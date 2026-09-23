"""The session loop: screenshot -> VLM -> action -> xdotool, until done / blocked
/ max_steps / timeout / abort. Desktop calls are sync (xdotool/scrot) so they run
in a thread; the VLM call is async.
"""
import asyncio
import logging
import os
import time
import uuid

from . import browser, desktop, frames, recorder, vlm

log = logging.getLogger(__name__)

# Semantic (CDP/Playwright) layer: give the model a ref-tagged accessibility map + intent actions
# (click_ref/type_ref/click_role/click_text) alongside the pixel path. Resolved once — if playwright
# isn't installed or the flag is off, this is False and the loop behaves exactly as pixel-only before.
_SEMANTIC = browser.available()

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "/home/sandbox/artifacts")

_READ_ONLY_NOTE = (" (READ-ONLY: observe and report only. Do NOT submit forms, "
                   "purchase, send messages, or change any settings. If the task "
                   "genuinely requires a write action, emit ask_founder.)")

# ── Stall gate (SHADOW / experimental) ───────────────────────────────────────
# The idea: cut a session off when it stops making OBSERVABLE progress, so the flat max_steps cap
# can recede to a pure safety backstop instead of guillotining productive long research. "No
# progress" is measured from the SCREEN, not the action: an action that leaves the screen
# perceptually unchanged (dhash distance <= STALL_MATERIAL_HAMMING) is a wasted step; `wait`/`noop`/
# errored steps never count; any material change resets the run to 0.
#
# IMPORTANT — this signal is NOT validated for enforcement. A backtest over 1,436 recorded sessions
# showed it (and every repetition-based variant tried) cannot separate doomed sessions from ones
# that thrash and then recover: at kill>=6 it would salvage ~68% of SUCCESSFUL sessions ~20 steps
# early. So the real fixes shipped alongside are the raised backstops + the escalating behavioural
# nudge (see _nudge_for). This gate defaults to SHADOW (telemetry only) so we keep accumulating the
# running counter in the trace; `enforce` stays an explicit opt-in for future experimentation once a
# more discriminative signal exists — do NOT enable it on the current signal.
#   off     — disabled, no counter
#   shadow  — track the counter + log "would have salvaged at step N", but do NOT terminate (default)
#   enforce — salvage-terminate at STALL_KILL_AT (experimental; not recommended on today's signal)
STALL_GATE = os.environ.get("SANDBOX_STALL_GATE", "shadow").strip().lower()
STALL_MATERIAL_HAMMING = int(os.environ.get("SANDBOX_STALL_MATERIAL_HAMMING", "6"))
STALL_KILL_AT = int(os.environ.get("SANDBOX_STALL_KILL_AT", "6"))

# Self-heal a wedged/black-screen browser: after this many CONSECUTIVE all-black frames, restart Chromium
# (bounded per session) rather than let the agent thrash ctrl+r against a dead compositor. Unlike the
# stall gate this is an ACTIVE recovery, not telemetry. Set SANDBOX_BROWSER_HEAL=0 to disable.
BROWSER_HEAL = os.environ.get("SANDBOX_BROWSER_HEAL", "1").strip().lower() not in ("0", "false", "no", "off")
BLANK_HEAL_AFTER = int(os.environ.get("SANDBOX_BLANK_HEAL_AFTER", "3"))
MAX_HEALS_PER_SESSION = int(os.environ.get("SANDBOX_MAX_HEALS", "2"))


def _heal_decision(blank: bool, blank_run: int, heals: int) -> str:
    """'restart' when the viewport has been PURE-black for BLANK_HEAL_AFTER consecutive frames and we
    still have restart budget, else 'none'. A pure function so the threshold logic is unit-testable.

    Why a plain bounded restart, not a cleverer probe: a pixel+xdotool harness can't observe renderer
    state (tabs aren't X windows, keystrokes aren't ack'd), so any 'is it really dead' probe is itself
    unreliable. Instead we lean on what IS reliable: `frames.is_blank` only fires on a ~fully PURE-black
    viewport (dark_level=10), which already excludes real dark-theme sites (luminance ~16-24+); and we
    require BLANK_HEAL_AFTER consecutive blank frames — during which the MODEL has itself had that many
    steps to reload/navigate out. If it's still pure-black after all that, the view is unusable to the
    agent whether it's a wedge or a rare all-black page, so a bounded restart (→ about:blank, model
    re-navigates) is the cheap, safe fallback. The ROOT cause — renderer crash from /dev/shm exhaustion —
    is fixed upstream (--disable-dev-shm-usage in entrypoint.sh + desktop._CHROMIUM_CMD); this is the net.
    """
    if not blank or blank_run < BLANK_HEAL_AFTER:
        return "none"
    return "restart" if heals < MAX_HEALS_PER_SESSION else "none"


class Session:
    def __init__(self, task: str, mode: str = "read_only", max_steps: int = 20,
                 wall_clock_s: float = 300.0):
        self.id = uuid.uuid4().hex[:12]
        self.task = task
        self.mode = mode
        self.max_steps = max_steps
        self.wall_clock_s = wall_clock_s
        self.status = "queued"          # queued|running|done|blocked|aborted|timeout|max_steps|stalled|error
        self.result: str | None = None
        self.steps: list[dict] = []
        self.abort = False
        self.error: str | None = None
        self.video = os.path.join(ARTIFACT_DIR, f"{self.id}.mp4")
        # Stall gate observability: running counter, first shadow-crossing step, and the events.
        self.stall = 0
        self.shadow_would_kill_at: int | None = None
        self.gate_events: list[dict] = []
        # Self-heal: consecutive pure-black frames + how many times we've restarted the browser this
        # session (bounded by MAX_HEALS_PER_SESSION).
        self.blank_run = 0
        self.heals = 0

    def public(self) -> dict:
        return {"id": self.id, "task": self.task, "mode": self.mode,
                "status": self.status, "result": self.result, "error": self.error,
                "steps": self.steps, "n_steps": len(self.steps), "video": self.video,
                "stall": self.stall, "shadow_would_kill_at": self.shadow_would_kill_at,
                "gate_events": self.gate_events}


def _num(v, default: float = 0.0) -> float:
    """Coerce a coordinate to a number. The VLM often emits x/y as STRINGS ('500'), which then blew up
    the pixel conversion (`'500' / 1000.0` -> TypeError) and errored every session on step 0."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


async def _goto(text: str):
    """Reliable one-shot address-bar entry: focus + select-all + type + Enter.
    Backs the compound 'search'/'navigate' actions so the model never has to
    sequence the fragile ctrl+l -> type -> Return dance across turns (it could
    not always perceive that the bar was focused, and looped on ctrl+l)."""
    text = (text or "").strip()
    if not text:
        return
    await asyncio.to_thread(desktop.key, "ctrl+l")
    await asyncio.sleep(0.25)
    await asyncio.to_thread(desktop.key, "ctrl+a")
    await asyncio.sleep(0.10)
    await asyncio.to_thread(desktop.type_text, text)
    await asyncio.sleep(0.15)
    await asyncio.to_thread(desktop.key, "Return")


async def _execute(action: dict) -> str | None:
    """Run one action. Returns None on success, or a short error string for a SEMANTIC action that
    didn't land (so the loop records it and the model can retry via the pixel path). Pixel/xdotool
    actions are best-effort and always return None (they can't report failure)."""
    t = action.get("type")
    x, y = _num(action.get("x")), _num(action.get("y"))
    # ── Semantic (CDP) actions: act by intent. On failure return the reason (→ model falls back). ──
    if t == "click_ref":
        ok, why = await browser.click_ref(str(action.get("ref", "")))
        return None if ok else why
    if t == "type_ref":
        ok, why = await browser.type_ref(str(action.get("ref", "")), str(action.get("text", "")),
                                         submit=bool(action.get("submit")))
        return None if ok else why
    if t == "click_role":
        ok, why = await browser.click_role(str(action.get("role", "")), str(action.get("name", "")))
        return None if ok else why
    if t == "click_text":
        ok, why = await browser.click_text(str(action.get("text", "")))
        return None if ok else why
    if t in ("search", "navigate", "goto"):
        # An explicit URL goes through the reliable CDP page.goto when the semantic layer is up; a bare
        # search query (or any failure) uses the pixel address-bar dance (real default search engine).
        url = str(action.get("url") or "").strip()
        if _SEMANTIC and t != "search" and url:
            ok, _why = await browser.goto(url)
            if ok:
                return None
        await _goto(action.get("query") or action.get("url") or action.get("text", ""))
        return None
    if t == "click":
        await asyncio.to_thread(desktop.click, x, y)
    elif t == "double_click":
        await asyncio.to_thread(desktop.double_click, x, y)
    elif t == "right_click":
        await asyncio.to_thread(desktop.right_click, x, y)
    elif t == "move":
        await asyncio.to_thread(desktop.move, x, y)
    elif t == "type":
        await asyncio.to_thread(desktop.type_text, str(action.get("text", "")))
    elif t == "key":
        await asyncio.to_thread(desktop.key, str(action.get("keys", "")))
    elif t == "scroll":
        await asyncio.to_thread(desktop.scroll, x, y, int(action.get("amount", 3)))
    elif t == "wait":
        await asyncio.sleep(min(int(action.get("ms", 500)), 5000) / 1000.0)


def _sig(a: dict):
    # Include the semantic targets (ref/role/name) so distinct intent-actions aren't seen as repeats.
    return (a.get("type"), a.get("keys"), a.get("text"), a.get("x"), a.get("y"),
            a.get("ref"), a.get("role"), a.get("name"))


def _searchish(a: dict) -> bool:
    t = a.get("type")
    k = str(a.get("keys", "")).lower()
    return t in ("type", "search", "navigate", "goto") or (t == "key" and ("ctrl+l" in k or "ctrl+a" in k))


def _scrollish(a: dict) -> bool:
    """Scrolling / paging is legitimate reading progress even when repeated — trace analysis found
    repeated scroll is the #1 long-run pattern in SUCCESSFUL sessions. Excluded from rut detection."""
    k = str(a.get("keys", "")).lower()
    return a.get("type") == "scroll" or (a.get("type") == "key"
            and any(x in k for x in ("page", "down", "up", "space", "end", "home")))


def _trailing_identical(recent: list[dict]) -> int:
    """Length of the current run of the SAME action signature at the tail (scroll/paging excluded)."""
    if not recent or _scrollish(recent[-1]):
        return 0
    last = _sig(recent[-1])
    n = 1
    for a in reversed(recent[:-1]):
        if _sig(a) == last:
            n += 1
        else:
            break
    return n


def _trailing_searchish(recent: list[dict]) -> int:
    """Length of the current run of search / address-bar-refocus actions at the tail."""
    n = 0
    for a in reversed(recent):
        if _searchish(a):
            n += 1
        else:
            break
    return n


def _nudge_for(recent: list[dict]) -> str:
    """Loop-breaker (a SOFT gate — a hint the model can ignore, never a kill): detect the two ruts
    these sessions fall into — repeating one action, or re-searching instead of reading — and return
    a corrective hint. Escalates: at a run of 3 it suggests a different move; at 5+ (the first nudge
    went unheeded) it firmly pushes a "done" so a genuinely-stuck session concludes rather than
    grinding to the cap. Scrolls/paging are excluded — repeating them is legitimate reading."""
    ident = _trailing_identical(recent)
    srch = _trailing_searchish(recent)
    if max(ident, srch) < 3:
        return ""
    if srch >= 3:               # a search / address-bar rut (repeated OR varying queries)
        base = (f"You have searched / re-focused the address bar {srch} times in a row. Results are "
                "almost certainly already on screen — STOP searching: scroll to read them, click the "
                "most promising result, or emit \"done\" with what you have.")
    else:                       # a non-search action repeated verbatim
        base = (f"You have repeated the SAME action {ident} times with no progress — it is not "
                "working. Do something DIFFERENT now: scroll to read, click a result, or emit "
                "\"done\" with what you already know.")
    if max(ident, srch) >= 5:   # escalation: the soft hint didn't land — push to conclude
        base += (" This is your LAST reminder: the same approach keeps failing, so emit \"done\" NOW "
                 "with your best honest answer (state what you could not confirm) instead of repeating it.")
    return base


def _brief(action: dict) -> str:
    t = action.get("type", "?")
    if t in ("search", "navigate", "goto"):
        return f"{t}({str(action.get('query') or action.get('url') or action.get('text',''))[:50]!r})"
    if t in ("click", "double_click", "right_click", "move", "scroll"):
        return f"{t}({action.get('x')},{action.get('y')})"
    if t in ("click_ref", "type_ref"):
        txt = f",{str(action.get('text',''))[:30]!r}" if t == "type_ref" else ""
        return f"{t}({action.get('ref')}{txt})"
    if t == "click_role":
        return f"click_role({action.get('role')},{str(action.get('name',''))[:30]!r})"
    if t == "click_text":
        return f"click_text({str(action.get('text',''))[:30]!r})"
    if t == "type":
        return f"type({str(action.get('text',''))[:40]!r})"
    if t == "key":
        return f"key({action.get('keys')})"
    return t


async def run_session(session: Session):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    session.status = "running"
    task = session.task + (_READ_ONLY_NOTE if session.mode == "read_only" else "")
    history: list[str] = []
    recent: list[dict] = []            # recent action dicts, for loop detection
    last_img: bytes | None = None
    prev_phash: str | None = None      # immediately-preceding observed frame's perceptual hash
    prev_judgeable = False             # did the previous step execute a progress-judgeable action?
    stalled = False                    # broke out because the stall gate fired (vs the step ceiling)
    rec = recorder.start(session.video)
    deadline = time.monotonic() + session.wall_clock_s
    try:
        for i in range(session.max_steps):
            if session.abort:
                session.status = "aborted"
                return
            if time.monotonic() > deadline:
                break                  # salvage below rather than lose the work
            # ── Observe, self-healing a wedged browser ── A crashed Chromium compositor paints an all-
            # BLACK viewport (the window + WM bar still render). If the viewport has been PURE-black for
            # BLANK_HEAL_AFTER consecutive frames — the model has had that many steps to reload/navigate
            # out and hasn't — restart Chromium (bounded), re-observe WITHIN this step so the model gets
            # to act on the recovered about:blank, and tell it to re-navigate. See _heal_decision for why
            # a plain bounded restart (not a cleverer, unobservable probe) is the right call here.
            img = await asyncio.to_thread(desktop.screenshot)
            if BROWSER_HEAL:
                blank = frames.is_blank(img)
                session.blank_run = session.blank_run + 1 if blank else 0
                if _heal_decision(blank, session.blank_run, session.heals) == "restart":
                    session.heals += 1
                    healed = await asyncio.to_thread(desktop.restart_browser)
                    session.gate_events.append({"i": i, "action": "browser-restart",
                                                "blank_run": session.blank_run, "ok": bool(healed)})
                    # Reset the run either way so we re-accumulate before the next attempt (bounded by
                    # MAX_HEALS) rather than firing every frame, giving the model steps in between.
                    session.blank_run = 0
                    if healed:
                        log.warning("self-heal session=%s restarted browser at step=%d (heal %d/%d)",
                                    session.id, i, session.heals, MAX_HEALS_PER_SESSION)
                        session.stall = 0        # fresh browser = new baseline; don't carry a stale run
                        prev_phash = None
                        prev_judgeable = False
                        # Restart dropped us to about:blank — pre-restart actions are moot. Clear the loop
                        # detector (else _nudge_for reads the stale run) and tell the model to re-navigate.
                        recent.clear()
                        history.append(f"step {i}: [browser was wedged — restarted; now at about:blank, "
                                       f"re-open the site you need]")
                        await asyncio.sleep(1.0)
                        img = await asyncio.to_thread(desktop.screenshot)   # re-observe within the step
                    else:
                        # Relaunch failed — do NOT claim about:blank. Record it honestly and let the
                        # remaining heal budget retry on subsequent blank frames; the salvage path handles
                        # a session that never recovers.
                        log.error("self-heal session=%s browser restart FAILED at step=%d (heal %d/%d)",
                                  session.id, i, session.heals, MAX_HEALS_PER_SESSION)
                        history.append(f"step {i}: [browser recovery attempted but the browser did not "
                                       f"come back — the screen may still be blank]")
            last_img = img
            cur_phash = frames.phash_of(img)

            # ── Stall gate ── Did the PREVIOUS judgeable action materially move the screen? If not,
            # the session made no observable progress this step. Any material change resets the run.
            if prev_judgeable and prev_phash is not None and cur_phash is not None:
                if frames.hamming(cur_phash, prev_phash) > STALL_MATERIAL_HAMMING:
                    session.stall = 0
                else:
                    session.stall += 1

            # Visual memory: persist a downscaled keyframe of this observed state + a perceptual
            # hash, so the agent can later pull worth-keeping frames. No-op unless capture is on.
            kf = frames.save_keyframe(session.id, i, img)
            if kf is not None:
                kf["screen_changed"] = (prev_phash is None
                                        or frames.hamming(kf["phash"], prev_phash) > 4)
            prev_phash = cur_phash
            prev_judgeable = False     # set True only after we execute a real action below

            # Stall gate (shadow telemetry by default; enforce is opt-in + experimental — the signal
            # is not validated for termination, see the module-level note). In shadow we only record
            # the first crossing; in enforce we salvage-terminate.
            if STALL_GATE != "off" and session.stall >= STALL_KILL_AT:
                if STALL_GATE == "enforce":
                    session.gate_events.append({"i": i, "stall": session.stall, "action": "salvage"})
                    log.info("stall-gate[enforce] session=%s salvaging at step=%d (stall=%d)",
                             session.id, i, session.stall)
                    stalled = True
                    break
                if session.shadow_would_kill_at is None:   # shadow: record the FIRST crossing only
                    session.shadow_would_kill_at = i
                    session.gate_events.append({"i": i, "stall": session.stall, "action": "would-salvage"})
                    log.info("stall-gate[shadow] session=%s WOULD salvage at step=%d (stall=%d)",
                             session.id, i, session.stall)

            # Semantic map: the page's ref-tagged accessibility tree, so the model can act by intent
            # (click_ref/type_ref) instead of guessing pixels. Best-effort — None when the layer is
            # unavailable or the page has no tree, in which case the model just gets the screenshot.
            semantic = ""
            if _SEMANTIC:
                try:
                    snap = await browser.snapshot()
                    if snap:
                        dlg = snap.get("dialog")
                        note = (f"NOTE: a JavaScript {dlg['type']} dialog appeared and was auto-handled "
                                f"(alerts accepted, confirm/prompt dismissed): {dlg['message']!r}\n"
                                if dlg else "")
                        if snap.get("tree"):
                            semantic = f"{note}page: {snap.get('url','')}\n{snap['tree']}"
                        elif note:
                            semantic = note
                except Exception as e:  # noqa: BLE001 — never let the semantic layer break a step
                    log.debug("snapshot failed session=%s step=%d: %s", session.id, i, e)

            # The behavioural loop-breaker: a soft, escalating hint (never a kill) that pushes a
            # genuinely-stuck session to conclude. Grounded in repetition, not phash — see _nudge_for.
            nudge = _nudge_for(recent)
            try:
                decision = await vlm.next_action(task, history, img,
                                                 steps_remaining=session.max_steps - i,
                                                 nudge=nudge, semantic=semantic)
            except Exception as e:  # a VLM hiccup shouldn't kill the session
                session.steps.append({"i": i, "error": f"vlm: {e}", **({"frame": kf} if kf else {})})
                await asyncio.sleep(1.0)
                continue

            action = decision.get("action", {})
            atype = action.get("type")
            session.steps.append({"i": i, "reasoning": decision.get("reasoning", ""),
                                  "action": action, **({"frame": kf} if kf else {})})

            if atype == "done":
                session.result = str(action.get("result", ""))
                session.status = "done"
                return
            if atype == "ask_founder":
                session.result = str(action.get("question", ""))
                session.status = "blocked"
                return
            if atype in (None, "noop"):
                await asyncio.sleep(0.5)
                continue

            err = await _execute(action)
            recent.append(action)
            # Audit: a ref is ephemeral, so persist WHAT that ref pointed at (its map line: role + name)
            # alongside the action, so the durable JSON records the real target, not just "f1e6".
            if atype in ("click_ref", "type_ref") and semantic:
                ref = str(action.get("ref", ""))
                tgt = next((ln.strip() for ln in semantic.splitlines() if f"[ref={ref}]" in ln), "")
                if tgt:
                    session.steps[-1]["target"] = tgt[:180]
            note = f"step {i}: {_brief(action)} — {decision.get('reasoning','')[:80]}"
            if err:                       # a semantic action that didn't land — the model sees this and
                note += f"  [FAILED: {err}; try a pixel click or a different element]"
                session.steps[-1]["exec_error"] = err
            history.append(note)
            prev_judgeable = (atype != "wait")   # a deliberate wait isn't a progress attempt
            await asyncio.sleep(0.6)  # let the UI settle before the next screenshot

        # Step budget / wall-clock exhausted (or the stall gate fired) without a clean 'done'.
        # Salvage: ask the model for its best answer from everything it saw, so a session that DID
        # gather info returns knowledge instead of a bare failure.
        fail_status = "stalled" if stalled else "max_steps"
        prefix = ("[no observable progress — best-effort summary] " if stalled
                  else "[reached step limit — best-effort summary] ")
        try:
            try:  # a FRESH final screen (reflects the last action); fall back if it fails
                simg = await asyncio.to_thread(desktop.screenshot)
            except Exception:  # noqa: BLE001
                simg = last_img
            if simg is None:
                session.status = fail_status
                return
            summary = await vlm.summarize_done(task, history, simg)
            if summary and len(summary) > 20:
                session.result = prefix + summary
                session.status = "done"
            else:
                session.status = fail_status
        except Exception as e:  # noqa: BLE001
            session.status = fail_status
            session.error = f"salvage: {e}"
    except Exception as e:
        session.status = "error"
        session.error = str(e)
    finally:
        recorder.stop(rec)
        # Durable audit record: the full step log + result next to the mp4, so
        # the "audit-only" security posture actually has something to audit even
        # after the container is gone (artifacts are host-mounted).
        try:
            import json
            with open(os.path.join(ARTIFACT_DIR, f"{session.id}.json"), "w") as f:
                json.dump(session.public(), f, indent=2, default=str)
        except Exception:
            pass
