"""The vision-model call + the computer-use action contract.

The model sees the current screenshot and the task, and returns ONE next action
as JSON. Coordinates are normalized [0,1000] per axis. NB: this convention was
verified for Qwen3.8-27B; Gemma 4's spatial grounding ACCURACY in [0,1000] is
still being calibrated (during model bring-up) — the convention is instructed
explicitly in the system prompt, so mismatch is an accuracy question, not a
format one. The parser is deliberately tolerant — the model's JSON shape varies
(e.g. {"x":N,"y":N} vs {"x":[N,N]}).
"""
import json
import re

from . import backends

# Wire format + backend/model config live in `backends.py` (pluggable per VLM_BACKEND). This module
# is the backend-agnostic computer-use ACTION CONTRACT: the system prompt, the action schema, and the
# tolerant JSON parser — nothing provider-specific.


SYSTEM_PROMPT = """You operate a Linux desktop by looking at screenshots and choosing ONE action at a time.

COORDINATES: normalized 0-1000 on each axis. (0,0) = top-left, (1000,1000) = bottom-right. Give the CENTER of what you want to click.

Respond with ONLY a JSON object, no prose, no markdown:
{"reasoning": "<one short sentence>", "action": {"type": "<type>", ...}}

Action types and their fields:
- {"type":"search","query":"<what to search the web for>"}   ← PREFERRED to search. One step: focuses the address bar, clears it, types the query, presses Enter.
- {"type":"navigate","url":"<full URL>"}                      ← PREFERRED to open a page. One step: focuses+clears the address bar, types the URL, presses Enter.
- {"type":"click_ref","ref":"<ref from the SEMANTIC MAP, e.g. f1e6>"}          ← PREFERRED to click a link/button/menu item that appears in the SEMANTIC MAP. Exact and reliable — no aiming.
- {"type":"type_ref","ref":"<ref>","text":"<text to enter>","submit":<true to press Enter in the field after filling — use for a search box / form that submits on Enter>}   ← PREFERRED to fill a text field / search box in the SEMANTIC MAP.
- {"type":"click_role","role":"<button|link|textbox|checkbox|tab|menuitem>","name":"<its visible/accessible name>"}   ← click by role + name when you know it.
- {"type":"click_text","text":"<visible text on the control>"}                  ← click the control that shows this text.
- {"type":"click","x":<0-1000>,"y":<0-1000>}
- {"type":"double_click","x":<0-1000>,"y":<0-1000>}
- {"type":"right_click","x":<0-1000>,"y":<0-1000>}
- {"type":"move","x":<0-1000>,"y":<0-1000>}
- {"type":"type","text":"<text to type>"}
- {"type":"key","keys":"<xdotool key combo, e.g. Return, ctrl+l, alt+Tab>"}
- {"type":"scroll","x":<0-1000>,"y":<0-1000>,"amount":<+down / -up integer>}
- {"type":"wait","ms":<milliseconds>}
- {"type":"done","result":"<what you observed / accomplished — the answer to the task>"}
- {"type":"ask_founder","question":"<blocked; need a human decision or a write approval>"}

RULES:
- One action per turn. After each action you get a fresh screenshot.
- SEMANTIC MAP: when a page's interactive elements are listed (roles, names, and [ref=…] handles), PREFER acting on them — click_ref / type_ref (by ref), or click_role / click_text. This is exact and reliable; a pixel click(x,y) is a GUESS. Use pixel click/type for things NOT in the map (canvas, images, custom-drawn widgets, a PDF) or when no map is given. AND if a semantic action FAILS (you'll see "[FAILED: …]" — the ref may be stale or the control unsupported), do NOT repeat it: fall back to a pixel click(x,y) on that element, or pick a different element.
- TO SEARCH OR OPEN A PAGE, ALWAYS USE "search" / "navigate" — they do the whole address-bar sequence reliably in ONE step. Do NOT hand-roll ctrl+l → type → Return: pressing ctrl+l on its own is a frequent dead-end (you cannot always tell the bar is focused, and you end up pressing it repeatedly). If you catch yourself about to press ctrl+l, use "search" or "navigate" instead.
- Prefer type_ref to fill a field that is in the map. type_ref FILLS but does NOT submit by itself — to submit a search box / form, add "submit":true (it presses Enter INSIDE the field). Do NOT use a pixel key("Return") to submit a type_ref'd field — that goes to the browser UI, not the field. Use pixel "type"/"key" only for fields NOT in the map. Not the address bar (use search/navigate).
- When the task is complete, emit "done" with the result. If you cannot proceed or the task needs a write/purchase/submit you are not allowed to do, emit "ask_founder".

CONCLUDING — READ THIS (most sessions fail by never concluding):
- You have a LIMITED step budget. The MOMENT you have enough to answer the task — even partially — emit {"type":"done","result":"..."} with what you found. A concise, honest partial answer via "done" is far better than running out of steps with nothing recorded.
- Do NOT keep re-searching. If search results or a page are ALREADY visible, do not type a new query — READ what's there: "scroll" to see more, or "click" the most promising result. Re-typing queries / repeatedly pressing ctrl+l when you already have relevant results just wastes your budget and records nothing.
- If after genuine effort you cannot find the answer, STILL emit "done" — state what you did find and what you could not confirm. Do not loop.

RULES OF ENGAGEMENT (non-negotiable):
- You are a researcher and operator, NOT an attacker. Explore, read, and learn freely across the open web.
- NEVER attempt to bypass authentication, exploit a vulnerability, brute-force or guess credentials, scan or probe systems for weaknesses, or access any system you have not been explicitly authorized to use.
- Only log in to systems you were given credentials for (these belong to Spwig). NEVER enter credentials into a site that is not the intended Spwig system.
- If a task appears to require attacking, breaking into, defrauding, or abusing any system, STOP and emit ask_founder instead.
- Everything you do here is recorded and reviewed."""


async def next_action(task: str, history: list[str], image_png: bytes,
                      steps_remaining: int | None = None, nudge: str = "",
                      semantic: str = "", route: str = "") -> dict:
    """Ask the VLM for the next action given the task, a text history, and the
    CURRENT screenshot only (we never resend past images — keeps context bounded).
    steps_remaining drives conclude-pressure; nudge is an optional loop-breaker;
    `semantic` is the current page's ref-tagged accessibility map (empty when the
    semantic layer is unavailable — then only the pixel actions apply); `route` is an
    optional mid-loop trajectory hint from route_check (empty unless a pivot is advised)."""
    hist = "\n".join(history[-12:]) if history else "(none yet)"
    budget = ""
    if steps_remaining is not None:
        budget = (f"\n\nSTEP BUDGET: {steps_remaining} step(s) left. "
                  + ("Few remain — wrap up and emit \"done\" with your best answer NOW."
                     if steps_remaining <= 4 else
                     "As soon as you have enough to answer, emit \"done\" rather than keep exploring."))
    sem = (f"\n\nSEMANTIC MAP of the current page (prefer acting on these — click_ref/type_ref by ref, "
           f"or click_role/click_text):\n{semantic}" if semantic else "")
    user_text = (f"TASK: {task}\n\nActions so far:\n{hist}{budget}{sem}"
                 + (f"\n\n{nudge}" if nudge else "")
                 + (f"\n\n{route}" if route else "")
                 + "\n\nHere is the current screen. Choose the next action.")
    content = await backends.chat_image(SYSTEM_PROMPT, user_text, image_png, max_tokens=400)
    return _parse_action(content)


async def route_check(task: str, history: list[str], image_png: bytes, semantic: str = "") -> dict:
    """Mid-loop landscape-routing (Michelle's pursue_knowledge-V2 ask #1): a periodic self-check that
    maps the search space BEFORE the sunk cost of the next several steps. Distinct from _nudge_for,
    which is repetition-based — THIS is TRAJECTORY/branch-based: it asks whether the current branch is
    still the most likely place the answer lives, or whether to pivot. PROBABILISTIC, not descriptive —
    it weighs candidate branches by likelihood rather than asserting one structure (the pre-flight-plan
    trap Michelle rejected). One VLM call. Returns {branches, current, most_promising, aligned,
    recommendation}; tolerant parse that FAILS OPEN (aligned=True → no pivot injected)."""
    hist = "\n".join(history[-16:]) if history else "(no actions)"
    sem = (f"\n\nSEMANTIC MAP of the current page:\n{semantic}" if semantic else "")
    user_text = (
        f"TASK: {task}\n\nYou have been pursuing this for several steps (your actions below, current "
        f"screen attached). Before you sink more steps into the current branch, step back and MAP THE "
        f"SEARCH SPACE — do NOT pick a click, just route:\n"
        f"1. List the 2-4 plausible branches (sites, sections, link paths, or sources) where the answer "
        f"could live, ordered by how LIKELY each is to actually contain it.\n"
        f"2. Name which branch your CURRENT trajectory is on.\n"
        f"3. Decide: is the current trajectory still the most promising route to the goal, or should you "
        f"pivot to a different branch?\n"
        f"This is a probabilistic sanity-check on your topology, not a commitment — you may be deep in a "
        f"convincing branch that does not hold the answer (the \"canyon\" trap). Prefer pivoting early "
        f"over sinking steps into a dead end.\n\n"
        f"Return ONLY a JSON object, no prose, no markdown:\n"
        f'{{"branches": ["<most likely>", "<next>", "..."], '
        f'"current": "<the branch you are on now>", '
        f'"most_promising": "<the branch most likely to hold the answer>", '
        f'"aligned": <true if the current trajectory is still the best route, false if you should pivot>, '
        f'"recommendation": "<if not aligned: the concrete pivot to make next; else empty>"}}\n\n'
        f"Actions:\n{hist}{sem}")
    content = await backends.chat_image(
        "You are a careful research strategist. Assess the search trajectory honestly; prefer pivoting "
        "early over sinking steps into a dead-end branch. Never fabricate.",
        user_text, image_png, max_tokens=400)
    return _parse_route(content)


def _parse_route(content: str) -> dict:
    """Tolerant parse of the route-check reply → {branches, current, most_promising, aligned,
    recommendation}. FAILS OPEN: an unparseable reply (or a missing `aligned`) returns aligned=True so
    no spurious pivot is injected — the loop then behaves as if no routing fired."""
    _OPEN = {"branches": [], "current": "", "most_promising": "", "aligned": True, "recommendation": ""}
    text = (content or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Unlike _parse_action (which MUST recover an action even from a truncated reply), route_check is
    # advisory — so an UNCLOSED <think> means the model was still reasoning and hadn't committed a
    # verdict. Do NOT mine it for JSON: fail open rather than inject a pivot from unfinished reasoning.
    if "<think>" in text:
        return dict(_OPEN)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict):
        return dict(_OPEN)
    branches = obj.get("branches")
    if isinstance(branches, str):
        branches = [branches]
    elif not isinstance(branches, list):
        branches = []
    # `aligned` fails OPEN: only an EXPLICIT false/no/0 (bool or recognised string) is a pivot; a bool
    # True, an unrecognised/empty string, or a missing/garbage value all mean "stay" (no spurious pivot).
    raw = obj.get("aligned")
    if isinstance(raw, bool):
        aligned = raw
    elif isinstance(raw, str):
        aligned = raw.strip().lower() not in ("false", "no", "0")
    else:
        aligned = True
    # Only trust STRING free-text for the injected fields; a non-string (dict/list/number) is garbage —
    # drop it to "" rather than stringify an object into a pivot instruction.
    def _s(v, n):
        return v.strip()[:n] if isinstance(v, str) else ""
    return {
        "branches": [s[:200] for s in (b for b in branches if isinstance(b, str)) if s.strip()][:4],
        "current": _s(obj.get("current"), 200),
        "most_promising": _s(obj.get("most_promising"), 200),
        "aligned": aligned,
        "recommendation": _s(obj.get("recommendation"), 300),
    }


async def summarize_done(task: str, history: list[str], image_png: bytes) -> str:
    """Salvage call: when the step budget is spent, ask the model to give its BEST
    answer from everything it saw, so a session that gathered info still returns
    knowledge instead of nothing. Returns a plain-text result (may be partial)."""
    hist = "\n".join(history[-16:]) if history else "(no actions)"
    user_text = (
        f"TASK: {task}\n\nYou have run out of steps. Based on EVERYTHING you saw "
        f"(your actions below and the final screen), give your best, honest answer "
        f"to the task now. Be concrete; state clearly what you could NOT confirm. "
        f"Do not invent facts.\n\nActions:\n{hist}\n\nFinal screen is attached.")
    text = await backends.chat_image(
        "You are a careful research assistant. Summarize findings honestly; never fabricate.",
        user_text, image_png, max_tokens=700)
    return (text or "").strip()


async def summarize_structured(task: str, history: list[str], image_png: bytes) -> dict:
    """Structured session result (Michelle's pursue_knowledge-V2 ask #2): return WHAT WAS SOUGHT, WHAT
    WAS FOUND, and the GAPS/unconfirmed as JSON — so the calling agent gets a reconciled delta to act on
    instead of prose it must re-synthesise. One call at session end. Tolerant: a non-JSON reply falls
    back to {sought: task, found: <text>, gaps: []} so this never fails the session."""
    hist = "\n".join(history[-16:]) if history else "(no actions)"
    user_text = (
        f"TASK: {task}\n\nBased on EVERYTHING you saw (your actions below and the final screen), return a "
        f"STRUCTURED summary as ONLY a JSON object, no prose, no markdown:\n"
        f'{{"sought": "<what the task was trying to find or learn, one line>", '
        f'"found": "<the concrete findings, honestly and specifically>", '
        f'"gaps": ["<each thing you could NOT confirm>", "..."]}}\n'
        f"Be concrete; never invent facts; an empty gaps list is fine if nothing is unconfirmed.\n\n"
        f"Actions:\n{hist}\n\nFinal screen is attached.")
    text = await backends.chat_image(
        "You are a careful research assistant. Return honest, structured JSON; never fabricate.",
        user_text, image_png, max_tokens=700)
    return _parse_structured(text, task)


def _parse_structured(content: str, task: str) -> dict:
    """Tolerant parse of the structured-summary reply → {sought, found, gaps}. Falls back to carrying the
    raw text as `found` (never loses the knowledge) when the model didn't emit clean JSON."""
    text = (content or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in text:
        text = text[text.index("{"):] if "{" in text else ""
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict):
        return {"sought": task, "found": (content or "").strip(), "gaps": []}
    gaps = obj.get("gaps")
    if isinstance(gaps, str):
        gaps = [gaps]
    elif not isinstance(gaps, list):
        gaps = []
    return {
        "sought": str(obj.get("sought") or task)[:500],
        "found": str(obj.get("found") or "").strip(),
        "gaps": [str(g)[:300] for g in gaps if str(g).strip()],
    }


def _parse_action(content: str) -> dict:
    """Extract {reasoning, action} from the model text, tolerant of format drift."""
    text = (content or "").strip()
    # Strip inline reasoning the server may NOT have split out — varies by model variant + vLLM
    # version (reasoning_content vs inline <think>). Remove closed <think>…</think> blocks (they can
    # contain braces that would wreck the greedy JSON fallback); for an unclosed/truncated <think>,
    # keep only from the first JSON brace. So the action JSON survives regardless of server behavior.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in text:
        text = text[text.index("{"):] if "{" in text else ""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None
    if not isinstance(obj, dict):
        return {"reasoning": "unparseable", "action": {"type": "noop", "raw": content[:200]}}

    action = obj.get("action", obj)  # tolerate a bare action object
    if not isinstance(action, dict):
        return {"reasoning": obj.get("reasoning", ""), "action": {"type": "noop"}}

    # Normalize coordinate shape: sometimes x is a [x,y] pair.
    if isinstance(action.get("x"), (list, tuple)) and len(action["x"]) == 2:
        action["x"], action["y"] = action["x"][0], action["x"][1]
    return {"reasoning": str(obj.get("reasoning", ""))[:200], "action": action}
