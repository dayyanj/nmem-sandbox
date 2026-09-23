"""Semantic browser layer — drives the SAME visible Chromium over CDP (Playwright).

The pixel path (scrot + xdotool) treats the browser as opaque pixels: every click is a coordinate
guess. This module attaches Playwright to the running Chromium via its CDP port and exposes an
INTENT-level interface instead — a compact accessibility snapshot of the page (roles + names + stable
`[ref=…]` handles), plus click/type BY REF or BY ROLE/TEXT. That is both more reliable (no mis-clicks)
and less bot-like (deliberate role-based interaction rather than erratic coordinates).

It is STRICTLY ADDITIVE and FAIL-OPEN. If Playwright isn't installed, the CDP port is unreachable, or
any call errors, every function returns None / (False, reason) and the caller falls back to the pixel
path — the screenshot remains the ground truth. The layer is active only when SANDBOX_SEMANTIC != 0
and the browser was launched with a CDP port (entrypoint.sh / desktop._CHROMIUM_CMD).

One browser, one active tab is the norm here (concurrency is capped at 1 and the harness drives a
single desktop), so page selection is a simple "last open page" heuristic. Playwright's `aria-ref`
handles are tied to the most recent `aria_snapshot(mode="ai")` on a page — the session loop snapshots
every step right before asking the model, so the refs the model returns always match a fresh snapshot.
"""
import asyncio
import os

CDP_URL = os.environ.get("SANDBOX_CDP_URL", "http://127.0.0.1:9222")
ENABLED = os.environ.get("SANDBOX_SEMANTIC", "1").strip().lower() not in ("0", "false", "no", "off")
# Keep the model's semantic map bounded — a huge page shouldn't blow the VLM context.
SNAPSHOT_MAX_CHARS = int(os.environ.get("SANDBOX_SNAPSHOT_MAX_CHARS", "6000"))
_ACT_TIMEOUT_MS = float(os.environ.get("SANDBOX_ACT_TIMEOUT_MS", "6000"))

_pw = None            # the started Playwright driver (reused across reconnects)
_browser = None       # the CDP-connected browser
_lock = asyncio.Lock()
_hooked: set = set()  # id(page) we've attached a dialog handler to (idempotent)
_last_dialog = None   # the most recent JS dialog, surfaced once in the next snapshot then cleared


def _handle_dialog(dialog):
    """Explicit, EXPLICITLY-SURFACED policy for JS dialogs. Attaching Playwright otherwise auto-DISMISSES
    every alert/confirm/prompt silently (verified) — so a dialog the pixel path used to show would just
    vanish. Instead: accept alerts + beforeunload (informational / allow-navigate), dismiss confirm/prompt
    (the safe default for a read-only agent — never auto-confirm a possibly-writing action), and record
    the dialog so the NEXT snapshot tells the model it happened. Runs as a task (accept/dismiss are async)."""
    global _last_dialog
    try:
        _last_dialog = {"type": dialog.type, "message": (dialog.message or "")[:200]}
    except Exception:  # noqa: BLE001
        _last_dialog = {"type": "dialog", "message": ""}

    async def _resolve():
        try:
            if dialog.type in ("alert", "beforeunload"):
                await dialog.accept()
            else:
                await dialog.dismiss()
        except Exception:  # noqa: BLE001
            pass
    try:
        asyncio.get_event_loop().create_task(_resolve())
    except Exception:  # noqa: BLE001
        pass


def _hook_dialogs(page):
    """Attach the dialog handler to a page once (idempotent)."""
    if id(page) in _hooked:
        return
    try:
        page.on("dialog", _handle_dialog)
        _hooked.add(id(page))
    except Exception:  # noqa: BLE001
        pass


def _arm_dialogs(br):
    """Install the dialog policy at CONTEXT level — every existing page AND every future one (via the
    context 'page' event) — so alerts/prompts are handled + recorded regardless of which tab is
    foreground or whether the semantic layer is acting on it. Idempotent per browser connection."""
    try:
        for ctx in br.contexts:
            try:
                ctx.on("page", _hook_dialogs)
            except Exception:  # noqa: BLE001
                pass
            for p in ctx.pages:
                _hook_dialogs(p)
    except Exception:  # noqa: BLE001
        pass


def available() -> bool:
    """Cheap pre-check: is the semantic layer even a possibility? (flag on + playwright importable).
    Does NOT open a connection — use is_ready() / connect() for that."""
    if not ENABLED:
        return False
    try:
        import playwright.async_api  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


async def connect():
    """(Re)connect to the CDP browser; return the browser or None. Best-effort, never raises. Reused
    across steps; transparently reconnects after a self-heal browser restart (the old handle reports
    disconnected)."""
    global _pw, _browser
    if not ENABLED:
        return None
    if _browser is not None:
        try:
            if _browser.is_connected():
                return _browser
        except Exception:  # noqa: BLE001
            pass
    try:
        from playwright.async_api import async_playwright
    except Exception:  # noqa: BLE001 — playwright not installed → pixel-only, silently
        return None
    async with _lock:
        if _browser is not None:
            try:
                if _browser.is_connected():
                    return _browser
            except Exception:  # noqa: BLE001
                pass
        try:
            if _pw is None:
                _pw = await async_playwright().start()
            _browser = await _pw.chromium.connect_over_cdp(CDP_URL, timeout=5000)
            _hooked.clear()          # fresh connection → page ids are new; re-arm from scratch
            _arm_dialogs(_browser)   # cover all pages (existing + future) independent of tab selection
            return _browser
        except Exception:  # noqa: BLE001
            _browser = None
            return None


async def is_ready() -> bool:
    return (await connect()) is not None


async def _get_page(br):
    """The page semantic reads/actions operate on — but ONLY when we can be SURE it's the one the
    screenshot shows. That's the single-tab case, which is the norm here: the sandbox boots one tab and
    navigates in place. With 2+ tabs the foreground is genuinely NOT observable in this Xvfb+CDP setup
    (every tab reports document.visibilityState 'visible', even with focus emulation disabled), so
    rather than risk acting on a background tab we return None and the caller degrades to the pixel path
    for that step. Creates a page only if somehow none exist. Never blocks on renderer introspection."""
    ctx = br.contexts[0] if br.contexts else await br.new_context()
    pages = [p for p in ctx.pages if not p.is_closed()]
    if not pages:
        return await ctx.new_page()      # dialog handler attaches via the context 'page' event
    if len(pages) == 1:
        return pages[0]
    return None   # ambiguous foreground → let the caller fall back to pixels (safe, never wrong-tab)


async def _click_unique(build, desc: str) -> tuple[bool, str]:
    """Resolve a locator via build(page), require it to match EXACTLY ONE element, then click it.
    Substring/`.first` matching would silently click the wrong control (e.g. 'Save' hitting 'Save and
    publish'); instead we report no-match / ambiguity so the model picks a ref or a more specific name."""
    br = await connect()
    if br is None:
        return False, "semantic-browser-unavailable"
    try:
        page = await _get_page(br)
        if page is None:
            return False, "ambiguous-foreground-tab (use a pixel click)"
        loc = build(page)
        # Locator.count() has no built-in action timeout — a dead renderer would hang it (and with it the
        # whole session, which only checks its deadline/abort BETWEEN steps). Bound it explicitly.
        n = await asyncio.wait_for(loc.count(), timeout=(_ACT_TIMEOUT_MS / 1000.0) + 1.0)
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:140]}"
    if n == 0:
        return False, f"no match for {desc}"
    if n > 1:
        return False, f"{n} matches for {desc} — be more specific, or click_ref one from the map"
    return await _perform(page, lambda: loc.first.click(timeout=_ACT_TIMEOUT_MS))


def _consume_dialog():
    """Read + clear the most recent auto-handled JS dialog (surfaced once). Standalone so it can be
    reported on EVERY path — including when there's no semantic map (2+ tabs, or aria extraction fails) —
    since the global handler still auto-handled it and the model must learn it happened."""
    global _last_dialog
    dlg = _last_dialog
    _last_dialog = None
    return dlg


async def snapshot() -> dict | None:
    """The page's accessibility tree as a compact, ref-tagged semantic map (roles + names + `[ref=…]`).
    `tree` is Playwright's `aria_snapshot(mode="ai")` YAML — the model reads it and acts BY ref (bounded
    to SNAPSHOT_MAX_CHARS). Returns None only when there's nothing to report; otherwise a dict that may
    carry a `dialog` even with an empty `tree` (so an auto-handled dialog is never lost on the pixel
    fallback path). Best-effort; never raises."""
    dlg = _consume_dialog()
    _degraded = {"url": "", "tree": "", "dialog": dlg} if dlg else None
    br = await connect()
    if br is None:
        return _degraded
    try:
        page = await _get_page(br)
    except Exception:  # noqa: BLE001
        return _degraded
    if page is None:                      # ambiguous foreground (2+ tabs) → no map, but still report dlg
        return _degraded
    try:  # let a mid-navigation DOM settle a little; not fatal if it doesn't
        await page.wait_for_load_state("domcontentloaded", timeout=1500)
    except Exception:  # noqa: BLE001
        pass
    try:
        tree = await page.locator("body").aria_snapshot(mode="ai", timeout=3000)
    except Exception:  # noqa: BLE001
        return _degraded
    if tree and len(tree) > SNAPSHOT_MAX_CHARS:
        tree = tree[:SNAPSHOT_MAX_CHARS] + "\n… (truncated — scroll or act on what's shown)"
    try:
        url = page.url
    except Exception:  # noqa: BLE001
        url = ""
    # `dlg` was consumed at the top so it's reported on every path, degraded or not.
    return {"url": url, "tree": tree or "", "dialog": dlg}


async def _page_or_err(br):
    """The foreground page, or (None, reason) — reason is a model-facing fallback hint."""
    try:
        page = await _get_page(br)
    except Exception as e:  # noqa: BLE001
        return None, f"no-page: {type(e).__name__}"
    if page is None:
        return None, "ambiguous-foreground-tab (use a pixel click)"
    return page, ""


async def _perform(page, action) -> tuple[bool, str]:
    """Run a click/press that MAY navigate. Clicking a link that navigates can make Playwright's
    post-action wait TIME OUT even though the click landed and the page navigated (observed live) — so
    if the URL changed, that IS success. Then let the new page settle. Returns (ok, reason)."""
    try:
        before = page.url
    except Exception:  # noqa: BLE001
        before = None
    ok, why = True, ""
    try:
        await action()
    except Exception as e:  # noqa: BLE001
        navigated = False
        try:
            # the click may have STARTED a navigation not yet reflected in page.url — wait for the load
            # to settle BEFORE judging, else we'd falsely report failure on an in-flight navigation.
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=3000)
            except Exception:  # noqa: BLE001
                pass
            navigated = before is not None and page.url != before
        except Exception:  # noqa: BLE001
            navigated = False
        if not navigated:
            ok, why = False, f"{type(e).__name__}: {str(e)[:140]}"
    if ok:
        try:  # success path: let any navigation settle; not fatal if there wasn't one
            await page.wait_for_load_state("domcontentloaded", timeout=4000)
        except Exception:  # noqa: BLE001
            pass
    return ok, why


async def click_ref(ref: str) -> tuple[bool, str]:
    ref = (ref or "").strip()
    if not ref:
        return False, "empty-ref"
    br = await connect()
    if br is None:
        return False, "semantic-browser-unavailable"
    page, err = await _page_or_err(br)
    if page is None:
        return False, err
    loc = page.locator(f"aria-ref={ref}").first
    return await _perform(page, lambda: loc.click(timeout=_ACT_TIMEOUT_MS))


async def type_ref(ref: str, text: str, submit: bool = False) -> tuple[bool, str]:
    ref = (ref or "").strip()
    if not ref:
        return False, "empty-ref"
    br = await connect()
    if br is None:
        return False, "semantic-browser-unavailable"
    page, err = await _page_or_err(br)
    if page is None:
        return False, err
    loc = page.locator(f"aria-ref={ref}").first
    try:
        await loc.fill(str(text), timeout=_ACT_TIMEOUT_MS)   # fill focuses the field; doesn't navigate
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:140]}"
    if submit:
        # Press Enter ON the filled element via Playwright (the field has page focus after fill, so this
        # submits it — a pixel key(Return) could land on the address bar instead). Submitting may
        # navigate, so tolerate the navigation-timeout artifact.
        return await _perform(page, lambda: loc.press("Enter", timeout=_ACT_TIMEOUT_MS))
    return True, ""


async def click_role(role: str, name: str) -> tuple[bool, str]:
    role = (role or "").strip()
    if not role:
        return False, "empty-role"
    nm = (name or "").strip()
    # EXACT name match — a supplied name should hit that control, not a superstring of it.
    build = (lambda page: page.get_by_role(role, name=nm, exact=True)) if nm \
        else (lambda page: page.get_by_role(role))
    return await _click_unique(build, f"{role} {nm!r}" if nm else f"{role} (no name)")


async def click_text(text: str) -> tuple[bool, str]:
    text = (text or "").strip()
    if not text:
        return False, "empty-text"
    return await _click_unique(lambda page: page.get_by_text(text, exact=True), f"text {text!r}")


async def goto(url: str) -> tuple[bool, str]:
    """Navigate to an explicit URL (prepending https:// if the scheme is missing). Used by the
    'navigate' action when the semantic layer is up — more reliable than typing the address bar. Bare
    search QUERIES stay on the pixel address-bar path (they use the browser's real default engine)."""
    br = await connect()
    if br is None:
        return False, "semantic-browser-unavailable"
    u = (url or "").strip()
    if not u:
        return False, "empty-url"
    if not u.startswith(("http://", "https://", "about:", "file:")):
        u = "https://" + u
    try:
        page = await _get_page(br)
        if page is None:
            return False, "ambiguous-foreground-tab"
        await page.goto(u, wait_until="domcontentloaded", timeout=20000)
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:140]}"
