"""Stall-gate + escalating-nudge logic (harness/agent.py).

These cover the behavioural loop-breaker that ships as the real fix (a soft, escalating hint that
never kills), plus the phash-based stall counter used for shadow telemetry. The hard-kill path is
deliberately NOT the default — a backtest over recorded sessions showed the signal can't separate
doomed sessions from ones that thrash then recover — so there is no test asserting it should fire in
normal operation; only that the counter tracks and resets correctly.
"""
from harness import agent as A
from harness import frames


# ── behavioural nudge (the shipped soft gate) ────────────────────────────────

def test_repeated_scroll_never_nudges():
    """Repeated scrolling is legitimate reading — the #1 long-run pattern in successful sessions."""
    scroll = [{"type": "scroll", "x": 500, "y": 500, "amount": 3} for _ in range(8)]
    assert A._nudge_for(scroll) == ""


def test_page_down_key_is_scrollish():
    paging = [{"type": "key", "keys": "Page_Down"} for _ in range(6)]
    assert A._nudge_for(paging) == ""


def test_repeated_click_nudges_at_three():
    click = [{"type": "click", "x": 100, "y": 200} for _ in range(3)]
    msg = A._nudge_for(click)
    assert "repeated the SAME action 3 times" in msg
    assert "LAST reminder" not in msg          # not escalated yet


def test_repeated_click_escalates_at_five():
    click = [{"type": "click", "x": 100, "y": 200} for _ in range(5)]
    msg = A._nudge_for(click)
    assert "LAST reminder" in msg               # firm push to conclude
    assert "done" in msg


def test_varying_searches_hit_the_searchish_branch():
    srch = [{"type": "search", "query": q} for q in ("a", "b", "c", "d")]
    assert "searched" in A._nudge_for(srch)


def test_scroll_interrupts_a_repeat_run():
    """A scroll between clicks resets the trailing identical-run, so no nudge yet."""
    mixed = [{"type": "click", "x": 1, "y": 1},
             {"type": "scroll", "x": 5, "y": 5},
             {"type": "click", "x": 1, "y": 1}]
    assert A._nudge_for(mixed) == ""


def test_short_history_is_quiet():
    assert A._nudge_for([{"type": "click", "x": 1, "y": 1}]) == ""


# ── phash progress signal (drives shadow telemetry) ──────────────────────────

def test_phash_stable_for_identical_frames():
    """phash_of is deterministic; hamming of a frame with itself is 0 (a no-progress step)."""
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.new("RGB", (200, 150), (30, 60, 90)).save(buf, "PNG")
    png = buf.getvalue()
    p1, p2 = frames.phash_of(png), frames.phash_of(png)
    assert p1 is not None and p1 == p2
    assert frames.hamming(p1, p2) == 0


def test_phash_of_is_capture_independent():
    """phash_of must work regardless of SANDBOX_CAPTURE_FRAMES (it feeds the runtime gate)."""
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 255, 255)).save(buf, "PNG")
    assert frames.phash_of(buf.getvalue()) is not None
    assert frames.phash_of(b"") is None
    assert frames.phash_of(b"not-an-image") is None


# ── blank-frame (wedged-compositor) detector — is_blank ──────────────────────

def _png(img):
    import io
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_is_blank_true_on_all_black():
    from PIL import Image
    assert frames.is_blank(_png(Image.new("RGB", (1280, 800), (0, 0, 0)))) is True


def test_is_blank_true_with_lit_chrome_over_black_viewport():
    """The real failure: page viewport is black but the tab/URL bar + WM toolbar still render. A
    whole-frame check would miss it at low res; measuring the content region must still flag it."""
    from PIL import Image
    im = Image.new("RGB", (1280, 800), (0, 0, 0))          # black viewport
    # a bright browser toolbar band across the top (~7% of height) + a WM bar at the bottom
    for y in range(0, 56):
        for x in range(0, 1280):
            im.putpixel((x, y), (235, 235, 235))
    for y in range(770, 800):
        for x in range(0, 1280):
            im.putpixel((x, y), (200, 200, 200))
    assert frames.is_blank(_png(im)) is True


def test_is_blank_true_with_fixed_chrome_at_small_resolution():
    """Chrome is a ~fixed pixel height, so a proportional-only crop under-cuts it on a short screen and a
    black viewport leaks lit toolbar. At 1280x600 with a 92px toolbar over black, is_blank must still fire."""
    from PIL import Image
    im = Image.new("RGB", (1280, 600), (0, 0, 0))         # black viewport
    for y in range(0, 92):                                # a tall, bright toolbar (fixed px, not scaled)
        for x in range(0, 1280):
            im.putpixel((x, y), (235, 235, 235))
    assert frames.is_blank(_png(im)) is True


def test_is_blank_false_on_real_page():
    """A page with actual content in the viewport is not blank (mostly-light body)."""
    from PIL import Image
    assert frames.is_blank(_png(Image.new("RGB", (1280, 800), (245, 245, 245)))) is False


def test_is_blank_false_on_dark_mode_page_with_content():
    """A dark-mode page is dark but has visible content — must NOT be treated as a dead viewport."""
    from PIL import Image
    im = Image.new("RGB", (1280, 800), (18, 18, 20))       # dark bg
    # scatter legible content (light text blocks) across the viewport
    for band in range(150, 750, 90):
        for y in range(band, band + 26):
            for x in range(120, 1160):
                im.putpixel((x, y), (210, 210, 214))
    assert frames.is_blank(_png(im)) is False


def test_is_blank_false_on_sparse_dark_page():
    """The precise false-positive to avoid: a HEALTHY dark page (#121214) that is almost entirely bg
    with only a small welcome link. A dark-GREY threshold would restart the browser and lose the page;
    a pure-black threshold (a wedge paints literal 0,0,0) leaves this healthy page well clear."""
    from PIL import Image
    im = Image.new("RGB", (1280, 800), (18, 18, 20))       # #121214, luminance ~18
    for y in range(300, 324):                              # one 24px light link, nothing else
        for x in range(120, 360):
            im.putpixel((x, y), (120, 150, 240))
    assert frames.is_blank(_png(im)) is False


def test_is_blank_false_on_junk():
    assert frames.is_blank(b"") is False
    assert frames.is_blank(b"not-an-image") is False


# ── self-heal threshold — _heal_decision ─────────────────────────────────────
# A bounded restart after BLANK_HEAL_AFTER consecutive PURE-black frames. is_blank already excludes real
# dark-theme sites (dark_level=10), and those consecutive frames are steps the MODEL had to reload out
# and didn't — so a still-black view is unusable either way and a bounded restart is the safe fallback.
# Signature: _heal_decision(blank, blank_run, heals).

def test_heal_decision_no_action_when_not_blank():
    assert A._heal_decision(False, 0, 0) == "none"


def test_heal_decision_waits_for_consecutive_blanks():
    """Below the consecutive-frame threshold we do nothing (a transient black frame isn't a wedge)."""
    assert A._heal_decision(True, A.BLANK_HEAL_AFTER - 1, 0) == "none"


def test_heal_decision_restarts_at_threshold():
    """At BLANK_HEAL_AFTER consecutive blanks, with budget left → restart."""
    assert A._heal_decision(True, A.BLANK_HEAL_AFTER, 0) == "restart"


def test_heal_decision_respects_heal_budget():
    """Once the per-session restart budget is spent, stop healing (let the normal gates handle it)."""
    assert A._heal_decision(True, A.BLANK_HEAL_AFTER + 3, A.MAX_HEALS_PER_SESSION) == "none"


def test_is_blank_true_on_sparse_pure_black_page():
    """A healthy page that is PURE-black with only a tiny link is pixel-indistinguishable from a wedge,
    so is_blank (a pixel test) returns True here — by design. is_blank's specificity (pure-black only,
    dark_level=10) keeps real dark-theme sites out; the residual pure-#000000-sparse page is unusable to
    the agent when black anyway, so the loop's bounded restart (→ about:blank, model re-navigates, capped
    by MAX_HEALS) is an accepted, cheap fallback for a throwaway browser."""
    from PIL import Image
    im = Image.new("RGB", (1280, 800), (0, 0, 0))          # pure black bg
    for y in range(300, 324):                              # one 24px light link, nothing else
        for x in range(120, 360):
            im.putpixel((x, y), (255, 255, 255))
    assert frames.is_blank(_png(im)) is True               # pixel-blank; the loop's probe is the guard
