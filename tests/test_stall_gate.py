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
