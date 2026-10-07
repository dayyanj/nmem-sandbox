"""pursue_knowledge-V2 #1: mid-loop landscape-routing (the "Canyon Effect" fix).

Pure units over the tolerant route parser, the cadence gate, the hint renderer, and the flag default +
next_action wiring. The live route_check VLM call is covered on the sandbox canary (SANDBOX_ROUTE_CHECK).
"""
import asyncio
from harness import vlm, agent as A


# ── _parse_route (tolerant, fail-open) ────────────────────────────────────────

def test_parse_clean_pivot():
    out = vlm._parse_route(
        '{"branches":["docs site","forum","github"],"current":"forum",'
        '"most_promising":"docs site","aligned":false,"recommendation":"open the docs site"}')
    assert out["aligned"] is False
    assert out["current"] == "forum" and out["most_promising"] == "docs site"
    assert out["recommendation"] == "open the docs site"
    assert out["branches"] == ["docs site", "forum", "github"]


def test_parse_clean_aligned():
    out = vlm._parse_route('{"branches":["a","b"],"current":"a","most_promising":"a","aligned":true,"recommendation":""}')
    assert out["aligned"] is True


def test_parse_fenced_and_think_wrapped():
    out = vlm._parse_route('<think>weighing branches</think>```json\n{"aligned":false,"recommendation":"pivot"}\n```')
    assert out["aligned"] is False and out["recommendation"] == "pivot"


def test_parse_aligned_as_string():
    assert vlm._parse_route('{"aligned":"false","recommendation":"go"}')["aligned"] is False
    assert vlm._parse_route('{"aligned":"true"}')["aligned"] is True


def test_parse_garbage_fails_open_aligned():
    out = vlm._parse_route("no json, just prose")
    assert out["aligned"] is True            # fail open → no spurious pivot
    assert out["recommendation"] == "" and out["branches"] == []


def test_parse_missing_aligned_fails_open():
    out = vlm._parse_route('{"branches":["a"],"current":"a"}')
    assert out["aligned"] is True            # missing `aligned` → treated as aligned


def test_parse_empty_aligned_string_fails_open():
    # "" / unrecognised aligned strings must NOT read as a pivot
    assert vlm._parse_route('{"aligned":"","most_promising":"docs"}')["aligned"] is True
    assert vlm._parse_route('{"aligned":"maybe"}')["aligned"] is True


def test_parse_non_string_recommendation_dropped():
    # a dict/list/number recommendation is garbage — must not be stringified into a pivot
    out = vlm._parse_route('{"aligned":false,"recommendation":{"error":"x"}}')
    assert out["recommendation"] == ""
    assert A._route_hint_text(out) == ""     # no branch + no rec → nothing to inject


def test_parse_non_string_most_promising_dropped():
    out = vlm._parse_route('{"aligned":false,"most_promising":{"a":1},"recommendation":""}')
    assert out["most_promising"] == ""
    assert A._route_hint_text(out) == ""


def test_parse_unclosed_think_fails_open():
    # a truncated / still-reasoning reply must not have a pivot mined out of it
    out = vlm._parse_route('<think>Possible answer: {"aligned":false,"recommendation":"try elsewhere"}')
    assert out["aligned"] is True and out["recommendation"] == ""


def test_int_env_tolerates_garbage(monkeypatch):
    monkeypatch.setenv("SANDBOX_ROUTE_EVERY", "")
    assert A._int_env("SANDBOX_ROUTE_EVERY", 5) == 5
    monkeypatch.setenv("SANDBOX_ROUTE_EVERY", "notanint")
    assert A._int_env("SANDBOX_ROUTE_EVERY", 5) == 5
    monkeypatch.setenv("SANDBOX_ROUTE_EVERY", "7")
    assert A._int_env("SANDBOX_ROUTE_EVERY", 5) == 7


def test_parse_branches_string_normalised_and_capped():
    out = vlm._parse_route('{"branches":"only one","aligned":true}')
    assert out["branches"] == ["only one"]
    out = vlm._parse_route('{"branches":["1","2","3","4","5","6"],"aligned":true}')
    assert len(out["branches"]) == 4         # capped at 4


# ── _route_hint_text ──────────────────────────────────────────────────────────

def test_hint_empty_when_aligned():
    assert A._route_hint_text({"aligned": True, "recommendation": "x", "most_promising": "y"}) == ""


def test_hint_empty_when_nothing_actionable():
    # not aligned but no recommendation AND no alternative branch → nothing to say
    assert A._route_hint_text({"aligned": False, "recommendation": "", "most_promising": "", "current": "a"}) == ""


def test_hint_built_on_pivot():
    h = A._route_hint_text({"aligned": False, "current": "forum", "most_promising": "docs site",
                            "recommendation": "open the docs site"})
    assert "TRAJECTORY CHECK" in h
    assert "forum" in h and "docs site" in h and "open the docs site" in h


def test_hint_built_with_branch_only():
    # a pivot with an alternative branch but no explicit recommendation still produces a usable hint
    h = A._route_hint_text({"aligned": False, "current": "forum", "most_promising": "docs site",
                            "recommendation": ""})
    assert "docs site" in h and h != ""


# ── _should_route cadence gate ────────────────────────────────────────────────

def _route(i, last=-1, remaining=20, nudge="", n_actions=10):
    return A._should_route(i, last, remaining, nudge, n_actions)


def test_should_route_off_by_default():
    assert A.ROUTE_CHECK is False
    assert _route(10) is False               # flag off → never


def test_should_route_gates(monkeypatch):
    monkeypatch.setattr(A, "ROUTE_CHECK", True)
    monkeypatch.setattr(A, "ROUTE_EVERY", 5)
    monkeypatch.setattr(A, "ROUTE_MIN_STEP", 4)
    # too early (step/actions below MIN_STEP)
    assert _route(3, n_actions=10) is False
    assert _route(10, n_actions=3) is False
    # a repetition nudge owns the step
    assert _route(10, nudge="repeat!") is False
    # conclude-pressure zone
    assert _route(10, remaining=4) is False
    # cadence: not enough steps since last route
    assert _route(6, last=4) is False        # 6-4 = 2 < 5
    # first eligible check and a later one
    assert _route(4, last=-1) is True        # 4-(-1) = 5 >= 5
    assert _route(10, last=4) is True         # 10-4 = 6 >= 5


# ── next_action wiring: route="" is byte-identical; a route is injected ────────

def test_next_action_route_injection(monkeypatch):
    captured = {}

    async def _fake_chat_image(system, user_text, image, max_tokens=400):
        captured["user_text"] = user_text
        return '{"reasoning":"x","action":{"type":"done","result":"ok"}}'

    monkeypatch.setattr(vlm.backends, "chat_image", _fake_chat_image)

    asyncio.run(vlm.next_action("T", ["step 0: search"], b"img", steps_remaining=10, route=""))
    assert "TRAJECTORY CHECK" not in captured["user_text"]      # off/empty → not present

    asyncio.run(vlm.next_action("T", ["step 0: search"], b"img", steps_remaining=10,
                                route="TRAJECTORY CHECK: pivot to docs"))
    assert "TRAJECTORY CHECK: pivot to docs" in captured["user_text"]


def test_flag_default_off():
    assert A.ROUTE_CHECK is False
