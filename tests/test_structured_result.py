"""pursue_knowledge-V2 #2: structured {sought,found,gaps} session result + the flag gate.

Pure units over the tolerant parser + the flag default. The live VLM call is covered on the sandbox
canary (SANDBOX_STRUCTURED_RESULT).
"""
import asyncio
from harness import vlm, agent as A


def test_parse_clean_json():
    out = vlm._parse_structured('{"sought":"find X","found":"X is 42","gaps":["cost"]}', "T")
    assert out == {"sought": "find X", "found": "X is 42", "gaps": ["cost"]}


def test_parse_gaps_string_normalised_to_list():
    out = vlm._parse_structured('{"sought":"a","found":"b","gaps":"only one"}', "T")
    assert out["gaps"] == ["only one"]


def test_parse_fenced_and_think_wrapped():
    out = vlm._parse_structured('<think>reasoning</think>```json\n{"sought":"a","found":"b","gaps":[]}\n```', "T")
    assert out["found"] == "b" and out["gaps"] == []


def test_parse_garbage_falls_back_without_losing_text():
    out = vlm._parse_structured("no json here, just prose findings", "the original task")
    assert out["sought"] == "the original task"          # task preserved
    assert out["found"] == "no json here, just prose findings"   # raw knowledge not lost
    assert out["gaps"] == []


def test_parse_drops_blank_gaps():
    out = vlm._parse_structured('{"sought":"a","found":"b","gaps":["real","","  "]}', "T")
    assert out["gaps"] == ["real"]


def test_flag_graduated_default_on():
    # GRADUATED default-on 2026-10-10 (validated on a reference deployment). Opt out with SANDBOX_STRUCTURED_RESULT=0.
    assert A.STRUCTURED_RESULT is True
    # opt-out semantics: an explicit falsey value still disables it (byte-identical off path)
    assert ("0".strip().lower() not in ("0", "false", "no", "off")) is False


def test_maybe_structured_noop_when_opted_out(monkeypatch):
    class _S:
        structured = None
    s = _S()
    calls = []
    monkeypatch.setattr(A, "STRUCTURED_RESULT", False)   # explicit opt-out (graduated default is now on)
    monkeypatch.setattr(A.vlm, "summarize_structured", lambda *a, **k: calls.append(1))
    asyncio.run(A._maybe_structured(s, "task", [], object()))
    assert s.structured is None and calls == []          # byte-identical off path: no VLM call at all
