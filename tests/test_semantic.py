"""Semantic-layer wiring (harness/agent.py action contract + harness/browser.py fail-open gate).

These are pure-function checks — no browser, no CDP. They lock in that the new intent-actions are
formatted + de-duplicated correctly and that the semantic layer is genuinely OFF (fail-open) when its
flag is disabled, so a build without Playwright behaves exactly like the pixel-only sandbox.
"""
import importlib
import os

from harness import agent as A


# ── _brief: the human-readable step label for the new intent actions ─────────

def test_brief_click_ref():
    assert A._brief({"type": "click_ref", "ref": "f1e6"}) == "click_ref(f1e6)"


def test_brief_type_ref_includes_text():
    s = A._brief({"type": "type_ref", "ref": "f1e9", "text": "hello world"})
    assert s.startswith("type_ref(f1e9,") and "hello world" in s


def test_brief_click_role_and_text():
    assert A._brief({"type": "click_role", "role": "button", "name": "Submit"}) == \
        "click_role(button,'Submit')"
    assert A._brief({"type": "click_text", "text": "Log in"}) == "click_text('Log in')"


# ── _sig: distinct semantic targets must NOT look like a repeat ───────────────

def test_sig_distinguishes_refs():
    a = {"type": "click_ref", "ref": "f1e6"}
    b = {"type": "click_ref", "ref": "f1e7"}
    assert A._sig(a) != A._sig(b)      # different elements → different signatures (no false repeat-nudge)


def test_sig_distinguishes_roles():
    a = {"type": "click_role", "role": "button", "name": "OK"}
    b = {"type": "click_role", "role": "button", "name": "Cancel"}
    assert A._sig(a) != A._sig(b)


# ── fail-open: the layer is OFF when the flag is off ─────────────────────────

def test_browser_available_false_when_disabled(monkeypatch):
    monkeypatch.setenv("SANDBOX_SEMANTIC", "0")
    import harness.browser as B
    importlib.reload(B)
    try:
        assert B.available() is False
    finally:
        monkeypatch.setenv("SANDBOX_SEMANTIC", "1")
        importlib.reload(B)
