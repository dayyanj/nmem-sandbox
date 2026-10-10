"""Regression: the scroll `amount` parser. The VLM frequently emits a DIRECTION WORD ('down'/'up') in
`amount` (e.g. {"type":"scroll","amount":"down"}); the old `int(action.get("amount", 3))` raised
ValueError on that and errored ~a third of pursue_knowledge sessions. `_scroll_amount` coerces to a signed
int (>0 down, <0 up): numeric passes through, words map to a signed default, junk -> default (down)."""
from harness.agent import _scroll_amount


def test_numeric_passthrough():
    assert _scroll_amount({"amount": 5}) == 5
    assert _scroll_amount({"amount": -2}) == -2
    assert _scroll_amount({"amount": "4"}) == 4          # numeric string
    assert _scroll_amount({"amount": "3.0"}) == 3        # float string → int


def test_direction_words_map_to_signed_default():
    assert _scroll_amount({"amount": "down"}) == 3        # the exact crash case
    assert _scroll_amount({"amount": "up"}) == -3
    assert _scroll_amount({"amount": "DOWN"}) == 3        # case-insensitive
    assert _scroll_amount({"direction": "up"}) == -3      # word in `direction` instead of `amount`


def test_missing_and_junk_default_down():
    assert _scroll_amount({}) == 3                        # nothing → default down
    assert _scroll_amount({"amount": "gibberish"}) == 3   # unparseable, no 'up' → down
    assert _scroll_amount({"amount": None}) == 3


def test_overflow_and_substring_do_not_crash_or_misfire():
    assert _scroll_amount({"amount": "inf"}) == 3         # int(float('inf')) OverflowError → down, no crash
    assert _scroll_amount({"amount": "1e309"}) == 3       # overflow → down
    assert _scroll_amount({"amount": "nan"}) == 3         # NaN → down
    assert _scroll_amount({"amount": "unsupported"}) == 3  # contains 'up' as a SUBSTRING but is not the word


def test_custom_default_magnitude():
    assert _scroll_amount({"amount": "up"}, default=5) == -5
    assert _scroll_amount({"amount": "down"}, default=5) == 5
