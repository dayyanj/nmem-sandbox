"""Desktop I/O: capture the screen (scrot) and inject input (xdotool) on the
Xvfb display. Coordinates from the VLM are normalized [0,1000] per axis; we map
them to real pixels here. All calls are synchronous subprocess — the agent loop
runs them via asyncio.to_thread so the event loop stays free.
"""
import os
import subprocess
import tempfile

DISPLAY = os.environ.get("DISPLAY", ":99")
SCREEN_W = int(os.environ.get("SCREEN_W", "1920"))
SCREEN_H = int(os.environ.get("SCREEN_H", "1080"))

_ENV = {**os.environ, "DISPLAY": DISPLAY}


def _run(args: list[str], timeout: float = 10.0) -> str:
    return subprocess.run(args, env=_ENV, capture_output=True, text=True,
                          timeout=timeout).stdout


def screen_size() -> tuple[int, int]:
    return SCREEN_W, SCREEN_H


def norm_to_px(nx: float, ny: float) -> tuple[int, int]:
    """Map normalized [0,1000] coords to pixels, clamped to the screen. Coerces to float defensively —
    the VLM sometimes emits coords as strings, which broke the division."""
    nx, ny = float(nx), float(ny)
    px = max(0, min(SCREEN_W - 1, round(nx / 1000.0 * SCREEN_W)))
    py = max(0, min(SCREEN_H - 1, round(ny / 1000.0 * SCREEN_H)))
    return px, py


def screenshot() -> bytes:
    """Capture the current screen as PNG bytes (scrot)."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        subprocess.run(["scrot", "-o", tmp.name], env=_ENV, capture_output=True,
                       timeout=15.0, check=True)
        with open(tmp.name, "rb") as f:
            return f.read()


# ── Input primitives (xdotool) — inputs are normalized [0,1000] ──

def move(nx: float, ny: float):
    px, py = norm_to_px(nx, ny)
    _run(["xdotool", "mousemove", str(px), str(py)])


def click(nx: float, ny: float, button: int = 1):
    px, py = norm_to_px(nx, ny)
    _run(["xdotool", "mousemove", str(px), str(py), "click", str(button)])


def double_click(nx: float, ny: float):
    px, py = norm_to_px(nx, ny)
    _run(["xdotool", "mousemove", str(px), str(py), "click", "--repeat", "2", "1"])


def right_click(nx: float, ny: float):
    click(nx, ny, button=3)


def type_text(text: str):
    # --clearmodifiers avoids a held modifier corrupting the typed string
    _run(["xdotool", "type", "--clearmodifiers", "--delay", "20", text], timeout=30.0)


def key(keys: str):
    # xdotool key syntax, e.g. "Return", "ctrl+l", "alt+Tab"
    _run(["xdotool", "key", "--clearmodifiers", keys])


def scroll(nx: float, ny: float, amount: int):
    """Scroll at (nx,ny). amount>0 scrolls down, <0 up. Wheel = button 4/5."""
    px, py = norm_to_px(nx, ny)
    button = "5" if amount > 0 else "4"
    args = ["xdotool", "mousemove", str(px), str(py)]
    for _ in range(min(abs(amount), 20)):
        args += ["click", button]
    _run(args)
