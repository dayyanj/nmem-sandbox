"""Desktop I/O: capture the screen (scrot) and inject input (xdotool) on the
Xvfb display. Coordinates from the VLM are normalized [0,1000] per axis; we map
them to real pixels here. All calls are synchronous subprocess — the agent loop
runs them via asyncio.to_thread so the event loop stays free.
"""
import os
import subprocess
import tempfile
import time

DISPLAY = os.environ.get("DISPLAY", ":99")
SCREEN_W = int(os.environ.get("SCREEN_W", "1920"))
SCREEN_H = int(os.environ.get("SCREEN_H", "1080"))

_ENV = {**os.environ, "DISPLAY": DISPLAY}

# Chromium launch flags — KEEP IN SYNC with entrypoint.sh (the boot launch). restart_browser() relaunches
# with these to recover a wedged/black compositor. --disable-dev-shm-usage is load-bearing: a container's
# default /dev/shm is tiny (64 MB), Chromium exhausts it, and the renderer crashes into a persistent black
# screen that keyboard reloads can't fix — the very failure this recovery exists for.
_CHROMIUM_CMD = ["chromium", "--no-sandbox", "--test-type", "--disable-gpu", "--disable-dev-shm-usage",
                 "--no-first-run", "--no-default-browser-check", "--disable-infobars",
                 "--disable-features=LensOverlay,LensStandalone",
                 # CDP port for the semantic layer — MUST match entrypoint.sh so the self-heal relaunch
                 # keeps Playwright able to reattach. Loopback-only; Chrome 136+ needs a non-default
                 # --user-data-dir for remote debugging.
                 "--remote-debugging-port=9222", "--remote-debugging-address=127.0.0.1",
                 "--remote-allow-origins=*", "--user-data-dir=/home/sandbox/.chromium-cdp",
                 "--start-maximized", "--window-position=0,0", f"--window-size={SCREEN_W},{SCREEN_H}",
                 "about:blank"]


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


def restart_browser(timeout: float = 25.0) -> bool:
    """Kill the (wedged) Chromium and relaunch it fresh — recovery for an unresponsive/black-screen
    compositor that keyboard reloads can't fix. Xvfb is left running (only the browser's rendering is
    dead, not the display). Returns True once a new Chromium window is visible. Synchronous; the caller
    runs it via asyncio.to_thread. Best-effort — never raises."""
    try:
        subprocess.run(["pkill", "-TERM", "-f", "chromium"], env=_ENV, capture_output=True, timeout=10.0)
        time.sleep(1.5)
        subprocess.run(["pkill", "-KILL", "-f", "chromium"], env=_ENV, capture_output=True, timeout=10.0)
        time.sleep(0.8)
        # A -KILL leaves the profile's Singleton* locks behind; the fresh Chromium can refuse to start
        # ("profile appears to be in use") until they're cleared. Safe to remove — the old process is dead.
        for lock in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
            try:
                os.remove(os.path.join("/home/sandbox/.chromium-cdp", lock))
            except OSError:
                pass
        subprocess.Popen(_CHROMIUM_CMD, env=_ENV,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        return False
    end = time.time() + timeout
    while time.time() < end:
        try:
            w = subprocess.run(["xdotool", "search", "--class", "chromium"], env=_ENV,
                               capture_output=True, text=True, timeout=5.0).stdout.strip()
        except Exception:  # noqa: BLE001
            w = ""
        if w:
            time.sleep(2.0)      # give the fresh window a beat to paint its first frame
            return True
        time.sleep(0.5)
    return False


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
