"""Keyframe capture for visual memory.

The session loop screenshots the desktop every step and the VLM reasons over it. Those pixels
are normally discarded. When SANDBOX_CAPTURE_FRAMES=1 we additionally persist a DOWNSCALED JPEG
of each observed frame to the session's artifact dir and hand the caller lightweight metadata
(a perceptual hash + dims) in the step trace — so the agent can decide, cheaply and WITHOUT
re-transferring pixels on every poll, which frames are worth pulling (GET /session/{id}/frame/{i})
and turning into visual memories.

PIL-only (already a sandbox dep); no numpy/opencv/imagehash needed. Fully best-effort: any failure
returns None and the session is unaffected.
"""
import io
import os

from PIL import Image

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "/home/sandbox/artifacts")
CAPTURE = os.environ.get("SANDBOX_CAPTURE_FRAMES", "0") == "1"
MAX_W = int(os.environ.get("SANDBOX_FRAME_MAX_W", "640"))
QUALITY = int(os.environ.get("SANDBOX_FRAME_QUALITY", "60"))

_HASH = 8  # dhash grid → 64-bit hash


def _dhash_hex(img: "Image.Image") -> str:
    """Row-wise difference hash (perceptual): grayscale → (HASH+1)×HASH → compare adjacent."""
    g = img.convert("L").resize((_HASH + 1, _HASH), Image.LANCZOS)
    px = list(g.getdata())
    val = 0
    for row in range(_HASH):
        base = row * (_HASH + 1)
        for col in range(_HASH):
            val = (val << 1) | (1 if px[base + col] > px[base + col + 1] else 0)
    return f"{val:016x}"


def hamming(a: str, b: str) -> int:
    """Bit distance between two dhash hexes (0=identical, 64=opposite). 64 on bad input."""
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except Exception:  # noqa: BLE001
        return 64


def phash_of(png_bytes: bytes) -> str | None:
    """Perceptual dhash of a screenshot, ALWAYS computed — independent of SANDBOX_CAPTURE_FRAMES
    (which only gates JPEG *persistence*). Cheap: PIL grayscale → 9×8 → 64-bit hash. Returns None on
    any failure. The stall gate uses this to tell whether an action MATERIALLY changed the screen, so
    it must be available even when keyframe capture is off. Best-effort — never raises."""
    if not png_bytes:
        return None
    try:
        return _dhash_hex(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
    except Exception:  # noqa: BLE001
        return None


# Browser chrome (tab strip + omnibox) is a roughly FIXED pixel height regardless of window size — so a
# purely proportional crop under-cuts it at small resolutions, leaving lit toolbar in the measured region
# and masking a black viewport. Crop max(proportional, these pixel floors), clamped so it can't eat the
# whole frame.
_TOP_CHROME_PX = 90     # tab strip + URL bar
_BOTTOM_CHROME_PX = 28  # WM/taskbar strip


def is_blank(png_bytes: bytes, *, dark_frac: float = 0.97, dark_level: int = 10,
             top_chrome: float = 0.12, bottom_chrome: float = 0.05, side_chrome: float = 0.02) -> bool:
    """True when the page CONTENT AREA is (near-)entirely BLACK — the wedged-compositor failure where
    Chromium's window + tab/URL bar + the WM bar still render but the viewport paints nothing. Keyboard
    reloads can't fix that (the wedge is below the page), so the session loop treats a persistent blank
    frame as a trigger to run its recovery.

    We threshold the CONTENT REGION, not the whole frame: the tab strip + omnibox (~top) and the WM
    toolbar (~bottom) still render on a black page, and that lit chrome is enough to keep a whole-frame
    check under the 0.97 threshold — so recovery would never fire. Chrome is a roughly FIXED pixel
    height, so the crop is max(proportional, a pixel floor) and clamped, keeping detection robust across
    resolutions. Distinguished from a merely dark page (dark-mode site, a video) two ways: the threshold
    is PURE-black (`dark_level`=10), not dark-grey — a wedged compositor paints literal 0,0,0, whereas a
    real dark theme sits at luminance ~16-24+; and almost the whole viewport must clear it — a real page
    (even sparse, even dark) still has visible content. Cheap PIL check; returns False on any error so it
    never causes a false restart."""
    if not png_bytes:
        return False
    try:
        im = Image.open(io.BytesIO(png_bytes)).convert("L")
        w, h = im.size
        if w < 4 or h < 4:
            return False
        # Crop the browser/desktop chrome so only the page viewport is measured. Pixel floors cover
        # fixed-height chrome at small resolutions; the clamps keep the crop from eating the whole frame.
        top = min(max(int(h * top_chrome), _TOP_CHROME_PX), h // 3)
        bot = min(max(int(h * bottom_chrome), _BOTTOM_CHROME_PX), h // 6)
        side = min(int(w * side_chrome), w // 8)
        box = (side, top, w - side, h - bot)
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:   # degenerate crop → measure whole frame
            box = (0, 0, w, h)
        im = im.crop(box).resize((64, 36), Image.BILINEAR)
    except Exception:  # noqa: BLE001
        return False
    px = list(im.getdata())
    if not px:
        return False
    dark = sum(1 for p in px if p <= dark_level)
    return (dark / len(px)) >= dark_frac


def frame_path(session_id: str, step_i: int) -> str:
    return os.path.join(ARTIFACT_DIR, session_id, f"frame_{step_i}.jpg")


def save_keyframe(session_id: str, step_i: int, png_bytes: bytes) -> dict | None:
    """Persist a downscaled JPEG of one observed frame; return {frame_id, phash, w, h} or None.

    No-op (returns None) unless SANDBOX_CAPTURE_FRAMES=1. Best-effort — never raises."""
    if not CAPTURE or not png_bytes:
        return None
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        phash = _dhash_hex(img)
        w, h = img.size
        if w > MAX_W:
            img = img.resize((MAX_W, max(1, round(h * MAX_W / w))), Image.LANCZOS)
        d = os.path.join(ARTIFACT_DIR, session_id)
        os.makedirs(d, exist_ok=True)
        img.save(frame_path(session_id, step_i), "JPEG", quality=QUALITY)
        return {"frame_id": step_i, "phash": phash, "w": img.size[0], "h": img.size[1]}
    except Exception:  # noqa: BLE001 — capture is additive, never breaks a session
        return None
