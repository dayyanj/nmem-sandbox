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
