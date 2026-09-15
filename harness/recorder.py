"""Per-session screen recording (ffmpeg x11grab -> mp4) for audit. Best-effort:
a recording failure must never break a session."""
import os
import signal
import subprocess

DISPLAY = os.environ.get("DISPLAY", ":99")
SCREEN_W = int(os.environ.get("SCREEN_W", "1920"))
SCREEN_H = int(os.environ.get("SCREEN_H", "1080"))


def start(out_path: str) -> subprocess.Popen | None:
    try:
        return subprocess.Popen(
            ["ffmpeg", "-y", "-f", "x11grab", "-framerate", "5",
             "-video_size", f"{SCREEN_W}x{SCREEN_H}", "-i", DISPLAY,
             "-codec:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             out_path],
            env={**os.environ, "DISPLAY": DISPLAY},
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None


def stop(proc: subprocess.Popen | None):
    if not proc:
        return
    try:
        proc.send_signal(signal.SIGINT)  # lets ffmpeg finalize the mp4 moov atom
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
