#!/usr/bin/env bash
# Bring up the virtual desktop, then (Phase 2+) exec the control-loop harness.
set -euo pipefail

: "${SCREEN_W:=1920}"
: "${SCREEN_H:=1080}"
: "${SCREEN_D:=24}"
export DISPLAY=:99

# Clear any stale X lock/socket from a previous run — otherwise `docker restart` leaves Xvfb unable
# to bind :99 (it silently fails and every session dies at the first screenshot). Recreate is clean;
# restart is not, without this.
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 2>/dev/null || true

echo "[sandbox] starting Xvfb at ${SCREEN_W}x${SCREEN_H}x${SCREEN_D}"
Xvfb :99 -screen 0 "${SCREEN_W}x${SCREEN_H}x${SCREEN_D}" -ac +extension RANDR +extension GLX >/tmp/xvfb.log 2>&1 &
# wait for the X server to accept connections
for _ in $(seq 1 30); do xdpyinfo -display :99 >/dev/null 2>&1 && break; sleep 0.3; done

echo "[sandbox] starting fluxbox"
fluxbox >/tmp/fluxbox.log 2>&1 &
sleep 1

# VNC bound to localhost only (reach noVNC via SSH tunnel) — no open VNC on the LAN.
echo "[sandbox] starting x11vnc (localhost) + noVNC on :6080"
x11vnc -display :99 -localhost -forever -shared -nopw -rfbport 5900 -quiet >/tmp/x11vnc.log 2>&1 &
websockify --web=/usr/share/novnc 6080 localhost:5900 >/tmp/novnc.log 2>&1 &
sleep 1

echo "[sandbox] desktop ready (DISPLAY=:99, ${SCREEN_W}x${SCREEN_H}); noVNC http://localhost:6080/vnc.html"

# Start a maximized browser so the agent always has something to drive.
# --test-type suppresses the "unsupported flag --no-sandbox" infobar.
# --disable-dev-shm-usage: a container's default /dev/shm is only 64 MB; Chromium exhausts it and the
#   renderer crashes into a PERSISTENT BLACK SCREEN that ctrl+r can't recover (observed live). Writing
#   shared memory to /tmp instead avoids it. --disable-features drops the noisy Lens overlay.
# KEEP THESE FLAGS IN SYNC with harness/desktop.py:_CHROMIUM_CMD (the self-heal relaunch).
echo "[sandbox] launching Chromium"
chromium --no-sandbox --test-type --disable-gpu --disable-dev-shm-usage --no-first-run \
    --no-default-browser-check --disable-infobars --disable-features=LensOverlay,LensStandalone \
    --start-maximized --window-position=0,0 --window-size="${SCREEN_W},${SCREEN_H}" \
    about:blank >/tmp/chromium.log 2>&1 &
sleep 3

# Phase 2 will replace this with: exec python3 -m harness.service
if [ -f /home/sandbox/harness/service.py ]; then
    echo "[sandbox] launching harness"
    exec python3 -m harness.service
else
    echo "[sandbox] no harness yet (Phase 1) — holding desktop open"
    exec tail -f /dev/null
fi
