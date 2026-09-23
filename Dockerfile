# DJ-AI computer-use sandbox — a headless Linux desktop the VLM drives.
# Debian (real chromium .deb, no snap). Xvfb desktop + fluxbox + input/capture
# tooling + x11vnc/noVNC for live viewing. The control-loop harness (Phase 2)
# is COPYed in and exec'd by the entrypoint; Phase 1 just brings up the desktop.
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:99 \
    SCREEN_W=1280 \
    SCREEN_H=800 \
    SCREEN_D=24
# NB: SCREEN_W/H are a RUNTIME knob (Xvfb reads them in entrypoint.sh). run.sh
# passes them via -e so resolution can change without an image rebuild. Lower res
# = less downscaling by the VLM's vision encoder = more legible UI. Default 1280x800.

RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb x11vnc fluxbox xdotool scrot ffmpeg xterm \
        chromium chromium-common \
        novnc websockify \
        python3 python3-pip \
        fonts-liberation dbus-x11 ca-certificates curl procps net-tools \
    && rm -rf /var/lib/apt/lists/*

# Harness runtime deps (used from Phase 2 onward).
# playwright drives the SAME visible Chromium over CDP for the semantic (accessibility) layer
# (harness/browser.py). We attach to the system chromium via connect_over_cdp, so NO `playwright
# install` browser download is needed — the pip package (its bundled Node driver) is enough.
RUN pip3 install --no-cache-dir --break-system-packages \
        fastapi "uvicorn[standard]" httpx pillow playwright

RUN useradd -m -s /bin/bash sandbox
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
COPY --chown=sandbox:sandbox harness /home/sandbox/harness
# Create the artifacts dir OWNED BY sandbox before dropping privileges, so a bind/named volume
# mounted here (e.g. the appliance's sandbox_artifacts) inherits sandbox ownership and the harness
# can actually write recordings / audit JSON / keyframes (Docker seeds a fresh volume from the
# mountpoint's image owner; without this it would be root-owned and the writes would silently fail).
RUN mkdir -p /home/sandbox/artifacts && chown sandbox:sandbox /home/sandbox/artifacts

USER sandbox
WORKDIR /home/sandbox
# 6080 = noVNC web viewer (reach via SSH tunnel); 8080 = harness API (Phase 2)
EXPOSE 6080 8080
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
