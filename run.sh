#!/usr/bin/env bash
# Run the nmem-sandbox — the LLM-agnostic computer-use desktop (a VLM drives a real Linux desktop;
# the agent POSTs an objective + polls). ALL per-agent settings are ENV, so one image serves every
# agent (Gemma, Qwen, …) — nothing here is agent-specific.
#
# Required: VLM_URL, VLM_MODEL. For a Qwen-family backend also set VLM_ENABLE_THINKING=false.
# Examples:
#   Gemma:  SANDBOX_NAME=my_sandbox VLM_URL=http://<vlm-host>:8003/v1/chat/completions \
#           VLM_MODEL=gemma-4-26B-A4B ./run.sh
#   Qwen:   SANDBOX_NAME=my_sandbox VLM_URL=http://<vlm-host>:8001/v1/chat/completions \
#           VLM_MODEL=Qwen3.8-27B VLM_ENABLE_THINKING=false ./run.sh
set -euo pipefail

NAME="${SANDBOX_NAME:-nmem_sandbox}"
IMAGE="${SANDBOX_IMAGE:-nmem-sandbox:dev}"
ART="${SANDBOX_ARTIFACTS:-$HOME/${NAME}_artifacts}"
: "${VLM_URL:?set VLM_URL to the served vision-model chat/completions endpoint}"
: "${VLM_MODEL:?set VLM_MODEL to the served model name}"

mkdir -p "$ART"; chmod 777 "$ART" 2>/dev/null || true
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" \
  -p 127.0.0.1:${NOVNC_PORT:-6080}:6080 \
  -p 127.0.0.1:${HARNESS_PORT:-8080}:8080 \
  -e VLM_URL="$VLM_URL" \
  -e VLM_MODEL="$VLM_MODEL" \
  -e VLM_ENABLE_THINKING="${VLM_ENABLE_THINKING:-}" \
  -e VLM_TEMPERATURE="${VLM_TEMPERATURE:-0.3}" \
  -e VLM_TIMEOUT="${VLM_TIMEOUT:-120}" \
  -e SCREEN_W="${SCREEN_W:-1280}" \
  -e SCREEN_H="${SCREEN_H:-800}" \
  -e SANDBOX_CAPTURE_FRAMES="${SANDBOX_CAPTURE_FRAMES:-1}" \
  -v "$ART:/home/sandbox/artifacts" \
  --restart no \
  "$IMAGE"
echo "$NAME up. harness=127.0.0.1:${HARNESS_PORT:-8080}  noVNC=127.0.0.1:${NOVNC_PORT:-6080}/vnc.html (SSH-tunnel to view)"
