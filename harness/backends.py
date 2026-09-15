"""Pluggable VLM backend adapters — the ONE place the sandbox knows a model's wire format.

The computer-use ACTION CONTRACT (`vlm.py`: system prompt, action schema, JSON parsing) is
backend-agnostic. Everything model/provider-specific — endpoint, auth, image encoding, request
shape, response parsing, family kwargs — lives HERE, behind `chat_image()`, selected by
`VLM_BACKEND`. Adding a provider is a localized adapter, never a fork of the harness.

Config (env — the appliance convention; one backend per sandbox container):
  VLM_BACKEND   openai (default) | anthropic | …   which wire adapter
  VLM_URL       served chat endpoint (openai: /v1/chat/completions)
  VLM_MODEL     served model name
  VLM_API_KEY   bearer/x-api-key for hosted APIs (optional; blank for local vLLM)
  VLM_TIMEOUT   per-call seconds (default 120)
  VLM_TEMPERATURE  default sampling temp (default 0.3)
  VLM_ENABLE_THINKING  openai/vLLM ONLY: set false for Qwen-family (adds
                       chat_template_kwargs.enable_thinking=false); unset for Gemma.
"""
from __future__ import annotations

import base64
import json
import logging
import os

import httpx

log = logging.getLogger(__name__)

VLM_BACKEND = os.environ.get("VLM_BACKEND", "openai").strip().lower()
VLM_URL = os.environ.get("VLM_URL", "http://localhost:8003/v1/chat/completions")
VLM_MODEL = os.environ.get("VLM_MODEL", "gemma-4-26B-A4B")
VLM_API_KEY = os.environ.get("VLM_API_KEY", "").strip()
VLM_TIMEOUT = float(os.environ.get("VLM_TIMEOUT", "120"))
VLM_TEMPERATURE = float(os.environ.get("VLM_TEMPERATURE", "0.3"))

# openai/vLLM-only family knob (chat_template_kwargs is a vLLM extension; invalid on hosted APIs).
# CONVENIENCE for the common vLLM+Qwen3 case only — the SAME flag behaves differently across Qwen
# variants (Qwen2.5 no thinking, Qwen3 hybrid, Instruct-2507 non-thinking, QwQ) AND vLLM versions
# (whether chat_template_kwargs is honored / reasoning split out). For anything off that path, use
# VLM_EXTRA_BODY below — the general escape hatch — and don't rely on this boolean.
_ET = os.environ.get("VLM_ENABLE_THINKING")
_CHAT_TEMPLATE_KWARGS = (
    {"enable_thinking": _ET.strip().lower() in ("1", "true", "yes", "on")}
    if _ET is not None else None)

# General request-extras passthrough: raw JSON merged (shallow, wins) into the openai payload, so ANY
# model+vLLM-version-specific fields work without a code change — e.g.
#   VLM_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'   (exact shape THIS model wants)
#   VLM_EXTRA_BODY='{"extra_body":{"top_k":20},"chat_template_kwargs":{"thinking":false}}'
# This is the durable answer to Qwen/vLLM variance; VLM_ENABLE_THINKING is just the common default.
_EXTRA_BODY: dict | None = None
_eb = os.environ.get("VLM_EXTRA_BODY")
if _eb:
    try:
        _parsed = json.loads(_eb)
        _EXTRA_BODY = _parsed if isinstance(_parsed, dict) else None
        if _EXTRA_BODY is None:
            log.warning("VLM_EXTRA_BODY is not a JSON object — ignored")
    except Exception:  # noqa: BLE001
        log.warning("VLM_EXTRA_BODY is not valid JSON — ignored", exc_info=True)


async def chat_image(system: str, user_text: str, image_png: bytes, *,
                     max_tokens: int, temperature: float | None = None) -> str:
    """Send (system + user_text + ONE image) to the configured VLM backend; return the model's raw
    text. The caller (`vlm.py`) parses the action/answer out of it — that logic is backend-agnostic."""
    temp = VLM_TEMPERATURE if temperature is None else temperature
    adapter = _ADAPTERS.get(VLM_BACKEND)
    if adapter is None:
        raise RuntimeError(
            f"unknown VLM_BACKEND={VLM_BACKEND!r}; known: {', '.join(sorted(_ADAPTERS))}")
    return await adapter(system, user_text, image_png, max_tokens=max_tokens, temperature=temp)


async def _openai(system: str, user_text: str, image_png: bytes, *,
                  max_tokens: int, temperature: float) -> str:
    """OpenAI-compatible /v1/chat/completions — covers vLLM AND any OpenAI-compatible API/proxy."""
    b64 = base64.b64encode(image_png).decode()
    payload: dict = {
        "model": VLM_MODEL, "max_tokens": max_tokens, "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ],
    }
    if _CHAT_TEMPLATE_KWARGS is not None:   # vLLM family knob (e.g. Qwen enable_thinking=false)
        payload["chat_template_kwargs"] = _CHAT_TEMPLATE_KWARGS
    if _EXTRA_BODY:                          # operator escape hatch — wins over the convenience knob
        payload.update(_EXTRA_BODY)
    headers = {"Authorization": f"Bearer {VLM_API_KEY}"} if VLM_API_KEY else {}
    async with httpx.AsyncClient(timeout=VLM_TIMEOUT) as client:
        r = await client.post(VLM_URL, json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def _anthropic(system: str, user_text: str, image_png: bytes, *,
                     max_tokens: int, temperature: float) -> str:
    """Anthropic Messages API (/v1/messages): different image block + response shape + x-api-key
    auth, and NO chat_template_kwargs. Extension point — implement when a native Anthropic vision
    backend is actually provisioned (kept explicit rather than a silently-wrong OpenAI shim)."""
    raise NotImplementedError(
        "VLM_BACKEND=anthropic not implemented yet — add the /v1/messages adapter here "
        "(base64 image source block, x-api-key + anthropic-version headers, parse content[0].text). "
        "Today's backends are vLLM/OpenAI-compatible; use VLM_BACKEND=openai.")


_ADAPTERS = {
    "openai": _openai,      # vLLM + OpenAI-compatible
    "vllm": _openai,        # alias — vLLM speaks the OpenAI wire format
    "anthropic": _anthropic,
}
