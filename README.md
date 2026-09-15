# nmem-sandbox

The **canonical, LLM-agnostic computer-use sandbox** for nmem agents — a headless Linux desktop a
vision model drives by `screenshot → action → xdotool`. The agent POSTs an objective and polls; it
never runs the loop itself. This is the appliance's sandbox **server** (the client is generic:
`nmem/agent_core/actors/computer_use.py` — `SandboxClient` / `drive_sandbox`).

One image serves **every** agent — differing only by env — so a Gemma-backed agent and a Qwen-backed
agent run the same image with different `VLM_*` settings.

## Run
```
docker build -t nmem-sandbox:dev .
# Gemma-family backend:
SANDBOX_NAME=my_sandbox VLM_URL=http://<vlm-host>:8003/v1/chat/completions \
  VLM_MODEL=gemma-4-26B-A4B ./run.sh
# Qwen-family backend (needs thinking disabled):
SANDBOX_NAME=my_sandbox VLM_URL=http://<vlm-host>:8001/v1/chat/completions \
  VLM_MODEL=Qwen3.8-27B VLM_ENABLE_THINKING=false ./run.sh
```

## Backend-agnostic
The computer-use **action contract** (`harness/vlm.py`) is provider-neutral; the **wire format** is a
pluggable adapter (`harness/backends.py`) selected by `VLM_BACKEND`. Today: `openai` (covers **vLLM
and any OpenAI-compatible API/proxy** — every backend in use). `anthropic` is a documented extension
point (native `/v1/messages` — implement when actually provisioned). Adding a provider is one adapter,
never a fork. Config is env (one backend per sandbox container — the appliance convention), not a file.

Model/version variance is handled two ways, so we never depend on one served model behaving: request
side, `VLM_EXTRA_BODY` (raw JSON passthrough) lets the operator send whatever `chat_template_kwargs`/
`extra_body` THIS Qwen-variant + vLLM-version wants; response side, the action parser **defensively
strips inline `<think>…</think>`**, so the action JSON is recovered even if the server left reasoning
in `content` (wrong/older `--reasoning-parser`, or a template that ignored `enable_thinking`).

## Config (all env — nothing agent-specific in code)
| var | default | meaning |
|---|---|---|
| `VLM_BACKEND` | `openai` | wire adapter: `openai` (vLLM/OpenAI-compat) \| `anthropic` (extension point) |
| `VLM_URL` | — (required) | served vision-model endpoint (openai: `/v1/chat/completions`) |
| `VLM_MODEL` | — (required) | served model name |
| `VLM_API_KEY` | unset | bearer/api-key for a hosted API; blank for local vLLM |
| `VLM_ENABLE_THINKING` | unset | **Convenience only** (common vLLM+Qwen3 case): set `false` → adds `chat_template_kwargs.enable_thinking=false`; UNSET for Gemma. Behaves differently across Qwen variants + vLLM versions — for anything off the common path use `VLM_EXTRA_BODY`. |
| `VLM_EXTRA_BODY` | unset | **Escape hatch for model/vLLM variance.** Raw JSON merged (shallow, wins) into the request — set the exact `chat_template_kwargs`/`extra_body`/etc. THIS model+vLLM needs, e.g. `{"chat_template_kwargs":{"enable_thinking":false}}`. |
| `VLM_TEMPERATURE` | `0.3` | slight sampling avoids deterministic ruts |
| `VLM_TIMEOUT` | `120` | per-call seconds |
| `SCREEN_W` / `SCREEN_H` | `1280`/`800` | Xvfb geometry (runtime) |
| `SANDBOX_CAPTURE_FRAMES` | `1` | persist a downscaled keyframe + perceptual dhash per step (visual memory) |
| `SANDBOX_NAME` / `SANDBOX_IMAGE` / `SANDBOX_ARTIFACTS` | `nmem_sandbox` / `nmem-sandbox:dev` / `~/<name>_artifacts` | container name / image / host artifact mount |

## Harness API
- `POST /session {task, mode, max_steps, wall_clock_s}` → `{session_id}`
- `GET  /session/{id}` → status + full step log (+ per-step `frame:{frame_id,phash,screen_changed}` when capture is on) + result
- `GET  /session/{id}/frame/{i}` → one captured keyframe (downscaled JPEG) — the agent pulls only worth-keeping frames (visual memory)
- `POST /session/{id}/abort` → kill-switch; `GET /health`, `GET /sessions`

Watch live: `ssh -L 6080:127.0.0.1:6080 <host>`, open `http://localhost:6080/vnc.html`.

## Visual memory (frame capture)
With `SANDBOX_CAPTURE_FRAMES=1`, each step persists a downscaled JPEG + perceptual dhash +
`screen_changed`, and exposes `GET /session/{id}/frame/{i}`. The agent's generic visual-memory sink
(`nmem.agent_core.visual_memory`) pulls only gated frames into sensory memory. **In-loop read-back
(P6b, planned):** the harness will additionally call the agent's `POST /visual-memory/seen-fail
{phash}` per step and, on a hit, inject "you failed on this screen before — try differently" into the
VLM prompt — a mid-pursuit redirect. That endpoint already exists agent-side; the harness callback is
the remaining piece.

## Security posture (founder-decided 2026-08-31)
Goal: **explore freely, don't harm.** Safety comes from *actions, credentials, and audit*, not from
caging the network.
- **No third-party credentials** baked in — the agent cannot authenticate into systems Spwig doesn't
  own. Spwig-owned creds are injected per-session, scoped, ephemeral.
- **Rules of engagement** in the system prompt (`harness/vlm.py`): researcher/operator not attacker;
  never bypass auth, exploit vulns, brute-force, scan/probe, or touch unauthorized systems.
- **Network: open** (founder decision) — exploration unrestricted; the guardrail is no-creds +
  rules-of-engagement + audit, not a firewall.
- **In-sandbox gating: none** — write-vs-read judgment lives at the agent layer + audit.
- **Audit is the backstop** — per-session mp4 + JSON action log (host-mounted), `max_steps` +
  `wall_clock_s` bounds, `/abort` kill-switch, global concurrency cap = 1.

Re-tightening later (egress rules, write-approval gate) is a config/flag change, not a redesign.

## The nmem suite

nmem-sandbox is one library in a family of composable, framework-agnostic cognitive layers for AI agents. Each is standalone — mix in only the ones you need.

| Repo | Layer | What it does |
|------|-------|--------------|
| [nmem](https://github.com/dayyanj/nmem) | Memory | Hierarchical, self-refining cognitive memory — 6 tiers + a consolidation engine |
| [nmem-sym](https://github.com/dayyanj/nmem-sym) | Reasoning | Symbolic cognition — typed knowledge graph, spreading activation, drives, prediction |
| [nmem-sym-sensor](https://github.com/dayyanj/nmem-sym-sensor) | Perception | Sensory memory — unlabeled visual/audio primitives clustered into grounded concepts |
| [nmem-identity](https://github.com/dayyanj/nmem-identity) | Perception | Self-supervised person identity (voice + face) — learns who it's talking to, no manual tagging |
| [nmem-act](https://github.com/dayyanj/nmem-act) | Action | Typed action/outcome contract + tiered autonomy gate — act to learn, safely |
| [nmem-sandbox](https://github.com/dayyanj/nmem-sandbox) | Action | LLM-agnostic computer-use sandbox — a headless desktop a vision model drives |
| [nmem-exchange](https://github.com/dayyanj/nmem-exchange) | Comms | End-to-end-secured message bus — agents talk without sharing memory |
| [nmem-immune](https://github.com/dayyanj/nmem-immune) | Integrity | Memory immune system — poisoning detection, drift, quarantine |
| [nmem-viz](https://github.com/dayyanj/nmem-viz) | Tooling | Real-time 3D "brain" visualization of any nmem agent |

**Just want to run one?** [**nmem-studio**](https://huggingface.co/dayyanj/nmem-studio) is the pull-and-run appliance — a single Docker image that stands up one fully-configured agent from a web wizard (memory + reasoning, plus optional identity and embodied perception), no config files by hand.
