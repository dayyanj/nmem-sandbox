"""FastAPI sandbox service — the API an agent (or a curl) drives the desktop through.

  POST /session {task, mode, max_steps, wall_clock_s} -> {session_id}
  GET  /session/{id}                                   -> full session state
  POST /session/{id}/abort                             -> kill-switch
  GET  /sessions                                       -> list
  GET  /health

Global concurrency is capped at 1 — one agent driving the single desktop at a time.
"""
import asyncio
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import frames
from .agent import Session, run_session

app = FastAPI(title="nmem-sandbox-harness")

_sessions: dict[str, Session] = {}
_tasks: dict[str, asyncio.Task] = {}


class SessionReq(BaseModel):
    task: str
    mode: str = "read_only"          # read_only | allow_writes (writes gated in Phase 3)
    max_steps: int = 20
    wall_clock_s: float = 300.0


def _busy() -> bool:
    return any(s.status in ("queued", "running") for s in _sessions.values())


@app.get("/health")
async def health():
    return {"ok": True, "busy": _busy(), "sessions": len(_sessions)}


@app.post("/session")
async def create_session(req: SessionReq):
    if _busy():
        raise HTTPException(status_code=409, detail="A session is already running (concurrency cap = 1)")
    if not req.task.strip():
        raise HTTPException(status_code=400, detail="task is required")
    session = Session(task=req.task, mode=req.mode,
                      # Ceiling raised 60→120: the flat cap is now a safety backstop, not a working
                      # limit — productive multi-hop research legitimately runs long, and the
                      # escalating nudge (agent.py) pushes genuinely-stuck sessions to conclude first.
                      max_steps=max(1, min(req.max_steps, 120)),
                      wall_clock_s=max(10.0, min(req.wall_clock_s, 1800.0)))
    _sessions[session.id] = session
    _tasks[session.id] = asyncio.create_task(run_session(session))
    return {"session_id": session.id, "status": session.status}


@app.get("/session/{sid}")
async def get_session(sid: str):
    s = _sessions.get(sid)
    if not s:
        raise HTTPException(status_code=404, detail="unknown session")
    return s.public()


@app.get("/session/{sid}/frame/{i}")
async def get_frame(sid: str, i: int):
    """Fetch one persisted keyframe (downscaled JPEG). The agent pulls only frames worth
    keeping — the step trace carries the phash/dims so it can decide before transferring pixels."""
    if sid not in _sessions:
        raise HTTPException(status_code=404, detail="unknown session")
    p = frames.frame_path(sid, i)
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="no frame for that step")
    return FileResponse(p, media_type="image/jpeg")


@app.post("/session/{sid}/abort")
async def abort_session(sid: str):
    s = _sessions.get(sid)
    if not s:
        raise HTTPException(status_code=404, detail="unknown session")
    s.abort = True
    return {"ok": True, "status": s.status}


@app.get("/sessions")
async def list_sessions():
    return {"sessions": [s.public() for s in _sessions.values()]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
