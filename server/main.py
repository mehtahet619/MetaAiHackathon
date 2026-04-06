"""
FastAPI server exposing the OpenEnv HTTP interface for TrafficControlEnv.

Endpoints:
  POST /reset            → TrafficObservation
  POST /step             → StepResult
  GET  /state            → TrafficState
  GET  /tasks            → list of task configs
  POST /grade            → {task_id, score, breakdown}
  GET  /health           → {status: ok}
"""

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .env import (
    TrafficAction,
    TrafficControlEnv,
    TrafficObservation,
    TrafficReward,
    TrafficState,
    StepResult,
    SignalPhase,
)
from pydantic import BaseModel

app = FastAPI(
    title="TrafficControlEnv",
    description="OpenEnv — Autonomous Intersection Traffic Control",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared environment instance (one per server; HF Space is single-container)
_env: TrafficControlEnv | None = None
_current_task = 1


def get_env() -> TrafficControlEnv:
    global _env
    if _env is None:
        _env = TrafficControlEnv(task_id=_current_task)
    return _env


# ─── Request/Response schemas ─────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = 1
    seed: int = 42


class GradeResponse(BaseModel):
    task_id: int
    task_name: str
    score: float
    breakdown: dict[str, Any]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "TrafficControlEnv", "version": "1.0.0"}


@app.post("/reset", response_model=TrafficObservation)
def reset(req: ResetRequest = ResetRequest()):
    global _env, _current_task
    if req.task_id not in (1, 2, 3):
        raise HTTPException(400, "task_id must be 1, 2, or 3")
    _current_task = req.task_id
    _env = TrafficControlEnv(task_id=req.task_id, seed=req.seed)
    return _env.reset()


@app.post("/step", response_model=StepResult)
def step(action: TrafficAction):
    env = get_env()
    result = env.step(action)
    return result


@app.get("/state", response_model=TrafficState)
def state():
    return get_env().state()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": tid,
                "name": cfg["name"],
                "description": cfg["description"],
                "max_steps": cfg["max_steps"],
                "difficulty": {1: "easy", 2: "medium", 3: "hard"}[tid],
            }
            for tid, cfg in TrafficControlEnv.TASK_CONFIGS.items()
        ]
    }


@app.post("/grade", response_model=GradeResponse)
def grade():
    env = get_env()
    score = env.grade()
    cfg = env.cfg
    cleared = len(env.cleared_vehicles)
    avg_wait = 0.0
    if env.cleared_vehicles:
        avg_wait = sum(v.wait_time for v in env.cleared_vehicles) / \
            len(env.cleared_vehicles)

    return GradeResponse(
        task_id=env.task_id,
        task_name=cfg["name"],
        score=score,
        breakdown={
            "total_cleared": cleared,
            "target_throughput": cfg["target_throughput"],
            "avg_wait_time": round(avg_wait, 2),
            "emergency_cleared": env.emergency_cleared,
            "steps_used": env.step_num,
            "max_steps": cfg["max_steps"],
        },
    )
