#!/usr/bin/env python3
"""
TrafficControlEnv — Inference Script
Runs an LLM agent against all 3 tasks and reports scores.

Required env vars:
  OPENAI_API_KEY   — API key (or compatible key)
  API_BASE_URL     — LLM base URL (e.g. https://api.openai.com/v1)
  MODEL_NAME       — Model identifier (e.g. gpt-4o-mini)
  HF_TOKEN         — Hugging Face token (for authenticated Space access)
  HF_SPACE_URL     — Base URL of your deployed HF Space
"""

import asyncio
import json
import os
import sys
import time
from typing import Any

import httpx
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────

API_KEY      = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
HF_SPACE_URL = os.environ.get("HF_SPACE_URL", "http://localhost:7860")

MAX_STEPS   = 60
TEMPERATURE = 0.2
MAX_TOKENS  = 512

SYSTEM_PROMPT = """You are an autonomous traffic signal controller at a 4-way intersection.

Your job: set signal phases to minimize vehicle wait times, prevent lane overflow,
and ALWAYS prioritize emergency vehicles (ambulances) by giving them green light ASAP.

Signal phases:
- ns_green: North-South lanes flow, East-West stops
- ew_green: East-West lanes flow, North-South stops
- all_red: All lanes stopped (use only for brief transitions or emergencies)

Rules:
1. If emergency_waiting=true, set the phase for that direction IMMEDIATELY (hold_steps=1)
2. Balance long queues: favor the direction with more vehicles
3. Don't hold one phase too long (>8 steps) — avoid starvation
4. Typical good hold duration: 3-5 steps per phase

Respond ONLY with a JSON object like:
{"phase": "ns_green", "hold_steps": 4}

No explanation, no markdown, just the JSON.
"""

# ─── Logging (required format) ─────────────────────────────────────────────────

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: Any, reward: float, done: bool, error: Any = None):
    print(
        f"[STEP] step={step} action={json.dumps(action)} "
        f"reward={reward:.4f} done={done} error={error}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: list[float]):
    avg_r = sum(rewards) / len(rewards) if rewards else 0.0
    print(
        f"[END] success={success} steps={steps} score={score:.4f} "
        f"avg_reward={avg_r:.4f} rewards={json.dumps([round(r,4) for r in rewards])}",
        flush=True,
    )

# ─── Environment client ────────────────────────────────────────────────────────

class TrafficEnvClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"
        self.client = httpx.Client(base_url=self.base, headers=headers, timeout=30)

    def reset(self, task_id: int = 1, seed: int = 42) -> dict:
        r = self.client.post("/reset", json={"task_id": task_id, "seed": seed})
        r.raise_for_status()
        return r.json()

    def step(self, phase: str, hold_steps: int) -> dict:
        r = self.client.post("/step", json={"phase": phase, "hold_steps": hold_steps})
        r.raise_for_status()
        return r.json()

    def grade(self) -> dict:
        r = self.client.post("/grade")
        r.raise_for_status()
        return r.json()

    def close(self):
        self.client.close()

# ─── Agent ────────────────────────────────────────────────────────────────────

def build_prompt(obs: dict, history: list[str]) -> str:
    lanes = obs.get("lanes", {})
    lane_summary = " | ".join(
        f"{d}:{info['queue_length']}" + (" [EMG!]" if info.get("has_emergency") else "")
        for d, info in lanes.items()
    )
    history_block = "\n".join(history[-5:]) if history else "none"
    emerg = "YES — direction: " + str(obs.get("emergency_direction")) if obs.get("emergency_waiting") else "no"

    return f"""Step {obs['step']}/{obs['step'] + obs['step_budget']}
Current phase: {obs['current_phase']} (held {obs['phase_duration']} steps)
Queues [{lane_summary}]
Emergency: {emerg} | Wait: {obs['emergency_wait_steps']} steps
Avg wait: {obs['avg_wait_time']:.1f} | Throughput: {obs['throughput']}
Recent actions:
{history_block}

Choose next phase and hold_steps:"""


def get_agent_action(client: OpenAI, obs: dict, history: list[str]) -> dict:
    prompt = build_prompt(obs, history)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Strip markdown if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        action = json.loads(raw)
        # Validate
        assert action.get("phase") in ("ns_green", "ew_green", "all_red")
        assert 1 <= int(action.get("hold_steps", 1)) <= 10
        return action
    except Exception as e:
        print(f"[DEBUG] Agent parse error: {e}", flush=True)
        # Fallback: safe default
        obs_dirs = obs.get("lanes", {})
        # Pick phase for busiest direction
        ns = obs_dirs.get("north", {}).get("queue_length", 0) + obs_dirs.get("south", {}).get("queue_length", 0)
        ew = obs_dirs.get("east",  {}).get("queue_length", 0) + obs_dirs.get("west",  {}).get("queue_length", 0)
        phase = "ns_green" if ns >= ew else "ew_green"
        # Emergency override
        if obs.get("emergency_waiting"):
            ed = obs.get("emergency_direction", "north")
            phase = "ns_green" if ed in ("north", "south") else "ew_green"
        return {"phase": phase, "hold_steps": 3}


# ─── Task runner ───────────────────────────────────────────────────────────────

def run_task(task_id: int, llm_client: OpenAI, env: TrafficEnvClient) -> float:
    task_names = {1: "BasicThroughput", 2: "EmergencyPriority", 3: "RushHour"}
    task_name = task_names.get(task_id, f"Task{task_id}")
    max_steps = {1: 60, 2: 80, 3: 120}[task_id]
    max_reward_est = {1: 30.0, 2: 40.0, 3: 80.0}[task_id]

    log_start(task=task_name, env="TrafficControlEnv", model=MODEL_NAME)

    obs = env.reset(task_id=task_id, seed=42)
    history: list[str] = []
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        for step in range(1, max_steps + 1):
            if obs.get("step_budget", 1) <= 0:
                break

            action = get_agent_action(llm_client, obs, history)
            result = env.step(action["phase"], action["hold_steps"])

            obs     = result["observation"]
            reward_val = result["reward"]["value"]
            done    = result["done"]
            error   = None

            rewards.append(reward_val)
            steps_taken = step
            history.append(
                f"Step {step}: phase={action['phase']} hold={action['hold_steps']} → r={reward_val:+.2f}"
            )

            log_step(step=step, action=action, reward=reward_val, done=done, error=error)

            if done:
                break

        # Grade via grader endpoint
        grade_resp = env.grade()
        score = grade_resp["score"]
        success = score >= 0.6

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = TrafficEnvClient(HF_SPACE_URL)

    print(f"\n{'='*60}", flush=True)
    print(f"TrafficControlEnv — Baseline Inference", flush=True)
    print(f"Model: {MODEL_NAME} | Space: {HF_SPACE_URL}", flush=True)
    print(f"{'='*60}\n", flush=True)

    task_scores = {}
    for task_id in [1, 2, 3]:
        print(f"\n--- Task {task_id} ---", flush=True)
        score = run_task(task_id, llm_client, env)
        task_scores[task_id] = score
        print(f"Task {task_id} score: {score:.3f}", flush=True)

    mean_score = sum(task_scores.values()) / len(task_scores)
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL SCORES: {task_scores}", flush=True)
    print(f"MEAN SCORE:   {mean_score:.3f}", flush=True)
    print(f"{'='*60}\n", flush=True)

    env.close()


if __name__ == "__main__":
    main()
