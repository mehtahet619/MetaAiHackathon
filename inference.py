#!/usr/bin/env python3
"""
TrafficControlEnv — Inference Script v2
Changes from v1:
  ① Smart reward awareness — agent prompt explains all reward components
  ② Emergency preemption   — agent sets emergency_preempt=true when urgency > 0.7
  ③ Multi-agent task 4     — sends intersection_phases for all 4 intersections,
                             coordinates green-wave patterns
Required env vars:
  OPENAI_API_KEY   — API key
  API_BASE_URL     — LLM base URL (default: https://api.openai.com/v1)
  MODEL_NAME       — Model name  (default: gpt-4o-mini)
  HF_TOKEN         — Hugging Face token
  HF_SPACE_URL     — Base URL of deployed HF Space
"""

import json
import os
from typing import Any

import httpx
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────

API_KEY      = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN")
HF_SPACE_URL = os.environ.get("HF_SPACE_URL", "http://localhost:7860")

TEMPERATURE = 0.2
MAX_TOKENS  = 768
VALID_PHASES = {"ns_green", "ew_green", "all_red"}

# ─── System prompts ────────────────────────────────────────────────────────────

# ① Smart-reward-aware prompt for tasks 1-3
SYSTEM_SINGLE = """You control one 4-way traffic signal intersection.
## Reward components (all matter):
  + throughput_bonus   — weighted: bus=2x, truck=1.5x, ambulance=5x, car=1x
  - wait_penalty       — EXPONENTIAL after 6 steps; long waits punish hard
  + balance_bonus      — reward for keeping all 4 lanes roughly equal
  + emergency_bonus    — up to +8.0 for clearing ambulance early; -3.0 if late
  - urgency_penalty    — up to -0.5/step while ambulance waits (sigmoid)
  - starvation_penalty — -0.5 per vehicle beyond queue cap of 15
  - phase_switch_cost  — -0.25 if you switch before holding 2 steps
## Phases:
  ns_green  North+South flow, East+West stop
  ew_green  East+West flow, North+South stop
  all_red   All stop (only for emergency transitions)
## Decision rules (in priority order):
1. If emergency_urgency > 0.7 OR preemption_active=true:
   set phase for ambulance direction AND emergency_preempt=true immediately.
   (north/south ambulance → ns_green; east/west → ew_green)
2. Avoid waits > 6 steps — exponential penalty kicks in fast.
3. Pick the busiest axis; keep queues balanced.
4. Hold 3-6 steps. Never switch before 2 steps (pays switch cost).
5. Use all_red only when switching directions for an incoming ambulance.
Respond ONLY with valid JSON, no markdown:
{"phase":"ns_green","hold_steps":4,"emergency_preempt":false}
"""

# ③ Multi-agent prompt for task 4
SYSTEM_MULTI = """You coordinate 4 traffic signal intersections in a 2x2 grid.
## Grid layout:
  [I0 NW] ── [I1 NE]
     |              |
  [I2 SW] ── [I3 SE]
Vehicles cleared from one intersection can flow into adjacent ones (40% probability).
## Green Wave Bonus (+0.8/step when active):
Coordinate adjacent pairs to the same phase:
  Horizontal wave: I0=I1=ns_green AND I2=I3=ns_green  (or both ew_green)
  Vertical wave:   I0=I2=ns_green AND I1=I3=ns_green  (or both ew_green)
Pick whichever axis has more total vehicles.
## ① Reward components (same as single-agent, multiplied across 4 intersections):
  + weighted throughput, + balance bonus, + green_wave_bonus
  - exponential wait penalty, - urgency penalty, - starvation penalty
## ② Emergency preemption:
If any intersection has emergency_urgency > 0.7:
  - Set that intersection's phase for the ambulance direction
  - Set emergency_preempt=true for that intersection
  - Set the same phase for downstream intersections on the ambulance's route
    (ambulance traveling south: set I0→I2 or I1→I3 to ns_green)
Respond ONLY with valid JSON (exactly 4 entries each), no markdown:
{"intersection_phases":["ns_green","ns_green","ns_green","ns_green"],
 "intersection_hold_steps":[4,4,4,4],
 "intersection_preempt":[false,false,false,false],
 "phase":"ns_green","hold_steps":4,"emergency_preempt":false}
(The plain phase/hold_steps/emergency_preempt are required for backward compat
and should match I0 values.)
"""

# ─── Logging ───────────────────────────────────────────────────────────────────

def log_start(task: str, model: str):
    print(f"START task={task} model={model}", flush=True)

def log_step(step: int, action: Any, reward: float, done: bool, info: str = ""):
    print(f"STEP step={step} action={json.dumps(action)} reward={reward:.4f} done={done}{' '+info if info else ''}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list[float]):
    avg = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"END success={success} steps={steps} score={score:.4f} avg_reward={avg:.4f}", flush=True)

# ─── HTTP client ───────────────────────────────────────────────────────────────

class EnvClient:
    def __init__(self, base: str):
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        self.c = httpx.Client(base_url=base.rstrip("/"), headers=headers, timeout=30)

    def reset(self, task_id: int, seed: int = 42) -> dict:
        r = self.c.post("/reset", json={"task_id": task_id, "seed": seed})
        r.raise_for_status()
        return r.json()

    def step(self, action: dict) -> dict:
        r = self.c.post("/step", json=action)
        r.raise_for_status()
        return r.json()

    def grade(self) -> dict:
        r = self.c.post("/grade")
        r.raise_for_status()
        return r.json()

    def close(self):
        self.c.close()

# ─── Prompt builders ───────────────────────────────────────────────────────────

def _lane_str(lanes: dict) -> str:
    parts = []
    for d, info in lanes.items():
        tag = ""
        if info.get("has_emergency"):
            tag = f" [EMG urg={info.get('emergency_urgency', 0):.2f}]"
        parts.append(f"{d}:{info.get('queue_length', 0)}{tag}")
    return " | ".join(parts)

def prompt_single(obs: dict, history: list[str]) -> str:
    urg = obs.get("emergency_urgency", 0.0)
    emerg_str = "NO"
    if obs.get("emergency_waiting"):
        emerg_str = (
            f"YES dir={obs.get('emergency_direction')} "
            f"waited={obs.get('emergency_wait_steps')}steps "
            f"urgency={urg:.2f}"
            + (" ⚠️ ACT NOW" if urg > 0.7 else "")
        )
    held_warn = "  ⚠️ TOO LONG" if obs.get("phase_held_too_long") else ""
    history_str = "\n".join(history[-5:]) if history else "none"

    return (
        f"Step {obs.get('step')}/{obs.get('step',0)+obs.get('step_budget',0)} | {obs.get('task_name','')}\n"
        f"Phase: {obs.get('current_phase')} (held {obs.get('phase_duration')} steps){held_warn}\n"
        f"Queues [{_lane_str(obs.get('lanes', {}))}]\n"
        f"Emergency: {emerg_str}\n"
        f"Balance: {obs.get('queue_balance_score',1.0):.2f} | AvgWait: {obs.get('avg_wait_time',0):.1f} | Throughput: {obs.get('throughput',0)}\n"
        f"Recent:\n{history_str}\n\nDecide:"
    )

def prompt_multi(obs: dict, history: list[str]) -> str:
    snaps = obs.get("intersection_snapshots") or []
    lines = []
    for s in snaps:
        urg = s.get("emergency_urgency", 0.0)
        lines.append(
            f"  I{s['id']} [{s['phase']} held={s['phase_duration']}]: "
            f"N={s['north_queue']} S={s['south_queue']} E={s['east_queue']} W={s['west_queue']}"
            + (f" | EMG urg={urg:.2f}{'⚠️' if urg>0.7 else ''}" if s.get("emergency_waiting") else "")
        )
    gw = "✓ ACTIVE" if obs.get("green_wave_active") else "✗ inactive"
    history_str = "\n".join(history[-4:]) if history else "none"

    return (
        f"Step {obs.get('step')}/{obs.get('step',0)+obs.get('step_budget',0)} | MultiAgentGrid\n"
        f"Network throughput: {obs.get('network_throughput',0)} | Green wave: {gw}\n"
        "\n".join(lines) + "\n"
        f"Recent:\n{history_str}\n\nCoordinate all 4:"
    )

# ─── Action parsers ────────────────────────────────────────────────────────────

def _fallback_single(obs: dict) -> dict:
    """② Fallback with emergency preemption logic and adaptive hold strategy."""
    urg = obs.get("emergency_urgency", 0.0)
    ed  = obs.get("emergency_direction")

    # (A+D) Emergency preemption — highest priority
    if urg > 0.7 and ed:
        ph = "ns_green" if ed in ("north", "south") else "ew_green"
        return {"phase": ph, "hold_steps": 2, "emergency_preempt": True}

    # (B) Smarter queue balancing — explicit NS vs EW totals
    lanes    = obs.get("lanes", {})
    total_ns = (lanes.get("north", {}).get("queue_length", 0) +
                lanes.get("south", {}).get("queue_length", 0))
    total_ew = (lanes.get("east",  {}).get("queue_length", 0) +
                lanes.get("west",  {}).get("queue_length", 0))

    # Prefer the direction with the higher total queue;
    # only switch preference if the other axis is meaningfully larger
    if total_ns >= total_ew:
        phase    = "ns_green"
        dominant = total_ns
        minor    = total_ew
    else:
        phase    = "ew_green"
        dominant = total_ew
        minor    = total_ns

    # (A) Adaptive hold_steps based on queue imbalance
    diff = dominant - minor
    if diff >= 6:
        # One axis heavily loaded — serve it longer
        hold_steps = 5
    elif diff >= 3:
        hold_steps = 4
    else:
        # Queues are roughly balanced — rotate quickly to prevent starvation
        hold_steps = 3

    # Clamp to valid range [1, 10]
    hold_steps = max(1, min(10, hold_steps))

    return {"phase": phase, "hold_steps": hold_steps, "emergency_preempt": False}


def _fallback_multi(obs: dict) -> dict:
    """③ Fallback: true green wave across all intersections + emergency propagation."""
    snaps = obs.get("intersection_snapshots") or [{}] * 4

    # (D) Scan all intersections for any active emergency first
    emerg_phase = None
    emerg_idx   = None
    for s in snaps:
        urg = s.get("emergency_urgency", 0.0)
        ed  = s.get("emergency_direction")
        if urg > 0.7 and ed:
            emerg_phase = "ns_green" if ed in ("north", "south") else "ew_green"
            emerg_idx   = s.get("id", 0)
            break  # first/highest-urgency emergency wins

    if emerg_phase is not None:
        # Propagate emergency phase to ALL intersections to clear the entire route
        phases  = [emerg_phase] * 4
        hold    = [2] * 4
        preempt = [False] * 4
        if emerg_idx is not None and 0 <= emerg_idx < 4:
            preempt[emerg_idx] = True
        return {
            "phase":                    phases[0],
            "hold_steps":               hold[0],
            "emergency_preempt":        preempt[0],
            "intersection_phases":      phases,
            "intersection_hold_steps":  hold,
            "intersection_preempt":     preempt,
        }

    # (C) Compute global NS and EW load across all intersections
    total_ns = 0
    total_ew = 0
    for s in snaps:
        total_ns += s.get("north_queue", 0) + s.get("south_queue", 0)
        total_ew += s.get("east_queue",  0) + s.get("west_queue",  0)

    # Choose dominant direction globally — set ALL intersections to the same
    # phase to maximise the green-wave bonus (+0.8/step)
    dominant = "ns_green" if total_ns >= total_ew else "ew_green"
    phases   = [dominant, dominant, dominant, dominant]

    # (A) Adaptive hold_steps based on global queue imbalance
    diff = abs(total_ns - total_ew)
    if diff >= 10:
        hold_val = 5
    elif diff >= 5:
        hold_val = 4
    else:
        hold_val = 3
    hold_val = max(1, min(10, hold_val))

    hold    = [hold_val] * 4
    preempt = [False] * 4

    return {
        "phase":                    phases[0],
        "hold_steps":               hold[0],
        "emergency_preempt":        preempt[0],
        "intersection_phases":      phases,
        "intersection_hold_steps":  hold,
        "intersection_preempt":     preempt,
    }


def parse_action(raw: str, obs: dict, task_id: int) -> dict:
    try:
        raw = raw.replace("```json", "").replace("```", "").strip()
        a = json.loads(raw)
        if task_id == 4:
            phases = a.get("intersection_phases", [a.get("phase", "ns_green")] * 4)
            hold   = a.get("intersection_hold_steps", [a.get("hold_steps", 3)] * 4)
            pre    = a.get("intersection_preempt", [a.get("emergency_preempt", False)] * 4)
            assert all(p in VALID_PHASES for p in phases), "bad phase"
            assert all(1 <= h <= 10 for h in hold), "bad hold"
            return {
                "phase": phases[0], "hold_steps": int(hold[0]), "emergency_preempt": bool(pre[0]),
                "intersection_phases": phases[:4],
                "intersection_hold_steps": [int(h) for h in hold[:4]],
                "intersection_preempt": [bool(p) for p in pre[:4]],
            }
        else:
            assert a.get("phase") in VALID_PHASES
            assert 1 <= int(a.get("hold_steps", 3)) <= 10
            a.setdefault("emergency_preempt", False)
            return a
    except Exception as e:
        print(f"[DEBUG] parse error: {e}", flush=True)
        return _fallback_multi(obs) if task_id == 4 else _fallback_single(obs)

def get_action(llm: OpenAI, obs: dict, history: list[str], task_id: int) -> dict:
    system = SYSTEM_MULTI if task_id == 4 else SYSTEM_SINGLE
    user   = prompt_multi(obs, history) if task_id == 4 else prompt_single(obs, history)
    try:
        resp = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        raw = ""
    return parse_action(raw, obs, task_id)

# ─── Task runner ───────────────────────────────────────────────────────────────

TASK_NAMES = {1: "BasicThroughput", 2: "EmergencyPriority", 3: "RushHour", 4: "MultiAgentGrid"}
MAX_STEPS  = {1: 60, 2: 80, 3: 120, 4: 150}

def run_task(task_id: int, llm: OpenAI, env: EnvClient) -> float:
    log_start(task=TASK_NAMES[task_id], model=MODEL_NAME)
    obs = env.reset(task_id=task_id, seed=42)
    history, rewards = [], []
    steps_taken, score, success = 0, 0.0, False

    try:
        for step in range(1, MAX_STEPS[task_id] + 1):
            if obs.get("step_budget", 1) <= 0:
                break
            action = get_action(llm, obs, history, task_id)
            result = env.step(action)
            obs    = result.get("observation", result)
            rval   = result.get("reward", {})
            if isinstance(rval, dict):
                reward_val = rval.get("value", 0.0)
            else:
                reward_val = float(rval)
            done = result.get("done", False)
            rewards.append(reward_val)
            steps_taken = step

            if task_id == 4:
                history.append(
                    f"Step {step}: phases={action.get('intersection_phases')} gw={obs.get('green_wave_active')} r={reward_val:+.2f}"
                )
            else:
                history.append(
                    f"Step {step}: phase={action.get('phase')} hold={action.get('hold_steps')} preempt={action.get('emergency_preempt')} r={reward_val:+.2f}"
                )
            log_step(step, action, reward_val, done)
            if done:
                break

        grade = env.grade()
        score = grade.get("score", 0.0)
        success = score >= 0.6
    except Exception as e:
        print(f"[DEBUG] task {task_id} error: {e}", flush=True)
    finally:
        log_end(success, steps_taken, score, rewards)
    return score

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EnvClient(HF_SPACE_URL)

    print(f"\n{'='*60}")
    print(f"TrafficControlEnv v2 | Model: {MODEL_NAME}")
    print(f"{'='*60}\n")

    scores = {}
    for tid in [1, 2, 3, 4]:
        print(f"\n--- Task {tid}: {TASK_NAMES[tid]} ---")
        scores[tid] = run_task(tid, llm, env)
        print(f"Score: {scores[tid]:.3f}")

    mean = sum(scores.values()) / len(scores)
    print(f"\n{'='*60}")
    for tid, s in scores.items():
        print(f"  Task {tid} ({TASK_NAMES[tid]}): {s:.3f}")
    print(f"  MEAN: {mean:.3f}")
    print(f"{'='*60}\n")
    env.close()

if __name__ == "__main__":
    main()