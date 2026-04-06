"""
TrafficControlEnv — Inference Script

Required env vars:
  OPENAI_API_KEY
  API_BASE_URL     e.g. https://api.openai.com/v1
  MODEL_NAME       e.g. gpt-4o-mini
  HF_TOKEN
  HF_SPACE_URL     e.g. https://mehtahet619-metaaihackathon.hf.space
"""

import json, os
import httpx
from openai import OpenAI

API_KEY      = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
HF_SPACE_URL = os.environ.get("HF_SPACE_URL", "http://localhost:7860")

TEMPERATURE = 0.2
MAX_TOKENS  = 256

SYSTEM_PROMPT = """You are a traffic signal controller at a 4-way intersection.
Minimise vehicle wait times. ALWAYS prioritise ambulances immediately.

Phases:
  ns_green — North+South flow, East+West stopped
  ew_green — East+West flow, North+South stopped
  all_red  — all stopped (avoid unless necessary)

Rules:
1. If emergency_waiting=true → set phase for that direction, hold_steps=1
2. Favour the direction with the longer queue
3. Do not hold one phase more than 8 steps

Respond ONLY with JSON like: {"phase": "ns_green", "hold_steps": 3}"""


# ── Logging (required format) ─────────────────────────────────────────────────

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(f"[STEP] step={step} action={json.dumps(action)} reward={reward:.4f} done={done} error={error}", flush=True)

def log_end(success, steps, score, rewards):
    avg = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"[END] success={success} steps={steps} score={score:.4f} avg_reward={avg:.4f} rewards={json.dumps([round(r,4) for r in rewards])}", flush=True)


# ── HTTP client ───────────────────────────────────────────────────────────────

class EnvClient:
    def __init__(self, base_url):
        self.base = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
        self.http = httpx.Client(base_url=self.base, headers=headers, timeout=30)

    def reset(self, task_id=1, seed=42):
        r = self.http.post("/reset", json={"task_id": task_id, "seed": seed})
        r.raise_for_status()
        return r.json()

    def step(self, phase, hold_steps):
        r = self.http.post("/step", json={"phase": phase, "hold_steps": hold_steps})
        r.raise_for_status()
        return r.json()

    def close(self):
        self.http.close()


# ── Agent ─────────────────────────────────────────────────────────────────────

def get_action(llm, obs, history):
    prompt = (
        f"Step {obs.get('step',0)} | Phase: {obs.get('current_phase')} held {obs.get('phase_duration',0)} steps\n"
        f"Queues: N={obs.get('north_queue',0)} S={obs.get('south_queue',0)} "
        f"E={obs.get('east_queue',0)} W={obs.get('west_queue',0)}\n"
        f"Emergency: waiting={obs.get('emergency_waiting')} "
        f"dir={obs.get('emergency_direction')} "
        f"waited={obs.get('emergency_wait_steps')} steps\n"
        f"Throughput: {obs.get('throughput',0)} | AvgWait: {obs.get('avg_wait_time',0)}\n"
        f"Recent:\n" + "\n".join(history[-4:])
    )
    try:
        resp = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        action = json.loads(raw)
        assert action.get("phase") in ("ns_green", "ew_green", "all_red")
        assert 1 <= int(action.get("hold_steps", 1)) <= 10
        return action
    except Exception as e:
        print(f"[DEBUG] agent parse error: {e}", flush=True)
        # Fallback greedy
        ns = obs.get("north_queue", 0) + obs.get("south_queue", 0)
        ew = obs.get("east_queue", 0)  + obs.get("west_queue", 0)
        if obs.get("emergency_waiting"):
            ed = obs.get("emergency_direction", "north")
            phase = "ns_green" if ed in ("north", "south") else "ew_green"
            return {"phase": phase, "hold_steps": 1}
        return {"phase": "ns_green" if ns >= ew else "ew_green", "hold_steps": 3}


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(task_id, llm, env):
    names    = {1: "BasicThroughput", 2: "EmergencyPriority", 3: "RushHour"}
    max_steps = {1: 60, 2: 80, 3: 120}[task_id]
    name = names[task_id]

    log_start(task=name, env="TrafficControlEnv", model=MODEL_NAME)

    raw = env.reset(task_id=task_id, seed=42)
    # handle both flat and nested observation responses
    obs = raw.get("observation", raw)

    history, rewards, steps_taken, score, success = [], [], 0, 0.0, False

    try:
        for step in range(1, max_steps + 1):
            if obs.get("step_budget", 1) <= 0:
                break

            action = get_action(llm, obs, history)
            raw    = env.step(action["phase"], action["hold_steps"])
            obs    = raw.get("observation", raw)
            reward = float(raw.get("reward") or 0.0)
            done   = bool(raw.get("done", False))

            rewards.append(reward)
            steps_taken = step
            history.append(
                f"Step {step}: phase={action['phase']} hold={action['hold_steps']} → r={reward:+.2f}"
            )

            log_step(step=step, action=action, reward=reward, done=done, error=None)

            if done:
                break

        # Score: clamp cumulative reward to [0, 1]
        max_possible = max_steps * 0.5
        score   = min(max(sum(rewards) / max_possible, 0.0), 1.0) if max_possible > 0 else 0.0
        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] task {task_id} error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EnvClient(HF_SPACE_URL)

    print(f"\n{'='*55}", flush=True)
    print(f"TrafficControlEnv Inference | model={MODEL_NAME}", flush=True)
    print(f"Space: {HF_SPACE_URL}", flush=True)
    print(f"{'='*55}\n", flush=True)

    scores = {}
    for task_id in [1, 2, 3]:
        print(f"\n--- Task {task_id} ---", flush=True)
        scores[task_id] = run_task(task_id, llm, env)
        print(f"Task {task_id} score: {scores[task_id]:.3f}", flush=True)

    mean = sum(scores.values()) / len(scores)
    print(f"\n{'='*55}", flush=True)
    print(f"SCORES: {scores}", flush=True)
    print(f"MEAN:   {mean:.3f}", flush=True)
    print(f"{'='*55}\n", flush=True)

    env.close()


if __name__ == "__main__":
    main()
