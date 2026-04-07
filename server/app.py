import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openenv.core.env_server.http_server import create_app
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi import Request

from models import TrafficControlAction, TrafficControlObservation
from traffic_control_env_environment import TrafficControlEnvironment, TASK_CONFIGS

app = create_app(
    TrafficControlEnvironment,
    TrafficControlAction,
    TrafficControlObservation,
    env_name="traffic_control_env",
    max_concurrent_envs=4,
)

# ─── Static UI ────────────────────────────────────────────────────────────────

STATIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static"
)

@app.get("/")
def root():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return JSONResponse({"status": "ok", "env": "TrafficControlEnv", "version": "2.0.0"})

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─── ① Reward component breakdown ─────────────────────────────────────────────

@app.get("/reward/components")
def reward_components():
    """Explains every reward component so agents/developers know what to optimise."""
    return {
        "components": {
            "throughput_bonus":    "Weighted vehicle throughput (car=1x, truck=1.5x, bus=2x, ambulance=5x)",
            "wait_penalty":        "Exponential penalty for vehicles waiting > 6 steps: 0.05 * exp(0.08*(w-6))",
            "balance_bonus":       "Reward for keeping all 4 lane queues evenly loaded (std-dev based)",
            "emergency_bonus":     "+8 cleared in first 50% of deadline, scaled down, -3 if late",
            "urgency_penalty":     "Sigmoid 0-0.5/step while ambulance is waiting (grows near deadline)",
            "starvation_penalty":  "-0.5 per vehicle exceeding queue cap of 15",
            "phase_switch_penalty":f"-{0.25} if phase switched after being held < 2 steps",
            "green_wave_bonus":    f"+{0.8}/step when adjacent intersections share phase (task 4 only)",
        },
        "auto_preemption": {
            "threshold": 0.75,
            "description": "When emergency_urgency >= 0.75, environment forces green for ambulance direction",
        },
    }


# ─── ③ Multi-agent convenience endpoints ──────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": tid,
                "name": cfg["name"],
                "max_steps": cfg["max_steps"],
                "target_throughput": cfg["target"],
                "num_intersections": 4 if tid == 4 else 1,
                "difficulty": {1: "easy", 2: "medium", 3: "hard", 4: "expert"}[tid],
                "features": (
                    ["smart_reward", "emergency_preemption", "multi_agent_grid", "green_wave"]
                    if tid == 4
                    else ["smart_reward", "emergency_preemption"]
                ),
            }
            for tid, cfg in TASK_CONFIGS.items()
        ]
    }


def main(host="0.0.0.0", port=7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(host=args.host, port=args.port)