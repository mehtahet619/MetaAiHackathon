import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openenv.core.env_server.http_server import create_app
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from models import TrafficControlAction, TrafficControlObservation
from traffic_control_env_environment import TrafficControlEnvironment, TASK_CONFIGS

app = create_app(
    TrafficControlEnvironment,
    TrafficControlAction,
    TrafficControlObservation,
    env_name="traffic_control_env",
    max_concurrent_envs=4,
)

# ── Static UI ─────────────────────────────────────────────────────────────────

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

# ── Reward component info ─────────────────────────────────────────────────────

@app.get("/reward/components")
def reward_components():
    return {
        "components": {
            "throughput_bonus":     "Weighted throughput (car=1x, truck=1.5x, bus=2x, ambulance=5x)",
            "wait_penalty":         "Exponential penalty for vehicles waiting > 6 steps",
            "balance_bonus":        "Reward for keeping all 4 lanes evenly loaded",
            "emergency_bonus":      "+8 if cleared in first 50% of deadline, -3 if late",
            "urgency_penalty":      "Sigmoid 0-0.5/step while ambulance waits",
            "starvation_penalty":   "-0.5 per vehicle over queue cap of 15",
            "phase_switch_penalty": "-0.25 if phase switched after < 2 steps held",
            "green_wave_bonus":     "+0.8/step when adjacent intersections share phase (task 4)",
        },
        "auto_preemption": {
            "threshold": 0.75,
            "description": "When emergency_urgency >= 0.75 the env forces green for ambulance direction",
        },
    }

# ── Tasks list ────────────────────────────────────────────────────────────────

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
            }
            for tid, cfg in TASK_CONFIGS.items()
        ]
    }

# ── Entry point ───────────────────────────────────────────────────────────────
# main() has no required arguments so it is safely callable by the entry-point
# loader (pyproject.toml scripts) as well as directly from the command line.

def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
