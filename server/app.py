import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openenv.core.env_server.http_server import create_app
from models import TrafficControlAction, TrafficControlObservation
from traffic_control_env_environment import TrafficControlEnvironment
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = create_app(
    TrafficControlEnvironment,
    TrafficControlAction,
    TrafficControlObservation,
    env_name="traffic_control_env",
    max_concurrent_envs=4,
)

# Serve the simulation UI at root
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")

@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main(host="0.0.0.0", port=7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(host=args.host, port=args.port)