import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openenv.core.env_server.http_server import create_app
from models import TrafficControlAction, TrafficControlObservation
from traffic_control_env_environment import TrafficControlEnvironment

app = create_app(
    TrafficControlEnvironment,
    TrafficControlAction,
    TrafficControlObservation,
    env_name="traffic_control_env",
    max_concurrent_envs=4,
)


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
