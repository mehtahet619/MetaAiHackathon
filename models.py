from typing import Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class TrafficControlAction(Action):
    phase: str = Field(..., description="ns_green | ew_green | all_red")
    hold_steps: int = Field(default=3, ge=1, le=10)


class TrafficControlObservation(Observation):
    current_phase: str = Field(default="ns_green")
    phase_duration: int = Field(default=0)
    north_queue: int = Field(default=0)
    south_queue: int = Field(default=0)
    east_queue: int = Field(default=0)
    west_queue: int = Field(default=0)
    total_waiting: int = Field(default=0)
    emergency_waiting: bool = Field(default=False)
    emergency_direction: Optional[str] = Field(default=None)
    emergency_wait_steps: int = Field(default=0)
    avg_wait_time: float = Field(default=0.0)
    throughput: int = Field(default=0)
    step_budget: int = Field(default=60)
    task_id: int = Field(default=1)
    task_name: str = Field(default="BasicThroughput")
    step: int = Field(default=0)
