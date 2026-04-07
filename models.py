from typing import Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class TrafficControlAction(Action):
    # ── Core (backward-compat) ──────────────────────────────────────────────
    phase: str = Field(..., description="ns_green | ew_green | all_red")
    hold_steps: int = Field(default=3, ge=1, le=10)

    # ② Emergency preemption — force green for ambulance direction immediately
    emergency_preempt: bool = Field(
        default=False,
        description="Override phase to clear emergency vehicle direction right now",
    )

    # ③ Multi-agent (task 4 only) — phases for all 4 intersections
    intersection_phases: Optional[list[str]] = Field(
        default=None,
        description="[I0,I1,I2,I3] phases for 2x2 grid (task 4 only)",
    )
    intersection_hold_steps: Optional[list[int]] = Field(
        default=None,
        description="Hold steps per intersection (task 4 only)",
    )
    intersection_preempt: Optional[list[bool]] = Field(
        default=None,
        description="Emergency preempt flag per intersection (task 4 only)",
    )


class TrafficControlObservation(Observation):
    # ── Core fields ─────────────────────────────────────────────────────────
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

    # ① Smart reward hints visible to the agent
    queue_balance_score: float = Field(
        default=1.0,
        description="0-1; how evenly loaded all 4 lanes are (1=perfectly balanced)",
    )
    phase_held_too_long: bool = Field(
        default=False,
        description="True when phase held > 8 steps — starvation risk",
    )

    # ② Emergency preemption signals
    emergency_urgency: float = Field(
        default=0.0,
        description="0-1 urgency score; auto-preemption fires above 0.75",
    )
    preemption_active: bool = Field(
        default=False,
        description="Environment overrode agent phase for emergency",
    )

    # ③ Multi-agent network fields
    num_intersections: int = Field(default=1)
    network_throughput: int = Field(
        default=0,
        description="Total vehicles cleared across all intersections (task 4)",
    )
    green_wave_active: bool = Field(
        default=False,
        description="Adjacent intersections are phase-coordinated (task 4)",
    )
    # Per-intersection snapshots as flat dicts (avoids nested Observation issues)
    intersection_snapshots: Optional[list[dict]] = Field(
        default=None,
        description="List of per-intersection state dicts for task 4",
    )