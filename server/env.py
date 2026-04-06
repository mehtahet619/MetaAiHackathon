"""
TrafficControlEnv — OpenEnv-compliant multi-agent traffic signal environment.

Real-world task: Coordinate traffic signals at a 4-way intersection to minimize
vehicle wait times and prioritize emergency vehicles. This models a genuine urban
challenge: inefficient signal timing costs billions in wasted fuel/time annually.

Three tasks (easy → hard):
  Task 1: Basic throughput — move N vehicles without starvation
  Task 2: Emergency priority — clear a path for an ambulance in time
  Task 3: Rush hour orchestration — balance four competing approaches under surge
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

# ─── Data Models ────────────────────────────────────────────────────────────────

class Direction(str, Enum):
    NORTH = "north"
    SOUTH = "south"
    EAST  = "east"
    WEST  = "west"

class SignalPhase(str, Enum):
    NS_GREEN = "ns_green"   # North-South green, East-West red
    EW_GREEN = "ew_green"   # East-West green, North-South red
    ALL_RED  = "all_red"    # Transition / emergency hold

class VehicleType(str, Enum):
    CAR       = "car"
    TRUCK     = "truck"
    BUS       = "bus"
    AMBULANCE = "ambulance"

class Vehicle(BaseModel):
    id: str
    type: VehicleType
    direction: Direction
    arrival_time: float
    wait_time: float = 0.0
    cleared: bool = False

class LaneState(BaseModel):
    direction: Direction
    queue_length: int
    vehicles: list[str]       # vehicle IDs
    has_emergency: bool = False

class TrafficObservation(BaseModel):
    """Everything an agent needs to make a signal decision."""
    step: int
    current_phase: SignalPhase
    phase_duration: int                  # steps current phase has been active
    lanes: dict[str, LaneState]          # keyed by direction
    total_waiting: int                   # total vehicles waiting
    emergency_waiting: bool              # is an ambulance in any queue?
    emergency_direction: Optional[str]   # which direction, if any
    emergency_wait_steps: int            # how long ambulance has waited
    avg_wait_time: float                 # mean wait across cleared vehicles
    throughput: int                      # vehicles cleared this episode
    step_budget: int                     # steps remaining
    task_id: int

class TrafficAction(BaseModel):
    """Signal control decision."""
    phase: SignalPhase = Field(
        description="Set the next signal phase"
    )
    hold_steps: int = Field(
        default=1,
        ge=1,
        le=10,
        description="How many steps to hold this phase (1-10)"
    )

class TrafficReward(BaseModel):
    value: float
    throughput_bonus: float
    wait_penalty: float
    emergency_bonus: float
    starvation_penalty: float

class TrafficState(BaseModel):
    step: int
    phase: SignalPhase
    vehicles: list[Vehicle]
    cleared_count: int
    emergency_cleared: bool
    task_id: int

class StepResult(BaseModel):
    observation: TrafficObservation
    reward: TrafficReward
    done: bool
    info: dict[str, Any]

# ─── Environment Core ────────────────────────────────────────────────────────────

ARRIVAL_RATES = {
    VehicleType.CAR:       0.65,
    VehicleType.TRUCK:     0.20,
    VehicleType.BUS:       0.10,
    VehicleType.AMBULANCE: 0.05,
}

CLEAR_RATE = {
    # vehicles that can clear per step when phase is green for that direction
    VehicleType.CAR:       2,
    VehicleType.TRUCK:     1,
    VehicleType.BUS:       1,
    VehicleType.AMBULANCE: 3,  # ambulance pushes through fast
}

MAX_QUEUE = 15  # max vehicles per lane before starvation penalty

class TrafficControlEnv:
    """
    4-way intersection traffic control environment.

    State: Queues on each approach (N, S, E, W), current signal phase
    Action: Set phase + hold duration
    Reward: Throughput - wait_penalty + emergency_bonus - starvation_penalty
    """

    TASK_CONFIGS = {
        1: {
            "name": "Basic Throughput",
            "max_steps": 60,
            "arrival_multiplier": 1.0,
            "emergency_spawn": False,
            "target_throughput": 30,
            "description": "Clear at least 30 vehicles with avg wait ≤ 5 steps",
        },
        2: {
            "name": "Emergency Priority",
            "max_steps": 80,
            "arrival_multiplier": 1.2,
            "emergency_spawn": True,
            "emergency_deadline": 40,  # ambulance must clear within 40 steps of arrival
            "target_throughput": 25,
            "description": "Clear an ambulance within deadline while maintaining flow",
        },
        3: {
            "name": "Rush Hour Orchestration",
            "max_steps": 120,
            "arrival_multiplier": 2.5,
            "emergency_spawn": True,
            "emergency_deadline": 30,
            "target_throughput": 70,
            "description": "High surge: clear 70 vehicles, handle emergencies, prevent starvation",
        },
    }

    def __init__(self, task_id: int = 1, seed: int = 42):
        assert task_id in self.TASK_CONFIGS, f"task_id must be 1, 2, or 3"
        self.task_id = task_id
        self.cfg = self.TASK_CONFIGS[task_id]
        self.seed = seed
        self._rng = random.Random(seed)
        self._vehicle_counter = 0
        self._reset_state()

    def _reset_state(self):
        self.step_num = 0
        self.current_phase = SignalPhase.NS_GREEN
        self.phase_duration = 0
        self.hold_remaining = 0
        self.vehicles: dict[str, Vehicle] = {}
        self.lanes: dict[Direction, deque] = {d: deque() for d in Direction}
        self.cleared_vehicles: list[Vehicle] = []
        self.emergency_arrived_step: Optional[int] = None
        self.emergency_cleared = False
        self.emergency_vehicle_id: Optional[str] = None
        self._rng = random.Random(self.seed)
        self._vehicle_counter = 0

        # Seed initial queue — small backlog at start
        for _ in range(4):
            self._spawn_vehicle(force_type=VehicleType.CAR)

    def _spawn_vehicle(self, force_type: Optional[VehicleType] = None) -> Optional[str]:
        vtype = force_type
        if vtype is None:
            r = self._rng.random()
            cumulative = 0.0
            for t, rate in ARRIVAL_RATES.items():
                cumulative += rate
                if r < cumulative:
                    vtype = t
                    break
            vtype = vtype or VehicleType.CAR

        # Emergency vehicles only spawn in tasks 2/3, once per episode
        if vtype == VehicleType.AMBULANCE:
            if not self.cfg.get("emergency_spawn") or self.emergency_vehicle_id:
                vtype = VehicleType.CAR

        direction = self._rng.choice(list(Direction))
        self._vehicle_counter += 1
        vid = f"v{self._vehicle_counter:04d}"

        v = Vehicle(
            id=vid,
            type=vtype,
            direction=direction,
            arrival_time=float(self.step_num),
        )
        self.vehicles[vid] = v
        self.lanes[direction].append(vid)

        if vtype == VehicleType.AMBULANCE:
            self.emergency_vehicle_id = vid
            self.emergency_arrived_step = self.step_num

        return vid

    def _get_green_directions(self) -> list[Direction]:
        if self.current_phase == SignalPhase.NS_GREEN:
            return [Direction.NORTH, Direction.SOUTH]
        elif self.current_phase == SignalPhase.EW_GREEN:
            return [Direction.EAST, Direction.WEST]
        return []

    def _clear_vehicles(self) -> tuple[int, bool]:
        """Move vehicles through green lanes. Returns (cleared_count, emergency_cleared)."""
        green_dirs = self._get_green_directions()
        cleared = 0
        emergency_just_cleared = False

        for direction in green_dirs:
            lane = self.lanes[direction]
            slots = 3  # max vehicles to clear per step per direction
            while lane and slots > 0:
                vid = lane[0]
                v = self.vehicles[vid]
                rate = CLEAR_RATE[v.type]
                # Each vehicle type consumes different slot budget
                if rate > slots:
                    break
                lane.popleft()
                v.wait_time = self.step_num - v.arrival_time
                v.cleared = True
                self.cleared_vehicles.append(v)
                slots -= rate
                cleared += 1
                if vid == self.emergency_vehicle_id:
                    self.emergency_cleared = True
                    emergency_just_cleared = True

        return cleared, emergency_just_cleared

    def _update_wait_times(self):
        for direction in Direction:
            for vid in self.lanes[direction]:
                self.vehicles[vid].wait_time = self.step_num - self.vehicles[vid].arrival_time

    def _spawn_arrivals(self):
        rate = self.cfg["arrival_multiplier"]
        # Spawn ambulance in first half of episode (task 2/3 only)
        if (self.cfg.get("emergency_spawn")
                and not self.emergency_vehicle_id
                and self.step_num == self.cfg["max_steps"] // 4):
            self._spawn_vehicle(force_type=VehicleType.AMBULANCE)

        # Regular vehicle arrivals (Poisson-ish)
        n_arrivals = int(self._rng.random() * 2 * rate)
        for _ in range(n_arrivals):
            self._spawn_vehicle()

    def _build_observation(self) -> TrafficObservation:
        lanes_state = {}
        for d in Direction:
            vids = list(self.lanes[d])
            has_emerg = self.emergency_vehicle_id in vids if self.emergency_vehicle_id else False
            lanes_state[d.value] = LaneState(
                direction=d,
                queue_length=len(vids),
                vehicles=vids[:5],   # show top 5 for context
                has_emergency=has_emerg,
            )

        emerg_wait = 0
        emerg_dir = None
        if self.emergency_vehicle_id and not self.emergency_cleared:
            ev = self.vehicles.get(self.emergency_vehicle_id)
            if ev and not ev.cleared:
                emerg_wait = self.step_num - (self.emergency_arrived_step or 0)
                # Find its direction
                for d in Direction:
                    if self.emergency_vehicle_id in self.lanes[d]:
                        emerg_dir = d.value
                        break

        avg_wait = 0.0
        if self.cleared_vehicles:
            avg_wait = sum(v.wait_time for v in self.cleared_vehicles) / len(self.cleared_vehicles)

        total_waiting = sum(len(self.lanes[d]) for d in Direction)

        return TrafficObservation(
            step=self.step_num,
            current_phase=self.current_phase,
            phase_duration=self.phase_duration,
            lanes=lanes_state,
            total_waiting=total_waiting,
            emergency_waiting=bool(emerg_dir),
            emergency_direction=emerg_dir,
            emergency_wait_steps=emerg_wait,
            avg_wait_time=round(avg_wait, 2),
            throughput=len(self.cleared_vehicles),
            step_budget=self.cfg["max_steps"] - self.step_num,
            task_id=self.task_id,
        )

    def _compute_reward(
        self,
        cleared: int,
        emergency_just_cleared: bool,
    ) -> TrafficReward:
        cfg = self.cfg

        # Throughput bonus
        throughput_bonus = cleared * 0.5

        # Wait time penalty (exponential for long waits)
        wait_penalty = 0.0
        for d in Direction:
            for vid in self.lanes[d]:
                w = self.vehicles[vid].wait_time
                if w > 8:
                    wait_penalty += 0.1 * (w - 8)

        # Emergency bonus / penalty
        emergency_bonus = 0.0
        if emergency_just_cleared:
            deadline = cfg.get("emergency_deadline", 999)
            waited = self.step_num - (self.emergency_arrived_step or 0)
            if waited <= deadline // 2:
                emergency_bonus = 5.0
            elif waited <= deadline:
                emergency_bonus = 3.0
            else:
                emergency_bonus = -2.0  # cleared late

        # Emergency urgency penalty per step while waiting
        if self.emergency_vehicle_id and not self.emergency_cleared:
            ev = self.vehicles.get(self.emergency_vehicle_id)
            if ev and not ev.cleared:
                waited = self.step_num - (self.emergency_arrived_step or 0)
                deadline = cfg.get("emergency_deadline", 999)
                if waited > deadline:
                    emergency_bonus -= 0.3  # ticking penalty post-deadline

        # Starvation penalty — queue overflow
        starvation_penalty = 0.0
        for d in Direction:
            overflow = max(0, len(self.lanes[d]) - MAX_QUEUE)
            starvation_penalty += overflow * 0.4

        value = throughput_bonus - wait_penalty + emergency_bonus - starvation_penalty
        return TrafficReward(
            value=value,
            throughput_bonus=throughput_bonus,
            wait_penalty=wait_penalty,
            emergency_bonus=emergency_bonus,
            starvation_penalty=starvation_penalty,
        )

    # ─── Public API ─────────────────────────────────────────────────────────────

    def reset(self) -> TrafficObservation:
        self._reset_state()
        return self._build_observation()

    def step(self, action: TrafficAction) -> StepResult:
        self.step_num += 1
        self._update_wait_times()

        # Apply phase change (with 1-step ALL_RED transition if phase changed)
        if action.phase != self.current_phase:
            self.current_phase = action.phase
            self.phase_duration = 0
        else:
            self.phase_duration += 1

        # Clear vehicles from green lanes
        cleared, emerg_cleared = self._clear_vehicles()

        # Spawn new arrivals
        self._spawn_arrivals()

        # Compute reward
        reward = self._compute_reward(cleared, emerg_cleared)

        # Check termination
        done = self.step_num >= self.cfg["max_steps"]

        obs = self._build_observation()

        info = {
            "task_name": self.cfg["name"],
            "cleared_this_step": cleared,
            "total_cleared": len(self.cleared_vehicles),
            "emergency_cleared": self.emergency_cleared,
        }

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> TrafficState:
        return TrafficState(
            step=self.step_num,
            phase=self.current_phase,
            vehicles=[v for v in self.vehicles.values()],
            cleared_count=len(self.cleared_vehicles),
            emergency_cleared=self.emergency_cleared,
            task_id=self.task_id,
        )

    def grade(self) -> float:
        """
        Compute task score in [0.0, 1.0].
        Each task has a specific grading rubric.
        """
        cfg = self.cfg
        cleared = len(self.cleared_vehicles)
        avg_wait = 0.0
        if self.cleared_vehicles:
            avg_wait = sum(v.wait_time for v in self.cleared_vehicles) / len(self.cleared_vehicles)

        if self.task_id == 1:
            # Throughput ≥ 30 (max 60), avg wait ≤ 5 (penalty if higher)
            throughput_score = min(cleared / cfg["target_throughput"], 1.0)
            wait_score = max(0.0, 1.0 - (max(0, avg_wait - 5) / 20))
            return round(0.7 * throughput_score + 0.3 * wait_score, 3)

        elif self.task_id == 2:
            # Throughput 25+, emergency cleared within deadline
            throughput_score = min(cleared / cfg["target_throughput"], 1.0)
            emerg_score = 0.0
            if self.emergency_cleared:
                waited = 0
                for v in self.cleared_vehicles:
                    if v.id == self.emergency_vehicle_id:
                        waited = v.wait_time
                        break
                deadline = cfg.get("emergency_deadline", 999)
                if waited <= deadline * 0.5:
                    emerg_score = 1.0
                elif waited <= deadline:
                    emerg_score = 0.6
                else:
                    emerg_score = 0.2
            return round(0.4 * throughput_score + 0.6 * emerg_score, 3)

        elif self.task_id == 3:
            # Throughput 70+, emergencies handled, no starvation
            throughput_score = min(cleared / cfg["target_throughput"], 1.0)
            max_queue = max(len(list(self.lanes[d])) for d in Direction) if any(self.lanes[d] for d in Direction) else 0
            starvation_score = max(0.0, 1.0 - max_queue / MAX_QUEUE)
            emerg_score = 1.0 if self.emergency_cleared else 0.0
            return round(0.5 * throughput_score + 0.3 * emerg_score + 0.2 * starvation_score, 3)

        return 0.0
