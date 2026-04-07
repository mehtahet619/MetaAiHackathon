"""
server/traffic_control_env_environment.py
All environment logic lives here — single-agent (tasks 1-3) and
multi-agent 2x2 grid (task 4).

Features:
  ① Smart reward  — weighted throughput, exponential wait, queue balance,
                    phase-switch cost, green-wave bonus
  ② Emergency preemption — sigmoid urgency score, auto-override at 0.75,
                           agent-triggered via emergency_preempt flag
  ③ Multi-agent  — 2x2 grid of 4 intersections, vehicle transfer between
                   adjacent nodes, green-wave coordination reward
"""

from __future__ import annotations

import math
import os
import random
import sys
from collections import deque
from typing import Optional
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.interfaces import Environment, EnvironmentMetadata
from openenv.core.env_server.types import State

from models import TrafficControlAction, TrafficControlObservation

# ─── Constants ─────────────────────────────────────────────────────────────────

DIRECTIONS = ["north", "south", "east", "west"]
VEHICLE_TYPES = ["car", "truck", "bus", "ambulance"]
VEHICLE_WEIGHTS = [0.65, 0.20, 0.10, 0.05]

# Slots each vehicle type consumes when passing through green
CLEAR_RATE = {"car": 2, "truck": 1, "bus": 1, "ambulance": 3}

# ① Weighted value for throughput bonus (bus/truck carry more people)
THROUGHPUT_WEIGHT = {"car": 1.0, "truck": 1.5, "bus": 2.0, "ambulance": 5.0}

MAX_QUEUE = 15
EMERGENCY_URGENCY_THRESHOLD = 0.75   # ② auto-preempt above this
PHASE_SWITCH_PENALTY = 0.25          # ① cost for switching too early
GREEN_WAVE_BONUS = 0.8               # ③ per-step bonus when neighbours coordinate

# ③ 2x2 grid: from intersection i, exiting Direction -> (neighbour_id, entry_direction)
# Grid:  [0 NW]--[1 NE]
#           |         |
#        [2 SW]--[3 SE]
GRID_NEIGHBOURS = {
    0: {"east": (1, "west"), "south": (2, "north")},
    1: {"west": (0, "east"), "south": (3, "north")},
    2: {"north": (0, "south"), "east": (3, "west")},
    3: {"north": (1, "south"), "west": (2, "east")},
}
TRANSFER_PROB = 0.40   # probability a cleared vehicle enters an adjacent intersection

# ─── Task configs ───────────────────────────────────────────────────────────────

TASK_CONFIGS = {
    1: {"name": "BasicThroughput",      "max_steps": 60,  "arrival_mult": 1.0,
        "emergency_spawn": False, "deadline": None, "target": 30},
    2: {"name": "EmergencyPriority",    "max_steps": 80,  "arrival_mult": 1.2,
        "emergency_spawn": True,  "deadline": 40,   "target": 25},
    3: {"name": "RushHour",             "max_steps": 120, "arrival_mult": 2.5,
        "emergency_spawn": True,  "deadline": 30,   "target": 70},
    4: {"name": "MultiAgentGrid",       "max_steps": 150, "arrival_mult": 1.8,
        "emergency_spawn": True,  "deadline": 50,   "target": 120},
}


# ─── Single intersection ────────────────────────────────────────────────────────

class Intersection:
    """One 4-way intersection. Used standalone (tasks 1-3) and inside the grid (task 4)."""

    def __init__(self, iid: int, cfg: dict, rng: random.Random, vc_start: int = 0):
        self.id = iid
        self.cfg = cfg
        self._rng = rng
        self._vc = vc_start
        self.step = 0
        self.phase = "ns_green"
        self.phase_dur = 0
        self._prev_phase = "ns_green"
        self.lanes: dict[str, deque] = {d: deque() for d in DIRECTIONS}
        self.vehicles: dict[str, dict] = {}
        self.cleared: list[dict] = []
        self.emerg_id: Optional[str] = None
        self.emerg_step: Optional[int] = None
        self.emerg_cleared = False
        # Seed backlog
        for _ in range(3):
            self._spawn("car")

    # ── Spawning ──────────────────────────────────────────────────────────

    def _spawn(self, force_type: Optional[str] = None, force_dir: Optional[str] = None) -> str:
        vtype = force_type
        if vtype is None:
            r = self._rng.random()
            cum = 0.0
            for t, w in zip(VEHICLE_TYPES, VEHICLE_WEIGHTS):
                cum += w
                if r < cum:
                    vtype = t
                    break
            vtype = vtype or "car"
        # Ambulance only once per episode, only if task allows
        if vtype == "ambulance" and (not self.cfg["emergency_spawn"] or self.emerg_id):
            vtype = "car"
        direction = force_dir or DIRECTIONS[int(self._rng.random() * 4)]
        self._vc += 1
        vid = f"i{self.id}_v{self._vc:04d}"
        self.vehicles[vid] = {"id": vid, "type": vtype, "dir": direction,
                              "arrival": self.step, "wait": 0, "cleared": False}
        self.lanes[direction].append(vid)
        if vtype == "ambulance":
            self.emerg_id = vid
            self.emerg_step = self.step
        return vid

    def inject(self, vtype: str, direction: str):
        """③ Transfer a vehicle in from an adjacent intersection."""
        self._spawn(force_type=vtype, force_dir=direction)

    # ② Urgency ────────────────────────────────────────────────────────────

    def urgency(self) -> float:
        """Sigmoid urgency score 0-1 for the waiting ambulance."""
        if not self.emerg_id or self.emerg_cleared:
            return 0.0
        v = self.vehicles.get(self.emerg_id)
        if not v or v["cleared"]:
            return 0.0
        waited = self.step - (self.emerg_step or 0)
        dl = self.cfg.get("deadline") or 60
        t = (waited - 0.6 * dl) / max(dl * 0.2, 1)
        return round(1.0 / (1.0 + math.exp(-t)), 3)

    def emerg_dir(self) -> Optional[str]:
        if not self.emerg_id or self.emerg_cleared:
            return None
        for d in DIRECTIONS:
            if self.emerg_id in self.lanes[d]:
                return d
        return None

    # ── Phase ─────────────────────────────────────────────────────────────

    def _green_dirs(self):
        if self.phase == "ns_green":
            return ["north", "south"]
        if self.phase == "ew_green":
            return ["east", "west"]
        return []

    @staticmethod
    def _phase_for_dir(d: str) -> str:
        return "ns_green" if d in ("north", "south") else "ew_green"

    # ① ② Apply action ─────────────────────────────────────────────────────

    def apply_action(self, phase: str, emergency_preempt: bool) -> float:
        """Apply phase (with preemption override). Returns phase-switch penalty."""
        target = phase
        urg = self.urgency()
        # ② Auto-preempt if urgency over threshold or flag set by agent
        if urg >= EMERGENCY_URGENCY_THRESHOLD or emergency_preempt:
            ed = self.emerg_dir()
            if ed:
                forced = self._phase_for_dir(ed)
                if forced != self.phase:
                    target = forced

        switch_cost = 0.0
        if target != self.phase:
            # ① Penalise switching too early
            if self.phase_dur < 2 and self._prev_phase != "all_red":
                switch_cost = PHASE_SWITCH_PENALTY
            self._prev_phase = self.phase
            self.phase = target
            self.phase_dur = 0
        else:
            self.phase_dur += 1
        return switch_cost

    # ── Step helpers ───────────────────────────────────────────────────────

    def update_waits(self):
        for d in DIRECTIONS:
            for vid in self.lanes[d]:
                self.vehicles[vid]["wait"] = self.step - self.vehicles[vid]["arrival"]

    def clear_vehicles(self) -> list[dict]:
        cleared_list = []
        for d in self._green_dirs():
            slots = 3
            while self.lanes[d] and slots > 0:
                vid = self.lanes[d][0]
                v = self.vehicles[vid]
                rate = CLEAR_RATE[v["type"]]
                if rate > slots:
                    break
                self.lanes[d].popleft()
                v["wait"] = self.step - v["arrival"]
                v["cleared"] = True
                self.cleared.append(v)
                slots -= rate
                cleared_list.append(v)
                if vid == self.emerg_id:
                    self.emerg_cleared = True
        return cleared_list

    def spawn_arrivals(self):
        mult = self.cfg["arrival_mult"]
        if (self.cfg["emergency_spawn"] and not self.emerg_id
                and self.step == self.cfg["max_steps"] // 4):
            self._spawn("ambulance")
        n = int(self._rng.random() * 2 * mult)
        for _ in range(n):
            self._spawn()

    # ① Smart reward ───────────────────────────────────────────────────────

    def compute_reward(self, cleared_list: list[dict], switch_cost: float,
                       gw_bonus: float = 0.0) -> dict:
        # ① Weighted throughput
        throughput_bonus = sum(THROUGHPUT_WEIGHT[v["type"]] for v in cleared_list)

        # ① Exponential wait penalty (grows fast after 6 steps)
        wait_penalty = 0.0
        for d in DIRECTIONS:
            for vid in self.lanes[d]:
                w = self.vehicles[vid]["wait"]
                if w > 6:
                    wait_penalty += 0.05 * math.exp(0.08 * (w - 6))

        # ① Queue balance bonus (reward even lane loading)
        ql = [len(self.lanes[d]) for d in DIRECTIONS]
        mean_q = sum(ql) / 4
        variance = sum((q - mean_q) ** 2 for q in ql) / 4
        balance_bonus = max(0.0, 1.0 - math.sqrt(variance) / 5)

        # ② Emergency bonus/penalty
        emergency_bonus = 0.0
        if any(v["type"] == "ambulance" for v in cleared_list):
            dl = self.cfg.get("deadline") or 999
            waited = self.step - (self.emerg_step or 0)
            ratio = waited / max(dl, 1)
            if ratio <= 0.5:
                emergency_bonus = 8.0
            elif ratio <= 1.0:
                emergency_bonus = 5.0 * (1.0 - ratio)
            else:
                emergency_bonus = -3.0

        # ② Urgency penalty per step while ambulance waits
        urgency_penalty = self.urgency() * 0.5

        # Starvation penalty
        starvation_penalty = sum(max(0, len(self.lanes[d]) - MAX_QUEUE) * 0.5
                                 for d in DIRECTIONS)

        total = (throughput_bonus - wait_penalty + emergency_bonus
                 + balance_bonus * 0.3 - urgency_penalty
                 - starvation_penalty - switch_cost + gw_bonus)

        return {
            "value": round(total, 4),
            "throughput_bonus": round(throughput_bonus, 4),
            "wait_penalty": round(wait_penalty, 4),
            "emergency_bonus": round(emergency_bonus, 4),
            "starvation_penalty": round(starvation_penalty, 4),
            "balance_bonus": round(balance_bonus, 4),
            "phase_switch_penalty": round(switch_cost, 4),
            "green_wave_bonus": round(gw_bonus, 4),
            "urgency_penalty": round(urgency_penalty, 4),
        }

    # ── Observation snapshot ───────────────────────────────────────────────

    def snapshot(self) -> dict:
        """③ Compact dict for intersection_snapshots field."""
        urg = self.urgency()
        ed = self.emerg_dir()
        ql = {d: len(self.lanes[d]) for d in DIRECTIONS}
        return {
            "id": self.id,
            "phase": self.phase,
            "phase_duration": self.phase_dur,
            "north_queue": ql["north"],
            "south_queue": ql["south"],
            "east_queue": ql["east"],
            "west_queue": ql["west"],
            "total_waiting": sum(ql.values()),
            "emergency_waiting": bool(ed),
            "emergency_direction": ed,
            "emergency_urgency": urg,
            "preemption_active": urg >= EMERGENCY_URGENCY_THRESHOLD,
            "throughput": len(self.cleared),
        }

    # ── Grading ───────────────────────────────────────────────────────────

    def grade(self) -> float:
        cfg = self.cfg
        cl = len(self.cleared)
        aw = sum(v["wait"] for v in self.cleared) / cl if cl else 0.0
        if cfg["target"] == 30:                  # task 1
            ts = min(cl / cfg["target"], 1.0)
            ws = max(0.0, 1.0 - max(0, aw - 5) / 20)
            return round(0.7 * ts + 0.3 * ws, 3)
        if cfg.get("deadline") == 40:            # task 2
            ts = min(cl / cfg["target"], 1.0)
            es = self._emergency_score(cfg)
            return round(0.4 * ts + 0.6 * es, 3)
        # task 3
        ts = min(cl / cfg["target"], 1.0)
        mq = max(len(self.lanes[d]) for d in DIRECTIONS)
        ss = max(0.0, 1.0 - mq / MAX_QUEUE)
        es = 1.0 if self.emerg_cleared else 0.0
        return round(0.5 * ts + 0.3 * es + 0.2 * ss, 3)

    def _emergency_score(self, cfg: dict) -> float:
        if not self.emerg_cleared:
            return 0.0
        for v in self.cleared:
            if v["id"] == self.emerg_id:
                dl = cfg.get("deadline") or 999
                if v["wait"] <= dl * 0.5:
                    return 1.0
                if v["wait"] <= dl:
                    return 0.6
                return 0.2
        return 0.0


# ─── OpenEnv Environment class ─────────────────────────────────────────────────

class TrafficControlEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._task_id = 1
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._ix: Optional[Intersection] = None          # single-agent
        self._grid: Optional[list[Intersection]] = None  # ③ multi-agent
        self._rng = random.Random(42)
        self._init(task_id=1, seed=42)

    # ── Init ──────────────────────────────────────────────────────────────

    def _init(self, task_id: int, seed: int = 42):
        self._task_id = task_id
        cfg = TASK_CONFIGS[task_id]
        self._rng = random.Random(seed)
        if task_id == 4:
            # ③ 2x2 grid — each intersection gets its own seeded RNG
            self._grid = [
                Intersection(i, cfg, random.Random(seed + i * 31), i * 200)
                for i in range(4)
            ]
            self._ix = None
        else:
            self._ix = Intersection(0, cfg, random.Random(seed))
            self._grid = None

    # ── OpenEnv interface ─────────────────────────────────────────────────

    def reset(self, seed=None, episode_id=None, **kwargs) -> TrafficControlObservation:
        task_id = int(kwargs.get("task_id", self._task_id))
        s = seed if seed is not None else 42
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._init(task_id=task_id, seed=s)
        return self._build_obs()

    def step(self, action: TrafficControlAction, **kwargs) -> TrafficControlObservation:
        self._state.step_count += 1

        if self._task_id == 4 and self._grid:
            return self._step_grid(action)
        return self._step_single(action)

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="TrafficControlEnv",
            description=(
                "Autonomous traffic signal control. Tasks 1-3: single 4-way intersection "
                "(smart rewards + emergency preemption). Task 4: 2x2 multi-agent grid "
                "with vehicle transfer and green-wave coordination."
            ),
            version="2.0.0",
        )

    # ── Single-agent step (tasks 1-3) ─────────────────────────────────────

    def _step_single(self, action: TrafficControlAction) -> TrafficControlObservation:
        ix = self._ix
        ix.step += 1
        ix.update_waits()

        switch_cost = ix.apply_action(action.phase, action.emergency_preempt)
        cleared_list = ix.clear_vehicles()
        ix.spawn_arrivals()

        reward = ix.compute_reward(cleared_list, switch_cost)
        cfg = TASK_CONFIGS[self._task_id]
        done = ix.step >= cfg["max_steps"]

        obs = self._build_obs()
        obs.done = done
        obs.reward = reward["value"]
        return obs

    # ③ Multi-agent step (task 4) ──────────────────────────────────────────

    def _step_grid(self, action: TrafficControlAction) -> TrafficControlObservation:
        grid = self._grid
        step = self._state.step_count

        # Resolve per-intersection actions
        phases = action.intersection_phases or [action.phase] * 4
        holds = action.intersection_hold_steps or [action.hold_steps] * 4
        preempts = action.intersection_preempt or [action.emergency_preempt] * 4

        # Update step counter for all intersections
        for ix in grid:
            ix.step = step
            ix.update_waits()

        # Apply actions
        switch_costs = [
            ix.apply_action(phases[i], preempts[i])
            for i, ix in enumerate(grid)
        ]

        # ③ Detect green wave BEFORE clearing
        gw = self._green_wave_active()
        gw_bonus = GREEN_WAVE_BONUS if gw else 0.0

        # Clear + transfer
        total_reward_val = 0.0
        for i, ix in enumerate(grid):
            cleared_list = ix.clear_vehicles()
            self._transfer(i, cleared_list)
            r = ix.compute_reward(cleared_list, switch_costs[i], gw_bonus / 4)
            total_reward_val += r["value"]
            ix.spawn_arrivals()

        cfg = TASK_CONFIGS[4]
        done = step >= cfg["max_steps"]

        obs = self._build_obs()
        obs.done = done
        obs.reward = round(total_reward_val, 4)
        return obs

    def _transfer(self, from_id: int, cleared: list[dict]):
        """③ Route a fraction of cleared vehicles into adjacent intersections."""
        for v in cleared:
            if v["type"] == "ambulance":
                continue
            if self._rng.random() > TRANSFER_PROB:
                continue
            nb = GRID_NEIGHBOURS.get(from_id, {}).get(v["dir"])
            if nb:
                neighbour_id, entry_dir = nb
                self._grid[neighbour_id].inject(v["type"], entry_dir)

    def _green_wave_active(self) -> bool:
        """③ True when horizontal OR vertical neighbour pairs share the same phase."""
        g = self._grid
        horiz = g[0].phase == g[1].phase and g[2].phase == g[3].phase
        vert  = g[0].phase == g[2].phase and g[1].phase == g[3].phase
        return horiz or vert

    # ── Observation builder ────────────────────────────────────────────────

    def _build_obs(self) -> TrafficControlObservation:
        cfg = TASK_CONFIGS[self._task_id]

        if self._task_id == 4 and self._grid:
            return self._build_obs_grid(cfg)
        return self._build_obs_single(cfg)

    def _build_obs_single(self, cfg: dict) -> TrafficControlObservation:
        ix = self._ix
        urg = ix.urgency()
        ed = ix.emerg_dir()
        ql = [len(ix.lanes[d]) for d in DIRECTIONS]
        mean_q = sum(ql) / 4
        variance = sum((q - mean_q) ** 2 for q in ql) / 4
        balance = round(max(0.0, 1.0 - math.sqrt(variance) / 5), 3)
        aw = sum(v["wait"] for v in ix.cleared) / len(ix.cleared) if ix.cleared else 0.0
        emerg_wait = ix.step - (ix.emerg_step or ix.step) if ix.emerg_id and not ix.emerg_cleared else 0

        return TrafficControlObservation(
            done=False, reward=0.0,
            current_phase=ix.phase,
            phase_duration=ix.phase_dur,
            north_queue=len(ix.lanes["north"]),
            south_queue=len(ix.lanes["south"]),
            east_queue=len(ix.lanes["east"]),
            west_queue=len(ix.lanes["west"]),
            total_waiting=sum(ql),
            emergency_waiting=bool(ed),
            emergency_direction=ed,
            emergency_wait_steps=emerg_wait,
            avg_wait_time=round(aw, 2),
            throughput=len(ix.cleared),
            step_budget=cfg["max_steps"] - ix.step,
            task_id=self._task_id,
            task_name=cfg["name"],
            step=ix.step,
            queue_balance_score=balance,
            phase_held_too_long=ix.phase_dur > 8,
            emergency_urgency=urg,
            preemption_active=urg >= EMERGENCY_URGENCY_THRESHOLD,
            num_intersections=1,
            network_throughput=len(ix.cleared),
            green_wave_active=False,
        )

    def _build_obs_grid(self, cfg: dict) -> TrafficControlObservation:
        grid = self._grid
        snaps = [ix.snapshot() for ix in grid]
        total_cleared = sum(len(ix.cleared) for ix in grid)
        total_waiting = sum(s["total_waiting"] for s in snaps)
        gw = self._green_wave_active()

        # Highest urgency intersection drives the top-level emergency fields
        most_urgent = max(snaps, key=lambda s: s["emergency_urgency"])

        # Primary intersection for phase/queue fields = whichever has most waiting
        primary_snap = max(snaps, key=lambda s: s["total_waiting"])
        primary_ix = grid[primary_snap["id"]]

        all_cleared = [v for ix in grid for v in ix.cleared]
        aw = sum(v["wait"] for v in all_cleared) / len(all_cleared) if all_cleared else 0.0

        ql = [primary_snap["north_queue"], primary_snap["south_queue"],
              primary_snap["east_queue"], primary_snap["west_queue"]]
        mean_q = sum(ql) / 4
        variance = sum((q - mean_q) ** 2 for q in ql) / 4
        balance = round(max(0.0, 1.0 - math.sqrt(variance) / 5), 3)

        return TrafficControlObservation(
            done=False, reward=0.0,
            current_phase=primary_ix.phase,
            phase_duration=primary_ix.phase_dur,
            north_queue=primary_snap["north_queue"],
            south_queue=primary_snap["south_queue"],
            east_queue=primary_snap["east_queue"],
            west_queue=primary_snap["west_queue"],
            total_waiting=total_waiting,
            emergency_waiting=most_urgent["emergency_waiting"],
            emergency_direction=most_urgent["emergency_direction"],
            emergency_wait_steps=most_urgent.get("emergency_wait_steps", 0),
            avg_wait_time=round(aw, 2),
            throughput=sum(s["throughput"] for s in snaps),
            step_budget=cfg["max_steps"] - self._state.step_count,
            task_id=4,
            task_name=cfg["name"],
            step=self._state.step_count,
            queue_balance_score=balance,
            phase_held_too_long=primary_ix.phase_dur > 8,
            emergency_urgency=most_urgent["emergency_urgency"],
            preemption_active=most_urgent["preemption_active"],
            num_intersections=4,
            network_throughput=total_cleared,
            green_wave_active=gw,
            intersection_snapshots=snaps,
        )

    # ── Grade (called by OpenEnv grader endpoint) ─────────────────────────

    def grade(self) -> float:
        if self._task_id == 4 and self._grid:
            return self._grade_grid()
        return self._ix.grade()

    def _grade_grid(self) -> float:
        cfg = TASK_CONFIGS[4]
        total = sum(len(ix.cleared) for ix in self._grid)
        ts = min(total / cfg["target"], 1.0)
        es = 1.0 if any(ix.emerg_cleared for ix in self._grid) else 0.0
        mq = max(max(len(ix.lanes[d]) for d in DIRECTIONS) for ix in self._grid)
        ss = max(0.0, 1.0 - mq / MAX_QUEUE)
        gw_score = sum(
            1 for ix in self._grid
            if ix.phase == self._grid[0].phase
        ) / 4
        return round(0.4 * ts + 0.25 * es + 0.2 * ss + 0.15 * gw_score, 3)