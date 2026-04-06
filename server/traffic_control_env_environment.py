from __future__ import annotations
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import deque
from typing import Optional
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import TrafficControlAction, TrafficControlObservation

DIRECTIONS = ["north","south","east","west"]
VEHICLE_TYPES = ["car","truck","bus","ambulance"]
VEHICLE_WEIGHTS = [0.65, 0.20, 0.10, 0.05]
CLEAR_RATE = {"car":2,"truck":1,"bus":1,"ambulance":3}
MAX_QUEUE = 15
TASK_CONFIGS = {
    1: {"name":"BasicThroughput",   "max_steps":60,  "arrival_mult":1.0, "emergency_spawn":False, "target":30, "deadline":None},
    2: {"name":"EmergencyPriority", "max_steps":80,  "arrival_mult":1.2, "emergency_spawn":True,  "target":25, "deadline":40},
    3: {"name":"RushHour",          "max_steps":120, "arrival_mult":2.5, "emergency_spawn":True,  "target":70, "deadline":30},
}

class TrafficControlEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._task_id = 1
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_env(seed=42)

    def _reset_env(self, seed=42, task_id=None):
        if task_id is not None:
            self._task_id = task_id
        self._cfg = TASK_CONFIGS[self._task_id]
        self._rng = random.Random(seed)
        self._step = 0
        self._phase = "ns_green"
        self._phase_dur = 0
        self._lanes = {d: deque() for d in DIRECTIONS}
        self._vehicles = {}
        self._cleared = []
        self._vctr = 0
        self._emerg_id = None
        self._emerg_step = None
        self._emerg_cleared = False
        for _ in range(4):
            self._spawn("car")

    def _spawn(self, force_type=None):
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
        if vtype == "ambulance" and (not self._cfg["emergency_spawn"] or self._emerg_id):
            vtype = "car"
        direction = DIRECTIONS[int(self._rng.random() * 4)]
        self._vctr += 1
        vid = f"v{self._vctr:04d}"
        self._vehicles[vid] = {"id":vid,"type":vtype,"dir":direction,"arrival":self._step,"wait":0,"cleared":False}
        self._lanes[direction].append(vid)
        if vtype == "ambulance":
            self._emerg_id = vid
            self._emerg_step = self._step
        return vid

    def _green_dirs(self):
        if self._phase == "ns_green": return ["north","south"]
        if self._phase == "ew_green": return ["east","west"]
        return []

    def _clear_vehicles(self):
        cleared = 0
        for d in self._green_dirs():
            slots = 3
            while self._lanes[d] and slots > 0:
                vid = self._lanes[d][0]
                v = self._vehicles[vid]
                rate = CLEAR_RATE[v["type"]]
                if rate > slots:
                    break
                self._lanes[d].popleft()
                v["wait"] = self._step - v["arrival"]
                v["cleared"] = True
                self._cleared.append(v)
                slots -= rate
                cleared += 1
                if vid == self._emerg_id:
                    self._emerg_cleared = True
        return cleared

    def _spawn_arrivals(self):
        mult = self._cfg["arrival_mult"]
        if self._cfg["emergency_spawn"] and not self._emerg_id and self._step == self._cfg["max_steps"] // 4:
            self._spawn("ambulance")
        n = int(self._rng.random() * 2 * mult)
        for _ in range(n):
            self._spawn()

    def _update_waits(self):
        for d in DIRECTIONS:
            for vid in self._lanes[d]:
                self._vehicles[vid]["wait"] = self._step - self._vehicles[vid]["arrival"]

    def _compute_reward(self, cleared):
        r = cleared * 0.5
        for d in DIRECTIONS:
            for vid in self._lanes[d]:
                w = self._vehicles[vid]["wait"]
                if w > 8:
                    r -= 0.1 * (w - 8)
        if self._emerg_id and self._emerg_cleared:
            ev = next((v for v in self._cleared if v["id"] == self._emerg_id), None)
            if ev:
                dl = self._cfg.get("deadline") or 999
                r += 5.0 if ev["wait"] <= dl*0.5 else (3.0 if ev["wait"] <= dl else -2.0)
        elif self._emerg_id and not self._emerg_cleared and self._emerg_step is not None:
            dl = self._cfg.get("deadline") or 999
            if self._step - self._emerg_step > dl:
                r -= 0.3
        for d in DIRECTIONS:
            r -= max(0, len(self._lanes[d]) - MAX_QUEUE) * 0.4
        return r

    def _grade(self):
        cl = len(self._cleared)
        aw = (sum(v["wait"] for v in self._cleared) / cl) if cl else 0.0
        cfg = self._cfg
        if self._task_id == 1:
            return round(0.7*min(cl/cfg["target"],1.0) + 0.3*max(0.0,1.0-max(0,aw-5)/20), 3)
        elif self._task_id == 2:
            ts = min(cl/cfg["target"],1.0)
            es = 0.0
            if self._emerg_cleared and self._emerg_id:
                ev = next((v for v in self._cleared if v["id"] == self._emerg_id), None)
                if ev:
                    dl = cfg["deadline"] or 999
                    es = 1.0 if ev["wait"] <= dl*0.5 else (0.6 if ev["wait"] <= dl else 0.2)
            return round(0.4*ts + 0.6*es, 3)
        else:
            ts = min(cl/cfg["target"],1.0)
            mq = max(len(self._lanes[d]) for d in DIRECTIONS)
            ss = max(0.0, 1.0 - mq/MAX_QUEUE)
            es = 1.0 if self._emerg_cleared else 0.0
            return round(0.5*ts + 0.3*es + 0.2*ss, 3)

    def _build_obs(self, reward=0.0, done=False):
        aw = (sum(v["wait"] for v in self._cleared)/len(self._cleared)) if self._cleared else 0.0
        emerg_dir = None
        emerg_wait = 0
        if self._emerg_id and not self._emerg_cleared:
            ev = self._vehicles.get(self._emerg_id)
            if ev and not ev["cleared"]:
                emerg_wait = ev["wait"]
                for d in DIRECTIONS:
                    if self._emerg_id in self._lanes[d]:
                        emerg_dir = d
                        break
        return TrafficControlObservation(
            done=done, reward=reward,
            current_phase=self._phase, phase_duration=self._phase_dur,
            north_queue=len(self._lanes["north"]), south_queue=len(self._lanes["south"]),
            east_queue=len(self._lanes["east"]),   west_queue=len(self._lanes["west"]),
            total_waiting=sum(len(self._lanes[d]) for d in DIRECTIONS),
            emergency_waiting=bool(emerg_dir), emergency_direction=emerg_dir,
            emergency_wait_steps=emerg_wait, avg_wait_time=round(aw,2),
            throughput=len(self._cleared), step_budget=self._cfg["max_steps"]-self._step,
            task_id=self._task_id, task_name=self._cfg["name"], step=self._step,
        )

    def reset(self, seed=None, episode_id=None, **kwargs):
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._reset_env(seed=seed if seed is not None else 42, task_id=kwargs.get("task_id", self._task_id))
        return self._build_obs()

    def step(self, action: TrafficControlAction, **kwargs):
        self._step += 1
        self._state.step_count = self._step
        self._update_waits()
        phase = action.phase if action.phase in {"ns_green","ew_green","all_red"} else "ns_green"
        if phase != self._phase:
            self._phase = phase
            self._phase_dur = 0
        else:
            self._phase_dur += 1
        cleared = self._clear_vehicles()
        self._spawn_arrivals()
        reward = self._compute_reward(cleared)
        done = self._step >= self._cfg["max_steps"]
        if done:
            reward += self._grade() * 10.0
        return self._build_obs(reward=reward, done=done)

    @property
    def state(self):
        return self._state

    def get_metadata(self):
        from openenv.core.env_server.interfaces import EnvironmentMetadata
        return EnvironmentMetadata(
            name="TrafficControlEnv",
            description="Autonomous traffic signal control at a 4-way intersection. 3 tasks: easy, medium, hard.",
            version="1.0.0",
        )
