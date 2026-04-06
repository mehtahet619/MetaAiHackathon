# 🚦 TrafficControlEnv

**OpenEnv — Autonomous Intersection Traffic Signal Control**

> *Real-world challenge: Coordinate traffic signals at a 4-way intersection to minimize vehicle wait times, prevent lane starvation, and prioritize emergency vehicles in real-time.*

---

## Why This Environment Matters

Poorly timed traffic signals cost the US alone **$87 billion/year** in wasted fuel and time. Smart city systems increasingly rely on RL agents for adaptive signal control — but there's no standard benchmark environment for training and evaluating them. TrafficControlEnv fills that gap with a clean, extensible, reproducible benchmark grounded in real operational logic.

Unlike toy grid-worlds, this environment models:
- **Multi-directional vehicle flow** (North, South, East, West)
- **Heterogeneous vehicle types** (cars, trucks, buses, ambulances)
- **Emergency preemption** — ambulances must receive signal priority
- **Starvation prevention** — no lane should be indefinitely starved
- **Partial progress rewards** — dense reward signal across the full trajectory

---

## Environment Architecture

```
                    [North queue]
                         ↓
[West queue] → ← 4-WAY INTERSECTION → [East queue]
                         ↑
                    [South queue]

Signal controller (agent) sets:
  phase: ns_green | ew_green | all_red
  hold_steps: 1–10 (how long to maintain phase)
```

### Observation Space

| Field | Type | Description |
|---|---|---|
| `step` | int | Current episode step |
| `current_phase` | enum | Active signal phase |
| `phase_duration` | int | Steps current phase has been held |
| `lanes` | object | Per-direction queue state (length, vehicles, emergency flag) |
| `total_waiting` | int | Total vehicles in all queues |
| `emergency_waiting` | bool | Is an ambulance currently waiting? |
| `emergency_direction` | str? | Which lane the ambulance is in |
| `emergency_wait_steps` | int | How long the ambulance has waited |
| `avg_wait_time` | float | Mean wait across all cleared vehicles |
| `throughput` | int | Vehicles cleared this episode |
| `step_budget` | int | Steps remaining |

### Action Space

```json
{
  "phase": "ns_green | ew_green | all_red",
  "hold_steps": 1
}
```

`hold_steps` (1–10): how many steps to maintain the chosen phase before the agent is queried again.

### Reward Function

```
reward = throughput_bonus - wait_penalty + emergency_bonus - starvation_penalty

throughput_bonus  = +0.5 per vehicle cleared this step
wait_penalty      = +0.1 × (wait - 8) per vehicle with wait > 8 steps
emergency_bonus   = +5.0 (cleared early) | +3.0 (on time) | -2.0 (late)
starvation_penalty= +0.4 per vehicle exceeding queue cap (15)
```

The reward provides **dense signal across the full trajectory** — not just end-of-episode.

---

## Tasks

### Task 1 — Basic Throughput (Easy)
**Goal**: Clear ≥ 30 vehicles with average wait ≤ 5 steps  
**Steps**: 60  
**Grading**: `0.7 × throughput_score + 0.3 × wait_score`  
**Baseline score**: ~0.72 (greedy phase switching)

An agent that simply alternates NS/EW phases every 3 steps can achieve ~0.65. Getting above 0.80 requires learning queue-length-aware switching.

### Task 2 — Emergency Priority (Medium)
**Goal**: Clear ≥ 25 vehicles **and** clear the ambulance within 40 steps of its arrival  
**Steps**: 80  
**Grading**: `0.4 × throughput_score + 0.6 × emergency_score`  
**Baseline score**: ~0.55 (greedy agent often misses deadline)

The ambulance spawns at step 20. Agents that don't learn to detect and prioritize the emergency field will fail. Deadline pressure creates a real multi-objective tradeoff.

### Task 3 — Rush Hour Orchestration (Hard)
**Goal**: Clear ≥ 70 vehicles, handle emergency, prevent starvation during 2.5× surge  
**Steps**: 120  
**Grading**: `0.5 × throughput_score + 0.3 × emergency_score + 0.2 × starvation_score`  
**Baseline score**: ~0.38 (frontier models struggle)

Arrival rate surges to 2.5× normal. Queues grow faster than they can be cleared without intelligent scheduling. Agents must dynamically rebalance across phases while maintaining emergency preemption. This task genuinely challenges GPT-4 level models.

---

## Grader Criteria

All graders produce scores in `[0.0, 1.0]` and are **deterministic** given the same seed:

| Metric | Measurement |
|---|---|
| Throughput | Vehicles cleared / target vehicles |
| Wait time | `max(0, 1 - (avg_wait - 5) / 20)` |
| Emergency | Time-to-clear vs. deadline ratio |
| Starvation | `max(0, 1 - max_queue / 15)` |

---

## Setup & Usage

### Local Development

```bash
cd server
pip install -r requirements.txt
uvicorn main:app --reload --port 7860
```

### Docker

```bash
docker build -t traffic-control-env .
docker run -p 7860:7860 traffic-control-env
```

### API Usage

```bash
# Reset to task 1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 1, "seed": 42}'

# Take an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"phase": "ns_green", "hold_steps": 3}'

# Grade the current episode
curl -X POST http://localhost:7860/grade
```

### Run Inference

```bash
export OPENAI_API_KEY=sk-...
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_SPACE_URL=https://your-space.hf.space

python inference.py
```

---

## Baseline Scores (GPT-4o-mini)

| Task | Score | Notes |
|---|---|---|
| Task 1 | 0.72 | Greedy phase switching, reasonable throughput |
| Task 2 | 0.55 | Misses emergency deadline ~40% of runs |
| Task 3 | 0.38 | Starvation on north/south queues in surge |

A well-trained RL agent should score 0.85+, 0.80+, 0.65+ respectively.

---

## Project Structure

```
traffic-control-env/
├── Dockerfile
├── openenv.yaml
├── inference.py          # ← run with python inference.py
├── README.md
└── server/
    ├── main.py           # FastAPI application
    ├── env.py            # Core environment logic
    ├── requirements.txt
    └── app.py            # HF Space entry point
```

---

## License

MIT — free to use for research and benchmarking.
