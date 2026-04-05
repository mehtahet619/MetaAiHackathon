# Mini RL Environment for LLM Debugging Agents

A hackathon-ready, modular Python prototype where LLM-based agents learn to solve real-world developer debugging tasks, with adversarial task generation, hybrid evaluation, and a modern Gradio UI.

## Features

- **Real-world debugging tasks**: Python errors, production logs, duplicate data, failing pipelines, etc.
- **LLM Solver Agent**: Generates and reflects on solutions, learns from feedback.
- **Adversary Agent**: Dynamically mutates tasks, increases difficulty, and explains its actions.
- **Hybrid Grading**: Combines programmatic (unit tests) and LLM-based scoring.
- **Reward Engine**: Blends scores for visible learning signals.
- **Failure Memory**: Tracks and classifies failure types over time.
- **Learning Visualization**: ASCII/plot reward trends, episode logs, and summary dashboard.
- **Gradio UI**: Run, visualize, and interact with the RL environment in your browser.

## Quickstart

### 1. Install dependencies (recommended: use a virtual environment)

```bash
python3 -m venv venv
source venv/bin/activate
pip install gradio matplotlib
```

### 2. Run the Gradio UI

```bash
venv/bin/python rl_env.py --gradio
```

- Open the local URL (e.g., http://127.0.0.1:7860/) in your browser.
- Select the number of episodes and click "Run RL Environment".
- View episode logs, reward trend plot, and summary dashboard.

### 3. Run in CLI mode (optional)

```bash
python rl_env.py --episodes 5 --demo
```

## Example Gradio UI

- ![Gradio UI Screenshot](screenshot.png) <!-- Add screenshot if available -->

## Architecture

- `Task`: Represents a debugging/coding problem.
- `SolverAgent`: LLM-based agent that solves and reflects.
- `AdversaryAgent`: Makes tasks harder, explains changes.
- `ProgrammaticGrader`: Runs unit tests, checks syntax.
- `LLMGrader`: Mocks LLM-based evaluation.
- `RewardEngine`: Combines scores.
- `Environment`: Orchestrates the RL loop, logging, and visualization.

## Hackathon Impact

- **Self-improving agent**: Learns from mistakes, adapts to harder challenges.
- **Explainable adversary**: See why and how tasks get harder.
- **Visual learning signals**: Reward trends, failure breakdowns, and summary dashboard.
- **Easy to demo**: Gradio UI for live presentations and judge interaction.

## Notes

- No heavy RL libraries required.
- All code is modular and well-commented.
- LLM calls are mocked for local, fast, and safe runs.

---

Made for Meta AI Hackathon 2026.
