# Swim Meet Lineup Optimizer

Automated, data-driven system for optimizing competitive swim meet lineups using Reinforcement Learning and deterministic constraint solving. Treats lineup generation as a combinatorial optimization and sequential decision-making problem to maximize total team points under city league constraints.

## Problem

Assigning a finite roster of athletes to a predefined schedule of events is heavily constrained:

- **Event limits:** Max 2 individual events + 2 relays (4 total) per athlete per meet
- **Fatigue dynamics:** Performance degrades after back-to-back events (modeled as a time decay coefficient)
- **Scoring mechanics:** Points are awarded by relative finish position against the opponent, not absolute time — making opponent strategy a critical variable

## Approach

Two parallel solvers are implemented and compared:

| Approach | Framework | Use Case |
|---|---|---|
| RL Agent | Gymnasium + Stable Baselines3 (MaskablePPO) | Handles fatigue, stochastic opponent lineups via domain randomization |
| Deterministic Baseline | Google OR-Tools (CP-SAT) | Guarantees optimal lineup under static, known conditions |

The RL environment models the meet as a Sequential Decision Process (MDP). Action masking is a hard requirement — it zeroes out illegal assignments so the agent never produces an invalid lineup.

## Architecture

```
Data Stores (CSV / DB)
    └── ETL Pipeline (clean → impute missing times → normalize to power points)
            └── Gymnasium Training Loop
                    ├── State Space (roster fatigue, event context, opponent projections)
                    ├── Action Masking (enforce event limits + eligibility)
                    ├── Meet Simulation & Physics (fatigue decay)
                    └── Reward (terminal: total meet points)
                            └── Trained Policy Weights → Optimal Lineup
```

## Project Structure

```
swim_rl/
├── data/
│   ├── raw/                    # Swimmer history CSVs, opponent scouting data
│   ├── processed/              # Cleaned feature vectors
│   └── league_rules.json       # Constraint config (event limits, scoring rules)
├── src/
│   ├── data/
│   │   ├── pipeline.py         # DataPipeline: clean, impute, normalize
│   │   └── imputation.py       # Stroke profiling imputation (linear regression)
│   ├── models/
│   │   ├── swimmer.py          # Swimmer dataclass + fatigue decay logic
│   │   ├── roster.py           # Roster: eligibility tracking, fatigue updates
│   │   └── meet_schedule.py    # MeetSchedule: events, scoring rules, opponent projections
│   ├── env/
│   │   └── swim_meet_env.py    # SwimMeetEnv (Gymnasium) + action_masks()
│   ├── solver/
│   │   └── or_tools_solver.py  # CP-SAT deterministic baseline
│   └── training/
│       └── train.py            # MaskablePPO training entry point
├── models/                     # Saved policy weights
├── notebooks/
│   └── exploration.ipynb       # EDA, imputation validation
└── requirements.txt
```

## Development Phases

- **Phase 1 — Baseline:** OR-Tools CP-SAT solver with full constraint logic validated against historical meets
- **Phase 2 — Gym Prototype:** Custom Gymnasium environment with action masking; sanity-checked with random rollouts
- **Phase 3 — RL Training:** MaskablePPO via Stable Baselines3; hyperparameter tuning; benchmark against OR-Tools baseline
- **Phase 4 — Simulation UI:** Frontend to input opponent roster and output recommended lineup

## Key Design Decisions

**Action Masking (MaskablePPO):** Required because unconstrained PPO will always assign the fastest swimmer to every event. `action_masks()` returns a boolean array that makes illegal assignments impossible at the policy level.

**Terminal Reward:** No intermediate rewards during the assignment phase. The full meet is simulated at episode end and total points become the reward signal. This prevents greedy short-sighted assignments in early events.

**Domain Randomization for Opponent:** Opponent lineup is randomly sampled from a distribution of plausible lineups across thousands of training episodes, producing a robust policy that doesn't overfit to a single guessed opponent.

**Stroke Profiling Imputation:** Swimmers rarely compete in every event. Missing times are extrapolated via linear regression against team benchmarks (e.g., predicting 200 IM time from 100 Breast + 100 Fly splits).

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run the deterministic OR-Tools baseline
python -m src.solver.or_tools_solver

# Train the RL agent
python -m src.training.train

# Run inference with a trained policy
python -m src.training.train --mode infer --model models/policy.zip
```

## Tech Stack

- `gymnasium` — RL environment interface
- `stable-baselines3` + `sb3-contrib` — MaskablePPO algorithm
- `ortools` — CP-SAT deterministic solver
- `pandas`, `numpy`, `scipy` — data pipeline and imputation
