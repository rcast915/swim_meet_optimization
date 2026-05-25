# Swim Meet Lineup Optimizer

Automated, data-driven system for optimizing competitive swim meet lineups using reinforcement learning and deterministic constraint solving. Treats lineup assignment as a sequential decision-making problem to maximize total team points under city league constraints.

Built for **HAW-ZZ (Hawkins)** competing in the **El Paso Gus & Goldie Summer Swim League, Public Division**.

---

## Problem

Assigning a finite roster of athletes to a predefined schedule of events is heavily constrained:

- **Age group and gender rules:** Each event is restricted to a specific age bracket and gender — a swimmer can only compete in events for their own group
- **Event limits:** Max 2 individual events + 2 relays (4 total) per athlete per meet
- **Fatigue dynamics:** Performance degrades after back-to-back events, modeled as a time decay coefficient
- **Scoring mechanics:** Points are awarded by relative finish position against the opponent, not absolute time — making opponent strategy a critical variable

The key insight: "put the fastest swimmer in every event" is a suboptimal greedy strategy. A slower swimmer who beats their opponent scores the same 6 points as an elite swimmer beating theirs. The optimizer needs to reason about matchups, not just times.

---

## Approach

Two parallel solvers are built and compared:

| Approach | Framework | Strengths |
|---|---|---|
| Deterministic Baseline | Google OR-Tools (CP-SAT) | Provably optimal under static, known conditions |
| RL Agent | Gymnasium + Stable Baselines3 (MaskablePPO) | Handles stochastic opponent times and swimmer fatigue across many episodes |

The RL environment models the meet as a Markov Decision Process. Action masking is required — it zeroes out illegal assignments (wrong age group, wrong gender, over event limit) so the agent only ever explores valid lineups.

---

## Architecture

```
Data Layer
  ├── data/raw/roster.csv          # Our swimmers — name, age, gender, best times
  └── data/raw/opponent_times.csv  # Historical opponent times by (age_group, gender, event)
        │
        ▼
  DataPipeline
  ├── load + clean                 # Normalize column names, drop bad rows
  ├── impute missing times         # Linear regression within distance tier
  └── build_roster / load_opponent_times
        │
        ▼
  Models
  ├── Swimmer                      # Dataclass: age_group, best_times, fatigue, eligibility
  ├── MeetEvent                    # age_group + gender + event_id + scoring_table
  ├── MeetSchedule                 # Ordered event list + opponent projection dict
  └── Roster                       # Action masks filtered by age_group + gender + event counts
        │
   ┌────┴────┐
   ▼         ▼
OR-Tools   SwimMeetEnv (Gymnasium)
CP-SAT     └── MaskablePPO training
Solver         └── Trained policy → optimal lineup
```

---

## Project Structure

```
swim_rl/
├── data/
│   ├── raw/
│   │   ├── roster.csv              # Our team's swimmers and best times
│   │   └── opponent_times.csv      # Historical opponent times (age_group, gender, event, time_seconds)
│   └── league_rules.json           # Age groups, events per group, scoring tables, event limits
├── src/
│   ├── data/
│   │   ├── pipeline.py             # DataPipeline: load, clean, impute, build roster
│   │   └── imputation.py           # StrokeProfileImputer: linear regression for missing times
│   ├── models/
│   │   ├── swimmer.py              # Swimmer dataclass + fatigue decay + eligibility
│   │   ├── roster.py               # Roster: age/gender-aware action masks, event tracking
│   │   └── meet_schedule.py        # MeetEvent + MeetSchedule + opponent time sampling
│   ├── env/
│   │   └── swim_meet_env.py        # SwimMeetEnv (Gymnasium) + action_masks()
│   ├── solver/
│   │   └── or_tools_solver.py      # CP-SAT deterministic baseline
│   └── training/
│       └── train.py                # build_schedule, make_env, MaskablePPO train + infer
├── models/                         # Saved policy weights (.zip)
├── notebooks/
└── requirements.txt
```

---

## Data Schemas

**`data/raw/roster.csv`** — our swimmers, one row per athlete:
```
name, age, gender,
25_freestyle, 25_butterfly, 25_backstroke, 25_breaststroke,
50_freestyle, 50_butterfly, 50_backstroke, 50_breaststroke, 100_im
```
Times in seconds. Blank = swimmer doesn't compete in that event. Ages 6–10 have 25-yard columns; 11+ have 50-yard columns.

**`data/raw/opponent_times.csv`** — pooled historical opponent times:
```
age_group, gender, event, time_seconds
```
Team identity is not tracked — only the distribution of times per `(age_group, gender, event)` matters.

> **All current data is dummy/placeholder.** Replace both files when real data is available.

---

## League Rules

| Age Group | Individual Events |
|-----------|------------------|
| 6 & Under | 25 fly, 25 back, 25 free, 25 breast |
| 7–8 | 25 fly, 25 back, 25 free, 25 breast |
| 9–10 | 25 fly, 25 back, 25 free, 25 breast, 100 IM |
| 11–12 | 50 fly, 50 back, 50 free, 50 breast, 100 IM |
| 13–14 | 50 fly, 50 back, 50 free, 50 breast, 100 IM |
| 15–18 | 50 fly, 50 back, 50 free, 50 breast, 100 IM |

Per-swimmer limits: max 2 individual events, max 2 relays, max 4 total.
Scoring: individual [6, 4, 3, 2, 1, 0] — relay [8, 4, 2, 0].

---

## Key Design Decisions

**No team tracking** — the tool is built around "our roster vs. a distribution of opponent times." Any coach can drop in their own roster CSV and historical meet data.

**Age group + gender on every event** — `MeetEvent` carries both fields. The action mask uses them to filter the roster to only eligible swimmers for that specific heat.

**Opponent projection key is a 3-tuple `(age_group, gender, event_id)`** — using just `event_id` would collapse all age groups together, producing nonsensical time distributions.

**Slot constraint is `<=`, not `==`** — if no eligible swimmer exists for an event bracket (e.g. no 6U girls on the roster), the solver skips it gracefully instead of raising an error.

**Imputation stays within the distance tier** — the `StrokeProfileImputer` only predicts 25-yard stroke times from 25-yard freestyle, and 50-yard stroke times from 50-yard freestyle. A young swimmer will never get a 50-yard time imputed.

**Terminal reward** — no intermediate rewards during assignment. The full meet is simulated at episode end and total points become the reward signal, preventing greedy short-sighted early assignments.

**Action masking (MaskablePPO)** — without masking, the agent wastes its entire learning budget figuring out that illegal actions are bad. MaskablePPO zeroes out illegal action probabilities before sampling so the agent only ever explores valid assignments.

---

## Development Phases

- **Phase 1** — Data layer + core models (Naomi): `Swimmer`, `MeetEvent`, `MeetSchedule`, `Roster`, `DataPipeline`, `StrokeProfileImputer`
- **Phase 2** — OR-Tools baseline (Ray): CP-SAT solver validated against full meet schedule
- **Phase 3** — Gymnasium environment (Ray): `SwimMeetEnv` with action masking, verified with random rollouts
- **Phase 4** — RL training (Ray): MaskablePPO trained and benchmarked against OR-Tools baseline
- **Phase 5** — Real data: replace dummy CSVs with actual HAW-ZZ roster and meet history

---

## Setup

```bash
# Create and activate the conda environment
conda create -n swim_rl python=3.11
conda activate swim_rl
pip install -r requirements.txt
```

> Use `/Users/raycast/anaconda3/envs/swim_rl/bin/python` — do **not** use base Anaconda Python (NumPy version conflicts break pandas and sklearn).

## Usage

```bash
conda activate swim_rl

# Train the RL agent
python -m src.training.train --mode train

# Run inference with a trained policy
python -m src.training.train --mode infer --model models/policy.zip
```

---

## Tech Stack

- `gymnasium` — RL environment interface
- `stable-baselines3` + `sb3-contrib` — MaskablePPO algorithm
- `ortools` — CP-SAT deterministic solver
- `pandas`, `numpy`, `scipy` — data pipeline
- `scikit-learn` — linear regression for time imputation
