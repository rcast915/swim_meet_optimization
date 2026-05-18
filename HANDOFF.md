# Project Handoff — Swim Meet Lineup Optimizer

## What This Project Is

A reinforcement learning system that optimizes competitive swim meet lineups for **HAW-ZZ (Hawkins)** in the **City of El Paso Gus & Goldie Summer Swim League (Public Division)**. The optimizer treats lineup generation as a sequential decision problem — assigning swimmers to events to maximize team points while obeying league rules.

Two solvers are being built in parallel:
- **RL agent** (Gymnasium + Stable Baselines3 MaskablePPO) — handles fatigue and stochastic opponent strategy
- **Deterministic baseline** (Google OR-Tools CP-SAT) — mathematically optimal under static conditions

Design doc is at [`Design Doc Swim Meet Lineup Optimization.pdf`](Design%20Doc%20Swim%20Meet%20Lineup%20Optimization.pdf). Read it first.

---

## League Facts (Critical Context)

**League:** Gus & Goldie Summer Swim League, Public Division  
**Our team:** HAW-ZZ (Hawkins)  
**Opponents:** COW (Cowan), VETS-ZZ (Veterans), MEM-ZZ (Memorial), LCMP  
**Pool:** Short course yards (25-yard pool)  
**Season format:** Regular season dual/tri-meets → City Championship (prelim/final, top 16 per event advance)  
**Championship date (2024):** July 13–14  

**Per-swimmer limits (hard rules):**
- Max 2 individual events per meet
- Max 2 relays per meet
- Max 4 total events per meet
- Must swim in at least 1 individual event to be eligible for championship

**Events by age group (individual only — these are the events we optimize):**

| Age Group | Events |
|-----------|--------|
| 6 & Under | 25 butterfly, 25 backstroke, 25 freestyle, 25 breaststroke |
| 7–8 | 25 butterfly, 25 backstroke, 25 freestyle, 25 breaststroke |
| 9–10 | 25 butterfly, 25 backstroke, 25 freestyle, 25 breaststroke, 100 IM |
| 11–12 | 50 butterfly, 50 backstroke, 50 freestyle, 50 breaststroke, 100 IM |
| 13–14 | 50 butterfly, 50 backstroke, 50 freestyle, 50 breaststroke, 100 IM |
| 15–18 | 50 butterfly, 50 backstroke, 50 freestyle, 50 breaststroke, 100 IM |

**Scoring:** Points by finish position against opponent. Individual: [6, 4, 3, 2, 1, 0]. Relay: [8, 4, 2, 0].

**Key quirk:** Scoring is relative finish vs. opponent — NOT absolute time. A slow swimmer who beats their opponent scores full points. This is why a greedy "put the fastest swimmer in every event" strategy is suboptimal and why the RL agent is valuable.

---

## What Has Been Built

### Source Code (`src/`)

| File | Status | What It Does |
|------|--------|--------------|
| `src/models/swimmer.py` | ✅ Done | `Swimmer` dataclass — age, gender, auto-computed `age_group`, best times, fatigue decay, eligibility checks |
| `src/models/roster.py` | ✅ Done | `Roster` — event count tracking, action masks filtered by age group + gender |
| `src/models/meet_schedule.py` | ✅ Done | `MeetSchedule` + `MeetEvent` — events carry `age_group` and `gender`; opponent time sampling keyed by `(age_group, gender, event_id)` |
| `src/env/swim_meet_env.py` | ✅ Done | `SwimMeetEnv` (Gymnasium) — full RL environment with `action_masks()` and terminal reward |
| `src/data/pipeline.py` | ✅ Done | `DataPipeline` — loads roster CSV, imputes missing times, builds swimmer list, loads opponent projections |
| `src/data/imputation.py` | ✅ Done | `StrokeProfileImputer` — linear regression to fill missing stroke times within the correct distance tier |
| `src/solver/or_tools_solver.py` | ✅ Done | CP-SAT deterministic baseline — enforces age group + gender hard constraints |
| `src/training/train.py` | ✅ Done | MaskablePPO training + inference entry point (untested) |

### Data (`data/`)

| File | What It Is |
|------|------------|
| `data/league_rules.json` | League constraints, event lists by age group, scoring tables |
| `data/raw/roster.csv` | HAW-ZZ roster — **dummy data**, will be replaced with real swimmer times |
| `data/raw/opponent_times.csv` | Historical opponent times — **dummy data**, will be replaced with real meet history |

> **All current data is dummy/placeholder.** Do not tune logic around these values. Replace both files when real data is available.

### `data/raw/roster.csv` Schema

```
name, age, gender,
25_freestyle, 25_butterfly, 25_backstroke, 25_breaststroke,
50_freestyle, 50_butterfly, 50_backstroke, 50_breaststroke,
100_im
```

Times in **seconds**. Blank = swimmer doesn't compete in that event. Ages 6–10 have 25-yard columns; ages 11+ have 50-yard columns. No team column — this file is always our roster.

### `data/raw/opponent_times.csv` Schema

```
age_group, gender, event, time_seconds
```

Each row is one historical opponent swim. Pool times from multiple meets/years — team identity is irrelevant. The pipeline groups these into distributions keyed by `(age_group, gender, event)` for the opponent projection system.

---

## What Needs to Be Done Next (Priority Order)

### 1. Test the OR-Tools baseline (NEXT)

`src/solver/or_tools_solver.py` has never been run. Build a full `MeetSchedule` with all individual events for a meet, run the solver, and verify the assignments are logically correct (right age groups, no swimmer over-assigned, etc.).

### 2. Test the Gymnasium environment

`src/env/swim_meet_env.py` has never been executed. Sanity check:
- Instantiate `SwimMeetEnv` with the HAW-ZZ roster and a full schedule
- Run random rollouts using `env.action_masks()` — verify no illegal assignments slip through
- Confirm episode terminates correctly and reward is returned only at the end

### 3. Training

Once 1–2 pass, run `python -m src.training.train` with the HAW-ZZ roster and opponent times. The `MaskablePPO` entry point in `src/training/train.py` is wired up but untested.

---

## Python Environment

- **Use this Python:** `/Users/raycast/anaconda3/envs/swim_rl/bin/python`
- **Conda env:** `swim_rl` — created specifically for this project, isolated from base Anaconda env
- **All packages installed:** `gymnasium`, `stable-baselines3`, `sb3-contrib`, `ortools`, `pandas`, `numpy`, `scipy`, `scikit-learn`
- **Do NOT use** the base Anaconda Python (`/Users/raycast/anaconda3/bin/python`) — it has NumPy version conflicts that break pandas and sklearn

To activate in terminal:
```bash
conda activate swim_rl
```

To run directly without activating:
```bash
/Users/raycast/anaconda3/envs/swim_rl/bin/python -m src.training.train
```

---

## Repository State

- Branch: `main`
- Last commit: `151dffd` — "updated read me and created file structure"
- All source files and data changes are uncommitted working tree changes

---

## Quick Orientation Commands

```bash
# Activate environment
conda activate swim_rl

# See data files
ls data/raw/

# Check roster
head -5 data/raw/roster.csv

# See league rules
cat data/league_rules.json

# Run smoke test (pipeline + action masks)
/Users/raycast/anaconda3/envs/swim_rl/bin/python - <<'EOF'
import json
from src.data.pipeline import DataPipeline
with open("data/league_rules.json") as f:
    rules = json.load(f)
pipeline = DataPipeline(rules)
df = pipeline.load("data/raw/roster.csv")
swimmers = pipeline.build_roster(df)
print(f"{len(swimmers)} swimmers loaded")
EOF
```
