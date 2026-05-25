# Ray's Project Plan — Swim Meet Optimizer

The idea is to frame lineup assignment as a sequential decision problem and solve it two ways: a deterministic CP-SAT solver for a provably optimal baseline, and an RL agent that can handle stochastic opponent
strategy and swimmer fatigue.

Naomi is handling the data and models layer. My work depends on her finishing first —
coordinate on the `Swimmer`, `Roster`, `MeetEvent`, and `MeetSchedule` interfaces
before I start the solver.

---

## What I Need to Understand First

- [ ] Be able to answer: why does greedy fail here? Why do we need two solvers?
- [ ] Nail down the MDP formulation — what is the state, action, and reward for this problem?
- [ ] Understand why terminal reward makes more sense than per-step reward here

---

## Phase 1 — OR-Tools Solver (`src/solver/or_tools_solver.py`)

I want a deterministic baseline before touching RL. CP-SAT is the right tool —
the per-swimmer limits and age/gender eligibility are hard constraints, not soft penalties.

**What I need to figure out:**
- The objective is currently a proxy (inverted time). A better objective would compare
  our time against a sampled opponent time and look up the scoring table. Worth revisiting
  after the RL side is working.
- The slot constraint must be `<=` not `==` — if we have no eligible swimmer for an event
  (e.g. no 6U girls on the roster), the solver needs to skip it gracefully, not error out.
- Result key must be `(age_group, gender, event_id)` — not just `event_id`. Multiple events
  share the same `event_id` across age groups and using a bare string key silently overwrites entries.

**Tasks:**
- [ ] Define binary variables `x[i][e]` — 1 if swimmer i assigned to event e
- [ ] Add slot constraint (`<=`) per event
- [ ] Add per-swimmer total, individual, and relay limit constraints
- [ ] Add eligibility hard constraints — zero out any (swimmer, event) pair where age_group
      or gender doesn't match, or swimmer has no time for that event
- [ ] Build the objective — inverted time proxy for now
- [ ] Return `dict[(age_group, gender, event_id), list[str]]`
- [ ] Validate: no swimmer over 2 individual events, all assignments in correct bracket,
      empty brackets produce empty lists not errors

---

## Phase 2 — Gymnasium Environment (`src/env/swim_meet_env.py`)

This is the core of the RL side. I need to frame lineup assignment as a proper MDP.

**My formulation:**
- State: current event index + event count per swimmer (roster status array)
- Action: swimmer index to assign to the current event
- Reward: total simulated meet points, only at episode termination
- Termination: all events have been assigned

**Why action masking matters:** Without it, the agent wastes its entire learning budget
figuring out that illegal actions (wrong age group, over-limit swimmers) are bad.
MaskablePPO zeros out illegal action probabilities before sampling so the agent only
ever explores legal assignments.

**Tasks:**
- [ ] Set up `observation_space` and `action_space`
- [ ] Implement `reset()` — reset roster, counters, return initial obs
- [ ] Implement `step(action)`:
  - Reject illegal actions with large negative reward and early termination
  - Get swimmer's decayed time (fatigue if they just swam)
  - Sample an opponent time from the schedule's projection for that event
  - Score the event, accumulate points
  - Assign the swimmer, advance to next event
  - Return terminal reward only on the last event
- [ ] Implement `action_masks()` — delegate to `roster.action_mask(current_event)`
- [ ] Implement `_score_event()` — compare times, return points from scoring table
- [ ] Validate with random rollouts — confirm no illegal assignments slip through
      and the episode terminates with a non-zero reward

---

## Phase 3 — Training Pipeline (`src/training/train.py`)

- [ ] Implement `build_schedule(rules, opponent_projections) -> MeetSchedule`
  - Iterate `rules["individual_events_by_age_group"]` × both genders
  - Each combination becomes one `MeetEvent` in the schedule
- [ ] Implement `make_env(rules) -> SwimMeetEnv`
  - Load + impute roster, load opponent times, build schedule, return wrapped env
- [ ] Implement `train(total_timesteps)`:
  - Wrap env with `ActionMasker`
  - Run `MaskablePPO("MultiInputPolicy", env, verbose=1).learn()`
  - Save policy to `models/policy.zip`
- [ ] Implement `infer(model_path)` — load saved policy, run one episode, print total points
- [ ] Smoke run at 10k timesteps first to catch any crashes before committing to a full run
- [ ] Full training run at 500k+ timesteps

---

## Phase 4 — Evaluation

I want to know whether the RL agent actually adds value over the deterministic baseline.

- [ ] OR-Tools expected score: for each assignment, compare swimmer time against mean
      of the opponent distribution for that event, look up scoring table
- [ ] RL agent score: run 100 episodes (opponent times sampled each time), record mean ± std
- [ ] Compare the two — does the RL agent beat OR-Tools on average?
- [ ] Identify conditions where RL wins (high opponent variance) vs. where OR-Tools is fine
- [ ] Revisit the OR-Tools objective — replace inverted time proxy with actual expected points
      and re-run to see if the gap narrows

---

## Phase 5 — Real Data (coordinate with Naomi)

- [ ] Get real roster from Naomi and Victoria — swap out `data/raw/roster.csv`
- [ ] Get real meet history (request `.hyv`/`.hy3` from Martin)
      and build a proper `data/raw/opponent_times.csv` from it
- [ ] Re-run everything on real data and sanity-check results against actual knowledge of the team

---

## Notes

- **Python env:** `conda activate swim_rl` or `/Users/raycast/anaconda3/envs/swim_rl/bin/python`
- **Do not use base Anaconda Python** — NumPy version conflicts break pandas and sklearn there
- **All current data is dummy** — don't tune thresholds or hyperparameters against it
- **Naomi's deliverables I depend on:** `Swimmer`, `MeetEvent`, `MeetSchedule`, `Roster`, `DataPipeline`
- **Interface note:** `opponent_projections` is `dict[(age_group, gender, event_id), (mean, std)]` — Gaussian parameters fitted from historical times, not raw lists. `sample_opponent_time` calls `rng.normal(mean, std)` clamped to min 1.0.
