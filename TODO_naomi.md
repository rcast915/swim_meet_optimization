# Naomi's TODO — Swim Meet Optimizer

You own the **data layer, opponent modeling, analysis, and the GUI**. My RL environment
and solver can't work without the objects you build here, and the whole thing is useless
without a good interface at the end.

Before writing any code, read `data/league_rules.json`, `data/raw/roster.csv`, and
`data/raw/opponent_times.csv`. You should be able to answer these before Phase 1:

- Why are the 50-yard columns blank for a 7-year-old swimmer?
- Why do we pool opponent times across all teams and years instead of tracking who swam what?
- Why is "put your fastest swimmer in every event" a bad strategy in this league?

---

## Phase 0 — Understand the Problem

- [ ] Read the design doc (`Design Doc Swim Meet Lineup Optimization.pdf`)
- [ ] Open `data/league_rules.json` — trace how age groups map to events
- [ ] Open `data/raw/roster.csv` — understand why some columns are blank per swimmer
- [ ] Open `data/raw/opponent_times.csv` — understand the schema and what it represents
- [ ] Answer: what does a swimmer need to qualify for the City Championship?

---

## Phase 1 — The Swimmer Model (`src/models/swimmer.py`)

This is the core object. Everything downstream uses it.

**Learn first:**
- What is a Python `dataclass`? Look up the `@dataclass` decorator and `field()`.
- What does `__post_init__` do? (It runs automatically after `__init__`.)

**Build:**
- [ ] Create the `Swimmer` dataclass with fields:
  - `name: str`
  - `age: int`
  - `gender: str` — `'M'` or `'F'`
  - `best_times: dict[str, float]` — maps event name to time in seconds
  - `age_group: str` — do NOT make this an `__init__` parameter; compute it automatically in `__post_init__` by calling a static method
  - `current_events_count: int = 0`
  - `individual_events_count: int = 0`
  - `relay_events_count: int = 0`
  - `fatigue_coefficient: float = 1.04`
- [ ] Implement `_age_to_group(age) -> str` as a `@staticmethod`
  - Maps age → one of: `"6U"`, `"7-8"`, `"9-10"`, `"11-12"`, `"13-14"`, `"15-18"`
- [ ] Implement `is_eligible(is_relay, max_individual=2, max_relay=2, max_total=4) -> bool`
  - Returns `False` if any limit would be exceeded by adding this event
- [ ] Implement `calculate_decayed_time(event_id, events_since_last) -> float`
  - If `events_since_last == 0` (swimmer just swam), return `best_times[event_id] * fatigue_coefficient`
  - Otherwise return `best_times[event_id]`
  - Raise `ValueError` if the swimmer has no time for that event

**Test in a Python shell:**
```python
from src.models.swimmer import Swimmer
s = Swimmer(name="Test", age=9, gender="F", best_times={"25_freestyle": 22.5})
print(s.age_group)           # should be "9-10"
print(s.is_eligible(False))  # should be True
s.individual_events_count = 2
print(s.is_eligible(False))  # should be False
```

---

## Phase 2 — Meet Schedule Models (`src/models/meet_schedule.py`)

**Learn first:**
- Python `dataclass` with `field(default_factory=dict)` — why you need `default_factory` for mutable defaults.

**Build:**
- [ ] Create `MeetEvent` dataclass:
  - `event_id: str` — e.g. `"25_freestyle"`
  - `age_group: str` — e.g. `"9-10"`
  - `gender: str` — `'M'` or `'F'`
  - `is_relay: bool`
  - `slots: int` — `1` for individual, `4` for relay
  - `scoring_table: list[int]` — points by finish position
- [ ] Create `MeetSchedule` dataclass:
  - `events: list[MeetEvent]`
  - `opponent_projections: dict[tuple[str, str, str], tuple[float, float]]` — keyed by `(age_group, gender, event_id)`, value is `(mean, std)` of a fitted Gaussian
  - Implement `__len__` and `__getitem__` so you can do `schedule[0]` and `len(schedule)`
- [ ] Implement `sample_opponent_time(event, rng) -> float`
  - Look up the `(age_group, gender, event_id)` key to get `(mean, std)`
  - Sample from a Gaussian: `rng.normal(mean, std)` — clamp to a minimum of 1.0 so times are never negative
  - Return `float("inf")` if no projection exists for that key

**Think about:** Why do we sample from a distribution instead of just using the mean opponent time every episode? (Hint: the RL agent trains across thousands of episodes)

---

## Phase 3 — Data Pipeline (`src/data/pipeline.py` + `src/data/imputation.py`)

**Learn first:**
- `pandas` basics: `pd.read_csv()`, `df.iterrows()`, `pd.notna()`, column selection.
- What is linear regression? (One variable predicts another — e.g., 25_freestyle time predicts 25_butterfly time.)
- Why does linear regression work for imputing swim times? (Strokes within the same distance tier are correlated.)

### 3a. Pipeline
- [ ] Implement `load(path) -> pd.DataFrame` — reads CSV, calls `_clean()`
- [ ] Implement `_clean(df) -> pd.DataFrame`
  - Normalize column names: lowercase, strip whitespace, replace spaces with underscores
  - Drop any rows where `name` is null
- [ ] Implement `build_roster(df) -> list[Swimmer]`
  - Identify event columns (everything that is not `name`, `age`, `gender`)
  - For each row, build `best_times` from only the non-null event columns
  - Construct a `Swimmer` for each row
- [ ] Implement `load_opponent_times(path) -> dict[tuple, tuple[float, float]]`
  - Read `opponent_times.csv`, group times by `(age_group, gender, event)`
  - For each group, compute the **mean and standard deviation** of the times
  - Return a dict mapping each key to `(mean, std)` — this is what `MeetSchedule` uses to sample opponents
  - Learn: what is standard deviation? why does a higher std mean more unpredictable opponents?
- [ ] Add `impute_missing_times(df) -> pd.DataFrame` that calls `StrokeProfileImputer.fit_transform(df)`

### 3b. Imputation
- [ ] Implement `StrokeProfileImputer` with this `PREDICTOR_MAP`:
  ```python
  {
      "25_butterfly":    ["25_freestyle"],
      "25_backstroke":   ["25_freestyle"],
      "25_breaststroke": ["25_freestyle"],
      "50_butterfly":    ["50_freestyle"],
      "50_backstroke":   ["50_freestyle"],
      "50_breaststroke": ["50_freestyle"],
      "100_im":          ["50_freestyle"],
  }
  ```
- [ ] Implement `fit(df)` — for each target in the map, train a `LinearRegression` using only rows that have both target and predictor values (use `.dropna()`)
  - Skip if fewer than 3 training rows
- [ ] Implement `transform(df)` — for each fitted model, fill in missing target values
  - Only fill rows where target is null AND predictor is not null
- [ ] Implement `fit_transform(df)` — fit then transform

**Test the full pipeline:**
```python
import json
from src.data.pipeline import DataPipeline
with open("data/league_rules.json") as f:
    rules = json.load(f)
pipeline = DataPipeline(rules)
df = pipeline.load("data/raw/roster.csv")
df = pipeline.impute_missing_times(df)
swimmers = pipeline.build_roster(df)
projections = pipeline.load_opponent_times("data/raw/opponent_times.csv")
for s in swimmers:
    print(s.name, s.age_group, s.gender, list(s.best_times.keys()))
print(projections[("9-10", "M", "25_freestyle")])  # should print (mean, std)
```

---

## Phase 4 — Roster (`src/models/roster.py`)

**Learn first:**
- `numpy` boolean arrays — creating them with `np.zeros(..., dtype=bool)` and indexing them.
- What is an "action mask"? (A boolean array that tells the RL agent which actions are legal right now.)

**Build:**
- [ ] Implement `Roster.__init__(swimmers)` — store the list, initialize `_last_event_idx` dict mapping each swimmer name to `-999`
- [ ] Implement `get_eligible_swimmers(event) -> list[int]` — returns indices of swimmers who:
  1. Match `event.age_group`
  2. Match `event.gender`
  3. Pass `swimmer.is_eligible(event.is_relay)`
- [ ] Implement `action_mask(event) -> np.ndarray` — boolean array, `True` at each eligible swimmer index
- [ ] Implement `assign(swimmer_idx, event_id, event_order, is_relay)` — increment the swimmer's counts, record the event order
- [ ] Implement `events_since_last(swimmer_idx, current_event_order) -> int`
- [ ] Implement `reset()` — zero all counts, reset `_last_event_idx`
- [ ] Implement `to_status_array() -> np.ndarray` — array of `current_events_count` per swimmer

**Test:**
```python
from src.models.roster import Roster
from src.models.meet_schedule import MeetEvent
roster = Roster(swimmers)
event = MeetEvent("25_freestyle", "9-10", "M", False, 1, [6,4,3,2,1,0])
mask = roster.action_mask(event)
print([swimmers[i].name for i in range(len(swimmers)) if mask[i]])
# Should only print 9-10 year old boys
```

---

## Sync Point — Hand Off to Me

Before moving on, confirm these are all working and tested:
1. `Swimmer` — `age_group`, `is_eligible()`, `calculate_decayed_time()`
2. `MeetEvent` + `MeetSchedule` — including `sample_opponent_time()` using the fitted Gaussian
3. `Roster` — `action_mask()` and `assign()`
4. `DataPipeline` — `build_roster()` and `load_opponent_times()` returning `(mean, std)` tuples

Run the full pipeline test from Phase 3 and confirm it prints sensible output. Then let me know — I need this before I can build the environment.

**Phases 5 and 6 below are blocked on me finishing the solver and environment. Start Phase 5 (EDA) while you wait since it only needs the pipeline.**

---

## Phase 5 — EDA Notebook (`notebooks/eda.ipynb`)

You can start this while I'm building the solver. It only needs your pipeline to be working.

Now that the data layer is working, actually analyze it. This is where you go from
"loading data" to "understanding data".

**Learn first:**
- `matplotlib` and `seaborn` for plotting
- What does a distribution of swim times look like? Is it symmetric? Skewed?
- What does it mean for two variables to be correlated? (This validates your imputation approach.)

**Build `notebooks/eda.ipynb`:**
- [ ] Load the full roster — how many swimmers per age group and gender? Are any brackets empty?
- [ ] Plot the distribution of times for each event — are they roughly Gaussian?
  - This matters because `sample_opponent_time` assumes a Gaussian — if times are heavily skewed, we may need a different distribution
- [ ] Validate the imputer: for swimmers who have both freestyle and butterfly times, hide the butterfly time, run the imputer, and compare predicted vs. actual. How accurate is it?
- [ ] Plot the opponent time distributions (mean ± std) for each bracket — which brackets have the most variance? High variance = unpredictable opponents = where the RL agent has the biggest advantage over OR-Tools
- [ ] Flag any age group + gender brackets where we have no swimmers — those are automatic forfeits

---

## Phase 6 — Exploring RL (`notebooks/eda.ipynb` continued)

> **Blocked on me finishing the Gymnasium environment.** I'll let you know when it's ready.

You won't build the RL agent yourself, but you need to understand what it's doing
well enough to display its output in the GUI meaningfully.

**Learn first:**
- What is a Markov Decision Process? Write down the state, action, and reward for *this specific problem* in your own words
- What does "episode" mean in RL? What is one episode in a swim meet context?
- What is "terminal reward" — why do we score the agent at the end of the meet, not after each event?
- What is an action mask and why does the agent need one?

**Explore:**
- [ ] Read `src/env/swim_meet_env.py` once I have it done — trace through `reset()` and `step()` line by line, add comments in plain English explaining what each section does
- [ ] Run random rollouts and record the score:
  ```python
  import numpy as np
  obs, _ = env.reset()
  done = False
  while not done:
      mask = env.action_masks()
      action = np.random.choice(np.where(mask)[0])
      obs, reward, terminated, truncated, info = env.step(action)
      done = terminated or truncated
  print(f"Random policy total points: {reward}")
  ```
- [ ] Run the OR-Tools solver on the same roster and compare its score to the random rollout — which wins and by how much?
- [ ] Watch a training run: `python -m src.training.train` — observe the reward climbing over episodes. Write a note in your own words explaining why it increases.
- [ ] Run the trained RL policy and compare its score against OR-Tools. Which wins? Are the lineups different?
- [ ] Add to your EDA notebook: plot the reward curve from training (reward vs. episode). What does the shape tell you about how the agent is learning?

---

## Phase 7 — GUI (`src/gui/app.py`)

> **Blocked on me finishing training.** You need a saved `models/policy.zip` to load the RL policy.

You own the interface coaches will actually use. Use **Streamlit** — it's pure Python,
no web dev experience needed.

**Learn first:**
- Run the Streamlit hello-world: `streamlit hello`
- Read the docs on: `st.title`, `st.file_uploader`, `st.dataframe`, `st.button`, `st.selectbox`
- Understand what the optimizer outputs: `dict[(age_group, gender, event_id), list[str]]` — a mapping of each event to the assigned swimmer

**Build:**
- [ ] Sidebar: upload a roster CSV — feed it through `DataPipeline`, show swimmer count on success
- [ ] Sidebar: choose solver — OR-Tools (fast, deterministic) or RL policy (from `models/policy.zip`)
- [ ] Main panel: show the uploaded roster as a table grouped by age group
- [ ] "Generate Lineup" button — runs the selected solver and displays results
- [ ] Results table: age group, gender, event, assigned swimmer, their best time for that event
- [ ] Highlight events where no eligible swimmer was found so the coach knows what to address
- [ ] Stretch goal: for each assignment, show whether our swimmer's time is faster or slower than the mean opponent time — a quick win/loss prediction per event

**Run it:**
```bash
conda activate swim_rl
streamlit run src/gui/app.py
```

---

## Reference

- **Python:** `conda activate swim_rl`
- **Run from project root:** `/Users/raycast/anaconda3/envs/swim_rl/bin/python`
- **Key reading:** `data/league_rules.json`, `TODO_ray.md` (so you understand what I'm building on top of your work)
