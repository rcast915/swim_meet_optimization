# Naomi's TODO — Swim Meet Optimizer

You own the **data and models layer** — the foundation everything else is built on.
My RL environment and solver can't work without the objects you build here.

Before writing any code, read
`data/league_rules.json`, `data/raw/roster.csv`, and `data/raw/opponent_times.csv`.
You should be able to answer these questions before phase 1:

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
print(s.age_group)        # should be "9-10"
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
  - `opponent_projections: dict[tuple[str, str, str], list[float]]` — keyed by `(age_group, gender, event_id)`
  - Implement `__len__` and `__getitem__` so you can do `schedule[0]` and `len(schedule)`
- [ ] Implement `sample_opponent_time(event, rng) -> float`
  - Build the key from `(event.age_group, event.gender, event.event_id)`
  - Randomly pick one time from the list using `rng.choice()`
  - Return `float("inf")` if no times exist for that key (means we automatically win)

**Think about:** Why do we randomly sample one opponent time instead of using the average?
(Hint: in a real meet, you don't know who the opponent will enter until the heat sheet.)

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
- [ ] Implement `load_opponent_times(path) -> dict[tuple, list[float]]`
  - Read `opponent_times.csv`
  - Group `time_seconds` values into a dict keyed by `(age_group, gender, event)`
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
for s in swimmers:
    print(s.name, s.age_group, s.gender, list(s.best_times.keys()))
```

---

## Phase 4 — Roster (`src/models/roster.py`)

This is where eligibility checking happens at the roster level.

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
# After building swimmers and a MeetEvent, verify the mask only shows eligible swimmers
from src.models.roster import Roster
from src.models.meet_schedule import MeetEvent
roster = Roster(swimmers)
event = MeetEvent("25_freestyle", "9-10", "M", False, 1, [6,4,3,2,1,0])
mask = roster.action_mask(event)
print([swimmers[i].name for i in range(len(swimmers)) if mask[i]])
# Should only print 9-10 year old boys
```

---

## Phase 5 — Sync with Ray

Once your phases are done, I need:
1. `Swimmer` — fully working with `age_group`, `is_eligible()`, `calculate_decayed_time()`
2. `MeetEvent` + `MeetSchedule` — including `sample_opponent_time()`
3. `Roster` — with working `action_mask()` and `assign()`
4. `DataPipeline` — `build_roster()` and `load_opponent_times()` working end-to-end

Run the full pipeline test above and confirm it prints sensible output before handing off.

---

## Reference

- **Python:** `conda activate swim_rl`
- **Run from project root:** `/Users/raycast/anaconda3/envs/swim_rl/bin/python`
- **Key reading:** `data/league_rules.json`, `TODO_ray.md` (so you understand what I'm building on top of your work)
