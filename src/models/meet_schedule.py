from dataclasses import dataclass, field


@dataclass
class MeetEvent:
    event_id: str
    age_group: str   # e.g. "9-10", "11-12"
    gender: str      # 'M' or 'F'
    is_relay: bool
    slots: int       # 1 for individual, 4 for relay
    scoring_table: list[int]


@dataclass
class MeetSchedule:
    events: list[MeetEvent]
    # keyed by (age_group, gender, event_id) -> list of historical opponent times
    opponent_projections: dict[tuple[str, str, str], list[float]] = field(default_factory=dict)

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx: int) -> MeetEvent:
        return self.events[idx]

    def sample_opponent_time(self, event: MeetEvent, rng) -> float:
        key = (event.age_group, event.gender, event.event_id)
        times = self.opponent_projections.get(key, [])
        if not times:
            return float("inf")
        return float(rng.choice(times))
