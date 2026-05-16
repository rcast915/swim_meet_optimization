from dataclasses import dataclass, field


@dataclass
class MeetEvent:
    event_id: str
    is_relay: bool
    slots: int  # 1 for individual, 4 for relay
    scoring_table: list[int]  # points by finish position


@dataclass
class MeetSchedule:
    events: list[MeetEvent]
    opponent_projections: dict[str, list[float]] = field(default_factory=dict)  # event_id -> list of opponent times

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx: int) -> MeetEvent:
        return self.events[idx]

    def sample_opponent_time(self, event_id: str, rng) -> float:
        times = self.opponent_projections.get(event_id, [])
        if not times:
            return float("inf")
        return float(rng.choice(times))
