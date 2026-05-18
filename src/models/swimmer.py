from dataclasses import dataclass, field


@dataclass
class Swimmer:
    name: str
    age: int
    gender: str  # 'M' or 'F'
    best_times: dict[str, float] = field(default_factory=dict)  # event -> seconds
    age_group: str = field(init=False)
    current_events_count: int = 0
    individual_events_count: int = 0
    relay_events_count: int = 0
    fatigue_coefficient: float = 1.04

    def __post_init__(self):
        self.age_group = self._age_to_group(self.age)

    @staticmethod
    def _age_to_group(age: int) -> str:
        if age <= 6:
            return "6U"
        if age <= 8:
            return "7-8"
        if age <= 10:
            return "9-10"
        if age <= 12:
            return "11-12"
        if age <= 14:
            return "13-14"
        return "15-18"

    def calculate_decayed_time(self, event_id: str, events_since_last: int) -> float:
        base_time = self.best_times.get(event_id)
        if base_time is None:
            raise ValueError(f"{self.name} has no time for event '{event_id}'")
        if events_since_last == 0:
            return base_time * self.fatigue_coefficient
        return base_time

    def is_eligible(self, is_relay: bool, max_individual: int = 2, max_relay: int = 2, max_total: int = 4) -> bool:
        if self.current_events_count >= max_total:
            return False
        if is_relay and self.relay_events_count >= max_relay:
            return False
        if not is_relay and self.individual_events_count >= max_individual:
            return False
        return True
