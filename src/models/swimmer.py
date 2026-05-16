from dataclasses import dataclass, field


@dataclass
class Swimmer:
    name: str
    best_times: dict[str, float] = field(default_factory=dict)  # event -> seconds
    current_events_count: int = 0
    individual_events_count: int = 0
    relay_events_count: int = 0
    fatigue_coefficient: float = 1.04

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
