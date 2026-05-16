import numpy as np
from src.models.swimmer import Swimmer


class Roster:
    def __init__(self, swimmers: list[Swimmer]):
        self.swimmers = swimmers
        self._last_event_idx: dict[str, int] = {s.name: -999 for s in swimmers}

    def get_eligible_swimmers(self, is_relay: bool) -> list[int]:
        return [i for i, s in enumerate(self.swimmers) if s.is_eligible(is_relay)]

    def action_mask(self, is_relay: bool) -> np.ndarray:
        mask = np.zeros(len(self.swimmers), dtype=bool)
        for i in self.get_eligible_swimmers(is_relay):
            mask[i] = True
        return mask

    def assign(self, swimmer_idx: int, event_id: str, event_order: int, is_relay: bool):
        swimmer = self.swimmers[swimmer_idx]
        swimmer.current_events_count += 1
        if is_relay:
            swimmer.relay_events_count += 1
        else:
            swimmer.individual_events_count += 1
        self._last_event_idx[swimmer.name] = event_order

    def events_since_last(self, swimmer_idx: int, current_event_order: int) -> int:
        name = self.swimmers[swimmer_idx].name
        return current_event_order - self._last_event_idx[name]

    def reset(self):
        for s in self.swimmers:
            s.current_events_count = 0
            s.individual_events_count = 0
            s.relay_events_count = 0
        self._last_event_idx = {s.name: -999 for s in self.swimmers}

    def to_status_array(self) -> np.ndarray:
        return np.array([s.current_events_count for s in self.swimmers], dtype=np.int8)
