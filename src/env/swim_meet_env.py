import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.models.roster import Roster
from src.models.meet_schedule import MeetSchedule


class SwimMeetEnv(gym.Env):
    """
    Sequential Decision Process over a full swim meet.

    Each step assigns one swimmer to the current event. Episodes terminate
    when all events have been assigned. Reward is the total simulated meet
    points, returned only at termination (terminal reward formulation).

    Action masking (via action_masks()) is required — use with MaskablePPO
    from sb3-contrib to prevent illegal assignments.
    """

    metadata = {"render_modes": []}

    def __init__(self, roster: Roster, schedule: MeetSchedule, seed: int | None = None):
        super().__init__()
        self.roster = roster
        self.schedule = schedule
        self.rng = np.random.default_rng(seed)

        n = len(roster.swimmers)
        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Dict({
            "roster_status": spaces.Box(low=0, high=4, shape=(n,), dtype=np.int8),
            "current_event": spaces.Discrete(len(schedule)),
        })

        self.current_event_idx = 0
        self.assignments: list[tuple[int, str]] = []  # (swimmer_idx, event_id)
        self._total_points = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.roster.reset()
        self.current_event_idx = 0
        self.assignments = []
        self._total_points = 0.0
        return self._obs(), {}

    def step(self, action: int):
        event = self.schedule[self.current_event_idx]

        if not self.action_masks()[action]:
            reward = -100.0
            return self._obs(), reward, True, False, {"illegal_action": True}

        events_since = self.roster.events_since_last(action, self.current_event_idx)
        swimmer = self.roster.swimmers[action]
        swimmer_time = swimmer.calculate_decayed_time(event.event_id, events_since)
        opponent_time = self.schedule.sample_opponent_time(event, self.rng)

        points = self._score_event(swimmer_time, opponent_time, event.scoring_table)
        self._total_points += points

        self.roster.assign(action, event.event_id, self.current_event_idx, event.is_relay)
        self.assignments.append((action, event.event_id))
        self.current_event_idx += 1

        terminated = self.current_event_idx >= len(self.schedule)
        reward = self._total_points if terminated else 0.0
        return self._obs(), reward, terminated, False, {}

    def action_masks(self) -> np.ndarray:
        event = self.schedule[self.current_event_idx]
        return self.roster.action_mask(event)

    def _obs(self) -> dict:
        return {
            "roster_status": self.roster.to_status_array(),
            "current_event": np.int64(min(self.current_event_idx, len(self.schedule) - 1)),
        }

    @staticmethod
    def _score_event(our_time: float, opponent_time: float, scoring_table: list[int]) -> float:
        if our_time < opponent_time:
            return float(scoring_table[0]) if scoring_table else 0.0
        return float(scoring_table[1]) if len(scoring_table) > 1 else 0.0
