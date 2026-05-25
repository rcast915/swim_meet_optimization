# MeetEvent dataclass: event_id, age_group, gender, is_relay, slots, scoring_table
# MeetSchedule dataclass:
#   events: list[MeetEvent]
#   opponent_projections: dict[(age_group, gender, event_id), (mean, std)]
#     — Gaussian parameters fitted from historical opponent times
# Methods:
#   sample_opponent_time(event, rng) -> float
#     samples rng.normal(mean, std), clamped to min 1.0
#     returns float("inf") if no projection exists for that key
