# OR-Tools CP-SAT deterministic baseline solver
# solve(swimmers, schedule, rules) -> dict[(age_group, gender, event_id), list[str]]
# Constraints: per-swimmer event limits, age_group + gender eligibility, time must exist
# Objective: maximize sum of inverted times (lower time = higher score proxy)
# Events with no eligible swimmer are left unassigned (slot constraint is <=, not ==)
