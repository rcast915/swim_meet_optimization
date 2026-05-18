from ortools.sat.python import cp_model

from src.models.swimmer import Swimmer
from src.models.meet_schedule import MeetSchedule


def solve(swimmers: list[Swimmer], schedule: MeetSchedule, rules: dict) -> dict[str, list[str]]:
    """
    Deterministic CP-SAT baseline. Finds the mathematically optimal assignment
    under static times (no fatigue, no opponent randomness).

    Returns: dict mapping event_id -> list of assigned swimmer names.
    """
    model = cp_model.CpModel()
    n_swimmers = len(swimmers)
    n_events = len(schedule)

    max_individual = rules.get("max_individual_events", 2)
    max_relay = rules.get("max_relay_events", 2)
    max_total = rules.get("max_total_events", 4)

    # x[i][e] = 1 if swimmer i is assigned to event e
    x = [[model.NewBoolVar(f"x_{i}_{e}") for e in range(n_events)] for i in range(n_swimmers)]

    # Each event must be filled (one swimmer for individual, four for relay)
    for e, event in enumerate(schedule.events):
        model.Add(sum(x[i][e] for i in range(n_swimmers)) == event.slots)

    # Per-swimmer event limits
    indiv_events = [e for e, ev in enumerate(schedule.events) if not ev.is_relay]
    relay_events = [e for e, ev in enumerate(schedule.events) if ev.is_relay]
    for i in range(n_swimmers):
        model.Add(sum(x[i][e] for e in range(n_events)) <= max_total)
        model.Add(sum(x[i][e] for e in indiv_events) <= max_individual)
        model.Add(sum(x[i][e] for e in relay_events) <= max_relay)

    # Eligibility: swimmer must match the event's age_group and gender, and have a time
    objective_terms = []
    for i, swimmer in enumerate(swimmers):
        for e, event in enumerate(schedule.events):
            wrong_bracket = (swimmer.age_group != event.age_group or swimmer.gender != event.gender)
            raw_time = swimmer.best_times.get(event.event_id)

            if wrong_bracket or raw_time is None:
                model.Add(x[i][e] == 0)
            else:
                inverted_score = int(10000 - raw_time * 100)
                objective_terms.append(inverted_score * x[i][e])

    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("OR-Tools could not find a feasible solution.")

    result: dict[str, list[str]] = {}
    for e, event in enumerate(schedule.events):
        result[event.event_id] = [
            swimmers[i].name for i in range(n_swimmers) if solver.Value(x[i][e])
        ]
    return result
