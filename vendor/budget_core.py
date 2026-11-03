"""Fixed-delay DAG scheduling, cumulative capacity and additive hard budgets.

The immutable legacy simulator is reused for insertion and event-sweep checks.
New algorithms all use the same minimum-cost reserve mask. Cost is an integer
number of simulation cost units per execution tick; no real hardware is implied.
"""
from copy import deepcopy
import math
import time

from ortools.sat.python import cp_model
import legacy_comm_core as legacy


def prepare(instance):
    if instance["model"] != "fixed" or not instance["overlap"]:
        raise ValueError("This experiment only supports fixed, overlapping communication")
    info = legacy.prepare(instance)
    rates = instance.get("cost_rates", [0] * info["k"])
    if len(rates) != info["k"] or any(type(x) is not int or x < 0 for x in rates):
        raise ValueError("cost_rates must be nonnegative integer costs per tick")
    costs = [[info["duration"][v][p] * rates[p] for p in range(info["k"])]
             for v in range(info["n"])]
    minimum = [min(costs[v][p] for p in info["eligible"][v]) for v in range(info["n"])]
    budget = instance.get("budget")
    if budget is not None and (type(budget) is not int or budget < sum(minimum)):
        raise ValueError("budget is invalid or below the minimum feasible assignment cost")
    info.update(costs=costs, min_cost=minimum, budget=budget)
    return info


def reserve_allows(info, state, v, p):
    if info["budget"] is None:
        return True
    used = sum(info["costs"][u][t["device"]] for u, t in state["tasks"].items())
    reserve = sum(info["min_cost"][u] for u in range(info["n"])
                  if u not in state["tasks"] and u != v)
    return used + info["costs"][v][p] + reserve <= info["budget"]


def validate(instance, solution):
    legacy.validate(instance, solution)
    info = prepare(instance)
    cost = sum(info["costs"][t["node"]][t["device"]] for t in solution["tasks"])
    if info["budget"] is not None and cost > info["budget"]:
        raise AssertionError(("budget", cost, info["budget"]))
    if "cost" in solution and solution["cost"] != cost:
        raise AssertionError("incorrect cost ledger")
    return True


def optimistic_cost_table(info):
    """PEFT recurrence using actual link-specific delay and feasible devices.

    This is an explicit extension for capacity eligibility and budget settings;
    the OCT itself remains optimistic and does not include future reservations.
    """
    octable = [[0.0] * info["k"] for _ in range(info["n"])]
    for v in reversed(info["order"]):
        for p in info["eligible"][v]:
            octable[v][p] = max((min(
                octable[c][q] + info["duration"][c][q] + info["comm"][v, c][p][q]
                for q in info["eligible"][c]) for c in info["children"][v]), default=0.0)
    return octable


def successor_score(instance, info, state, v, p, candidate):
    """One-level join lookahead, optimistic about unassigned parents.

    Known parent mappings and calendars are respected. Other unscheduled tasks
    contribute a no-communication earliest-finish lower bound. This is our
    simple heuristic control, not a reproduction of a named published method.
    """
    tasks = {**state["tasks"], v: candidate["task"]}
    lower = [0] * info["n"]
    for u in info["order"]:
        lower[u] = tasks[u]["end"] if u in tasks else (
            min(info["duration"][u][q] for q in info["eligible"][u])
            + max((lower[a] for a in info["parents"][u]), default=0))
    scores = [candidate["task"]["end"]]
    for c in info["children"][v]:
        choices = []
        for q in info["eligible"][c]:
            arrivals = []
            for u in info["parents"][c]:
                if u in tasks:
                    arrivals.append(tasks[u]["end"] + info["comm"][u, c][tasks[u]["device"]][q])
                else:
                    arrivals.append(lower[u] + min(info["comm"][u, c][a][q]
                                                   for a in info["eligible"][u]))
            start = legacy.find_slot(max(arrivals, default=0), info["duration"][c][q],
                                     instance["demands"][c], instance["capacities"][q],
                                     candidate["calendar"][q])
            choices.append(start + info["duration"][c][q])
        scores.append(min(choices))
    return max(scores)


def schedule(instance, algorithm="HEFT", assignment=None, order=None, logits=None, price=0.0):
    started = time.perf_counter()
    info = prepare(instance)
    if assignment is not None:
        if len(assignment) != info["n"] or any(p not in info["eligible"][v]
                                               for v, p in enumerate(assignment)):
            raise ValueError("infeasible fixed assignment")
        if info["budget"] is not None and sum(info["costs"][v][p] for v, p in enumerate(assignment)) > info["budget"]:
            raise ValueError("fixed assignment exceeds budget")
    octable = optimistic_cost_table(info) if algorithm == "PEFT" else None
    rank = [sum(octable[v][p] for p in info["eligible"][v]) / len(info["eligible"][v])
            for v in range(info["n"])] if octable else info["ranks"]
    state = legacy.blank_state(info["k"])
    while len(state["tasks"]) < info["n"]:
        ready = [v for v in range(info["n"]) if v not in state["tasks"]
                 and all(u in state["tasks"] for u in info["parents"][v])]
        if order is not None:
            v = order[len(state["tasks"])]
            if v not in ready:
                raise ValueError("supplied order violates dependencies")
            ready = [v]
        elif algorithm in ("HEFT", "PEFT", "JoinLookahead", "replay", "CostFirst"):
            ready = [min(ready, key=lambda v: (-rank[v], v))]
        elif algorithm not in ("Greedy", "FrozenPolicy"):
            raise ValueError(algorithm)
        candidates = []
        for v in ready:
            devices = [assignment[v]] if assignment is not None else info["eligible"][v]
            for p in devices:
                if not reserve_allows(info, state, v, p):
                    continue
                cand = legacy.placement(instance, info, state, v, p)
                eft = cand["task"]["end"]
                score = eft
                if algorithm == "PEFT":
                    score += octable[v][p]
                    if price:
                        scale = sum(min(row) for row in info["duration"]) / info["n"]
                        score += price * scale * (info["costs"][v][p] - info["min_cost"][v]) / max(1, info["min_cost"][v])
                elif algorithm == "CostFirst":
                    score = info["costs"][v][p]
                elif algorithm == "JoinLookahead":
                    score = successor_score(instance, info, state, v, p, cand)
                elif algorithm == "FrozenPolicy":
                    if logits is None:
                        raise ValueError("FrozenPolicy requires a precomputed score matrix")
                    score = -float(logits[v][p])
                candidates.append((score, eft, cand["task"]["start"], v, p, cand))
        if not candidates:
            raise AssertionError("minimum-cost reserve should leave a completion path")
        _, _, _, v, p, selected = min(candidates, key=lambda c: c[:5])
        state["tasks"][v] = selected["task"]
        state["calendar"] = selected["calendar"]
        state["communications"].extend(selected["communications"])
        state["decision_order"].append(v)
    result = legacy.finish_schedule(instance, state, algorithm, rank)
    if price:
        result["algorithm"] = f"{algorithm}-price{price:g}"
    result["cost"] = sum(info["costs"][t["node"]][t["device"]] for t in result["tasks"])
    validate(instance, result)
    result["runtime_seconds"] = time.perf_counter() - started
    return result


def single_device(instance):
    started = time.perf_counter()
    info = prepare(instance)
    candidates = []
    for p in range(info["k"]):
        if all(p in x for x in info["eligible"]):
            if info["budget"] is None or sum(info["costs"][v][p] for v in range(info["n"])) <= info["budget"]:
                candidates.append(schedule(instance, "HEFT", assignment=[p] * info["n"]))
    if not candidates:
        return None
    best = deepcopy(min(candidates, key=lambda s: s["makespan"]))
    best.update(algorithm="BestSingleDevice", candidate_count=len(candidates),
                runtime_seconds=time.perf_counter() - started)
    return best


def mapping_search(instance, seeds, passes=2, name="HEFT-LS"):
    started = time.perf_counter()
    info = prepare(instance)
    best = deepcopy(min(seeds, key=lambda s: s["makespan"]))
    initial = best["makespan"]
    order = best["decision_order"]
    evaluations = len(seeds)
    moves = []
    for _ in range(passes):
        assigned = [t["device"] for t in best["tasks"]]
        chosen = None
        trial_best = best
        for v in range(info["n"]):
            for p in info["eligible"][v]:
                if p == assigned[v]:
                    continue
                changed = assigned.copy()
                changed[v] = p
                if info["budget"] is not None and sum(info["costs"][u][q] for u, q in enumerate(changed)) > info["budget"]:
                    continue
                trial = schedule(instance, "replay", assignment=changed, order=order)
                evaluations += 1
                if trial["makespan"] < trial_best["makespan"]:
                    trial_best, chosen = trial, [v, p]
        if chosen is None:
            break
        moves.append({"node": chosen[0], "device": chosen[1], "before": best["makespan"],
                      "after": trial_best["makespan"]})
        best = trial_best
    best = deepcopy(best)
    best.update(algorithm=name, runtime_seconds=time.perf_counter() - started,
                local_search={"candidate_count": evaluations, "passes_limit": passes,
                              "initial_makespan": initial, "moves": moves,
                              "converged": len(moves) < passes})
    validate(instance, best)
    return best


def pair_exchange_search(instance, seed, passes=2):
    """Exhaust all feasible two-task remappings, preserving the decision order.

    Intended as a strong diagnostic baseline. The full candidate search and
    intervening single-move convergence are charged to runtime, not inference
    for a purported one-forward neural policy.
    """
    started = time.perf_counter()
    info = prepare(instance)
    best = deepcopy(seed)
    initial = best["makespan"]
    evaluated = 0
    pair_moves = []
    single_converged = []
    for _ in range(passes):
        refined = mapping_search(instance, [best], 20, "SingleMoveConverged")
        single_converged.append(refined["local_search"]["converged"])
        best = refined
        assigned = [t["device"] for t in best["tasks"]]
        order = best["decision_order"]
        cost = best["cost"]
        trial_best, chosen = best, None
        for u in range(info["n"]):
            for v in range(u + 1, info["n"]):
                for p in info["eligible"][u]:
                    if p == assigned[u]:
                        continue
                    for q in info["eligible"][v]:
                        if q == assigned[v]:
                            continue
                        changed_cost = cost - info["costs"][u][assigned[u]] - info["costs"][v][assigned[v]] + info["costs"][u][p] + info["costs"][v][q]
                        if info["budget"] is not None and changed_cost > info["budget"]:
                            continue
                        changed = assigned.copy()
                        changed[u], changed[v] = p, q
                        trial = schedule(instance, "replay", assignment=changed, order=order)
                        evaluated += 1
                        if trial["makespan"] < trial_best["makespan"]:
                            trial_best, chosen = trial, [u, p, v, q]
        if chosen is None:
            break
        pair_moves.append({"change": chosen, "before": best["makespan"], "after": trial_best["makespan"]})
        best = trial_best
    refined = mapping_search(instance, [best], 20, "SingleMoveConverged")
    single_converged.append(refined["local_search"]["converged"])
    best = deepcopy(refined)
    best.update(algorithm="Exchange-LS", runtime_seconds=time.perf_counter() - started,
                pair_search={"passes_limit": passes, "evaluated_pairs": evaluated,
                             "pair_moves": pair_moves, "single_convergence_checks": single_converged,
                             "initial_makespan": initial})
    validate(instance, best)
    return best


def exact_solve(instance, incumbent, seconds=1.0, fixed_assignment=None,
                reference_assignment=None, max_changes=None):
    """CP-SAT exact for this integer fixed-delay model, with budget included."""
    validate(instance, incumbent)
    info = prepare(instance)
    n, k, horizon = info["n"], info["k"], incumbent["makespan"]
    if reference_assignment is not None:
        if len(reference_assignment) != n or any(p not in info["eligible"][v] for v,p in enumerate(reference_assignment)):
            raise ValueError("invalid reference assignment")
        if type(max_changes) is not int or not 0 <= max_changes <= n:
            raise ValueError("max_changes must be an integer in [0,n]")
        if sum(t["device"] != reference_assignment[t["node"]] for t in incumbent["tasks"]) > max_changes:
            raise ValueError("incumbent must belong to the requested neighborhood")
    elif max_changes is not None:
        raise ValueError("max_changes requires reference_assignment")
    model = cp_model.CpModel()
    starts = [model.new_int_var(0, horizon, f"s{v}") for v in range(n)]
    ends = [model.new_int_var(0, horizon, f"e{v}") for v in range(n)]
    assigned = [[model.new_bool_var(f"x{v}_{p}") for p in range(k)] for v in range(n)]
    intervals = [[] for _ in range(k)]
    demands = [[] for _ in range(k)]
    for v in range(n):
        model.add_exactly_one(assigned[v])
        model.add(ends[v] == starts[v] + sum(info["duration"][v][p] * assigned[v][p] for p in range(k)))
        for p in range(k):
            if p not in info["eligible"][v]:
                model.add(assigned[v][p] == 0)
            interval = model.new_optional_interval_var(starts[v], info["duration"][v][p],
                                                       ends[v], assigned[v][p], f"task{v}_{p}")
            intervals[p].append(interval)
            demands[p].append(instance["demands"][v])
        if fixed_assignment is not None:
            model.add(assigned[v][fixed_assignment[v]] == 1)
    for p in range(k):
        model.add_cumulative(intervals[p], demands[p], instance["capacities"][p])
    for u, v in instance["edges"]:
        for p in range(k):
            for q in range(k):
                model.add(starts[v] >= ends[u] + info["comm"][u, v][p][q]).only_enforce_if(
                    [assigned[u][p], assigned[v][q]])
    if info["budget"] is not None:
        model.add(sum(info["costs"][v][p] * assigned[v][p] for v in range(n) for p in range(k))
                  <= info["budget"])
    if reference_assignment is not None:
        model.add(sum(1-assigned[v][reference_assignment[v]] for v in range(n)) <= max_changes)
    cmax = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(cmax, ends)
    model.minimize(cmax)
    for task in incumbent["tasks"]:
        v = task["node"]
        model.add_hint(starts[v], task["start"])
        model.add_hint(ends[v], task["end"])
        for p in range(k):
            model.add_hint(assigned[v][p], int(task["device"] == p))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 23
    status = solver.solve(model)
    result = {"status": solver.status_name(status), "proven_optimal": status == cp_model.OPTIMAL,
              "lower_bound": max(legacy.analytic_lower_bound(instance), math.ceil(solver.best_objective_bound - 1e-7)),
              "incumbent": None, "schedule": None, "wall_seconds": solver.wall_time,
              "time_limit_seconds": seconds, "workers": 1, "solver_seed": 23}
    if reference_assignment is not None:
        result["max_mapping_changes"] = max_changes
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        tasks = []
        for v in range(n):
            p = next(p for p in range(k) if solver.value(assigned[v][p]))
            tasks.append({"node": v, "device": p, "start": solver.value(starts[v]),
                          "end": solver.value(ends[v]), "duration": info["duration"][v][p],
                          "demand": instance["demands"][v]})
        records = []
        for u, v in instance["edges"]:
            p, q = tasks[u]["device"], tasks[v]["device"]
            start = tasks[u]["end"]
            length = info["comm"][u, v][p][q]
            records.append({"u": u, "v": v, "src": p, "dst": q, "start": start,
                            "end": start + length, "duration": length, "release": start,
                            "queue_wait": 0, "local": p == q, "data_mb": instance["data_mb"][u][v]})
        for task in tasks:
            task["data_ready"] = max((r["end"] for r in records if r["v"] == task["node"]), default=0)
            task["resource_wait"] = task["start"] - task["data_ready"]
        solution = {"algorithm": "CP-SAT", "makespan": solver.value(cmax), "tasks": tasks,
                    "communications": records, "decision_order": info["order"]}
        solution["cost"] = sum(info["costs"][t["node"]][t["device"]] for t in tasks)
        validate(instance, solution)
        result.update(incumbent=solution["makespan"], schedule=solution)
    if status in (cp_model.INFEASIBLE, cp_model.MODEL_INVALID):
        raise AssertionError("known feasible incumbent contradicted: " + solver.response_stats())
    return result
