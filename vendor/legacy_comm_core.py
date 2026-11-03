"""Discrete-time heterogeneous DAG scheduling with independently checked schedules.

All times are integer ticks. Fixed communication has unlimited parallel links;
shared_bus reserves one global non-preemptive channel. Both can overlap CPU work
unless overlap=False, in which case a transfer blocks both endpoint devices.
"""
from copy import deepcopy
import math
import time

import numpy as np
from ortools.sat.python import cp_model


def topology(n, edges):
    parents, children = [[] for _ in range(n)], [[] for _ in range(n)]
    for u, v in edges:
        if u == v or not (0 <= u < n and 0 <= v < n) or v in children[u]:
            raise ValueError("invalid/duplicate edge")
        parents[v].append(u)
        children[u].append(v)
    degrees = [len(p) for p in parents]
    ready = sorted(v for v in range(n) if not degrees[v])
    order = []
    while ready:
        v = ready.pop(0)
        order.append(v)
        for c in children[v]:
            degrees[c] -= 1
            if not degrees[c]:
                ready.append(c)
                ready.sort()
    if len(order) != n:
        raise ValueError("cycle")
    return parents, children, order


def prepare(instance):
    n, k = len(instance["workload"]), len(instance["speeds"])
    parents, children, order = topology(n, instance["edges"])
    if n == 0 or k == 0 or any(x <= 0 for x in instance["workload"] + instance["speeds"]):
        raise ValueError("nonpositive workload/speed")
    capacities, demands = instance["capacities"], instance["demands"]
    if len(capacities) != k or len(demands) != n:
        raise ValueError("resource dimensions")
    if any(type(x) is not int or x <= 0 for x in capacities + demands):
        raise ValueError("capacities and demands must be positive integers")
    if instance["model"] not in ("fixed", "shared_bus"):
        raise ValueError("communication model")
    tick = instance["ticks_per_unit"]
    duration = [[max(1, math.ceil(w / f * tick - 1e-10)) for f in instance["speeds"]]
                for w in instance["workload"]]
    eligible = [[p for p in range(k) if demands[v] <= capacities[p]] for v in range(n)]
    if any(not e for e in eligible):
        raise ValueError("task without a feasible device")
    communication, average_comm = {}, {}
    bandwidth = np.asarray(instance["bandwidth"], dtype=float)
    if bandwidth.shape != (k, k) or (k > 1 and np.any(bandwidth[~np.eye(k, dtype=bool)] <= 0)):
        raise ValueError("invalid cross-device bandwidth")
    mean_bw = float(bandwidth[~np.eye(k, dtype=bool)].mean()) if k > 1 else 1.0
    latency = instance.get("latency", 0.0)
    for u, v in instance["edges"]:
        data = instance["data_mb"][u][v]
        if data < 0 or latency < 0:
            raise ValueError("negative communication parameter")
        communication[u, v] = [[0 if p == q else math.ceil(
            (latency + data / bandwidth[p, q]) * tick - 1e-10)
            for q in range(k)] for p in range(k)]
        # Original HEFT paper: data / average cross-processor bandwidth.
        average_comm[u, v] = (latency + data / mean_bw) * tick if k > 1 else 0.0
    ranks = [0.0] * n
    for v in reversed(order):
        mean_compute = float(np.mean([duration[v][p] for p in eligible[v]]))
        ranks[v] = mean_compute + max(
            (average_comm[v, c] + ranks[c] for c in children[v]), default=0.0)
    return {"n": n, "k": k, "parents": parents, "children": children, "order": order,
            "duration": duration, "eligible": eligible, "comm": communication, "ranks": ranks}


def fits(start, duration, demand, capacity, calendar):
    end = start + duration
    overlaps = [r for r in calendar if r["start"] < end and r["end"] > start]
    times = sorted({start, end} | {max(start, r["start"]) for r in overlaps}
                   | {min(end, r["end"]) for r in overlaps})
    return all(demand + sum(r["demand"] for r in overlaps
                           if r["start"] <= t < r["end"]) <= capacity
               for t in times[:-1])


def find_slot(release, duration, demand, capacity, calendar):
    if demand > capacity:
        raise ValueError("infeasible resource demand")
    candidates = sorted({release} | {r["end"] for r in calendar if r["end"] >= release})
    for start in candidates:
        if fits(start, duration, demand, capacity, calendar):
            return start
    raise AssertionError("last calendar end must be feasible")


def blank_state(k):
    return {"tasks": {}, "calendar": [[] for _ in range(k)], "bus": [],
            "communications": [], "decision_order": []}


def placement(instance, info, state, v, device):
    calendars = [list(c) for c in state["calendar"]]
    bus = list(state["bus"])
    communications = []
    arrivals = [0]
    parents = sorted(info["parents"][v], key=lambda u: (state["tasks"][u]["end"], u))
    for u in parents:
        parent = state["tasks"][u]
        src = parent["device"]
        length = info["comm"][u, v][src][device]
        release = parent["end"]
        start = release
        if length:
            while True:
                previous = start
                if instance["model"] == "shared_bus":
                    start = max(start, find_slot(start, length, 1, 1, bus))
                if not instance["overlap"]:
                    for p in (src, device):
                        cap = instance["capacities"][p]
                        start = max(start, find_slot(start, length, cap, cap, calendars[p]))
                if start == previous:
                    break
        end = start + length
        record = {"u": u, "v": v, "src": src, "dst": device, "start": start,
                  "end": end, "duration": length, "release": release,
                  "queue_wait": start - release, "local": src == device,
                  "data_mb": instance["data_mb"][u][v]}
        communications.append(record)
        if length and instance["model"] == "shared_bus":
            bus.append({"start": start, "end": end, "demand": 1})
        if length and not instance["overlap"]:
            for p in (src, device):
                calendars[p].append({"start": start, "end": end,
                                     "demand": instance["capacities"][p], "kind": "transfer"})
        arrivals.append(end)
    release = max(arrivals)
    start = find_slot(release, info["duration"][v][device], instance["demands"][v],
                      instance["capacities"][device], calendars[device])
    end = start + info["duration"][v][device]
    task = {"node": v, "device": device, "start": start, "end": end,
            "duration": end - start, "demand": instance["demands"][v],
            "data_ready": release, "resource_wait": start - release}
    calendars[device].append({**task, "kind": "compute"})
    return {"task": task, "calendar": calendars, "bus": bus, "communications": communications}


def finish_schedule(instance, state, algorithm, rank=None):
    tasks = [state["tasks"][v] for v in sorted(state["tasks"])]
    return {"algorithm": algorithm, "makespan": max(t["end"] for t in tasks),
            "tasks": tasks, "communications": state["communications"],
            "decision_order": state["decision_order"],
            "execution_order": {str(p): [t["node"] for t in sorted(tasks, key=lambda t: (t["start"], t["node"]))
                                        if t["device"] == p] for p in range(len(instance["speeds"]))},
            "ranks": rank}


def schedule(instance, algorithm="HEFT", seed=0, assignment=None, order=None):
    info = prepare(instance)
    state = blank_state(info["k"])
    rng = np.random.default_rng(seed)
    while len(state["tasks"]) < info["n"]:
        ready = [v for v in range(info["n"]) if v not in state["tasks"]
                 and all(u in state["tasks"] for u in info["parents"][v])]
        if order is not None:
            v = order[len(state["tasks"])]
            if v not in ready:
                raise ValueError("supplied decision order violates dependencies")
            ready = [v]
        elif algorithm in ("HEFT", "replay"):
            ready = [min(ready, key=lambda v: (-info["ranks"][v], v))]
        elif algorithm == "Random":
            ready = [int(rng.choice(ready))]
        elif algorithm != "Greedy":
            raise ValueError(algorithm)
        candidates = []
        for v in ready:
            devices = info["eligible"][v]
            if assignment is not None:
                devices = [assignment[v]]
                if devices[0] not in info["eligible"][v]:
                    raise ValueError("infeasible assignment")
            elif algorithm == "Random":
                devices = [int(rng.choice(devices))]
            for p in devices:
                cand = placement(instance, info, state, v, p)
                candidates.append((cand["task"]["end"], cand["task"]["start"], v, p, cand))
        _, _, v, _, selected = min(candidates, key=lambda x: x[:4])
        state["tasks"][v] = selected["task"]
        state["calendar"] = selected["calendar"]
        state["bus"] = selected["bus"]
        state["communications"].extend(selected["communications"])
        state["decision_order"].append(v)
    result = finish_schedule(instance, state, algorithm, info["ranks"] if algorithm == "HEFT" else None)
    validate(instance, result)
    return result


def enhanced_heft(instance, passes=2):
    """Monotone best-improvement remapping; includes feasible single-device seeds.

This is a budgeted local search baseline, not a new neural architecture. Replay
uses the original HEFT topological decision order and recalculates all transfers.
"""
    info = prepare(instance)
    best = schedule(instance, "HEFT")
    order = best["decision_order"]
    evaluated = 1
    improvements = []
    for p in range(info["k"]):
        if all(p in devices for devices in info["eligible"]):
            trial = schedule(instance, "replay", assignment=[p] * info["n"], order=order)
            evaluated += 1
            if trial["makespan"] < best["makespan"]:
                improvements.append({"move": "all_on_one_device", "device": p,
                                     "before": best["makespan"], "after": trial["makespan"]})
                best = trial
    for _ in range(passes):
        assignments = [t["device"] for t in best["tasks"]]
        candidate_best, chosen_move = best, None
        for v in range(info["n"]):
            for p in info["eligible"][v]:
                if p == assignments[v]:
                    continue
                changed = assignments.copy()
                changed[v] = p
                trial = schedule(instance, "replay", assignment=changed, order=order)
                evaluated += 1
                if trial["makespan"] < candidate_best["makespan"]:
                    candidate_best = trial
                    chosen_move = {"move": "one_task_remap", "node": v, "device": p,
                                   "before": best["makespan"], "after": trial["makespan"]}
        if chosen_move is None:
            break
        improvements.append(chosen_move)
        best = candidate_best
    best["algorithm"] = "HEFT-LS"
    best["local_search"] = {"candidate_count": evaluated, "passes_limit": passes,
                            "moves": improvements}
    validate(instance, best)
    return best


def analytic_lower_bound(instance):
    info = prepare(instance)
    shortest = [min(info["duration"][v][p] for p in info["eligible"][v]) for v in range(info["n"])]
    paths = [0] * info["n"]
    for v in info["order"]:
        paths[v] = shortest[v] + max((paths[u] for u in info["parents"][v]), default=0)
    area = sum(instance["demands"][v] * shortest[v] for v in range(info["n"]))
    return max(max(paths), math.ceil(area / sum(instance["capacities"])))


def exact_solve(instance, incumbent, seconds=1.0, fixed_assignment=None, fixed_order=None):
    info = prepare(instance)
    n, k = info["n"], info["k"]
    horizon = incumbent["makespan"]
    model = cp_model.CpModel()
    starts = [model.new_int_var(0, horizon, f"s{v}") for v in range(n)]
    ends = [model.new_int_var(0, horizon, f"e{v}") for v in range(n)]
    assigned = [[model.new_bool_var(f"x{v}_{p}") for p in range(k)] for v in range(n)]
    intervals, demands = [[] for _ in range(k)], [[] for _ in range(k)]
    for v in range(n):
        model.add_exactly_one(assigned[v])
        model.add(ends[v] == starts[v] + sum(info["duration"][v][p] * assigned[v][p] for p in range(k)))
        for p in range(k):
            if p not in info["eligible"][v]:
                model.add(assigned[v][p] == 0)
            itv = model.new_optional_interval_var(starts[v], info["duration"][v][p], ends[v], assigned[v][p], f"task{v}_{p}")
            intervals[p].append(itv)
            demands[p].append(instance["demands"][v])
        if fixed_assignment is not None:
            model.add(assigned[v][fixed_assignment[v]] == 1)
    if fixed_order is not None:
        # Restrict actual relative start order on each device under a fixed map.
        if fixed_assignment is None:
            raise ValueError("fixed_order requires fixed_assignment")
        for p in range(k):
            nodes = [v for v in fixed_order if fixed_assignment[v] == p]
            for u, v in zip(nodes, nodes[1:]):
                model.add(starts[u] <= starts[v])
    bus_intervals, transfer_vars = [], {}
    for u, v in instance["edges"]:
        for p in range(k):
            for q in range(k):
                chosen = [assigned[u][p], assigned[v][q]]
                length = info["comm"][u, v][p][q]
                if length == 0 or (instance["model"] == "fixed" and instance["overlap"]):
                    model.add(starts[v] >= ends[u] + length).only_enforce_if(chosen)
                    continue
                active = model.new_bool_var(f"transfer{u}_{v}_{p}_{q}")
                model.add_bool_and(chosen).only_enforce_if(active)
                model.add_bool_or([chosen[0].Not(), chosen[1].Not(), active])
                ts = model.new_int_var(0, horizon, f"ts{u}_{v}_{p}_{q}")
                te = model.new_int_var(0, horizon, f"te{u}_{v}_{p}_{q}")
                interval = model.new_optional_interval_var(ts, length, te, active, f"comm{u}_{v}_{p}_{q}")
                model.add(ts >= ends[u]).only_enforce_if(active)
                model.add(starts[v] >= te).only_enforce_if(active)
                if instance["model"] == "shared_bus":
                    bus_intervals.append(interval)
                if not instance["overlap"]:
                    for device in (p, q):
                        intervals[device].append(interval)
                        demands[device].append(instance["capacities"][device])
                transfer_vars[u, v, p, q] = ts, te
    for p in range(k):
        model.add_cumulative(intervals[p], demands[p], instance["capacities"][p])
    if bus_intervals:
        model.add_no_overlap(bus_intervals)
    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, ends)
    model.minimize(makespan)
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
    lower_bound = max(analytic_lower_bound(instance), math.ceil(solver.best_objective_bound - 1e-7))
    result = {"status": solver.status_name(status), "proven_optimal": status == cp_model.OPTIMAL,
              "lower_bound": lower_bound, "incumbent": None, "schedule": None,
              "wall_seconds": solver.wall_time, "time_limit_seconds": seconds,
              "workers": 1, "solver_seed": 23, "response_stats": solver.response_stats()}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        tasks = []
        for v in range(n):
            p = next(p for p in range(k) if solver.value(assigned[v][p]))
            start, end = solver.value(starts[v]), solver.value(ends[v])
            tasks.append({"node": v, "device": p, "start": start, "end": end,
                          "duration": end - start, "demand": instance["demands"][v]})
        records = []
        for u, v in instance["edges"]:
            p, q = tasks[u]["device"], tasks[v]["device"]
            duration = info["comm"][u, v][p][q]
            if (u, v, p, q) in transfer_vars:
                ts, te = transfer_vars[u, v, p, q]
                start, end = solver.value(ts), solver.value(te)
            else:
                start, end = tasks[u]["end"], tasks[u]["end"] + duration
            records.append({"u": u, "v": v, "src": p, "dst": q, "start": start,
                            "end": end, "duration": duration, "release": tasks[u]["end"],
                            "queue_wait": start - tasks[u]["end"], "local": p == q,
                            "data_mb": instance["data_mb"][u][v]})
        for task in tasks:
            task["data_ready"] = max((r["end"] for r in records if r["v"] == task["node"]), default=0)
            task["resource_wait"] = task["start"] - task["data_ready"]
        sol = {"algorithm": "CP-SAT", "makespan": solver.value(makespan), "tasks": tasks,
               "communications": records, "decision_order": info["order"],
               "execution_order": {str(p): [t["node"] for t in sorted(tasks, key=lambda t: (t["start"], t["node"]))
                                           if t["device"] == p] for p in range(k)}}
        validate(instance, sol)
        result.update(incumbent=sol["makespan"], schedule=sol)
    if status in (cp_model.MODEL_INVALID, cp_model.INFEASIBLE):
        raise AssertionError(f"known feasible incumbent but solver status {result['status']}: {solver.response_stats()}")
    return result


def validate(instance, solution):
    """Separate event-sweep validation; no reliance on list-scheduler calendars."""
    info = prepare(instance)
    tasks = {t["node"]: t for t in solution["tasks"]}
    assert len(tasks) == len(solution["tasks"]) == info["n"]
    assert set(tasks) == set(range(info["n"]))
    for v, t in tasks.items():
        assert t["device"] in info["eligible"][v]
        assert t["start"] >= 0 and t["end"] - t["start"] == info["duration"][v][t["device"]]
        assert type(t["start"]) is int and type(t["end"]) is int
        assert t["demand"] == instance["demands"][v]
    records = {(c["u"], c["v"]): c for c in solution["communications"]}
    assert len(records) == len(solution["communications"]) == len(instance["edges"])
    assert set(records) == {tuple(e) for e in instance["edges"]}
    for (u, v), r in records.items():
        p, q = tasks[u]["device"], tasks[v]["device"]
        assert (r["src"], r["dst"]) == (p, q)
        assert r["start"] >= tasks[u]["end"]
        assert r["end"] - r["start"] == info["comm"][u, v][p][q]
        assert tasks[v]["start"] >= r["end"]
    for p, capacity in enumerate(instance["capacities"]):
        events = []
        for t in tasks.values():
            if t["device"] == p:
                events.extend([(t["start"], t["demand"]), (t["end"], -t["demand"])])
        if not instance["overlap"]:
            for r in records.values():
                if r["duration"] and p in (r["src"], r["dst"]):
                    events.extend([(r["start"], capacity), (r["end"], -capacity)])
        occupancy = 0
        for timestamp in sorted({t for t, _ in events}):
            occupancy += sum(delta for t, delta in events if t == timestamp)
            assert 0 <= occupancy <= capacity, ("capacity", p, timestamp, occupancy)
    if instance["model"] == "shared_bus":
        transfers = sorted((r["start"], r["end"]) for r in records.values() if r["duration"])
        assert all(a[1] <= b[0] for a, b in zip(transfers, transfers[1:])), "network contention"
    assert solution["makespan"] == max(t["end"] for t in tasks.values())
    return True


def measured_run(function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    result["runtime_seconds"] = time.perf_counter() - start
    return result
