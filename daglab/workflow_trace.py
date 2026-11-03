"""WfFormat trace adapter; original physical conversion with handoff metadata.

Ported from experiments/autoresearch/src/workflow_trace.py. The physics are
unchanged: recorded runtime and file bytes, assumed three-pool platform, fixed
communication lag with overlap, additive execution cost, and hard budget.
"""
import hashlib
import json
import math
from pathlib import Path

from .reference import core


def load_trace(path, source, profile, eta, family=None):
    path = Path(path)
    raw = json.loads(path.read_text())
    spec = raw["workflow"]["specification"]
    tasks = spec["tasks"]
    measured = {t["id"]: t for t in raw["workflow"]["execution"]["tasks"]}
    files = {f["id"]: f["sizeInBytes"] for f in spec["files"]}
    index = {t["id"]: i for i, t in enumerate(tasks)}
    n = len(tasks)
    if len(index) != n or any(t["id"] not in measured for t in tasks):
        raise ValueError("incomplete/duplicate execution records")
    edges = []
    data = [[0.] * n for _ in range(n)]
    for u, t in enumerate(tasks):
        for child in t["children"]:
            v = index[child]
            common = set(t["outputFiles"]) & set(tasks[v]["inputFiles"])
            edges.append([u, v])
            data[u][v] = sum(files[f] for f in common) / 1e6
    runtimes = [measured[t["id"]]["runtimeInSeconds"] for t in tasks]
    if any(x < 0 for x in runtimes):
        raise ValueError("negative runtime")
    workload = [4 * max(1, math.ceil(x * 1000 / 4)) for x in runtimes]
    factor = {"10_40_MBps": 1, "100_400_MBps": 10}[profile]
    base = {
        "model": "fixed", "overlap": True, "ticks_per_unit": 1,
        "n": n, "edges": edges, "workload": workload,
        "demands": [1] * n, "capacities": [2, 2, 2],
        "speeds": [1, 2, 4], "cost_rates": [1, 4, 16],
        "bandwidth": [[0, .01 * factor, .02 * factor],
                      [.01 * factor, 0, .04 * factor],
                      [.02 * factor, .04 * factor, 0]],
        "data_mb": data, "latency": 0, "budget": None,
        "source": source, "family": source, "link_profile": profile,
        "budget_eta": eta,
        "units": "milliseconds; simulated platform and cost units",
        "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "zero_runtime_tasks": sum(x == 0 for x in runtimes),
        "source_task_ids": [t["id"] for t in tasks],
        "source_runtime_seconds": runtimes, "platform_calibrated": False,
        "note": "Complete measured-trace graph, runtimes and file bytes; uniform relative speeds, unit slot demands, capacities, rates and links are assumptions."
    }
    info = core.prepare(base)
    bmin = sum(info["min_cost"])
    fastest = [min(info["eligible"][v], key=lambda p: (info["duration"][v][p], p))
               for v in range(n)]
    bfast = sum(info["costs"][v][p] for v, p in enumerate(fastest))
    base.update(budget=bmin + int(eta * (bfast - bmin)), bmin=bmin, bfast=bfast)
    # Preserve the original adapter's instance ID before adding bookkeeping.
    base["id"] = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()[:16]
    base.update(
        seed=int(base["trace_sha256"][:8], 16),
        latent_id="wfinstances:" + base["trace_sha256"][:20],
        target_ccr=None, resource_mode="pool", speed_ratio=4,
        budget_mode={.25: "tight25", .6: "tight60", 1.: "nonbinding"}.get(eta, "eta" + str(eta)),
        budget_degenerate=bmin == bfast,
        workflow_family=family or source,
        split_role="known_public_smoke",
        previously_inspected=True,
        dataset_kind="measured_workflow_simulated_platform",
    )
    # Canonical family groups the two Epigenomics scales; no physics is changed.
    if family is not None:
        base["family"] = family
    return base
