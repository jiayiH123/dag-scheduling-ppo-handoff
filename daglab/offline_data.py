"""Dependency-free sparse physical records for the complete offline datasets.

Large graphs stay sparse on disk. The existing research simulator uses a dense
matrix; materialize only an explicitly bounded training/development subset.
"""
from collections import deque
from copy import deepcopy
import hashlib
import json
import math

SCHEMA = 'daglab-sparse-physical-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def topological_order(n, edges):
    children = [[] for _ in range(n)]
    incoming = [0] * n
    seen = set()
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v or (u, v) in seen:
            raise ValueError(('invalid/duplicate edge', u, v))
        seen.add((u, v)); children[u].append(v); incoming[v] += 1
    ready = deque(v for v in range(n) if incoming[v] == 0)
    order = []
    while ready:
        u = ready.popleft(); order.append(u)
        for v in children[u]:
            incoming[v] -= 1
            if incoming[v] == 0:
                ready.append(v)
    if len(order) != n:
        raise ValueError('cyclic source graph')
    return order


def topology_fingerprint(n, edges):
    # Same directed WL16 fingerprint as the prior public pilot, ID invariant.
    h = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    parents, children = [[] for _ in range(n)], [[] for _ in range(n)]
    for u, v in edges:
        parents[v].append(u); children[u].append(v)
    colors = [h([len(parents[v]), len(children[v])]) for v in range(n)]
    for _ in range(16):
        colors = [h([colors[v], sorted(colors[u] for u in parents[v]),
                     sorted(colors[w] for w in children[v])]) for v in range(n)]
    return h([n, sorted(colors), sorted((colors[u], colors[v]) for u, v in edges)])


def workflow_base(raw, source_path, source_sha256):
    spec = raw['workflow']['specification']
    tasks = spec['tasks']
    measured = {t['id']: t for t in raw['workflow']['execution']['tasks']}
    files = {f['id']: f['sizeInBytes'] for f in spec['files']}
    ids = [t['id'] for t in tasks]
    index = {name: v for v, name in enumerate(ids)}
    if not tasks or len(index) != len(tasks) or any(name not in measured for name in ids):
        raise ValueError('incomplete/duplicate execution records')
    edges = sorted((u, index[name]) for u, t in enumerate(tasks) for name in t['children'])
    edge_data = []
    for u, v in edges:
        shared = set(tasks[u]['outputFiles']) & set(tasks[v]['inputFiles'])
        size = sum(files[name] for name in shared)
        if size < 0 or not math.isfinite(size):
            raise ValueError('invalid measured file bytes')
        edge_data.append(size / 1e6)
    runtimes = [measured[name]['runtimeInSeconds'] for name in ids]
    if any(not math.isfinite(t) or t < 0 for t in runtimes):
        raise ValueError('invalid measured runtime')
    n = len(tasks)
    topological_order(n, edges)
    return {'base_id': 'wf_' + digest([source_path, source_sha256])[:20],
            'dataset': 'wfinstances', 'n': n, 'edges': edges, 'edge_data_mb': edge_data,
            'workload': [4 * max(1, math.ceil(t * 1000 / 4)) for t in runtimes],
            'demands': [1] * n, 'pool_capacity': 2, 'source_path': source_path,
            'source_sha256': source_sha256, 'source_task_ids': ids,
            'source_runtime_seconds': runtimes,
            'family': '/'.join(source_path.split('/')[:-1]),
            'topology_fingerprint': topology_fingerprint(n, edges),
            'source_kind': 'measured_execution' if not source_path.startswith('helloworld/') else 'synthetic_example',
            'communication_origin': 'measured shared input/output file sizes',
            'resource_origin': 'simulated unit slot demands and capacity2, not measured memory allocation',
            'duration_origin': 'measured seconds rounded up to4ms; relative device speeds simulated',
            'source_commit': 'd7bfead08a4dbf985ebbf427b3685a6512c259e2'}


def simulated_tpch_traffic(base):
    # One stable missing-traffic assumption per edge, shared across all variants.
    mean_fast_ticks = sum(w / 4 for w in base['workload']) / base['n']
    mean_slow_bandwidth_mb_per_tick = (.01 + .02 + .04) / 3
    base['edge_data_mb'] = [mean_fast_ticks * mean_slow_bandwidth_mb_per_tick *
                            (.5 + int(digest([base['base_id'], u, v])[:8], 16) / 0xffffffff)
                            for u, v in base['edges']]
    base['communication_origin'] = 'simulated edge traffic; no measured TPC-H shuffle bytes supplied'


def physical_variants(base):
    for pools in [3, 6]:
        speeds, rates = [1, 2, 4] * (pools // 3), [1, 4, 16] * (pools // 3)
        durations = [[math.ceil(w / speed) for speed in speeds] for w in base['workload']]
        costs = [[duration * rate for duration, rate in zip(row, rates)] for row in durations]
        bmin = sum(min(row) for row in costs)
        bfast = sum(row[min(range(pools), key=lambda p: (durations[v][p], p))] for v, row in enumerate(costs))
        for profile, factor in [('10_40_MBps', 1), ('100_400_MBps', 10)]:
            bandwidth3 = [[0, .01 * factor, .02 * factor], [.01 * factor, 0, .04 * factor],
                          [.02 * factor, .04 * factor, 0]]
            bandwidth = [[0 if p == q else bandwidth3[p % 3][q % 3] or bandwidth3[0][1]
                          for q in range(pools)] for p in range(pools)]
            for eta in [.25, .6]:
                key = f"{base['split']}_{base['base_id']}_{pools}_{profile}_{eta}"
                inst = {k: base[k] for k in ['n', 'edges', 'workload', 'demands', 'family']}
                inst.update(model='fixed', overlap=True, ticks_per_unit=1, latency=0,
                            capacities=[base['pool_capacity']] * pools, speeds=speeds,
                            cost_rates=rates, bandwidth=bandwidth,
                            budget=bmin + int(eta * (bfast - bmin)), bmin=bmin, bfast=bfast,
                            data_mb_sparse=[[u, v, value] for (u, v), value in zip(base['edges'], base['edge_data_mb'])],
                            id=key, source=base['source_path'], seed=int(digest(key)[:8], 16),
                            latent_id=base['base_id'], topology_fingerprint=base['topology_fingerprint'],
                            budget_eta=eta, budget_mode='tight25' if eta == .25 else 'tight60',
                            budget_degenerate=bmin == bfast, target_ccr=None, resource_mode='pool',
                            speed_ratio=4, link_profile=profile, workflow_family=base['family'],
                            split_role=base['split'] + '_offline_full',
                            previously_inspected=base.get('previously_inspected', False),
                            dataset_kind=base['source_kind'] + '_simulated_platform',
                            platform_calibrated=False, source_commit=base['source_commit'],
                            units='milliseconds / assumed TPC-H duration units; simulation cost units',
                            communication_origin=base['communication_origin'],
                            note='Sparse physical instance; full source and assumptions in offline manifest. No task truncation.')
                yield {'schema': SCHEMA, 'key': key, 'instance_sparse': inst,
                       'provenance': {k: base[k] for k in ['base_id', 'dataset', 'source_path', 'source_sha256',
                                                         'source_kind', 'communication_origin', 'resource_origin', 'duration_origin']}}


def validate_sparse(record):
    inst = record['instance_sparse']
    n, k = inst['n'], len(inst['speeds'])
    topological_order(n, inst['edges'])
    if len(inst['workload']) != n or len(inst['demands']) != n:
        raise ValueError('wrong task arrays')
    if any(type(w) is not int or w < 1 for w in inst['workload']):
        raise ValueError('invalid task work')
    if any(type(q) is not int or q < 1 or not any(q <= c for c in inst['capacities']) for q in inst['demands']):
        raise ValueError('invalid/ineligible task demand')
    if len(inst['data_mb_sparse']) != len(inst['edges']):
        raise ValueError('edge data missing')
    for edge, (u, v, value) in zip(inst['edges'], inst['data_mb_sparse']):
        if list(edge) != [u, v] or value < 0 or not math.isfinite(value):
            raise ValueError('invalid edge traffic')
    if len(inst['bandwidth']) != k or any(len(row) != k for row in inst['bandwidth']):
        raise ValueError('wrong bandwidth dimensions')
    for p in range(k):
        for q in range(k):
            if (p == q and inst['bandwidth'][p][q] != 0) or (p != q and inst['bandwidth'][p][q] <= 0):
                raise ValueError('invalid bandwidth')
    # A topological serial all-pool0 schedule is an explicit feasible witness:
    # same-pool transfer=0, each task fits, cost is the minimum-cost reserve.
    if any(q > inst['capacities'][0] for q in inst['demands']):
        raise ValueError('serial witness needs eligible pool0')
    cost0 = sum(math.ceil(w / inst['speeds'][0]) * inst['cost_rates'][0] for w in inst['workload'])
    bmin = sum(min(math.ceil(w / inst['speeds'][p]) * inst['cost_rates'][p]
                   for p in range(k) if demand <= inst['capacities'][p])
               for w, demand in zip(inst['workload'], inst['demands']))
    if cost0 != bmin or inst['bmin'] != bmin or not bmin <= inst['budget'] <= inst['bfast']:
        raise ValueError('invalid hard budget / feasible witness')
    return True


def dense_instance(record, max_tasks=512):
    inst = record['instance_sparse']
    if inst['n'] > max_tasks:
        raise ValueError(f"{inst['n']}tasks exceed explicit dense materialization limit{max_tasks}; sparse source is complete")
    result = deepcopy({k: v for k, v in inst.items() if k != 'data_mb_sparse'})
    data = [[0.] * inst['n'] for _ in range(inst['n'])]
    for u, v, value in inst['data_mb_sparse']:
        data[u][v] = value
    result['data_mb'] = data
    return result
