"""Model-independent structured data and numeric-token view (no Torch/network)."""
from pathlib import Path
import gzip
import hashlib
import json
import math

SCHEMA = 'daglab-structured-scheduling-v1'
TOKEN_TYPES = ['global', 'task', 'pool', 'edge', 'link', 'assignment', 'query']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def iter_rows(root, split=None, limit=None):
    root = Path(root)
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    count = 0
    for item in manifest['shards']:
        if split is not None and item['split'] != split:
            continue
        with gzip.open(root / item['path'], 'rt') as f:
            for line in f:
                yield json.loads(line)
                count += 1
                if limit and count >= limit:
                    return


def to_instance(row):
    x = row['input']; t, p, net, b = x['tasks'], x['pools'], x['network'], x['budget']
    n = x['n']; data = [[0.] * n for _ in range(n)]
    for u, v, value in net['edges']:
        data[u][v] = value
    return dict(n=n, workload=t['workload_seconds'], demands=t['demand'],
                speeds=p['speed'], capacities=p['capacity'], cost_rates=p['cost_rate'],
                edges=[[u, v] for u, v, _ in net['edges']], data_mb=data,
                bandwidth=net['bandwidth_MBps'], latency=net['latency_seconds'],
                ticks_per_unit=x['physics']['ticks_per_second'], model='fixed', overlap=True,
                budget=b['limit'], bmin=b['minimum'], bfast=b['fast'],
                budget_mode=b['mode'], budget_eta=b['eta'],
                target_ccr=row['meta']['target_ccr'], resource_mode='pool',
                family=row['meta']['family'], source='synthetic-structured-v1',
                latent_id=row['base_id'], seed=row['meta']['instance_seed'],
                topology_fingerprint=row['meta']['topology_fingerprint'],
                units='reference seconds, millisecond ticks and simulated integer fee units')


def numeric_view(model_input):
    """One token per typed row. Complete assignment prefix BEFORE all queries.

    Payloads have type-specific widths for separate small numeric projectors.
    ID pairs are also provided as integer indices, not only normalized floats.
    No target, teacher trajectory or baseline result is accepted by this API.
    """
    x = model_input; n, k = x['n'], x['k']
    t, p, net, b, norm = x['tasks'], x['pools'], x['network'], x['budget'], x['normalization']
    rows, indices, types = [], [], []
    def add(kind, ids, values):
        if any(not math.isfinite(v) for v in values):
            raise ValueError('nonfinite numeric input')
        types.append(TOKEN_TYPES.index(kind)); indices.append(ids); rows.append(values)
    ts, cs = norm['time_ticks'], norm['cost_per_task']
    add('global', [-1, -1], [n / 200, k / 8, b['limit'] / max(1, b['minimum']),
         (b['limit'] - b['minimum']) / max(1, b['fast'] - b['minimum']),
         b['fast'] / max(1, b['minimum']), float(b['mode'] == 'nonbinding')])
    parents, children = [0] * n, [0] * n
    for u, v, _ in net['edges']:
        children[u] += 1; parents[v] += 1
    max_rank = max(t['heft_rank'])
    for v in range(n):
        add('task', [v, -1], [t['workload_seconds'][v] / norm['workload_seconds'],
            t['demand'][v] / max(p['capacity']), parents[v] / n, children[v] / n,
            t['heft_rank'][v] / max_rank])
    for q in range(k):
        add('pool', [-1, q], [p['speed'][q] / max(p['speed']),
            p['capacity'][q] / max(p['capacity']), p['cost_rate'][q] / max(p['cost_rate'])])
    inv_bw = sum(1 / net['bandwidth_MBps'][i][j] for i in range(k) for j in range(k) if i != j) / (k * (k - 1))
    for u, v, data in net['edges']:
        add('edge', [u, v], [data / norm['edge_MB'],
            data * inv_bw * x['physics']['ticks_per_second'] / ts])
    max_bw = max(max(a) for a in net['bandwidth_MBps'])
    for i in range(k):
        for j in range(k):
            add('link', [i, j], [net['bandwidth_MBps'][i][j] / max_bw, float(i == j)])
    for v in range(n):
        for q in range(k):
            add('assignment', [v, q], [t['duration_ticks'][v][q] / ts,
                t['cost'][v][q] / cs, t['demand'][v] / p['capacity'][q],
                float(q in t['eligible_pools'][v])])
    prefix_length = len(rows)
    query_positions = []
    for v in range(n):
        for q in range(k):
            query_positions.append(len(rows))
            add('query', [v, q], [])
    return dict(types=types, ids=indices, payloads=rows, prefix_length=prefix_length,
                query_positions=query_positions, n=n, k=k,
                output_shape=[n, k, 2], position_count=len(rows))


def validate_record(row):
    if row['schema'] != SCHEMA:
        raise ValueError('unknown schema')
    x = row['input']; n, k = x['n'], x['k']; t, p = x['tasks'], x['pools']
    if any(len(t[a]) != n for a in ['workload_seconds', 'demand', 'duration_ticks', 'cost', 'eligible_pools', 'heft_rank']):
        raise AssertionError('task arrays')
    if any(len(p[a]) != k for a in ['speed', 'capacity', 'cost_rate']):
        raise AssertionError('pool arrays')
    if any(type(v) is not int or v < 1 for v in t['demand'] + p['capacity'] + p['cost_rate']):
        raise AssertionError('positive integer physics')
    order = x['heft_order']; pos = {v: i for i, v in enumerate(order)}
    if len(order) != n or set(order) != set(range(n)):
        raise AssertionError('order permutation')
    seen = set()
    for u, v, data in x['network']['edges']:
        if not (0 <= u < n and 0 <= v < n and pos[u] < pos[v]) or (u, v) in seen or data < 0:
            raise AssertionError('invalid edge')
        seen.add((u, v))
    for v in range(n):
        eligible = [q for q in range(k) if t['demand'][v] <= p['capacity'][q]]
        if not eligible or eligible != t['eligible_pools'][v]:
            raise AssertionError('eligibility')
        for q in range(k):
            dur = max(1, math.ceil(t['workload_seconds'][v] / p['speed'][q] * 1000 - 1e-10))
            if t['duration_ticks'][v][q] != dur or t['cost'][v][q] != dur * p['cost_rate'][q]:
                raise AssertionError('duration/cost arithmetic')
    b = x['budget']; minimum = sum(min(t['cost'][v][q] for q in t['eligible_pools'][v]) for v in range(n))
    maximum = sum(max(t['cost'][v][q] for q in t['eligible_pools'][v]) for v in range(n))
    if b['minimum'] != minimum or b['maximum'] != maximum or not minimum <= b['limit'] <= maximum:
        raise AssertionError('budget')
    if (row['split'] == 'train') != ('supervision' in row):
        raise AssertionError('TRAIN-only target contract')
    view = numeric_view(x)
    if view['position_count'] > 4096 or len(view['query_positions']) != n * k:
        raise AssertionError('numeric position contract')
    if 'supervision' in row:
        sup = row['supervision']; mapping = sup['task_to_pool']
        if len(mapping) != n or len(sup['decisions']) != n:
            raise AssertionError('target dimensions')
        spent = 0; reserve = minimum
        for v, d in zip(order, sup['decisions']):
            reserve -= min(t['cost'][v][q] for q in t['eligible_pools'][v])
            available = [v*k+q for q in t['eligible_pools'][v] if spent+t['cost'][v][q]+reserve <= b['limit']]
            if d['available'] != available or d['action'] != v*k+mapping[v] or d['action'] != available[d['selected']]:
                raise AssertionError('teacher mask/target')
            if d['spent_before'] != spent or d['future_min_reserve'] != reserve:
                raise AssertionError('teacher budget prefix')
            if not all(-1-1e-10 <= f <= 1e-10 for f in d['offsets']):
                raise AssertionError('normalized EFT range')
            # +/-7 is expressible by cap8 heads and wins every feasible teacher action.
            scores = [7. if a == d['action'] else -7. for a in available]
            best = max(range(len(scores)), key=lambda z: scores[z]+d['offsets'][z])
            if best != d['selected']:
                raise AssertionError('bounded-head oracle target')
            spent += t['cost'][v][mapping[v]]
        if spent != sup['cost'] or spent > b['limit'] or max(sup['ends']) != sup['makespan']:
            raise AssertionError('target objective')
    return view['position_count']
