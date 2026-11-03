"""Deterministic new simulation and unchanged Teacher64 recipe with trace metadata."""
import hashlib
import math
import random
from .offline_data import topology_fingerprint, topological_order
from .structured_dataset import SCHEMA, validate_record


def seed_of(*args):
    return int.from_bytes(hashlib.sha256(repr(args).encode()).digest()[:8], 'little')


def topology(family, n, seed):
    r = random.Random(seed); edges = set()
    if family == 'layered':
        levels = r.randint(3, min(10, n//2)); groups = [[] for _ in range(levels)]
        for v in range(n):
            groups[min(levels-1, v*levels//n)].append(v)
        for z in range(1, levels):
            for v in groups[z]:
                for u in r.sample(groups[z-1], r.randint(1, min(3, len(groups[z-1])))):
                    edges.add((u, v))
                if z > 1 and r.random() < .5:
                    edges.add((r.choice(sum(groups[:z-1], [])), v))
    elif family == 'forkjoin':
        branches = r.randint(2, min(8, (n-2)//2)); cuts = sorted(r.sample(range(2, n-1), branches-1))
        groups = [list(range(a, b)) for a, b in zip([1]+cuts, cuts+[n-1])]
        for g in groups:
            edges.add((0, g[0])); edges.add((g[-1], n-1))
            edges.update(zip(g, g[1:]))
    elif family == 'aggregation':
        roots = r.randint(2, min(6, n//4))
        for v in range(roots, n):
            for u in r.sample(range(v), r.randint(1, min(3, v))):
                edges.add((u, v))
        join = r.randint(n//2, n-2)
        edges.update((u, join) for u in r.sample(range(join), min(join, r.randint(3, n//2))))
    elif family == 'chain_side':
        chain = [0] + sorted(r.sample(range(1, n-1), r.randint(n//3, 2*n//3)-2)) + [n-1]
        edges.update(zip(chain, chain[1:]))
        for v in set(range(n))-set(chain):
            edges.add((r.choice([u for u in chain if u < v]), v))
            edges.add((v, r.choice([u for u in chain if u > v])))
    elif family == 'cross':
        roots = r.randint(2, min(6, n//4))
        for v in range(roots, n):
            candidates = list(range(max(0, v-n//2), v))
            edges.update((u, v) for u in r.sample(candidates, r.randint(1, min(3, len(candidates)))))
    else:
        raise ValueError(family)
    for _ in range(n//2):
        u = r.randrange(n-1); v = r.randrange(u+1, n); edges.add((u, v))
    if len(edges) > 4*n:
        raise AssertionError('edge bound')
    # IDs are arbitrary; generation and teacher use this same relabeling.
    permutation = list(range(n)); r.shuffle(permutation)
    edges = sorted((permutation[u], permutation[v]) for u, v in edges)
    topological_order(n, edges)
    return edges


def balanced(options, count, seed):
    values = [options[i % len(options)] for i in range(count)]
    random.Random(seed).shuffle(values)
    return values


def base_index(config, progress=None):
    seen = set(); rows = []; rejected = 0
    for role, count in config['counts'].items():
        ns = config['ood_tasks'] if role == 'ood_size' else config['tasks']
        cells = [(n, family) for n in ns for family in config['families']]
        pools = config['ood_pools'] if role == 'ood_pools' else config['pools']
        factors = {name: balanced(values, count, seed_of(config['seed'], role, name)) for name, values in
                   [('k', pools), ('ccr', config['ccrs']), ('H', config['speed_ratios']),
                    ('pressure', config['pressure_modes']), ('budget_pair', config['budget_pairs'])]}
        for i in range(count):
            n, family = cells[i % len(cells)]; attempt = 0
            while True:
                seed = seed_of(config['seed'], role, i, attempt)
                edges = topology(family, n, seed); fp = topology_fingerprint(n, edges)
                if fp not in seen:
                    seen.add(fp); break
                attempt += 1; rejected += 1
                if attempt > 10000:
                    raise RuntimeError('topology generator exhausted; no performance-based resampling')
            row = dict(base_id=f'simv1_{role}_{i:05d}', split=role, index=i, n=n, family=family,
                       seed=seed, topology_fingerprint=fp, duplicate_attempts=attempt,
                       **{name: values[i] for name, values in factors.items()})
            rows.append(row)
            if progress and (i+1) % 500 == 0:
                progress(dict(indexed_role=role, bases=i+1, topology_duplicates_rejected=rejected))
    return rows


def instances(base, config):
    n, k = base['n'], base['k']; r = random.Random(seed_of(base['seed'], 'physics'))
    edges = topology(base['family'], n, base['seed'])
    work = [round(min(10., max(.02, r.lognormvariate(-.4, .9))), 6) for _ in range(n)]
    if base['family'] == 'aggregation':
        indegree = [0] * n
        for u, v in edges: indegree[v] += 1
        v = max(range(n), key=lambda v: indegree[v]); work[v] = min(10., 2*work[v])
    weights = {'light': [.7, .2, .08, .02], 'mixed': [.25]*4, 'heavy': [.05, .15, .3, .5]}[base['pressure']]
    demand = r.choices(config['demands'], weights, k=n)
    speeds = [1., float(base['H'])] + [math.exp(r.uniform(0, math.log(base['H']))) for _ in range(k-2)]
    r.shuffle(speeds); gamma = r.uniform(1.2, 2.)
    rates = [max(1, round(10*s**gamma*r.uniform(.9, 1.1))) for s in speeds]
    caps = r.choices(config['capacities'], k=k)
    bw = [[0.] * k for _ in range(k)]
    for p in range(k):
        for q in range(p+1, k):
            bw[p][q] = bw[q][p] = round(r.choice(config['bandwidth_MBps'])*r.uniform(.8, 1.2), 6)
    duration = [[max(1, math.ceil(w/s*1000-1e-10)) for s in speeds] for w in work]
    cost = [[d*rate for d, rate in zip(row, rates)] for row in duration]
    bmin = sum(min(row) for row in cost); bmax = sum(max(row) for row in cost)
    fastest = [min(range(k), key=lambda p: (duration[v][p], cost[v][p], p)) for v in range(n)]
    bfast = sum(cost[v][p] for v, p in enumerate(fastest))
    mean_compute = sum(sum(row) for row in duration)/(n*k)
    inverse = sum(1/bw[p][q] for p in range(k) for q in range(k) if p != q)/(k*(k-1))
    factors = [min(4., max(.25, r.lognormvariate(0, .5))) for _ in edges]
    unit = base['ccr']*(mean_compute/1000)/inverse
    avgfactor = sum(factors)/len(factors)
    sizes = [unit*f/avgfactor for f in factors]
    actual_ccr = sum(math.ceil(d/bw[p][q]*1000-1e-10) for d in sizes for p in range(k)
                     for q in range(k) if p != q)/(len(edges)*k*(k-1)*mean_compute)
    data = [[0.] * n for _ in range(n)]
    for (u, v), size in zip(edges, sizes): data[u][v] = size
    for j, eta in enumerate(base['budget_pair']):
        mode = 'nonbinding' if eta == 'nonbinding' else 'eta'+str(eta)
        budget = bmax if eta == 'nonbinding' else bmin+int(eta*(bfast-bmin))
        key = f"{base['split']}_{base['base_id']}_b{j}"
        inst = dict(n=n, edges=[list(e) for e in edges], workload=work, demands=demand,
                    capacities=caps, speeds=speeds, cost_rates=rates, bandwidth=bw, data_mb=data,
                    latency=0, model='fixed', overlap=True, ticks_per_unit=1000,
                    budget=budget, bmin=bmin, bfast=bfast, budget_mode=mode,
                    latent_id=base['base_id'], family=base['family'], resource_mode='pool',
                    target_ccr=base['ccr'], seed=seed_of(base['seed'], j) % (2**32),
                    source='synthetic-structured-v1', units='reference seconds / millisecond ticks / simulated integer fee units')
        yield key, inst, dict(eta=None if eta == 'nonbinding' else eta, maximum=bmax,
                            actual_ccr=actual_ccr, price_gamma=gamma)


def teacher_with_quality(inst, decoder, names):
    """Same ten seeds, RNG and64 moves as daglab.reference.teacher; extra provenance."""
    from .reference import portfolio, mapping_of
    engine = decoder.engine; seeds, _ = portfolio(inst)
    mappings = [mapping_of(s) for s in seeds]
    values = [engine.evaluate(m, decoder.order) for m in mappings]
    best = min(range(len(values)), key=lambda i: values[i]); mapping = mappings[best]; value = values[best]
    rng = random.Random(inst['seed']+3917); feasible = 0
    for _ in range(64):
        changed = list(mapping)
        for v in rng.sample(range(engine.n), rng.choice([1, 2])):
            changed[v] = rng.choice(engine.info['eligible'][v])
        if not engine.feasible(changed): continue
        feasible += 1; candidate = engine.evaluate(changed, decoder.order)
        if candidate < value: mapping, value = changed, candidate
    solution = engine.evaluate(mapping, decoder.order, materialize=True)
    quality = dict(rule_names=names, native_makespans=[s['makespan'] for s in seeds],
                   fixed_order_makespans=[v[0] for v in values], fixed_order_costs=[v[1] for v in values],
                   initial_rule_index=best, local_attempts=64, feasible_local_proposals=feasible,
                   mapping_evaluations=engine.evaluations, optimality='not certified')
    return solution, mapping, quality


def teacher_trace(inst, decoder, mapping, solution):
    from .reference import core
    from prepared_sgs import find_slot
    info = decoder.info; n, k = decoder.n, decoder.k
    forced = decoder.rollout([[0., 0.]]*(n*k), 'local', forced_mapping=mapping, materialize=True)
    if forced['schedule']['tasks'] != solution['tasks']:
        raise AssertionError('teacher physical replay')
    ends = [0]*n; calendars = [[] for _ in range(k)]; spent = 0; reserve = sum(info['min_cost'])
    for v, d in zip(decoder.order, forced['decisions']):
        reserve -= info['min_cost'][v]; candidate_starts = []; candidate_ends = []
        for action in d['available']:
            q = action % k
            release = max((ends[u]+info['comm'][u, v][mapping[u]][q] for u in info['parents'][v]), default=0)
            start = find_slot(release, info['duration'][v][q], inst['demands'][v], inst['capacities'][q], calendars[q])
            candidate_starts.append(start); candidate_ends.append(start+info['duration'][v][q])
        d.update(spent_before=spent, future_min_reserve=reserve,
                 candidate_start_ticks=candidate_starts, candidate_end_ticks=candidate_ends)
        q = mapping[v]; selected = d['selected']; start = candidate_starts[selected]; end = candidate_ends[selected]
        calendars[q].append((start, end, inst['demands'][v])); ends[v] = end; spent += info['costs'][v][q]
    core.validate(inst, forced['schedule'])
    return dict(task_to_pool=mapping, decisions=forced['decisions'], makespan=solution['makespan'],
                cost=solution['cost'], starts=[s['start'] for s in solution['tasks']],
                ends=[s['end'] for s in solution['tasks']],
                target='conditional feasible-pool action likelihood, not a/w regression',
                decoder='NormalizedEFTDecoder/local/fixed-HEFT', teacher='ten-rule portfolio plus64local proposals')


def build_base_job(args):
    import time
    from .reference import core
    from .module_lab.decoder import NormalizedEFTDecoder
    base, config = args; started = time.perf_counter(); records = []
    for key, inst, extra in instances(base, config):
        info = core.prepare(inst); n, k = info['n'], info['k']
        order = sorted(range(n), key=lambda v: (-info['ranks'][v], v))
        normal = dict(time_ticks=sum(min(a) for a in info['duration'])/n,
                      cost_per_task=sum(info['min_cost'])/n,
                      workload_seconds=sum(inst['workload'])/n,
                      edge_MB=max(1e-12, sum(inst['data_mb'][u][v] for u, v in inst['edges'])/len(inst['edges'])))
        x = dict(n=n, k=k, physics=dict(nonpreemptive=True, fixed_delay=True, overlap=True,
                    link_contention=False, same_pool_zero=True, ticks_per_second=1000,
                    fee='duration_ticks*pool_cost_rate'),
                 tasks=dict(workload_seconds=inst['workload'], demand=inst['demands'], duration_ticks=info['duration'],
                            cost=info['costs'], eligible_pools=info['eligible'], heft_rank=info['ranks']),
                 pools=dict(speed=inst['speeds'], capacity=inst['capacities'], cost_rate=inst['cost_rates']),
                 network=dict(edges=[[u, v, inst['data_mb'][u][v]] for u, v in inst['edges']],
                              bandwidth_MBps=inst['bandwidth'], latency_seconds=0),
                 budget=dict(limit=inst['budget'], minimum=inst['bmin'], fast=inst['bfast'],
                             maximum=extra['maximum'], eta=extra['eta'], mode=inst['budget_mode']),
                 heft_order=order, normalization=normal)
        row = dict(schema=SCHEMA, key=key, base_id=base['base_id'], split=base['split'], input=x,
                   meta=dict(source='new synthetic simulation, not measured workflow', family=base['family'],
                             instance_seed=inst['seed'], topology_seed=base['seed'],
                             topology_fingerprint=base['topology_fingerprint'],
                             target_ccr=base['ccr'], actual_ccr=extra['actual_ccr'],
                             speed_ratio=base['H'], demand_pressure=base['pressure'], price_gamma=extra['price_gamma']))
        if base['split'] == 'train':
            d = NormalizedEFTDecoder(inst); solution, mapping, quality = teacher_with_quality(inst, d, config['teacher_rules'])
            row['supervision'] = teacher_trace(inst, d, mapping, solution); row['quality'] = quality
        elif base['split'] == 'val':
            row['quality'] = dict(reference_HEFT=core.schedule(inst, 'HEFT')['makespan'])
        validate_record(row); records.append(row)
    return records, time.perf_counter()-started
