"""Public plan graphs and explicitly constructed derivatives; no model calls."""
import collections
import hashlib
import json
import math
import random
import re
from pathlib import Path

from .offline_data import topological_order, topology_fingerprint
from .structured_synthetic import seed_of

DIGITAL_SOURCES = {'toolbench', 'toolalpaca', 'lumos', 'environment/os', 'os'}


def parse_plan(text):
    """Read only final reference answer. START/END are virtual, not compute."""
    nodes = {}
    for line in text.splitlines():
        m = re.match(r'^\s*(\d+)\s*[:.]\s*(.+)$', line)
        if m:
            key = int(m[1])
            if key in nodes:
                raise ValueError('duplicate_node_id')
            nodes[key] = m[2].strip()
    if not nodes:
        raise ValueError('missing_nodes')
    ids = sorted(nodes)
    index = {key: i for i, key in enumerate(ids)}
    edges, virtual = set(), []
    pairs = re.findall(r'\(\s*(START|END|\d+)\s*,\s*(START|END|\d+)\s*\)', text, re.I)
    if not pairs:
        raise ValueError('missing_edges')
    for left, right in pairs:
        left, right = left.upper(), right.upper()
        if any(v not in {'START', 'END'} and int(v) not in index for v in (left, right)):
            raise ValueError('unknown_edge_endpoint')
        if left in {'START', 'END'} or right in {'START', 'END'}:
            if left == 'END' or right == 'START':
                raise ValueError('invalid_virtual_direction')
            virtual.append([left, right])
        else:
            edges.add((index[int(left)], index[int(right)]))
    edges = sorted(edges)
    topological_order(len(ids), edges)
    if len(edges) > 4 * len(ids):
        raise ValueError('above_train_edge_bound')
    return dict(n=len(ids), edges=[list(e) for e in edges],
                node_descriptions=[nodes[key] for key in ids], source_node_ids=ids,
                virtual_edges=virtual)


def graph_stats(n, edges):
    order = topological_order(n, edges)
    parents, children = [[] for _ in range(n)], [[] for _ in range(n)]
    for u, v in edges:
        parents[v].append(u); children[u].append(v)
    depth = [0] * n
    for v in order:
        depth[v] = 1 + max((depth[u] for u in parents[v]), default=0)
    levels = collections.Counter(depth)
    return dict(n=n, edges=len(edges), edges_per_node=len(edges)/n,
                depth=max(depth), max_level_width=max(levels.values()),
                unit_duration_critical_path_fraction=max(depth)/n,
                roots=sum(not p for p in parents), sinks=sum(not c for c in children),
                max_indegree=max(map(len, parents)), max_outdegree=max(map(len, children)))


def load_public_plans(raw_dir):
    raw_dir = Path(raw_dir)
    files = [(raw_dir/'worfbench_hf_train/worfbench_train.json', None)]
    files += [(p, p.parent.name) for p in sorted((raw_dir/'worfbench/gold_traj').glob('*/graph_eval.json'))]
    plans, audit = [], collections.Counter()
    for path, source_hint in files:
        for index, row in enumerate(json.loads(path.read_text())):
            source = row.get('source', source_hint)
            messages = row.get('messages', row.get('conversations'))
            try:
                graph = parse_plan(messages[-1]['content'])
            except ValueError as e:
                audit[f'rejected:{source}:{e}'] += 1
                continue
            question = next(m['content'] for m in reversed(messages[:-1]) if m['role'] == 'user')
            # Hash the actual final problem/answer, excluding few-shot examples.
            identity = hashlib.sha256(json.dumps([source, question, messages[-1]['content']],
                                                  ensure_ascii=False).encode()).hexdigest()
            graph.update(base_id='worf_'+identity[:20], source=source,
                         source_file=str(path.relative_to(raw_dir)), source_row_index=index,
                         source_record_id=row.get('id'), request=question,
                         content_sha256=identity, family='worfbench/'+source,
                         source_kind='public_reference_plan',
                         topology_fingerprint=topology_fingerprint(graph['n'], graph['edges']))
            plans.append(graph)
            audit[f'parsed:{source}'] += 1
    return plans, dict(audit)


def refine_plan(plan):
    """Construct 3 invocations per coarse subtask; retain ALL source edges."""
    n = plan['n'] * 3
    edges = [(3*v, 3*v+1) for v in range(plan['n'])]
    edges += [(3*v+1, 3*v+2) for v in range(plan['n'])]
    edges += [(3*u+2, 3*v) for u, v in plan['edges']]
    descriptions = [f'{stage}: {desc}' for desc in plan['node_descriptions']
                    for stage in ('Prepare inputs', 'Execute subtask', 'Validate and format output')]
    return dict(n=n, edges=[list(e) for e in sorted(edges)], node_descriptions=descriptions,
                base_id='refine3_'+plan['base_id'], parent_base_id=plan['base_id'],
                source=plan['source'], family='refine3/'+plan['source'],
                source_file=plan['source_file'], source_row_index=plan['source_row_index'],
                source_record_id=plan['source_record_id'], request=plan['request'],
                source_kind='constructed_three_stage_refinement',
                refinement='Three fixed stages per reference subtask, not observed executions',
                source_node_ids=plan['source_node_ids'],
                topology_fingerprint=topology_fingerprint(n, edges))


def application_templates():
    """Parrot-inspired fixed application graphs, not published execution traces."""
    out = []
    for files in (4, 8, 12, 16):
        labels = ['Architect: plan project files and interfaces']; edges = []
        def node(text):
            labels.append(text); return len(labels)-1
        codes = [node(f'Initial implementation of file {i}') for i in range(files)]
        edges += [(0, v) for v in codes]
        barrier = node('Integrate initial file implementations')
        edges += [(v, barrier) for v in codes]
        for round_id in range(3):
            revised = []
            for i in range(files):
                review = node(f'Review file {i}, round {round_id+1}')
                revision = node(f'Revise file {i}, round {round_id+1}')
                edges += [(barrier, review), (review, revision)]
                revised.append(revision)
            barrier = node(f'Integrate revisions, round {round_id+1}')
            edges += [(v, barrier) for v in revised]
        out.append(dict(n=len(labels), edges=[list(e) for e in sorted(edges)],
                        node_descriptions=labels, base_id=f'parrot_style_code_files{files}_rounds3',
                        source='parrot_paper_section8.4', family='template/multifile_coding',
                        source_kind='constructed_application_template', files=files, rounds=3,
                        source_url='https://www.usenix.org/system/files/osdi24-lin-chaofan.pdf',
                        note='Our invocation graph based on published pattern; not author trace/count'))
    for n in (20, 40, 80, 120):
        out.append(dict(n=n, edges=[[i, n-1] for i in range(n-1)],
                        node_descriptions=[f'Summarize document chunk {i}' for i in range(n-1)] +
                                          ['Synthesize all chunk summaries'],
                        base_id=f'parrot_style_mapreduce_n{n}', source='parrot_code',
                        family='template/document_mapreduce', source_kind='constructed_application_template',
                        source_file='parrot/examples/summarization_map_reduce.py',
                        note='Our parameterized structure from public code; no document executed'))
    for plan in out:
        plan['topology_fingerprint'] = topology_fingerprint(plan['n'], plan['edges'])
    return out


def physical_instances(base, config):
    """TRAIN-v1 physics recipe with public edges; no family-specific work boost.

    RNG order and formulas intentionally match structured_synthetic.instances.
    This function never changes the frozen generator or source graph.
    """
    n, k = base['n'], base['k']; edges = [list(e) for e in base['edges']]
    r = random.Random(seed_of(base['seed'], 'physics'))
    work = [round(min(10., max(.02, r.lognormvariate(-.4, .9))), 6) for _ in range(n)]
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
        inst = dict(n=n, edges=edges, workload=work, demands=demand, capacities=caps,
                    speeds=speeds, cost_rates=rates, bandwidth=bw, data_mb=data,
                    latency=0, model='fixed', overlap=True, ticks_per_unit=1000,
                    budget=budget, bmin=bmin, bfast=bfast, budget_mode=mode,
                    latent_id=base['base_id'], family=base['family'], resource_mode='pool',
                    target_ccr=base['ccr'], seed=seed_of(base['seed'], j) % (2**32),
                    source='agent-workflow-public-and-constructed-v1',
                    units='reference seconds / millisecond ticks / simulated integer fee units')
        yield f"val_{base['base_id']}_b{j}", inst, dict(eta=None if eta == 'nonbinding' else eta,
                  maximum=bmax, actual_ccr=actual_ccr, price_gamma=gamma)


def to_public_instance(row):
    """Existing physics adapter with accurate public/constructed provenance."""
    from .structured_dataset import to_instance
    inst = to_instance(row)
    inst.update(source=row['meta']['source'], panel=row['meta']['panel'],
                source_kind=row['meta']['source_kind'], public_plan_id=row['meta']['public_plan_id'])
    return inst
