"""Pinned author TPC-H records, read once for the offline handoff build.

No author training code is executed. Only a small numeric NumPy pickle whitelist
is accepted; offline training reads the resulting ordinary JSON, not pickle.
"""
from collections import Counter
import io
import math
from pathlib import Path
import pickle
import random
import numpy as np
from .offline_data import digest, simulated_tpch_traffic, topological_order, topology_fingerprint

COMMIT = 'dbb9a1a2613d8e306fa726089e2c294a29f9f133'


class NumericUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ['numpy.core.multiarray', 'numpy._core.multiarray'] and name in ['_reconstruct', 'scalar']:
            return getattr(np._core.multiarray, name)
        if module == 'numpy' and name in ['ndarray', 'dtype']:
            return getattr(np, name)
        raise pickle.UnpicklingError(f'unsupported source pickle class {module}.{name}')


def numeric_npy(path):
    blob = Path(path).read_bytes()
    stream = io.BytesIO(blob)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(('unhandled NPY format', version))
    if dtype.hasobject:
        value = NumericUnpickler(stream).load()
        if shape != () or value.shape != ():
            raise ValueError('expected scalar dictionary source')
        return value.item()
    return np.load(io.BytesIO(blob), allow_pickle=False)


def stage_values(stage):
    first = stage['first_wave']
    setting = next(iter(first))
    demand = len(first[setting]) + len(stage['rest_wave'][setting])
    clean, previous = {}, []
    for e in first:
        fresh = Counter(stage['fresh_durations'][e])
        clean[e] = []
        for duration in first[e]:
            if fresh[duration]:
                fresh[duration] -= 1
            else:
                clean[e].append(duration)
    for e in sorted(clean):
        if not clean[e]:
            clean[e] = previous
        previous = clean[e]
    values = [x for bank in [clean, stage['rest_wave'], stage['fresh_durations']]
              for row in bank.values() for x in row]
    if not values or demand < 1 or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('invalid TPC-H task samples')
    return sum(values) / len(values), demand


def query_bases(root, source_hashes, roles):
    root = Path(root)
    result = []
    for adjacency in sorted((root / 'dag_data/tpch').glob('*/adj_mat_*.npy')):
        query = int(adjacency.stem.rsplit('_', 1)[1])
        scale = adjacency.parent.name
        duration_file = adjacency.parent / f'task_duration_{query}.npy'
        matrix, duration = numeric_npy(adjacency), numeric_npy(duration_file)
        n = len(matrix)
        if matrix.shape != (n, n) or set(duration) != set(range(n)):
            raise ValueError('TPC-H node/sample mismatch')
        # Historical PPO-BiHyb import convention, retained for frozen v1/v3 replay.
        # Decima uses i->j on these SAME matrices. New experiments must use the
        # verified forward adapter in performance_data.prepare_inputs (v4).
        edges = sorted((int(j), int(i)) for i, j in np.argwhere(matrix == 1))
        topological_order(n, edges)
        values = [stage_values(duration[v]) for v in range(n)]
        source = f'dag_data/tpch/{scale}/query_{query}'
        base = {'base_id': f'tpch_{scale}_q{query}', 'dataset': 'tpch_primitive',
                'n': n, 'edges': edges, 'workload': [4 * max(1, math.ceil(t / 4)) for t, _ in values],
                'demands': [d for _, d in values], 'pool_capacity': 600,
                'source_path': source, 'source_sha256': digest([source_hashes[str(p.relative_to(root))] for p in [adjacency, duration_file]]),
                'source_commit': COMMIT, 'source_kind': 'measured_query_stage_samples',
                'family': 'tpch/query' + str(query), 'query_id': query, 'input_scale': scale,
                'source_mean_stage_durations': [t for t, _ in values],
                'topology_fingerprint': topology_fingerprint(n, edges),
                'split': roles[query], 'previously_inspected': True,
                'resource_origin': 'author first-setting stage task counts; assumed pool capacity600',
                'duration_origin': 'author cleaned pooled mean duration samples, interpreted as ms, rounded up4; simulated relative speeds',
                'benchmark_role': 'primitive source; not a strict unseen-topology holdout'}
        simulated_tpch_traffic(base)
        result.append(base)
    if len(result) != 154:
        raise ValueError(('expected all154query/scale records', len(result)))
    return result


def make_compositions(primitives, roles, counts=None):
    counts = counts or {'train': 800, 'val': 100, 'test': 100}
    by_role = {s: [b for b in primitives if roles[b['query_id']] == s] for s in counts}
    used, fingerprints, result = set(), {}, []
    for role, count in counts.items():
        rng = random.Random('20261019:' + role)
        for i in range(count):
            size = [5, 10, 20][i % 3]
            for attempt in range(10000):
                selected = [rng.choice(by_role[role]) for _ in range(size)]
                signature = digest([b['base_id'] for b in selected])
                if signature in used:
                    continue
                edges, work, demands, components, offset = [], [], [], [], 0
                for b in selected:
                    edges.extend((u + offset, v + offset) for u, v in b['edges'])
                    work.extend(b['workload']); demands.extend(b['demands'])
                    components.append({'base_id': b['base_id'], 'query_id': b['query_id'],
                                       'input_scale': b['input_scale'], 'offset': offset, 'n': b['n']})
                    offset += b['n']
                fp = topology_fingerprint(offset, edges)
                if fp in fingerprints and fingerprints[fp] != role:
                    continue
                break
            else:
                raise RuntimeError('cannot build requested disjoint composition groups')
            used.add(signature); fingerprints[fp] = role
            base = {'base_id': f'tpchmix_{role}_{i:04}_{signature[:12]}', 'dataset': 'tpch_composite',
                    'n': offset, 'edges': sorted(edges), 'workload': work, 'demands': demands,
                    'pool_capacity': 600, 'source_path': 'generated-query-composition/' + signature,
                    'source_sha256': signature, 'source_commit': COMMIT,
                    'source_kind': 'generated_disjoint_union_of_measured_query_DAGs',
                    'family': 'tpch_composite', 'components': components, 'num_queries': size,
                    'topology_fingerprint': fp, 'split': role, 'previously_inspected': False,
                    'resource_origin': 'author stage task counts; assumed capacity600 perpool',
                    'duration_origin': 'author stage duration aggregation; generated composition, not measured whole-workload runtime'}
            simulated_tpch_traffic(base)
            result.append(base)
    return result


def query_roles():
    ids = list(range(1, 23)); random.Random(20261019).shuffle(ids)
    return {q: 'train' if i < 14 else 'val' if i < 18 else 'test' for i, q in enumerate(ids)}
