"""Independent synthetic motifs and immutable focused-data overlays."""
import json
import os
import random
import sqlite3
from .sft_three_panel import ROOT, Store, connect
from .offline_data import topological_order

CONFIG = ROOT/os.environ.get('DAGLAB_FOCUSED_CONFIG','configs/focused_augmentation_20261101.json')


def config():
    return json.loads(CONFIG.read_text())


def partition(total, parts, rng):
    values = [1]*parts
    for _ in range(total-parts):
        values[rng.randrange(parts)] += 1
    rng.shuffle(values)
    return values


def topology(family, n, seed):
    r = random.Random(seed)
    edges = set()
    if family == 'iterative_branch_review':
        stages = r.randint(2, min(6, (n-1)//4))
        stage_nodes = partition(n-stages-1, stages, r)
        cursor, barrier = 1, 0
        for count in stage_nodes:
            branches = r.randint(max(1, (count+2)//3), count)
            ends = []
            for length in partition(count, branches, r):
                nodes = list(range(cursor, cursor+length)); cursor += length
                edges.add((barrier, nodes[0])); edges.update(zip(nodes, nodes[1:])); ends.append(nodes[-1])
            joined = cursor; cursor += 1
            edges.update((v, joined) for v in ends); barrier = joined
        assert cursor == n
    elif family == 'wide_scatter_gather':
        roots, sinks = r.randint(1, 2), r.randint(1, 2)
        middle = n-roots-sinks
        branches = r.randint(max(2, (middle+2)//3), middle)
        cursor = roots
        for length in partition(middle, branches, r):
            nodes = list(range(cursor, cursor+length)); cursor += length
            edges.update((u, nodes[0]) for u in range(roots))
            edges.update(zip(nodes, nodes[1:]))
            edges.update((nodes[-1], v) for v in range(n-sinks, n))
        # Sparse optional cross-branch dependencies vary shapes without
        # destroying the deliberately shallow, wide regime.
        for _ in range(r.randint(0, n//12)):
            u = r.randrange(roots, n-sinks-1); v = r.randrange(u+1, n-sinks)
            edges.add((u, v))
    elif family == 'scatter_pipeline_join':
        prefix = r.randint(max(3, n//10), min(32, n//3))
        suffix = r.randint(1, max(2, n//12))
        middle = n-prefix-suffix
        branches = r.randint(2, min(24, middle//2))
        edges.update(zip(range(prefix-1), range(1, prefix)))
        cursor = prefix
        ends = []
        for length in partition(middle, branches, r):
            nodes = list(range(cursor, cursor+length)); cursor += length
            edges.add((r.randrange(max(0, prefix-4), prefix), nodes[0]))
            edges.update(zip(nodes, nodes[1:])); ends.append(nodes[-1])
        edges.update((v, cursor) for v in ends)
        edges.update(zip(range(cursor, n-1), range(cursor+1, n)))
    else:
        raise ValueError(family)
    assert edges and len(edges) <= 4*n
    permutation = list(range(n)); r.shuffle(permutation)
    edges = sorted([permutation[u], permutation[v]] for u, v in edges)
    topological_order(n, edges)
    return edges


class FocusedStore(Store):
    """Read-only union; the 1.6GB original cache is never copied or edited."""
    def __init__(self, cfg=None):
        self.cfg = cfg or config()
        self.train = sqlite3.connect(':memory:', uri=True)
        for alias, path in [('original', ROOT/self.cfg['original_cache']/'public.sqlite3'),
                            ('augmented', ROOT/self.cfg['data']/'training.sqlite3')]:
            self.train.execute('ATTACH DATABASE ? AS '+alias, ('file:'+str(path.resolve())+'?mode=ro',))
        self.train.execute('CREATE TEMP VIEW records AS SELECT * FROM original.records UNION ALL SELECT * FROM augmented.records')
        self.extra = connect(ROOT/self.cfg['data']/'validation.sqlite3')
        self.wf_keys = json.loads((ROOT/self.cfg['scope_lock']).read_text())['validation']['wf_by_application']['keys']
        self.panel_keys = {name: [r[0] for r in self.extra.execute('SELECT key FROM records WHERE panel=? ORDER BY key',(name,))]
                           for name in ('synthetic_val862', 'parrot_pattern_templates')}
        self.panel_keys['wf_by_application'] = sorted(self.wf_keys)
