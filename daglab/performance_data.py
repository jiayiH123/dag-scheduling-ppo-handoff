"""V4 working corpus. Legacy bundles stay immutable; simulation runtime only."""
import collections
import copy
import gzip
import io
import json
import random
import sqlite3
import tarfile
import time
from pathlib import Path

from .unified_public_data import ROOT, BUNDLE, SYNTHETIC, sha, pack, unpack, write_json, database, adapted, physical_sha
from .offline_data import digest, topology_fingerprint, simulated_tpch_traffic, physical_variants, validate_sparse, dense_instance

CONFIG = ROOT / 'configs/performance_autoresearch_20261030.json'


def config():
    return json.loads(CONFIG.read_text())


def connection(directory, readonly=True):
    return sqlite3.connect('file:' + str(Path(directory).resolve() / 'public.sqlite3') + ('?mode=ro' if readonly else ''), uri=True)


def instance(raw, source):
    if source == 'synthetic':
        from .structured_dataset import to_instance
        return to_instance(raw)
    return dense_instance(raw, max_tasks=300)


def wf_identity(row):
    parts = row['source_path'].split('/')
    engine = parts[0]
    if engine == 'nextflow':
        if parts[1] == 'legacy':
            app = row['source_name']
        elif len(parts) > 3:
            app = parts[2]
        else:
            app = parts[-1].split('-')[0]
    elif engine == 'streamflow':
        app = parts[-1].split('-')[0]
    else:
        app = parts[1]
    return dict(engine=engine, application=app, pipeline_version='unknown',
                engine_version=row['source_engine'].get('version', 'unknown'),
                execution=row['base_id'], executed_at=row['executed_at'],
                source_path=row['source_path'], identity_basis='source directory and workflow name; not inferred from model performance')


def prepare_inputs(out):
    """Bind raw matrices, correct their direction and freeze new composite identities."""
    import numpy as np
    out = Path(out)
    with gzip.open(BUNDLE / 'base_catalog.jsonl.gz', 'rt') as f:
        catalog = [json.loads(line) for line in f]
    sources = json.loads((BUNDLE / 'SOURCES.json').read_text())
    archive = sources['tpch']['archive']
    blob = b''.join((BUNDLE / p).read_bytes() for p in archive['parts'])
    import hashlib
    assert hashlib.sha256(blob).hexdigest() == archive['archive_sha256']
    matrices = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as f:
        for m in f:
            if m.isfile() and '/adj_mat_' in m.name and m.name.endswith('.npy'):
                rel = m.name.split('/', 1)[1]
                matrices[rel] = np.load(io.BytesIO(f.extractfile(m).read()), allow_pickle=False)
    primitives = []
    for old in catalog:
        if old['dataset'] != 'tpch_primitive':
            continue
        b = copy.deepcopy(old)
        rel = f"dag_data/tpch/{b['input_scale']}/adj_mat_{b['query_id']}.npy"
        matrix = matrices[rel]
        edges = sorted([int(i), int(j)] for i, j in np.argwhere(matrix == 1))
        assert sorted([v, u] for u, v in b['edges']) == edges
        b.update(edges=edges, topology_fingerprint=topology_fingerprint(b['n'], edges),
                 split='shared_primitive', dependency_convention='Decima adj[i,j]=1 means i->j')
        simulated_tpch_traffic(b)  # Anchored once on query/scale/edge; never regenerated on composites.
        b['communication_origin'] += '; v4 primitive-anchored, copied unchanged into compositions'
        primitives.append(b)
    assert len(primitives) == 154
    canonical = lambda bs: digest(sorted((int(b['query_id']), b['input_scale']) for b in bs))
    used = {canonical(b['components']) for b in catalog if b['dataset'] == 'tpch_composite'}
    assert len(used) == 1000
    fingerprint_role, composites, rejected = {}, [], collections.Counter()
    cfg = config()
    for role, count in cfg['tpch_counts'].items():
        rng = random.Random(f"{cfg['composition_seed']}:{role}")
        for i in range(count):
            size = [5, 10, 20][i % 3]
            for attempt in range(10000):
                selected = sorted((rng.choice(primitives) for _ in range(size)), key=lambda b:(b['query_id'], b['input_scale']))
                signature = canonical(selected)
                if signature in used:
                    rejected['multiset'] += 1
                    continue
                n = sum(b['n'] for b in selected)
                if not 20 <= n <= 300:
                    rejected['size'] += 1
                    continue
                edges, traffic, work, demands, components = [], [], [], [], []
                offset = 0
                for b in selected:
                    edges.extend([u + offset, v + offset] for u, v in b['edges'])
                    traffic.extend(b['edge_data_mb'])
                    work.extend(b['workload']); demands.extend(b['demands'])
                    components.append(dict(base_id=b['base_id'], query_id=b['query_id'], input_scale=b['input_scale'],
                                           offset=offset, n=b['n'], source_sha256=b['source_sha256']))
                    offset += b['n']
                fp = topology_fingerprint(n, edges)
                if fp in fingerprint_role and fingerprint_role[fp] != role:
                    rejected['topology'] += 1
                    continue
                break
            else:
                raise RuntimeError('composition isolation exhausted')
            used.add(signature); fingerprint_role[fp] = role
            base = dict(base_id=f'tpchv4_{role}_{i:04d}_{signature[:12]}', dataset='tpch_composite',
                        n=n, edges=edges, edge_data_mb=traffic, workload=work, demands=demands, pool_capacity=600,
                        source_path='generated-id22-composition/' + signature, source_sha256=signature,
                        source_commit=primitives[0]['source_commit'], source_kind='generated_disjoint_union_of_public_query_stages',
                        family='tpch_composite', components=components, num_queries=size, topology_fingerprint=fp,
                        split=role, previously_inspected=False, resource_origin='original stage task counts retained; simulated resources applied later',
                        duration_origin=primitives[0]['duration_origin'], communication_origin=primitives[0]['communication_origin'],
                        signature=signature, dependency_convention='Decima forward')
            assert base['edges'] == sorted(base['edges'])
            composites.append(base)
    wf_path = ROOT / 'runs/dataset_composition_discussion_20261029/wf_source_metadata.jsonl'
    wf_rows = [json.loads(line) for line in wf_path.read_text().splitlines()]
    identities = {r['base_id']: wf_identity(r) for r in wf_rows if 20 <= r['n'] <= 300}
    seen = {(identities[r['base_id']]['engine'], identities[r['base_id']]['application'])
            for r in wf_rows if r['base_id'] in identities and r['split'] == 'train'}
    for ident in identities.values():
        ident['training_supported_engine_application'] = (ident['engine'], ident['application']) in seen
    result = dict(primitives=primitives, composites=composites, wf_identity=identities,
                  evidence=dict(raw_adjacencies_checked=154, raw_archive_sha256=archive['archive_sha256'],
                                old_multisets_excluded=1000, rejected=dict(rejected),
                                wf_identity_source_sha256=sha(wf_path), config_sha256=sha(CONFIG)))
    path = out / 'INPUTS.json'
    if path.exists():
        assert json.loads(path.read_text()) == result, 'input generation changed on resume'
    else:
        write_json(path, result)
    return result


def reference_values(inst):
    from scripts.unified_public_worker import baselines
    from .reference import portfolio
    result = baselines(inst)
    seeds, _ = portfolio(inst)
    for i, name in [(1, 'PEFT'), (2, 'JoinLookahead'), (3, 'CostFirst')]:
        result[name] = dict(C=seeds[i]['makespan'], cost=seeds[i]['cost'], feasible=True)
    return result


def payload_for(raw, source, role, base_id, cohort, metadata, existing=None):
    from .reference import graph_features, core, teacher, CachedBudgetDecoder, mapping_of
    from .module_lab.features import enrich
    from .state_score import StateScoreDecoder
    inst = instance(raw, source)
    dec = StateScoreDecoder(inst)
    g, _ = graph_features(inst)
    g = enrich(dict(instance=inst, graph=g), dec)
    out = dict(id=raw['key'], base_id=base_id, n=inst['n'], graph=g,
               reference=core.schedule(inst, 'HEFT')['makespan'], meta=metadata)
    if role == 'train':
        if existing is not None:
            supervision = existing['supervision']
        elif source == 'synthetic':
            supervision = raw['supervision']
        else:
            solution, _, seconds = teacher(inst, CachedBudgetDecoder(inst))
            assert solution['search']['attempted'] == 64
            supervision = dict(task_to_pool=mapping_of(solution), makespan=solution['makespan'], cost=solution['cost'],
                               starts=[t['start'] for t in solution['tasks']], ends=[t['end'] for t in solution['tasks']],
                               search=solution['search'], algorithm=solution['algorithm'], teacher_seconds=seconds)
        replay = dec.rollout([[0.] * 4] * (dec.n * dec.k), forced_mapping=supervision['task_to_pool'])
        assert replay['makespan'] == supervision['makespan'] and replay['cost'] == supervision['cost']
        for field, target in [('start', 'starts'), ('end', 'ends')]:
            assert [t[field] for t in replay['schedule']['tasks']] == supervision[target]
        if existing is not None:
            assert out['graph'] == existing['graph'] and out['reference'] == existing['reference']
            assert replay['decisions'] == existing['decisions']
        out.update(supervision=supervision, decisions=replay['decisions'], teacher=supervision['makespan'])
    else:
        out['references'] = reference_values(inst)
        assert out['reference'] == out['references']['HEFT']['C']
    return out


def generate(job):
    begin = time.perf_counter()
    records, actions = [], 0
    cfg = config()
    source, role = job['source'], job['role']
    rows = []
    if source == 'synthetic':
        for raw in job['rows']:
            rows.append((raw, raw['base_id'], 0, raw['meta']['topology_fingerprint'], None, {}))
    elif source == 'wfinstances':
        con = connection(ROOT / cfg['legacy_data'])
        for key in job['keys']:
            r = con.execute('SELECT raw,payload,base_id,cohort,topology FROM records WHERE key=?', (key,)).fetchone()
            raw, existing = unpack(r[0]), unpack(r[1])
            ident = job['identities'][r[2]]
            raw['provenance']['source_identity'] = ident
            rows.append((raw, r[2], r[3], r[4], existing, ident))
        con.close()
    else:
        base = job['base']
        for original in physical_variants(base):
            raw = adapted(original, dict(source=source, role=role, cohort=job['cohort'], base=base, pressure=job['pressure']))
            raw['key'] = raw['key'].replace('publicv3_', 'publicv4_', 1)
            raw['instance_sparse']['id'] = raw['key']
            raw['provenance'].update(adaptation_protocol=cfg['protocol'], composition_signature=base['signature'],
                                     components=base['components'], dependency_convention=base['dependency_convention'])
            validate_sparse(raw)
            rows.append((raw, base['base_id'], job['cohort'], base['topology_fingerprint'], None, {}))
    for raw, base_id, cohort, topology, existing, identity in rows:
        inst = instance(raw, source)
        main = source != 'wfinstances' or identity['training_supported_engine_application']
        meta = dict(key=raw['key'], base_id=base_id, source=source, role=role, cohort=cohort,
                    topology=topology, n=inst['n'], k=len(inst['speeds']), family=inst['family'],
                    pressure=inst.get('resource_adaptation', {}).get('pressure', inst.get('demand_pressure', 'unknown')),
                    eta=inst.get('budget_eta'), main_dev=main, identity=identity)
        payload = payload_for(raw, source, role, base_id, cohort, meta, existing)
        actions += len(payload.get('decisions', []))
        fingerprint = digest(raw['input']) if source == 'synthetic' else physical_sha(raw)
        records.append((raw['key'], source, role, cohort, base_id, topology, inst['n'], len(inst['speeds']),
                        fingerprint, pack(raw), pack(payload)))
    return dict(job=job['id'], rows=records, actions=actions, seconds=time.perf_counter()-begin)


def jobs(inputs):
    from .structured_unique import iter_unique_rows
    for split, role in [('train', 'train'), ('val', 'dev')]:
        chunk, part = [], 0
        for row in iter_unique_rows(SYNTHETIC, split):
            chunk.append(row)
            if len(chunk) == 64:
                yield dict(id=f'synthetic:{role}:{part}', source='synthetic', role=role, rows=chunk)
                chunk = []; part += 1
        if chunk:
            yield dict(id=f'synthetic:{role}:{part}', source='synthetic', role=role, rows=chunk)
    con = connection(ROOT / config()['legacy_data'])
    for old_role, role in [('train', 'train'), ('fresh_val', 'dev')]:
        keys = [r[0] for r in con.execute('SELECT key FROM records WHERE source="wfinstances" AND role=? ORDER BY key', (old_role,))]
        for start in range(0, len(keys), 16):
            yield dict(id=f'wfinstances:{role}:{start}', source='wfinstances', role=role,
                       keys=keys[start:start+16], identities=inputs['wf_identity'])
    con.close()
    for role, cohorts in [('train', [0, 1]), ('dev', [3])]:
        bases = [b for b in inputs['composites'] if b['split'] == role]
        for cohort in cohorts:
            ordered = sorted(bases, key=lambda b:digest([config()['resource_seed'], 'tpch_composite', role, cohort, b['base_id']]))
            for i, b in enumerate(ordered):
                yield dict(id=f'tpch:{role}:{cohort}:{b["base_id"]}', source='tpch_composite', role=role,
                           cohort=cohort, base=b, pressure=['light', 'mixed', 'heavy'][i % 3])


def probe_keys(con):
    def select(source, count, variants):
        groups = collections.defaultdict(list)
        sizes = {}
        for key, base, n in con.execute('SELECT key,base_id,n FROM records WHERE source=? AND role="train" ORDER BY key', (source,)):
            groups[base].append(key); sizes[base] = n
        bins = collections.defaultdict(list)
        for base, n in sizes.items():
            bins[0 if n <= 40 else 1 if n <= 80 else 2 if n <= 120 else 3].append(base)
        ordered = [iter(sorted(bs, key=lambda b:digest([2026103002, source, b]))) for _, bs in sorted(bins.items())]
        bases = []
        while len(bases) < min(count, len(groups)):
            for it in ordered:
                b = next(it, None)
                if b is not None and len(bases) < count:
                    bases.append(b)
        return [k for b in bases for k in sorted(groups[b], key=lambda k:digest(['probe_variant', k]))[:variants]]
    result = dict(synthetic=select('synthetic', 1536, 1),
                  wfinstances=[r[0] for r in con.execute('SELECT key FROM records WHERE source="wfinstances" AND role="train" ORDER BY key')],
                  tpch_composite=select('tpch_composite', 512, 2))
    assert [len(result[s]) for s in config()['sources']] == [1536, 1968, 1024]
    return result
