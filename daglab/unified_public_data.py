"""Versioned public inputs/Teacher64 labels; no torch or network imports."""
import collections
import copy
import gzip
import hashlib
import json
import math
import random
import sqlite3
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/unified_public_v3_20261029.json'
BUNDLE = ROOT / 'data/offline_20261019'
SYNTHETIC = ROOT / 'data/structured_scheduling_v1'
PROTOCOL_COMMIT = '59a3bfd'
SOURCES = ['wfinstances', 'tpch_composite']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def pack(value):
    return zlib.compress(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode(), 3)


def unpack(value):
    return json.loads(zlib.decompress(value))


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def config():
    return json.loads(CONFIG.read_text())


def readonly(directory):
    return sqlite3.connect('file:' + str(Path(directory).resolve()/'public.sqlite3') + '?mode=ro', uri=True)


def database(directory):
    con = sqlite3.connect(Path(directory)/'public.sqlite3')
    con.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    con.execute('''CREATE TABLE IF NOT EXISTS records (
        key TEXT PRIMARY KEY, source TEXT, role TEXT, cohort INTEGER, base_id TEXT,
        topology TEXT, n INTEGER, k INTEGER, physical_sha TEXT, raw BLOB, payload BLOB)''')
    con.execute('CREATE INDEX IF NOT EXISTS source_role ON records(source,role)')
    con.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, count INTEGER, actions INTEGER, seconds REAL)')
    return con


def selectors():
    """Only input metadata/hashes determine membership, never any quality scores."""
    from .offline_data import digest
    cfg = config()
    index = json.loads((BUNDLE/'INDEX.json').read_text())
    groups = {}
    for source in SOURCES:
        for role in ['train', 'val', 'test']:
            groups[source, role] = sorted([r for r in index['bases'] if r['dataset'] == source and
                r['split'] == role and cfg['task_range'][0] <= r['n'] <= cfg['task_range'][1]], key=lambda r:r['base_id'])
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            assert not {r['base_id'] for r in groups[source,a]} & {r['base_id'] for r in groups[source,b]}
            assert not {r['topology_fingerprint'] for r in groups[source,a]} & {r['topology_fingerprint'] for r in groups[source,b]}
    assert [len(groups[s,r]) for s in SOURCES for r in ['train','val','test']] == [82,25,13,799,100,100]
    jobs = []
    for source in SOURCES:
        plans = [('train','train',cfg['train_cohorts'][source]), ('val','fresh_val',cfg['fresh_val_cohorts']),
                 ('test','test_primary',cfg['primary_test_cohorts'][source]),
                 ('test','test_reserve',cfg['reserve_test_cohorts'][source])]
        for original_role, role, cohorts in plans:
            blocks = [(cohort, base) for cohort in cohorts for base in groups[source,original_role]]
            if role == 'test_primary' and source == 'wfinstances':
                blocks.sort(key=lambda b:digest([cfg['selection_seed'],source,role,b[0],b[1]['base_id']]))
                blocks = blocks[:25]
                assert len({b['base_id'] for _,b in blocks}) == 13
            for cohort, base in blocks:
                ordered = sorted(groups[source,original_role], key=lambda b:digest([cfg['resource_seed'], source, original_role, cohort, b['base_id']]))
                mode = ['light','mixed','heavy'][next(i for i,b in enumerate(ordered) if b['base_id']==base['base_id']) % 3]
                jobs.append(dict(id=f'{source}:{role}:{cohort}:{base["base_id"]}', source=source, role=role,
                    cohort=cohort, base=base, pressure=mode, selected_keys=None))
    # Expansion selectors are frozen now but generated only after run one finishes.
    extra = []
    for source, cohort, count in [('wfinstances',2,656),('tpch_composite',1,4379)]:
        candidates = []
        for base in groups[source,'train']:
            with gzip.open(BUNDLE/base['adapted_file'],'rt') as f:
                candidates.extend((digest([cfg['selection_seed'],'expansion',r['key']]),base['base_id'],r['key']) for r in map(json.loads,f))
        chosen = collections.defaultdict(list)
        for _, base_id, key in sorted(candidates)[:count]:
            chosen[base_id].append(key)
        assert sum(map(len,chosen.values())) == count
        ordered = sorted(groups[source,'train'],key=lambda b:digest([cfg['resource_seed'],source,'train',cohort,b['base_id']]))
        for i, base in enumerate(ordered):
            if base['base_id'] in chosen:
                extra.append(dict(id=f'{source}:train:{cohort}:{base["base_id"]}',source=source,role='train',cohort=cohort,
                    base=base,pressure=['light','mixed','heavy'][i%3],selected_keys=sorted(chosen[base['base_id']])) )
    return dict(jobs=jobs, expansion_jobs=extra, source_groups={f'{s}:{r}':dict(bases=len(bs),
        topology_groups=len({b['topology_fingerprint'] for b in bs}), min_tasks=min(b['n'] for b in bs),
        max_tasks=max(b['n'] for b in bs)) for (s,r),bs in groups.items()},
        source_manifest_sha256=sha(BUNDLE/'MANIFEST.json'), index_sha256=sha(BUNDLE/'INDEX.json'),
        config_sha256=sha(CONFIG), protocol_commit=PROTOCOL_COMMIT,
        tpch_query_roles=index['tpch_query_roles'], selection_uses_scores=False)


def adapted(source, job):
    from .offline_data import digest, validate_sparse
    out = copy.deepcopy(source)
    x = out['instance_sparse'];cfg=config();base=job['base']['base_id'];cohort=job['cohort']
    weights={'light':[.7,.2,.08,.02],'mixed':[.25]*4,'heavy':[.05,.15,.3,.5]}
    def rng(kind):
        return random.Random(int(digest([cfg['resource_seed'],job['source'],base,cohort,kind])[:16],16))
    demands=rng('demands').choices([1,2,3,4],weights=weights[job['pressure']],k=x['n'])
    capacities=rng('capacities').choices([4,8,12],k=6)[:len(x['speeds'])]
    durations=[[math.ceil(w/s) for s in x['speeds']] for w in x['workload']]
    costs=[[d*r for d,r in zip(ds,x['cost_rates'])] for ds in durations]
    bmin=sum(min(ds) for ds in costs)
    bfast=sum(costs[v][min(range(len(capacities)),key=lambda p:(durations[v][p],costs[v][p],p))] for v in range(x['n']))
    key=f'publicv3_{job["role"]}_c{cohort}_{source["key"]}'
    x.update(id=key,demands=demands,capacities=capacities,bmin=bmin,bfast=bfast,
        budget=bmin+int(x['budget_eta']*(bfast-bmin)),
        resource_adaptation=dict(schema='public-resource-v3',seed=cfg['resource_seed'],cohort=cohort,
            pressure=job['pressure'],unit='simulated abstract cumulative units',legacy_key=source['key'],
            legacy_demands=source['instance_sparse']['demands'],legacy_capacities=source['instance_sparse']['capacities']))
    out['key']=key
    out['provenance'].update(adaptation_protocol=cfg['protocol'],original_split=job['base']['split'],
        adaptation_role=job['role'],cohort=cohort,resource_origin='simulated demand1..4 / capacity4,8,12')
    changed={'id','demands','capacities','bmin','bfast','budget','resource_adaptation'}
    assert all(x[k]==v for k,v in source['instance_sparse'].items() if k not in changed)
    assert all(d<=min(capacities) for d in demands)
    assert sum(math.ceil(w/x['speeds'][0])*x['cost_rates'][0] for w in x['workload'])==bmin<=x['budget']
    validate_sparse(out)
    return out


def physical_sha(row):
    from .offline_data import digest
    x=row['instance_sparse']
    keys=['n','edges','workload','demands','capacities','speeds','cost_rates','bandwidth','budget','data_mb_sparse','ticks_per_unit','latency','model','overlap']
    return digest({k:x[k] for k in keys})


def generate_job(job):
    import time
    from .offline_data import dense_instance
    from .reference import CachedBudgetDecoder, core, graph_features, mapping_of, teacher
    from .module_lab.features import enrich
    from .state_score import StateScoreDecoder
    begin=time.perf_counter();path=BUNDLE/job['base']['adapted_file']
    expected={r['path']:r['sha256'] for r in json.loads((BUNDLE/'MANIFEST.json').read_text())['files']}
    assert sha(path)==expected[job['base']['adapted_file']]
    with gzip.open(path,'rt') as f:
        source=list(map(json.loads,f))
    output=[];actions=0
    for original in source:
        if job['selected_keys'] is not None and original['key'] not in job['selected_keys']:
            continue
        row=adapted(original,job);s=row['instance_sparse'];payload=None
        if job['role'] in ['train','fresh_val']:
            inst=dense_instance(row,max_tasks=300);dec=StateScoreDecoder(inst)
            graph,_=graph_features(inst);graph=enrich(dict(instance=inst,graph=graph),dec)
            payload=dict(id=row['key'],base_id=job['base']['base_id'],n=inst['n'],graph=graph,
                reference=core.schedule(inst,'HEFT')['makespan'],meta=dict(source=job['source'],role=job['role'],
                cohort=job['cohort'],family=inst['family'],topology=inst['topology_fingerprint'],
                n=inst['n'],k=len(inst['speeds']),pressure=job['pressure'],eta=inst['budget_eta']))
            if job['role']=='train':
                solution,_,seconds=teacher(inst,CachedBudgetDecoder(inst))
                mapping=mapping_of(solution)
                replay=dec.rollout([[0.]*4]*(dec.n*dec.k),forced_mapping=mapping)
                assert replay['schedule']['tasks']==solution['tasks']
                assert replay['makespan']==solution['makespan'] and replay['cost']==solution['cost']
                assert solution['search']['attempted']==64
                payload.update(teacher=solution['makespan'],decisions=replay['decisions'],
                    supervision=dict(task_to_pool=mapping,makespan=solution['makespan'],cost=solution['cost'],
                    starts=[t['start'] for t in solution['tasks']],ends=[t['end'] for t in solution['tasks']],
                    algorithm=solution['algorithm'],search=solution['search']),teacher_seconds=seconds)
                actions+=len(replay['decisions'])
        output.append((row['key'],job['source'],job['role'],job['cohort'],job['base']['base_id'],
            job['base']['topology_fingerprint'],s['n'],len(s['speeds']),physical_sha(row),pack(row),pack(payload) if payload else None))
    assert len(output)==(len(job['selected_keys']) if job['selected_keys'] is not None else 8)
    return dict(job=job['id'],rows=output,actions=actions,seconds=time.perf_counter()-begin)
