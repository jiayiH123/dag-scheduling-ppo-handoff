"""V5 synthetic-only transfer inputs and labels; previous corpora stay immutable."""
import collections
import copy
import gzip
import json
import math
import random
import sqlite3
import time
from pathlib import Path

from .unified_public_data import ROOT, BUNDLE, SYNTHETIC, sha, pack, unpack, write_json, database, physical_sha, adapted
from .offline_data import digest, topology_fingerprint, topological_order, dense_instance, physical_variants, validate_sparse
from .performance_data import connection, wf_identity, reference_values

CONFIG = ROOT/'configs/synthetic_transfer_20261031.json'


def config():
    return json.loads(CONFIG.read_text())


def instance(raw, source=None):
    if 'input' in raw:
        from .structured_dataset import to_instance
        return to_instance(raw)
    return dense_instance(raw, max_tasks=config()['max_tasks'])


def fingerprint(raw):
    if 'input' in raw:
        return digest(raw['input'])
    return physical_sha(raw)


def prepare_inputs():
    """Freeze input identities only. No policy or schedule-quality selection."""
    cfg=config(); out=ROOT/cfg['run'];out.mkdir(parents=True,exist_ok=True)
    old=json.loads((ROOT/cfg['legacy_data']/'INPUTS.json').read_text())
    with gzip.open(BUNDLE/'base_catalog.jsonl.gz','rt') as f:
        catalog=[json.loads(l) for l in f]
    wfs=[b for b in catalog if b['dataset']=='wfinstances' and 20<=b['n']<=300]
    wfs.sort(key=lambda b:b['base_id']);assert len(wfs)==120
    wf_meta=[json.loads(l) for l in (ROOT/'runs/dataset_composition_discussion_20261029/wf_source_metadata.jsonl').read_text().splitlines()]
    identities={r['base_id']:wf_identity(r) for r in wf_meta if 20<=r['n']<=300}
    small=[b for b in old['composites'] if b['split']=='dev'];assert len(small)==200
    used={b['signature'] for b in old['composites']}
    for b in catalog:
        if b['dataset']=='tpch_composite':
            used.add(digest(sorted((int(c['query_id']),c['input_scale']) for c in b['components'])))
    protected={b['topology_fingerprint'] for b in old['composites']}
    protected.update(b['topology_fingerprint'] for b in catalog)
    rng=random.Random(cfg['input_seed']);extra=[];rejected=collections.Counter()
    primitives=old['primitives']
    for size,count in cfg['large_tpch'].items():
        size=int(size)
        for i in range(count):
            for attempt in range(10000):
                selected=sorted([rng.choice(primitives) for _ in range(size)],key=lambda b:(b['query_id'],b['input_scale']))
                signature=digest(sorted((int(b['query_id']),b['input_scale']) for b in selected))
                if signature in used:rejected['multiset']+=1;continue
                edges=[];traffic=[];work=[];demands=[];components=[];offset=0
                for b in selected:
                    edges.extend([u+offset,v+offset] for u,v in b['edges']);traffic.extend(b['edge_data_mb'])
                    work.extend(b['workload']);demands.extend(b['demands'])
                    components.append(dict(base_id=b['base_id'],query_id=b['query_id'],input_scale=b['input_scale'],
                                           offset=offset,n=b['n'],source_sha256=b['source_sha256']))
                    offset+=b['n']
                fp=topology_fingerprint(offset,edges)
                if fp in protected:rejected['topology']+=1;continue
                assert offset<=cfg['max_tasks'];break
            else:raise RuntimeError('new TPC composition isolation exhausted')
            used.add(signature);protected.add(fp)
            base=copy.deepcopy(small[0])
            base.update(base_id=f'tpchv5_eval_{size}_{i:03d}_{signature[:12]}',n=offset,edges=edges,edge_data_mb=traffic,
                workload=work,demands=demands,components=components,num_queries=size,signature=signature,
                source_path='generated-id22-composition/'+signature,source_sha256=signature,
                topology_fingerprint=fp,split='eval',previously_inspected=False)
            extra.append(base)
    # Include every old synthetic role in the topology exclusion set.
    with gzip.open(SYNTHETIC/'BASES.jsonl.gz','rt') as f:
        synthetic_bases=[json.loads(l) for l in f]
    protected.update(b['topology_fingerprint'] for b in synthetic_bases)
    with gzip.open(ROOT/cfg['candidate_file'],'rt') as f:
        candidates=[json.loads(l) for l in f]
    assert len(candidates)==5000
    seen=set(protected)
    for b in candidates:
        assert b['topology']==topology_fingerprint(b['n'],b['edges'])
        assert b['topology'] not in seen,'candidate overlaps expanded evaluation or previous corpus'
        seen.add(b['topology'])
    from scripts.generate_coverage_candidates_20261031 import topology
    shape=json.loads((ROOT/'configs/synthetic_target_coverage_draft_20261031.json').read_text())
    from scripts.audit_target_structures_20261031 import features
    for components,n,count in cfg['extra_synthetic_cells']:
        spec=copy.deepcopy(shape);spec['independent_jobs']['components']=[components]
        for i in range(count):
            while True:
                edges=topology('independent_jobs',n,rng,spec);fp=topology_fingerprint(n,edges)
                if fp not in seen:break
                rejected['large_synthetic_topology']+=1
            seen.add(fp)
            params={k:rng.choice(v) for k,v in shape['two_population'].items()}
            work=[rng.lognormvariate(0,params['within_population_sigma'])*(1 if rng.random()<params['small_probability'] else params['large_small_ratio']) for _ in range(n)]
            mean=sum(work)/n;work=[w/mean for w in work]
            candidates.append(dict(base_id=f'coverage_large_c{components}_n{n}_{i:03d}',kind='independent_jobs',
                topology=fp,work_mode='two_population',work_parameters=params,n=n,edges=edges,
                relative_work=work,features=features(n,edges,work)))
    result=dict(schema=cfg['schema'],config_sha256=sha(CONFIG),old_inputs_sha256=sha(ROOT/cfg['legacy_data']/'INPUTS.json'),
        candidate_source_sha256=sha(ROOT/cfg['candidate_file']),wfs=wfs,wf_identity=identities,
        tpch_existing=small,tpch_extra=extra,rejected=dict(rejected),selection_uses_performance=False)
    path=out/'INPUTS.json'
    if path.exists():assert json.loads(path.read_text())==result,'Frozen inputs changed'
    else:write_json(path,result)
    cpath=out/'CANDIDATES.jsonl.gz'
    if cpath.exists():
        with gzip.open(cpath,'rt') as f:assert [json.loads(l) for l in f]==candidates
    else:
        with gzip.open(cpath,'wt') as f:
            for row in candidates:f.write(json.dumps(row,separators=(',',':'))+'\n')
    report={}
    for name,bs in [('WF',wfs),('TPC',small+extra),('augmentation',candidates)]:
        report[name]=dict(bases=len(bs),node_min=min(b['n'] for b in bs),node_max=max(b['n'] for b in bs),
            topology_groups=len({b.get('topology_fingerprint',b.get('topology')) for b in bs}))
    report['augmentation']['cells']=dict(collections.Counter(f"{b['kind']}:n{b['n']}:components{b['features']['components']}" for b in candidates))
    # Input-only expanded coverage audit; the original standardization stays fixed.
    audit=json.loads((ROOT/'runs/target_structure_audit_20261031/report.json').read_text())
    import numpy as np
    from scripts.audit_target_structures_20261031 import FEATURES,quantiles
    mean=np.array(audit['standardization_mean']);scale=np.array(audit['standardization_scale'])
    cx=(np.array([[b['features'][f] for f in FEATURES] for b in candidates])-mean)/scale
    report['expanded_target_features']={}
    for name,bs in [('WF',wfs),('TPC',small+extra)]:
        fs=[features(b['n'],b['edges'],b['workload']) for b in bs]
        nearest=[]
        for b in fs:
            x=(np.array([b[f] for f in FEATURES])-mean)/scale
            nearest.append(float(np.sqrt(((cx-x)**2).sum(1).min())))
        report['expanded_target_features'][name]=dict(nearest_candidate_distance=quantiles(nearest),
            features={f:quantiles([b[f] for b in fs]) for f in FEATURES+['work_p90_over_median']})
    write_json(out/'INPUT_AUDIT.json',report)
    return result,candidates


def synthetic_variants(base,attributes=False):
    """Paired graph/physics streams. Attributes change only declared input groups."""
    cfg=config();n=base['n'];edges=base['edges'];seed=[cfg['input_seed'],base['base_id']]
    def rng(kind):return random.Random(int(digest([seed,kind])[:16],16))
    r=rng('original_physics');k=r.choice([3,4,6]);H=r.choice([2,4,8]);pressure=r.choice(['light','mixed','heavy'])
    work=[round(min(10.,max(.02,r.lognormvariate(-.4,.9))),6) for _ in range(n)]
    weights={'light':[.7,.2,.08,.02],'mixed':[.25]*4,'heavy':[.05,.15,.3,.5]}[pressure]
    demands=r.choices([1,2,3,4],weights=weights,k=n);caps=r.choices([4,8,12],k=k)
    speeds=[1.,float(H)]+[math.exp(r.uniform(0,math.log(H))) for _ in range(k-2)];r.shuffle(speeds)
    gamma=r.uniform(1.2,2.);rates=[max(1,round(10*s**gamma*r.uniform(.9,1.1))) for s in speeds]
    bw=[[0.]*k for _ in range(k)]
    for p in range(k):
        for q in range(p+1,k):bw[p][q]=bw[q][p]=round(r.choice([10,100,1000])*r.uniform(.8,1.2),6)
    ccr=r.choice([0.,.1,1.,5.]);pair=r.choice([[.1,.5],[.25,.75],[1.,'nonbinding']])
    changes=dict(work=False,platform=False,CCR=False,zero_fraction=0.)
    a=cfg['augmentation']
    if attributes:
        if rng('work_change').random()<a['work_heavy_tail_probability']:
            # Re-generate two populations even for pilot rows tagged original_lognormal.
            rr=rng('work_values');params=base['work_parameters'] or dict(small_probability=rr.choice([.2,.5,.7]),large_small_ratio=rr.choice([10,30,100]),within_population_sigma=rr.choice([.3,.6]))
            new=[rr.lognormvariate(0,params['within_population_sigma'])*(1 if rr.random()<params['small_probability'] else params['large_small_ratio']) for _ in range(n)]
            factor=sum(work)/sum(new);work=[round(w*factor,8) for w in new];changes.update(work=True,work_parameters=params)
        if rng('platform_change').random()<a['public_platform_probability']:
            rr=rng('public_platform');k=rr.choice([3,6]);speeds=[1,2,4]*(k//3);rates=[10,40,160]*(k//3)
            demands=rng('public_demands').choices([1,2,3,4],weights=weights,k=n);caps=rng('public_capacities').choices([4,8,12],k=k)
            factor=rr.choice([1,10]);b3=[[0,10,20],[10,0,40],[20,40,0]]
            bw=[[0 if p==q else factor*(b3[p%3][q%3] or 10) for q in range(k)] for p in range(k)]
            pair=[.25,.6];changes['platform']=True
        if rng('ccr_change').random()<a['low_CCR_probability']:
            rr=rng('low_ccr');ccr=math.exp(rr.uniform(*map(math.log,a['low_CCR_range'])))
            changes['CCR']=True;changes['zero_fraction']=rr.choice(a['zero_byte_edge_fractions'])
    durations=[[max(1,math.ceil(w/s*1000-1e-10)) for s in speeds] for w in work]
    costs=[[d*r for d,r in zip(ds,rates)] for ds in durations]
    bmin=sum(min(ds) for ds in costs);bmax=sum(max(ds) for ds in costs)
    bfast=sum(ds[min(range(k),key=lambda p:(durations[v][p],ds[p],p))] for v,ds in enumerate(costs))
    mean_compute=sum(map(sum,durations))/(n*k);inv=sum(1/bw[p][q] for p in range(k) for q in range(k) if p!=q)/(k*(k-1))
    rr=rng('traffic');factors=[min(4.,max(.25,rr.lognormvariate(0,.5))) for _ in edges]
    zr=rng('zero_edges');factors=[0. if zr.random()<changes['zero_fraction'] else f for f in factors]
    if not any(factors):factors[0]=1.
    factor_mean=sum(factors)/len(factors);sizes=[ccr*(mean_compute/1000)/inv*f/factor_mean for f in factors]
    tag='attributes' if attributes else 'structure';seen=set()
    for j,eta in enumerate(pair):
        budget=bmax if eta=='nonbinding' else bmin+int(eta*(bfast-bmin))
        if budget in seen:continue
        seen.add(budget);key=f'v5_train_{tag}_{base["base_id"]}_b{j}'
        x=dict(id=key,n=n,edges=edges,workload=work,demands=demands,capacities=caps,speeds=speeds,cost_rates=rates,
            bandwidth=bw,data_mb_sparse=[[u,v,s] for (u,v),s in zip(edges,sizes)],latency=0,model='fixed',overlap=True,
            ticks_per_unit=1000,budget=budget,bmin=bmin,bfast=bfast,budget_eta=None if eta=='nonbinding' else eta,
            budget_mode=str(eta),target_ccr=ccr,latent_id=base['base_id'],family=base['kind'],resource_mode='pool',
            source='new_random_synthetic_v5',seed=int(digest([seed,'schedule'])[:8],16),
            topology_fingerprint=base['topology'],resource_adaptation=dict(pressure=pressure),augmentation=changes)
        topological_order(n,edges)
        assert min(caps)>=max(demands) and budget>=bmin
        yield dict(key=key,instance_sparse=x,provenance=dict(source_kind='independently_generated_synthetic',
            base_id=base['base_id'],target_input_statistics_informed=True,recipe=tag,seed=cfg['input_seed']))


def make_payload(raw,source,role,meta):
    from .reference import graph_features,core,teacher,CachedBudgetDecoder,mapping_of
    from .module_lab.features import enrich
    from .state_score import StateScoreDecoder
    inst=instance(raw);dec=StateScoreDecoder(inst);g,_=graph_features(inst)
    g=enrich(dict(instance=inst,graph=g),dec)
    out=dict(id=raw['key'],base_id=meta['base_id'],n=inst['n'],graph=g,meta=meta)
    if role=='train':
        solution,_,seconds=teacher(inst,CachedBudgetDecoder(inst));mapping=mapping_of(solution)
        replay=dec.rollout([[0.]*4]*(dec.n*dec.k),forced_mapping=mapping)
        assert solution['search']['attempted']==64
        assert replay['schedule']['tasks']==solution['tasks']
        assert (replay['makespan'],replay['cost'])==(solution['makespan'],solution['cost'])
        out.update(reference=core.schedule(inst,'HEFT')['makespan'],teacher=solution['makespan'],decisions=replay['decisions'],
            supervision=dict(task_to_pool=mapping,makespan=solution['makespan'],cost=solution['cost'],
                starts=[t['start'] for t in solution['tasks']],ends=[t['end'] for t in solution['tasks']],
                search=solution['search'],algorithm=solution['algorithm'],teacher_seconds=seconds))
    else:
        out['references']=reference_values(inst);out['reference']=out['references']['HEFT']['C']
        zero=dec.rollout([[0.]*4]*(dec.n*dec.k))
        assert zero['makespan']==out['reference']
    return out


def metadata(raw,source,base_id,topology,role='eval',**extra):
    x=instance(raw)
    return dict(key=raw['key'],source=source,role=role,base_id=base_id,topology=topology,n=x['n'],k=len(x['speeds']),
        family=x['family'],pressure=x.get('resource_adaptation',{}).get('pressure',x.get('demand_pressure','unknown')),
        eta=x.get('budget_eta'),**extra)


def row_tuple(raw,payload):
    m=payload['meta']
    return (raw['key'],m['source'],m['role'],0,m['base_id'],m['topology'],m['n'],m['k'],fingerprint(raw),pack(raw),pack(payload))


def generate_job(job):
    begin=time.perf_counter();rows=[]
    for raw,meta in job['rows']:
        payload=make_payload(raw,meta['source'],meta['role'],meta);rows.append(row_tuple(raw,payload))
    return dict(id=job['id'],rows=rows,seconds=time.perf_counter()-begin,
        actions=sum(len(unpack(row[-1]).get('decisions',[])) for row in rows))


def grouped_metrics(rows):
    """Paired ratios; explicit equal group weighting for all methods and sources."""
    result={}
    refs=sorted(set.intersection(*(set(r['references']) for r in rows)))
    for source in ['synthetic','wfinstances','tpch_composite']:
        rr=[r for r in rows if r['source']==source];bases=collections.defaultdict(list)
        for r in rr:bases[r['base_id']].append(r)
        ratios={ref:{} for ref in refs}
        for ref in refs:
            for b,rs in bases.items():ratios[ref][b]=sum(r['C']/r['references'][ref]['C'] for r in rs)/len(rs)
        groups=collections.defaultdict(list)
        for b,rs in bases.items():
            m=rs[0]
            if source=='synthetic':g=(m['n'],m['family'])
            elif source=='tpch_composite':g=m['num_queries']
            else:g=(m['identity']['engine'],m['identity']['application'],m['topology'])
            groups[g].append(b)
        values={};strata={}
        for ref in refs:
            avg={g:sum(ratios[ref][b] for b in bs)/len(bs) for g,bs in groups.items()}
            if source=='wfinstances':
                apps=collections.defaultdict(list)
                for (engine,app,topo),v in avg.items():apps[f'{engine}/{app}'].append(v)
                avg={g:sum(v)/len(v) for g,v in apps.items()}
            values[ref]=sum(avg.values())/len(avg)
            strata[ref]={str(g):v for g,v in avg.items()}
        result[source]=dict(configs=len(rr),bases=len(bases),C_over=values,strata=strata,
            C_HEFT_execution_mean=sum(ratios['HEFT'].values())/len(bases),feasible=all(r['feasible'] for r in rr))
    return result
