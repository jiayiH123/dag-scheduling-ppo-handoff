#!/usr/bin/env python3
"""Generate new random input candidates and measure shape coverage; no teacher/model calls."""
import collections
import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from daglab.offline_data import topology_fingerprint, topological_order
from scripts.audit_target_structures_20261031 import features, FEATURES, sha

OUT = ROOT/'runs/target_structure_audit_20261031'
CONFIG = ROOT/'configs/synthetic_target_coverage_draft_20261031.json'


def topology(kind,n,r,cfg):
    edges=set()
    if kind=='independent_jobs':
        spec=cfg[kind]; lo=spec['component_nodes_min'];hi=spec['component_nodes_max']
        c=r.choice([v for v in spec['components'] if c_valid(n,v,lo,hi)])
        sizes=[lo]*c
        for _ in range(n-lo*c):
            j=r.choice([j for j in range(c) if sizes[j]<hi]);sizes[j]+=1
        offset=0
        for m in sizes:
            roots=r.randint(1,max(1,min(m-1,round(.55*m))))
            for u in range(roots):edges.add((offset+u,offset+roots))
            for v in range(roots+1,m):edges.add((offset+v-1,offset+v))
            for _ in range(m):
                if r.random()<spec['extra_edge_probability']:
                    u=r.randrange(m-1);v=r.randrange(u+1,m);edges.add((offset+u,offset+v))
            offset+=m
    elif kind=='broadcast_join':
        b=max(2,min((n-2)//2,round((n-2)*r.uniform(*cfg[kind]['branch_fraction_range']))))
        # Random numbers of tasks in each preparation/processing chain avoid a
        # near-single-template family with only superficial random shortcuts.
        lengths=[[1,1] for _ in range(b)]
        for _ in range(n-2-2*b):lengths[r.randrange(b)][r.randrange(2)]+=1
        offset=1; preparations=[];processing_heads=[]
        for j in range(b):
            prep,compute=lengths[j]
            edges.update((offset+i,offset+i+1) for i in range(prep-1))
            prep_end=offset+prep-1;compute_head=offset+prep
            edges.update((compute_head+i,compute_head+i+1) for i in range(compute-1))
            edges.update([(0,compute_head),(prep_end,compute_head),(compute_head+compute-1,n-1)])
            preparations.append(prep_end);processing_heads.append(compute_head)
            offset+=prep+compute
        for _ in range(r.randint(0,int(cfg[kind]['extra_edges_per_node_max']*n))):
            edges.add((r.choice(preparations),r.choice(processing_heads)))
    elif kind=='wide_forkjoin':
        choices=[b for b in cfg[kind]['branch_choices'] if 2*b<=n-2]
        b=r.choice(choices)
        sizes=[2]*b
        for _ in range(n-2-2*b):sizes[r.randrange(b)]+=1
        offset=1; layers={0:0,n-1:n}
        for m in sizes:
            edges.add((0,offset));edges.add((offset+m-1,n-1))
            for i in range(m):layers[offset+i]=i+1
            edges.update((offset+i,offset+i+1) for i in range(m-1));offset+=m
        for _ in range(r.randint(0,int(cfg[kind]['extra_edges_per_node_max']*n))):
            u,v=r.sample(range(n),2)
            if layers[u]>layers[v]:u,v=v,u
            if layers[u]<layers[v]:edges.add((u,v))
    else:raise ValueError(kind)
    # Relabel independently so IDs do not convey the generation order.
    permutation=list(range(n));r.shuffle(permutation)
    edges=sorted((permutation[u],permutation[v]) for u,v in edges)
    assert len(edges)<=cfg['edge_bound_per_node']*n
    topological_order(n,edges)
    return [list(e) for e in edges]


def c_valid(n,c,lo,hi):return lo*c<=n<=hi*c


def main():
    cfg=json.loads(CONFIG.read_text()); report=json.loads((OUT/'report.json').read_text())
    public=json.loads((OUT/'members.json').read_text())
    protected={r['topology'] for r in public['configs']}
    with gzip.open(ROOT/'data/structured_scheduling_v1/BASES.jsonl.gz','rt') as f:
        protected.update(json.loads(l)['topology_fingerprint'] for l in f if l.strip())
    # All synthetic roles are protected using topology fingerprints; no held-out performance or labels read.
    seen=set(protected);r=random.Random(cfg['seed']);rejected=collections.Counter();rows=[];attempts=0
    for kind,count in cfg['candidate_counts'].items():
        kept=0
        while kept<count:
            attempts+=1
            if attempts>30000:raise RuntimeError('Too many duplicate structural candidates')
            n=r.choice(cfg['tasks']);edges=topology(kind,n,r,cfg)
            fp=topology_fingerprint(n,edges)
            if fp in seen:rejected[kind]+=1;continue
            seen.add(fp)
            mode=r.choice(cfg['work_modes']);params={}
            if mode=='original_lognormal':work=[min(10.,max(.02,r.lognormvariate(-.4,.9))) for _ in range(n)]
            else:
                params={k:r.choice(v) for k,v in cfg['two_population'].items()}
                work=[r.lognormvariate(0,params['within_population_sigma'])*
                      (1 if r.random()<params['small_probability'] else params['large_small_ratio']) for _ in range(n)]
            mean=sum(work)/n;work=[w/mean for w in work]
            item=dict(base_id=f'coverage_candidate_{kind}_{kept:04d}',kind=kind,topology=fp,work_mode=mode,
                      work_parameters=params,n=n,edges=edges,relative_work=work,
                      features=features(n,edges,work))
            rows.append(item);kept+=1
            if kept%500==0:print(f'{kind}: {kept}/{count}; cumulative attempts={attempts}',flush=True)
        print(f'{kind}: {kept} unique input graphs generated',flush=True)
    path=OUT/'candidate_bases.jsonl.gz'
    with gzip.open(path,'wt') as f:
        for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
    targets=[]
    import csv
    with (OUT/'graph_features.csv').open() as f:
        for row in csv.DictReader(f):
            if row['source']!='synthetic':targets.append(row)
    mean=np.array(report['standardization_mean']);scale=np.array(report['standardization_scale'])
    tx=(np.array([[float(x[f]) for f in FEATURES] for x in targets])-mean)/scale
    sx=(np.array([[x['features'][f] for f in FEATURES] for x in rows])-mean)/scale
    dist=np.sqrt(np.maximum(0,(tx*tx).sum(1)[:,None]+(sx*sx).sum(1)[None,:]-2*tx@sx.T))
    comparisons={}
    for source in ['wfinstances','tpch_composite']:
        ids=[i for i,x in enumerate(targets) if x['source']==source]
        original=np.array([float(targets[i]['nearest_shape_distance']) for i in ids])
        augmented=np.minimum(original,dist[ids].min(1))
        comparisons[source]=dict(bases=len(ids),original_median_nearest_shape_distance=float(np.median(original)),
                                original_plus_candidates_median_nearest_shape_distance=float(np.median(augmented)),
                                fraction_with_strictly_closer_candidate=float(np.mean(dist[ids].min(1)<original-1e-10)))
    result=dict(scope='5,000 input-only candidate graphs; no resources/budgets/teacher labels materialized, no training or quality claim',
                config_sha256=sha(CONFIG),candidate_sha256=sha(path),audit_membership_sha256=report['membership_sha256'],
                protected_topology_fingerprints=len(protected),candidate_counts=cfg['candidate_counts'],
                duplicate_candidates_rejected=dict(rejected),attempts=attempts,
                accepted_n_counts={kind:dict(collections.Counter(x['n'] for x in rows if x['kind']==kind)) for kind in cfg['candidate_counts']},
                accepted_component_counts={kind:dict(collections.Counter(x['features']['components'] for x in rows if x['kind']==kind)) for kind in cfg['candidate_counts']},
                candidate_features={kind:{f:float(np.median([x['features'][f] for x in rows if x['kind']==kind])) for f in
                    ['n','components','edges_per_node','depth_fraction','max_out_fraction','work_cv','work_p90_over_median']}
                    for kind in cfg['candidate_counts']},descriptive_coverage=comparisons,
                limitations='Target-input-informed metric. More candidates can mechanically reduce nearest distance. These distances are not makespan evidence or proof that training helps.')
    (OUT/'candidate_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
