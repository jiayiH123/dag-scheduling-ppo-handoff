#!/usr/bin/env python3
"""Final key-joined multi-DEV comparison; require all corrected primary runs complete."""
import hashlib,json,sys,csv,io
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'runs/external_baselines_20261028';RUN=ROOT/'runs/external_baselines_budget_visible_20261028'

def readrows(path):return [json.loads(line) for line in path.open()]

def main():
    control=readrows(ROOT/'runs/score_ranges_multival_20261028/cases.jsonl')
    meta={r['key']:r for r in control};assert len(meta)==1862
    methods={'GNN_B':{r['key']:r['methods']['control'] for r in control}}
    artifacts=[];work={}
    def add(path,rename=None):
        rows=readrows(path);assert len(rows)==len({r['key'] for r in rows})==1862
        assert {r['key'] for r in rows}==set(meta)
        for r in rows:
            assert r['references']['HEFT']==meta[r['key']]['references']['HEFT']
            assert all(v['feasible'] for v in r['methods'].values())
            for name,v in r['methods'].items():methods.setdefault((rename or {}).get(name,name),{})[r['key']]=v
        artifacts.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    add(OLD/'heuristics/cases.jsonl')
    for kind in ['cai','graphormer','cai_sparse']:
        path=RUN/kind/'summary.json';s=json.loads(path.read_text())
        assert s['status']=='complete' and not s.get('intermediate',True)
        work[kind]=s['training'];name=next(iter(next(iter(s['panels'].values()))['methods']))
        epoch=s['training']['epochs'];add(RUN/kind/f'epoch{epoch}_cases.jsonl',{name:name+'_last'})
        if kind!='cai_sparse':
            best=s['best_by_synthetic']['epoch'];add(RUN/kind/f'epoch{best}_cases.jsonl',{name:name+f'_selectedSyntheticEpoch{best}'})
    s=json.loads((RUN/'giph_fast_r1/summary.json').read_text());assert s['status']=='complete'
    add(RUN/'giph_fast_r1/cases.jsonl');work['giph']=s['training']
    refnames=set.union(*(set(r['references']) for r in control))
    for name in refnames:
        methods[name]={r['key']:dict(C=r['references'][name],feasible=True) for r in control if name in r['references']}
    panels={};rng=np.random.default_rng(2026102805);flat=[]
    for panel in ['synthetic','wfinstances','tpch_composite']:
        rows=[r for r in control if r['dataset']==panel];keys=[r['key'] for r in rows]
        groups=sorted({r['topology'] for r in rows});ix={g:i for i,g in enumerate(groups)}
        labels=np.array([ix[r['topology']] for r in rows]);counts=np.bincount(labels,minlength=len(groups))
        samples=rng.integers(len(groups),size=(2000,len(groups)));denom=counts[samples].sum(1)
        b=np.array([methods['GNN_B'][key]['C'] for key in keys],float)
        heft=np.array([meta[key]['references']['HEFT'] for key in keys],float)
        teacher=np.array([meta[key]['references']['Teacher64'] for key in keys],float)
        pm={}
        for name,values in methods.items():
            if not set(keys)<=set(values):continue
            c=np.array([values[key]['C'] for key in keys],float);ratio=c/heft;delta=(c-b)/heft
            sums=np.bincount(labels,weights=delta,minlength=len(groups));ci=np.quantile(sums[samples].sum(1)/denom,[.025,.975]).tolist()
            family={f:float(np.mean([values[r['key']]['C']/r['references']['HEFT'] for r in rows if r['family']==f])) for f in sorted({r['family'] for r in rows})}
            pm[name]=dict(C_HEFT=float(ratio.mean()),C_Teacher=float((c/teacher).mean()),family_macro_C_HEFT=float(np.mean(list(family.values()))),
                family_C_HEFT=family,mean_relative_C_vs_GNN=float((c/b-1).mean()),delta_C_HEFT_vs_GNN=float(delta.mean()),
                conditional_topology_cluster_CI95=ci,wins_GNN=int((c<b).sum()),ties_GNN=int((c==b).sum()),losses_GNN=int((c>b).sum()),
                improves_GNN_gt5pct=int((c/b<.95).sum()),regresses_GNN_gt5pct=int((c/b>1.05).sum()))
            flat.append(dict(dataset=panel,method=name,configs=len(rows),**{k:v for k,v in pm[name].items() if not isinstance(v,(dict,list))}))
        panels[panel]=dict(configs=len(rows),topology_groups=len(groups),methods=pm)
    # Search retention must be monotone and HEFT-start must retain HEFT itself.
    g=readrows(RUN/'giph_fast_r1/cases.jsonl')
    for r in g:
        for start in ['HEFT','random']:
            m=r['methods'];assert m[f'GiPH_{start}_2n']['C']<=m[f'GiPH_{start}_64']['C']<=m[f'GiPH_{start}_16']['C']
        assert r['methods']['GiPH_HEFT_16']['C']<=r['references']['HEFT']
    result=dict(status='complete_and_recomputed',panels=panels,training_work=work,bindings=artifacts,
        initialization_seed='external11; currentGNN28lineage',single_training_seed=True,TEST=False,
        interval_scope='conditional fixed-model topology-cluster bootstrap; not training seed or model-selection uncertainty',
        old_feature_normalization_attempts_excluded_from_primary=True,GiPH_retention_checked=True)
    (RUN/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    (RUN/'comparison.csv').write_text(buf.getvalue())
    print('PASS: all1862keysets,physicalflags,conditionalintervals,GiPHmonotonicretention; comparison saved')

if __name__=='__main__':main()
