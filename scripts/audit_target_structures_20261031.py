#!/usr/bin/env python3
"""Input-only structural coverage audit; never loads policies or scheduling results."""
import collections
import csv
import gzip
import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daglab.structured_synthetic import topology, instances
from daglab.offline_data import topological_order

RUN = ROOT / 'runs/performance_autoresearch_20261030'
OUT = ROOT / 'runs/target_structure_audit_20261031'
FEATURES = ['edges_per_node', 'source_fraction', 'sink_fraction', 'fork_fraction', 'join_fraction',
            'max_out_fraction', 'max_in_fraction', 'depth_fraction', 'layer_width_fraction',
            'log_components', 'largest_component_fraction', 'reachability_fraction']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def quantiles(v):
    return dict(zip(['min', 'p10', 'median', 'p90', 'max'], map(float, np.quantile(v, [0, .1, .5, .9, 1]))))


def features(n, edges, work):
    parents = [[] for _ in range(n)]; children = [[] for _ in range(n)]
    root = list(range(n))
    def find(a):
        while root[a] != a:
            root[a] = root[root[a]]; a = root[a]
        return a
    for u, v in edges:
        children[u].append(v); parents[v].append(u)
        root[find(u)] = find(v)
    order = topological_order(n, edges)
    level = [0] * n; path_work = [0.] * n; descendants = [0] * n
    for v in order:
        level[v] = max((level[u] + 1 for u in parents[v]), default=0)
        path_work[v] = float(work[v]) + max((path_work[u] for u in parents[v]), default=0.)
    for u in reversed(order):
        for v in children[u]:
            descendants[u] |= (1 << v) | descendants[v]
    cs = sorted(collections.Counter(find(v) for v in range(n)).values())
    indeg = np.array(list(map(len, parents))); outdeg = np.array(list(map(len, children)))
    w = np.array(work, dtype=float)
    assert len(w) == n and np.all(w > 0)
    return dict(n=n, edges=len(edges), edges_per_node=len(edges)/n,
                source_fraction=float(np.mean(indeg == 0)), sink_fraction=float(np.mean(outdeg == 0)),
                fork_fraction=float(np.mean(outdeg > 1)), join_fraction=float(np.mean(indeg > 1)),
                max_out_fraction=float(outdeg.max()/n), max_in_fraction=float(indeg.max()/n),
                depth=max(level)+1, depth_fraction=(max(level)+1)/n,
                layer_width=max(collections.Counter(level).values()),
                layer_width_fraction=max(collections.Counter(level).values())/n,
                components=len(cs), log_components=math.log1p(len(cs)), largest_component_fraction=max(cs)/n,
                component_size_median=float(np.median(cs)),
                reachability_fraction=sum(x.bit_count() for x in descendants)/max(1, n*(n-1)/2),
                work_cv=float(w.std()/w.mean()), work_p90_over_median=float(np.quantile(w,.9)/np.median(w)),
                max_work_fraction=float(w.max()/w.sum()), critical_path_work_fraction=max(path_work)/float(w.sum()))


def physical(x):
    work = x['workload']; n = len(work); speeds = x['speeds']; k = len(speeds)
    ticks = x['ticks_per_unit']
    durations = [[max(1, math.ceil(w/s*ticks-1e-10)) for s in speeds] for w in work]
    mean_duration = np.mean([d for v in range(n) for p, d in enumerate(durations[v]) if x['demands'][v] <= x['capacities'][p]])
    data = ([z[2] for z in x['data_mb_sparse']] if 'data_mb_sparse' in x else [x['data_mb'][u][v] for u,v in x['edges']])
    invbw = np.mean([1/x['bandwidth'][p][q] for p in range(k) for q in range(k) if p != q])
    return dict(k=k, speed_ratio=max(speeds)/min(speeds), rate_ratio=max(x['cost_rates'])/min(x['cost_rates']),
                budget_eta=x.get('budget_eta'), resource_fraction=float(np.mean([d/c for d in x['demands'] for c in x['capacities']])),
                zero_edge_fraction=sum(d == 0 for d in data)/max(1,len(data)),
                estimated_CCR=float(np.mean(data)*invbw*ticks/mean_duration) if data else 0.)


def weighted_kmeans(x, weights, k, seed):
    rng = np.random.default_rng(seed)
    ids = [int(rng.choice(len(x), p=weights/weights.sum()))]
    for _ in range(1,k):
        dist = np.min(((x[:,None,:]-x[ids][None,:,:])**2).sum(2), axis=1)
        prob = weights * dist
        if prob.sum() <= 0:
            raise ValueError('Too few distinct feature vectors')
        ids.append(int(rng.choice(len(x), p=prob/prob.sum())))
    centers = x[ids].copy(); old = None
    for _ in range(200):
        dd = ((x[:,None,:]-centers[None,:,:])**2).sum(2)
        labels = dd.argmin(1)
        if old is not None and np.array_equal(old, labels): break
        old = labels.copy()
        for j in range(k):
            mask = labels == j
            centers[j] = np.average(x[mask], axis=0, weights=weights[mask]) if mask.any() else x[np.argmax(weights*dd.min(1))]
    dd = ((x[:,None,:]-centers[None,:,:])**2).sum(2)
    return dict(seed=seed, labels=dd.argmin(1), centers=centers, objective=float((dd.min(1)*weights).sum()))


def ari(a,b):
    def pairs(n): return n*(n-1)/2
    tab = collections.Counter(zip(map(int,a),map(int,b)))
    both = sum(pairs(n) for n in tab.values())
    aa = sum(pairs(n) for n in collections.Counter(a).values())
    bb = sum(pairs(n) for n in collections.Counter(b).values())
    expected = aa*bb/max(1,pairs(len(a))); den = .5*(aa+bb)-expected
    return (both-expected)/den if den else 1.


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT/'configs/structured_scheduling_v1.json').read_text())
    con = sqlite3.connect(f'file:{RUN / "data_v4/public.sqlite3"}?mode=ro', uri=True)
    members = [dict(zip(['key','source','role','base_id','topology','n','k','physical_sha256'], r)) for r in con.execute(
        'SELECT key,source,role,base_id,topology,n,k,physical_sha FROM records WHERE role="dev" ORDER BY key')]
    assert len(members) == 2662
    rows = []; physical_rows = []; public_seen = {}; identities = {}; components = {}
    for key, source, base, topo, blob in con.execute('SELECT key,source,base_id,topology,raw FROM records WHERE role="dev" AND source!="synthetic" ORDER BY key'):
        raw = json.loads(zlib.decompress(blob)); x = raw['instance_sparse']; prov = raw['provenance']
        physical_rows.append(dict(key=key,base_id=base,source=source,**physical(x)))
        if base in public_seen:
            assert public_seen[base] == (x['edges'], x['workload'])
            continue
        public_seen[base] = (x['edges'], x['workload'])
        identity = prov.get('source_identity',{})
        main = identity.get('training_supported_engine_application',True)
        row = dict(base_id=base, source=source, role='dev', family=x['family'], topology=topo, main_panel=main,
                   **features(x['n'],x['edges'],x['workload']))
        rows.append(row); identities[base] = identity; components[base] = prov.get('components',[])
    assert sum(r['source']=='wfinstances' for r in rows)==25
    assert sum(r['source']=='tpch_composite' for r in rows)==200
    print('Public inputs profiled: 25 WF + 200 TPC bases; 1,800 physical configs',flush=True)
    with gzip.open(ROOT/'data/structured_scheduling_v1/BASES.jsonl.gz','rt') as f:
        bases = [json.loads(l) for l in f if l.strip()]
    checks = 0
    for b in bases:
        if b['split'] not in ['train','val']: continue
        insts = list(instances(b,cfg)); x = insts[0][1]
        rows.append(dict(base_id=b['base_id'],source='synthetic',role='train' if b['split']=='train' else 'dev',
                         family=b['family'],topology=b['topology_fingerprint'],main_panel=True,
                         **features(b['n'],x['edges'],x['workload'])))
        if checks < 5:
            packed = con.execute('SELECT raw FROM records WHERE base_id=? ORDER BY key LIMIT 1',(b['base_id'],)).fetchone()[0]
            original = json.loads(zlib.decompress(packed))['input']
            assert [[u,v] for u,v,_ in original['network']['edges']] == x['edges']
            assert original['tasks']['workload_seconds'] == x['workload']
            checks += 1
        seen_budgets = set()
        for key,inst,extra in insts:
            if inst['budget'] in seen_budgets:continue
            seen_budgets.add(inst['budget'])
            inst['budget_eta'] = extra['eta']
            physical_rows.append(dict(key=key,base_id=b['base_id'],source='synthetic_'+b['split'],**physical(inst)))
        if b['split']=='train' and (b['index']+1)%2000 == 0: print(f"Synthetic TRAIN profiled: {b['index']+1}/10000",flush=True)
    con.close()
    for r in rows:
        assert r['edges'] <= 4*r['n'] if r['source']=='synthetic' else True
    targets = [r for r in rows if r['source']!='synthetic']
    train = [r for r in rows if r['source']=='synthetic' and r['role']=='train']
    tx = np.array([[r[f] for f in FEATURES] for r in targets])
    weights = np.array([.5/25 if r['source']=='wfinstances' else .5/200 for r in targets])
    mean = np.average(tx,axis=0,weights=weights)
    scale = np.sqrt(np.average((tx-mean)**2,axis=0,weights=weights)); scale = np.maximum(scale,1e-8)
    tx = (tx-mean)/scale
    allx = (np.array([[r[f] for f in FEATURES] for r in rows])-mean)/scale
    sx = (np.array([[r[f] for f in FEATURES] for r in train])-mean)/scale
    fit_stats = {}; fit = None
    for k in [4,6,8]:
        fits = [weighted_kmeans(tx,weights,k,seed) for seed in range(28,33)]
        best = min(fits,key=lambda r:r['objective'])
        fit_stats[str(k)] = dict(best_seed=best['seed'],weighted_sse=best['objective'],
                                 ARI_to_best=[ari(best['labels'],f['labels']) for f in fits],
                                 target_counts=dict(collections.Counter(map(int,best['labels']))))
        if k==6:fit=best
    assert fit is not None
    assigned = ((allx[:,None,:]-fit['centers'][None,:,:])**2).sum(2).argmin(1)
    for r,label in zip(rows,assigned):r['cluster']=int(label)
    distances = np.sqrt(np.maximum(0,(tx*tx).sum(1)[:,None]+(sx*sx).sum(1)[None,:]-2*tx@sx.T))
    for i,r in enumerate(targets):
        j = int(distances[i].argmin());r['nearest_synthetic']=train[j]['base_id'];r['nearest_shape_distance']=float(distances[i,j])
        r['synthetic_n_support']=r['n'] in cfg['tasks']
        r['above_synthetic_max_n']=r['n']>max(cfg['tasks'])
        r['components_outside_synthetic_range']=not min(t['components'] for t in train)<=r['components']<=max(t['components'] for t in train)
    summaries = {}
    groups = {'synthetic_train':train,'synthetic_dev':[r for r in rows if r['source']=='synthetic' and r['role']=='dev'],
              'WF_all': [r for r in targets if r['source']=='wfinstances'],
              'WF_main':[r for r in targets if r['source']=='wfinstances' and r['main_panel']],
              'TPC':[r for r in targets if r['source']=='tpch_composite']}
    measurements = ['n','edges_per_node','components','source_fraction','sink_fraction','fork_fraction','join_fraction',
                    'max_out_fraction','max_in_fraction','depth','depth_fraction','layer_width_fraction',
                    'reachability_fraction','work_cv','work_p90_over_median','critical_path_work_fraction']
    for name,rs in groups.items():
        summaries[name]=dict(bases=len(rs),features={f:quantiles([r[f] for r in rs]) for f in measurements},
                            cluster_counts=dict(collections.Counter(r['cluster'] for r in rs)))
        if rs[0]['source']!='synthetic':
            summaries[name].update(above_synthetic_max_n=sum(r['above_synthetic_max_n'] for r in rs),
                                   components_outside_synthetic_range=sum(r['components_outside_synthetic_range'] for r in rs),
                                   nearest_shape_distance=quantiles([r['nearest_shape_distance'] for r in rs]))
    clusters=[]
    for j in range(6):
        rr=[r for r in targets if r['cluster']==j]
        center=fit['centers'][j]
        representative=min(rr,key=lambda r:sum((((r[f]-mean[i])/scale[i])-center[i])**2 for i,f in enumerate(FEATURES)))
        clusters.append(dict(cluster=j,target_counts=dict(collections.Counter(r['source'] for r in rr)),
                             synthetic_count=sum(r['cluster']==j for r in train),representative=representative['base_id'],
                             median={f:float(np.median([r[f] for r in rr])) for f in measurements},
                             n_range=[min(r['n'] for r in rr),max(r['n'] for r in rr)]))
    phys={}
    for source in sorted(set(r['source'] for r in physical_rows)):
        rr=[r for r in physical_rows if r['source']==source]
        phys[source]=dict(configs=len(rr),**{f:quantiles([r[f] for r in rr if r[f] is not None]) for f in
             ['k','speed_ratio','rate_ratio','budget_eta','resource_fraction','zero_edge_fraction','estimated_CCR']})
    fieldnames=sorted(set().union(*(r.keys() for r in rows)))
    for name,rr,fields in [('graph_features.csv',rows,fieldnames),('physical_features.csv',physical_rows,list(physical_rows[0]))]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rr)
    (OUT/'members.json').write_text(json.dumps(dict(configs=members,public_identities=identities,TPC_components=components),indent=2)+'\n')
    evidence=dict(protocol='protocols/target_structure_audit_20261031.md',scope='Exploratory target-input-informed synthesis design; no training/quality evaluation',
                  input_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'configs/structured_scheduling_v1.json',
                    ROOT/'data/structured_scheduling_v1/BASES.jsonl.gz',RUN/'data_v4/MANIFEST.json',Path(__file__)]},
                  manifest_database_sha256=json.loads((RUN/'data_v4/MANIFEST.json').read_text())['database_sha256'],
                  membership_sha256=sha(OUT/'members.json'),reconstruction_spot_checks=checks,
                  features=FEATURES,standardization_mean=mean.tolist(),standardization_scale=scale.tolist(),
                  clustering=fit_stats,cluster_summaries=clusters,source_summaries=summaries,physical_summaries=phys)
    (OUT/'report.json').write_text(json.dumps(evidence,indent=2)+'\n')
    plot(groups,clusters,phys)
    print(json.dumps(dict(output=str(OUT),clusters=clusters,
                         target_coverage={k:{z:v for z,v in d.items() if z in ['bases','above_synthetic_max_n','components_outside_synthetic_range','nearest_shape_distance']} for k,d in summaries.items()}),indent=2),flush=True)


def plot(groups,clusters,phys):
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'dag-performance-mpl-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'pdf.fonttype':42,'savefig.bbox':'tight'})
    names=['synthetic_train','synthetic_dev','WF_all','TPC']; labels=['Synthetic TRAIN','Synthetic DEV','WF (25)','TPC (200)']
    colors=['#009E73','#56B4E9','#E69F00','#0072B2']
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,(f,title) in zip(axes.flat,[('n','Task count'),('components','Weakly connected components'),
         ('depth_fraction','Longest path nodes / total nodes'),('max_out_fraction','Maximum out-degree / nodes'),
         ('work_cv','Task workload coefficient of variation'),('critical_path_work_fraction','Longest-work path / total work')]):
        bp=ax.boxplot([[r[f] for r in groups[s]] for s in names],patch_artist=True,showfliers=False)
        for b,c in zip(bp['boxes'],colors):b.set_facecolor(c);b.set_alpha(.7)
        ax.set_xticks(range(1,5),labels,rotation=18,ha='right',fontsize=8);ax.set_title(title)
    fig.suptitle('Input distributions: full original synthetic TRAIN vs fixed evaluation workloads\nBoxes: median/IQR; whiskers: 1.5 IQR; outliers hidden (exact ranges in CSV)',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(OUT/f'input_distributions.{ext}',dpi=300)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,3.6),layout='constrained')
    for i,name in enumerate(['synthetic_train','WF_all','TPC']):
        rs=groups[name];counts=[sum(r['cluster']==j for r in rs)/len(rs)*100 for j in range(6)]
        ax.bar(np.arange(6)+(i-1)*.25,counts,.23,label={'synthetic_train':'Synthetic TRAIN','WF_all':'WF','TPC':'TPC'}[name],
               color=[colors[0],colors[2],colors[3]][i])
    ax.set_xticks(range(6),[f'C{j}' for j in range(6)]);ax.set_ylabel('Fraction of source base graphs (%)')
    ax.set_title('Descriptive shape clusters (k=6; WF/TPC equally weighted for fitting)');ax.legend()
    for ext in ['png','pdf']:fig.savefig(OUT/f'cluster_coverage.{ext}',dpi=300)
    plt.close(fig)


if __name__=='__main__':main()
