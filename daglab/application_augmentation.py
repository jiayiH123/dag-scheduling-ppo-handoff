"""Input-statistic-informed synthetic DAGs; never copy a public adjacency."""
import collections
import math
import random
import statistics
from .offline_data import topological_order


def roles(n,edges):
    parents=[[] for _ in range(n)];children=[[] for _ in range(n)]
    for u,v in edges:parents[v].append(u);children[u].append(v)
    tags=['root' if not parents[v] else 'join' if len(parents[v])>=8 else 'sink' if not children[v] else 'internal' for v in range(n)]
    return parents,children,tags


def log_stats(values):
    values=[max(.004,float(x)) for x in values]
    logs=[math.log(v) for v in values]
    return dict(mu=statistics.mean(logs),sigma=max(.12,statistics.pstdev(logs)),
                minimum=min(values),maximum=max(values),count=len(values))


def describe(base,application):
    n=base['n'];edges=base['edges'];parents,children,tags=roles(n,edges)
    levels=[0]*n
    for v in topological_order(n,edges):levels[v]=1+max((levels[u] for u in parents[v]),default=0)
    counts=collections.Counter(levels);work=[w/1000 for w in base['workload']]
    sizes=base['edge_data_mb'];positive=[v for v in sizes if v>0]
    log_sizes=[math.log(v) for v in positive]
    return dict(application=application,source_base_id=base['base_id'],source_sha256=base['source_sha256'],
        n=n,depth=max(levels),level_widths=[counts[i] for i in range(1,max(levels)+1)],
        edge_density=len(edges)/n,join_fractions=[d/n for d in sorted(map(len,parents),reverse=True)[:2]],
        workload_global=log_stats(work),
        workload_roles={tag:log_stats([w for w,t in zip(work,tags) if t==tag]) for tag in set(tags)},
        traffic_zero_probability=sum(v==0 for v in sizes)/len(sizes),
        traffic_log_mu=statistics.mean(log_sizes) if log_sizes else 0.,
        traffic_log_sigma=max(.12,statistics.pstdev(log_sizes)) if log_sizes else .12,
        traffic_max=max(positive,default=0.),
        origin='Descriptive public INPUT statistics only; graph/values generated independently')


def sample_graph(profile,n,k,rng):
    depth=max(2,min(n//2,round(profile['depth']*rng.uniform(.85,1.15))))
    source=profile['level_widths'];weights=[]
    for i in range(depth):
        x=i*(len(source)-1)/(depth-1);lo=int(x);hi=min(lo+1,len(source)-1)
        weights.append(((1-(x-lo))*source[lo]+(x-lo)*source[hi])*rng.uniform(.75,1.25))
    widths=[1]*depth
    for level in rng.choices(range(depth),weights=weights,k=n-depth):widths[level]+=1
    layers=[];offset=0
    for size in widths:layers.append(list(range(offset,offset+size)));offset+=size
    levels={v:i for i,group in enumerate(layers) for v in group}
    edges=set()
    for i in range(1,depth):
        for v in layers[i]:edges.add((rng.choice(layers[i-1]),v))
    cap=min(4*n,4096-(1+n+k+k*k+2*n*k))
    if len(edges)>cap:raise ValueError('numeric-position edge budget')
    # Strong gather nodes plus random ordinary dependencies; no source edges read.
    hubs=[v for group in reversed(layers[1:]) for v in group]
    for fraction,v in zip(profile['join_fractions'],hubs):
        prior=[u for group in layers[:levels[v]] for u in group]
        desired=min(len(prior),max(1,round(fraction*n*rng.uniform(.8,1.1))))
        for u in rng.sample(prior,len(prior)):
            if sum(w==v for _,w in edges)>=desired or len(edges)>=cap:break
            edges.add((u,v))
    target=min(cap,max(len(edges),round(profile['edge_density']*n*rng.uniform(.85,1.15))))
    for _ in range(max(1,target-len(edges))*60):
        if len(edges)>=target:break
        level=rng.randrange(1,depth);v=rng.choice(layers[level])
        parent_level=level-1 if rng.random()<.75 else rng.randrange(level)
        edges.add((rng.choice(layers[parent_level]),v))
    permutation=list(range(n));rng.shuffle(permutation)
    edges=sorted([permutation[u],permutation[v]] for u,v in edges)
    topological_order(n,edges)
    assert len(edges)<=cap
    return edges


def sample_instance(base,profile,seed,pressure,eta,network_profile):
    """Simulation in seconds/millisecond ticks, near existing WF parameter ranges."""
    r=random.Random(seed);n=base['n'];k=base['k'];edges=base['edges']
    _,_,tags=roles(n,edges);work=[];global_scale=math.exp(r.uniform(-.15,.15))
    for tag in tags:
        stats=profile['workload_roles'].get(tag,profile['workload_global'])
        value=r.lognormvariate(stats['mu'],stats['sigma']*r.uniform(.85,1.15))*global_scale
        work.append(round(min(2*stats['maximum'],max(.004,value)),6))
    weights={'light':[.7,.2,.08,.02],'mixed':[.25]*4,'heavy':[.05,.15,.3,.5]}[pressure]
    demands=r.choices([1,2,3,4],weights=weights,k=n);caps=r.choices([4,8,12],k=k)
    pool_types=list(range(3))*(k//3);r.shuffle(pool_types)
    speeds=[round([1.,2.,4.][t]*r.uniform(.985,1.015),6) for t in pool_types]
    assert len(set(speeds))==len(speeds)
    rates=[[1,4,16][t] for t in pool_types]
    bw=[[0.]*k for _ in range(k)];bw3=[[0,10,20],[10,0,40],[20,40,0]]
    factor=1 if network_profile==0 else 10
    for p in range(k):
        for q in range(p+1,k):
            bw[p][q]=bw[q][p]=round((bw3[pool_types[p]][pool_types[q]] or 10)*factor*r.uniform(.9,1.1),6)
    data=[[0.]*n for _ in range(n)]
    for u,v in edges:
        if r.random()>=profile['traffic_zero_probability']:
            data[u][v]=min(2*profile['traffic_max'],math.exp(r.gauss(profile['traffic_log_mu'],profile['traffic_log_sigma'])))
    durations=[[max(1,math.ceil(w/s*1000-1e-10)) for s in speeds] for w in work]
    costs=[[d*c for d,c in zip(ds,rates)] for ds in durations]
    bmin=sum(min(cs) for cs in costs);bmax=sum(max(cs) for cs in costs)
    fastest=[min(range(k),key=lambda p:(durations[v][p],costs[v][p],p)) for v in range(n)]
    bfast=sum(costs[v][p] for v,p in enumerate(fastest));budget=bmin+int(eta*(bfast-bmin))
    mean_compute=sum(map(sum,durations))/(n*k)
    actual_ccr=sum(math.ceil(data[u][v]/bw[p][q]*1000-1e-10) for u,v in edges for p in range(k) for q in range(k) if p!=q)/(len(edges)*k*(k-1)*mean_compute)
    inst=dict(n=n,edges=edges,workload=work,demands=demands,capacities=caps,speeds=speeds,
        cost_rates=rates,bandwidth=bw,data_mb=data,latency=0,model='fixed',overlap=True,ticks_per_unit=1000,
        budget=budget,bmin=bmin,bfast=bfast,budget_mode='eta'+str(eta),budget_eta=eta,
        latent_id=base['base_id'],family=base['family'],resource_mode='pool',target_ccr=None,
        seed=seed%(2**32),source='independent-application-profile-synthetic-v1')
    return inst,dict(eta=eta,maximum=bmax,actual_ccr=actual_ccr,price_gamma=2.)
