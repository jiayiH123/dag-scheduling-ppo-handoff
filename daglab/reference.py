"""Exact extracted generator/features/teacher functions; source hashes in SOURCE_MANIFEST.json."""
from pathlib import Path
import sys, hashlib, itertools, random, time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
import budget_core as core
import legacy_comm_core as legacy
from cached_budget_decoder import CachedBudgetDecoder
from search_controls import portfolio,mapping_of
from budget_controls import mslbl
FAMILIES=['pipeline','forkjoin','costly_downstream','random_packing']


def base(family, n, resource, seed, ccr):
    latent = f'{family}:{n}:{resource}:{seed}'
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(latent.encode()).digest()[:8], 'little'))
    work = (4*rng.integers(5, 61, n)).tolist()
    if family == 'pipeline':
        width = 4
        edges = [[v, v+width] for v in range(n-width)]
        edges += [[v, v+width+1] for v in range(n-width-1) if rng.random() < .25]
    elif family == 'forkjoin':
        edges = [[0, v] for v in range(1, n-1)]+[[v, n-1] for v in range(1, n-1)]
    elif family == 'costly_downstream':
        groups = [list(range(0, n-1, 2)), list(range(1, n-1, 2))]
        edges = []
        for group in groups:
            edges += [[u, v] for u, v in zip(group, group[1:])]+[[group[-1], n-1]]
        for v in groups[0]:
            work[v] *= 2
        work[-1] *= 3
    elif family == 'random_packing':
        es = {(u, v) for u in range(n) for v in range(u+1, n) if rng.random() < 2.5/n}
        for u in range(n-1):
            if not any(a == u for a, b in es):
                es.add((u, int(rng.integers(u+1, n))))
        edges = [list(e) for e in sorted(es)]
    else:
        raise ValueError(family)
    demands = [1]*n if resource == 'serial' else rng.integers(1, 5, n).tolist()
    capacities = [1, 1, 1] if resource == 'serial' else [3, 4, 4]
    speeds = [1, 2, 4]
    bandwidth = np.zeros((3, 3))
    for p, q in itertools.combinations(range(3), 2):
        bandwidth[p, q] = bandwidth[q, p] = int(rng.choice([2, 4, 8]))
    factors = rng.uniform(.5, 1.5, len(edges))
    if family == 'costly_downstream':
        for i, (u, v) in enumerate(edges):
            if v == n-1:
                factors[i] *= 3
    eligible = [[p for p in range(3) if demands[v] <= capacities[p]] for v in range(n)]
    average = float(np.mean([work[v]/speeds[p] for v in range(n) for p in eligible[v]]))
    unit_transfer = float(np.mean([f/bandwidth[p, q] for f in factors for p in range(3)
                                  for q in range(3) if p != q]))
    data = np.zeros((n, n))
    for (u, v), f in zip(edges, factors):
        data[u, v] = float(ccr*average/unit_transfer*f)
    inst = {'source': 'synthetic-large/'+family, 'family': family, 'seed': seed,
            'latent_id': hashlib.sha256(latent.encode()).hexdigest()[:20], 'n': n,
            'target_ccr': ccr, 'resource_mode': resource, 'speed_ratio': 4,
            'model': 'fixed', 'overlap': True, 'ticks_per_unit': 1,
            'edges': edges, 'workload': work, 'demands': demands, 'capacities': capacities,
            'speeds': speeds, 'bandwidth': bandwidth.tolist(), 'data_mb': data.tolist(),
            'latency': 0, 'cost_rates': [1, 4, 16], 'budget': None,
            'units': 'synthetic ticks and additive costs; not measured hardware'}
    return inst

def add_budget(base, name):
    info=core.prepare(base)
    bmin=sum(info["min_cost"])
    fastest=[min(info["eligible"][v],key=lambda p:(info["duration"][v][p],info["costs"][v][p],p)) for v in range(info["n"])]
    bfast=sum(info["costs"][v][p] for v,p in enumerate(fastest))
    eta={"nonbinding":1,"tight25":.25,"tight60":.6}[name]
    base.update(budget=bmin+int(eta*(bfast-bmin)),budget_mode=name,bmin=bmin,bfast=bfast,budget_degenerate=bmin==bfast)
    return base

def graph_features(inst):
    started=time.perf_counter()
    info=core.prepare(inst);n,k=info["n"],info["k"]
    mean_work=sum(inst["workload"])/n
    mean_compute=sum(min(t) for t in info["duration"])/n
    mean_cost=sum(info["min_cost"])/n
    max_speed=max(inst["speeds"]);max_cap=max(inst["capacities"]);max_rate=max(inst["cost_rates"])
    eta=(inst["budget"]-inst["bmin"])/max(1,inst["bfast"]-inst["bmin"])
    x=[];eligible=[]
    for v in range(n):
        for p in range(k):
            x.append([inst["workload"][v]/mean_work,inst["demands"][v]/max_cap,
                      len(info["parents"][v])/n,len(info["children"][v])/n,
                      info["duration"][v][p]/mean_compute,info["costs"][v][p]/max(1,mean_cost),
                      inst["speeds"][p]/max_speed,inst["capacities"][p]/max_cap,
                      inst["cost_rates"][p]/max_rate,inst["demands"][v]/inst["capacities"][p],
                      info["ranks"][v]/max(info["ranks"]),eta])
            eligible.append(p in info["eligible"][v])
    src,dst,weight=[],[],[]
    for u,v in inst["edges"]:
        for p in info["eligible"][u]:
            for q in info["eligible"][v]:
                src.append(u*k+p);dst.append(v*k+q)
                weight.append(1/(1+info["comm"][u,v][p][q]/mean_compute))
    graph={"x":x,"eligible":eligible,"src":src,"dst":dst,"edge_weight":weight,"n":n,"k":k}
    return graph,time.perf_counter()-started

def teacher(inst, decoder):
    started = time.perf_counter()
    seeds, _ = portfolio(inst)
    engine = decoder.engine
    candidates = [mapping_of(s) for s in seeds]
    values = [engine.evaluate(m, decoder.order) for m in candidates]
    best = min(range(len(values)), key=lambda i: values[i])
    mapping, value = candidates[best], values[best]
    rng = random.Random(inst['seed']+3917)
    feasible, attempted = 0, 0
    for _ in range(64):
        attempted += 1
        changed = list(mapping)
        for v in rng.sample(range(engine.n), rng.choice([1, 2])):
            changed[v] = rng.choice(engine.info['eligible'][v])
        if not engine.feasible(changed):
            continue
        feasible += 1
        candidate = engine.evaluate(changed, decoder.order)
        if candidate < value:
            mapping, value = changed, candidate
    solution = engine.evaluate(mapping, decoder.order, materialize=True)
    solution['algorithm'] = 'Offline-Portfolio-Capped-Mapping-Teacher'
    solution['search'] = {'attempted': attempted, 'feasible': feasible,
                          'simulations': engine.evaluations, 'converged': False}
    forced = decoder.rollout([[0., 0.]]*(engine.n*engine.k), 'initial',
                             forced_mapping=mapping, materialize=True)
    if forced['schedule']['tasks'] != solution['tasks']:
        raise AssertionError('teacher replay mismatch')
    return solution, forced['decisions'], time.perf_counter()-started

def random_policy_rollout(inst, logits, mode, sample, seed):
    started=time.perf_counter();info=core.prepare(inst);state=legacy.blank_state(info["k"])
    rng=np.random.default_rng(seed);decisions=[]
    while len(state["tasks"])<info["n"]:
        ready=[v for v in range(info["n"]) if v not in state["tasks"] and all(u in state["tasks"] for u in info["parents"][v])]
        choices={}
        for v in ready:
            candidates=[]
            for p in info["eligible"][v]:
                if core.reserve_allows(info,state,v,p):
                    c=legacy.placement(inst,info,state,v,p)
                    candidates.append((c["task"]["end"],c["task"]["start"],p,c))
                    if mode=="joint":choices[v*info["k"]+p]=(v,p,c)
            if mode=="order" and candidates:
                _,_,p,c=min(candidates,key=lambda t:t[:3]);choices[v]=(v,p,c)
        available=sorted(choices)
        vals=np.asarray([logits[a] for a in available]);prob=np.exp(vals-vals.max());prob/=prob.sum()
        selected=int(rng.choice(len(available),p=prob)) if sample else int(np.argmax(vals))
        action=available[selected];v,p,c=choices[action]
        decisions.append({"available":available,"selected":selected,"action":action})
        state["tasks"][v]=c["task"];state["calendar"]=c["calendar"]
        state["communications"].extend(c["communications"]);state["decision_order"].append(v)
    sol=legacy.finish_schedule(inst,state,"RL-"+mode)
    sol["cost"]=sum(info["costs"][t["node"]][t["device"]] for t in sol["tasks"])
    core.validate(inst,sol)
    sol["runtime_seconds"]=time.perf_counter()-started
    return {"decisions":decisions,"makespan":sol["makespan"],"cost":sol["cost"],"schedule":sol,
            "decoder_seconds":sol["runtime_seconds"]}
