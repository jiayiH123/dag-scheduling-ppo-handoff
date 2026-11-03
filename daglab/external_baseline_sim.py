"""Explicit paper adaptations on the unchanged budget/cumulative SGS physics."""
import hashlib
import math
import random
import time
from fractions import Fraction
import numpy as np
from .reference import core
from prepared_sgs import PreparedSGS, find_slot


class Problem:
    def __init__(self, inst):
        self.inst=inst;self.engine=PreparedSGS(inst);self.info=self.engine.info
        self.n,self.k=self.engine.n,self.engine.k
        self.order=sorted(range(self.n),key=lambda v:(-self.info['ranks'][v],v))
        self.scale=sum(min(self.info['duration'][v][p] for p in self.info['eligible'][v]) for v in range(self.n))/self.n
        self.eligible=np.array([[p in self.info['eligible'][v] for p in range(self.k)] for v in range(self.n)])
        self.costs=np.array(self.info['costs']);self.duration=np.array(self.info['duration'])
        self.budget=inst['budget'];self.parents=self.info['parents'];self.children=self.info['children']
        levels=[0]*self.n
        for v in self.info['order']:levels[v]=max((levels[u]+1 for u in self.parents[v]),default=0)
        self.levels=[[v for v in range(self.n) if levels[v]==l] for l in range(max(levels)+1)]
        self.level_of=levels

    def state(self,mapping,order=None):
        """Same insertion SGS, without repeated full validation inside a search."""
        order=self.order if order is None else order
        calendars=[[] for _ in range(self.k)];starts=[0]*self.n;ends=[0]*self.n
        for v in order:
            p=mapping[v];duration=int(self.duration[v,p])
            release=max((ends[u]+self.info['comm'][u,v][mapping[u]][p] for u in self.parents[v]),default=0)
            start=find_slot(release,duration,self.inst['demands'][v],self.inst['capacities'][p],calendars[p])
            starts[v]=start;ends[v]=start+duration;calendars[p].append((start,ends[v],self.inst['demands'][v]))
        return dict(C=max(ends),cost=sum(int(self.costs[v,p]) for v,p in enumerate(mapping)),starts=starts,ends=ends)

    def finish(self,mapping,order=None):
        order=self.order if order is None else order
        solution=self.engine.evaluate(mapping,order,materialize=True)
        core.validate(self.inst,solution)
        return dict(C=solution['makespan'],cost=solution['cost'],feasible=True,
            mapping=list(mapping),order=list(order),ends=[t['end'] for t in solution['tasks']])

    def schedule(self,order=None,rule='EFT'):
        info=self.info;n,k=self.n,self.k
        if order is None:order=list(self.order)
        if rule.startswith('BUDA'):
            coefficient=1.
            if rule=='BUDA_unitbalanced':
                coefficient=float(self.duration[self.eligible].mean()/max(1.,self.costs[self.eligible].mean()))
            table=np.zeros((n,k))
            for v in reversed(info['order']):
                for p in info['eligible'][v]:
                    future=max((table[c,p]+sum(info['comm'][v,c][p])/k for c in self.children[v]),default=0.)
                    table[v,p]=info['duration'][v][p]+coefficient*info['costs'][v][p]+future
            ranks=[sum(table[v,p] for p in info['eligible'][v])/len(info['eligible'][v]) for v in range(n)]
            # Ready-list priority instead of relying on monotone ranks under eligibility.
            order=[];done=set()
            while len(order)<n:
                ready=[v for v in range(n) if v not in done and all(u in done for u in self.parents[v])]
                v=min(ready,key=lambda v:(-ranks[v],v));order.append(v);done.add(v)
        averages=[Fraction(sum(info['costs'][v][p] for p in info['eligible'][v]),len(info['eligible'][v])) for v in range(n)]
        remaining_avg=sum(averages);reserve=sum(info['min_cost']);ledger=Fraction(reserve)
        mapping=[-1]*n;ends=[0]*n;starts=[0]*n;cal=[[] for _ in range(k)];spent=0
        stats=dict(fallbacks=0,reserve_masked=0)
        for v in order:
            reserve-=info['min_cost'][v];candidates=[]
            for p in info['eligible'][v]:
                cost=info['costs'][v][p]
                if spent+cost+reserve>self.budget:stats['reserve_masked']+=1;continue
                release=max((ends[u]+info['comm'][u,v][mapping[u]][p] for u in self.parents[v]),default=0)
                start=find_slot(release,info['duration'][v][p],self.inst['demands'][v],self.inst['capacities'][p],cal[p])
                candidates.append(dict(p=p,start=start,end=start+info['duration'][v][p],cost=cost))
            if not candidates:raise AssertionError('reserve must leave feasible suffix')
            eft=lambda c:(c['end'],c['start'],c['p'])
            if rule=='BHEFT':
                sab=self.budget-spent-remaining_avg
                ctb=averages[v]+max(sab,0)*averages[v]/remaining_avg
                affordable=[c for c in candidates if c['cost']<=ctb]
                if affordable:chosen=min(affordable,key=eft)
                else:
                    stats['fallbacks']+=1
                    chosen=min(candidates,key=eft if sab>=0 else lambda c:(c['cost'],*eft(c)))
            elif rule.startswith('BUDA'):
                ledger-=averages[v] if rule=='BUDA_printedledger' else info['min_cost'][v]
                atb=self.budget-spent-ledger
                best=min(candidates,key=eft)
                affordable=[c for c in candidates if c['cost']<=best['cost'] and c['cost']<=atb]
                if not affordable:
                    stats['fallbacks']+=1;affordable=[min(candidates,key=lambda c:(c['cost'],*eft(c)))]
                chosen=min(affordable,key=lambda c:(c['end']+table[v,c['p']],*eft(c)))
            else:chosen=min(candidates,key=eft)
            p=chosen['p'];mapping[v]=p;starts[v]=chosen['start'];ends[v]=chosen['end'];spent+=chosen['cost']
            cal[p].append((starts[v],ends[v],self.inst['demands'][v]));remaining_avg-=averages[v]
        return dict(mapping=mapping,order=order,C=max(ends),cost=spent,starts=starts,ends=ends,stats=stats)

    def random_mapping(self,seed):
        rng=random.Random(seed);remaining=sum(self.info['min_cost']);spent=0;mapping=[-1]*self.n
        for v in self.order:
            remaining-=self.info['min_cost'][v]
            feasible=[p for p in self.info['eligible'][v] if spent+int(self.costs[v,p])+remaining<=self.budget]
            p=rng.choice(feasible);mapping[v]=p;spent+=int(self.costs[v,p])
        return mapping

    def priority_features(self):
        inst,info=self.inst,self.info
        mean_fee=max(1.,float(self.costs[self.eligible].mean()));rank=max(info['ranks'])
        x=[]
        for v in range(self.n):
            traffic=sum(inst['data_mb'][v])+sum(inst['data_mb'][u][v] for u in self.parents[v])
            x.append([math.log1p(inst['workload'][v]),math.log1p(traffic),
                inst['demands'][v]/max(inst['capacities']),float(self.duration[v].min())/self.scale,
                float(self.duration[v].mean())/self.scale,float(self.costs[v].min())/mean_fee,
                float(self.costs[v].mean())/mean_fee,info['ranks'][v]/rank,
                self.budget/max(1,sum(info['min_cost'])),self.k/6.])
        a=np.asarray(x);a=(a-a.mean(0))/(a.std(0)+1e-8)
        return dict(x=a.tolist(),edges=[list(e) for e in inst['edges']],n=self.n)

    def priority_order(self,priorities):
        chosen=[];done=set();decisions=[]
        while len(chosen)<self.n:
            ready=[v for v in range(self.n) if v not in done and all(u in done for u in self.parents[v])]
            v=min(ready,key=lambda v:(-priorities[v],v));decisions.append(dict(available=ready,action=v))
            chosen.append(v);done.add(v)
        return chosen,decisions

    def relocation_graph(self,mapping,state,last_task=None):
        n,k=self.n,self.k;inst=self.inst;info=self.info
        current_cost=state['cost'];pairs=np.arange(n*k).reshape(n,k)
        mask=self.eligible & (current_cost-self.costs[np.arange(n),mapping,None]+self.costs<=self.budget)
        mask[np.arange(n),mapping]=False
        if last_task is not None:
            exclusion=mask.copy();exclusion[last_task,:]=False
            if exclusion.any():mask=exclusion
        # Deterministic current path indicator, not accumulated visit counts.
        critical=np.zeros(n);v=max(range(n),key=lambda v:state['ends'][v])
        while True:
            critical[v]=1
            if not self.parents[v]:break
            v=max(self.parents[v],key=lambda u:state['ends'][u]+info['comm'][u,v][mapping[u]][mapping[v]])
        node=[];src=[];dst=[];edge=[]
        for v in range(n):
            for p in range(k):
                release=max((state['ends'][u]+info['comm'][u,v][mapping[u]][p] for u in self.parents[v]),default=0)
                potential=0 if p==mapping[v] else release-state['starts'][v]
                node.append([inst['workload'][v],1/inst['speeds'][p],info['duration'][v][p],potential,critical[v],
                    info['costs'][v][p],inst['demands'][v]/inst['capacities'][p],float(p==mapping[v]),
                    (self.budget-current_cost)/max(1,info['costs'][v][p])])
        for u,v in inst['edges']:
            pairs_uv=[(mapping[u],p) for p in range(k) if self.eligible[v,p]]
            pairs_uv += [(p,mapping[v]) for p in range(k) if self.eligible[u,p] and p!=mapping[u]]
            for p,q in pairs_uv:
                src.append(u*k+p);dst.append(v*k+q)
                edge.append([inst['data_mb'][u][v],0.,info['comm'][u,v][p][q],
                    0. if p==q else 1/inst['bandwidth'][p][q],critical[u]*critical[v]])
        def normalize(a,width):
            a=np.asarray(a,dtype=float).reshape(-1,width)
            if not len(a):return a.tolist()
            low=a.min(0);span=a.max(0)-low
            return np.where(span>0,2*(a-low)/np.maximum(span,1e-12)-1,0.).tolist()
        return dict(x=normalize(node,9),src=src,dst=dst,edge=normalize(edge,5),
            levels=[[v*k+p for v in level for p in range(k)] for level in self.levels],
            mask=mask.flatten().tolist(),n=n,k=k)


def stable_seed(key,offset=0):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')+offset
