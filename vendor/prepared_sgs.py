"""Prepared fixed-lag serial schedule generation; saved schedules checked independently.

No policy inference, no search, no change to the scheduling problem. One prepared
instance can be reused across many candidate mappings/orders. The implementation
uses tuple calendars and a sweep to find the same earliest slot as legacy SGS.
"""
from copy import deepcopy
import math
import time
import budget_core as core


def fits(start,duration,demand,capacity,calendar):
    end=start+duration;occupied=0;events={}
    for s,e,d in calendar:
        if s<=start<e:occupied+=d
        if start<s<end:events[s]=events.get(s,0)+d
        if start<e<end:events[e]=events.get(e,0)-d
    if occupied+demand>capacity:return False
    for t in sorted(events):
        occupied+=events[t]
        if occupied+demand>capacity:return False
    return True


def find_slot(release,duration,demand,capacity,calendar):
    if demand>capacity:raise ValueError("ineligible placement")
    for start in sorted({release}|{e for _,e,_ in calendar if e>=release}):
        if fits(start,duration,demand,capacity,calendar):return start
    raise AssertionError("last calendar end must be feasible")


class PreparedSGS:
    def __init__(self,instance):
        started=time.perf_counter();self.instance=deepcopy(instance);self.info=core.prepare(self.instance)
        self.n,self.k=self.info["n"],self.info["k"];self.order_cache=set();self.evaluations=0
        self.compilation_seconds=time.perf_counter()-started

    def cost(self,mapping):
        return sum(self.info["costs"][v][p] for v,p in enumerate(mapping))

    def feasible(self,mapping):
        return len(mapping)==self.n and all(p in self.info["eligible"][v] for v,p in enumerate(mapping)) and (
            self.info["budget"] is None or self.cost(mapping)<=self.info["budget"])

    def check_order(self,order):
        order=tuple(order)
        if order not in self.order_cache:
            if len(order)!=self.n or set(order)!=set(range(self.n)):raise ValueError("invalid priority permutation")
            position={v:i for i,v in enumerate(order)}
            if any(position[u]>=position[v] for u,v in self.instance["edges"]):raise ValueError("order is not topological")
            self.order_cache.add(order)
        return order

    def evaluate(self,mapping,order=None,materialize=False,enforce_budget=True):
        if len(mapping)!=self.n or any(p not in self.info["eligible"][v] for v,p in enumerate(mapping)):raise ValueError("invalid mapping")
        cost=self.cost(mapping)
        if enforce_budget and self.info["budget"] is not None and cost>self.info["budget"]:raise ValueError("mapping exceeds budget")
        order=self.check_order(self.info["order"] if order is None else order)
        calendars=[[] for _ in range(self.k)];starts=[0]*self.n;ends=[0]*self.n;ready_times=[0]*self.n
        for v in order:
            p=mapping[v];duration=self.info["duration"][v][p]
            release=max((ends[u]+self.info["comm"][u,v][mapping[u]][p] for u in self.info["parents"][v]),default=0)
            start=find_slot(release,duration,self.instance["demands"][v],self.instance["capacities"][p],calendars[p])
            starts[v]=start;ends[v]=start+duration;ready_times[v]=release
            calendars[p].append((start,ends[v],self.instance["demands"][v]))
        self.evaluations+=1;value=(max(ends),cost)
        if not materialize:return value
        tasks=[{"node":v,"device":p,"start":starts[v],"end":ends[v],"duration":self.info["duration"][v][p],
                "demand":self.instance["demands"][v],"data_ready":ready_times[v],"resource_wait":starts[v]-ready_times[v]}
               for v,p in enumerate(mapping)]
        communications=[]
        for v in order:
            for u in sorted(self.info["parents"][v],key=lambda u:(ends[u],u)):
                p,q=mapping[u],mapping[v];length=self.info["comm"][u,v][p][q]
                communications.append({"u":u,"v":v,"src":p,"dst":q,"start":ends[u],"end":ends[u]+length,
                                       "duration":length,"release":ends[u],"queue_wait":0,"local":p==q,
                                       "data_mb":self.instance["data_mb"][u][v]})
        sol={"algorithm":"Prepared-SGS","makespan":value[0],"cost":cost,"tasks":tasks,"communications":communications,
             "decision_order":list(order)}
        # This checker retains the separate legacy event-sweep and cost ledger.
        core.validate(self.instance,sol);return sol


def dag_regions(engine):
    info=engine.info;n=engine.n;down=[{v} for v in range(n)];up=[{v} for v in range(n)]
    for v in reversed(info["order"]):
        for c in info["children"][v]:down[v]|=down[c]
    for v in info["order"]:
        for p in info["parents"][v]:up[v]|=up[p]
    groups={tuple(range(n))}
    for u in range(n):
        for v in down[u]:groups.add(tuple(sorted(down[u]&up[v])))
        groups.add(tuple(sorted({u}|set(info["parents"][u]))));groups.add(tuple(sorted({u}|set(info["children"][u]))))
    return sorted(g for g in groups if len(g)>=3)
