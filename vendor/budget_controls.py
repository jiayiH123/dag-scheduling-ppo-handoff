"""Literature-grounded budget allocation and plateau-aware diagnostic controls."""
from copy import deepcopy
from fractions import Fraction
import itertools
import time
import budget_core as core
import legacy_comm_core as legacy


def mslbl(instance):
    """Chen et al. FGCS2017 Eqs8,9,22 and EFT selection, with explicit adaptations.

    Original task budget logic is retained exactly with rational arithmetic.
    Actual link lags, capacity eligibility and Cumulative insertion are our base-
    model adaptations. At B=Bmin we allow the feasible cheapest completion,
    rather than the pseudocode's boundary rejection. This is not author code.
    """
    started=time.perf_counter();info=core.prepare(instance)
    minimum=info["min_cost"]
    maximum=[max(info["costs"][v][p] for p in info["eligible"][v]) for v in range(info["n"])]
    bmin,bmax=sum(minimum),sum(maximum)
    budget=info["budget"] if info["budget"] is not None else bmax
    level=min(Fraction(1),Fraction(budget-bmin,max(1,bmax-bmin)))
    reserve=[minimum[v]+level*(maximum[v]-minimum[v]) for v in range(info["n"])]
    state=legacy.blank_state(info["k"]);ledger=[]
    while len(state["tasks"])<info["n"]:
        ready=[v for v in info["order"] if v not in state["tasks"] and all(u in state["tasks"] for u in info["parents"][v])]
        v=min(ready,key=lambda v:(-info["ranks"][v],v))
        used=sum(info["costs"][u][t["device"]] for u,t in state["tasks"].items())
        task_budget=budget-used-sum(reserve[u] for u in range(info["n"]) if u not in state["tasks"] and u!=v)
        candidates=[]
        for p in info["eligible"][v]:
            if info["costs"][v][p]<=task_budget:
                candidate=legacy.placement(instance,info,state,v,p)
                candidates.append((candidate["task"]["end"],candidate["task"]["start"],p,candidate))
        if not candidates:raise AssertionError("budget-level reserve must admit a completion")
        _,_,p,c=min(candidates,key=lambda t:t[:3])
        state["tasks"][v]=c["task"];state["calendar"]=c["calendar"]
        state["communications"].extend(c["communications"]);state["decision_order"].append(v)
        ledger.append({"node":v,"task_budget":str(task_budget),"cost":info["costs"][v][p]})
    solution=legacy.finish_schedule(instance,state,"MSLBL-adapted")
    solution["cost"]=sum(info["costs"][t["node"]][t["device"]] for t in solution["tasks"])
    solution.update(runtime_seconds=time.perf_counter()-started,budget_level=str(level),budget_ledger=ledger)
    core.validate(instance,solution);return solution


def lex_pair_search(instance,seed):
    """Accept strictly lower Cmax OR equal Cmax with lower cost; no new method claim."""
    started=time.perf_counter();info=core.prepare(instance);best=deepcopy(seed);moves=[];trials=0
    while True:
        assignment=[t["device"] for t in best["tasks"]];key=(best["makespan"],best["cost"])
        chosen=None;trial_best=best
        for size in [1,2]:
            for group in itertools.combinations(range(info["n"]),size):
                for devices in itertools.product(*(info["eligible"][v] for v in group)):
                    if any(p==assignment[v] for v,p in zip(group,devices)):continue
                    changed=assignment.copy()
                    for v,p in zip(group,devices):changed[v]=p
                    if info["budget"] is not None and sum(info["costs"][v][p] for v,p in enumerate(changed))>info["budget"]:continue
                    trial=core.schedule(instance,"replay",assignment=changed,order=best["decision_order"]);trials+=1
                    trial_key=(trial["makespan"],trial["cost"])
                    if trial_key<key:
                        trial_best=trial;chosen=list(zip(group,devices));key=trial_key
        if chosen is None:break
        moves.append({"change":chosen,"before":[best["makespan"],best["cost"]],"after":[trial_best["makespan"],trial_best["cost"]]})
        best=trial_best
    best.update(algorithm="Lex-Pair",runtime_seconds=time.perf_counter()-started,
                lex_search={"converged":True,"moves":moves,"candidate_count":trials})
    core.validate(instance,best);return best
