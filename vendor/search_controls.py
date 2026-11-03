"""Common prepared-engine search controls; all work is recorded, no neural claim."""
from copy import deepcopy
import itertools
import time
import budget_core as core
from budget_controls import mslbl
from prepared_sgs import PreparedSGS,dag_regions


def portfolio(instance):
    started=time.perf_counter()
    schedules=[core.schedule(instance,a) for a in ["HEFT","PEFT","JoinLookahead","CostFirst"]]+[mslbl(instance)]
    schedules += [core.schedule(instance,"PEFT",price=x) for x in [.1,.3,1,3,10]]
    return schedules,time.perf_counter()-started


def mapping_of(s):return [t["device"] for t in sorted(s["tasks"],key=lambda t:t["node"])]


def group_actions(engine,mapping,groups):
    for group in groups:
        for p in range(engine.k):
            if all(p in engine.info["eligible"][v] for v in group):
                changed=list(mapping)
                for v in group:changed[v]=p
                if changed!=mapping:yield changed,{"kind":"region","group":list(group),"device":p}
        for template,key in [("cheap","costs"),("fast","duration")]:
            changed=list(mapping)
            for v in group:changed[v]=min(engine.info["eligible"][v],key=lambda p:(engine.info[key][v][p],p))
            if changed!=mapping:yield changed,{"kind":"region","group":list(group),"template":template}


def lex_search(instance,seed,seconds=60,include_regions=True,include_order=True):
    started=time.perf_counter();engine=PreparedSGS(instance);groups=dag_regions(engine) if include_regions else []
    mapping=mapping_of(seed);order=tuple(seed["decision_order"]);value=engine.evaluate(mapping,order)
    moves=[];passes=0;proposals=0;converged=False;deadline=started+seconds
    while time.perf_counter()<deadline:
        passes+=1;winner=value;chosen=None;winner_mapping=mapping;winner_order=order;completed=True
        def candidates():
            for size in [1,2]:
                for group in itertools.combinations(range(engine.n),size):
                    for ps in itertools.product(*(engine.info["eligible"][v] for v in group)):
                        if any(p==mapping[v] for v,p in zip(group,ps)):continue
                        changed=list(mapping)
                        for v,p in zip(group,ps):changed[v]=p
                        yield changed,order,{"kind":"remap","changes":list(zip(group,ps))}
            if include_order:
                for i in range(engine.n-1):
                    u,v=order[i:i+2]
                    if v not in engine.info["children"][u]:
                        changed=list(order);changed[i],changed[i+1]=v,u
                        yield mapping,tuple(changed),{"kind":"order_swap","tasks":[u,v]}
            for changed,action in group_actions(engine,mapping,groups):yield changed,order,action
        for changed,new_order,action in candidates():
            if time.perf_counter()>=deadline:completed=False;break
            proposals+=1
            if not engine.feasible(changed):continue
            result=engine.evaluate(changed,new_order)
            if result<winner:winner=result;chosen=action;winner_mapping=changed;winner_order=new_order
        if chosen is not None:
            moves.append({"action":chosen,"before":value,"after":winner});value=winner;mapping=winner_mapping;order=winner_order
        if not completed:break
        if chosen is None:converged=True;break
    schedule=engine.evaluate(mapping,order,materialize=True)
    schedule.update(algorithm="Prepared-Lex-Region-Order",runtime_seconds=time.perf_counter()-started,
                    search={"converged":converged,"passes":passes,"moves":moves,"attempted_actions":proposals,
                            "simulations":engine.evaluations,"seconds_limit":seconds,"include_regions":include_regions,
                            "include_order":include_order,"preparation_seconds":engine.compilation_seconds})
    return schedule


def donor_candidates(instance,reference,seconds=10,positive_slack_only=False):
    """Co-location regions paid for by one or greedy multiple cheaper assignments.

    Explicit diagnostic heuristic, not a proposed neural or one-forward policy.
    Only regions whose unfinanced mapping EXCEEDS budget receive donor actions.
    """
    started=time.perf_counter();deadline=started+seconds;engine=PreparedSGS(instance);groups=dag_regions(engine)
    mapping=mapping_of(reference);order=tuple(reference["decision_order"]);initial=engine.evaluate(mapping,order)
    winner=initial;winner_mapping=mapping;chosen=None;proposals=0;unfinanced=0;completed=True
    # Dependency-only latest-finish slack; it is not an exact resource criticality.
    tasks={t["node"]:t for t in reference["tasks"]};latest_finish=[reference["makespan"]]*engine.n
    for v in reversed(engine.info["order"]):
        for c in engine.info["children"][v]:
            latest_finish[v]=min(latest_finish[v],latest_finish[c]-engine.info["duration"][c][mapping[c]]-engine.info["comm"][v,c][mapping[v]][mapping[c]])
    donors=[]
    for v,p in enumerate(mapping):
        if positive_slack_only and latest_finish[v]<=tasks[v]["end"]:continue
        for q in engine.info["eligible"][v]:
            saving=engine.info["costs"][v][p]-engine.info["costs"][v][q]
            if saving<=0:continue
            changed=list(mapping);changed[v]=q;result=engine.evaluate(changed,order)
            damage=max(0,result[0]-initial[0]);donors.append((v,q,saving,damage))
    for grouped,action in group_actions(engine,mapping,groups):
        if time.perf_counter()>=deadline:completed=False;break
        group=action["group"];group_set=set(group);cost=engine.cost(grouped);deficit=cost-instance["budget"]
        if deficit<=0:continue
        unfinanced+=1;allowed=[d for d in donors if d[0] not in group_set]
        candidates=[]
        for v,q,saving,damage in allowed:
            if saving>=deficit:
                changed=list(grouped);changed[v]=q;candidates.append((changed,[(v,q,saving)]))
        for ranking in [lambda d:(d[3]/d[2],d[3],-d[2],d[0],d[1]),lambda d:(-d[2],d[3],d[0],d[1])]:
            changed=list(grouped);used=set();remaining=deficit;selected=[]
            for v,q,saving,damage in sorted(allowed,key=ranking):
                if v in used:continue
                used.add(v);changed[v]=q;remaining-=saving;selected.append((v,q,saving))
                if remaining<=0:break
            if remaining<=0:candidates.append((changed,selected))
        seen=set()
        for changed,selected in candidates:
            if time.perf_counter()>=deadline:completed=False;break
            key=tuple(changed)
            if key in seen:continue
            seen.add(key);proposals+=1
            if not engine.feasible(changed):raise AssertionError("funded proposal must obey budget")
            result=engine.evaluate(changed,order)
            if result<winner:
                winner=result;winner_mapping=changed
                chosen={"region":action,"donors":selected,"unfunded_cost":cost,"deficit":deficit,
                        "original_spare_budget":instance["budget"]-initial[1]}
        if not completed:break
    solution=engine.evaluate(winner_mapping,order,materialize=True)
    solution.update(algorithm="Region-With-Budget-Compensation" if not positive_slack_only else "Region-Positive-Slack-Donors",
                    runtime_seconds=time.perf_counter()-started,donor_search={"completed":completed,"chosen":chosen,
                    "unfinanced_regions":unfinanced,"feasible_candidates":proposals,"simulations":engine.evaluations,
                    "positive_slack_only":positive_slack_only,"seconds_limit":seconds,
                    "preparation_seconds":engine.compilation_seconds})
    return solution
