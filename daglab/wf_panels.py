"""Application-level WF reporting from existing per-case evaluation results.

No models, schedulers, source data mutation, or performance-based membership.
Can be called after each future VAL observation without repeating inference.
"""
from collections import defaultdict
import math
import statistics

SIZE_BINS = [(20,40),(41,80),(81,120),(121,200),(201,300)]


def size_group(n):
    for lo,hi in SIZE_BINS:
        if lo<=n<=hi:
            return f'{lo}-{hi}'
    return 'outside20-300'


def app_group(row):
    identity=row['identity']
    return identity['engine']+'/'+identity['application']


def coverage(rows):
    return dict(configs=len(rows),executions=len({r['base_id'] for r in rows}),
                topology_groups_WL16=len({r['topology'] for r in rows}),
                n_min=min(r['n'] for r in rows),n_max=max(r['n'] for r in rows))


def summarize(rows):
    """Config -> execution -> topology -> application equal weighting.

    Retain configuration means and feasibility separately. A failed/missing
    baseline or policy result invalidates that ratio rather than being dropped.
    """
    result=coverage(rows)
    result['feasible']=all(r.get('feasible',False) for r in rows)
    refs=sorted(set.union(*(set(r['references']) for r in rows)))
    values={};flat={};wins={};incomplete={}
    for ref in refs:
        invalid=[r['key'] for r in rows if not r.get('feasible',False) or ref not in r['references']
                 or not r['references'][ref].get('feasible',False)
                 or not isinstance(r['references'][ref].get('C'),(int,float))
                 or not math.isfinite(r['references'][ref]['C']) or r['references'][ref]['C']<=0]
        if invalid:
            values[ref]=None;flat[ref]=None;wins[ref]=None;incomplete[ref]=invalid
            continue
        bases=defaultdict(list);identities={}
        ratios=[r['C']/r['references'][ref]['C'] for r in rows]
        for r,ratio in zip(rows,ratios):
            bases[r['base_id']].append(ratio)
            identities[r['base_id']]=(app_group(r),r['topology'])
        topologies=defaultdict(list)
        for base,rs in bases.items():topologies[identities[base]].append(statistics.mean(rs))
        apps=defaultdict(list)
        for (app,topology),rs in topologies.items():apps[app].append(statistics.mean(rs))
        values[ref]=statistics.mean(statistics.mean(rs) for rs in apps.values())
        flat[ref]=statistics.mean(ratios)
        wins[ref]=sum(r['C']<r['references'][ref]['C'] for r in rows)/len(rows)
    result.update(C_over=values,improvement_percent={k:None if v is None else 100*(1-v) for k,v in values.items()},
                  configuration_mean_C_over=flat,configuration_win_rate=wins,incomplete_references=incomplete)
    return result


def report(rows):
    rows=[r for r in rows if r.get('source')=='wfinstances']
    if not rows:raise ValueError('No WF evaluation rows')
    groups={name:defaultdict(list) for name in ('application_engine','application','task_count','application_engine_task_count')}
    for row in rows:
        groups['application_engine'][app_group(row)].append(row)
        groups['application'][row['identity']['application']].append(row)
        groups['task_count'][size_group(row['n'])].append(row)
        groups['application_engine_task_count'][app_group(row)+'|'+size_group(row['n'])].append(row)
    return dict(schema='wf-group-evaluation-v1',all=summarize(rows),
                groups={name:{key:summarize(rs) for key,rs in sorted(group.items())} for name,group in groups.items()},
                weighting='configuration -> execution -> WL16 topology within application/engine -> application/engine',
                scope='Known tuning evaluation; all groups retained. No automatic subset selection.')
