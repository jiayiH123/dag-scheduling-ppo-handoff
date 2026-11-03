#!/usr/bin/env python3
"""Simulator-only read bridge. TEST commands require the frozen model selection."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from daglab.unified_public_data import ROOT,SYNTHETIC,config,readonly,sha,unpack


def baselines(inst):
    from daglab.reference import CachedBudgetDecoder,core,portfolio,teacher
    from daglab.diagnostic_decoder import DiagnosticDecoder
    from daglab.external_baseline_sim import Problem
    seeds,seconds=portfolio(inst);values={}
    def add(name,solution,elapsed=None):
        core.validate(inst,solution)
        values[name]=dict(C=solution['makespan'],cost=solution['cost'],feasible=True,
            seconds=elapsed if elapsed is not None else solution.get('runtime_seconds'))
    add('HEFT',seeds[0]);add('MSLBL',seeds[4]);add('Greedy',core.schedule(inst,'Greedy'))
    add('Portfolio',min(seeds,key=lambda s:(s['makespan'],s['cost'])),seconds)
    solution,_,elapsed=teacher(inst,CachedBudgetDecoder(inst));add('Teacher64',solution,elapsed)
    problem=Problem(inst)
    for name in ['BHEFT','BUDA']:
        begin=time.perf_counter();r=problem.schedule(rule=name);f=problem.finish(r['mapping'],r['order'])
        assert (f['C'],f['cost'])==(r['C'],r['cost'])
        values[name]=dict(C=f['C'],cost=f['cost'],feasible=True,seconds=time.perf_counter()-begin)
    dec=DiagnosticDecoder(inst);parts=[];begin=time.perf_counter()
    for j in range(5):
        r=dec.rollout([[0.]*4]*(dec.n*dec.k),uniform=True,seed=131+inst['seed']*17+j)
        parts.append(dict(C=r['makespan'],cost=r['cost'],feasible=True))
    values['RandomMean5']=dict(C=sum(r['C'] for r in parts)/5,cost=sum(r['cost'] for r in parts)/5,
        seconds=time.perf_counter()-begin,feasible=True,components=parts)
    return values


def worker(directory,run):
    from daglab.offline_data import dense_instance
    from daglab.reference import graph_features
    from daglab.module_lab.features import enrich
    from daglab.state_score import StateScoreDecoder
    cfg=config();manifest=json.loads((directory/'MANIFEST.json').read_text())
    assert manifest['status']=='complete' and manifest['database_sha256']==sha(directory/'public.sqlite3')
    con=readonly(directory);current=None;inst=None;test_rows=None
    def ready_test():
        path=run/'frozen_selection.json'
        assert path.exists(),'TEST unavailable before final DEV checkpoint selection is frozen'
        freeze=json.loads(path.read_text())
        assert freeze['all_training_epochs_completed']==7
        return freeze
    def load(key):
        nonlocal current,inst,test_rows
        if key.startswith('synthetic_test:'):
            ready_test()
            if test_rows is None:
                from daglab.structured_unique import iter_unique_rows
                keys=set(json.loads((directory/'synthetic_test_keys.json').read_text())['keys'])
                test_rows={r['key']:r for r in iter_unique_rows(SYNTHETIC,'test') if r['key'] in keys}
                assert len(test_rows)==862
            from daglab.structured_dataset import to_instance
            r=test_rows[key.split(':',1)[1]];inst=to_instance(r)
            meta=dict(source='synthetic',role='test_primary',cohort=0,base_id=r['base_id'],
                family=r['meta']['family'],topology=r['meta']['topology_fingerprint'],
                pressure=r['meta']['demand_pressure'],eta=inst['budget_eta'],n=inst['n'],k=len(inst['speeds']))
            payload=None
        else:
            rec=con.execute('SELECT raw,payload,source,role,cohort,base_id,topology FROM records WHERE key=?',(key,)).fetchone()
            assert rec is not None,key
            raw=unpack(rec[0]);payload=unpack(rec[1]) if rec[1] else None
            if rec[3].startswith('test_'):ready_test()
            inst=dense_instance(raw,max_tasks=300)
            meta=dict(source=rec[2],role=rec[3],cohort=rec[4],base_id=rec[5],topology=rec[6],
                family=inst['family'],pressure=inst['resource_adaptation']['pressure'],eta=inst['budget_eta'],
                n=inst['n'],k=len(inst['speeds']))
        current=StateScoreDecoder(inst)
        if payload is None:
            g,_=graph_features(inst);g=enrich(dict(instance=inst,graph=g),current)
        else:g=payload['graph']
        meta['key']=key
        return dict(meta=meta,graph=g,reference=payload['reference'] if payload else None)
    for line in sys.stdin:
        q=json.loads(line);cmd=q['cmd']
        if cmd=='init':
            out=dict(manifest=manifest,train_keys={s:[r[0] for r in con.execute('SELECT key FROM records WHERE source=? AND role="train" ORDER BY key',(s,))]
                for s in ['wfinstances','tpch_composite']},fresh_val_keys=[r[0] for r in con.execute('SELECT key FROM records WHERE role="fresh_val" ORDER BY source,key')])
        elif cmd=='batch':
            out=[]
            for key in q['keys']:
                row=con.execute('SELECT payload,role FROM records WHERE key=?',(key,)).fetchone()
                assert row is not None and row[1]=='train'
                payload=unpack(row[0]);payload.pop('supervision',None);out.append(payload)
        elif cmd=='get':out=load(q['key'])
        elif cmd=='roll':
            assert current is not None
            out={}
            for name,x in q['coefficients'].items():
                r=current.rollout(x,sample=q.get('sample',False),seed=q.get('seed',0),forced_mapping=q.get('forced_mapping'))
                out[name]=dict(C=r['makespan'],cost=r['cost'],feasible=True,
                    mapping_sha256=hashlib.sha256(json.dumps([t['device'] for t in r['schedule']['tasks']]).encode()).hexdigest())
                if q.get('keep_decisions'):out[name]['decisions']=r['decisions']
        elif cmd=='baseline':out=baselines(inst)
        elif cmd=='training_checks':
            keys=[]
            for source in ['wfinstances','tpch_composite']:
                for k in [3,6]:
                    keys.append(con.execute('SELECT key FROM records WHERE role="train" AND source=? AND k=? ORDER BY n,key LIMIT 1',(source,k)).fetchone()[0])
            out=keys
        elif cmd=='test_keys':
            ready_test()
            out=['synthetic_test:'+k for k in json.loads((directory/'synthetic_test_keys.json').read_text())['keys']]
            out += [r[0] for r in con.execute('SELECT key FROM records WHERE role LIKE "test_%" ORDER BY role,source,cohort,key')]
            assert len(out)==3566
        elif cmd=='close':
            print('{}',flush=True);con.close();return
        else:raise ValueError(cmd)
        print(json.dumps(out,allow_nan=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    worker(args.data,args.run)
