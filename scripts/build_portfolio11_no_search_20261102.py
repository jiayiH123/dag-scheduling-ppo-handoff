#!/usr/bin/env python3
"""Eleven candidate mappings only: no local search, no legacy-final retention."""
import collections
import concurrent.futures as cf
import fcntl
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from daglab.teacher_search_ablation import config,code_binding
from daglab.sft_three_panel import connect,unpack,pack,write_json,to_instance
from daglab.structured_dataset import sha256
from daglab.offline_data import digest
from daglab.state_score import StateScoreDecoder
from daglab.reference import mapping_of
from daglab.external_baseline_sim import Problem
from scripts.build_teacher64_bheft_v2_20261102 import seed_schedule

CON=None


def initialize():
    global CON
    CON=connect(ROOT/config()['original_cache']/'public.sqlite3')


def make(key):
    record=CON.execute('SELECT * FROM records WHERE key=? AND source="synthetic" AND role="train"',(key,)).fetchone()
    raw,payload=unpack(record[9]),unpack(record[10])
    inst=to_instance(raw)
    dec=StateScoreDecoder(inst)
    q=raw['quality']
    values=list(zip(q['fixed_order_makespans'],q['fixed_order_costs']))
    old_index=min(range(10),key=lambda i:values[i])
    assert old_index==q['initial_rule_index']
    mapping=mapping_of(seed_schedule(inst,old_index))
    assert dec.engine.evaluate(mapping,dec.order)==values[old_index]
    problem=Problem(inst)
    b=problem.schedule(rule='BHEFT')
    assert tuple(b['order'])==tuple(dec.order)
    checked=problem.finish(b['mapping'],b['order'])
    b_value=(checked['C'],checked['cost'])
    assert dec.engine.evaluate(b['mapping'],dec.order)==b_value
    values.append(b_value)
    selected=min(range(11),key=lambda i:values[i])
    mapping=b['mapping'] if selected==10 else mapping
    replay=dec.rollout([[0.]*4]*(dec.n*dec.k),forced_mapping=mapping)
    value=(replay['makespan'],replay['cost'])
    assert value==values[selected]
    assert dec.engine.evaluate(mapping,dec.order,materialize=True)['tasks']==replay['schedule']['tasks']
    name='Portfolio11-no-search'
    supervision=dict(task_to_pool=mapping,makespan=value[0],cost=value[1],decisions=replay['decisions'],
        starts=[t['start'] for t in replay['schedule']['tasks']],ends=[t['end'] for t in replay['schedule']['tasks']],
        decoder='StateScoreDecoder/fixed-HEFT/forced-mapping',teacher=name,
        target='conditional feasible-pool action likelihood; own prefix')
    quality=dict(rule_names=q['rule_names']+['BHEFT-adapted'],fixed_order_makespans=[v[0] for v in values],
        fixed_order_costs=[v[1] for v in values],initial_rule_index=selected,local_attempts=0,
        legacy_final_retention=False,legacy_search_not_used=True,optimality='not certified')
    new_raw=dict(raw,supervision=supervision,quality=quality,teacher_version=name)
    new_payload=dict(payload,supervision=supervision,decisions=replay['decisions'],teacher=value[0],teacher_version=name)
    assert new_payload['graph']==payload['graph'] and new_raw['input']==raw['input']
    metric=dict(key=key,n=dec.n,k=dec.k,initial_rule=quality['rule_names'][selected],
        C=value[0],cost=value[1],legacy64_C=payload['teacher'],actions=len(replay['decisions']),local_attempts=0)
    return tuple(record[:9])+(pack(new_raw),pack(new_payload)),metric


def main():
    cfg=config()
    out=ROOT/cfg['no_search_cache']
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'builder.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=ROOT/cfg['original_cache']/'public.sqlite3'
    binding=dict(config=sha256(ROOT/'configs/teacher_search_ablation_ppo_20261102.json'),source=sha256(source),
        code=code_binding(['scripts/build_portfolio11_no_search_20261102.py','scripts/build_teacher64_bheft_v2_20261102.py',
            'daglab/reference.py','daglab/state_score.py','daglab/external_baseline_sim.py']))
    if (out/'BINDING.json').exists():
        assert json.loads((out/'BINDING.json').read_text())==binding
    else:
        write_json(out/'BINDING.json',binding)
    if (out/'MANIFEST.json').exists():
        assert json.loads((out/'MANIFEST.json').read_text())['database_sha256']==sha256(out/'public.sqlite3')
        return
    initialize()
    keys=[r[0] for r in CON.execute('SELECT key FROM records WHERE source="synthetic" AND role="train" ORDER BY n,k,key')]
    assert len(keys)==17261
    con=sqlite3.connect(out/'public.sqlite3')
    con.executescript('CREATE TABLE IF NOT EXISTS records(key TEXT PRIMARY KEY,source TEXT,role TEXT,cohort INTEGER,base_id TEXT,topology TEXT,n INTEGER,k INTEGER,physical_sha TEXT,raw BLOB,payload BLOB);CREATE TABLE IF NOT EXISTS provenance(key TEXT PRIMARY KEY,detail TEXT);')
    if not (out/'ENGINEERING_CHECKS.json').exists():
        checks=[]
        for n in [20,40,80,120]:
            for k in [3,4,6]:
                key=CON.execute('SELECT key FROM records WHERE source="synthetic" AND role="train" AND n=? AND k=? ORDER BY key LIMIT 1',(n,k)).fetchone()[0]
                record,metric=make(key)
                assert unpack(record[-2])['quality']['local_attempts']==0
                checks.append(metric)
        write_json(out/'ENGINEERING_CHECKS.json',dict(status='PASS',configs=12,rows=checks,local_proposals=0))
    done={r[0] for r in con.execute('SELECT key FROM records')}
    pending=[k for k in keys if k not in done]
    started=time.time()
    try:
        with cf.ProcessPoolExecutor(max_workers=2,initializer=initialize) as pool:
            for index,(record,metric) in enumerate(pool.map(make,pending,chunksize=4),1):
                con.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?)',record)
                con.execute('INSERT INTO provenance VALUES (?,?)',(metric['key'],json.dumps(metric)))
                if index%8==0 or index==len(pending):
                    con.commit()
                if index%40==0 or index==len(pending):
                    write_json(out/'progress.json',dict(status='labeling',configs_done=len(done)+index,target=17261,seconds=time.time()-started))
        assert con.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        actual=list(con.execute('SELECT key,physical_sha FROM records ORDER BY key'))
        expected=list(CON.execute('SELECT key,physical_sha FROM records WHERE source="synthetic" AND role="train" ORDER BY key'))
        assert actual==expected
        metrics=[json.loads(r[0]) for r in con.execute('SELECT detail FROM provenance')]
        assert sum(r['actions'] for r in metrics)==1122860
        assert all(r['local_attempts']==0 for r in metrics)
        con.commit()
        con.close()
        assert sha256(source)==binding['source']
        summary=dict(status='PASS',configs=17261,bases=10000,actions=1122860,local_attempts=0,
            no_legacy_final_retention=True,input_keys_and_physics_unchanged=True,
            selected_rules=dict(collections.Counter(r['initial_rule'] for r in metrics)))
        write_json(out/'SUMMARY.json',summary)
        write_json(out/'MANIFEST.json',dict(status='complete',database_sha256=sha256(out/'public.sqlite3'),
            binding=binding,summary=summary,teacher='Portfolio11-no-search'))
        write_json(out/'progress.json',dict(status='complete',configs_done=17261,target=17261))
    except BaseException:
        write_json(out/'FAILURE.json',dict(error=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
