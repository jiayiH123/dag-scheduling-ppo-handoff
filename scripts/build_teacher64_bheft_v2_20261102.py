#!/usr/bin/env python3
"""Finite, resumable, immutable-input Teacher64+BHEFT relabeling. No neural calls."""
import argparse
import collections
import concurrent.futures as cf
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from daglab.sft_three_panel import write_json,pack,unpack
from daglab.structured_dataset import sha256,to_instance
from daglab.offline_data import dense_instance
from daglab.reference import CachedBudgetDecoder,core,portfolio,mapping_of
from daglab.state_score import StateScoreDecoder
from daglab.external_baseline_sim import Problem
from budget_controls import mslbl

CONFIG=ROOT/'configs/teacher64_bheft_v2_20261102.json'
SOURCES={}
CFG={}


def init_worker(cfg):
    global CFG,SOURCES
    CFG=cfg
    SOURCES={s['id']:sqlite3.connect('file:'+str((ROOT/s['database']).resolve())+'?mode=ro',uri=True) for s in cfg['sources']}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def seed_schedule(inst,index):
    if index==4:return mslbl(inst)
    if index>=5:return core.schedule(inst,'PEFT',price=[.1,.3,1,3,10][index-5])
    return core.schedule(inst,['HEFT','PEFT','JoinLookahead','CostFirst'][index])


def relabel(source_id,key):
    begin=time.perf_counter()
    record=SOURCES[source_id].execute('SELECT * FROM records WHERE key=?',(key,)).fetchone()
    assert record and record[2]=='train'
    raw=unpack(record[9]);payload=unpack(record[10]);old_supervision=payload['supervision']
    inst=to_instance(raw) if 'input' in raw else dense_instance(raw,max_tasks=300)
    dec=StateScoreDecoder(inst);engine=dec.engine
    old_mapping=old_supervision['task_to_pool']
    old_value=(old_supervision['makespan'],old_supervision['cost'])
    old_solution=engine.evaluate(old_mapping,dec.order,materialize=True)
    assert (old_solution['makespan'],old_solution['cost'])==old_value
    for field,vector in [('start','starts'),('end','ends')]:
        assert [t[field] for t in old_solution['tasks']]==old_supervision[vector]
    q=raw.get('quality',{})
    if all(k in q for k in ['rule_names','fixed_order_makespans','fixed_order_costs','initial_rule_index']):
        assert q['rule_names']==CFG['legacy_rules']
        values=list(zip(q['fixed_order_makespans'],q['fixed_order_costs']))
        old_index=min(range(10),key=lambda i:values[i])
        assert old_index==q['initial_rule_index']
        schedule=seed_schedule(inst,old_index)
        initial_mapping=mapping_of(schedule)
        assert engine.evaluate(initial_mapping,dec.order)==values[old_index]
        assert schedule['makespan']==q['native_makespans'][old_index]
        native=list(q['native_makespans']);seed_mode='cached_ten_values_verified_best_mapping'
    else:
        seeds,_=portfolio(inst);mappings=[mapping_of(s) for s in seeds]
        values=[engine.evaluate(m,dec.order) for m in mappings]
        old_index=min(range(10),key=lambda i:values[i]);initial_mapping=mappings[old_index]
        native=[s['makespan'] for s in seeds];seed_mode='recomputed_ten_candidates'
    assert len(values)==10 and old_value<=values[old_index]
    problem=Problem(inst);bheft=problem.schedule(rule='BHEFT')
    assert tuple(bheft['order'])==tuple(dec.order)
    bheft_checked=problem.finish(bheft['mapping'],bheft['order'])
    bheft_value=(bheft_checked['C'],bheft_checked['cost'])
    assert bheft_value==(bheft['C'],bheft['cost'])
    assert engine.evaluate(bheft['mapping'],dec.order)==bheft_value
    values.append(bheft_value);native.append(bheft_value[0])
    initial_index=min(range(11),key=lambda i:values[i])
    mapping=list(bheft['mapping'] if initial_index==10 else initial_mapping)
    value=values[initial_index]
    rng=random.Random(inst['seed']+CFG['local_seed_offset']);feasible=0;accepted=[]
    for attempt in range(CFG['local_attempts']):
        changed=list(mapping)
        for v in rng.sample(range(engine.n),rng.choice([1,2])):
            changed[v]=rng.choice(engine.info['eligible'][v])
        if not engine.feasible(changed):continue
        feasible+=1;candidate=engine.evaluate(changed,dec.order)
        if candidate<value:
            accepted.append(dict(attempt=attempt+1,changes=[[v,p] for v,p in enumerate(changed) if p!=mapping[v]],before=value,after=candidate))
            mapping,value=changed,candidate
    search_value=value
    if old_value<=value:
        mapping=list(old_mapping);value=old_value;incumbent='legacy_final'
    else:incumbent='eleven_seed_search'
    assert value<=old_value and value<=bheft_value
    replay=dec.rollout([[0.]*4]*(dec.n*dec.k),forced_mapping=mapping)
    solution=replay['schedule'];assert (replay['makespan'],replay['cost'])==value
    checked=engine.evaluate(mapping,dec.order,materialize=True)
    assert checked['tasks']==solution['tasks']
    if mapping==old_mapping:assert replay['decisions']==payload['decisions']
    algorithm='Teacher64+BHEFT-v2-retain-legacy'
    quality=dict(rule_names=CFG['legacy_rules']+['BHEFT-adapted'],native_makespans=native,
        fixed_order_makespans=[v[0] for v in values],fixed_order_costs=[v[1] for v in values],
        initial_rule_index=initial_index,local_attempts=64,feasible_local_proposals=feasible,
        accepted_local_proposals=len(accepted),accepted_moves=accepted,
        search_final_objective=search_value,legacy_final_objective=old_value,
        final_incumbent=incumbent,seed_evaluation_mode=seed_mode,
        fresh_engine_evaluations=engine.evaluations,optimality='not certified',
        legacy64_work_reused=True,legacy_quality=q)
    supervision=dict(task_to_pool=mapping,makespan=value[0],cost=value[1],
        starts=[t['start'] for t in solution['tasks']],ends=[t['end'] for t in solution['tasks']],
        decisions=replay['decisions'],decoder='StateScoreDecoder/fixed-HEFT/forced-mapping',teacher=algorithm,
        target='conditional feasible-pool action likelihood; own prefix')
    graph_hash=digest(payload['graph']);old_reference=payload['reference']
    new_payload=dict(payload,supervision=supervision,decisions=replay['decisions'],teacher=value[0],
        teacher_version=algorithm,legacy_teacher_value=dict(C=old_value[0],cost=old_value[1]))
    new_raw=dict(raw,supervision=supervision,quality=quality,teacher_version=algorithm)
    assert new_payload['graph']==payload['graph'] and new_payload['reference']==old_reference
    assert new_raw.get('input',new_raw.get('instance_sparse'))==raw.get('input',raw.get('instance_sparse'))
    m=payload['meta']
    metrics=dict(key=key,source_partition=source_id,source=record[1],base_id=record[4],n=record[6],k=record[7],
        family=m['family'],target_application=m.get('target_application'),eta=m.get('eta'),
        initial_rule=quality['rule_names'][initial_index],initial_BHEFT_better_than_legacy_ten=bheft_value<values[old_index],
        BHEFT_C=bheft_value[0],BHEFT_cost=bheft_value[1],old_C=old_value[0],old_cost=old_value[1],
        new_C=value[0],new_cost=value[1],relative_makespan_gain=1-value[0]/old_value[0],
        C_improved=value[0]<old_value[0],cost_only_improved=value[0]==old_value[0] and value[1]<old_value[1],
        mapping_changed=mapping!=old_mapping,changed_pool_labels=sum(a!=b for a,b in zip(mapping,old_mapping)),
        incumbent=incumbent,retention_prevented_regression=old_value<search_value,
        teacher_actions=len(replay['decisions']),new_proposals=64,feasible_proposals=feasible,
        accepted_proposals=len(accepted),graph_sha256=graph_hash,physical_sha=record[8],seconds=time.perf_counter()-begin)
    return tuple(record[:9])+(pack(new_raw),pack(new_payload)),metrics


def job_worker(job):
    return job['id'],[relabel(job['source'],key) for key in job['keys']]


def input_plan(cfg):
    jobs=[];parts={}
    for source in cfg['sources']:
        con=sqlite3.connect('file:'+str((ROOT/source['database']).resolve())+'?mode=ro',uri=True)
        rows=con.execute('SELECT key,n,k FROM records WHERE '+source['where']+' ORDER BY n,k,key').fetchall();con.close()
        assert len(rows)==source['expected_configs']
        assert sum(r[1] for r in rows)==source['expected_actions']
        parts[source['id']]=dict(configs=len(rows),actions=sum(r[1] for r in rows),keys_sha256=digest([r[0] for r in rows]))
        for index,start in enumerate(range(0,len(rows),cfg['batch_configs'])):
            jobs.append(dict(id=f"{source['id']}:{index:05d}",source=source['id'],keys=[r[0] for r in rows[start:start+cfg['batch_configs']]]))
    return jobs,parts


def engineering(cfg,out):
    init_worker(cfg);selected={}
    for s in cfg['sources']:
        con=SOURCES[s['id']]
        for n,k in con.execute('SELECT DISTINCT n,k FROM records WHERE '+s['where']+' ORDER BY n,k'):
            if s['id']=='original':
                key=con.execute('SELECT key FROM records WHERE '+s['where']+' AND n=? AND k=? ORDER BY key LIMIT 1',(n,k)).fetchone()[0]
                selected[s['id'],key]=True
        if s['id']=='augmented':
            seen=set()
            for key,blob in con.execute('SELECT key,payload FROM records WHERE '+s['where']+' ORDER BY n DESC,k DESC,key'):
                meta=unpack(blob)['meta'];app=meta.get('target_application',meta['family'])
                if app not in seen:selected[s['id'],key]=True;seen.add(app)
            key=con.execute('SELECT key FROM records WHERE '+s['where']+' ORDER BY n DESC,k DESC,key LIMIT 1').fetchone()[0]
            selected[s['id'],key]=True
    examples=[];first=None
    for source,key in selected:
        record,metric=relabel(source,key);examples.append(metric)
        if first is None:first=(source,key,record,metric)
    with cf.ProcessPoolExecutor(max_workers=1,initializer=init_worker,initargs=(cfg,)) as pool:
        source,key,serial,metric=first
        _,items=pool.submit(job_worker,dict(id='engineering',source=source,keys=[key])).result()
        other,othermetric=items[0]
        assert serial==other
        assert {k:v for k,v in metric.items() if k!='seconds'}=={k:v for k,v in othermetric.items() if k!='seconds'}
    value=dict(status='PASS',examples=len(examples),checks=['legacy_objective_replay','cached_seed_exactness',
        'BHEFT_same_order_and_physical_replay','64_proposals','nonregression_vs_old_and_BHEFT',
        'own_prefix_StateScore_replay','unchanged_graph_and_physics','serial_worker_exactness'],
        model_forwards=0,optimizer_updates=0,rows=examples)
    write_json(out/'ENGINEERING_CHECKS.json',value)
    print(json.dumps(dict(status='engineering_PASS',examples=len(examples))),flush=True)


def summarize(con):
    groups=collections.defaultdict(list)
    for (blob,) in con.execute('SELECT detail FROM provenance'):
        r=json.loads(blob)
        for key in [f"partition:{r['source_partition']}",f"family:{r['family']}",f"n:{r['n']}",f"k:{r['k']}"]:
            groups[key].append(r)
        if r['target_application']:groups['application:'+r['target_application']].append(r)
    result={}
    for key,rr in groups.items():
        result[key]=dict(configs=len(rr),actions=sum(r['teacher_actions'] for r in rr),
            C_improved=sum(r['C_improved'] for r in rr),cost_only_improved=sum(r['cost_only_improved'] for r in rr),
            mappings_changed=sum(r['mapping_changed'] for r in rr),changed_pool_labels=sum(r['changed_pool_labels'] for r in rr),
            mean_relative_makespan_gain=sum(r['relative_makespan_gain'] for r in rr)/len(rr),
            BHEFT_initial_wins=sum(r['initial_BHEFT_better_than_legacy_ten'] for r in rr),
            retention_prevented_regression=sum(r['retention_prevented_regression'] for r in rr),
            incumbent_counts=dict(collections.Counter(r['incumbent'] for r in rr)))
    return result


def build(cfg,out):
    lock=(out/'builder.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs,parts=input_plan(cfg)
    files=['scripts/build_teacher64_bheft_v2_20261102.py','daglab/state_score.py','daglab/reference.py',
        'daglab/structured_dataset.py','vendor/budget_core.py','vendor/budget_controls.py','vendor/prepared_sgs.py',
        'vendor/cached_budget_decoder.py','vendor/search_controls.py','daglab/external_baseline_sim.py']
    binding=dict(config=sha256(CONFIG),protocol=sha256(ROOT/cfg['protocol']),
        code={f:sha256(ROOT/f) for f in files},inputs={s['id']:sha256(ROOT/s['database']) for s in cfg['sources']},partitions=parts)
    if (out/'INPUT_BINDING.json').exists():assert json.loads((out/'INPUT_BINDING.json').read_text())==binding
    else:write_json(out/'INPUT_BINDING.json',binding)
    if (out/'MANIFEST.json').exists():
        assert json.loads((out/'MANIFEST.json').read_text())['database_sha256']==sha256(out/'public.sqlite3');return
    assert json.loads((out/'ENGINEERING_CHECKS.json').read_text())['status']=='PASS'
    con=sqlite3.connect(out/'public.sqlite3')
    con.executescript('''CREATE TABLE IF NOT EXISTS records(key TEXT PRIMARY KEY,source TEXT,role TEXT,cohort INTEGER,base_id TEXT,topology TEXT,n INTEGER,k INTEGER,physical_sha TEXT,raw BLOB,payload BLOB);
CREATE TABLE IF NOT EXISTS provenance(key TEXT PRIMARY KEY,source_partition TEXT,detail TEXT);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,count INTEGER);''')
    done={r[0] for r in con.execute('SELECT id FROM jobs')};pending=[j for j in jobs if j['id'] not in done]
    begin=time.time();last=0;completed=len(done)
    def progress(status):
        nonlocal last
        counts=dict(con.execute('SELECT source_partition,count(*) FROM provenance GROUP BY source_partition'))
        total=sum(counts.values())
        value=dict(status=status,pid=os.getpid(),updated_unix=time.time(),configs_done=total,target_configs=21261,
            partitions_done=counts,targets={s['id']:s['expected_configs'] for s in cfg['sources']},
            jobs_done=completed,jobs_total=len(jobs),session_seconds=time.time()-begin,workers=cfg['workers'],
            automatic_training=False)
        write_json(out/'progress.json',value);last=time.time();return value
    awake=subprocess.Popen(['caffeinate','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    progress('generating')
    try:
        with cf.ProcessPoolExecutor(max_workers=cfg['workers'],initializer=init_worker,initargs=(cfg,)) as pool:
            iterator=iter(pending);active={}
            def submit():
                job=next(iterator,None)
                if job:active[pool.submit(job_worker,job)]=job['id']
            for _ in range(cfg['workers']*2):submit()
            while active:
                ready,_=cf.wait(active,timeout=1,return_when=cf.FIRST_COMPLETED)
                for future in ready:
                    expected=active.pop(future);ident,items=future.result();assert ident==expected
                    with con:
                        for record,metric in items:
                            con.execute('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?)',record)
                            con.execute('INSERT INTO provenance VALUES (?,?,?)',(metric['key'],metric['source_partition'],json.dumps(metric,separators=(',',':'))))
                        con.execute('INSERT INTO jobs VALUES (?,?)',(ident,len(items)))
                    completed+=1
                    with (out/'jobs.jsonl').open('a') as f:f.write(json.dumps(dict(job=ident,configs=len(items),seconds=sum(m['seconds'] for _,m in items)))+'\n')
                    submit()
                if time.time()-last>=cfg['progress_interval_seconds']:progress('generating')
        assert con.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        result=summarize(con)
        for s in cfg['sources']:
            g=result['partition:'+s['id']];assert g['configs']==s['expected_configs'] and g['actions']==s['expected_actions']
        expected_keys={key for j in jobs for key in j['keys']}
        assert {r[0] for r in con.execute('SELECT key FROM records')}==expected_keys and len(expected_keys)==21261
        for (detail,) in con.execute('SELECT detail FROM provenance'):
            r=json.loads(detail);assert (r['new_C'],r['new_cost'])<=(r['old_C'],r['old_cost'])
            assert (r['new_C'],r['new_cost'])<=(r['BHEFT_C'],r['BHEFT_cost'])
        write_json(out/'QUALITY_SUMMARY.json',dict(status='PASS',groups=result,no_regression=True,
            configs=21261,teacher_actions=1589632,training_started=False))
        con.commit();con.close()
        for s in cfg['sources']:assert sha256(ROOT/s['database'])==binding['inputs'][s['id']]
        write_json(out/'MANIFEST.json',dict(status='complete',schema=cfg['schema'],partitions=parts,
            database_sha256=sha256(out/'public.sqlite3'),binding_sha256=sha256(out/'INPUT_BINDING.json'),
            protocol=cfg['protocol'],code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            old_sources_unchanged=True,labels_from_VAL_or_TEST=False,training_started=False))
        con=sqlite3.connect(out/'public.sqlite3');progress('complete');con.close()
        print(json.dumps(dict(status='complete',groups={k:v for k,v in result.items() if k.startswith('partition:')})),flush=True)
    except BaseException:
        try:progress('failed')
        except BaseException:pass
        write_json(out/'FAILURE.json',dict(error=traceback.format_exc(),updated_unix=time.time()))
        raise
    finally:
        if awake.poll() is None:awake.terminate()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    cfg=json.loads(CONFIG.read_text());out=ROOT/cfg['run'];out.mkdir(parents=True,exist_ok=True)
    engineering(cfg,out) if args.check else build(cfg,out)
