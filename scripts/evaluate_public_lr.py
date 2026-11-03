#!/usr/bin/env python3
"""Frozen one-forward ABC policies and fresh baselines on public resource-v2 DEV."""
import argparse
import copy
import gzip
import hashlib
import json
import math
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
TRAIN=ROOT/'runs/scratch_lr_three_arm_20261028'
DATA=ROOT/'data/public_resource_v2_20261028'
OUT=ROOT/'runs/public_lr_evaluation_20261028'
PROTOCOL='protocols/public_lr_evaluation_20261028.md'
SIM='/opt/miniconda3/envs/dag-research/bin/python'


def save(path,value):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def load_inputs():
    summary=json.loads((DATA/'SUMMARY.json').read_text());rows=[];hashes=[]
    for file in summary['files']:
        path=DATA/file['path'];sha=hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha==file['sha256'];hashes.append(dict(path=str(path.relative_to(ROOT)),sha256=sha))
        with gzip.open(path,'rt') as stream:
            part=[json.loads(line) for line in stream]
        assert len(part)==file['rows'];rows.extend(part)
    assert len(rows)==len({r['key'] for r in rows})==1000
    assert all(r['key'].startswith('val_resourcev2_') and 'supervision' not in r for r in rows)
    first={}
    for row in rows:
        group=row['provenance']['dataset'],row['instance_sparse']['resource_adaptation']['pressure']
        first.setdefault(group,row['key'])
    return rows,hashes,set(first.values())


def sim_worker():
    from daglab.offline_data import dense_instance
    from daglab.reference import graph_features
    from daglab.state_score import StateScoreDecoder
    from daglab.module_lab.features import enrich
    rows,hashes,checks=load_inputs();decoder=None;row=None
    for line in sys.stdin:
        q=json.loads(line);cmd=q['cmd']
        if cmd=='init':
            result=dict(count=len(rows),hashes=hashes,checks=sorted(checks))
        elif cmd=='get':
            row=rows[q['index']];begin=time.perf_counter();inst=dense_instance(row,max_tasks=300)
            decoder=StateScoreDecoder(inst);prepare_seconds=time.perf_counter()-begin
            graph,feature_seconds=graph_features(inst);begin=time.perf_counter()
            graph=enrich(dict(instance=inst,graph=graph),decoder);feature_seconds+=time.perf_counter()-begin
            n,k=decoder.n,decoder.k
            comm=[z for u,v in inst['edges'] for p in range(k) for h in range(k) if p!=h for z in [decoder.info['comm'][u,v][p][h]]]
            compute=sum(sum(ds) for ds in decoder.info['duration'])/(n*k)
            meta=dict(key=row['key'],base_id=row['provenance']['base_id'],dataset=row['provenance']['dataset'],
                family=inst['family'],topology=inst['topology_fingerprint'],n=n,k=k,eta=inst['budget_eta'],
                pressure=inst['resource_adaptation']['pressure'],network=inst['link_profile'],budget=inst['budget'],
                actual_ccr=sum(comm)/max(1,len(comm))/compute)
            result=dict(meta=meta,graph=graph,prepare_seconds=prepare_seconds,feature_seconds=feature_seconds,check=row['key'] in checks)
        elif cmd=='roll':
            result={}
            for name,x in q['coefficients'].items():
                begin=time.perf_counter();r=decoder.rollout(x)
                tasks=r['schedule']['tasks']
                value=dict(C=r['makespan'],cost=r['cost'],decode_seconds=time.perf_counter()-begin,
                    feasible=True,mapping_sha256=hashlib.sha256(json.dumps([t['device'] for t in tasks]).encode()).hexdigest(),
                    resource_wait_sum=sum(t['resource_wait'] for t in tasks))
                if row['key'] in checks:value['decisions']=r['decisions']
                result[name]=value
        elif cmd=='close':
            print('{}',flush=True);return
        else:raise ValueError(cmd)
        print(json.dumps(result,allow_nan=False),flush=True)


def audit_and_models(out):
    import torch
    from daglab.state_score_torch import StateActor
    from scripts.run_scratch_lr_three_arm import ARMS,rate,same
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    summary=json.loads((TRAIN/'summary.json').read_text());meta=json.loads((TRAIN/'metadata.json').read_text())
    assert summary['status']=='complete' and summary['added_updates_per_arm']==2158
    expected=sorted(set([0,1079,2158]+list(range(50,2159,50))))
    assert [h['new_update'] for h in summary['history']]==expected
    curves=[json.loads(line) for line in (TRAIN/'training_curves.jsonl').open()]
    assert [r['new_update'] for r in curves]==list(range(1,2159))
    assert sum(r['batch'] for r in curves)==34522 and curves[-1]['added_actions']==2245720
    assert all(r['arms'][a]['LR']==[rate(a,r['new_update'])] for r in curves for a in ARMS)
    source=ROOT/meta['source'];assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['source_sha256']
    ck=torch.load(source,map_location='cpu',weights_only=False)
    for row in json.loads((TRAIN/'exposure.json').read_text()):
        assert row['configs']==17261 and row['actions']==1122860
    grid_checks=0
    for update in expected:
        ev=json.loads((TRAIN/f'eval_{update:04d}.json').read_text())
        for arm in ARMS:
            path=TRAIN/f'{arm}_update{update:04d}.pt';c=torch.load(path,map_location='cpu',weights_only=False)
            assert c['new_update']==update and c['step']==5500+update
            assert all(bool(torch.isfinite(v).all()) for v in c['model'].values())
            m=StateActor('both');m.load_state_dict(c['model'],strict=True)
            assert all(float(v['step'])==5500+update for v in c['optimizer']['state'].values())
            vals=ev['VAL'][arm];assert len(vals)==len({v['id'] for v in vals})==862
            assert abs(sum(v['C']/v['HEFT'] for v in vals)/862-ev['metrics'][arm]['C_HEFT'])<1e-12
            grid_checks+=1
        for tag in ['best','last']:
            pass
    aliases={'source5500':source}
    for letter,arm in zip('ABC',ARMS):
        for tag in ['best','last']:
            path=TRAIN/f'{arm}_{tag}.pt';c=torch.load(path,map_location='cpu',weights_only=False)
            update=summary['best'][arm]['new_update'] if tag=='best' else 2158
            assert c['new_update']==update
            grid=torch.load(TRAIN/f'{arm}_update{update:04d}.pt',map_location='cpu',weights_only=False)
            assert same(c['model'],grid['model'],torch) and same(c['optimizer'],grid['optimizer'],torch)
            aliases[letter+'_'+tag]=path
        assert min(summary['history'],key=lambda h:h['metrics'][arm]['C_HEFT'])['new_update']==summary['best'][arm]['new_update']
    def fingerprint(state):
        h=hashlib.sha256()
        for name,t in state.items():
            h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
            h.update(bytes(t.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()))
        return h.hexdigest()
    models={};seen={};mapping={};bindings={}
    for alias,path in aliases.items():
        c=torch.load(path,map_location='cpu',weights_only=False);fp=fingerprint(c['model'])
        if fp not in seen:
            seen[fp]=alias;m=StateActor('both');m.load_state_dict(c['model']);models[alias]=m.eval()
        mapping[alias]=seen[fp]
        bindings[alias]=dict(path=str(path.relative_to(ROOT)),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            model_sha256=fp,global_step=c['step'],canonical=seen[fp])
    assert len(models)==5 and mapping['B_best']==mapping['C_best']=='source5500'
    assert all(sum(p.numel() for p in m.parameters())==32612 for m in models.values())
    save(out/'ABC_audit.json',dict(status='PASS',evaluated_grid_checkpoint_checks=grid_checks,
        actual_update_count=2158,configs_per_arm=34522,LR_schedule_verified=True,source_unchanged=True,
        all_saved_best_last_match_grids=True,aliases=bindings,unique_models=len(models),
        initial_first_update_audit=json.loads((TRAIN/'startup_verified.json').read_text()),
        best=summary['best'],last=summary['history'][-1]['metrics']))
    return torch,models,mapping,bindings


def model_stage(out):
    from daglab.module_lab.models import tensors
    from daglab.state_score_torch import distributions
    torch,models,aliases,bindings=audit_and_models(out)
    worker=subprocess.Popen([SIM,str(Path(__file__).resolve()),'--sim-worker'],cwd=ROOT,
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    def call(**q):
        worker.stdin.write(json.dumps(q,allow_nan=False)+'\n');worker.stdin.flush();line=worker.stdout.readline()
        if not line:raise RuntimeError('physical worker terminated')
        return json.loads(line)
    started=time.perf_counter();checks=[]
    try:
        identity=call(cmd='init')
        save(out/'metadata.json',dict(protocol=PROTOCOL,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            data=identity,models=bindings,aliases=aliases,parameters=32612,torch_threads=1,
            candidate_selection='frozen old862VAL best/last plus source; no new public score search',
            deployment='one forward and one deterministic construction',TEST=False,training_updates=0))
        with (out/'models.jsonl').open('w') as stream:
            for i in range(identity['count']):
                payload=call(cmd='get',index=i);graph=tensors(payload['graph'],'cpu');xs={};forward_seconds={}
                with torch.no_grad():
                    for name,m in models.items():
                        before=m.forward_calls,m.base.forward_calls;begin=time.perf_counter();x=m(graph)
                        forward_seconds[name]=time.perf_counter()-begin
                        assert (m.forward_calls-before[0],m.base.forward_calls-before[1])==(1,1)
                        assert bool(torch.isfinite(x).all()) and tuple(x.shape)==(payload['meta']['n']*payload['meta']['k'],4)
                        xs[name]=x
                result=call(cmd='roll',coefficients={k:x.tolist() for k,x in xs.items()})
                for name,r in result.items():
                    r['forward_seconds']=forward_seconds[name];r['actor_calls']=r['base_calls']=1
                    if 'decisions' in r:
                        ds=r.pop('decisions');lp,_=distributions(xs[name],ds)
                        selected=lp.gather(1,torch.tensor([d['selected'] for d in ds])[:,None]).squeeze(1)
                        err=float((selected-torch.tensor([d['selected_log_probability'] for d in ds])).abs().max())
                        assert err<5e-5;checks.append(dict(key=payload['meta']['key'],model=name,error=err))
                value=dict(**payload['meta'],shared_prepare_seconds=payload['prepare_seconds'],shared_feature_seconds=payload['feature_seconds'],models=result)
                stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush()
                if (i+1)%50==0:
                    save(out/'progress.json',dict(stage='model_evaluation',completed=i+1,total=1000,seconds=time.perf_counter()-started))
                    print('MODELS',i+1,'/1000',round(time.perf_counter()-started,1),'s',flush=True)
        save(out/'model_checks.json',dict(status='PASS',rows=1000,unique_model_schedules=5000,
            exactly_one_actor_and_base_forward=True,all_physical_validation=True,probability_checks=checks,
            source_checkpoint_files_unchanged=all(hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['file_sha256'] for r in bindings.values()),
            seconds=time.perf_counter()-started))
    finally:
        if worker.poll() is None:call(cmd='close');worker.wait(timeout=30)


def one_baseline(job):
    row,original_check=job
    from daglab.offline_data import dense_instance
    from daglab.reference import core,portfolio,mapping_of
    from daglab.state_score import StateScoreDecoder
    from daglab.diagnostic_decoder import DiagnosticDecoder
    from daglab.module_lab.decoder import NormalizedEFTDecoder
    from daglab.structured_synthetic import teacher_with_quality
    started=time.perf_counter();inst=dense_instance(row,max_tasks=300);dec=StateScoreDecoder(inst)
    schedules,portfolio_seconds=portfolio(inst)
    for s in schedules:core.validate(inst,s)
    values={name:dict(C=s['makespan'],cost=s['cost'],seconds=s['runtime_seconds'],feasible=True)
        for name,s in [('HEFT',schedules[0]),('PEFT',schedules[1]),('MSLBL',schedules[4])]}
    greedy=core.schedule(inst,'Greedy');values['Greedy']=dict(C=greedy['makespan'],cost=greedy['cost'],seconds=greedy['runtime_seconds'],feasible=True)
    native=min(schedules,key=lambda s:(s['makespan'],s['cost']))
    values['Portfolio_native']=dict(C=native['makespan'],cost=native['cost'],seconds=portfolio_seconds,feasible=True)
    maps=[mapping_of(s) for s in schedules];fixed=[dec.engine.evaluate(m,dec.order) for m in maps]
    winner=min(range(10),key=lambda j:fixed[j]);mapping=maps[winner];score=fixed[winner]
    solution=dec.engine.evaluate(mapping,dec.order,materialize=True);core.validate(inst,solution)
    values['Portfolio_fixed']=dict(C=score[0],cost=score[1],feasible=True)
    begin=time.perf_counter();rng=random.Random(inst['seed']+3917);feasible=0
    for _ in range(64):
        changed=list(mapping)
        for v in rng.sample(range(dec.n),rng.choice([1,2])):changed[v]=rng.choice(dec.info['eligible'][v])
        if not dec.engine.feasible(changed):continue
        feasible+=1;candidate=dec.engine.evaluate(changed,dec.order)
        if candidate<score:mapping,score=changed,candidate
    solution=dec.engine.evaluate(mapping,dec.order,materialize=True);core.validate(inst,solution)
    replay=dec.rollout([[0.,0.,0.,0.]]*(dec.n*dec.k),forced_mapping=mapping)
    assert (replay['makespan'],replay['cost'])==(score[0],score[1])
    assert replay['schedule']['tasks']==solution['tasks']
    values['Teacher64']=dict(C=score[0],cost=score[1],seconds_after_portfolio=time.perf_counter()-begin,
        feasible=True,feasible_proposals=feasible)
    if original_check:
        names=json.loads((ROOT/'data/structured_scheduling_v1/CONFIG.json').read_text())['teacher_rules']
        original,omap,_=teacher_with_quality(inst,NormalizedEFTDecoder(inst),names)
        assert (original['makespan'],original['cost'],omap)==(score[0],score[1],mapping)
    rand=DiagnosticDecoder(inst);random_rows=[]
    for j in range(5):
        seed=131+inst['seed']*17+j
        r=rand.rollout([[0.,0.,0.,0.]]*(dec.n*dec.k),uniform=True,seed=seed)
        random_rows.append(dict(seed=seed,C=r['makespan'],cost=r['cost'],feasible=True))
    values['RandomPoolMean5_fixedHEFT']=dict(C=sum(r['C'] for r in random_rows)/5,
        cost=sum(r['cost'] for r in random_rows)/5,components=random_rows,feasible=True)
    assert values['Teacher64']['C']<=values['Portfolio_fixed']['C']<=values['HEFT']['C']
    return dict(key=row['key'],values=values,original_teacher_exact=original_check,seconds=time.perf_counter()-started)


def baseline_stage(out,workers):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    rows,_,checks=load_inputs();started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool,(out/'baselines.jsonl').open('w') as stream:
        futures=[pool.submit(one_baseline,(r,r['key'] in checks)) for r in rows]
        for i,future in enumerate(as_completed(futures),1):
            row=future.result();stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if i%25==0:
                save(out/'progress.json',dict(stage='baselines',completed=i,total=1000,workers=workers,seconds=time.perf_counter()-started))
                print('BASELINES',i,'/1000',round(time.perf_counter()-started,1),'s',flush=True)
    save(out/'baseline_checks.json',dict(status='PASS',rows=1000,all_fresh=True,all_schedules_validated=True,
        original_teacher_crosschecks=len(checks),random_components=5000,random_not_bestof5=True,
        workers=workers,seconds=time.perf_counter()-started))


def main(args):
    if args.sim_worker:sim_worker();return
    if args.baselines:baseline_stage(args.output.resolve(),args.workers);return
    out=args.output.resolve();out.mkdir(exist_ok=False)
    started=time.perf_counter()
    try:
        model_stage(out)
        subprocess.run([SIM,str(Path(__file__).resolve()),'--baselines','--output',str(out),'--workers',str(args.workers)],cwd=ROOT,check=True)
        save(out/'execution_complete.json',dict(status='complete',models=1000,baselines=1000,total_seconds=time.perf_counter()-started,TEST=False))
        print('COMPLETE execution',round(time.perf_counter()-started,1),'seconds',flush=True)
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sim-worker',action='store_true');p.add_argument('--baselines',action='store_true')
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--workers',type=int,default=4)
    main(p.parse_args())
