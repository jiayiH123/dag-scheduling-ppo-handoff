#!/usr/bin/env python3
"""Finite two-method queues: rates, full-scale training, seeds, final reporting."""
import argparse
import concurrent.futures
import fcntl
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from daglab.faithful_external_data import Inputs,config,summarize
from daglab.faithful_external import CAI,GiPH,Problem,GiPHProblem,cai_input,torch,RUN
from daglab.sft_three_panel import write_json
from scripts.train_faithful_external_20261102 import evaluate,binding


def train_arm(method,rate,seed):
    name=f'{method}_s{seed}_lr{rate:g}'; out=RUN/name; out.mkdir(exist_ok=True)
    command=[config()['python'],str(ROOT/'scripts/train_faithful_external_20261102.py'),
             '--method',method,'--lr',str(rate),'--seed',str(seed),'--arm',name]
    for attempt in range(3):
        if (out/'COMPLETE.json').exists():
            return out,json.loads((out/'COMPLETE.json').read_text())
        with (out/'worker.log').open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            write_json(out/'PROCESS.json',dict(pid=process.pid,command=command,attempt=attempt))
            code=process.wait()
        if code==0 and (out/'COMPLETE.json').exists():
            return out,json.loads((out/'COMPLETE.json').read_text())
        if (out/'FAILURE.json').exists():
            failure=json.loads((out/'FAILURE.json').read_text())
            if 'Nonfinite' in failure.get('error',''):
                break
    return out,dict(status='failed')


def method_queue(method):
    torch.set_num_threads(1)
    cfg=config(); candidates=[]
    rates=cfg[method]['rates']
    if method=='giph':
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(train_arm,method,rate,11) for rate in rates]
            rate_results=[future.result() for future in futures]
    else:
        rate_results=[train_arm(method,rate,11) for rate in rates]
    for rate,(out,result) in zip(rates,rate_results):
        if result.get('status')=='complete':
            candidates.append((result['best']['primary'],rate,out))
    if not candidates:
        raise RuntimeError(f'All {method} rates failed; see preserved arm failures.')
    _,rate,out=min(candidates,key=lambda row:row[0])
    write_json(RUN/f'{method}_SELECTED_RECIPE.json',dict(rate=rate,selection='minimum random-start/base-macro VAL over completed seed11 recipes',arm=out.name))
    arms=[out]
    if method=='giph':
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(train_arm,method,rate,seed) for seed in [22,33]]
            confirmations=[future.result() for future in futures]
    else:
        confirmations=[train_arm(method,rate,seed) for seed in [22,33]]
    for directory,result in confirmations:
        if result.get('status')!='complete':
            raise RuntimeError(f'Selected recipe confirmation failed: {directory}')
        arms.append(directory)
    inputs=Inputs(); final=[]
    for directory in arms:
        complete=json.loads((directory/'COMPLETE.json').read_text())
        saved=torch.load(directory/'best.pt',weights_only=False)
        assert saved['binding']==binding(inputs)
        model=CAI(cai_input(Problem(inputs.instance(inputs.keys[0])))['numeric'].shape[-1]) if method=='cai' else GiPH()
        model.load_state_dict(saved['model'])
        observations=[]
        seeds=cfg['giph']['final_sampling_seeds'] if method=='giph' else [101]
        for seed in seeds:
            observations.append(evaluate(model,inputs,method,directory,complete['best']['unit'],
                                         action_seed=seed,tag=f'final_seed{seed}'))
        variant_names=list(observations[0]['variants'])
        averages={variant:{metric:sum(o['variants'][variant][metric] for o in observations)/len(observations)
                            for metric in ['base_macro','config_mean']} for variant in variant_names}
        final.append(dict(arm=directory.name,selected_unit=complete['best']['unit'],metrics=averages,
                          sampling_repetitions=len(seeds),training=complete))
    variants=list(final[0]['metrics'])
    means={variant:{metric:sum(row['metrics'][variant][metric] for row in final)/3
                    for metric in ['base_macro','config_mean']} for variant in variants}
    write_json(RUN/f'{method}_FINAL.json',dict(status='complete',selected_rate=rate,
        final_three_seed_mean=means,runs=final,configs=862,bases=500,
        VAL_used_for_recipe_and_checkpoint_selection=True,replicate_aggregation='mean, not best-of'))


def main():
    RUN.mkdir(exist_ok=True)
    lock=(RUN/'controller.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    checks=json.loads((RUN/'engineering/CHECKS.json').read_text()); assert checks['status']=='PASS'
    assert json.loads((RUN/'engineering/CAI_RESUME.json').read_text())['status']=='PASS'
    assert json.loads((RUN/'engineering/GIPH_UPDATE_PARITY.json').read_text())['status']=='PASS'
    inputs=Inputs(); inputs.audit()
    started=time.time(); write_json(RUN/'controller_status.json',dict(status='running',pid=os.getpid(),started_unix=started,
        protocol=config()['protocol'],queues=['cai','giph'],bounded=True))
    # Separate interpreters avoid shared model RNG. Each method has a finite queue.
    processes={}
    for method in ['cai','giph']:
        log=(RUN/f'{method}_queue.log').open('a')
        command=[config()['python'],str(Path(__file__).resolve()),'--method-queue',method]
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        processes[method]=(process,log)
        write_json(RUN/f'{method}_QUEUE_PROCESS.json',dict(pid=process.pid,command=command))
    heartbeat=0; report=0
    while any(p.poll() is None for p,_ in processes.values()):
        now=time.time()
        if now-report>=120:
            subprocess.run([config()['plot_python'],str(ROOT/'scripts/report_faithful_external_20261102.py')],cwd=ROOT,check=False)
            report=now
        if now-heartbeat>=1200:
            write_json(RUN/'HEARTBEAT.json',dict(unix=now,elapsed_seconds=now-started,
                methods={name:dict(pid=p.pid,returncode=p.poll()) for name,(p,_) in processes.items()}))
            heartbeat=now
        time.sleep(5)
    codes={name:p.returncode for name,(p,_) in processes.items()}
    for _,log in processes.values():log.close()
    status='complete' if all(code==0 for code in codes.values()) else 'failed'
    write_json(RUN/'controller_status.json',dict(status=status,pid=os.getpid(),returncodes=codes,elapsed_seconds=time.time()-started))
    subprocess.run([config()['plot_python'],str(ROOT/'scripts/report_faithful_external_20261102.py')],cwd=ROOT,check=True)
    if status!='complete':raise RuntimeError(str(codes))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--method-queue',choices=['cai','giph']); args=parser.parse_args()
    try:
        method_queue(args.method_queue) if args.method_queue else main()
    except BaseException:
        write_json(RUN/((args.method_queue+'_QUEUE_FAILURE.json') if args.method_queue else 'CONTROLLER_FAILURE.json'),
                   dict(error=traceback.format_exc(),unix=time.time()))
        raise
