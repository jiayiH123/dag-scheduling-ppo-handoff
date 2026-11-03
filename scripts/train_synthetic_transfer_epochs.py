#!/usr/bin/env python3
"""Seven full shuffled epochs; five complete test/probe observations per epoch."""
import argparse
import collections
import gzip
import json
import math
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from daglab.synthetic_transfer_data import config,connection,sha,write_json,digest,unpack
from daglab.epoch_training_plan import make_epoch_plan,observation_steps
from scripts.train_synthetic_transfer import Trial as PhysicalTrial,now
from scripts.check_direct_score import Bridge
EPOCH_CONFIG=ROOT/'configs/synthetic_transfer_epochs_20261031.json'


class EpochTrial(PhysicalTrial):
    def __init__(self,arm,resume=False,run_override=None,data_override=None,epochs_override=None,batch_override=None):
        import torch
        from daglab.state_score_torch import StateActor,bc_loss,distributions
        from daglab.module_lab.models import tensors
        arm={k:v for k,v in arm.items() if k!='weights'}
        self.epoch_cfg=json.loads(EPOCH_CONFIG.read_text());self.cfg=config();self.cfg['sources']=arm['sources']
        self.cfg['batch_size']=batch_override or self.epoch_cfg['batch_size'];self.epochs=epochs_override or self.epoch_cfg['epochs']
        self.arm=arm;self.torch=torch;self.bc_loss=bc_loss;self.distributions=distributions;self.tensors=tensors
        self.observation=dict(eval_sources=['synthetic','wfinstances','tpch_composite'],per_epoch=5,
            original_train_scope='fixed1536 original bases, one configuration each',full_test=True)
        self.data=Path(data_override) if data_override else ROOT/self.cfg['data' if arm['data_stage']=='base' else 'augmented_data']
        self.out=Path(run_override) if run_override else ROOT/self.cfg['run']/arm['id'];self.out.mkdir(parents=True,exist_ok=True)
        self.manifest=json.loads((self.data/'MANIFEST.json').read_text())
        assert self.manifest['status']=='complete' and sha(self.data/'public.sqlite3')==self.manifest['database_sha256']
        audit=json.loads((self.data/'FULL_AUDIT.json').read_text())
        assert audit['status']=='PASS' and audit['database_sha256']==self.manifest['database_sha256']
        assert audit['no_public_training_instances'] and audit['no_cross_role_topologies']
        self.con=connection(self.data);self.batches,self.source_for_key,self.plan_sha=make_epoch_plan(self.con,arm['sources'],self.cfg['seed'],self.epochs,self.cfg['batch_size'])
        self.corpus_size=len(self.source_for_key);self.updates_per_epoch=math.ceil(self.corpus_size/self.cfg['batch_size']);self.target=len(self.batches)
        self.observation_steps=set(observation_steps(self.updates_per_epoch,self.epochs,5))
        self.diagnostic_steps={self.updates_per_epoch*e for e in self.epoch_cfg['diagnostic_epochs'] if e<=self.epochs}
        if data_override is None:
            expected=self.epoch_cfg['arms'][arm['id']]
            assert self.corpus_size==expected['total_configs'] and self.target==expected['total_updates_7epochs']
            assert len(self.observation_steps)==36
            if arm['data_stage']=='augmented':
                path=ROOT/self.epoch_cfg['input_plan'];assert sha(path)==self.epoch_cfg['input_plan_sha256']
                plan=json.loads(path.read_text());extra=next(s for s in arm['sources'] if s!='synthetic')
                expected_inputs=path.parent/f'{arm["id"]}_INPUT_PLAN.jsonl.gz'
                assert sha(expected_inputs)==plan['arms'][arm['id']]['input_plan_sha256']
                with gzip.open(expected_inputs,'rt') as f:planned={r['key']:r['physical_sha256'] for r in map(json.loads,f)}
                actual=dict(self.con.execute('SELECT key,physical_sha FROM records WHERE source=? AND role="train"',(extra,)))
                assert actual==planned,'Labelled corpus differs from frozen physical input plan'
        self.binding=dict(data=sha(self.data/'MANIFEST.json'),epoch_config=sha(EPOCH_CONFIG),arm=arm,
            implementation=sha(Path(__file__)),physical_training_helpers=sha(ROOT/'scripts/train_synthetic_transfer.py'),
            sampler=sha(ROOT/'daglab/epoch_training_plan.py'),worker=sha(ROOT/'scripts/synthetic_transfer_worker.py'),
            data_helpers=sha(ROOT/'daglab/synthetic_transfer_data.py'))
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);random.seed(self.cfg['seed']);torch.manual_seed(self.cfg['seed'])
        self.model=StateActor('both');self.opt=torch.optim.Adam(self.model.parameters(),lr=self.cfg['lr'])
        self.worker=Bridge('scripts/synthetic_transfer_worker.py','--data',str(self.data));self.analysis_metadata={}
        self.step=0;self.actions=0;self.exposures=collections.Counter();self.unique=collections.defaultdict(set)
        self.training_seconds=0.;self.eval_seconds=0.;self.history=[];self.best=None;self.started=time.perf_counter();self.resumed_from=None
        if resume:self.restore()
        else:
            assert not (self.out/'metadata.json').exists(),'Use --resume; never overwrite a trial'
            write_json(self.out/'metadata.json',dict(status='initialized',created=now(),binding=self.binding,
                seed=self.cfg['seed'],architecture='StateActor both edge_attention32x2',parameters=sum(p.numel() for p in self.model.parameters()),
                optimizer=dict(name='Adam',lr=self.cfg['lr'],betas=[.9,.999],eps=1e-8,weight_decay=0,clip=1,scheduler=None),
                epochs=self.epochs,corpus_configs=self.corpus_size,updates_per_epoch=self.updates_per_epoch,total_updates=self.target,
                total_graph_presentations=self.corpus_size*self.epochs,batch_size=self.cfg['batch_size'],
                last_batch_graphs=self.corpus_size%self.cfg['batch_size'] or self.cfg['batch_size'],
                loss='mean teacher CE within graph; mean over actual batch size including epoch tail',
                sampling='full independent shuffle of all configurations once per epoch; no source reweighting',
                score_bounds=[8,8,2,2],plan_sha256=self.plan_sha,full_test_observation_steps=sorted(self.observation_steps),
                diagnostic_steps=sorted(self.diagnostic_steps),evaluation_used_for_tuning=True,automatic_extension=False,
                one_forward_per_graph=True,single_training_seed=True,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                initial_parameters_sha256=digest({k:v.tolist() for k,v in self.model.state_dict().items()})))
            write_json(self.out/'observation_metadata.json',self.observation)
        self.model.train()

    def publish(self,status,**extra):
        write_json(self.out/'progress.json',dict(status=status,updated=now(),step=self.step,target_steps=self.target,
            epoch_progress=self.step/self.updates_per_epoch,target_epochs=self.epochs,
            graph_presentations=sum(self.exposures.values()),actions=self.actions,source_exposures=dict(self.exposures),
            unique_configs_seen={s:len(v) for s,v in self.unique.items()},best=self.best,
            training_seconds=self.training_seconds,evaluation_seconds=self.eval_seconds,**extra))

    def update(self):
        begin=time.perf_counter();keys=self.batches[self.step];batch=len(keys)
        assert 1<=batch<=self.cfg['batch_size'];losses=collections.defaultdict(list);self.opt.zero_grad(set_to_none=True)
        for key in keys:
            r=self.row(key);assert r['meta']['role']=='train'
            loss=self.bc_loss(self.forward(r),r['decisions']);assert bool(self.torch.isfinite(loss))
            (loss/batch).backward();s=self.source_for_key[key];losses[s].append(float(loss.detach()))
            self.exposures[s]+=1;self.unique[s].add(key);self.actions+=len(r['decisions'])
        norm=float(self.torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.));assert math.isfinite(norm)
        self.opt.step();self.step+=1;self.training_seconds+=time.perf_counter()-begin
        record=dict(step=self.step,epoch=(self.step-1)//self.updates_per_epoch+1,epoch_step=(self.step-1)%self.updates_per_epoch+1,
            presentations=sum(self.exposures.values()),batch_graphs=batch,NLL=sum(map(sum,losses.values()))/batch,
            source_NLL={s:sum(v)/len(v) for s,v in losses.items()},gradient_norm=norm,lr=self.cfg['lr'],actions=self.actions,
            source_counts={s:len(v) for s,v in losses.items()},training_seconds=self.training_seconds,keys_sha256=digest(keys))
        with (self.out/'training.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        if self.step==1:
            self.checkpoint('initial_update');write_json(self.out/'startup_verified.json',dict(status='PASS',record=record,finite_gradients=True))
        return record

    def probe_original(self):
        begin=time.perf_counter();self.model.eval();keys=json.loads((self.data/'TRAIN_PROBE.json').read_text())['synthetic'];rows=[]
        self.publish('original_train_probe',cases=0,total=len(keys))
        with self.torch.no_grad():
            for start in range(0,len(keys),16):
                batch=[self.row(k) for k in keys[start:start+16]]
                results=self.worker(cmd='roll',items=[dict(key=r['id'],coefficients=self.forward(r).tolist()) for r in batch])
                for r,res in zip(batch,results):
                    assert res['key']==r['id'] and res['feasible']
                    rows.append(dict(key=r['id'],base_id=r['base_id'],C=res['C'],C_HEFT=res['C']/r['reference'],
                        Teacher64_HEFT=r['teacher']/r['reference'],feasible=True))
                if start%256==0:self.publish('original_train_probe',cases=len(rows),total=len(keys))
        summary=dict(configs=len(rows),bases=len({r['base_id'] for r in rows}),C_HEFT=sum(r['C_HEFT'] for r in rows)/len(rows),
            Teacher64_HEFT=sum(r['Teacher64_HEFT'] for r in rows)/len(rows))
        assert summary['configs']==summary['bases']
        path=self.out/f'original_train_{self.step:05d}.json.gz';tmp=path.with_name(path.name+'.tmp')
        with gzip.open(tmp,'wt') as f:json.dump(dict(step=self.step,summary=summary,rows=rows),f,allow_nan=False)
        tmp.replace(path);path=self.out/'train_original_curve.json';curve=json.loads(path.read_text()) if path.exists() else []
        curve=[p for p in curve if p['step']<self.step]+[dict(step=self.step,presentations=sum(self.exposures.values()),**summary)]
        write_json(path,curve);self.eval_seconds+=time.perf_counter()-begin;self.model.train();self.checkpoint('last');self.publish('training')

    def observe(self):
        marker=self.out/f'observation_complete_{self.step:05d}.json'
        if marker.exists():return
        if not any(p['step']==self.step for p in self.history):self.evaluate()
        path=self.out/'train_original_curve.json';curve=json.loads(path.read_text()) if path.exists() else []
        if not any(p['step']==self.step for p in curve):self.probe_original()
        if self.step in self.diagnostic_steps:
            path=self.out/'diagnostic_curve.json';curve=json.loads(path.read_text()) if path.exists() else []
            if not any(p['step']==self.step for p in curve):self.diagnose()
        self.plot();self.checkpoint('last');write_json(marker,dict(step=self.step,epoch=self.step/self.updates_per_epoch,complete=True))

    def run(self):
        if self.step==0 and not (self.out/'preflight.json').exists():self.gate()
        if self.step in self.observation_steps:self.observe()
        while self.step<self.target:
            self.update()
            if self.step%50==0:
                self.checkpoint('last');self.publish('training')
                if self.step not in self.observation_steps:self.plot()
            if self.step in self.observation_steps:self.observe()
        assert sum(self.exposures.values())==self.corpus_size*self.epochs
        assert sum(len(v) for v in self.unique.values())==self.corpus_size
        assert len(self.history)==len(self.observation_steps)
        self.checkpoint('last')
        write_json(self.out/'summary.json',dict(status='complete',updated=now(),arm=self.arm,epochs=self.epochs,steps=self.step,
            presentations=sum(self.exposures.values()),corpus_configs=self.corpus_size,source_exposures=dict(self.exposures),
            best=self.best,history=self.history,training_seconds=self.training_seconds,evaluation_seconds=self.eval_seconds,
            data_binding=self.binding,evaluation_used_for_tuning=True,single_seed_exploratory=True,automatic_extension=False))
        self.publish('complete')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    cfg=config();arm=next(r for r in cfg['arms'] if r['id']==a.arm)
    trial=EpochTrial(arm,a.resume)
    try:trial.run()
    finally:trial.close()
