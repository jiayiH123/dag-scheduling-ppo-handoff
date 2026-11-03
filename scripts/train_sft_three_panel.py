#!/usr/bin/env python3
"""Old S0 training recipe, new full three-panel VAL every100 updates."""
import argparse
import collections
import gzip
import json
import math
from functools import lru_cache
from pathlib import Path
import random
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from daglab.sft_three_panel import CONFIG,Store,config,panel_summary,write_json
from daglab.structured_dataset import sha256
from daglab.epoch_training_plan import make_epoch_plan
from daglab.offline_data import digest
from scripts.train_synthetic_transfer_epochs import EpochTrial
from scripts.train_synthetic_transfer import now
from scripts.check_direct_score import Bridge


def save_gzip(path,value):
    tmp=path.with_name(path.name+'.tmp')
    with gzip.open(tmp,'wt') as f:json.dump(value,f,allow_nan=False)
    tmp.replace(path)


class ThreePanelTrial(EpochTrial):
    def __init__(self,resume=False,out_override=None):
        import torch
        from daglab.state_score_torch import StateActor,bc_loss,distributions
        from daglab.module_lab.models import tensors
        self.cfg=config();self.cfg['lr']=self.cfg['optimizer']['lr'];self.cfg['sources']=['synthetic']
        self.epochs=self.cfg['epochs'];self.arm=dict(id='three_panel_sft',sources=['synthetic'],data_stage='base')
        self.torch=torch;self.bc_loss=bc_loss;self.distributions=distributions;self.tensors=tensors
        self.out=Path(out_override) if out_override else ROOT/self.cfg['run'];self.out.mkdir(parents=True,exist_ok=True)
        self.data=ROOT/self.cfg['training_cache'];self.store=Store(self.cfg);self.con=self.store.train
        overlay_manifest=ROOT/self.cfg['data']/'MANIFEST.json'
        assert json.loads(overlay_manifest.read_text())['status']=='complete'
        self.batches,self.source_for_key,self.plan_sha=make_epoch_plan(self.con,['synthetic'],self.cfg['seed'],self.epochs,self.cfg['batch_size'])
        self.corpus_size=len(self.source_for_key);self.updates_per_epoch=math.ceil(self.corpus_size/self.cfg['batch_size']);self.target=len(self.batches)
        assert (self.corpus_size,self.updates_per_epoch,self.target)==(17261,1079,7553)
        self.observation_steps={0,self.target,*range(self.cfg['eval_every'],self.target+1,self.cfg['eval_every'])}
        self.diagnostic_steps=set();self.observation=dict(eval_sources=list(self.store.panel_keys),eval_every=100)
        self.binding=dict(config=sha256(CONFIG),scope_lock=sha256(ROOT/self.cfg['scope_lock']),
            source_training_manifest=sha256(self.data/'MANIFEST.json'),overlay_manifest=sha256(overlay_manifest),
            implementation=sha256(Path(__file__)),worker=sha256(ROOT/'scripts/sft_three_panel_worker.py'),
            helpers=sha256(ROOT/'daglab/sft_three_panel.py'),wf_metrics=sha256(ROOT/'daglab/wf_panels.py'),
            epoch_update=sha256(ROOT/'scripts/train_synthetic_transfer_epochs.py'),
            physical_helpers=sha256(ROOT/'scripts/train_synthetic_transfer.py'),sampler=sha256(ROOT/'daglab/epoch_training_plan.py'))
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        random.seed(self.cfg['seed']);torch.manual_seed(self.cfg['seed'])
        self.model=StateActor('both')
        opt=self.cfg['optimizer'];self.opt=torch.optim.Adam(self.model.parameters(),lr=opt['lr'],betas=tuple(opt['betas']),eps=opt['eps'],weight_decay=opt['weight_decay'])
        assert sum(p.numel() for p in self.model.parameters())==32612
        self.worker=Bridge('scripts/sft_three_panel_worker.py');self.analysis_metadata={}
        self.step=0;self.actions=0;self.exposures=collections.Counter();self.unique=collections.defaultdict(set)
        self.training_seconds=0.;self.eval_seconds=0.;self.history=[];self.best=None;self.best_by_panel={}
        self.started=time.perf_counter();self.resumed_from=None
        old=json.loads((ROOT/'runs/synthetic_transfer_20261031/S0_original/metadata.json').read_text())
        init_sha=digest({k:v.tolist() for k,v in self.model.state_dict().items()})
        assert self.plan_sha==old['plan_sha256'] and init_sha==old['initial_parameters_sha256']
        if resume:
            self.restore()
            saved=torch.load(self.out/'last.pt',map_location='cpu',weights_only=False)
            self.best_by_panel=saved.get('best_by_panel',{})
            self.trim_auxiliary_logs()
        else:
            assert not (self.out/'metadata.json').exists(),'Use --resume; preserve prior output'
            write_json(self.out/'config.json',config())
            write_json(self.out/'metadata.json',dict(created=now(),binding=self.binding,seed=self.cfg['seed'],
                architecture=self.cfg['architecture'],parameters=32612,optimizer=opt,epochs=7,batch_size=16,
                corpus_configs=17261,updates_per_epoch=1079,total_updates=7553,last_batch_graphs=13,
                total_graph_presentations=120827,plan_sha256=self.plan_sha,initial_parameters_sha256=init_sha,
                initial_and_epoch_plan_exactly_match_old_S0=True,validation_counts={k:len(v) for k,v in self.store.panel_keys.items()},
                observation_steps=sorted(self.observation_steps),evaluation_used_for_tuning=True,
                best_alias_primary_panel=self.cfg['best_alias_primary_panel'],best_for_each_panel=True,
                one_forward_per_graph=True,device='cpu',torch_version=torch.__version__,
                train_probe=1536,scope='single-seed exploratory recipe reproduction, new validation scope',
                protocol_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        self.model.train()

    @lru_cache(maxsize=32)
    def row(self,key):return self.store.row(key)

    def checkpoint(self,name):
        self.torch.save(dict(model=self.model.state_dict(),optimizer=self.opt.state_dict(),step=self.step,actions=self.actions,
            exposures=dict(self.exposures),unique={s:sorted(v) for s,v in self.unique.items()},history=self.history,
            best=self.best,best_by_panel=self.best_by_panel,training_seconds=self.training_seconds,eval_seconds=self.eval_seconds,
            torch_rng=self.torch.get_rng_state(),python_rng=random.getstate(),plan_sha=self.plan_sha,binding=self.binding),
            self.out/(name+'.tmp.pt'))
        (self.out/(name+'.tmp.pt')).replace(self.out/(name+'.pt'))

    def trim_auxiliary_logs(self):
        for name in ('train_original_curve.json',*('val_'+p+'_curve.json' for p in self.store.panel_keys)):
            path=self.out/name
            if path.exists():write_json(path,[p for p in json.loads(path.read_text()) if p['step']<=self.step])

    def evaluate(self):
        begin=time.perf_counter();self.model.eval();metrics={};improved=[]
        with self.torch.no_grad():
            for panel,keys in self.store.panel_keys.items():
                rows=[];self.publish('evaluating',evaluation_source=panel,panel_cases=0,panel_total=len(keys))
                for start in range(0,len(keys),16):
                    batch=[self.row(k) for k in keys[start:start+16]]
                    items=[dict(key=r['id'],coefficients=self.forward(r).tolist()) for r in batch]
                    results=self.worker(cmd='roll',items=items)
                    for r,res in zip(batch,results):
                        assert r['id']==res['key'] and res['feasible']
                        rows.append(dict(**r['meta'],C=res['C'],cost=res['cost'],feasible=True,references=r['references']))
                    if start%256==0:self.publish('evaluating',evaluation_source=panel,panel_cases=len(rows),panel_total=len(keys))
                assert len(rows)==len(keys)
                metrics[panel]=panel_summary(panel,rows)
                point=dict(step=self.step,epoch_progress=self.step/self.updates_per_epoch,**metrics[panel])
                save_gzip(self.out/f'val_{panel}_{self.step:05d}.json.gz',dict(**point,rows=rows))
                path=self.out/f'val_{panel}_curve.json';curve=json.loads(path.read_text()) if path.exists() else []
                write_json(path,[p for p in curve if p['step']<self.step]+[point])
                ratio=metrics[panel]['C_over']['HEFT']
                if panel not in self.best_by_panel or ratio<self.best_by_panel[panel]['C_HEFT']:
                    self.best_by_panel[panel]=dict(step=self.step,C_HEFT=ratio);improved.append(panel)
        primary=self.cfg['best_alias_primary_panel'];objective=metrics[primary]['C_over']['HEFT']
        point=dict(step=self.step,presentations=sum(self.exposures.values()),objective=objective,metrics=metrics)
        self.history=sorted([p for p in self.history if p['step']!=self.step]+[point],key=lambda p:p['step'])
        is_best=self.best is None or objective<self.best['objective']
        if is_best:self.best=point
        self.eval_seconds+=time.perf_counter()-begin
        for panel in improved:self.checkpoint('best_'+panel)
        if is_best:self.checkpoint('best')
        write_json(self.out/'curve.json',self.history);write_json(self.out/'best_by_panel.json',self.best_by_panel)
        self.checkpoint('last');self.checkpoint(f'step{self.step:05d}')
        self.model.train();self.publish('training')
        print('VAL',self.step,{p:m['C_over'] for p,m in metrics.items()},flush=True)

    def plot(self):
        subprocess.run([self.cfg['sim_python'],str(ROOT/'figures/gen_fig_sft_three_panel_20261101.py'),
                        '--run',str(self.out)],cwd=ROOT,check=True)

    def observe(self):
        marker=self.out/f'observation_complete_{self.step:05d}.json'
        if marker.exists():return
        if not any(p['step']==self.step for p in self.history):self.evaluate()
        curve_path=self.out/'train_original_curve.json';curve=json.loads(curve_path.read_text()) if curve_path.exists() else []
        if not any(p['step']==self.step for p in curve):self.probe_original()
        self.plot();self.checkpoint('last');write_json(marker,dict(step=self.step,complete=True))

    def close(self):
        self.worker.close();self.store.close();self.row.cache_clear()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
    trial=ThreePanelTrial(a.resume)
    try:trial.run()
    finally:trial.close()
