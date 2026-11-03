#!/usr/bin/env python3
"""V5 synthetic-only full-corpus one-forward SFT with exact optimizer/RNG/sampler resumption."""
import argparse
import collections
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from daglab.synthetic_transfer_data import CONFIG,config,connection,unpack,sha,write_json,digest,grouped_metrics
from scripts.check_direct_score import Bridge
OBSERVATION_CONFIG=ROOT/'configs/synthetic_transfer_observation_20261031.json'


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def make_plan(con,weights,seed,total):
    """Separate reproducible source and base streams; rotate all variants per base."""
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    source_for_key={}
    for key,source,base in con.execute('SELECT key,source,base_id FROM records WHERE role="train" ORDER BY key'):
        groups[source][base].append(key);source_for_key[key]=source
    def stream(source):
        variants={b:sorted(keys,key=lambda k:digest([seed,'variant',k])) for b,keys in groups[source].items()}
        cycle=0
        while True:
            bases=sorted(groups[source]);random.Random(f'{seed}:{source}:basecycle:{cycle}').shuffle(bases)
            for b in bases:yield variants[b][cycle%len(variants[b])]
            cycle+=1
    sources=[s for s in config()['sources'] if s in weights];streams={s:stream(s) for s in sources}
    rng=random.Random(seed+30000)
    tags=rng.choices(sources,weights=[weights[s] for s in sources],k=total)
    keys=[next(streams[s]) for s in tags]
    return keys,source_for_key,digest(keys)


class Trial:
    def __init__(self,arm,target,resume=False,run_override=None,data_override=None):
        import torch
        from daglab.state_score_torch import StateActor,bc_loss,distributions
        from daglab.module_lab.models import tensors
        self.torch=torch;self.cfg=config();self.cfg['sources']=arm['sources'];self.arm=arm;self.target=target;self.bc_loss=bc_loss;self.distributions=distributions;self.tensors=tensors
        self.observation=json.loads(OBSERVATION_CONFIG.read_text())
        for key in ['eval_every','diagnostic_every']:self.cfg[key]=self.observation[key]
        self.data=Path(data_override) if data_override else ROOT/self.cfg['data' if arm['data_stage']=='base' else 'augmented_data']
        self.out=Path(run_override) if run_override else ROOT/self.cfg['run']/arm['id']
        self.out.mkdir(parents=True,exist_ok=True)
        self.manifest=json.loads((self.data/'MANIFEST.json').read_text())
        assert self.manifest['status']=='complete'
        assert self.manifest['database_sha256']==sha(self.data/'public.sqlite3')
        audit=json.loads((self.data/'FULL_AUDIT.json').read_text())
        assert audit['status']=='PASS' and audit['database_sha256']==self.manifest['database_sha256']
        assert audit['no_public_training_instances'] and audit['no_cross_role_topologies']
        assert self.manifest['evaluation_used_for_tuning'] is True
        self.analysis_metadata=json.loads((self.data/'ANALYSIS_METADATA.json').read_text()) if (self.data/'ANALYSIS_METADATA.json').exists() else {}
        self.binding=dict(data=sha(self.data/'MANIFEST.json'),config=sha(CONFIG),arm=arm,
                          implementation=sha(Path(__file__)),worker=sha(ROOT/'scripts/synthetic_transfer_worker.py'),helpers=sha(ROOT/'daglab/synthetic_transfer_data.py'),
                          observation_config=sha(OBSERVATION_CONFIG))
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);random.seed(self.cfg['seed']);torch.manual_seed(self.cfg['seed'])
        self.model=StateActor('both');self.opt=torch.optim.Adam(self.model.parameters(),lr=self.cfg['lr'])
        self.con=connection(self.data)
        self.plan,self.source_for_key,self.plan_sha=make_plan(self.con,dict(zip(arm['sources'],arm['weights'])),self.cfg['seed'],self.cfg['extension_steps']*self.cfg['batch_size'])
        self.worker=Bridge('scripts/synthetic_transfer_worker.py','--data',str(self.data))
        self.step=0;self.actions=0;self.exposures=collections.Counter();self.unique=collections.defaultdict(set)
        self.training_seconds=0.;self.eval_seconds=0.;self.history=[];self.best=None;self.started=time.perf_counter()
        self.resumed_from=None
        if resume:
            self.restore()
        else:
            assert not (self.out/'metadata.json').exists(),'Use --resume; no silent overwrite'
            write_json(self.out/'metadata.json',dict(status='initialized',created=now(),binding=self.binding,seed=self.cfg['seed'],
                architecture='StateActor both edge_attention32x2',parameters=sum(p.numel() for p in self.model.parameters()),
                optimizer=dict(name='Adam',lr=self.cfg['lr'],betas=[.9,.999],eps=1e-8,weight_decay=0,clip=1,scheduler=None),
                loss='mean teacher CE within graph, equal graph weight',source_weights=arm['weights'],
                source_sampling='weighted draws; source-specific uniform base cycles and rotating variants',
                score_bounds=[8,8,2,2],batch_size=self.cfg['batch_size'],minimum_updates=self.cfg['steps'],extension_updates=self.cfg['extension_steps'],
                data_manifest=self.manifest,plan_sha256=self.plan_sha,plan_total_graphs=len(self.plan),
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                initial_parameters_sha256=digest({k:v.tolist() for k,v in self.model.state_dict().items()}),
                one_forward=True,evaluation_used_for_tuning=True,scope='exploratory seed28; full-size synthetic-only transfer'))
            write_json(self.out/'observation_metadata.json',self.observation)
        self.model.train()

    @lru_cache(maxsize=32)
    def row(self,key):
        r=self.con.execute('SELECT payload,role FROM records WHERE key=?',(key,)).fetchone()
        assert r is not None and r[1] in ['train','eval']
        value=unpack(r[0]);value['meta'].update(self.analysis_metadata.get(key,{}));return value

    def forward(self,row):
        before=(self.model.forward_calls,self.model.base.forward_calls)
        x=self.model(self.tensors(row['graph'],'cpu'))
        assert (self.model.forward_calls-before[0],self.model.base.forward_calls-before[1])==(1,1)
        assert x.shape==(row['graph']['n']*row['graph']['k'],4) and bool(self.torch.isfinite(x).all())
        return x

    def checkpoint(self,name):
        p=self.out/(name+'.pt');tmp=p.with_suffix('.tmp.pt')
        self.torch.save(dict(model=self.model.state_dict(),optimizer=self.opt.state_dict(),step=self.step,actions=self.actions,
            exposures=dict(self.exposures),unique={s:sorted(v) for s,v in self.unique.items()},history=self.history,best=self.best,
            training_seconds=self.training_seconds,eval_seconds=self.eval_seconds,torch_rng=self.torch.get_rng_state(),
            python_rng=random.getstate(),plan_sha=self.plan_sha,binding=self.binding),tmp)
        tmp.replace(p)

    def restore(self):
        path=self.out/'last.pt';s=self.torch.load(path,map_location='cpu',weights_only=False)
        assert s['binding']==self.binding,'Data/config/code binding changed; explicit migration needed'
        assert s['plan_sha']==self.plan_sha
        self.model.load_state_dict(s['model']);self.opt.load_state_dict(s['optimizer'])
        for key in ['step','actions','history','best','training_seconds','eval_seconds']:setattr(self,key,s[key])
        self.exposures=collections.Counter(s['exposures']);self.unique=collections.defaultdict(set,{k:set(v) for k,v in s['unique'].items()})
        self.torch.set_rng_state(s['torch_rng']);random.setstate(s['python_rng']);self.resumed_from=dict(step=self.step,sha256=sha(path),at=now())
        log=self.out/'training.jsonl'
        if log.exists():
            lines=log.read_text().splitlines();keep=[line for line in lines if json.loads(line)['step']<=self.step]
            tail=[line for line in lines if json.loads(line)['step']>self.step]
            if tail:
                orphan=self.out/f'interrupted_uncheckpointed_{time.time_ns()}.jsonl';orphan.write_text('\n'.join(tail)+'\n')
                temp=log.with_suffix('.tmp');temp.write_text('\n'.join(keep)+'\n');temp.replace(log)
        with (self.out/'resume_history.jsonl').open('a') as f:f.write(json.dumps(self.resumed_from)+'\n')

    def publish(self,status,**extra):
        write_json(self.out/'progress.json',dict(status=status,updated=now(),step=self.step,target_steps=self.target,
            graph_presentations=sum(self.exposures.values()),actions=self.actions,source_exposures=dict(self.exposures),
            unique_configs_seen={s:len(v) for s,v in self.unique.items()},best=self.best,
            training_seconds=self.training_seconds,evaluation_seconds=self.eval_seconds,**extra))

    def gate(self):
        """Independent CE, gradient, simulator likelihood and exact teacher checks."""
        t=self.torch;checks=[]
        for source in self.cfg['sources']:
            for k in [3,6]:
                rec=self.con.execute('SELECT key FROM records WHERE role="train" AND source=? AND k=? ORDER BY n,key LIMIT 1',(source,k)).fetchone()
                if rec is None:continue
                row=self.row(rec[0]);x=self.forward(row)
                assert bool((x==0).all()),'Fresh model must start from zero coefficients'
                loss=self.bc_loss(x,row['decisions']);manual=[]
                for d in row['decisions']:
                    z=x[d['available']];scores=z[:,0]+z[:,1]*x.new_tensor(d['budget_local'])+x.new_tensor(d['offsets'])+z[:,2]*x.new_tensor(d['gap'])+z[:,3]*x.new_tensor(d['wait'])
                    manual.append(t.logsumexp(scores,0)-scores[d['selected']])
                independent=t.stack(manual).mean();assert t.allclose(loss,independent,atol=2e-6)
                ga=t.autograd.grad(loss,x,retain_graph=True)[0];gb=t.autograd.grad(independent,x)[0];assert t.allclose(ga,gb,atol=2e-7)
                zero=self.worker(cmd='roll',items=[dict(key=row['id'],coefficients=x.detach().tolist())])[0]
                assert zero['C']==row['reference']
                forced=self.worker(cmd='roll',items=[dict(key=row['id'],coefficients=x.detach().tolist(),forced_mapping=row['supervision']['task_to_pool'])],keep_decisions=True)[0]
                assert forced['C']==row['teacher'] and forced['decisions']==row['decisions']
                sample=self.worker(cmd='roll',items=[dict(key=row['id'],coefficients=x.detach().tolist(),sample=True,seed=28)],keep_decisions=True)[0]
                logs,_=self.distributions(x,sample['decisions']);labels=t.tensor([d['selected'] for d in sample['decisions']])
                err=float((logs.gather(1,labels[:,None]).squeeze(1)-t.tensor([d['selected_log_probability'] for d in sample['decisions']])).abs().max().detach())
                assert err<5e-5
                checks.append(dict(key=row['id'],source=source,k=k,probability_error=err,independent_gradient=True,zero_HEFT=True,teacher_replay=True))
        self.model.zero_grad(set_to_none=True)
        write_json(self.out/'preflight.json',dict(status='PASS',checks=checks,quality_conclusion=False,one_forward=True))

    def update(self):
        begin=time.perf_counter();batch=self.cfg['batch_size'];keys=self.plan[self.step*batch:(self.step+1)*batch]
        assert len(keys)==batch
        rows=[self.row(k) for k in keys];losses=collections.defaultdict(list)
        self.opt.zero_grad(set_to_none=True)
        for key,row in zip(keys,rows):
            assert row['meta']['role']=='train'
            loss=self.bc_loss(self.forward(row),row['decisions']);assert bool(self.torch.isfinite(loss))
            (loss/batch).backward();s=self.source_for_key[key];losses[s].append(float(loss.detach()))
            self.exposures[s]+=1;self.unique[s].add(key);self.actions+=len(row['decisions'])
        norm=float(self.torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.));assert math.isfinite(norm)
        assert self.opt.param_groups[0]['lr']==self.cfg['lr']
        self.opt.step();self.step+=1;self.training_seconds+=time.perf_counter()-begin
        record=dict(step=self.step,presentations=self.step*batch,actions=self.actions,
                    NLL=sum(sum(v) for v in losses.values())/batch,source_NLL={s:sum(v)/len(v) for s,v in losses.items()},
                    gradient_norm=norm,lr=self.cfg['lr'],training_seconds=self.training_seconds,
                    source_counts={s:len(v) for s,v in losses.items()},keys_sha256=digest(keys))
        with (self.out/'training.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        if self.step==1:
            self.checkpoint('initial_update');write_json(self.out/'startup_verified.json',dict(status='PASS',record=record,finite_gradients=True))
        return record

    def evaluate(self):
        begin=time.perf_counter();self.model.eval();rows=[]
        panels={}
        with self.torch.no_grad():
            for source in self.observation['eval_sources']:
                keys=[r[0] for r in self.con.execute('SELECT key FROM records WHERE role="eval" AND source=? ORDER BY key',(source,))]
                panels[source]=[]
                self.publish('evaluating',evaluation_source=source,panel_cases=0,panel_total=len(keys),completed_sources=list(panels)[:-1])
                for start in range(0,len(keys),16):
                    batch=[self.row(k) for k in keys[start:start+16]];items=[];forward_times=[]
                    for r in batch:
                        tick=time.perf_counter();x=self.forward(r);forward_times.append(time.perf_counter()-tick)
                        items.append(dict(key=r['id'],coefficients=x.tolist()))
                    results=self.worker(cmd='roll',items=items)
                    for r,res,seconds in zip(batch,results,forward_times):
                        assert res['key']==r['id'] and res['feasible']
                        panels[source].append(dict(**r['meta'],C=res['C'],cost=res['cost'],feasible=True,references=r['references'],
                                         forward_seconds=seconds,decoder_seconds=res['decoder_seconds']))
                    if start%256==0:self.publish('evaluating',evaluation_source=source,panel_cases=len(panels[source]),panel_total=len(keys))
                rows.extend(panels[source])
        metrics=grouped_metrics(rows)
        objective=sum(metrics[s]['C_over']['HEFT'] for s in ['synthetic','wfinstances','tpch_composite'])/3
        point=dict(step=self.step,presentations=sum(self.exposures.values()),objective=objective,metrics=metrics)
        self.history=[p for p in self.history if p['step']!=self.step]+[point]
        self.history.sort(key=lambda p:p['step'])
        improved=self.best is None or objective<self.best['objective']
        if improved:self.best=point
        path=self.out/f'eval_{self.step:05d}.json.gz';tmp=path.with_name(path.name+'.tmp')
        with gzip.open(tmp,'wt') as f:json.dump(dict(**point,rows=rows),f,allow_nan=False)
        tmp.replace(path)
        for source,panel in panels.items():
            path=self.out/f'test_{source}_{self.step:05d}.json.gz';tmp=path.with_name(path.name+'.tmp')
            with gzip.open(tmp,'wt') as f:json.dump(dict(step=self.step,source=source,metrics=metrics[source],rows=panel),f,allow_nan=False)
            tmp.replace(path)
            write_json(self.out/f'test_{source}_curve.json',[dict(step=p['step'],presentations=p['presentations'],**p['metrics'][source]) for p in self.history])
        self.eval_seconds+=time.perf_counter()-begin
        if improved:self.checkpoint('best')
        self.checkpoint('last');self.checkpoint(f'step{self.step:05d}');write_json(self.out/'curve.json',self.history)
        self.model.train();self.publish('training')
        print('EVAL',self.arm['id'],self.step,{s:round(v['C_over']['HEFT'],6) for s,v in metrics.items()},flush=True)

    def diagnose(self):
        begin=time.perf_counter();self.model.eval()
        probe=json.loads((self.data/'TRAIN_PROBE.json').read_text());keys=[k for s in self.cfg['sources'] for k in probe[s]]
        records=[];t=self.torch
        self.publish('diagnosing',diagnostic_cases=0,diagnostic_total=len(keys))
        with t.no_grad():
            for start in range(0,len(keys),16):
                batch=[self.row(k) for k in keys[start:start+16]];items=[];local=[]
                for r in batch:
                    x=self.forward(r);ds=r['decisions'];logs,mask=self.distributions(x,ds);labels=t.tensor([d['selected'] for d in ds]);active=mask.sum(1)>1
                    ce=-logs.gather(1,labels[:,None]).squeeze(1);correct=(logs.argmax(1)==labels)&active
                    spans=[];flips=[]
                    for d in ds:
                        if len(d['available'])<2:continue
                        z=x[d['available']];terms=t.stack([z[:,0],z[:,1]*x.new_tensor(d['budget_local']),x.new_tensor(d['offsets']),z[:,2]*x.new_tensor(d['gap']),z[:,3]*x.new_tensor(d['wait'])],dim=1)
                        scores=terms.sum(1);spans.append((terms.amax(0)-terms.amin(0)).tolist())
                        flips.append([int((scores-terms[:,j]).argmax())!=int(scores.argmax()) for j in range(5)])
                    local.append(dict(**r['meta'],teacher=r['teacher'],reference=r['reference'],NLL=float(ce.mean()),active_NLL=float(ce[active].mean()) if bool(active.any()) else None,
                                      active_actions=int(active.sum()),correct_active=int(correct.sum()),actions=len(ds),
                                      coefficient_saturation=((x.abs()>=.95*x.new_tensor([8,8,2,2])).float().mean(0)).tolist(),
                                      teacher_term_span=t.tensor(spans).mean(0).tolist() if spans else [0.]*5,
                                      local_term_removal_flip=t.tensor(flips,dtype=t.float32).mean(0).tolist() if flips else [0.]*5))
                    items.append(dict(key=r['id'],coefficients=x.tolist()))
                results=self.worker(cmd='roll',items=items)
                for r,res in zip(local,results):
                    assert r['key']==res['key'] and res['feasible']
                    r.update(C=res['C'],C_teacher=res['C']/r['teacher'],C_HEFT=res['C']/r['reference'],feasible=True);records.append(r)
                if start%256==0:self.publish('diagnosing',diagnostic_cases=len(records),diagnostic_total=len(keys))
        summary={}
        for source in self.cfg['sources']:
            rr=[r for r in records if r['source']==source]
            by_base=collections.defaultdict(list)
            for r in rr:by_base[r['base_id']].append(r)
            summary[source]=dict(configs=len(rr),bases=len(by_base),NLL=sum(r['NLL'] for r in rr)/len(rr),
                 active_accuracy=sum(r['correct_active'] for r in rr)/max(1,sum(r['active_actions'] for r in rr)),
                 C_teacher=sum(sum(r['C_teacher'] for r in bs)/len(bs) for bs in by_base.values())/len(by_base),
                 C_HEFT=sum(sum(r['C_HEFT'] for r in bs)/len(bs) for bs in by_base.values())/len(by_base),
                 Teacher64_HEFT=sum(sum(r['teacher']/r['reference'] for r in bs)/len(bs) for bs in by_base.values())/len(by_base),
                 coefficient_saturation=t.tensor([r['coefficient_saturation'] for r in rr]).mean(0).tolist(),
                 teacher_term_span=t.tensor([r['teacher_term_span'] for r in rr]).mean(0).tolist())
        path=self.out/f'diagnostic_{self.step:05d}.json.gz';tmp=path.with_name(path.name+'.tmp')
        with gzip.open(tmp,'wt') as f:json.dump(dict(step=self.step,summary=summary,rows=records,
            interpretation='teacher-prefix classification vs autonomous rollout; local term-removal flips are not causal full-rollout ablations'),f,allow_nan=False)
        tmp.replace(path);write_json(self.out/'diagnostic_latest.json',dict(step=self.step,summary=summary))
        curve_path=self.out/'diagnostic_curve.json'
        curve=json.loads(curve_path.read_text()) if curve_path.exists() else []
        curve=[p for p in curve if p['step']<self.step]+[dict(step=self.step,summary=summary)]
        write_json(curve_path,curve)
        self.eval_seconds+=time.perf_counter()-begin;self.model.train();self.checkpoint('last');self.publish('training')
        print('DIAGNOSTIC',self.arm['id'],self.step,summary,flush=True)

    def plot(self):
        subprocess.run(['/opt/miniconda3/envs/dag-research/bin/python',str(ROOT/'figures/gen_fig_synthetic_transfer_20261031.py')],cwd=ROOT,check=True)

    def run(self):
        if self.step==0:
            self.gate();self.evaluate();self.diagnose();self.plot()
        elif self.step%self.cfg['eval_every']==0 or self.step==self.target:
            # Recover an interrupted observation after its model checkpoint was saved.
            if not any(p['step']==self.step for p in self.history):self.evaluate()
            curve=json.loads((self.out/'diagnostic_curve.json').read_text()) if (self.out/'diagnostic_curve.json').exists() else []
            if not any(p['step']==self.step for p in curve):self.diagnose()
            self.plot()
        while self.step<self.target:
            self.update()
            if self.step%50==0:self.publish('training')
            if self.step%250==0:self.checkpoint('last')
            if self.step%self.cfg['eval_every']==0 or self.step==self.target:self.evaluate()
            if self.step%self.cfg['diagnostic_every']==0 or self.step==self.target:
                self.diagnose();self.plot()
        self.checkpoint('last')
        summary=dict(status='complete',updated=now(),arm=self.arm,steps=self.step,
            presentations=self.step*self.cfg['batch_size'],actions=self.actions,source_exposures=dict(self.exposures),
            unique_configs_seen={s:len(v) for s,v in self.unique.items()},best=self.best,history=self.history,
            training_seconds=self.training_seconds,evaluation_seconds=self.eval_seconds,data_binding=self.binding,
            evaluation_used_for_tuning=True,single_seed_exploratory=True)
        write_json(self.out/f'summary_{self.step:05d}.json',summary);write_json(self.out/'summary.json',summary)
        for tag in ['best','last']:
            dest=self.out/f'budget{self.step:05d}_{tag}.pt'
            if not dest.exists():shutil.copy2(self.out/(tag+'.pt'),dest)
        self.publish('complete')

    def close(self):
        self.worker.close();self.con.close();self.row.cache_clear()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--steps',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args()
    cfg=config();arm=next(x for x in cfg['arms'] if x['id']==a.arm)
    review=ROOT/cfg['run']/'TRAINING_PLAN_REVIEW.json'
    if review.exists():
        # User is setting actual epoch counts and evaluation cadence. This entry
        # point must not fall through into the superseded 16320-update plan.
        out=ROOT/cfg['run']/arm['id'];out.mkdir(parents=True,exist_ok=True)
        print('WAITING: epoch-based training plan and observation cadence under discussion; no model initialized',flush=True)
        while review.exists():
            write_json(out/'progress.json',dict(status='awaiting_epoch_and_cadence_plan',updated=now(),step=0,
                target_epochs=7,reason=json.loads(review.read_text())['reason']))
            time.sleep(5)
        raise SystemExit('Training plan reviewed; restart through the epoch-based controller, not the old update-budget command')
    assert (a.steps or cfg['steps']) in [cfg['steps'],cfg['extension_steps']],'Production arms use the predeclared full budgets'
    trial=Trial(arm,a.steps or cfg['steps'],a.resume)
    try:trial.run()
    finally:trial.close()
