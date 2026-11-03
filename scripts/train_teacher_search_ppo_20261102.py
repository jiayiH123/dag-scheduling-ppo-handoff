#!/usr/bin/env python3
"""Matched repeat512 classical PPO from either teacher's SFT checkpoint."""
import argparse
import collections
import copy
import gzip
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from daglab.teacher_search_ablation import config,Store,train_keys,macro,code_binding,serve
from daglab.sft_three_panel import write_json
from daglab.structured_dataset import sha256
from daglab.offline_data import digest
from scripts.check_direct_score import Bridge


class PPO:
    def __init__(self,arm,seed,out_override=None,resume=False):
        import torch
        from daglab.state_score_torch import StateActor,distributions
        from daglab.module_lab.models import tensors
        from daglab.classic_ppo_torch import Critic,gae,clipped_loss
        self.torch,self.distributions,self.tensors,self.gae,self.clipped_loss=torch,distributions,tensors,gae,clipped_loss
        self.cfg=config()
        self.hp=self.cfg['PPO']
        self.arm,self.seed=arm,seed
        self.root=ROOT/self.cfg['run']
        self.out=Path(out_override) if out_override else self.root/f'ppo_{arm}_seed{seed}'
        self.out.mkdir(parents=True,exist_ok=True)
        self.source=ROOT/self.cfg['strong_source'] if arm=='strong' else ROOT/self.cfg['no_search_SFT_run']/'best.pt'
        if self.cfg.get('portable_source_override'):self.source=ROOT/self.cfg['portable_source_override']
        self.source_state=torch.load(self.source,map_location='cpu',weights_only=False)
        self.binding=dict(runtime_config_sha256=digest(self.cfg),portable_runtime_sha256=sha256(ROOT/'daglab/portable_runtime.py'),source=sha256(self.source),config=sha256(ROOT/'configs/teacher_search_ablation_ppo_20261102.json'),
            arm=arm,seed=seed,code=code_binding(['scripts/train_teacher_search_ppo_20261102.py',
                'daglab/teacher_search_ablation.py','daglab/state_score.py','daglab/state_score_torch.py',
                'daglab/classic_ppo.py','daglab/classic_ppo_torch.py','daglab/critic_transfer_torch.py',
                'figures/gen_fig_teacher_search_ablation_20261102.py']),
            validation_manifest=sha256(ROOT/'runs/sft_three_panel_20261101/data/MANIFEST.json'),
            training_manifest=sha256(ROOT/self.cfg['original_cache']/'MANIFEST.json'))
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        unused=StateActor('both')
        self.actor=StateActor('both')
        self.actor.load_state_dict(self.source_state['model'],strict=True)
        self.critic=Critic()
        assert sum(p.numel() for p in self.actor.parameters())==32612
        assert sum(p.numel() for p in self.critic.parameters())==9153
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=self.hp['actor_lr'])
        self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=self.hp['critic_lr'])
        self.rng=random.Random(seed)
        self.keys=train_keys()
        random.Random(seed).shuffle(self.keys)
        self.order_sha=digest(self.keys)
        self.collection=self.actor_updates=self.critic_updates=self.slots=self.prewarm_updates=0
        self.history=[]
        self.best=self.best_config_mean=None
        self.training_seconds=self.evaluation_seconds=0.
        self.worker=Bridge('scripts/train_teacher_search_ppo_20261102.py','--worker')
        self.store=Store()
        self.actor.train()
        self.critic.train()
        if (self.out/'metadata.json').exists():
            assert json.loads((self.out/'metadata.json').read_text())['binding']==self.binding
        else:
            write_json(self.out/'metadata.json',dict(binding=self.binding,arm=arm,RL_seed=seed,SFT_seed=28,
                SFT_source=str(self.source.relative_to(ROOT)),SFT_step=self.source_state['step'],
                parameters=32612,critic_parameters=9153,PPO=self.hp,order_sha256=self.order_sha,
                main_keys=self.keys[:512],warm_keys=self.keys[2048:2304],
                historical_order_sha256=__import__('hashlib').sha256(('\n'.join(self.keys)+'\n').encode()).hexdigest(),
                evaluation_used_for_tuning=True,primary_metric='base_macro_C_HEFT',
                supplementary_metric='configuration_mean_C_HEFT',VAL_configs=862,TEST_evaluated=False,
                one_actor_forward_per_deployment=True,teacher_loss_in_PPO=False,
                CPU_threads=1,deterministic_algorithms=True,
                initial_critic_sha256=digest({k:v.tolist() for k,v in self.critic.state_dict().items()}),
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        if resume:
            self.restore()

    def forward(self,row):
        before=(self.actor.forward_calls,self.actor.base.forward_calls)
        x=self.actor(self.tensors(row['graph'],'cpu'))
        assert (self.actor.forward_calls-before[0],self.actor.base.forward_calls-before[1])==(1,1)
        assert x.shape==(row['n']*row['graph']['k'],4) and bool(self.torch.isfinite(x).all())
        return x

    def rows(self,keys):
        return self.worker(cmd='rows',keys=keys)

    def trajectory(self,row,result):
        t=self.torch
        ds=result['decisions']
        states=t.tensor([d['critic_state'] for d in ds])
        rewards=t.tensor([-(d['partial_after']-d['partial_before'])/row['HEFT'] for d in ds])
        assert states.shape==(row['n'],76) and bool(t.isfinite(states).all())
        assert abs(float(rewards.sum())+result['C']/row['HEFT'])<2e-5
        return dict(row=row,ds=ds,states=states,rewards=rewards,mc=rewards.flip(0).cumsum(0).flip(0),
            old=t.tensor([d['selected_log_probability'] for d in ds]),
            labels=t.tensor([d['selected'] for d in ds]),active=t.tensor([len(d['available'])>1 for d in ds]),C=result['C'])

    def value_train(self,batch,target):
        t=self.torch
        self.critic_opt.zero_grad(set_to_none=True)
        pred=self.critic(t.cat([r['states'] for r in batch]))
        y=t.cat([r[target] for r in batch])
        loss=.5*(pred-y).square().mean()
        assert bool(t.isfinite(loss))
        loss.backward()
        norm=float(t.nn.utils.clip_grad_norm_(self.critic.parameters(),1.))
        assert math.isfinite(norm)
        self.critic_opt.step()
        return dict(value_loss=float(loss.detach()),critic_gradient_norm=norm)

    def publish(self,status,**extra):
        write_json(self.out/'progress.json',dict(status=status,arm=self.arm,seed=self.seed,collection=self.collection,
            actor_updates=self.actor_updates,actor_update_slots=self.slots,critic_updates=self.critic_updates,
            prewarm_updates=self.prewarm_updates,target_collections=self.hp['collections'],max_actor_updates=self.hp['collections']*8,
            best=self.best,training_seconds=self.training_seconds,evaluation_seconds=self.evaluation_seconds,**extra))

    def state(self):
        return dict(model=self.actor.state_dict(),critic=self.critic.state_dict(),optimizer=self.actor_opt.state_dict(),
            critic_optimizer=self.critic_opt.state_dict(),binding=self.binding,collection=self.collection,
            actor_updates=self.actor_updates,critic_updates=self.critic_updates,slots=self.slots,prewarm_updates=self.prewarm_updates,
            history=self.history,best=self.best,best_config_mean=self.best_config_mean,python_rng=self.rng.getstate(),
            torch_rng=self.torch.get_rng_state(),training_seconds=self.training_seconds,evaluation_seconds=self.evaluation_seconds)

    def checkpoint(self,name):
        tmp=self.out/f'{name}.tmp.pt'
        self.torch.save(self.state(),tmp)
        tmp.replace(self.out/f'{name}.pt')

    def restore(self):
        s=self.torch.load(self.out/'resume.pt',map_location='cpu',weights_only=False)
        assert s['binding']==self.binding
        self.actor.load_state_dict(s['model'])
        self.critic.load_state_dict(s['critic'])
        self.actor_opt.load_state_dict(s['optimizer'])
        self.critic_opt.load_state_dict(s['critic_optimizer'])
        for name in ['collection','actor_updates','critic_updates','slots','prewarm_updates','history','best','best_config_mean',
                     'training_seconds','evaluation_seconds']:
            setattr(self,name,s[name])
        self.rng.setstate(s['python_rng'])
        self.torch.set_rng_state(s['torch_rng'])
        for name,field in [('training.jsonl','collection'),('updates.jsonl','collection')]:
            path=self.out/name
            if path.exists():
                records=[r for r in path.read_text().splitlines() if json.loads(r)[field]<=self.collection]
                path.write_text('\n'.join(records)+ ('\n' if records else ''))
        write_json(self.out/'curve.json',self.history)

    def warmup(self):
        begin=time.perf_counter()
        self.publish('critic_warmup')
        before={k:v.detach().clone() for k,v in self.actor.state_dict().items()}
        trajectories=[]
        for offset in range(2048,2304,16):
            rows=self.rows(self.keys[offset:offset+16])
            with self.torch.no_grad():
                items=[dict(id=r['id'],x=self.forward(r).tolist(),sample=True,temperature=self.hp['temperature'],
                    seed=731000+self.seed*100+offset+i) for i,r in enumerate(rows)]
            results=self.worker(cmd='rlroll',items=items)
            trajectories.extend(self.trajectory(r,v) for r,v in zip(rows,results))
        wrng=random.Random(7290+self.seed)
        losses=[]
        for step in range(200):
            result=self.value_train([trajectories[i] for i in wrng.sample(range(224),8)],'mc')
            self.prewarm_updates+=1
            losses.append(dict(step=step+1,**result))
        with self.torch.no_grad():
            hold=trajectories[224:]
            pred=self.critic(self.torch.cat([r['states'] for r in hold]))
            y=self.torch.cat([r['mc'] for r in hold])
            mse=float((pred-y).square().mean())
        assert all(self.torch.equal(v,before[k]) for k,v in self.actor.state_dict().items())
        self.training_seconds+=time.perf_counter()-begin
        write_json(self.out/'warmup.json',dict(configs=256,fit=224,holdout=32,updates=200,
            actor_unchanged=True,holdout_MSE=mse,curve=losses,seconds=time.perf_counter()-begin))

    def evaluate(self,reason):
        if any(p['actor_updates']==self.actor_updates and p['collection']==self.collection for p in self.history):
            return
        begin=time.perf_counter()
        self.actor.eval()
        rows=[]
        self.publish('evaluating',reason=reason)
        expected=None
        if self.actor_updates==0 and (self.source.parent/f'val_synthetic_val862_{self.source_state["step"]:05d}.json.gz').exists():
            with gzip.open(self.source.parent/f'val_synthetic_val862_{self.source_state["step"]:05d}.json.gz','rt') as f:
                expected={r['key']:(r['C'],r['cost']) for r in json.load(f)['rows']}
        with self.torch.no_grad():
            for start in range(0,862,16):
                batch=self.rows(self.store.val_keys[start:start+16])
                items=[dict(id=r['id'],x=self.forward(r).tolist()) for r in batch]
                for row,result in zip(batch,self.worker(cmd='roll',items=items)):
                    assert row['id']==result['id'] and result['feasible']
                    if expected is not None:
                        assert (result['C'],result['cost'])==expected[row['id']]
                    rows.append(dict(key=row['id'],base=row['base'],n=row['n'],C=result['C'],cost=result['cost'],
                        HEFT=row['HEFT'],feasible=True,references=row['references']))
        assert len(rows)==862
        primary=macro(rows)
        point=dict(actor_updates=self.actor_updates,actor_update_slots=self.slots,collection=self.collection,reason=reason,
            C_HEFT_base_macro=primary,C_HEFT_config_mean=sum(r['C']/r['HEFT'] for r in rows)/862,
            configs=862,bases=500,feasible=True)
        self.history.append(point)
        self.evaluation_seconds+=time.perf_counter()-begin
        if self.best is None or primary<self.best['C_HEFT_base_macro']:
            self.best=point
            self.checkpoint('best')
        if self.best_config_mean is None or point['C_HEFT_config_mean']<self.best_config_mean['C_HEFT_config_mean']:
            self.best_config_mean=point
            self.checkpoint('best_config_mean')
        name=f'eval_u{self.actor_updates:04d}_c{self.collection:03d}'
        tmp=self.out/(name+'.tmp.json.gz')
        with gzip.open(tmp,'wt') as f:
            json.dump(dict(**point,rows=rows),f,allow_nan=False)
        tmp.replace(self.out/(name+'.json.gz'))
        self.checkpoint(name)
        write_json(self.out/'curve.json',self.history)
        self.actor.train()
        self.publish('training')
        self.plot()
        print('VAL',self.arm,self.seed,self.actor_updates,self.collection,primary,point['C_HEFT_config_mean'],flush=True)

    def update_collection(self,collection,observe=True):
        begin=time.perf_counter()
        self.collection=collection
        start=((collection-1)*16)%512
        rows=self.rows(self.keys[start:start+16])
        with self.torch.no_grad():
            items=[dict(id=r['id'],x=self.forward(r).tolist(),sample=True,temperature=self.hp['temperature'],
                seed=self.seed*10000+collection*16+i) for i,r in enumerate(rows)]
        trajectories=[self.trajectory(r,v) for r,v in zip(rows,self.worker(cmd='rlroll',items=items))]
        with self.torch.no_grad():
            for tr,it in zip(trajectories,items):
                v=self.critic(tr['states'])
                tr['adv'],tr['returns']=self.gae(tr['rewards'],v,lam=1.)
                assert self.torch.allclose(tr['returns'],tr['mc'],atol=1e-5,rtol=0)
                logs,_=self.distributions(self.torch.tensor(it['x']),tr['ds'],self.hp['temperature'])
                error=float((logs.gather(1,tr['labels'][:,None]).squeeze(1)-tr['old']).abs().max())
                assert error<5e-5
        active=self.torch.cat([t['adv'][t['active']] for t in trajectories])
        am=active.mean() if active.numel() else active.new_zeros(())
        sd=active.std(unbiased=False).clamp_min(1e-8) if active.numel() else active.new_ones(())
        for tr in trajectories:
            tr['normalized_adv']=(tr['adv']-am)/sd
        stopped=False
        for epoch in range(4):
            perm=list(range(16))
            self.rng.shuffle(perm)
            for offset in [0,8]:
                batch=[trajectories[i] for i in perm[offset:offset+8]]
                record=self.value_train(batch,'returns')
                self.critic_updates+=1
                self.slots+=1
                record.update(collection=collection,epoch=epoch,offset=offset,slot=self.slots)
                took_update=False
                if not stopped and any(bool(tr['active'].any()) for tr in batch):
                    self.actor_opt.zero_grad(set_to_none=True)
                    new,old,adv,entropy=[],[],[],[]
                    for tr in batch:
                        lp,mask=self.distributions(self.forward(tr['row']),tr['ds'],self.hp['temperature'])
                        new.append(lp.gather(1,tr['labels'][:,None]).squeeze(1)[tr['active']])
                        old.append(tr['old'][tr['active']])
                        adv.append(tr['normalized_adv'][tr['active']])
                        entropy.append(-(lp.exp()*lp.masked_fill(~mask,0)).sum(-1)[tr['active']])
                    policy,kl,frac=self.clipped_loss(self.torch.cat(new),self.torch.cat(old),self.torch.cat(adv),clip=self.hp['clip'])
                    ent=self.torch.cat(entropy).mean()
                    record.update(policy=float(policy.detach()),KL=float(kl.detach()),clip_fraction=float(frac),entropy=float(ent.detach()))
                    if float(kl.detach())>self.hp['KL_stop']:
                        stopped=True
                    else:
                        loss=policy-self.hp['entropy']*ent
                        assert bool(self.torch.isfinite(loss))
                        loss.backward()
                        norm=float(self.torch.nn.utils.clip_grad_norm_(self.actor.parameters(),1.))
                        assert math.isfinite(norm)
                        self.actor_opt.step()
                        self.actor_updates+=1
                        took_update=True
                        record.update(actor_gradient_norm=norm,actor_loss=float(loss.detach()))
                record.update(actor_updates=self.actor_updates,actor_stopped=stopped,actual_actor_update=took_update)
                with (self.out/'updates.jsonl').open('a') as f:
                    f.write(json.dumps(record,allow_nan=False)+'\n')
                if observe and took_update and self.actor_updates%self.cfg['eval_every_actor_updates']==0:
                    self.training_seconds+=time.perf_counter()-begin
                    self.evaluate('100_actual_actor_updates')
                    begin=time.perf_counter()
        self.training_seconds+=time.perf_counter()-begin
        entry=dict(collection=collection,actor_updates=self.actor_updates,critic_updates=self.critic_updates,
            slots=self.slots,sampled_C_HEFT=sum(t['C']/t['row']['HEFT'] for t in trajectories)/16,
            trajectories=16,unique_main_configs_seen=min(collection*16,512),max_logprob_error=error,
            sampled_keys=[r['id'] for r in rows],training_seconds=self.training_seconds)
        with (self.out/'training.jsonl').open('a') as f:
            f.write(json.dumps(entry,allow_nan=False)+'\n')
        if observe and collection in [32,64,96,128]:
            self.evaluate('historical_collection_grid')
        self.checkpoint('resume')
        self.publish('training')

    def plot(self):
        subprocess.run([self.cfg['sim_python'],str(ROOT/'figures/gen_fig_teacher_search_ablation_20261102.py')],cwd=ROOT,check=True)

    def run(self):
        if self.collection==0 and self.prewarm_updates==0:
            self.warmup()
            self.evaluate('SFT_start')
            self.checkpoint('resume')
        for collection in range(self.collection+1,129):
            self.update_collection(collection)
        if not any(p['actor_updates']==self.actor_updates for p in self.history):
            self.evaluate('final')
        self.checkpoint('last')
        assert self.critic_updates==1024 and self.prewarm_updates==200 and self.slots==1024
        write_json(self.out/'summary.json',dict(status='complete',arm=self.arm,seed=self.seed,best=self.best,
            best_config_mean=self.best_config_mean,history=self.history,actor_updates=self.actor_updates,
            critic_updates=1024,prewarm_updates=200,main_trajectories=2048,main_unique_configs=512,
            warmup_trajectories=256,training_seconds=self.training_seconds,evaluation_seconds=self.evaluation_seconds,
            binding=self.binding,VAL_used_for_tuning=True))
        self.publish('complete')
        self.plot()

    def close(self):
        self.worker.close()
        self.store.close()


def engineering():
    root=ROOT/config()['run']/'ppo_engineering'
    p=PPO('strong',41,out_override=root/'uninterrupted')
    try:
        checks=[]
        con=p.store.train
        keys=[con.execute('SELECT key FROM records WHERE source="synthetic" AND role="train" AND n=? AND k=? ORDER BY key LIMIT 1',(n,k)).fetchone()[0] for n in [20,40,80,120] for k in [3,4,6]]
        rows=p.rows(keys)
        for row in rows:
            with p.torch.no_grad():
                x=p.forward(row)
            for temperature in [1.,.5]:
                it=dict(id=row['id'],x=x.tolist(),sample=True,temperature=temperature,seed=4100)
                plain=p.worker(cmd='roll',items=[it],decisions=True,schedule=True)[0]
                res=p.worker(cmd='rlroll',items=[it],schedule=True)[0]
                assert plain['schedule']==res['schedule']
                tr=p.trajectory(row,res)
                logs,_=p.distributions(x,tr['ds'],temperature)
                error=float((logs.gather(1,tr['labels'][:,None]).squeeze(1)-tr['old']).abs().max())
                assert error<5e-5
                p.actor.zero_grad(set_to_none=True)
                p.critic.zero_grad(set_to_none=True)
                loss=(p.critic(tr['states'])-tr['mc']).square().mean()
                loss.backward()
                assert all(v.grad is None for v in p.actor.parameters())
                assert sum(float(v.grad.abs().sum()) for v in p.critic.parameters() if v.grad is not None)>0
                checks.append(dict(key=row['id'],n=row['n'],k=row['graph']['k'],T=temperature,logprob_error=error))
        p.actor.zero_grad(set_to_none=True)
        p.critic.zero_grad(set_to_none=True)
        p.update_collection(1,observe=False)
        saved=p.torch.load(p.out/'resume.pt',map_location='cpu',weights_only=False)
        p.update_collection(2,observe=False)
        expected=copy.deepcopy(p.state())
        q=PPO('strong',41,out_override=root/'restored')
        try:
            q.torch.save(saved,q.out/'resume.pt')
            q.restore()
            q.update_collection(2,observe=False)
            actual=q.state()
            def equal(a,b):
                if isinstance(a,p.torch.Tensor):
                    assert p.torch.equal(a,b)
                elif isinstance(a,dict):
                    assert a.keys()==b.keys()
                    for k in a:equal(a[k],b[k])
                elif isinstance(a,(list,tuple)):
                    assert len(a)==len(b)
                    for x,y in zip(a,b):equal(x,y)
                else:
                    assert a==b
            for k in ['model','critic','optimizer','critic_optimizer','python_rng','torch_rng','actor_updates','critic_updates']:
                equal(expected[k],actual[k])
            write_json(ROOT/config()['run']/'PPO_ENGINEERING_CHECKS.json',dict(status='PASS',checks=checks,
                shapes=12,temperature_shapes=24,instrumentation_same_schedule=True,critic_actor_gradient_isolation=True,
                exact_collection_boundary_resume=True,engineering_updates_not_formal=True,binding=p.binding))
        finally:
            q.close()
    finally:
        p.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--worker',action='store_true')
    ap.add_argument('--check',action='store_true')
    ap.add_argument('--arm',choices=['strong','no_search'])
    ap.add_argument('--seed',type=int,default=41)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if args.worker:
        serve()
    elif args.check:
        engineering()
    else:
        assert args.arm
        trial=PPO(args.arm,args.seed,resume=args.resume)
        try:
            trial.run()
        finally:
            trial.close()
