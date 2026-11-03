#!/usr/bin/env python3
"""Corrected primary external policies: visible budget, full data, explicit search."""
import argparse,copy,hashlib,itertools,json,math,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,'/private/tmp/dag-baseline-deps-20261028')
import torch
from daglab.external_budget_visible import BudgetProblem,BudgetCAIActor,BudgetGraphormer,BudgetGiPHActor
from daglab.external_baseline_models import CAIActor,GiPHActor
from daglab.external_graphormer import CAIGraphormer
from daglab.external_baseline_sim import Problem,stable_seed
from daglab.structured_unique import iter_unique_rows
from daglab.structured_gnn_simulator import shuffled
from daglab.structured_dataset import to_instance
from scripts.run_external_baselines import save,cases,cai_roll,giph_paths,summarize,DATA
from scripts.run_scratch_lr_three_arm import same
BASE=ROOT/'runs/external_baselines_budget_visible_20261028'


def train_rows(epoch):return shuffled(iter_unique_rows(DATA,'train'),2026102803+epoch)


def check(out):
    seen=set();details=[];cai_gradient=False;giph_gradient=False
    for row in train_rows(1):
        inst=to_instance(row);shape=inst['n'],len(inst['capacities'])
        if shape in seen:continue
        seen.add(shape);p=BudgetProblem(inst);old=Problem(inst)
        lower=copy.deepcopy(inst);lower['budget']=sum(p.info['min_cost']);lo=BudgetProblem(lower)
        a,b=p.priority_features(),lo.priority_features()
        assert a['x'][0][9]==p.k/6 and b['x'][0][8]==0
        if p.budget>lower['budget']:assert a['x'][0][8]!=b['x'][0][8]
        stats={}
        for factory,previous in [(BudgetCAIActor,CAIActor),(BudgetGraphormer,CAIGraphormer)]:
            torch.manual_seed(11);m=factory().eval();torch.manual_seed(11);ref=previous().eval()
            z,_=m(a);rz,_=ref(old.priority_features());err=float((z-rz).detach().abs().max());assert err<5e-5
            order,ds=p.priority_order(z.detach().tolist());active=next((d for d in ds if len(d['available'])>1),None)
            if active:
                loss=-torch.log_softmax(z[active['available']],0)[active['available'].index(active['action'])];loss.backward()
                weights=[d[0].linear.weight for d in m.directions] if isinstance(m,BudgetCAIActor) else [m.input.weight]
                cai_gradient |= any(w.grad is not None and bool((w.grad[:,8:10]!=0).any()) for w in weights)
            stats[factory.__name__]=dict(initial_logit_max_error=err)
        mapping=p.random_mapping(131);s=p.state(mapping);g=p.relocation_graph(mapping,s);og=old.relocation_graph(mapping,s)
        assert len(g['x'][0])==10 and all(x[9]==(p.budget-s['cost'])/max(1,p.budget) for x in g['x'])
        torch.manual_seed(11);m=BudgetGiPHActor();torch.manual_seed(11);ref=GiPHActor()
        z=m(g);rz=ref(og);mask=torch.tensor(g['mask'])
        if mask.any():
            err=float((z[mask]-rz[mask]).detach().abs().max());assert err<2e-6
            (-torch.log_softmax(z,0)[z.argmax()]).backward()
            giph_gradient |= bool((m.node[0].weight.grad[:,9]!=0).any())
        else:err=0.
        stats['BudgetGiPHActor']=dict(initial_logit_max_error=err)
        details.append(dict(n=p.n,k=p.k,eta=a['x'][0][8],models=stats))
        if len(seen)==12:break
    assert len(seen)==12 and cai_gradient and giph_gradient
    save(out/'budget_visibility_checks.json',dict(status='PASS',details=details,CAI_budget_columns_nonzero_gradient=cai_gradient,
        GiPH_budget_column_nonzero_gradient=giph_gradient,features_differ_with_budget=True,initialization_tolerance=True))
    print('BUDGET VISIBILITY PASS',flush=True)


def cai_epoch_eval(out,model,name,epoch):
    records=[];model.eval()
    with (out/f'epoch{epoch}_cases.jsonl').open('w') as stream,torch.no_grad():
        for c in cases():
            p=BudgetProblem(c['inst']);base=p.schedule();assert base['C']==c['references']['HEFT']
            before=model.forward_calls;z,_=model(p.priority_features());r,_=cai_roll(p,z,False)
            assert model.forward_calls-before==1
            f=p.finish(r['mapping'],r['order']);assert (f['C'],f['cost'])==(r['C'],r['cost'])
            value=dict(**c['meta'],references=c['references'],methods={name:dict(C=f['C'],cost=f['cost'],feasible=True,actor_calls=1)})
            records.append(value);stream.write(json.dumps(value)+'\n')
    summarize(out,records,dict(epoch=epoch,intermediate=True,budget_visible=True));s=json.loads((out/'summary.json').read_text())
    save(out/f'epoch{epoch}_summary.json',s)
    return s


def fit_cai(method,out):
    isgraph=method=='graphormer';sparse=method=='cai_sparse';factory=BudgetGraphormer if isgraph else BudgetCAIActor
    name='CAI_DR_Graphormer_visible' if isgraph else 'CAI_noDR_BiGAT_visible' if sparse else 'CAI_DR_BiGAT_visible'
    batchsize=16 if isgraph else 8;perpass=1079 if isgraph else 2158;epochs=1 if sparse else 7
    torch.manual_seed(11);model=factory()
    opt=torch.optim.Adam(model.parameters(),lr=5e-4) if isgraph else torch.optim.RMSprop(model.parameters(),lr=7e-4,alpha=.99,eps=1e-5)
    torch.save(dict(model=model.state_dict(),torch_rng=torch.get_rng_state()),out/'initial.pt')
    history=[];best=None;graphs=0;updates=0;actions=0;exposures=[];started=time.perf_counter()
    zero=cai_epoch_eval(out,model,name,0);history.append(dict(epoch=0,panels={p:r['methods'][name] for p,r in zero['panels'].items()}))
    with (out/'training.jsonl').open('w') as log:
        for epoch in range(1,epochs+1):
            iterator=iter(train_rows(epoch));count=0;steps=0;digest=hashlib.sha256();model.train()
            while True:
                batch=list(itertools.islice(iterator,batchsize))
                if not batch:break
                opt.zero_grad(set_to_none=True);items=[];raws=[];ratios=[]
                for row in batch:
                    p=BudgetProblem(to_instance(row));base=p.schedule();z,raw=model(p.priority_features());raws.append(raw.square().mean())
                    for _ in range(4):
                        r,lp=cai_roll(p,z,True)
                        reward=torch.full((p.n,),(base['C']-r['C'])/base['C'],dtype=torch.float32) if sparse else torch.tensor([(base['ends'][v]-r['ends'][v])/base['C'] for v in r['order']],dtype=torch.float32)
                        items.append((lp,reward));ratios.append(r['C']/base['C']);actions+=p.n
                    graphs+=1;count+=1;digest.update((row['key']+'\n').encode())
                active=torch.cat([r[r!=0] for _,r in items]);mean=active.mean() if len(active) else 0.;std=active.std(unbiased=False) if len(active) else 1.
                loss=sum((-(r-mean).div(std+1e-10).clamp(-2,2)*lp).mean()+.01*lp.mean() for lp,r in items)/len(items)
                loss=loss+.001*torch.stack(raws).mean();assert bool(torch.isfinite(loss));loss.backward()
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),.5));assert math.isfinite(norm);opt.step();updates+=1;steps+=1
                rec=dict(epoch=epoch,epoch_step=steps,update=updates,graphs=graphs,actions=actions,
                    loss=float(loss.detach()),sampled_C_HEFT=sum(ratios)/len(ratios),gradient_norm=norm,lr=opt.param_groups[0]['lr'])
                log.write(json.dumps(rec)+'\n');log.flush()
                if steps%100==0:save(out/'progress.json',dict(stage='training',**rec));print(method,'TRAIN',epoch,steps,flush=True)
            assert (steps,count)==(perpass,17261)
            exposures.append(dict(epoch=epoch,configs=count,actions=1122860*4,ordered_keys_sha256=digest.hexdigest()))
            ck=dict(model=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),epoch=epoch,graphs=graphs,updates=updates,actions=actions)
            torch.save(ck,out/f'epoch{epoch}.pt');torch.save(ck,out/'final.pt')
            s=cai_epoch_eval(out,model,name,epoch);history.append(dict(epoch=epoch,panels={p:r['methods'][name] for p,r in s['panels'].items()}))
            metric=history[-1]['panels']['synthetic']['C_HEFT']
            if best is None or metric<best['C_HEFT']:best=dict(epoch=epoch,C_HEFT=metric);torch.save(ck,out/'best.pt')
            save(out/'history.json',dict(history=history,best_by_synthetic=best,exposures=exposures));save(out/'progress.json',dict(stage='training',epoch=epoch,graphs=graphs,updates=updates))
            print(method,'EPOCH',epoch,{p:r['C_HEFT'] for p,r in history[-1]['panels'].items()},flush=True)
    assert (graphs,actions,updates)==(17261*epochs,4491440*epochs,perpass*epochs)
    s.update(status='complete',intermediate=False,training=dict(epochs=epochs,graphs=graphs,samples=graphs*4,actions=actions,updates=updates,
        parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-started),history=history,best_by_synthetic=best,exposures=exposures)
    save(out/'summary.json',s);save(out/'progress.json',dict(stage='complete',epoch=epochs,graphs=graphs,updates=updates));print('COMPLETE VISIBLE',method,best,flush=True)


def giph_optimize(model,opt,ps,maps):
    opt.zero_grad(set_to_none=True);lps,rs,_,counts,ncalls=giph_paths(model,ps,maps,[16]*len(ps),train=True)
    losses=[]
    for lp,rewards in zip(lps,rs):
        if not lp:continue
        running=0.;ret=[]
        for r in reversed(rewards):running=r+.95*running;ret.insert(0,running)
        baseline=[sum(rewards[:i+1])/(i+1) for i in range(len(rewards))]
        losses.append(-(torch.stack(lp)*torch.tensor([r-b for r,b in zip(ret,baseline)])).sum())
    loss=sum(losses)/len(ps) if losses else torch.tensor(0.)
    if losses:
        assert bool(torch.isfinite(loss));loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert math.isfinite(norm);opt.step()
    else:norm=0.
    return dict(loss=float(loss.detach()),gradient_norm=norm,batch_return=sum(sum(r) for r in rs)/len(ps)),counts,ncalls


def fit_giph(out,source=None,fast=False):
    if fast:
        from daglab.external_giph_fast import FastBudgetGiPHActor,FastBudgetProblem
        factory,problem_class=FastBudgetGiPHActor,FastBudgetProblem
        assert json.loads((BASE/'giph_fast_equivalence.json').read_text())['status']=='PASS'
    else:factory,problem_class=BudgetGiPHActor,BudgetProblem
    torch.manual_seed(11);model=factory();opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    torch.save(dict(model=model.state_dict(),torch_rng=torch.get_rng_state()),out/'initial.pt')
    updates=0;graphs=0;actions=0;calls=0;digest=hashlib.sha256();started=time.perf_counter();iterator=iter(train_rows(1));source_ck=None
    if source:
        source_ck=torch.load(source,weights_only=False);model.load_state_dict(source_ck['model']);opt.load_state_dict(copy.deepcopy(source_ck['optimizer']))
        updates=source_ck['updates'];graphs=source_ck['graphs'];actions=source_ck['actions']
        prefix=[json.loads(line) for line in (source.parent/'training.jsonl').open()]
        calls=next(r['batch_forward_calls'] for r in prefix if r['update']==updates)
        for _ in range(graphs):digest.update((next(iterator)['key']+'\n').encode())
        initial=torch.load(source.parent/'initial.pt',weights_only=False);torch.save(initial,out/'initial.pt')
        torch.set_rng_state(source_ck['torch_rng'])
        save(out/'resume_audit.json',dict(source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_update=updates,source_graphs=graphs,source_actions=actions,model_and_Adam_exact=True,source_rng=True,skipped_order_prefix=graphs,
            same_model_features_rewards_and_training_protocol=True,optimization='vectorized graph and level mask only'))
    first=True
    with (out/'training.jsonl').open('w') as log:
        while True:
            batch=list(itertools.islice(iterator,16))
            if not batch:break
            ps=[problem_class(to_instance(row)) for row in batch];maps=[p.random_mapping(stable_seed(r['key'],11000)) for p,r in zip(ps,batch)]
            before=torch.get_rng_state().clone();rec,counts,ncalls=giph_optimize(model,opt,ps,maps)
            if first and source:
                after=torch.get_rng_state();reference=BudgetGiPHActor();reference.load_state_dict(source_ck['model'])
                ropt=torch.optim.Adam(reference.parameters(),lr=3e-4);ropt.load_state_dict(copy.deepcopy(source_ck['optimizer']))
                slowps=[BudgetProblem(to_instance(row)) for row in batch];slowmaps=[p.random_mapping(stable_seed(r['key'],11000)) for p,r in zip(slowps,batch)]
                torch.set_rng_state(before);rrec,rcounts,rcalls=giph_optimize(reference,ropt,slowps,slowmaps)
                assert rec==rrec and counts==rcounts and ncalls==rcalls
                assert same(model.state_dict(),reference.state_dict(),torch) and same(opt.state_dict(),ropt.state_dict(),torch)
                torch.set_rng_state(after);save(out/'first_update_audit.json',dict(actual_slow_fast_update_model_Adam_exact=True,counts_and_rewards_exact=True))
            first=False
            updates+=1;graphs+=len(batch);actions+=sum(counts);calls+=ncalls
            for row in batch:digest.update((row['key']+'\n').encode())
            rec=dict(update=updates,graphs=graphs,actions=actions,batch_forward_calls=calls,lr=3e-4,**rec)
            log.write(json.dumps(rec)+'\n');log.flush()
            if updates%25==0:
                save(out/'progress.json',dict(stage='training',**rec));print('GIPH VISIBLE TRAIN',updates,graphs,actions,flush=True)
            if updates%100==0:torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),updates=updates,graphs=graphs,actions=actions),out/f'update{updates}.pt')
    assert (updates,graphs)==(1079,17261)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),updates=updates,graphs=graphs,actions=actions),out/'final.pt')
    save(out/'training_summary.json',dict(updates=updates,graphs=graphs,actions=actions,batch_forward_calls=calls,
        ordered_keys_sha256=digest.hexdigest(),seconds=time.perf_counter()-started,parameters=sum(p.numel() for p in model.parameters()),seed=11))
    evaluate_giph(out,model,factory,problem_class)


def evaluate_giph(out,model,factory=BudgetGiPHActor,problem_class=BudgetProblem):
    initial=factory();initial.load_state_dict(torch.load(out/'initial.pt',weights_only=False)['model'])
    records=[];iterator=iter(cases());started=time.perf_counter();equiv=0;batchcalls=0
    with (out/'cases.jsonl').open('w') as stream,torch.no_grad():
        while True:
            batch=list(itertools.islice(iterator,16))
            if not batch:break
            ps=[problem_class(c['inst']) for c in batch];heft=[p.schedule() for p in ps]
            assert all(h['C']==c['references']['HEFT'] for h,c in zip(heft,batch));methods=[{} for _ in batch]
            for kind,m,steps in [('random',model,[2*p.n for p in ps]),('HEFT',model,[2*p.n for p in ps]),('initial_HEFT16',initial,[16]*len(ps))]:
                maps=[p.random_mapping(stable_seed(c['meta']['key'],11)) for p,c in zip(ps,batch)] if kind=='random' else [list(h['mapping']) for h in heft]
                _,_,snaps,counts,calls=giph_paths(m,ps,maps,steps);equiv+=sum(counts);batchcalls+=calls
                for j,p in enumerate(ps):
                    points=[('',16)] if kind=='initial_HEFT16' else [('_16',min(16,steps[j])),('_64',min(64,steps[j])),('_2n',steps[j])]
                    for suffix,point in points:
                        value=snaps[j][point];f=p.finish(value['mapping']);assert (f['C'],f['cost'])==(value['C'],value['cost'])
                        methods[j]['GiPH_'+kind+suffix]=dict(C=f['C'],cost=f['cost'],feasible=True,actual_steps=value['actual_steps'],budget=point,includes_initial=True)
            for c,values in zip(batch,methods):
                record=dict(**c['meta'],references=c['references'],methods=values);records.append(record);stream.write(json.dumps(record)+'\n')
            stream.flush();save(out/'progress.json',dict(stage='evaluation',completed=len(records),total=1862,graph_equivalent_forwards=equiv,batch_forward_calls=batchcalls))
            print('GIPH VISIBLE EVAL',len(records),'seconds',round(time.perf_counter()-started,1),flush=True)
    summarize(out,records,dict(budget_visible=True,training=json.loads((out/'training_summary.json').read_text()),evaluation_seconds=time.perf_counter()-started,
        graph_equivalent_forwards=equiv,batch_forward_calls=batchcalls,HEFT_identity_cases=1862))


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['cai','graphormer','cai_sparse','giph'],required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--fast-giph',action='store_true');p.add_argument('--output',type=Path);args=p.parse_args()
    out=(args.output or BASE/args.method).resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    try:
        bindings=[]
        manifest=json.loads((DATA/'MANIFEST.json').read_text())
        for shard in manifest['shards']:
            if shard['split'] in ('train','val'):
                digest=hashlib.sha256((DATA/shard['path']).read_bytes()).hexdigest();assert digest==shard['sha256'];bindings.append(dict(path=shard['path'],sha256=digest))
        save(out/'metadata.json',dict(protocol='protocols/external_budget_visibility_fix_20261028.md',method=args.method,
            checked_shards=bindings,seed=11,TRAIN=17261,VAL=862,WF=200,TPC=800,teacher_labels=False,TEST=False,
            initial_trained_weights_used=False,code={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(__file__).resolve(),ROOT/'daglab/external_budget_visible.py',ROOT/'daglab/external_baseline_sim.py',ROOT/'daglab/external_baseline_models.py',ROOT/'daglab/external_graphormer.py',ROOT/'scripts/run_external_baselines.py']}))
        check(out)
        fit_giph(out,args.resume.resolve() if args.resume else None,args.fast_giph) if args.method=='giph' else fit_cai(args.method,out)
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
