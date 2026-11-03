#!/usr/bin/env python3
"""Frozen external adaptations: budget heuristics or TRAIN-only learned policies."""
import argparse
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if Path('/private/tmp/dag-baseline-deps-20261028').exists():
    sys.path.insert(0,'/private/tmp/dag-baseline-deps-20261028')
DATA=ROOT/'data/structured_scheduling_v1'
OUT=ROOT/'runs/external_baselines_20261028'


def save(path,obj):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def get_train():
    from daglab.structured_unique import iter_unique_rows
    from daglab.structured_gnn_simulator import shuffled
    return shuffled(iter_unique_rows(DATA,'train'),2026102804)


def cases():
    from daglab.structured_unique import iter_unique_rows
    from daglab.structured_dataset import to_instance
    from daglab.offline_data import dense_instance
    from scripts.evaluate_public_lr import load_inputs
    refs=[json.loads(line) for line in (ROOT/'runs/score_ranges_multival_20261028/cases.jsonl').open()]
    source=list(iter_unique_rows(DATA,'val'))+load_inputs()[0]
    assert len(refs)==len(source)==1862
    for i,(r,ref) in enumerate(zip(source,refs)):
        assert r['key']==ref['key']
        yield dict(meta={k:v for k,v in ref.items() if k not in ['methods','references']},
            references=dict(ref['references'],GNN_B=ref['methods']['control']['C']),
            inst=to_instance(r) if i<862 else dense_instance(r,max_tasks=300))


def preflight(out):
    import torch
    import numpy as np
    from daglab.external_baseline_sim import Problem
    from daglab.external_baseline_models import GATLayer,TopoMP,CAIActor,GiPHActor,merge_relocation_graphs
    from daglab.structured_dataset import to_instance
    from daglab.reference import core
    seen=set();problems=[]
    for row in get_train():
        inst=to_instance(row);shape=inst['n'],len(inst['capacities'])
        if shape not in seen:seen.add(shape);problems.append(Problem(inst))
        if len(seen)==12:break
    results=[]
    for p in problems:
        eft=p.schedule();original=core.schedule(p.inst,'HEFT')
        assert (eft['C'],eft['cost'])==(original['makespan'],original['cost'])
        for rule in ['BHEFT','BUDA','BUDA_printedledger','BUDA_unitbalanced']:
            r=p.schedule(rule=rule);f=p.finish(r['mapping'],r['order']);assert (r['C'],r['cost'])==(f['C'],f['cost'])
        rng=np.random.default_rng(131);priorities=rng.normal(size=p.n).tolist()
        order,ds=p.priority_order(priorities);p.engine.check_order(order)
        assert all(d['action']==order[j] and d['action'] in d['available'] for j,d in enumerate(ds))
        model=CAIActor();logits,raw=model(p.priority_features());(-logits[order[0]]+.001*(raw**2).mean()).backward()
        assert any(q.grad is not None and bool((q.grad!=0).any()) for q in model.parameters())
        model=GiPHActor();mapping=p.random_mapping(131);state=p.state(mapping)
        assert (state['C'],state['cost'])==p.engine.evaluate(mapping,p.order)
        g=p.relocation_graph(mapping,state);g2=p.relocation_graph(eft['mapping'],eft)
        z=model(g);batch,sizes=merge_relocation_graphs([g,g2]);zb=model(batch)[:len(z)]
        valid=torch.tensor(g['mask']);assert torch.equal(torch.isfinite(z),valid)
        if valid.any():
            assert float((z[valid]-zb[valid]).abs().max())<2e-6
            chosen=int(z.argmax());v,pool=divmod(chosen,p.k);mapping[v]=pool
            after=p.state(mapping);assert after['cost']<=p.budget
            (-torch.log_softmax(z,0)[chosen]).backward()
            assert all(q.grad is None or bool(torch.isfinite(q.grad).all()) for q in model.parameters())
        results.append(dict(n=p.n,k=p.k,HEFT_exact=True,valid_order=True,finite_gradients=True))
    # Independent explicit per-edge/per-node references for operator ports.
    layer=GATLayer(3,2);x=torch.tensor([[1.,2.,3.],[0.,1.,2.],[2.,3.,1.]])
    src=torch.tensor([0,1,0,1,2]);dst=torch.tensor([1,2,0,1,2]);actual=layer(x,src,dst)
    h=layer.linear(x);reference=[]
    for v in range(3):
        ids=(dst==v).nonzero().flatten();s=src[ids]
        weights=torch.softmax(torch.nn.functional.leaky_relu((h[s]*layer.att_src).sum(-1)+(h[v]*layer.att_dst).sum(),.2),0)
        reference.append((weights[:,None]*h[s]).sum(0)+layer.bias)
    assert torch.allclose(actual,torch.stack(reference),atol=1e-6,rtol=1e-6)
    layer=TopoMP(h=2,e=1);initial=torch.tensor([[1.,0.],[0.,1.],[2.,1.]])
    src=torch.tensor([0,1]);dst=torch.tensor([2,2]);edges=torch.tensor([[1.],[2.]])
    actual=layer(initial,src,dst,edges,[[0,1],[2]])
    hs=[]
    for v in range(3):
        incoming=[i for i in range(len(src)) if int(dst[i])==v]
        msgs=[torch.relu(layer.pre(torch.cat([hs[int(src[i])],edges[i]]))) for i in incoming]
        agg=torch.stack(msgs).mean(0) if msgs else initial.new_zeros(3)
        hs.append(initial[v]+torch.relu(layer.update(agg)))
    assert torch.allclose(actual,torch.stack(hs),atol=1e-6,rtol=1e-6)
    save(out/'preflight.json',dict(status='PASS',TRAIN_shapes=results,GAT_reference=True,GiPH_reference=True,
        disjoint_batch_logits_tolerance=2e-6,source_registry_sha256=hashlib.sha256((ROOT/'vendor/external_baseline_sources_20261028/MANIFEST.json').read_bytes()).hexdigest()))


def evaluate_heuristics(out):
    from daglab.external_baseline_sim import Problem
    rules=['BHEFT','BUDA','BUDA_printedledger','BUDA_unitbalanced'];started=time.perf_counter();records=[]
    with (out/'cases.jsonl').open('w') as log:
        for i,case in enumerate(cases(),1):
            p=Problem(case['inst']);eft=p.schedule();assert eft['C']==case['references']['HEFT']
            methods={}
            for rule in rules:
                begin=time.perf_counter();r=p.schedule(rule=rule);f=p.finish(r['mapping'],r['order'])
                assert (r['C'],r['cost'])==(f['C'],f['cost'])
                methods[rule]=dict(C=f['C'],cost=f['cost'],feasible=True,seconds=time.perf_counter()-begin,**r['stats'])
            value=dict(**case['meta'],references=case['references'],methods=methods)
            records.append(value);log.write(json.dumps(value,allow_nan=False)+'\n');log.flush()
            if i%100==0:save(out/'progress.json',dict(stage='evaluation',completed=i,total=1862));print('HEURISTICS',i,flush=True)
    summarize(out,records,dict(seconds=time.perf_counter()-started,HEFT_identity_cases=1862))


def summarize(out,records,extra):
    summary={}
    for panel in sorted({r['dataset'] for r in records}):
        rows=[r for r in records if r['dataset']==panel];methods={}
        for name in rows[0]['methods']:
            ratios=[r['methods'][name]['C']/r['references']['HEFT'] for r in rows]
            families=sorted({r['family'] for r in rows})
            family={f:sum(r['methods'][name]['C']/r['references']['HEFT'] for r in rows if r['family']==f)/sum(r['family']==f for r in rows) for f in families}
            methods[name]=dict(C_HEFT=sum(ratios)/len(rows),family_macro_C_HEFT=sum(family.values())/len(families),
                family_C_HEFT=family,C_Teacher=sum(r['methods'][name]['C']/r['references']['Teacher64'] for r in rows)/len(rows),
                wins_HEFT=sum(r<1-1e-12 for r in ratios),ties_HEFT=sum(abs(r-1)<1e-12 for r in ratios),
                losses_HEFT=sum(r>1+1e-12 for r in ratios),
                mean_C_GNN=sum(r['methods'][name]['C']/r['references']['GNN_B'] for r in rows)/len(rows))
        summary[panel]=dict(configs=len(rows),methods=methods)
    save(out/'summary.json',dict(status='complete',panels=summary,configs=len(records),all_feasible=True,TEST_evaluated=False,**extra))
    print('COMPLETE',out.name,{p:{a:round(v['C_HEFT'],6) for a,v in r['methods'].items()} for p,r in summary.items()},flush=True)


def cai_roll(p,logits,sample):
    import torch
    scores=logits.detach()
    if sample:scores=scores-torch.log(-torch.log(torch.rand_like(scores).clamp(1e-7,1-1e-7)))
    order,decisions=p.priority_order(scores.tolist());r=p.schedule(order=order)
    mask=torch.zeros((p.n,p.n),dtype=torch.bool)
    for i,d in enumerate(decisions):mask[i,d['available']]=True
    lp=logits[order]-torch.logsumexp(logits[None,:].expand(p.n,-1).masked_fill(~mask,-torch.inf),-1)
    return r,lp


def train_cai(out):
    import torch
    from daglab.external_baseline_models import CAIActor
    from daglab.external_baseline_sim import Problem
    from daglab.structured_dataset import to_instance
    torch.manual_seed(11);model=CAIActor();opt=torch.optim.RMSprop(model.parameters(),lr=7e-4,alpha=.99,eps=1e-5)
    torch.save(dict(model=model.state_dict()),out/'initial.pt')
    iterator=iter(get_train());updates=0;presentations=0;actions=0;digest=hashlib.sha256();started=time.perf_counter()
    with (out/'training.jsonl').open('w') as log:
        while True:
            batch=list(itertools.islice(iterator,8))
            if not batch:break
            opt.zero_grad(set_to_none=True);items=[];raws=[];ratios=[]
            for row in batch:
                p=Problem(to_instance(row));baseline=p.schedule();g=p.priority_features();logits,raw=model(g);raws.append(raw.square().mean())
                for j in range(4):
                    r,lp=cai_roll(p,logits,True)
                    reward=torch.tensor([(baseline['ends'][v]-r['ends'][v])/baseline['C'] for v in r['order']],dtype=torch.float32)
                    items.append((lp,reward));ratios.append(r['C']/baseline['C']);actions+=p.n
                digest.update((row['key']+'\n').encode());presentations+=1
            all_rewards=torch.cat([reward for _,reward in items]);active=all_rewards[all_rewards!=0]
            mean=active.mean() if len(active) else 0.;std=active.std(unbiased=False) if len(active) else 1.
            loss=sum((-(reward-mean).div(std+1e-10).clamp(-2,2)*lp).mean()+.01*lp.mean() for lp,reward in items)/len(items)
            loss=loss+.001*torch.stack(raws).mean();assert bool(torch.isfinite(loss));loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),.5));assert math.isfinite(norm);opt.step();updates+=1
            record=dict(update=updates,graphs=presentations,samples=presentations*4,actions=actions,loss=float(loss.detach()),
                sampled_C_HEFT=sum(ratios)/len(ratios),gradient_norm=norm,lr=opt.param_groups[0]['lr'])
            log.write(json.dumps(record)+'\n');log.flush()
            if updates%100==0:save(out/'progress.json',dict(stage='training',**record));print('CAI TRAIN',updates,presentations,round(record['sampled_C_HEFT'],4),flush=True)
    assert (updates,presentations)==(2158,17261)
    assert actions==1122860*4
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),updates=updates,graphs=presentations,
        torch_rng=torch.get_rng_state()),out/'final.pt')
    save(out/'training_summary.json',dict(updates=updates,graphs=presentations,samples=4*presentations,actions=actions,
        ordered_keys_sha256=digest.hexdigest(),seconds=time.perf_counter()-started,parameters=sum(p.numel() for p in model.parameters()),seed=11))
    evaluate_cai(out,model)


def evaluate_cai(out,model):
    import torch
    from daglab.external_baseline_models import CAIActor
    from daglab.external_baseline_sim import Problem
    initial=CAIActor();initial.load_state_dict(torch.load(out/'initial.pt',weights_only=False)['model'])
    records=[];started=time.perf_counter()
    with (out/'cases.jsonl').open('w') as stream,torch.no_grad():
        for i,case in enumerate(cases(),1):
            p=Problem(case['inst']);methods={};assert p.schedule()['C']==case['references']['HEFT']
            for name,m in [('CAI_initial',initial),('CAI_DR_BiGAT',model)]:
                begin=time.perf_counter();before=m.forward_calls;logits,_=m(p.priority_features());r,_=cai_roll(p,logits,False)
                assert m.forward_calls-before==1
                f=p.finish(r['mapping'],r['order']);assert (f['C'],f['cost'])==(r['C'],r['cost'])
                methods[name]=dict(C=f['C'],cost=f['cost'],feasible=True,actor_calls=1,seconds=time.perf_counter()-begin)
            r=dict(**case['meta'],references=case['references'],methods=methods);records.append(r);stream.write(json.dumps(r)+'\n');stream.flush()
            if i%100==0:save(out/'progress.json',dict(stage='evaluation',completed=i,total=1862));print('CAI EVAL',i,flush=True)
    summarize(out,records,dict(evaluation_seconds=time.perf_counter()-started,training=json.loads((out/'training_summary.json').read_text()),HEFT_identity_cases=1862))


def giph_paths(model,problems,mappings,steps,train=False):
    import torch
    from daglab.external_baseline_models import merge_relocation_graphs
    states=[p.state(m) for p,m in zip(problems,mappings)];initial_C=[s['C'] for s in states]
    best=[dict(C=s['C'],cost=s['cost'],mapping=list(m)) for s,m in zip(states,mappings)]
    logs=[[] for _ in problems];rewards=[[] for _ in problems];snapshots=[{} for _ in problems]
    previous=[None]*len(problems);actual=[0]*len(problems);done=set();batch_forwards=0
    for step in range(1,max(steps)+1):
        ids=[];graphs=[]
        for j,p in enumerate(problems):
            if j in done or step>steps[j]:continue
            g=p.relocation_graph(mappings[j],states[j],previous[j])
            if not any(g['mask']):done.add(j);continue
            ids.append(j);graphs.append(g)
        if ids:
            graph,sizes=merge_relocation_graphs(graphs);z=model(graph);batch_forwards+=1
            for j,(a,b) in zip(ids,sizes):
                lp=torch.log_softmax(z[a:b],0)
                selected=int(torch.multinomial(lp.exp(),1)) if train else int(lp.argmax())
                assert graph['mask'][a+selected];logs[j].append(lp[selected])
                v,pool=divmod(selected,problems[j].k);mappings[j][v]=pool;previous[j]=v
                new=problems[j].state(mappings[j]);assert new['cost']<=problems[j].budget
                rewards[j].append((states[j]['C']-new['C'])/initial_C[j]);states[j]=new;actual[j]+=1
                if (new['C'],new['cost'])<(best[j]['C'],best[j]['cost']):
                    best[j]=dict(C=new['C'],cost=new['cost'],mapping=list(mappings[j]))
        for j in range(len(problems)):
            if step in [16,64,steps[j]]:snapshots[j][step]=dict(**best[j],actual_steps=actual[j])
        if not ids and len(done)==len(problems):break
    for j in range(len(problems)):
        for point in sorted({min(16,steps[j]),min(64,steps[j]),steps[j]}):
            snapshots[j].setdefault(point,dict(**best[j],actual_steps=actual[j]))
    return logs,rewards,snapshots,actual,batch_forwards


def train_giph(out):
    import torch
    from daglab.external_baseline_sim import Problem,stable_seed
    from daglab.external_baseline_models import GiPHActor
    from daglab.structured_dataset import to_instance
    torch.manual_seed(11);model=GiPHActor();opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    torch.save(dict(model=model.state_dict()),out/'initial.pt')
    iterator=iter(get_train());updates=0;presentations=0;actions=0;batch_forwards=0;digest=hashlib.sha256();started=time.perf_counter()
    with (out/'training.jsonl').open('w') as stream:
        while True:
            batch=list(itertools.islice(iterator,16))
            if not batch:break
            ps=[Problem(to_instance(row)) for row in batch];maps=[p.random_mapping(stable_seed(row['key'],11000)) for p,row in zip(ps,batch)]
            opt.zero_grad(set_to_none=True)
            logs,rewards,_,counts,calls=giph_paths(model,ps,maps,[16]*len(ps),train=True)
            losses=[]
            for lp,rs in zip(logs,rewards):
                if not lp:continue
                running=0.;returns=[]
                for r in reversed(rs):running=r+.95*running;returns.insert(0,running)
                baseline=[sum(rs[:i+1])/(i+1) for i in range(len(rs))]
                advantage=torch.tensor([r-b for r,b in zip(returns,baseline)],dtype=torch.float32)
                losses.append(-(torch.stack(lp)*advantage).sum())
            loss=sum(losses)/len(ps) if losses else torch.tensor(0.)
            if losses:
                assert bool(torch.isfinite(loss));loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert math.isfinite(norm);opt.step()
            else:norm=0.
            updates+=1;presentations+=len(batch);actions+=sum(counts);batch_forwards+=calls
            for row in batch:digest.update((row['key']+'\n').encode())
            record=dict(update=updates,graphs=presentations,actions=actions,batch_forward_calls=batch_forwards,
                loss=float(loss.detach()),gradient_norm=norm,lr=opt.param_groups[0]['lr'],batch_return=sum(sum(r) for r in rewards)/len(ps))
            stream.write(json.dumps(record)+'\n');stream.flush()
            if updates%25==0:save(out/'progress.json',dict(stage='training',**record));print('GIPH TRAIN',updates,presentations,actions,flush=True)
    assert (updates,presentations)==(1079,17261)
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),updates=updates,graphs=presentations,
        torch_rng=torch.get_rng_state()),out/'final.pt')
    save(out/'training_summary.json',dict(updates=updates,graphs=presentations,actions=actions,batch_forward_calls=batch_forwards,
        ordered_keys_sha256=digest.hexdigest(),seconds=time.perf_counter()-started,parameters=sum(p.numel() for p in model.parameters()),seed=11))
    evaluate_giph(out,model)


def evaluate_giph(out,model):
    import torch
    from daglab.external_baseline_sim import Problem,stable_seed
    from daglab.external_baseline_models import GiPHActor
    initial=GiPHActor();initial.load_state_dict(torch.load(out/'initial.pt',weights_only=False)['model'])
    iterator=iter(cases());records=[];started=time.perf_counter();equivalent=0;actualcalls=0
    with (out/'cases.jsonl').open('w') as stream,torch.no_grad():
        while True:
            batch=list(itertools.islice(iterator,16))
            if not batch:break
            ps=[Problem(case['inst']) for case in batch];heft=[p.schedule() for p in ps]
            assert all(s['C']==c['references']['HEFT'] for s,c in zip(heft,batch))
            result=[{} for _ in batch]
            for start_kind,m,budgets in [('random',model,[2*p.n for p in ps]),('HEFT',model,[2*p.n for p in ps]),('initial_HEFT16',initial,[16]*len(ps))]:
                mappings=[p.random_mapping(stable_seed(c['meta']['key'],11)) for p,c in zip(ps,batch)] if start_kind=='random' else [list(h['mapping']) for h in heft]
                begin=time.perf_counter();before=m.forward_calls
                _,_,snaps,counts,calls=giph_paths(m,ps,mappings,budgets)
                assert m.forward_calls-before==calls;equivalent+=sum(counts);actualcalls+=calls
                seconds=time.perf_counter()-begin
                for j,p in enumerate(ps):
                    points=[('',16)] if start_kind=='initial_HEFT16' else [('_16',min(16,budgets[j])),('_64',min(64,budgets[j])),('_2n',budgets[j])]
                    for suffix,point in points:
                        value=snaps[j][point];f=p.finish(value['mapping']);assert (f['C'],f['cost'])==(value['C'],value['cost'])
                        tag='GiPH_'+start_kind+suffix
                        result[j][tag]=dict(C=f['C'],cost=f['cost'],feasible=True,actual_steps=value['actual_steps'],
                            budget=point,batch_amortized_fullpath_seconds=seconds/len(ps),includes_initial=True,
                            truncated_at_2n=(suffix=='_64' and budgets[j]<64))
            for case,methods in zip(batch,result):
                r=dict(**case['meta'],references=case['references'],methods=methods);records.append(r);stream.write(json.dumps(r)+'\n')
            stream.flush();save(out/'progress.json',dict(stage='evaluation',completed=len(records),total=1862,graph_equivalent_forwards=equivalent,batch_forward_calls=actualcalls))
            print('GIPH EVAL',len(records),'elapsed',round(time.perf_counter()-started,1),flush=True)
    summarize(out,records,dict(evaluation_seconds=time.perf_counter()-started,training=json.loads((out/'training_summary.json').read_text()),
        graph_equivalent_forwards=equivalent,batch_forward_calls=actualcalls,HEFT_identity_cases=1862,timing='shared disjoint batching and concurrent LR; not standalone latency'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['heuristics','cai','giph'],required=True);parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();out=OUT/args.method;out.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(131);random.seed(131)
    try:
        bindings=[]
        manifest=json.loads((DATA/'MANIFEST.json').read_text())
        for shard in manifest['shards']:
            if shard['split'] in ('train','val'):
                path=DATA/shard['path'];digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==shard['sha256'];bindings.append(dict(path=shard['path'],sha256=digest))
        save(out/'metadata.json',dict(protocol='protocols/external_baselines_20261028.md',method=args.method,seed=11,TRAIN=17261,
            VAL=862,WF=200,TPC=800,checked_shards=bindings,TEST=False,teacher_mapping_labels_used=False,
            torch=torch.__version__,source_manifest=json.loads((ROOT/'vendor/external_baseline_sources_20261028/MANIFEST.json').read_text()),
            adaptation_not_original_reproduction=True,training_budget_not_matched_to_our7pass=True))
        preflight(out);print('PREFLIGHT PASS',args.method,flush=True)
        if args.preflight_only:return
        dict(heuristics=evaluate_heuristics,cai=train_cai,giph=train_giph)[args.method](out)
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':main()
