#!/usr/bin/env python3
"""Equivalent-feature/logit/gradient proof before accepting a GiPH speed refactor."""
import copy,itertools,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,'/private/tmp/dag-baseline-deps-20261028')
import torch,numpy as np
from daglab.external_budget_visible import BudgetProblem,BudgetGiPHActor
from daglab.external_giph_fast import FastBudgetProblem,FastBudgetGiPHActor
from daglab.structured_dataset import to_instance
from scripts.run_external_baselines import get_train,save,giph_paths

def main():
    torch.set_num_threads(1);rows=[];seen=set()
    for r in get_train():
        shape=r['input']['n'],r['input']['k']
        if shape not in seen:rows.append(r);seen.add(shape)
        if len(rows)==12:break
    checks=[];ps=[];fps=[]
    for row in rows:
        p=BudgetProblem(to_instance(row));fp=FastBudgetProblem(p.inst);ps.append(p);fps.append(fp)
        for seed in [11,22,33]:
            mapping=p.random_mapping(seed);s=p.state(mapping);g=p.relocation_graph(mapping,s);fg=fp.relocation_graph(mapping,s)
            for key in ['src','dst','mask','levels']:assert g[key]==fg[key]
            err=max(float(np.max(np.abs(np.asarray(g[k])-np.asarray(fg[k])))) if len(g[k]) else 0. for k in ['x','edge']);assert err<1e-12
            torch.manual_seed(11);m=BudgetGiPHActor();fm=FastBudgetGiPHActor();fm.load_state_dict(m.state_dict())
            z=m(g);fz=fm(fg);mask=torch.tensor(g['mask'])
            if mask.any():
                assert torch.equal(z[mask],fz[mask])
                action=int(z.argmax());(-torch.log_softmax(z,0)[action]).backward();(-torch.log_softmax(fz,0)[action]).backward()
                assert all(torch.equal(a.grad,b.grad) for a,b in zip(m.parameters(),fm.parameters()))
            checks.append(dict(n=p.n,k=p.k,seed=seed,feature_max_error=err,logits_gradients_exact=True))
    measurements={}
    for name,problems,factory in [('original',ps,BudgetGiPHActor),('fast',fps,FastBudgetGiPHActor)]:
        model=factory();maps=[p.random_mapping(11) for p in problems];begin=time.perf_counter()
        with torch.no_grad():giph_paths(model,problems,maps,[4]*len(ps))
        measurements[name]=time.perf_counter()-begin
    out=ROOT/'runs/external_baselines_budget_visible_20261028/giph_fast_equivalence.json'
    save(out,dict(status='PASS',checks=checks,forward_gradient_exact=True,timings=measurements,speedup=measurements['original']/measurements['fast']))
    print('PASS',measurements,'speedup',measurements['original']/measurements['fast'])

if __name__=='__main__':main()
