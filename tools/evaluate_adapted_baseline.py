#!/usr/bin/env python3
"""Historical learned controls, explicitly separate from author-core replicas."""
import argparse,gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['cai','graphormer','giph_heft','giph_random'],required=True)
    parser.add_argument('--panel',choices=['synthetic_val862','wf960','wf400','parrot498'],default='synthetic_val862')
    parser.add_argument('--out',required=True);parser.add_argument('--limit',type=int);args=parser.parse_args()
    import torch
    from daglab.portable_runtime import Data
    from daglab.sft_three_panel import unpack,to_instance,panel_summary
    from daglab.external_budget_visible import BudgetCAIActor,BudgetGraphormer,BudgetProblem
    from daglab.external_giph_fast import FastBudgetGiPHActor,FastBudgetProblem
    from daglab.external_baseline_sim import stable_seed
    from scripts.run_external_baselines import cai_roll,giph_paths
    torch.set_num_threads(1)
    base=ROOT/'runs/external_baselines_budget_visible_20261028'
    if args.method=='cai':model=BudgetCAIActor();path=base/'cai/epoch1.pt'
    elif args.method=='graphormer':model=BudgetGraphormer();path=base/'graphormer/epoch7.pt'
    else:model=FastBudgetGiPHActor();path=base/'giph_fast_r1/final.pt'
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);model.eval()
    data=Data(panel=args.panel);rows=[]
    try:
        with torch.no_grad():
            for key in data.val_keys[:args.limit]:
                raw,payload=[unpack(v) for v in data.val.execute('SELECT raw,payload FROM records WHERE key=?',(key,)).fetchone()]
                inst=to_instance(raw)
                if args.method in ['cai','graphormer']:
                    problem=BudgetProblem(inst);scores,_=model(problem.priority_features());s,_=cai_roll(problem,scores,False)
                else:
                    problem=FastBudgetProblem(inst)
                    mapping=problem.random_mapping(stable_seed(key,11)) if args.method=='giph_random' else list(problem.schedule()['mapping'])
                    _,_,snapshots,_,_=giph_paths(model,[problem],[mapping],[2*problem.n])
                    s=dict(snapshots[0][2*problem.n],order=problem.order)
                checked=problem.finish(s['mapping'],s['order']);assert (checked['C'],checked['cost'])==(s['C'],s['cost'])
                rows.append(dict(**payload['meta'],C=checked['C'],cost=checked['cost'],feasible=True,references=payload['references']))
                if len(rows)%25==0:print('HISTORICAL_BASELINE',args.method,len(rows),flush=True)
        panel='wf_by_application' if args.panel.startswith('wf') else 'parrot_pattern_templates' if args.panel=='parrot498' else 'synthetic_val862'
        summary=panel_summary(panel,rows);out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(out,'wt') as stream:json.dump(dict(historical_adaptation=True,author_core_reproduction=False,
            method=args.method,checkpoint=str(path.relative_to(ROOT)),panel=args.panel,summary=summary,rows=rows),stream)
        print(summary['C_over']['HEFT'])
    finally:data.close()

if __name__=='__main__':main()
