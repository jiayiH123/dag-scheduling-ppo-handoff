#!/usr/bin/env python3
"""Portable original StateActor SFT recipe, both teacher variants, exact resume."""
import argparse,gzip,json,math,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--teacher',choices=['strong','no_search'],required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--epochs',type=int,default=7)
    parser.add_argument('--seed',type=int,default=28);parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--lr',type=float,default=.001);parser.add_argument('--eval-every',type=int,default=100)
    parser.add_argument('--max-updates',type=int,help='Optional explicitly recorded short engineering run')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    import torch
    from daglab.portable_runtime import Data,evaluation
    from daglab.state_score_torch import StateActor,bc_loss
    from daglab.module_lab.models import tensors
    from daglab.epoch_training_plan import make_epoch_plan
    from daglab.sft_three_panel import write_json
    from daglab.structured_dataset import sha256
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed);random.seed(args.seed)
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True);data=Data(args.teacher)
    batches,_,plan_sha=make_epoch_plan(data.train,['synthetic'],args.seed,args.epochs,args.batch)
    target=min(len(batches),args.max_updates) if args.max_updates else len(batches)
    model=StateActor('both');optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    binding=dict(manifest=sha256(ROOT/'data/MANIFEST.json'),plan=plan_sha,
        code={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'daglab/portable_runtime.py',
              ROOT/'daglab/state_score.py',ROOT/'daglab/state_score_torch.py',ROOT/'daglab/module_lab/models.py']},
        teacher=args.teacher,seed=args.seed,batch=args.batch,epochs=args.epochs,lr=args.lr,target=target,eval_every=args.eval_every)
    step=0;history=[];records=[];best=None
    if args.resume:
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False);assert saved['binding']==binding
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng'])
        step=saved['step'];history=saved['history'];records=saved['records'];best=saved['best']
    else:assert not (out/'last.pt').exists(),'Use --resume or choose a new output path'
    write_json(out/'config.json',dict(args=vars(args),binding=binding,parameters=32612))
    def save(name):
        temp=out/(name+'.tmp.pt');torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),step=step,history=history,records=records,best=best,binding=binding),temp);temp.replace(out/(name+'.pt'))
        (out/'training.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    def observe():
        nonlocal best
        result=evaluation(model);point=dict(step=step,**result['summary']);history.append(point)
        with gzip.open(out/f'val_synthetic_val862_{step:05d}.json.gz','wt') as stream:json.dump(dict(**point,rows=result['rows']),stream)
        write_json(out/'val_synthetic_val862_curve.json',history)
        if best is None or point['C_over']['HEFT']<best['C_over']['HEFT']:best=point;save('best')
        save('last');print('VAL',step,point['C_over']['HEFT'],flush=True)
    if step==0 and not history:observe()
    try:
        while step<target:
            begin=time.perf_counter();optimizer.zero_grad(set_to_none=True);model.train();batch=batches[step];losses=[]
            for key in batch:
                _,row=data.record(key);loss=bc_loss(model(tensors(row['graph'],'cpu')),row['decisions'])
                assert bool(torch.isfinite(loss));(loss/len(batch)).backward();losses.append(float(loss.detach()))
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));assert math.isfinite(norm)
            optimizer.step();step+=1;records.append(dict(step=step,NLL=sum(losses)/len(losses),gradient_norm=norm,seconds=time.perf_counter()-begin))
            if step%50==0:save('last')
            if step%args.eval_every==0 or step==target:observe()
            write_json(out/'progress.json',dict(stage='training',step=step,target=target,loss=records[-1]['NLL']))
        save('last');write_json(out/'COMPLETE.json',dict(status='complete',step=step,best=best,binding=binding))
    finally:data.close()

if __name__=='__main__':main()
