#!/usr/bin/env python3
"""Start a new repeat512 PPO experiment from either supplied SFT model."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from daglab.sft_three_panel import write_json
import scripts.train_teacher_search_ppo_20261102 as original

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['strong','no_search'],required=True)
    parser.add_argument('--seed',type=int,default=41)
    parser.add_argument('--out',required=True)
    parser.add_argument('--source',help='Optional other compatible StateActor checkpoint')
    parser.add_argument('--collections',type=int,default=128)
    parser.add_argument('--actor-lr',type=float,default=3e-5)
    parser.add_argument('--critic-lr',type=float,default=.001)
    parser.add_argument('--temperature',type=float,default=.5)
    parser.add_argument('--entropy',type=float,default=.001)
    parser.add_argument('--clip',type=float,default=.2)
    parser.add_argument('--kl-stop',type=float,default=.02)
    parser.add_argument('--eval-every',type=int,default=100)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();assert 1<=args.collections<=128 and args.eval_every>0
    cfg=original.config();cfg['run']=str(Path(args.out))
    cfg['eval_every_actor_updates']=args.eval_every
    cfg['PPO'].update(collections=args.collections,actor_lr=args.actor_lr,critic_lr=args.critic_lr,
        temperature=args.temperature,entropy=args.entropy,clip=args.clip,KL_stop=args.kl_stop)
    if args.source:
        if args.arm=='strong':cfg['strong_source']=args.source
        else:
            # Original constructor expects no_search_SFT_run/best.pt; add explicit override.
            cfg['portable_source_override']=args.source
    original.config=lambda:cfg
    trial=original.PPO(args.arm,args.seed,out_override=ROOT/args.out,resume=args.resume)
    write_json(trial.out/'portable_config.json',dict(configuration=cfg,args=vars(args),
        initialization='model weights only from supplied SFT; fresh PPO optimizers unless --resume'))
    try:
        if trial.collection==0 and trial.prewarm_updates==0:
            trial.warmup();trial.evaluate('SFT_start');trial.checkpoint('resume')
        for collection in range(trial.collection+1,args.collections+1):trial.update_collection(collection)
        if not any(p['actor_updates']==trial.actor_updates for p in trial.history):trial.evaluate('final')
        trial.checkpoint('last');trial.publish('complete')
        write_json(trial.out/'summary.json',dict(status='complete',arm=args.arm,seed=args.seed,best=trial.best,
            history=trial.history,actor_updates=trial.actor_updates,critic_updates=trial.critic_updates,
            prewarm_updates=trial.prewarm_updates,collections=trial.collection,binding=trial.binding,
            training_seconds=trial.training_seconds,evaluation_seconds=trial.evaluation_seconds,
            VAL_used_for_tuning=True))
    finally:trial.close()

if __name__=='__main__':main()
