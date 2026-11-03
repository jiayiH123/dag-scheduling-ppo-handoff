#!/usr/bin/env python3
import argparse,gzip,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--panel',choices=['synthetic_val862','wf960','wf400','parrot498'],default='synthetic_val862')
    parser.add_argument('--out',required=True);parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    import torch
    from daglab.state_score_torch import StateActor
    from daglab.portable_runtime import evaluation
    torch.set_num_threads(1);model=StateActor('both')
    model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=False)['model'])
    result=evaluation(model,args.panel,args.limit);path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt') as stream:json.dump(result,stream)
    print(json.dumps(result['summary'],indent=2))

if __name__=='__main__':main()
