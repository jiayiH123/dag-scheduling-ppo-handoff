"""CPU reference model RPC, usable even when torch and simulation live in separate envs."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'vendor'))
import torch
from budget_feedback_actor import BudgetFeedbackActor

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);args=ap.parse_args()
    torch.set_num_threads(1);model=BudgetFeedbackActor()
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(saved['state_dict']);model.eval()
    for line in sys.stdin:
        q=json.loads(line)
        if q.get('command')=='close':print(json.dumps({'closed':True}),flush=True);return
        before=model.forward_calls;start=time.perf_counter()
        with torch.no_grad():x=model(q['graph'])
        elapsed=time.perf_counter()-start;calls=model.forward_calls-before
        if calls!=1:raise AssertionError('expected one forward')
        print(json.dumps({'logits':x.tolist(),'network_calls':calls,'forward_seconds':elapsed,'parameters':sum(p.numel() for p in model.parameters())},allow_nan=False),flush=True)
if __name__=='__main__':main()
