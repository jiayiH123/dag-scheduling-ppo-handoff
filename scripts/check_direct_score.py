#!/usr/bin/env python3
"""Physical/probability/gradient gates before direct-score training."""
import hashlib,json,math,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/direct_score_autoresearch_20261029'
SIM=__import__('os').environ.get('DAGLAB_SIM_PYTHON',sys.executable)


class Bridge:
    def __init__(self,script,*args):
        self.p=subprocess.Popen([SIM,str(ROOT/script),*args],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    def __call__(self,**q):
        self.p.stdin.write(json.dumps(q,allow_nan=False)+'\n');self.p.stdin.flush();line=self.p.stdout.readline()
        if not line:raise RuntimeError('physical bridge exited')
        return json.loads(line)
    def close(self):
        if self.p.poll() is None:self(cmd='close');self.p.wait(timeout=30)


def main():
    import torch
    from daglab.direct_score_torch import DirectActor,distributions,bc_loss
    from daglab.state_score_torch import StateActor
    from daglab.module_lab.models import tensors
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=False)
    worker=Bridge('scripts/run_state_score_followup.py','--worker')
    try:
        identity=worker(cmd='init');assert identity['VAL']==862 and identity['TRAIN_probe']==128
        checks=[]
        for row in worker(cmd='checks'):
            g=tensors(row['graph'],'cpu');n,k=g['n'],g['k'];torch.manual_seed(28);historical=StateActor('both')
            torch.manual_seed(28);m=DirectActor('direct')
            for name,value in m.state_dict().items():
                if not name.startswith('head.2.'):assert torch.equal(value,historical.base.state_dict()[name])
            before=m.forward_calls;x=m(g);assert m.forward_calls-before==1 and x.shape==(n*k,1) and bool((x==0).all())
            samples=[]
            for mode in ['direct','direct_eft']:
                model=DirectActor(mode);x=model(g);loss=bc_loss(x,row['decisions'],mode);loss.backward()
                assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                lp,mask=distributions(x,row['decisions'],mode)
                if mode=='direct':
                    for i,d in enumerate(row['decisions']):
                        assert torch.allclose(lp[i,:len(d['available'])],lp.new_full((len(d['available']),),-math.log(len(d['available']))),atol=1e-6)
                # Independent loop CE and exact output-logit gradient.
                torch.manual_seed(281);z=torch.randn(n*k,1,requires_grad=True);actual=bc_loss(z,row['decisions'],mode)
                manual=[]
                for d in row['decisions']:
                    logits=z[d['available'],0]
                    if mode=='direct_eft':logits=logits+z.new_tensor(d['offsets'])
                    manual.append(torch.logsumexp(logits,0)-logits[d['selected']])
                expected=torch.stack(manual).mean()
                ga=torch.autograd.grad(actual,z,retain_graph=True)[0];gb=torch.autograd.grad(expected,z)[0]
                assert torch.allclose(actual,expected,atol=2e-6) and torch.allclose(ga,gb,atol=2e-7)
                for seed in [28,29]:samples.append(dict(id=row['id'],arm=mode+str(seed),x=z.detach().tolist(),score_mode=mode,sample=True,seed=seed))
            outputs=worker(cmd='roll',items=samples,keep_decisions=True)
            for item,roll in zip(samples,outputs):
                z=torch.tensor(item['x']);lp,_=distributions(z,roll['decisions'],item['score_mode'])
                labels=torch.tensor([d['selected'] for d in roll['decisions']]);selected=lp.gather(1,labels[:,None]).squeeze(1)
                err=float((selected-torch.tensor([d['selected_log_probability'] for d in roll['decisions']])).abs().max());assert err<5e-6
            zeros=[[0.] for _ in range(n*k)]
            items=[dict(id=row['id'],arm='old',x=[[0.]*4 for _ in zeros]),dict(id=row['id'],arm='eft',x=zeros,score_mode='direct_eft'),dict(id=row['id'],arm='direct',x=zeros,score_mode='direct')]
            rolls=worker(cmd='roll',items=items,keep_decisions=True,keep_schedule=True)
            assert rolls[0]['schedule']==rolls[1]['schedule']
            assert all(d['action']==min(d['available']) for d in rolls[2]['decisions'])
            mapping=[None]*n
            for d in row['decisions']:mapping[d['action']//k]=d['action']%k
            forced=[dict(id=row['id'],arm=mode,x=zeros,score_mode=mode,forced_mapping=mapping) for mode in ['direct','direct_eft']]
            forced.append(dict(id=row['id'],arm='old',x=[[0.]*4 for _ in zeros],forced_mapping=mapping))
            replay=worker(cmd='roll',items=forced,keep_schedule=True)
            assert replay[0]['schedule']==replay[1]['schedule']==replay[2]['schedule'] and replay[0]['C']==row['teacher']
            checks.append(dict(n=n,k=k,initial_encoder_exact=True,zero_EFT_HEFT_identity=True,direct_ties_pool_index=True,
                teacher_mapping_timing_cost_exact=True,independent_CE_and_gradient=True,probability_prefix_checks=len(outputs)))
        # Full old-VAL regression of the unchanged default coefficient path.
        source=ROOT/'runs/scratch_both_seven_epoch_20261028/step5500.pt'
        original=json.loads((source.parent/'eval_5500.json').read_text())['VAL']['scratch_both'];expected={r['id']:r['C'] for r in original}
        old=StateActor('both');old.load_state_dict(torch.load(source,weights_only=False)['model']);old.eval();count=0
        with torch.no_grad():
            for offset in range(0,862,16):
                rows=worker(cmd='val',offset=offset,count=16)
                items=[dict(id=r['id'],arm='historical',x=old(tensors(r['graph'],'cpu')).tolist()) for r in rows]
                for roll in worker(cmd='roll',items=items):assert roll['C']==expected[roll['id']];count+=1
        assert count==862
        data=ROOT/'data/structured_scheduling_v1';manifest=data/'MANIFEST.json'
        assert hashlib.sha256(manifest.read_bytes()).hexdigest()==identity['manifest']
        checked=[]
        for shard in json.loads(manifest.read_text())['shards']:
            if shard['split'] not in ['train','val']:continue
            path=data/shard['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==shard['sha256'];checked.append(shard)
        result=dict(status='PASS',identity=identity,shapes=checks,historical_full_VAL_exact=count,
            checked_train_val_shards=checked,TEST=False,parameters=sum(p.numel() for p in DirectActor('direct').parameters()),
            sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'daglab/direct_score_torch.py',ROOT/'daglab/state_score.py',ROOT/'daglab/module_lab/models.py']})
        (OUT/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS direct score gates:12shapes/teacher/probabilities/gradients/862historicalVAL',flush=True)
    except BaseException:
        import traceback
        (OUT/'preflight_failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc()),indent=2));raise
    finally:worker.close()


if __name__=='__main__':main()
