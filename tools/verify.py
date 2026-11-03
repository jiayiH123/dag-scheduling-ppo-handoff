#!/usr/bin/env python3
"""Physical/feature/label parity, full retained SFT replay and PPO resume gate."""
import copy,gzip,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main():
    import torch
    from daglab.portable_runtime import Data,evaluation
    from daglab.state_score_torch import StateActor
    torch.set_num_threads(1)
    checks=[];expected=json.loads((ROOT/'evidence/EXPECTED_FEATURES_AND_TRACES.json').read_text())
    for variant in ['legacy','strong','no_search']:
        data=Data(variant)
        try:
            for row in expected:
                _,payload=data.record(row['key'])
                assert digest(payload['graph'])==row['graph'],('graph',variant,row['key'])
                assert digest(payload['decisions'])==row['traces'][variant],('trace',variant,row['key'])
                checks.append(dict(variant=variant,n=row['n'],k=row['k'],graph_and_trace_exact=True))
        finally:data.close()
    cfg=json.loads((ROOT/'configs/teacher_search_ablation_ppo_20261102.json').read_text());validation={}
    for arm,path in [('strong',cfg['strong_source']),('no_search',cfg['no_search_SFT_run']+'/best.pt')]:
        saved=torch.load(ROOT/path,map_location='cpu',weights_only=False)
        model=StateActor('both');model.load_state_dict(saved['model']);assert sum(p.numel() for p in model.parameters())==32612
        with gzip.open(ROOT/path.rsplit('/',1)[0]/f'val_synthetic_val862_{saved["step"]:05d}.json.gz','rt') as stream:
            original={r['key']:(r['C'],r['cost']) for r in json.load(stream)['rows']}
        result=evaluation(model)
        assert len(result['rows'])==len(original)==862
        assert all((r['C'],r['cost'])==original[r['key']] for r in result['rows'])
        validation[arm]=dict(step=saved['step'],exact_C_and_cost_cases=862,summary=result['summary'])
        print('SFT_FULL_VAL_PARITY',arm,saved['step'],result['summary']['C_over']['HEFT'],flush=True)
    # Use the actual PPO collection implementation, without starting a full run.
    import scripts.train_teacher_search_ppo_20261102 as module
    out=ROOT/'evidence/portable_smoke';out.mkdir(exist_ok=True)
    p=module.PPO('strong',41,out_override=out/'uninterrupted')
    q=None
    try:
        p.update_collection(1,observe=False)
        p.update_collection(2,observe=False)
        expected_actor=copy.deepcopy(p.actor.state_dict());expected_critic=copy.deepcopy(p.critic.state_dict())
        q=module.PPO('strong',41,out_override=out/'resumed')
        shutil.copy2(p.out/'resume.pt',q.out/'resume.pt');q.restore()
        assert q.collection==2
        p.update_collection(3,observe=False);q.update_collection(3,observe=False)
        assert all(torch.equal(v,q.actor.state_dict()[k]) for k,v in p.actor.state_dict().items())
        assert all(torch.equal(v,q.critic.state_dict()[k]) for k,v in p.critic.state_dict().items())
        ppo=dict(actual_collections=3,actor_updates=p.actor_updates,critic_updates=p.critic_updates,
                 exact_actor_critic_resume=True,own_prefix_log_probability_checks=True,
                 engineering_only_no_critic_warmup_or_VAL_selection=True)
    finally:
        p.close()
        if q:q.close()
    report=dict(status='PASS',feature_and_teacher_trace_cases=checks,retained_SFT_full_VAL_parity=validation,PPO=ppo,
        packed_manifest_sha256=hashlib.sha256((ROOT/'data/MANIFEST.json').read_bytes()).hexdigest())
    (ROOT/'evidence/PORTABLE_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: 36 feature/trace cases, both full862 retained SFT replays, physical PPO and exact resume.',flush=True)

if __name__=='__main__':main()
