#!/usr/bin/env python3
"""Exact optimizer/RNG restoration for the actual CAI paper-network update."""
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from daglab.faithful_external import *
from daglab.faithful_external_data import Inputs
from daglab.sft_three_panel import write_json


def main():
    torch.set_num_threads(1); torch.manual_seed(1202)
    inputs=Inputs(); p=Problem(inputs.instance(inputs.keys[0])); data=cai_input(p)
    model=CAI(data['numeric'].shape[-1]); optimizer=torch.optim.Adam(model.parameters(),lr=5e-4)
    initial=copy.deepcopy(model.state_dict()); opt=copy.deepcopy(optimizer.state_dict()); rng=torch.get_rng_state()
    def update():
        optimizer.zero_grad(); baseline=p.schedule(); z,raw=model(data); items=[]
        for _ in range(4):
            result,lp=model.sample(p,data,z,True)
            reward=torch.tensor([baseline['ends'][v]-result['ends'][v] for v in range(p.n)],dtype=torch.float32)
            items.append((lp,reward))
        loss=cai_loss(items,[raw.square().mean()]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),.5); optimizer.step()
        return loss.detach().clone()
    loss=update(); expected=copy.deepcopy(model.state_dict()); expected_opt=copy.deepcopy(optimizer.state_dict())
    model.load_state_dict(initial); optimizer.load_state_dict(opt); torch.set_rng_state(rng)
    repeated=update()
    assert torch.equal(loss,repeated)
    assert all(torch.equal(expected[name],value) for name,value in model.state_dict().items())
    for key,value in optimizer.state_dict()['state'].items():
        assert all(torch.equal(v,expected_opt['state'][key][name]) for name,v in value.items())
    write_json(RUN/'engineering/CAI_RESUME.json',dict(status='PASS',weights_bitwise=True,optimizer_bitwise=True,
        torch_RNG_replay=True,teacher_labels=False,formal_updates=False))
    print('CAI EXACT RESUME PASS',flush=True)


if __name__=='__main__':main()
