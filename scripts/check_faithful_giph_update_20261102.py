#!/usr/bin/env python3
"""Compare our online episode update against the exact author finish_episode."""
import copy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from daglab.faithful_external import *
import daglab.faithful_external as faithful
from daglab.faithful_external_data import Inputs
from daglab.sft_three_panel import write_json


def main():
    torch.set_num_threads(1);torch.manual_seed(1203);inputs=Inputs()
    instance=inputs.instance(inputs.keys[0]);model=GiPH();optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    initial=copy.deepcopy(model.state_dict());opt=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state()
    def trajectory():
        p=GiPHProblem(instance);mapping=p.random_mapping(91);states=[];logs=[]
        original_state=p.state
        def capture_state(*args,**kwargs):
            state=original_state(*args,**kwargs);states.append(state['C']);return state
        p.state=capture_state
        original_log_prob=torch.distributions.Categorical.log_prob
        def capture_lp(distribution,action):
            lp=original_log_prob(distribution,action);logs.append(lp);return lp
        torch.distributions.Categorical.log_prob=capture_lp
        try:best,loss,info=giph_episode(model,p,mapping,train=True)
        finally:torch.distributions.Categorical.log_prob=original_log_prob
        rewards=[float(a-b) for a,b in zip(states,states[1:])]
        assert len(logs)==len(rewards)==info['moves']
        return best,loss,info,logs,rewards
    optimizer.zero_grad();best,loss,info,logs,rewards=trajectory();loss.backward();optimizer.step()
    expected=copy.deepcopy(model.state_dict())
    model.load_state_dict(initial);optimizer.load_state_dict(opt);torch.set_rng_state(rng)
    repeated,_,again,logs,rewards=trajectory();assert repeated==best and again==info
    author=faithful._agent['PlacementAgent'].__new__(faithful._agent['PlacementAgent'])
    author.device=torch.device('cpu');author.gamma=.97;author.log_probs=logs;author.saved_rewards=rewards;author.optim=optimizer
    author.finish_episode(update_network=True,use_baseline=True)
    maximum=max(float((expected[name]-value).abs().max()) for name,value in model.state_dict().items())
    assert all(torch.allclose(expected[name],value,atol=3e-7,rtol=3e-6) for name,value in model.state_dict().items())
    write_json(RUN/'engineering/GIPH_UPDATE_PARITY.json',dict(status='PASS',
        actual_author_finish_episode_executed=True,full_moves=info['moves'],max_parameter_difference=maximum,
        original_Adam_and_prefix_reward_baseline=True,no_formal_state_used=True))
    print('AUTHOR GIPH UPDATE PARITY PASS',maximum)


if __name__=='__main__':main()
