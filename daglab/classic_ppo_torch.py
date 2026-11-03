"""Classic clipped actor surrogate, learned state value and GAE, without BC loss."""
import torch
from torch import nn

OBS_DIM=76


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(OBS_DIM,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self,states):
        return self.net(states).squeeze(-1)


def gae(rewards,values,gamma=1.,lam=.95):
    """One complete, unterminated-by-time-limit episode; terminal bootstrap is0."""
    assert rewards.ndim==values.ndim==1 and len(rewards)==len(values)
    advantage=torch.zeros_like(rewards)
    carry=rewards.new_zeros(())
    next_value=rewards.new_zeros(())
    for t in range(len(rewards)-1,-1,-1):
        delta=rewards[t]+gamma*next_value-values[t]
        carry=delta+gamma*lam*carry
        advantage[t]=carry
        next_value=values[t]
    return advantage.detach(),(advantage+values).detach()


def clipped_loss(new_logprob,old_logprob,advantage,clip=.2):
    ratio=(new_logprob-old_logprob).exp()
    loss=-torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).mean()
    approx_kl=((ratio-1)-(new_logprob-old_logprob)).mean()
    return loss,approx_kl,((ratio-1).abs()>clip).float().mean()
