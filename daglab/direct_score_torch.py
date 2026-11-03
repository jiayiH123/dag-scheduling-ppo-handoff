"""One encoder call, one unbounded logit per task-pool, optional fixed EFT."""
import math
import torch
from torch import nn
from .module_lab.models import ModuleActor

MODES=('direct','direct_eft')


class DirectActor(ModuleActor):
    def __init__(self,mode):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('edge_attention',32,2)
        self.mode=mode
        self.head[-1]=nn.Linear(32,1)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def output_transform(self,raw):return raw


def distributions(x,decisions,mode,temperature=1.):
    if mode not in MODES:raise ValueError(mode)
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('temperature')
    width=max(len(d['available']) for d in decisions)
    ids=torch.tensor([d['available']+[0]*(width-len(d['available'])) for d in decisions],device=x.device)
    mask=torch.tensor([[True]*len(d['available'])+[False]*(width-len(d['available'])) for d in decisions],device=x.device)
    score=x[ids,0]
    if mode=='direct_eft':
        score=score+x.new_tensor([d['offsets']+[0.]*(width-len(d['available'])) for d in decisions])
    return torch.log_softmax((score/temperature).masked_fill(~mask,-torch.inf),-1),mask


def bc_loss(x,decisions,mode):
    logp,_=distributions(x,decisions,mode)
    labels=torch.tensor([d['selected'] for d in decisions],device=x.device)
    return -logp.gather(1,labels[:,None]).mean()
