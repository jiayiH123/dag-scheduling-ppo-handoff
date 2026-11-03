"""One-forward static position residual plus coefficients for prefix budget features."""
import math
import torch
from torch import nn
from .state_score_torch import StateActor as Trunk
from .module_lab.models import tensors as base_tensors
MODES=['original','position','budget','both','capacity']
def tensors(graph,device):
    g=base_tensors(graph,device);g['heft_position']=torch.tensor(graph['heft_position'],device=device,dtype=torch.float32)
    return g
class BudgetActor(nn.Module):
    def __init__(self,mode):
        super().__init__();assert mode in MODES;self.mode=mode;self.forward_calls=0
        self.trunk=Trunk('both')
        self.position=nn.Sequential(nn.Linear(65,32),nn.Tanh(),nn.Linear(32,4))
        self.budget=nn.Sequential(nn.Linear(64,32),nn.Tanh(),nn.Linear(32,2))
        for head in [self.position,self.budget]:nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
    def forward(self,g):
        self.forward_calls+=1;x=self.trunk(g);context=self.trunk.context
        pos=g['heft_position'] if self.mode in ['position','both'] else torch.full_like(g['heft_position'],.5)
        residual=2*torch.tanh(self.position(torch.cat([context,pos[:,None]],-1))/2)
        x=x+residual*float(self.mode in ['position','both','capacity'])
        y=2*torch.tanh(self.budget(context)/2)
        return torch.cat([x,y*float(self.mode in ['budget','both']),y*float(self.mode=='capacity')],-1)
def distributions(x,ds,temperature=1.):
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('temperature')
    width=max(len(d['available']) for d in ds);device=x.device
    ids=torch.tensor([d['available']+[0]*(width-len(d['available'])) for d in ds],device=device)
    mask=torch.tensor([[True]*len(d['available'])+[False]*(width-len(d['available'])) for d in ds],device=device)
    def f(name):return x.new_tensor([d[name]+[0.]*(width-len(d['available'])) for d in ds])
    p=x[ids];phi=f('budget_local')
    score=p[...,0]+p[...,1]*phi+f('offsets')+p[...,2]*f('gap')+p[...,3]*f('wait')+p[...,4]*f('pressure')+p[...,5]*f('coverage')+p[...,6]*phi+p[...,7]*(1-phi)
    return torch.log_softmax((score/temperature).masked_fill(~mask,-torch.inf),-1),mask
def bc_loss(x,ds):
    lp,_=distributions(x,ds);label=torch.tensor([d['selected'] for d in ds],device=x.device)
    return -lp.gather(1,label[:,None]).mean()
