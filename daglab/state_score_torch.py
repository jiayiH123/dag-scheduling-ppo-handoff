"""One full encoder/head call; signed zero-initialized physical-state coefficients."""
import math
import torch
from torch import nn
from .module_lab.models import ModuleActor

MODES={'original':(0.,0.),'gap':(1.,0.),'wait':(0.,1.),'both':(1.,1.)}

class StateActor(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode;self.forward_calls=0
        self.base=ModuleActor('edge_attention',32,2)
        self.extra=nn.Sequential(nn.Linear(64,32),nn.Tanh(),nn.Linear(32,2))
        nn.init.zeros_(self.extra[-1].weight);nn.init.zeros_(self.extra[-1].bias)
        self.base.head.register_forward_pre_hook(self.capture)
        self.context=None
    def capture(self,module,args):self.context=args[0]
    def forward(self,g):
        self.forward_calls+=1;x=self.base(g)
        y=2*torch.tanh(self.extra(self.context)/2)
        mask=y.new_tensor(MODES[self.mode])
        return torch.cat([x,y*mask],-1)

def distributions(x,decisions,temperature=1.):
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('temperature must be finite and positive')
    width=max(len(d['available']) for d in decisions);device=x.device
    ids=torch.tensor([d['available']+[0]*(width-len(d['available'])) for d in decisions],device=device)
    mask=torch.tensor([[True]*len(d['available'])+[False]*(width-len(d['available'])) for d in decisions],device=device)
    def feature(name):return x.new_tensor([d[name]+[0.]*(width-len(d['available'])) for d in decisions])
    pairs=x[ids]
    scores=pairs[...,0]+pairs[...,1]*feature('budget_local')+feature('offsets')+pairs[...,2]*feature('gap')+pairs[...,3]*feature('wait')
    return torch.log_softmax((scores/temperature).masked_fill(~mask,-torch.inf),-1),mask

def bc_loss(x,ds):
    logs,_=distributions(x,ds);labels=torch.tensor([d['selected'] for d in ds],device=x.device)
    return -logs.gather(1,labels[:,None]).mean()
