"""Single graph forward emits intercept and analytic budget-state coefficient."""
import torch
from torch import nn
from single_forward_actor import SingleForwardActor

class BudgetFeedbackActor(SingleForwardActor):
    def __init__(self,hidden=32):
        super().__init__("joint",hidden)
        self.budget_head=nn.Sequential(nn.Linear(2*hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        with torch.no_grad():self.budget_head[-1].weight.zero_();self.budget_head[-1].bias.zero_()

    def forward(self,graph):
        self.forward_calls+=1
        x=torch.tensor(graph["x"],dtype=torch.float32);eligible=torch.tensor(graph["eligible"],dtype=torch.bool)
        src=torch.tensor(graph["src"],dtype=torch.long);dst=torch.tensor(graph["dst"],dtype=torch.long)
        weights=torch.tensor(graph["edge_weight"],dtype=torch.float32).unsqueeze(1);h=torch.tanh(self.input(x));count=h.shape[0]
        indegree=torch.zeros(count,1).index_add(0,dst,torch.ones_like(weights)).clamp_min(1)
        outdegree=torch.zeros(count,1).index_add(0,src,torch.ones_like(weights)).clamp_min(1)
        for fw,bw,update in zip(self.fwd,self.bwd,self.update):
            f=torch.zeros_like(h).index_add(0,dst,fw(h[src])*weights)/indegree
            b=torch.zeros_like(h).index_add(0,src,bw(h[dst])*weights)/outdegree
            h=torch.tanh(update(torch.cat([h,f,b],dim=-1)))
        context=h[eligible].mean(0,keepdim=True);joint=torch.cat([h,context.expand(count,-1)],dim=-1)
        intercept=8*torch.tanh(self.head(joint).squeeze(-1)/8)
        coefficient=8*torch.tanh(self.budget_head(joint).squeeze(-1)/8)
        return torch.stack([intercept,coefficient],dim=-1)
