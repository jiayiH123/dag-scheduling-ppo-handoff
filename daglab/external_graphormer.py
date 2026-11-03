"""CAI paper-family numeric Graphormer adaptation, not original Fairseq execution."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class GraphormerBlock(nn.Module):
    def __init__(self):
        super().__init__();self.qkv=nn.Linear(64,192);self.out=nn.Linear(64,64)
        self.ff1=nn.Linear(64,64);self.ff2=nn.Linear(64,64)
        self.norm1=nn.LayerNorm(64);self.norm2=nn.LayerNorm(64);self.drop=nn.Dropout(.1)
    def forward(self,x,bias):
        n=len(x);q,k,v=self.qkv(x).reshape(n,3,4,16).permute(1,2,0,3).unbind(0)
        logits=q@k.transpose(-1,-2)/4+bias
        attention=F.dropout(torch.softmax(logits,-1),.1,training=self.training)
        y=(attention@v).transpose(0,1).reshape(n,64)
        x=self.norm1(x+self.drop(self.out(y)))
        return self.norm2(x+self.drop(self.ff2(self.drop(F.gelu(self.ff1(x))))))


class CAIGraphormer(nn.Module):
    def __init__(self):
        super().__init__();self.forward_calls=0
        self.input=nn.Linear(10,64);self.indegree=nn.Embedding(513,64);self.outdegree=nn.Embedding(513,64)
        self.distance=nn.Embedding(12,4);self.global_token=nn.Parameter(torch.zeros(1,64))
        self.global_bias=nn.Parameter(torch.zeros(4));self.blocks=nn.ModuleList([GraphormerBlock() for _ in range(4)])
        self.head=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,g):
        self.forward_calls+=1;n=g['n'];parents=[[] for _ in range(n)];children=[[] for _ in range(n)]
        for u,v in g['edges']:parents[v].append(u);children[u].append(v)
        indegree=[len(p) for p in parents];outdegree=[len(c) for c in children]
        pending=indegree.copy();ready=[v for v in range(n) if pending[v]==0];order=[]
        while ready:
            v=ready.pop();order.append(v)
            for c in children[v]:
                pending[c]-=1
                if pending[c]==0:ready.append(c)
        assert len(order)==n
        dist=np.full((n,n),10000,dtype=np.int64);np.fill_diagonal(dist,0)
        for v in order:
            for u in parents[v]:dist[:,v]=np.minimum(dist[:,v],dist[:,u]+1)
        dist=np.where(dist>=10000,11,np.minimum(dist,10))
        b=self.distance(torch.tensor(dist,dtype=torch.long)).permute(2,0,1)
        bias=self.global_bias[:,None,None].expand(4,n+1,n+1).clone();bias[:,1:,1:]=b
        x=self.input(torch.tensor(g['x'],dtype=torch.float32))
        x=x+self.indegree(torch.tensor(indegree).clamp_max(512))+self.outdegree(torch.tensor(outdegree).clamp_max(512))
        x=torch.cat([self.global_token,x],0)
        for block in self.blocks:x=block(x,bias)
        raw=self.head(x[1:]).flatten();logits=5*(raw-raw.mean())/(raw.std()+1e-20)
        return logits,raw
