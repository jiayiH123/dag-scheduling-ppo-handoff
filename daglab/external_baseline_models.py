"""Portable operator reimplementations, not untouched author packages.

DAG-CAI-DR: Qi/Zhang/Liu/Wang, NeurIPS2025, authorCC-BY-NC4 source.
GiPH: Hu etal., MLSys2023, authorMIT source. See vendor/external_baseline_sources_20261028.
"""
import torch
from torch import nn
from torch.nn import functional as F


class GATLayer(nn.Module):
    def __init__(self,din,dout):
        super().__init__();self.linear=nn.Linear(din,dout,bias=False)
        self.att_src=nn.Parameter(torch.empty(dout));self.att_dst=nn.Parameter(torch.empty(dout));self.bias=nn.Parameter(torch.zeros(dout))
        nn.init.xavier_uniform_(self.linear.weight);nn.init.xavier_uniform_(self.att_src[None]);nn.init.xavier_uniform_(self.att_dst[None])
    def forward(self,x,src,dst):
        h=self.linear(x);z=F.leaky_relu((h[src]*self.att_src).sum(-1)+(h[dst]*self.att_dst).sum(-1),.2)
        mx=z.new_full((len(x),),-torch.inf).scatter_reduce(0,dst,z,reduce='amax',include_self=True)
        ex=torch.exp(z-mx[dst]);den=ex.new_zeros(len(x)).index_add(0,dst,ex)
        out=h.new_zeros(h.shape).index_add(0,dst,h[src]*(ex/den[dst])[:,None])
        return out+self.bias


class CAIActor(nn.Module):
    def __init__(self):
        super().__init__();self.forward_calls=0
        self.directions=nn.ModuleList([nn.ModuleList([GATLayer(10,8),GATLayer(8,8)]) for _ in range(2)])
        self.head=nn.Sequential(nn.Linear(16,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,g):
        self.forward_calls+=1;x=torch.tensor(g['x'],dtype=torch.float32);n=len(x)
        edges=torch.tensor(g['edges'],dtype=torch.long).reshape(-1,2)
        src=torch.cat([edges[:,0],torch.arange(n)]);dst=torch.cat([edges[:,1],torch.arange(n)])
        hs=[]
        for i,layers in enumerate(self.directions):
            a,b=(src,dst) if i==0 else (dst,src)
            hs.append(layers[1](F.relu(layers[0](x,a,b)),a,b))
        raw=self.head(torch.cat(hs,-1)).flatten()
        logits=5*(raw-raw.mean())/(raw.std()+1e-20)
        return logits,raw


class TopoMP(nn.Module):
    def __init__(self,h=16,e=5):
        super().__init__();self.pre=nn.Linear(h+e,h+e);self.update=nn.Linear(h+e,h)
    def forward(self,initial,src,dst,edge,levels):
        h=initial
        for ids in levels:
            nodes=torch.tensor(ids,dtype=torch.long)
            # All incoming parents have been updated; all task-device nodes retained.
            take=torch.isin(dst,nodes)
            a,b=src[take],dst[take]
            msgs=F.relu(self.pre(torch.cat([h[a],edge[take]],-1)))
            aggregate=msgs.new_zeros((len(h),msgs.shape[1])).index_add(0,b,msgs)
            degree=msgs.new_zeros(len(h)).index_add(0,b,torch.ones(len(b)))
            value=initial[nodes]+F.relu(self.update(aggregate[nodes]/degree[nodes,None].clamp_min(1)))
            h=h.index_copy(0,nodes,value)
        return h


class GiPHActor(nn.Module):
    def __init__(self):
        super().__init__();self.forward_calls=0
        self.node=nn.Sequential(nn.Linear(9,9),nn.ReLU(),nn.Linear(9,16))
        self.fmp=TopoMP();self.bmp=TopoMP();self.head=nn.Sequential(nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,g):
        self.forward_calls+=1
        x=torch.tensor(g['x'],dtype=torch.float32);src=torch.tensor(g['src'],dtype=torch.long);dst=torch.tensor(g['dst'],dtype=torch.long)
        edge=torch.tensor(g['edge'],dtype=torch.float32).reshape(-1,5);initial=self.node(x)
        hf=self.fmp(initial,src,dst,edge,g['levels']);hb=self.bmp(initial,dst,src,edge,list(reversed(g['levels'])))
        logits=self.head(torch.cat([hf,hb],-1)).flatten();mask=torch.tensor(g['mask'],dtype=torch.bool)
        return logits.masked_fill(~mask,-torch.inf)


def merge_relocation_graphs(graphs):
    """Disjoint union; no cross-instance messages or policy normalization."""
    out=dict(x=[],src=[],dst=[],edge=[],levels=[],mask=[]);sizes=[]
    for g in graphs:
        offset=len(out['x']);sizes.append((offset,offset+len(g['x'])))
        out['x'].extend(g['x']);out['edge'].extend(g['edge']);out['mask'].extend(g['mask'])
        out['src'].extend(v+offset for v in g['src']);out['dst'].extend(v+offset for v in g['dst'])
        while len(out['levels'])<len(g['levels']):out['levels'].append([])
        for i,ids in enumerate(g['levels']):out['levels'][i].extend(v+offset for v in ids)
    return out,sizes
