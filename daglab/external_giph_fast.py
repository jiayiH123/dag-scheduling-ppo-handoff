"""Numerically checked equivalent GiPH graph construction and level edge selection."""
import numpy as np
import torch
from .external_budget_visible import BudgetProblem,BudgetGiPHActor
from .external_baseline_models import TopoMP
from torch.nn import functional as F


class FastBudgetProblem(BudgetProblem):
    def __init__(self,inst):
        super().__init__(inst)
        n,k=self.n,self.k
        self.eu=np.asarray([u for u,v in inst['edges']],dtype=int)
        self.ev=np.asarray([v for u,v in inst['edges']],dtype=int)
        self.edge_comm=np.asarray([self.info['comm'][u,v] for u,v in inst['edges']],dtype=float).reshape(-1,k,k)
        self.edge_bytes=np.asarray([inst['data_mb'][u][v] for u,v in inst['edges']],dtype=float)
        self.inv_bw=np.divide(1.,np.asarray(inst['bandwidth'],dtype=float),out=np.zeros((k,k)),where=~np.eye(k,dtype=bool))
        self.node_template=np.zeros((n,k,9),dtype=float)
        self.node_template[:,:,0]=np.asarray(inst['workload'])[:,None]
        self.node_template[:,:,1]=1/np.asarray(inst['speeds'])[None,:]
        self.node_template[:,:,2]=self.duration
        self.node_template[:,:,5]=self.costs
        self.node_template[:,:,6]=np.asarray(inst['demands'])[:,None]/np.asarray(inst['capacities'])[None,:]
        self.fast_levels=[[v*k+p for v in level for p in range(k)] for level in self.levels]
    def relocation_graph(self,mapping,state,last_task=None):
        n,k=self.n,self.k;info=self.info;m=np.asarray(mapping,dtype=int)
        mask=self.eligible & (state['cost']-self.costs[np.arange(n),m,None]+self.costs<=self.budget)
        mask[np.arange(n),m]=False
        if last_task is not None:
            excluded=mask.copy();excluded[last_task]=False
            if excluded.any():mask=excluded
        critical=np.zeros(n);v=max(range(n),key=lambda v:state['ends'][v])
        while True:
            critical[v]=1
            if not self.parents[v]:break
            v=max(self.parents[v],key=lambda u:state['ends'][u]+info['comm'][u,v][mapping[u]][mapping[v]])
        node=self.node_template.copy();node[:,:,4]=critical[:,None];node[np.arange(n),m,7]=1
        for v in range(n):
            release=np.zeros(k)
            for u in self.parents[v]:release=np.maximum(release,state['ends'][u]+np.asarray(info['comm'][u,v][mapping[u]]))
            node[v,:,3]=release-state['starts'][v];node[v,m[v],3]=0
        node[:,:,8]=(self.budget-state['cost'])/np.maximum(1,self.costs)
        q=np.broadcast_to(np.arange(k),(len(self.eu),k));mu=m[self.eu];mv=m[self.ev]
        others=q[q!=mu[:,None]].reshape(-1,k-1)
        pi=np.concatenate([np.broadcast_to(mu[:,None],q.shape),others],axis=1)
        qi=np.concatenate([q,np.broadcast_to(mv[:,None],others.shape)],axis=1)
        src=self.eu[:,None]*k+pi;dst=self.ev[:,None]*k+qi
        edge=np.zeros((len(self.eu),2*k-1,5),dtype=float)
        edge[:,:,0]=self.edge_bytes[:,None]
        edge[:,:,2]=self.edge_comm[np.arange(len(self.eu))[:,None],pi,qi]
        edge[:,:,3]=self.inv_bw[pi,qi];edge[:,:,4]=(critical[self.eu]*critical[self.ev])[:,None]
        eligible=self.eligible[self.eu[:,None],pi] & self.eligible[self.ev[:,None],qi]
        def normalize(a,width):
            a=a.reshape(-1,width)
            if not len(a):return a
            low=a.min(0);span=a.max(0)-low
            return np.where(span>0,2*(a-low)/np.maximum(span,1e-12)-1,0.)
        x=normalize(node,9);x=np.column_stack([x,np.full(len(x),(self.budget-state['cost'])/max(1,self.budget))])
        return dict(x=x.tolist(),src=src[eligible].tolist(),dst=dst[eligible].tolist(),edge=normalize(edge[eligible],5).tolist(),
            levels=self.fast_levels,mask=mask.flatten().tolist(),n=n,k=k)


class FastTopoMP(TopoMP):
    def forward(self,initial,src,dst,edge,levels):
        h=initial;level_of=[0]*len(h)
        for i,ids in enumerate(levels):
            for v in ids:level_of[v]=i
        edge_level=torch.tensor(level_of,dtype=torch.long)[dst]
        for i,ids in enumerate(levels):
            nodes=torch.tensor(ids,dtype=torch.long);take=edge_level==i;a,b=src[take],dst[take]
            msgs=F.relu(self.pre(torch.cat([h[a],edge[take]],-1)))
            aggregate=msgs.new_zeros((len(h),msgs.shape[1])).index_add(0,b,msgs)
            degree=msgs.new_zeros(len(h)).index_add(0,b,torch.ones(len(b)))
            value=initial[nodes]+F.relu(self.update(aggregate[nodes]/degree[nodes,None].clamp_min(1)))
            h=h.index_copy(0,nodes,value)
        return h


class FastBudgetGiPHActor(BudgetGiPHActor):
    def __init__(self):
        super().__init__();rng=torch.get_rng_state()
        for name in ['fmp','bmp']:
            old=getattr(self,name);new=FastTopoMP();new.load_state_dict(old.state_dict());setattr(self,name,new)
        torch.set_rng_state(rng)
