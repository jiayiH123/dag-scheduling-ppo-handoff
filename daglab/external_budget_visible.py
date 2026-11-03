"""Preserve global physical parameters after local feature normalization."""
import torch
from torch import nn
from .external_baseline_sim import Problem
from .external_baseline_models import CAIActor,GiPHActor
from .external_graphormer import CAIGraphormer


class BudgetProblem(Problem):
    def priority_features(self):
        g=super().priority_features()
        minimum=sum(self.info['min_cost'])
        fastest=[min(self.info['eligible'][v],key=lambda p:(self.info['duration'][v][p],self.info['costs'][v][p],p)) for v in range(self.n)]
        fastcost=sum(self.info['costs'][v][p] for v,p in enumerate(fastest))
        eta=(self.budget-minimum)/max(1,fastcost-minimum)
        for x in g['x']:x[8]=eta;x[9]=self.k/6
        return g
    def relocation_graph(self,mapping,state,last_task=None):
        g=super().relocation_graph(mapping,state,last_task)
        slack=(self.budget-state['cost'])/max(1,self.budget)
        for x in g['x']:x.append(slack)
        return g


class BudgetCAIActor(CAIActor):
    def __init__(self):
        super().__init__()
        with torch.no_grad():
            for direction in self.directions:direction[0].linear.weight[:,8:10].zero_()


class BudgetGraphormer(CAIGraphormer):
    def __init__(self):
        super().__init__()
        with torch.no_grad():self.input.weight[:,8:10].zero_()


class BudgetGiPHActor(GiPHActor):
    def __init__(self):
        super().__init__();rng=torch.get_rng_state();old=self.node[0];new=nn.Linear(10,9)
        with torch.no_grad():
            new.weight[:,:9].copy_(old.weight);new.weight[:,9].zero_();new.bias.copy_(old.bias)
        self.node[0]=new;torch.set_rng_state(rng)
