"""Simulator-only, pre-action observations for a training-only value network."""
import math
from .state_score import StateScoreDecoder

OBS_DIM=76  # 16 global features, 10 per pool, padded to six pools.


class InstrumentedDecoder(StateScoreDecoder):
    def rollout(self, coefficients, **kwargs):
        assert self.k<=6
        self.observed_steps=0
        return super().rollout(coefficients,observer=self.observe,**kwargs)

    def observe(self,v,choices,calendars,mapping,starts,ends,spent,reserve):
        tau=max(1.,self.scale)
        logtime=lambda t: math.log1p(max(0.,t)/tau)
        budget=max(1,self.inst['budget'])
        caps=self.inst['capacities']
        feasible={c[1]:c for c in choices}
        step=self.observed_steps
        self.observed_steps+=1
        obs=[self.n/120,self.k/6,step/self.n,1-step/self.n,math.log1p(tau)/16,
             (budget-spent)/budget,reserve/budget,(budget-spent-reserve)/budget,
             logtime(max(ends)),logtime(min(c[3] for c in choices)),
             logtime(max(c[3] for c in choices)),logtime(sum(c[3] for c in choices)/len(choices)),
             logtime(self.info['ranks'][v]),self.inst['demands'][v]/max(caps),
             len(self.info['parents'][v])/self.n,len(choices)/self.k]
        for p in range(6):
            if p>=self.k:
                obs.extend([0.]*10)
                continue
            c=feasible.get(p)
            busy=sum((e-s)*d for s,e,d in calendars[p])/(max(1,caps[p])*tau*self.n)
            parent_fraction=sum(mapping[u]==p for u in self.info['parents'][v])/max(1,len(self.info['parents'][v]))
            obs.extend([float(p in self.info['eligible'][v]),float(c is not None),
                        logtime(self.info['duration'][v][p]),logtime(c[3]) if c else 0.,
                        logtime(c[2]-c[4]) if c else 0.,logtime(c[4]) if c else 0.,
                        logtime(max((e for s,e,d in calendars[p]),default=0)),
                        busy,self.info['costs'][v][p]/budget,parent_fraction])
        assert len(obs)==OBS_DIM and all(math.isfinite(x) for x in obs)
        return obs
