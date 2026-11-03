"""Physical state scoring; NumPy environment, no modification of frozen decoder."""
import math
import time
import numpy as np
from .reference import core
from .module_lab.decoder import NormalizedEFTDecoder
from prepared_sgs import find_slot


class StateScoreDecoder(NormalizedEFTDecoder):
    def rollout(self, coefficients, forced_mapping=None, sample=False, seed=0, observer=None, temperature=1., score_mode='coefficients'):
        if len(coefficients)!=self.n*self.k:raise ValueError('coefficient shape')
        if score_mode not in ['coefficients','direct','direct_eft']:raise ValueError('score mode')
        expected=4 if score_mode=='coefficients' else 1
        if any(len(row)!=expected for row in coefficients):raise ValueError('score output width')
        if not math.isfinite(temperature) or temperature<=0:raise ValueError('temperature must be finite and positive')
        started=time.perf_counter();rng=np.random.default_rng(seed)
        calendars=[[] for _ in range(self.k)]
        mapping=[-1]*self.n;starts=[0]*self.n;ends=[0]*self.n;ready=[0]*self.n
        spent=0;reserve=sum(self.info['min_cost']);decisions=[];tau=max(1.,self.scale)
        for v in self.order:
            reserve-=self.info['min_cost'][v];choices=[]
            den=max(1,max(self.info['costs'][v][p]-self.info['min_cost'][v] for p in self.info['eligible'][v]))
            for p in self.info['eligible'][v]:
                if spent+self.info['costs'][v][p]+reserve>self.inst['budget']:continue
                release=max((ends[u]+self.info['comm'][u,v][mapping[u]][p] for u in self.info['parents'][v]),default=0)
                duration=self.info['duration'][v][p]
                start=find_slot(release,duration,self.inst['demands'][v],self.inst['capacities'][p],calendars[p])
                phi=(self.info['costs'][v][p]-self.info['min_cost'][v])/den
                choices.append((v*self.k+p,p,start,start+duration,release,phi))
            old=[-c[3]/tau for c in choices];upper=max(old);lower=min(old)
            offsets=[(z-upper)/max(1.,upper-lower) for z in old]
            eftmin=min(c[3] for c in choices)
            gap=[min(4.,math.log1p((c[3]-eftmin)/tau)) for c in choices]
            wait=[min(4.,math.log1p((c[2]-c[4])/tau)) for c in choices]
            if score_mode=='coefficients':
                score=np.asarray([coefficients[c[0]][0]+coefficients[c[0]][1]*c[5]+off+
                    coefficients[c[0]][2]*g+coefficients[c[0]][3]*h
                    for c,off,g,h in zip(choices,offsets,gap,wait)])
            else:
                score=np.asarray([coefficients[c[0]][0]+(off if score_mode=='direct_eft' else 0.)
                    for c,off in zip(choices,offsets)])
            if not np.isfinite(score).all():raise ValueError('nonfinite score')
            logprob=(score-score.max())/temperature;logprob-=math.log(float(np.exp(logprob).sum()))
            # Optional training-only instrumentation; observes the prefix before
            # action selection and never changes candidate scores or sampling.
            observation=observer(v,choices,calendars,mapping,starts,ends,spent,reserve) if observer else None
            partial_before=max(ends) if observer else None
            if forced_mapping is not None:selected=next(i for i,c in enumerate(choices) if c[1]==forced_mapping[v])
            elif sample:selected=int(rng.choice(len(choices),p=np.exp(logprob)))
            elif score_mode=='direct':selected=min(range(len(choices)),key=lambda i:(-score[i],choices[i][1]))
            else:selected=min(range(len(choices)),key=lambda i:(-score[i],choices[i][3],choices[i][2],choices[i][1]))
            c=choices[selected];_,p,start,end,release,_=c
            mapping[v]=p;starts[v]=start;ends[v]=end;ready[v]=release
            calendars[p].append((start,end,self.inst['demands'][v]));spent+=self.info['costs'][v][p]
            decisions.append(dict(available=[x[0] for x in choices],offsets=offsets,
                budget_local=[x[5] for x in choices],gap=gap,wait=wait,
                selected=selected,action=c[0],selected_log_probability=float(logprob[selected])))
            if observer:
                decisions[-1].update(critic_state=observation,partial_before=partial_before,partial_after=max(ends))
        tasks=[dict(node=v,device=mapping[v],start=starts[v],end=ends[v],duration=self.info['duration'][v][mapping[v]],
            demand=self.inst['demands'][v],data_ready=ready[v],resource_wait=starts[v]-ready[v]) for v in range(self.n)]
        communications=[]
        for v in self.order:
            for u in sorted(self.info['parents'][v],key=lambda u:(ends[u],u)):
                p,q=mapping[u],mapping[v];duration=self.info['comm'][u,v][p][q]
                communications.append(dict(u=u,v=v,src=p,dst=q,start=ends[u],end=ends[u]+duration,
                    duration=duration,release=ends[u],queue_wait=0,local=p==q,data_mb=self.inst['data_mb'][u][v]))
        schedule=dict(algorithm='StateScore',makespan=max(ends),cost=spent,tasks=tasks,
            communications=communications,decision_order=list(self.order))
        core.validate(self.inst,schedule)
        return dict(decisions=decisions,makespan=max(ends),cost=spent,schedule=schedule,
                    decoder_seconds=time.perf_counter()-started)
