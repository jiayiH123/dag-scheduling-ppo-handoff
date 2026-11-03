# Isolated copy of our frozen CachedBudgetDecoder; vendor is unchanged.
# Only semantic change: per-decision relative EFT offsets are bounded to[-1,0].
# Original source SHA256: 2694e0d3561e3798e1a882ac7a66f3083f576b42eb2d985f4d258e0196d5e38b
"""Prepared one-pass analytic decoder; no neural calls or candidate search.

Initial and local arms share the exact same two-head actor and hard mask.
Local removes the global budget denominator from the explicit scoring term,
but retains budget information in graph features and in the feasibility mask.
"""
import math
import time
import numpy as np
import budget_core as core
from prepared_sgs import PreparedSGS, find_slot


class NormalizedEFTDecoder:
    def __init__(self, instance, order=None):
        self.engine = PreparedSGS(instance)
        self.inst, self.info = self.engine.instance, self.engine.info
        self.n, self.k = self.engine.n, self.engine.k
        if order is None:
            # The original HEFT rank is strictly descending along dependencies.
            order = sorted(range(self.n), key=lambda v: (-self.info['ranks'][v], v))
        self.order = self.engine.check_order(order)
        self.scale = sum(min(self.info['duration'][v][p] for p in self.info['eligible'][v])
                         for v in range(self.n)) / self.n
        self.initial_allowance = self.inst['budget'] - sum(self.info['min_cost'])

    def rollout(self, logits, mode, sample=False, seed=0, forced_mapping=None,
                materialize=False):
        if mode not in ('initial', 'local'):
            raise ValueError(mode)
        started = time.perf_counter()
        if len(logits) != self.n * self.k:
            raise ValueError('wrong score count')
        rng = np.random.default_rng(seed)
        calendars = [[] for _ in range(self.k)]
        mapping, starts, ends, ready = [-1]*self.n, [0]*self.n, [0]*self.n, [0]*self.n
        spent, reserve = 0, sum(self.info['min_cost'])
        decisions = []
        for v in self.order:
            reserve -= self.info['min_cost'][v]
            choices = []
            local_denominator = max(1, max(self.info['costs'][v][p]-self.info['min_cost'][v]
                                          for p in self.info['eligible'][v]))
            for p in self.info['eligible'][v]:
                if spent + self.info['costs'][v][p] + reserve > self.inst['budget']:
                    continue
                release = max((ends[u]+self.info['comm'][u, v][mapping[u]][p]
                               for u in self.info['parents'][v]), default=0)
                duration = self.info['duration'][v][p]
                start = find_slot(release, duration, self.inst['demands'][v],
                                  self.inst['capacities'][p], calendars[p])
                extra = self.info['costs'][v][p]-self.info['min_cost'][v]
                choices.append((v*self.k+p, p, start, start+duration, release,
                                extra/max(1, self.initial_allowance), extra/local_denominator))
            offsets = [-c[3]/max(1, self.scale) for c in choices]
            upper, lower = max(offsets), min(offsets)
            offsets = [(value-upper)/max(1., upper-lower) for value in offsets]
            phi = [c[5 if mode == 'initial' else 6] for c in choices]
            if any(not 0 <= f <= 1+1e-10 for f in phi):
                raise AssertionError('score feature outside range')
            values = np.asarray([logits[c[0]][0] + logits[c[0]][1]*f + off
                                 for c, f, off in zip(choices, phi, offsets)])
            if not np.isfinite(values).all():
                raise ValueError('nonfinite score')
            logprob = values-values.max()
            logprob -= math.log(float(np.exp(logprob).sum()))
            if forced_mapping is not None:
                selected = next(i for i, c in enumerate(choices) if c[1] == forced_mapping[v])
            elif sample:
                selected = int(rng.choice(len(choices), p=np.exp(logprob)))
            else:
                selected = min(range(len(choices)), key=lambda i: (-values[i], choices[i][3],
                                                                   choices[i][2], choices[i][1]))
            c = choices[selected]
            a, p, start, end, release = c[:5]
            mapping[v], starts[v], ends[v], ready[v] = p, start, end, release
            calendars[p].append((start, end, self.inst['demands'][v]))
            spent += self.info['costs'][v][p]
            decisions.append({'available': [c[0] for c in choices], 'offsets': offsets,
                              'budget_initial': [c[5] for c in choices],
                              'budget_local': [c[6] for c in choices], 'selected': selected,
                              'action': a, 'selected_log_probability': float(logprob[selected])})
        result = {'decisions': decisions, 'makespan': max(ends), 'cost': spent}
        if materialize:
            tasks = [{'node': v, 'device': mapping[v], 'start': starts[v], 'end': ends[v],
                      'duration': self.info['duration'][v][mapping[v]],
                      'demand': self.inst['demands'][v], 'data_ready': ready[v],
                      'resource_wait': starts[v]-ready[v]} for v in range(self.n)]
            communications = []
            for v in self.order:
                for u in sorted(self.info['parents'][v], key=lambda u: (ends[u], u)):
                    p, q = mapping[u], mapping[v]
                    length = self.info['comm'][u, v][p][q]
                    communications.append({'u': u, 'v': v, 'src': p, 'dst': q,
                                           'start': ends[u], 'end': ends[u]+length,
                                           'duration': length, 'release': ends[u],
                                           'queue_wait': 0, 'local': p == q,
                                           'data_mb': self.inst['data_mb'][u][v]})
            solution = {'algorithm': 'Cached-Budget-'+mode, 'makespan': max(ends), 'cost': spent,
                        'tasks': tasks, 'communications': communications,
                        'decision_order': list(self.order)}
            core.validate(self.inst, solution)
            result['schedule'] = solution
        result['decoder_seconds'] = time.perf_counter()-started
        return result
