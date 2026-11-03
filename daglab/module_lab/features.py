"""Complete-instance physical features. Runs in the simulation interpreter.

The OCT feature/prior borrows the existing PEFT adaptation. New features are
deterministic input information, not labels or simulator rollout lookahead.
"""
import math
from daglab.reference import core


def enrich(record, decoder):
    inst, info = record['instance'], decoder.info
    graph = dict(record['graph'])
    n, k, scale = info['n'], info['k'], max(1., decoder.scale)
    octable = core.optimistic_cost_table(info)
    depth = [0] * n
    for v in info['order']:
        depth[v] = 1 + max((depth[u] for u in info['parents'][v]), default=0)
    maxdepth = max(depth)
    # Raw normalized delays survive explicitly; weighted legacy aggregation is
    # retained only in its named control. Missing edges have zero-size arrays.
    delays = [info['comm'][s // k, t // k][s % k][t % k] / scale
              for s, t in zip(graph['src'], graph['dst'])]
    extra, priors = [], []
    for v in range(n):
        eligible = info['eligible'][v]
        octmin = min(octable[v][p] for p in eligible)
        min_duration = min(info['duration'][v][p] for p in eligible)
        for p in range(k):
            incoming = [info['comm'][u, v][q][p] / scale
                        for u in info['parents'][v] for q in info['eligible'][u]]
            outgoing = [info['comm'][v, u][p][q] / scale
                        for u in info['children'][v] for q in info['eligible'][u]]
            average = lambda xs: sum(xs) / max(1, len(xs))
            extra.append([
                math.log1p(octable[v][p] / scale),
                math.log1p(max(0., octable[v][p] - octmin) / scale),
                depth[v] / maxdepth,
                math.log1p(info['duration'][v][p] / max(1, min_duration)),
                math.log1p(average(incoming)),
                math.log1p(max(incoming, default=0.)),
                math.log1p(average(outgoing)),
                math.log1p(max(outgoing, default=0.)),
            ])
            # Only differences across candidate pools matter. The cap remains
            # the original8*tanh(raw/8); this is not an uncapped PEFT decoder.
            priors.append(-(octable[v][p] - octmin) / scale)
    graph['rich_x'] = [x + e for x, e in zip(graph['x'], extra)]
    graph['edge_delay'] = delays
    graph['oct_prior'] = priors
    graph['task_edges'] = inst['edges']
    graph['pool_delay'] = [[
        sum(info['comm'][u, v][p][q] / scale for u, v in inst['edges']) /
        max(1, len(inst['edges'])) for q in range(k)] for p in range(k)]
    return graph
