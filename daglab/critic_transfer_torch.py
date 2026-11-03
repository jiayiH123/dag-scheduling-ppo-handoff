"""Training-only graph value context, without additional deployment actor calls.

GOODRL-inspired value organization; this is a cumulative-pool adaptation, not
the author's GAT implementation. No critic gradient is shared with the actor.
"""
import torch
from torch import nn
from .classic_ppo_torch import Critic


class SummaryCritic(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(76, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, states):
        return self.net(states).squeeze(-1)


def graph_inputs(row, result, g):
    """Reconstruct pre-action prefixes, masking *all* future schedule fields.

    HEFT order and complete static DAG are known at time0. Normalization uses
    the fixed HEFT reference, never the sampled trajectory's final makespan.
    """
    n, k = row['n'], g['k']
    tasks = sorted(result['schedule']['tasks'], key=lambda t: t['node'])
    order = torch.tensor(result['schedule']['decision_order'])
    rank = torch.empty(n, dtype=torch.long)
    rank[order] = torch.arange(n)
    prefix = rank[None, :] < torch.arange(n)[:, None]
    current = rank[None, :] == torch.arange(n)[:, None]
    p = torch.tensor([t['device'] for t in tasks])
    assignment = torch.nn.functional.one_hot(p, 6).float()
    times = torch.tensor([[t['start'], t['end']] for t in tasks]).float()
    times = torch.log1p(times.clamp_min(0) / max(1., row['HEFT']))
    # Ten dynamic channels: assigned, current, start/end, six assigned pools.
    dynamic = torch.cat([prefix[..., None].float(), current[..., None].float(),
                         times[None].expand(n, -1, -1) * prefix[..., None],
                         assignment[None].expand(n, -1, -1) * prefix[..., None]], -1)
    assert dynamic.shape == (n, n, 10) and bool(torch.isfinite(dynamic).all())
    # The static input averages candidate-pool physical features per task and
    # per-pool task features; all task-pool information is known before rollout.
    rich = g['rich_x'].reshape(n, k, 20)
    src, dst = g['src'] // k, g['dst'] // k
    adj = torch.zeros(n, n)
    adj.index_put_((dst, src), torch.ones_like(src).float(), accumulate=True)
    adj = (adj > 0).float()
    incoming = adj / adj.sum(-1, keepdim=True).clamp_min(1)
    outgoing = adj.T / adj.T.sum(-1, keepdim=True).clamp_min(1)
    # Preserve explicit DAG-edge communication magnitudes in addition to static
    # node summaries; average across the known pool-pair links for each edge.
    delay_sum, delay_count = torch.zeros(n,n), torch.zeros(n,n)
    delay_sum.index_put_((dst,src),g['edge_delay'],accumulate=True)
    delay_count.index_put_((dst,src),torch.ones_like(g['edge_delay']),accumulate=True)
    delay = delay_sum / delay_count.clamp_min(1)
    edge_dst,edge_src = adj.nonzero(as_tuple=True)
    edge_delay = delay[edge_dst,edge_src]
    return dict(dynamic=dynamic, current=current.float(), task=rich.mean(1),
                pool=rich.mean(0), incoming=incoming, outgoing=outgoing,
                edge_dst=edge_dst,edge_src=edge_src,edge_delay=edge_delay,pool_delay=g['pool_delay'],
                states=torch.tensor([d['critic_state'] for d in result['decisions']]))


class GraphCritic(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.task_input = nn.Linear(20, hidden)
        self.dynamic_input = nn.Linear(10, hidden, bias=False)
        self.pool_input = nn.Linear(20, hidden)
        self.global_input = nn.Linear(76, hidden)
        self.edge_input = nn.Linear(2, hidden)
        self.fwd = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(2)])
        self.bwd = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(2)])
        self.update = nn.ModuleList([nn.Linear(3 * hidden, hidden) for _ in range(2)])
        self.attention = nn.MultiheadAttention(hidden, 4, dropout=0., batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        h = torch.tanh(self.task_input(x['task'])[None] + self.dynamic_input(x['dynamic']))
        d=x['edge_delay']
        edge=torch.tanh(self.edge_input(torch.stack([torch.log1p(d),d/(1+d)],-1)))
        ef=torch.zeros_like(h[0]).index_add(0,x['edge_dst'],edge)
        eb=torch.zeros_like(h[0]).index_add(0,x['edge_src'],edge)
        ef=ef/torch.bincount(x['edge_dst'],minlength=h.shape[1])[:,None].clamp_min(1)
        eb=eb/torch.bincount(x['edge_src'],minlength=h.shape[1])[:,None].clamp_min(1)
        for fw, bw, update in zip(self.fwd, self.bwd, self.update):
            f = torch.matmul(x['incoming'], fw(h)) + ef[None]
            b = torch.matmul(x['outgoing'], bw(h)) + eb[None]
            h = torch.tanh(update(torch.cat([h, f, b], -1)))
        k = len(x['pool'])
        assigned = x['dynamic'][..., 4:4+k]
        reservations = torch.einsum('tnk,tnh->tkh', assigned, h)
        reservations = reservations / assigned.sum(1)[..., None].clamp_min(1)
        pools = torch.tanh(self.pool_input(x['pool'])[None] + reservations)
        current = (h * x['current'][..., None]).sum(1)
        glob = torch.tanh(self.global_input(x['states']))
        tokens = torch.cat([h.mean(1)[:, None], h.max(1).values[:, None],
                            current[:, None], pools, glob[:, None]], 1)
        bias=tokens.new_zeros(tokens.shape[1],tokens.shape[1])
        bias[3:3+k,3:3+k]=-torch.log1p(x['pool_delay'])
        context = self.norm(tokens + self.attention(tokens, tokens, tokens, attn_mask=bias,need_weights=False)[0]).mean(1)
        return self.head(torch.cat([context, glob], -1)).squeeze(-1)


def make_critic(kind):
    if kind == 'small':
        return Critic()
    if kind == 'graph':
        return GraphCritic()
    if kind == 'wide':
        target = sum(p.numel() for p in GraphCritic().parameters())
        width = min(range(32, 257), key=lambda w: abs(w*w + 79*w + 1 - target))
        model = SummaryCritic(width)
        assert abs(sum(p.numel() for p in model.parameters()) / target - 1) < .1
        return model
    raise ValueError(kind)


def predict(critic, trajectories, kind):
    if kind == 'graph':
        return torch.cat([critic(t['graph_input']) for t in trajectories])
    return critic(torch.cat([t['states'] for t in trajectories]))


def value_metrics(pred, target, trajectories):
    bins = torch.cat([torch.arange(len(t['states'])) * 10 // len(t['states']) for t in trajectories])
    a, b = pred.detach().clone(), target.detach().clone()
    for i in range(10):
        mask = bins == i
        if bool(mask.any()):
            a[mask] -= a[mask].mean()
            b[mask] -= b[mask].mean()
    return dict(MSE=float((pred-target).square().mean()),
                EV=1-float((pred-target).var(unbiased=False)/target.var(unbiased=False).clamp_min(1e-8)),
                residual_EV=1-float((a-b).var(unbiased=False)/b.var(unbiased=False).clamp_min(1e-8)))
