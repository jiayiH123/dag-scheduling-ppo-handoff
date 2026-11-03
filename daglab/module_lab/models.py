"""Small switchable representations, all one full forward and parallel a/w heads.

Product-graph messaging is GiPH-inspired; axial task/pool interaction borrows the
scheduling-attention idea, not ReSched's sequential policy or its exact network.
"""
import torch
from torch import nn

PROBABILITY_DTYPE = torch.float32


def set_probability_dtype(name):
    global PROBABILITY_DTYPE
    if name not in ['float32', 'float64']:
        raise ValueError('probability dtype must be float32 or float64')
    PROBABILITY_DTYPE = getattr(torch, name)


ARMS = {
    'legacy_gnn': dict(features='legacy', encoder='gnn', explicit=False, prior=False),
    'physical_gnn': dict(features='physical', encoder='gnn', explicit=False, prior=False),
    'edge_gnn': dict(features='physical', encoder='gnn', explicit=True, prior=False),
    'edge_gnn_ln': dict(features='physical', encoder='gnn', explicit=True, prior=False, normalize=True),
    'edge_attention': dict(features='physical', encoder='attention', explicit=True, prior=False),
    'physical_mlp': dict(features='physical', encoder='mlp', explicit=False, prior=False),
    'edge_gnn_oct': dict(features='physical', encoder='gnn', explicit=True, prior=True),
}


def tensors(graph, device):
    result = {k: torch.tensor(graph[k], dtype=torch.float32, device=device)
              for k in ['x', 'rich_x', 'edge_weight', 'edge_delay', 'oct_prior', 'pool_delay']}
    result.update({k: torch.tensor(graph[k], dtype=torch.long, device=device) for k in ['src', 'dst']})
    result['eligible'] = torch.tensor(graph['eligible'], dtype=torch.bool, device=device)
    result['n'], result['k'] = graph['n'], graph['k']
    adjacency = torch.zeros(graph['n'], graph['n'], device=device)
    for u, v in graph['task_edges']:
        adjacency[v, u], adjacency[u, v] = 1., -1.
    result['adjacency'] = adjacency
    return result


class ModuleActor(nn.Module):
    def __init__(self, arm, hidden=32, layers=2):
        super().__init__()
        self.arm, self.spec, self.forward_calls = arm, ARMS[arm], 0
        dim = 12 if self.spec['features'] == 'legacy' else 20
        self.input = nn.Linear(dim, hidden)
        self.fwd = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) if self.spec['encoder'] != 'mlp'
                                  else nn.Identity() for _ in range(layers)])
        self.bwd = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) if self.spec['encoder'] != 'mlp'
                                  else nn.Identity() for _ in range(layers)])
        self.update = nn.ModuleList([nn.Linear(3 * hidden, hidden) for _ in range(layers)])
        self.delay = nn.ModuleList([nn.Sequential(nn.Linear(2, hidden), nn.Tanh()) for _ in range(layers)]) \
            if self.spec['explicit'] else nn.ModuleList()
        self.task_attn = nn.ModuleList([nn.MultiheadAttention(hidden, 4, dropout=0., batch_first=True)
                                       for _ in range(layers)]) if self.spec['encoder'] == 'attention' else nn.ModuleList()
        self.pool_attn = nn.ModuleList([nn.MultiheadAttention(hidden, 4, dropout=0., batch_first=True)
                                       for _ in range(layers)]) if self.spec['encoder'] == 'attention' else nn.ModuleList()
        norm_count = 2 * layers if self.spec['encoder'] == 'attention' else layers if self.spec.get('normalize') else 0
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(norm_count)])
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, g):
        self.forward_calls += 1
        x = g['x'] if self.spec['features'] == 'legacy' else g['rich_x']
        h = torch.tanh(self.input(x))
        src, dst = g['src'], g['dst']
        w = g['edge_weight'][:, None]
        d = g['edge_delay']
        nd = h.new_zeros((len(h), 1)).index_add(0, dst, torch.ones_like(w)).clamp_min(1)
        od = h.new_zeros((len(h), 1)).index_add(0, src, torch.ones_like(w)).clamp_min(1)
        for layer, (fw, bw, update) in enumerate(zip(self.fwd, self.bwd, self.update)):
            if self.spec['encoder'] == 'mlp':
                f, b = torch.zeros_like(h), torch.zeros_like(h)
            elif self.spec['explicit']:
                # Unlike legacy attenuation, large delays enter the message as
                # data instead of suppressing the entire predecessor vector.
                edge = self.delay[layer](torch.stack([torch.log1p(d), d / (1 + d)], -1))
                f = torch.zeros_like(h).index_add(0, dst, torch.tanh(fw(h[src]) + edge)) / nd
                b = torch.zeros_like(h).index_add(0, src, torch.tanh(bw(h[dst]) + edge)) / od
            else:
                f = torch.zeros_like(h).index_add(0, dst, fw(h[src]) * w) / nd
                b = torch.zeros_like(h).index_add(0, src, bw(h[dst]) * w) / od
            h = torch.tanh(update(torch.cat([h, f, b], -1)))
            if self.spec.get('normalize'):
                h = self.norm[layer](h)
            if self.spec['encoder'] == 'attention':
                z = h.reshape(g['n'], g['k'], -1).transpose(0, 1)
                # Pool is the batch axis for task attention, then task is batch
                # axis for pool interaction; no quadratic (N*P)^2 token matrix.
                z = self.norm[2 * layer](z + self.task_attn[layer](
                    z, z, z, attn_mask=g['adjacency'], need_weights=False)[0])
                z = z.transpose(0, 1)
                z = self.norm[2 * layer + 1](z + self.pool_attn[layer](
                    z, z, z, attn_mask=-torch.log1p(g['pool_delay']), need_weights=False)[0])
                h = z.reshape(len(h), -1)
        context = h[g['eligible']].mean(0, keepdim=True).expand(len(h), -1)
        raw = self.head(torch.cat([h, context], -1))
        if self.spec['prior']:
            raw = raw + torch.stack([g['oct_prior'], torch.zeros_like(g['oct_prior'])], -1)
        return self.output_transform(raw)

    def output_transform(self, raw):
        return 8 * torch.tanh(raw / 8)


def distributions(logits, decisions):
    """Full feasible-pool log distributions at this trajectory's own prefixes."""
    width = max(len(d['available']) for d in decisions)
    device = logits.device
    logits = logits.to(PROBABILITY_DTYPE)
    indices = torch.tensor([d['available'] + [0] * (width - len(d['available'])) for d in decisions], device=device)
    mask = torch.tensor([[True] * len(d['available']) + [False] * (width - len(d['available']))
                         for d in decisions], device=device)
    offsets = torch.tensor([d['offsets'] + [0.] * (width - len(d['available'])) for d in decisions], device=device, dtype=logits.dtype)
    phi = torch.tensor([d['budget_local'] + [0.] * (width - len(d['available'])) for d in decisions], device=device, dtype=logits.dtype)
    pairs = logits[indices]
    score = pairs[..., 0] + pairs[..., 1] * phi + offsets
    logs = torch.log_softmax(score.masked_fill(~mask, -torch.inf), -1)
    return logs, mask


def conditionals(logits, decisions):
    """Score every action with *that trajectory's* masks, EFT and budget prefix."""
    logs, mask = distributions(logits, decisions)
    device = logits.device
    selected = torch.tensor([d['selected'] for d in decisions], device=device)
    # Avoid0*(-inf) in entropy of padding entries.
    entropy = -(logs.exp() * logs.masked_fill(~mask, 0.)).sum(-1)
    return logs.gather(1, selected[:, None]).squeeze(1), entropy


def conditional_kl(logits, reference_logits, decisions):
    logs, mask = distributions(logits, decisions)
    baseline, _ = distributions(reference_logits.detach(), decisions)
    delta = (logs - baseline).masked_fill(~mask, 0.)
    return (logs.exp() * delta).sum(-1).mean()
