"""Tiny, permutation-equivariant product-graph actor; pure PyTorch, CPU.

No external graph library. Product nodes are task-device candidates. This is a
feasibility prototype using prior graph ideas, not a claimed new architecture.
"""
import torch
from torch import nn


class SingleForwardActor(nn.Module):
    def __init__(self, mode, hidden=32):
        super().__init__()
        if mode not in ("order", "joint"):
            raise ValueError(mode)
        self.mode = mode
        self.input = nn.Linear(12, hidden)
        self.fwd = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(2)])
        self.bwd = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(2)])
        self.update = nn.ModuleList([nn.Linear(3*hidden, hidden) for _ in range(2)])
        self.head = nn.Sequential(nn.Linear(2*hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.forward_calls = 0

    def forward(self, graph):
        self.forward_calls += 1
        x = torch.tensor(graph["x"], dtype=torch.float32)
        eligible = torch.tensor(graph["eligible"], dtype=torch.bool)
        src = torch.tensor(graph["src"], dtype=torch.long)
        dst = torch.tensor(graph["dst"], dtype=torch.long)
        weights = torch.tensor(graph["edge_weight"], dtype=torch.float32).unsqueeze(1)
        h = torch.tanh(self.input(x))
        count = h.shape[0]
        in_degree = torch.zeros(count, 1).index_add(0, dst, torch.ones_like(weights)).clamp_min(1)
        out_degree = torch.zeros(count, 1).index_add(0, src, torch.ones_like(weights)).clamp_min(1)
        for fw, bw, update in zip(self.fwd, self.bwd, self.update):
            forward = torch.zeros_like(h).index_add(0, dst, fw(h[src])*weights)/in_degree
            backward = torch.zeros_like(h).index_add(0, src, bw(h[dst])*weights)/out_degree
            h = torch.tanh(update(torch.cat([h, forward, backward], dim=-1)))
        context = h[eligible].mean(0, keepdim=True)
        if self.mode == "order":
            shaped = h.reshape(graph["n"], graph["k"], -1)
            mask = eligible.reshape(graph["n"], graph["k"], 1)
            h = (shaped*mask).sum(1)/mask.sum(1).clamp_min(1)
        logits = self.head(torch.cat([h, context.expand(h.shape[0], -1)], dim=-1)).squeeze(-1)
        return 8*torch.tanh(logits/8)
