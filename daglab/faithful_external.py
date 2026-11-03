"""Pinned author neural modules with explicit shared-physics input adapters.

No old learned-control network is imported. See the frozen reproduction protocol.
"""
import ast
import importlib.util
import itertools
import math
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/faithful_external_20261102'
sys.path.insert(0, str(RUN / 'deps'))
os.environ.setdefault('DGLBACKEND', 'pytorch')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import networkx as nx
import dgl
import dgl.function as fn

from .external_baseline_sim import Problem, stable_seed


def author_definitions(path, names, namespace=None):
    """Execute exact selected definitions, without unused environment imports."""
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in selected} == set(names)
    scope = dict(torch=torch, nn=nn, F=F, np=np, math=math, nx=nx, dgl=dgl,
                 fn=fn, Linear=nn.Linear, product=itertools.product, __name__=__name__)
    if namespace:
        scope.update(namespace)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), scope)
    return scope


_GIPH = RUN / 'sources/giph'
_fnn = author_definitions(_GIPH / 'placement_rl/primative_nn.py', ['weights_init_', 'reset', 'FNN'])
_agent = author_definitions(_GIPH / 'placement_rl/placement_agent.py',
                           ['MPLayer', 'GiPHEmbedding', 'SoftmaxActor', 'PlacementAgent'], {'FNN': _fnn['FNN']})
_program = author_definitions(_GIPH / 'env/program.py', ['Program'])['Program']
_cai = author_definitions(RUN / 'sources/cai/RL_models/utils.py', ['sample_gumbel', 'GumbelTopoSortAdj'])


def _fairseq_compat():
    """Only wrappers exercised by the pinned Graphormer at q_noise=layerdrop=0."""
    if 'fairseq' in sys.modules:
        return
    fairseq = types.ModuleType('fairseq')
    utils = types.ModuleType('fairseq.utils')
    utils.get_activation_fn = lambda name: {'gelu': F.gelu, 'relu': F.relu, 'tanh': torch.tanh}[name]
    utils.softmax = lambda x, dim, onnx_trace=False: F.softmax(x, dim=dim, dtype=torch.float32)
    modules = types.ModuleType('fairseq.modules')

    class FairseqDropout(nn.Module):
        def __init__(self, p, module_name=None):
            super().__init__(); self.p = p

        def forward(self, x, inplace=False):
            return F.dropout(x, self.p, self.training, inplace=inplace)

    def layer_norm(size, export=False):
        assert not export
        return nn.LayerNorm(size)

    def quant_noise(module, p, block_size):
        assert p == 0, 'Nonzero quantization noise is intentionally unsupported.'
        return module

    modules.FairseqDropout = FairseqDropout
    modules.LayerNorm = layer_norm
    modules.LayerDropModuleList = nn.ModuleList
    drop = types.ModuleType('fairseq.modules.fairseq_dropout'); drop.FairseqDropout = FairseqDropout
    noise = types.ModuleType('fairseq.modules.quant_noise'); noise.quant_noise = quant_noise
    fairseq.modules = modules; fairseq.utils = utils
    for name, module in [('fairseq', fairseq), ('fairseq.utils', utils), ('fairseq.modules', modules),
                         ('fairseq.modules.fairseq_dropout', drop), ('fairseq.modules.quant_noise', noise)]:
        sys.modules[name] = module


def _graphormer_encoder():
    _fairseq_compat()
    directory = RUN / 'sources/cai/Graphormer/graphormer/modules'
    package = types.ModuleType('_pinned_cai_graphormer'); package.__path__ = [str(directory)]
    sys.modules[package.__name__] = package
    name = package.__name__ + '.graphormer_graph_encoder'
    spec = importlib.util.spec_from_file_location(name, directory / 'graphormer_graph_encoder.py')
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module.GraphormerGraphEncoder


GraphormerGraphEncoder = _graphormer_encoder()


class NumericNodeFeature(nn.Module):
    """Original degree/token construction; continuous input required by Appendix C.1."""
    def __init__(self, original, features):
        super().__init__()
        self.atom_encoder = nn.Linear(features, 64)
        self.in_degree_encoder = original.in_degree_encoder
        self.out_degree_encoder = original.out_degree_encoder
        self.graph_token = original.graph_token
        nn.init.normal_(self.atom_encoder.weight, std=.02 / math.sqrt(4))
        nn.init.zeros_(self.atom_encoder.bias)

    def forward(self, data):
        x = self.atom_encoder(data['numeric'])
        x = x + self.in_degree_encoder(data['in_degree']) + self.out_degree_encoder(data['out_degree'])
        token = self.graph_token.weight[None].expand(len(x), -1, -1)
        return torch.cat([token, x], dim=1)


class CAI(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.encoder = GraphormerGraphEncoder(num_atoms=1, num_in_degree=121, num_out_degree=121,
            num_edges=32, num_spatial=130, num_edge_dis=1, edge_type=None, multi_hop_max_dist=0,
            num_encoder_layers=4, embedding_dim=64, ffn_embedding_dim=64, num_attention_heads=4,
            q_noise=0, layerdrop=0)
        self.encoder.graph_node_feature = NumericNodeFeature(self.encoder.graph_node_feature, features)
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.sampler = _cai['GumbelTopoSortAdj']()
        self.forward_calls = 0

    def forward(self, data):
        self.forward_calls += 1
        states, _ = self.encoder(data)
        raw = self.head(states[-1][1:, 0]).flatten()
        return 5 * (raw - raw.mean()) / (raw.std() + 1e-20), raw

    def sample(self, problem, data, logits, stochastic):
        if stochastic:
            lp, priorities = self.sampler(logits, data['precedence'])
        else:
            lp, priorities = self.sampler(logits, data['precedence'], True, logits.detach())
        order, _ = problem.priority_order(priorities.detach().tolist())
        state = problem.schedule(order=order)
        return state, lp


def cai_input(problem):
    p = problem; n, k = p.n, p.k
    dur = np.zeros((n, 6)); fee = np.zeros((n, 6)); eligibility = np.zeros((n, 6))
    dur[:, :k] = np.log1p(p.duration); fee[:, :k] = np.log1p(p.costs)
    eligibility[:, :k] = p.eligible
    traffic = np.asarray(p.inst['data_mb'])
    local = np.column_stack([np.log1p(p.inst['workload']), np.log1p(traffic.sum(0)),
        np.log1p(traffic.sum(1)), np.asarray(p.inst['demands']), dur, fee])
    local = (local - local.mean(0)) / (local.std(0) + 1e-8)
    speeds = np.zeros(6); capacities = np.zeros(6); prices = np.zeros(6); bandwidth = np.zeros((6, 6))
    speeds[:k] = np.log1p(p.inst['speeds']); capacities[:k] = np.asarray(p.inst['capacities']) / 12
    prices[:k] = np.log1p(p.inst['cost_rates']); bandwidth[:k, :k] = np.log1p(p.inst['bandwidth'])
    minimum = sum(p.info['min_cost'])
    maximum = sum(max(int(p.costs[v, q]) for q in p.info['eligible'][v]) for v in range(n))
    eta = (p.budget - minimum) / max(1, maximum - minimum)
    assert 0 <= eta <= 1, 'Original panel budgets must lie within physical min/max cost.'
    globals_ = np.r_[speeds, capacities, prices, bandwidth.ravel(), k / 6, eta]
    numeric = np.column_stack([local, eligibility, np.broadcast_to(globals_, (n, len(globals_)))])
    adjacency = np.zeros((n, n), dtype=np.int64)
    edge_type = np.zeros((n, n, 1), dtype=np.int64)
    distance = np.full((n, n), 128, dtype=np.int64); np.fill_diagonal(distance, 0)
    for u, v in p.inst['edges']:
        adjacency[v, u] = 1
        edge_type[u, v, 0] = min(31, 1 + int(math.log2(1 + float(traffic[u, v]))))
    for v in p.info['order']:
        for u in p.parents[v]:
            distance[:, v] = np.minimum(distance[:, v], distance[:, u] + 1)
    return dict(x=torch.ones((1, n, 1), dtype=torch.long), numeric=torch.tensor(numeric[None], dtype=torch.float32),
        in_degree=torch.tensor([len(p.parents[v]) for v in range(n)])[None],
        out_degree=torch.tensor([len(p.children[v]) for v in range(n)])[None],
        spatial_pos=torch.tensor(distance[None]), attn_bias=torch.zeros((1, n + 1, n + 1)),
        edge_input=None, attn_edge_type=torch.tensor(edge_type[None]), precedence=torch.tensor(adjacency))


def cai_loss(items, raws):
    """Author loss on task-indexed, padded120 dense advantages and logprobs."""
    lp = torch.stack([F.pad(l, (0, 120 - len(l))) for l, _ in items])
    rewards = torch.stack([F.pad(r, (0, 120 - len(r))) for _, r in items])
    active = rewards[rewards != 0]
    mean = active.mean() if len(active) else rewards.new_tensor(0.)
    std = active.std() if len(active) > 1 else rewards.new_tensor(1.)
    advantages = ((rewards - mean) / (std + 1e-10)).clamp(-2, 2)
    mask = (rewards != 0) | (lp != 0)
    entropy_term = lp.sum(-1) / mask.sum(-1).clamp_min(1)
    return -(advantages * lp).mean() + .001 * torch.stack(raws).mean() + .01 * entropy_term.mean()


class GiPH(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = _agent['GiPHEmbedding'](8, 5, 10, torch.device('cpu'))
        self.policy = _agent['SoftmaxActor'](10, torch.device('cpu'), hidden_dim=16)
        self.forward_calls = 0

    def forward(self, graph, mask):
        self.forward_calls += 1
        return self.policy(self.embedding(graph), mask)


def minmax(a):
    a = np.asarray(a, dtype=np.float32)
    if not len(a):
        return a
    span = np.ptp(a, axis=0)
    return np.divide(2 * (a - a.min(0)), span, out=np.ones_like(a), where=span != 0) - 1


class GiPHProblem(Problem):
    def __init__(self, inst):
        super().__init__(inst)
        graph = nx.DiGraph(); graph.add_nodes_from(range(self.n)); graph.add_edges_from(inst['edges'])
        for u, v in graph.edges:
            graph.edges[u, v]['bytes'] = float(inst['data_mb'][u][v])
        self.program = _program.__new__(_program); self.program.P = graph
        self.program.n_operators = self.n
        self.program.op_compute = torch.tensor(inst['workload'])
        self.program.data_bytes = torch.tensor(inst['data_mb'], dtype=torch.float32)
        self.program.op_parents = self.parents
        self.program.op_children = self.children
        # Original init_criticality's finite equal-data limit when all bytes are0.
        replaced = []
        for u in range(self.n):
            es = list(graph.out_edges(u))
            if es and max(graph.edges[e]['bytes'] for e in es) == 0:
                for e in es:
                    replaced.append((e, graph.edges[e]['bytes'])); graph.edges[e]['bytes'] = 1.
        self.program.init_criticality()
        for e, original in replaced:
            graph.edges[e]['bytes'] = original
        self.pairs = [(v, q) for v in range(self.n) for q in self.info['eligible'][v]]
        self.pair_index = {pair: i for i, pair in enumerate(self.pairs)}
        self.pair_v = np.array([a for a, _ in self.pairs]); self.pair_p = np.array([b for _, b in self.pairs])

    def critical_path(self, mapping, state):
        path = [max(range(self.n), key=lambda v: state['ends'][v])]
        while self.parents[path[-1]]:
            v = path[-1]
            path.append(max(self.parents[v], key=lambda u: state['ends'][u] + self.info['comm'][u, v][mapping[u]][mapping[v]]))
        return list(reversed(path))

    def graph(self, mapping, state, previous=None):
        self.program.update_criticality(self.critical_path(mapping, state))
        graph = self.program.P; n, k = self.n, self.k
        vs, ps = self.pair_v, self.pair_p; mapping = np.asarray(mapping)
        node = np.zeros((len(vs), 5), dtype=np.float32)
        node[:, 0] = np.asarray(self.inst['workload'])[vs]
        node[:, 1] = 1 / np.asarray(self.inst['speeds'])[ps]
        node[:, 2] = self.duration[vs, ps]
        node[:, 4] = [graph.nodes[int(v)]['criticality'] for v in vs]
        for v in range(n):
            if not self.parents[v]:
                continue
            release = np.zeros(k)
            for u in self.parents[v]:
                release = np.maximum(release, state['ends'][u] + np.asarray(self.info['comm'][u, v][mapping[u]]))
            ids = np.nonzero(vs == v)[0]
            node[ids, 3] = release[ps[ids]] - state['starts'][v]
        node[ps == mapping[vs], 3] = 0
        # Original features stay first and retain original [-1,1] scaling.
        fees = minmax(self.costs[vs, ps, None])
        capacity = np.asarray(self.inst['demands'])[vs] / np.asarray(self.inst['capacities'])[ps]
        slack = np.full(len(vs), (self.budget - state['cost']) / max(1, self.budget))
        x = np.column_stack([minmax(node), fees, capacity, slack]).astype(np.float32)
        mask = state['cost'] - self.costs[vs, mapping[vs]] + self.costs[vs, ps] <= self.budget
        mask &= ps != mapping[vs]
        if previous is not None:
            mask &= vs != previous
        src = []; dst = []; features = []
        for v, p in self.pairs:
            target = self.pair_index[v, p]
            for u in self.parents[v]:
                source = self.pair_index[u, int(mapping[u])]
                q = int(mapping[u]); src.append(source); dst.append(target)
                features.append([self.inst['data_mb'][u][v], 0., self.info['comm'][u, v][q][p],
                    0. if q == p else 1 / self.inst['bandwidth'][q][p], graph.edges[u, v]['criticality']])
            if p != mapping[v]:
                for c in self.children[v]:
                    q = int(mapping[c]); src.append(target); dst.append(self.pair_index[c, q])
                    features.append([self.inst['data_mb'][v][c], 0., self.info['comm'][v, c][p][q],
                        0. if p == q else 1 / self.inst['bandwidth'][p][q], graph.edges[v, c]['criticality']])
        g = dgl.graph((src, dst), num_nodes=len(vs))
        g.ndata['x'] = torch.from_numpy(x)
        g.edata['x'] = torch.from_numpy(minmax(np.asarray(features).reshape(-1, 5)))
        return g, torch.from_numpy(mask)

    def history(self):
        return [self.program.P.edges[u, v]['criticality'] for u, v in self.inst['edges']]

    def set_history(self, history):
        for edge, value in zip(self.inst['edges'], history):
            self.program.P.edges[tuple(edge)]['criticality'] = value
        self.program.populate_criticality()


def giph_episode(model, p, mapping, train=False, gamma=.97, generator=None):
    state = p.state(mapping); best = dict(C=state['C'], cost=state['cost'], mapping=list(mapping))
    log_probs = []; rewards = []; previous = None; visited = []
    for _ in range(2 * p.n):
        graph, mask = p.graph(mapping, state, previous)
        if not bool(mask.any()):
            break
        probabilities = model(graph, mask)
        assert bool(torch.isfinite(probabilities).all()) and float(probabilities.sum()) > 0
        # Categorical sampling, including at evaluation, as in author code.
        if generator is None:
            distribution = torch.distributions.Categorical(probabilities)
            action = distribution.sample(); logp = distribution.log_prob(action)
        else:
            action = torch.multinomial(probabilities, 1, generator=generator)[0]
            logp = probabilities[action].clamp_min(torch.finfo(probabilities.dtype).eps).log()
        v, q = p.pairs[int(action)]; mapping[v] = q; previous = v
        new = p.state(mapping)
        assert new['cost'] <= p.budget
        rewards.append(float(state['C'] - new['C']))
        if train:
            log_probs.append(logp)
        state = new; visited.append(new['C'])
        if (new['C'], new['cost']) < (best['C'], best['cost']):
            best = dict(C=new['C'], cost=new['cost'], mapping=list(mapping))
    if train and log_probs:
        returns = []; value = 0.
        for reward in reversed(rewards):
            value = reward + gamma * value; returns.insert(0, value)
        returns = [value - sum(rewards[:i + 1]) / (i + 1) for i, value in enumerate(returns)]
        loss = -(torch.stack(log_probs) * torch.tensor(returns, dtype=torch.float32)).sum()
    else:
        loss = None
    return best, loss, dict(moves=len(rewards), total_reward=sum(rewards), final_C=state['C'], visited=visited)
