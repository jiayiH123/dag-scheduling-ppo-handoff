#!/usr/bin/env python3
"""Real-shape operator/physics fidelity gates and throughput, not formal training."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import ast
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daglab.faithful_external import *
import daglab.faithful_external as faithful
from daglab.sft_three_panel import unpack, to_instance, write_json


def main():
    torch.set_num_threads(1); torch.manual_seed(1201)
    out = RUN / 'engineering'; out.mkdir(exist_ok=True)
    con = sqlite3.connect('file:' + str(ROOT / 'runs/synthetic_transfer_20261031/data_base/public.sqlite3') + '?mode=ro', uri=True)
    shapes = con.execute('select n,k,min(key) from records where source="synthetic" and role="train" group by n,k').fetchall()
    assert len(shapes) == 12
    results = []
    for n, k, key in shapes:
        raw = unpack(con.execute('select raw from records where key=?', (key,)).fetchone()[0])
        p = GiPHProblem(to_instance(raw)); heft = p.schedule(); checked = p.finish(heft['mapping'])
        assert checked['C'] == heft['C']
        data = cai_input(p); cai = CAI(data['numeric'].shape[-1]); cai.eval()
        z, raw_z = cai(data)
        state, lp = cai.sample(p, data, z, False)
        p.finish(state['mapping'], state['order'])
        order, decisions = p.priority_order(z.detach().tolist())
        independent = z.new_empty(p.n)
        for v, decision in zip(order, decisions):
            independent[v] = z[v] - torch.logsumexp(z[decision['available']], 0)
        assert torch.allclose(lp, independent, atol=3e-5, rtol=2e-5)
        a = torch.autograd.grad(lp.sum(), z, retain_graph=True)[0]
        b = torch.autograd.grad(independent.sum(), z, retain_graph=True)[0]
        assert torch.allclose(a, b, atol=3e-5, rtol=2e-5)
        rewards = torch.tensor([heft['ends'][v] - state['ends'][v] for v in range(n)], dtype=torch.float32)
        loss = cai_loss([(lp, rewards)], [raw_z.square().mean()]); loss.backward()
        assert all(q.grad is None or bool(torch.isfinite(q.grad).all()) for q in cai.parameters())

        mapping = p.random_mapping(1201); model = GiPH()
        begin = time.perf_counter(); best, loss, info = giph_episode(model, p, mapping, train=True)
        if loss is not None:
            loss.backward()
        seconds = time.perf_counter() - begin
        assert all(q.grad is None or bool(torch.isfinite(q.grad).all()) for q in model.parameters())
        final = p.finish(best['mapping']); assert final['C'] == best['C']
        assert info['moves'] == 2 * n or not bool(p.graph(mapping, p.state(mapping))[1].any())
        results.append(dict(n=n, k=k, key=key, CAI_logprob_and_gradient=True, GiPH_author_DGL=True,
                            full_moves=info['moves'], giph_episode_forward_backward_seconds=seconds,
                            CAI_parameters=sum(x.numel() for x in cai.parameters()),
                            GiPH_parameters=sum(x.numel() for x in model.parameters())))
        print(json.dumps(results[-1]), flush=True)

    # Pinned author attention versus independent QK-softmax-V, including gradients.
    layer = cai.encoder.layers[0].self_attn; layer.eval()
    x = torch.randn(7, 1, 64, requires_grad=True); bias = torch.randn(1, 4, 7, 7)
    actual, _ = layer(x, x, x, attn_bias=bias, need_weights=False)
    q = layer.q_proj(x).view(7, 1, 4, 16).permute(1, 2, 0, 3)
    ke = layer.k_proj(x).view(7, 1, 4, 16).permute(1, 2, 0, 3)
    v = layer.v_proj(x).view(7, 1, 4, 16).permute(1, 2, 0, 3)
    expected = (torch.softmax(q @ ke.transpose(-1, -2) / 4 + bias, -1) @ v).permute(2, 0, 1, 3).reshape(7, 1, 64)
    expected = layer.out_proj(expected)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    ga = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
    gb = torch.autograd.grad(expected.square().sum(), x)[0]
    assert torch.allclose(ga, gb, atol=3e-6, rtol=1e-5)
    # Compare all original node/edge features with author environment methods.
    source_path = RUN / 'sources/giph/placement_rl/placement_env.py'
    source = source_path.read_text().replace('if op == 0 or mapping[op] == dev:',
                                           'if not parents or mapping[op] == dev:')
    # Exact declared constant-column finite repair, applied to reference only.
    source = source.replace('feat[feature] = (feat[feature] - mi) * 2 / (ma-mi) - 1',
                            'feat[feature] = torch.nan_to_num((feat[feature] - mi) * 2 / (ma-mi) - 1)')
    tree = ast.parse(source)
    environment_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name=='PlacementEnv')
    namespace = dict(torch=torch, np=np, nx=nx, dgl=dgl, product=itertools.product,
        computation_latency=lambda program, network, op, dev: float(p.duration[op,dev]),
        communication_latency=lambda program, network, u,v,q,r: p.info['comm'][u,v][q][r])
    exec(compile(ast.Module(body=[environment_class], type_ignores=[]), str(source_path), 'exec'), namespace)
    env_class = namespace['PlacementEnv']; env = env_class.__new__(env_class)
    mapping = p.random_mapping(123); state = p.state(mapping); g, mask = p.graph(mapping, state)
    env.full_graph_node_dict = {0:{0:{v:{q:p.pair_index[v,q] for q in p.info['eligible'][v]} for v in range(p.n)}}}
    env.full_graph_action_dict = {0:{0:dict(enumerate(p.pairs))}}
    env.programs = [p.program]
    network = types.SimpleNamespace(comp_rate=torch.tensor([1/s for s in p.inst['speeds']]),
        comm_delay=np.zeros((p.k,p.k)), comm_rate=np.divide(1,np.asarray(p.inst['bandwidth']),
        out=np.zeros((p.k,p.k)),where=~np.eye(p.k,dtype=bool)))
    env.networks = [network]; env.node_feature_mean = {}; env.node_feature_std = {}
    stats = nx.DiGraph()
    for v in range(p.n):
        stats.add_node(v,start_time=[state['starts'][v]],end_time=[state['ends'][v]])
    native_node = env.get_full_node_feature(0,0,mapping,stats)
    expected_node = torch.stack([native_node[name] for name in env_class.NODE_FEATURES],dim=1)
    assert torch.allclose(g.ndata['x'][:,:5],expected_node,atol=3e-6,rtol=1e-5)
    u,v,edge = env.get_full_edge_feature(0,0,mapping,stats)
    assert torch.equal(g.edges()[0],u) and torch.equal(g.edges()[1],v)
    expected_edge = torch.stack([edge[name] for name in env_class.EDGE_FEATURES],dim=1).float()
    assert torch.allclose(g.edata['x'],expected_edge,atol=3e-6,rtol=1e-5)

    # Exact save/restore of one online author-loss update on a real20-task shape.
    n,k,key = shapes[0]
    raw = unpack(con.execute('select raw from records where key=?',(key,)).fetchone()[0])
    model = GiPH(); optimizer = torch.optim.Adam(model.parameters(),lr=.01)
    saved_model = copy.deepcopy(model.state_dict()); saved_opt = copy.deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state()
    def one_update():
        instance = GiPHProblem(to_instance(raw)); mapping = instance.random_mapping(91)
        optimizer.zero_grad(); best,loss,info = giph_episode(model,instance,mapping,train=True)
        loss.backward(); optimizer.step()
        return best, info
    first = one_update(); expected_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(saved_model); optimizer.load_state_dict(saved_opt); torch.set_rng_state(rng)
    repeated = one_update()
    assert first==repeated
    assert all(torch.equal(expected_weights[name],value) for name,value in model.state_dict().items())
    write_json(out / 'CHECKS.json', dict(status='PASS', shapes=results, author_attention_forward_gradient=True,
        GiPH_node_edge_features_match_author_with_declared_repairs=True,
        exact_online_update_resume=True,
        torch=torch.__version__, numpy=np.__version__, dgl=dgl.__version__,
        mean_GiPH_episode_seconds=sum(r['giph_episode_forward_backward_seconds'] for r in results) / len(results),
        checkpoint_or_training_state_used=False))
    print('ENGINEERING PASS', flush=True)


if __name__ == '__main__':
    main()
