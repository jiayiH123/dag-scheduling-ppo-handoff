#!/usr/bin/env python3
"""Finite replacement GiPH queue; frozen CAI files and active workers unchanged."""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import daglab.faithful_external as f
from daglab.faithful_external_data import Inputs
from daglab.sft_three_panel import write_json
from daglab.structured_dataset import sha256
from scripts.train_faithful_external_20261102 import binding as original_binding, evaluate, save_checkpoint

torch = f.torch
RUN = ROOT / 'runs/giph_paper200_20261102'
CONFIG = ROOT / 'configs/giph_paper200_20261102.json'
PROTOCOL = ROOT / 'protocols/giph_paper200_20261102.md'


class PaperGiPH(f.GiPH):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.embedding = f._agent['GiPHEmbedding'](7, 4, 10, torch.device('cpu'))
        self.policy = f._agent['SoftmaxActor'](10, torch.device('cpu'), hidden_dim=16)
        self.forward_calls = 0

    def forward(self, graph, mask):
        # Tables4/5 omit released-default criticality (native fifth column).
        graph = graph.local_var()
        assert graph.ndata['x'].shape[1] == 8 and graph.edata['x'].shape[1] == 5
        graph.ndata['x'] = graph.ndata['x'][:, [0, 1, 2, 3, 5, 6, 7]]
        graph.edata['x'] = graph.edata['x'][:, :4]
        return super().forward(graph, mask)


def bound(inputs):
    return dict(shared=original_binding(inputs), replacement={str(p.relative_to(ROOT)): sha256(p)
                for p in [Path(__file__), CONFIG, PROTOCOL]})


def episode(model, optimizer, inputs, key, episode_id, seed, history):
    optimizer.zero_grad(set_to_none=True)
    model.train()
    problem = f.GiPHProblem(inputs.instance(key))
    if key in history:
        problem.set_history(history[key])
    initial = problem.random_mapping(f.stable_seed(key, seed * 100000 + episode_id))
    initial_C = problem.state(initial)['C']
    best, loss, work = f.giph_episode(model, problem, initial, train=True, gamma=.97)
    norm = 0.
    if loss is not None:
        assert bool(torch.isfinite(loss)), 'Nonfinite policy loss'
        loss.backward()
        norm = math.sqrt(sum(float(q.grad.square().sum()) for q in model.parameters() if q.grad is not None))
        assert math.isfinite(norm), 'Nonfinite gradient'
        assert all(bool(torch.isfinite(q.grad).all()) for q in model.parameters() if q.grad is not None)
        optimizer.step()
    history[key] = problem.history()
    checked = problem.finish(best['mapping'])
    assert checked['C'] == best['C'] and checked['cost'] <= problem.budget
    return dict(episode=episode_id, key=key, n=problem.n, k=problem.k,
                loss=float(loss.detach()) if loss is not None else 0.,
                optimizer_update=loss is not None, gradient_norm=norm, moves=work['moves'],
                initial_C=initial_C, best_C=best['C'], reward=work['total_reward'])


def checks():
    torch.set_num_threads(1)
    inputs = Inputs()
    for name in ['CHECKS.json', 'CAI_RESUME.json', 'GIPH_UPDATE_PARITY.json']:
        assert json.loads((f.RUN / 'engineering' / name).read_text())['status'] == 'PASS'
    key = next(r[0] for r in inputs.metadata if r[2:4] == (20, 3))
    torch.manual_seed(9182)
    model = PaperGiPH(); optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    history = {}
    episode(model, optimizer, inputs, key, 1, 11, history)
    snapshot = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                  rng=torch.get_rng_state(), history=history))
    first = episode(model, optimizer, inputs, key, 2, 11, history)
    expected = copy.deepcopy(model.state_dict())
    model2 = PaperGiPH(); optimizer2 = torch.optim.Adam(model2.parameters(), lr=.01)
    model2.load_state_dict(snapshot['model']); optimizer2.load_state_dict(snapshot['optimizer'])
    torch.set_rng_state(snapshot['rng'])
    second = episode(model2, optimizer2, inputs, key, 2, 11, snapshot['history'])
    assert first == second
    assert all(torch.equal(v, model2.state_dict()[k]) for k, v in expected.items())
    large = next(r[0] for r in inputs.metadata if r[2:4] == (120, 6))
    p = f.GiPHProblem(inputs.instance(large)); mapping = p.random_mapping(9921)
    graph, mask = p.graph(mapping, p.state(mapping))
    probabilities = model2(graph, mask)
    assert bool(torch.isfinite(probabilities).all())
    assert abs(float(probabilities.sum()) - 1) < 1e-5
    result = dict(status='PASS', exact_two_episode_replay=True, real_large_shape=[120, 6],
                  native_features=[4, 4], adapted_features=[7, 4],
                  parameters=sum(q.numel() for q in model.parameters()), binding=bound(inputs))
    RUN.mkdir(exist_ok=True); write_json(RUN / 'ENGINEERING_CHECKS.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'binding'}), flush=True)


def worker(seed):
    torch.set_num_threads(1)
    cfg = json.loads(CONFIG.read_text()); inputs = Inputs(); binding = bound(inputs)
    assert json.loads((RUN / 'ENGINEERING_CHECKS.json').read_text())['binding'] == binding
    out = RUN / f'seed{seed}'; out.mkdir(exist_ok=True)
    lock = (out / 'worker.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'COMPLETE.json').exists():
        assert json.loads((out / 'COMPLETE.json').read_text())['binding'] == binding
        return
    torch.manual_seed(seed)
    model = PaperGiPH(); optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    # Sampling with replacement, fixed before any rollout or VAL observation.
    indices = f.np.random.default_rng(seed).integers(len(inputs.keys), size=cfg['episodes_per_seed'])
    keys = [inputs.keys[int(i)] for i in indices]
    write_json(out / 'SAMPLE_PLAN.json', dict(seed=seed, pool=17261, keys=keys,
        unique_configurations=len(set(keys)), sampling=cfg['sampling']))
    state = dict(binding=binding, seed=seed, episodes=0, optimizer_updates=0, moves=0,
                 history={}, records=[], training_seconds=0.)
    checkpoint = out / 'resume.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=False)
        assert saved['binding'] == binding
        model.load_state_dict(saved.pop('model')); optimizer.load_state_dict(saved.pop('optimizer'))
        torch.set_rng_state(saved.pop('rng')); state = saved
    else:
        save_checkpoint(out / 'initial.pt', dict(model=model.state_dict(), binding=binding))

    def persist():
        save_checkpoint(checkpoint, dict(state, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                         rng=torch.get_rng_state()))
        # Entire committed history is tiny (200 records); no uncommitted tail survives.
        (out / 'training.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in state['records']))

    for index in range(state['episodes'], cfg['episodes_per_seed']):
        begin = time.perf_counter()
        row = episode(model, optimizer, inputs, keys[index], index + 1, seed, state['history'])
        state['training_seconds'] += time.perf_counter() - begin
        state['episodes'] = index + 1; state['moves'] += row['moves']
        state['optimizer_updates'] += int(row['optimizer_update'])
        row['training_seconds'] = state['training_seconds']; state['records'].append(row)
        persist()
        write_json(out / 'progress.json', dict(stage='training', **row, total_episodes=200))
        print('TRAIN', seed, index + 1, row['n'], row['moves'], row['loss'], flush=True)
    save_checkpoint(out / 'final.pt', dict(model=model.state_dict(), binding=binding, episodes=200))
    write_json(out / 'TRAINING_COMPLETE.json', dict(episodes=200, optimizer_updates=state['optimizer_updates'],
        moves=state['moves'], training_seconds=state['training_seconds'], binding=binding))
    # One terminal sweep per training seed; same policy, both initializations.
    observation = evaluate(model, inputs, 'giph', out, 200, action_seed=cfg['evaluation_action_seed'], tag='terminal')
    write_json(out / 'COMPLETE.json', dict(status='complete', seed=seed, binding=binding,
        episodes=200, optimizer_updates=state['optimizer_updates'], moves=state['moves'],
        training_seconds=state['training_seconds'], final=observation, VAL_checkpoint_selection=False))


def controller():
    import psutil
    lock = (RUN / 'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((RUN / 'ENGINEERING_CHECKS.json').read_text())['status'] == 'PASS'
    cfg = json.loads(CONFIG.read_text()); python = sys.executable; processes = {}; attempts = {}
    started = time.time()
    write_json(RUN / 'controller_status.json', dict(status='running', pid=os.getpid(), started_unix=started,
        episodes_per_seed=200, learning_rate=.01, seeds=cfg['seeds'], prior_GiPH_queue_stopped=True))

    def launch(seed):
        attempts[seed] = attempts.get(seed, 0) + 1
        out = RUN / f'seed{seed}'; out.mkdir(exist_ok=True)
        command = [python, '-u', str(Path(__file__).resolve()), '--seed', str(seed)]
        with (out / 'worker.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        write_json(out / 'PROCESS.json', dict(pid=process.pid, command=command, attempt=attempts[seed]))
        processes[seed] = process

    for seed in cfg['seeds']:
        if not (RUN / f'seed{seed}' / 'COMPLETE.json').exists(): launch(seed)
    report = 0
    while True:
        for seed, process in list(processes.items()):
            if process.poll() is not None and not (RUN / f'seed{seed}' / 'COMPLETE.json').exists():
                if attempts[seed] <= cfg['maximum_restarts']: launch(seed)
        if time.time() - report >= 60:
            subprocess.run(['/opt/miniconda3/envs/dag-research/bin/python',
                            str(ROOT / 'scripts/report_giph_paper200_20261102.py')], cwd=ROOT, check=True)
            report = time.time()
        done = all((RUN / f'seed{seed}' / 'COMPLETE.json').exists() or
                   (seed in processes and processes[seed].poll() is not None and
                    attempts[seed] > cfg['maximum_restarts']) for seed in cfg['seeds'])
        if done: break
        time.sleep(5)
    success = all((RUN / f'seed{seed}' / 'COMPLETE.json').exists() for seed in cfg['seeds'])
    write_json(RUN / 'controller_status.json', dict(status='complete' if success else 'failed',
        pid=os.getpid(), elapsed_seconds=time.time()-started, seeds=cfg['seeds'], episodes_per_seed=200))
    subprocess.run(['/opt/miniconda3/envs/dag-research/bin/python',
                    str(ROOT / 'scripts/report_giph_paper200_20261102.py')], cwd=ROOT, check=True)
    # Keep former CAI reports alive without reviving the canceled GiPH queue.
    cai_pid = json.loads((f.RUN / 'cai_QUEUE_PROCESS.json').read_text())['pid']
    while psutil.pid_exists(cai_pid) and not (f.RUN / 'cai_FINAL.json').exists():
        try:
            if psutil.Process(cai_pid).status() == psutil.STATUS_ZOMBIE: break
        except psutil.NoSuchProcess: break
        subprocess.run(['/opt/miniconda3/envs/dag-research/bin/python',
                        str(ROOT / 'scripts/report_giph_paper200_20261102.py')], cwd=ROOT, check=False)
        time.sleep(30)
    subprocess.run(['/opt/miniconda3/envs/dag-research/bin/python',
                    str(ROOT / 'scripts/report_giph_paper200_20261102.py')], cwd=ROOT, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int)
    parser.add_argument('--check', action='store_true'); args = parser.parse_args()
    try:
        checks() if args.check else worker(args.seed) if args.seed is not None else controller()
    except BaseException:
        out = RUN / f'seed{args.seed}' if args.seed is not None else RUN
        out.mkdir(exist_ok=True)
        write_json(out / 'FAILURE.json', dict(error=traceback.format_exc(), unix=time.time()))
        raise
