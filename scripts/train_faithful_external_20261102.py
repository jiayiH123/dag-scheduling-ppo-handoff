#!/usr/bin/env python3
"""One bound, exactly resumable paper-faithful training arm on original17K."""
import argparse
import contextlib
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daglab.faithful_external import *
from daglab.faithful_external_data import Inputs, config, summarize
from daglab.sft_three_panel import write_json
from daglab.structured_dataset import sha256


def binding(inputs):
    paths = [Path(__file__), ROOT / 'daglab/faithful_external.py', ROOT / 'daglab/faithful_external_data.py',
             ROOT / 'configs/faithful_external_20261102.json', ROOT / 'protocols/faithful_external_20261102.md',
             ROOT / 'daglab/external_baseline_sim.py', ROOT / 'vendor/prepared_sgs.py',
             ROOT / 'vendor/budget_core.py', ROOT / 'vendor/legacy_comm_core.py']
    paths += list((RUN / 'sources/cai/Graphormer/graphormer/modules').glob('*.py'))
    paths += [RUN / 'sources/cai/RL_models/utils.py', RUN / 'sources/giph/placement_rl/placement_agent.py',
              RUN / 'sources/giph/placement_rl/primative_nn.py', RUN / 'sources/giph/env/program.py']
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    audit = json.loads((RUN / 'INPUT_AUDIT.json').read_text())
    assert audit['train_key_sha256'] == hashlib.sha256(('\n'.join(inputs.keys) + '\n').encode()).hexdigest()
    return dict(code=sources, input_audit_sha256=sha256(RUN / 'INPUT_AUDIT.json'),
                engineering_sha256=sha256(RUN / 'engineering/CHECKS.json'),
                CAI_resume_sha256=sha256(RUN / 'engineering/CAI_RESUME.json'),
                GiPH_update_parity_sha256=sha256(RUN / 'engineering/GIPH_UPDATE_PARITY.json'),
                feature_range_sha256=sha256(RUN / 'engineering/FEATURE_RANGE_AUDIT.json'))


def save_checkpoint(path, state):
    temporary = path.with_suffix('.tmp.pt'); torch.save(state, temporary); temporary.replace(path)


def evaluate(model, inputs, method, out, unit, action_seed=101, tag=None):
    suffix = tag or f'{unit:07d}'
    target = out / f'val_{suffix}.json.gz'
    rng = torch.get_rng_state(); was_training = model.training; model.eval()
    rows = []; start = time.perf_counter(); method_variants = ['CAI'] if method == 'cai' else ['random', 'HEFT']
    partial = out / f'val_{suffix}.partial.jsonl'
    # A completed immutable evaluation can be reused after a crash.
    if target.exists():
        with gzip.open(target, 'rt') as stream:
            summary = json.load(stream)['summary']
        torch.set_rng_state(rng); model.train(was_training)
        return summary
    with partial.open('w') as stream, torch.no_grad():
        for index, case in enumerate(inputs.validation):
            results = {}
            for variant in method_variants:
                p = Problem(case['inst']) if method == 'cai' else GiPHProblem(case['inst'])
                assert p.schedule()['C'] == case['references']['HEFT']['C']
                before = model.forward_calls
                if method == 'cai':
                    data = cai_input(p); z, _ = model(data); state, _ = model.sample(p, data, z, False)
                    final = p.finish(state['mapping'], state['order']); work = dict(moves=0)
                else:
                    mapping = p.random_mapping(stable_seed(case['key'], 1700)) if variant == 'random' else list(p.schedule()['mapping'])
                    generator = torch.Generator().manual_seed(stable_seed(case['key'] + variant, action_seed))
                    best, _, work = giph_episode(model, p, mapping, generator=generator)
                    final = p.finish(best['mapping']); assert final['C'] == best['C']
                    if variant == 'HEFT':
                        assert final['C'] <= case['references']['HEFT']['C']
                results[variant] = dict(C=final['C'], cost=final['cost'], mapping=final['mapping'],
                                       order=final['order'], actor_calls=model.forward_calls-before, moves=work['moves'])
            record = {k: case[k] for k in ['key', 'base_id', 'n', 'family', 'references']}
            record.update(C=results[method_variants[0]]['C'], feasible=True, methods=results)
            rows.append(record); stream.write(json.dumps(record) + '\n')
            if (index + 1) % 50 == 0:
                stream.flush()
                write_json(out / 'progress.json', dict(stage='evaluation', unit=unit, cases=index+1, total=862,
                                                       seconds=time.perf_counter()-start))
    summaries = {}
    for variant in method_variants:
        converted = [dict(row, C=row['methods'][variant]['C']) for row in rows]
        summaries[variant] = summarize(converted)
    result = dict(unit=unit, action_seed=action_seed, variants=summaries,
                  primary=summaries[method_variants[0]]['base_macro'], seconds=time.perf_counter()-start)
    temporary = target.with_name(target.name + '.tmp')
    with gzip.open(temporary, 'wt') as stream:
        json.dump(dict(summary=result, rows=rows), stream)
    temporary.replace(target); partial.unlink()
    torch.set_rng_state(rng); model.train(was_training)
    return result


def run(args):
    torch.set_num_threads(1); torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    inputs = Inputs(); cfg = config(); out = RUN / args.arm; out.mkdir(exist_ok=True)
    lock = (out / 'worker.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    bound = binding(inputs)
    if (out / 'COMPLETE.json').exists():
        assert json.loads((out / 'COMPLETE.json').read_text())['binding'] == bound
        return
    initial_p = Problem(inputs.instance(inputs.keys[0]))
    model = CAI(cai_input(initial_p)['numeric'].shape[-1]) if args.method == 'cai' else GiPH()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    state = dict(binding=bound, method=args.method, seed=args.seed, lr=args.lr,
                 epoch=0, position=0, updates=0, presentations=0, moves=0, node_decisions=0,
                 curve=[], best=None, history={}, training_seconds=0., finished=False)
    checkpoint = out / 'resume.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=False)
        assert saved['binding'] == bound and saved['method'] == args.method and saved['seed'] == args.seed and saved['lr'] == args.lr
        model.load_state_dict(saved.pop('model')); optimizer.load_state_dict(saved.pop('optimizer'))
        torch.set_rng_state(saved.pop('torch_rng')); state = saved
        if (out / 'training.jsonl').exists():
            lines = [json.loads(line) for line in (out / 'training.jsonl').read_text().splitlines()]
            (out / 'training.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in lines if row['update'] <= state['updates']))
    else:
        save_checkpoint(out / 'initial.pt', dict(model=model.state_dict(), binding=bound))
        write_json(out / 'metadata.json', dict(binding=bound, method=args.method, lr=args.lr, seed=args.seed,
            parameters=sum(q.numel() for q in model.parameters()), train_configs=17261, VAL=862,
            network='author Graphormer 4x64/4head + 64-64-1' if args.method=='cai' else 'author DGL GiPHEmbedding10 / SoftmaxActor16',
            optimizer='Adam', gamma=None if args.method=='cai' else .97,
            no_teacher_supervision=True, original_episode_update=args.method=='giph'))

    def persist():
        save_checkpoint(checkpoint, dict(state, model=model.state_dict(), optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state()))

    def observe(unit):
        observation = evaluate(model, inputs, args.method, out, unit)
        state['curve'].append(observation)
        if state['best'] is None or observation['primary'] < state['best']['primary']:
            state['best'] = observation
            save_checkpoint(out / 'best.pt', dict(model=model.state_dict(), binding=bound, observation=observation))
        write_json(out / 'curve.json', state['curve']); persist()
        print('VAL', args.arm, unit, observation['primary'], flush=True)

    if not state['curve']:
        observe(0)
    epochs = cfg[args.method]['epochs']
    log = (out / 'training.jsonl').open('a')
    for epoch in range(state['epoch'], epochs):
        keys = inputs.epoch_keys(args.seed, epoch)
        batchsize = 16 if args.method == 'cai' else 1
        for start in range(state['position'], len(keys), batchsize):
            begin = time.perf_counter(); optimizer.zero_grad(set_to_none=True); model.train()
            batch = keys[start:start+batchsize]; sampled = []; loss_items = []; raws = []; moves = 0
            for key in batch:
                inst = inputs.instance(key)
                if args.method == 'cai':
                    p = Problem(inst); baseline = p.schedule(); data = cai_input(p); z, raw = model(data)
                    raws.append(raw.square().mean())
                    for _ in range(4):
                        schedule, lp = model.sample(p, data, z, True)
                        reward = torch.tensor([baseline['ends'][v] - schedule['ends'][v] for v in range(p.n)], dtype=torch.float32)
                        loss_items.append((lp, reward)); sampled.append(schedule['C']/baseline['C'])
                        state['node_decisions'] += p.n
                else:
                    p = GiPHProblem(inst)
                    if key in state['history']:
                        p.set_history(state['history'][key])
                    mapping = p.random_mapping(stable_seed(key, args.seed * 100 + epoch))
                    baseline_C = p.state(mapping)['C']
                    best, loss, work = giph_episode(model, p, mapping, train=True)
                    moves += work['moves']
                    if epochs > 1:
                        state['history'][key] = p.history()
                    sampled.append(best['C']/baseline_C)
            if args.method == 'cai':
                loss = cai_loss(loss_items, raws)
            if loss is not None:
                assert bool(torch.isfinite(loss)), 'Nonfinite loss'
                loss.backward()
                if args.method == 'cai':
                    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), .5))
                else:
                    norm = math.sqrt(sum(float(q.grad.square().sum()) for q in model.parameters() if q.grad is not None))
                assert math.isfinite(norm) and all(q.grad is None or bool(torch.isfinite(q.grad).all()) for q in model.parameters())
                optimizer.step()
            else:
                norm = 0.
            state['updates'] += 1; state['presentations'] += len(batch); state['moves'] += moves
            state['epoch'] = epoch; state['position'] = start + len(batch)
            state['training_seconds'] += time.perf_counter() - begin
            unit = state['updates']
            record = dict(update=unit, epoch=epoch+1, position=state['position'], presentations=state['presentations'],
                moves=state['moves'], node_decisions=state['node_decisions'], loss=float(loss.detach()) if loss is not None else 0.,
                gradient_norm=norm, sampled_ratio=sum(sampled)/len(sampled), training_seconds=state['training_seconds'])
            log.write(json.dumps(record) + '\n'); log.flush()
            if unit % (10 if args.method=='cai' else 25) == 0:
                persist(); write_json(out / 'progress.json', dict(stage='training', arm=args.arm, **record))
            interval = cfg[args.method]['eval_updates' if args.method=='cai' else 'eval_episodes']
            if unit % interval == 0:
                observe(unit)
        state['epoch'] = epoch + 1; state['position'] = 0
        if state['curve'][-1]['unit'] != state['updates']:
            observe(state['updates'])
        persist()
    log.close(); state['finished'] = True; persist()
    write_json(out / 'COMPLETE.json', dict(status='complete', binding=bound, best=state['best'],
        updates=state['updates'], presentations=state['presentations'], moves=state['moves'],
        node_decisions=state['node_decisions'], training_seconds=state['training_seconds'],
        parameters=sum(q.numel() for q in model.parameters())))
    print('COMPLETE', args.arm, state['best']['primary'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['cai', 'giph'], required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--arm', required=True)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException:
        directory = RUN / args.arm; directory.mkdir(exist_ok=True)
        write_json(directory / 'FAILURE.json', dict(error=traceback.format_exc(), unix=time.time()))
        raise
