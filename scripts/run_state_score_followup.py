#!/usr/bin/env python3
"""Locked equal-work timing and duration experiments; two bounded-memory runtimes."""
import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'data/structured_scheduling_v1'
SOURCE = ROOT / 'runs/state_score_factorial_20261027_r1'
PROBE = ROOT / 'runs/structured_gnn_failure_audit_20261026/payload.json'
MANIFEST = '8ef25bfef98975e8a79999c44df263f9650124abd4cac3773d4dce75a2b9a4ab'


def worker():
    from daglab.structured_unique import iter_unique_rows
    from daglab.structured_dataset import to_instance
    from daglab.structured_gnn_simulator import shuffled
    from daglab.reference import graph_features
    from daglab.module_lab.features import enrich
    from daglab.state_score import StateScoreDecoder

    val = list(iter_unique_rows(DATA, 'val'))
    probes = [x['row'] for x in json.loads(PROBE.read_text()) if x['row']['split'] == 'train']
    iterator = None
    cache = {}
    exposure = {}

    def record(row):
        exposure['configs'] += 1
        exposure['actions'] += len(row['supervision']['decisions'])
        exposure['digest'].update((row['key'] + '\n').encode())

    for line in sys.stdin:
        q = json.loads(line)
        cmd = q['cmd']
        if cmd == 'init':
            out = dict(VAL=len(val), TRAIN_probe=len(probes), manifest=hashlib.sha256((DATA / 'MANIFEST.json').read_bytes()).hexdigest())
        elif cmd == 'start':
            iterator = iter(shuffled(iter_unique_rows(DATA, 'train'), q['seed']))
            exposure = dict(configs=0, actions=0, digest=hashlib.sha256(), seed=q['seed'])
            for _ in range(q.get('skip', 0)):
                record(next(iterator))
            out = dict(skipped=q.get('skip', 0))
        elif cmd == 'exposure':
            out = {k: v for k, v in exposure.items() if k != 'digest'}
            out['ordered_keys_sha256'] = exposure['digest'].hexdigest()
        elif cmd in ['next', 'val', 'probe', 'checks']:
            cache.clear()
            rows = []
            if cmd == 'next':
                for _ in range(q['count']):
                    row = next(iterator, None)
                    if row is None:
                        break
                    rows.append(row)
                    record(row)
            elif cmd == 'checks':
                shapes = set()
                for row in iter_unique_rows(DATA, 'train'):
                    shape = row['input']['n'], row['input']['k']
                    if shape not in shapes:
                        shapes.add(shape)
                        rows.append(row)
                    if len(shapes) == 12:
                        break
            else:
                rows = (val if cmd == 'val' else probes)[q['offset']:q['offset'] + q['count']]
            out = []
            for row in rows:
                inst = to_instance(row)
                dec = StateScoreDecoder(inst)
                cache[row['key']] = dec
                graph, _ = graph_features(inst)
                graph = enrich(dict(instance=inst, graph=graph), dec)
                item = dict(id=row['key'], base_id=row['base_id'], n=inst['n'], graph=graph,
                            reference=row['quality'].get('reference_HEFT'), teacher=None)
                if 'supervision' in row:
                    target = row['supervision']
                    replay = dec.rollout([[0., 0., 0., 0.]] * (dec.n * dec.k), forced_mapping=target['task_to_pool'])
                    assert replay['makespan'] == target['makespan'] and replay['cost'] == target['cost']
                    for name, key in [('start', 'starts'), ('end', 'ends')]:
                        assert [t[name] for t in replay['schedule']['tasks']] == target[key]
                    for a, b in zip(replay['decisions'], target['decisions']):
                        for name in ['available', 'action', 'selected', 'offsets', 'budget_local']:
                            assert a[name] == b[name]
                    item.update(decisions=replay['decisions'], teacher=target['makespan'])
                out.append(item)
        elif cmd == 'roll':
            out = []
            for item in q['items']:
                result = cache[item['id']].rollout(item['x'], sample=item.get('sample', False), seed=item.get('seed', 0),
                    score_mode=item.get('score_mode','coefficients'),forced_mapping=item.get('forced_mapping'))
                value = dict(id=item['id'], arm=item['arm'], C=result['makespan'], cost=result['cost'])
                if q.get('keep_decisions'):
                    value['decisions'] = result['decisions']
                if q.get('keep_schedule'):
                    value['schedule'] = result['schedule']
                out.append(value)
        elif cmd == 'close':
            print('{}', flush=True)
            return
        else:
            raise ValueError(cmd)
        print(json.dumps(out, allow_nan=False), flush=True)


def train(args):
    import torch
    from daglab.module_lab.models import tensors
    from daglab.state_score_torch import StateActor, bc_loss, distributions

    torch.set_num_threads(4)
    random.seed(28)
    torch.manual_seed(28)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

    process = subprocess.Popen([args.sim_python, str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def call(**q):
        process.stdin.write(json.dumps(q, allow_nan=False) + '\n')
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError('physical worker terminated')
        return json.loads(line)

    if args.stage == 'timing':
        specs = {'gap_after270': ('original', 270, 'gap'),
                 'gap_after540': ('original', 540, 'gap'),
                 'resume_original540': ('original', 540, 'original')}
    else:
        audit = json.loads((args.timing_run / 'summary.json').read_text())
        assert audit['status'] == 'complete' and audit['resume_audit']['passed']
        specs = {mode + '_' + schedule: (mode, 1079, mode)
                 for mode in ['original', 'gap'] for schedule in ['constant', 'cosine']}

    models, opts, initial, hashes, reset = {}, {}, {}, {}, {}
    for key, (source_mode, boundary, mode) in specs.items():
        name = f'scratch_{source_mode}_' + ('last' if boundary == 1079 else f'stage{boundary}') + '.pt'
        path = SOURCE / name
        source = torch.load(path, map_location='cpu', weights_only=False)
        assert source['step'] == boundary and source['mode'] == source_mode
        m = StateActor(mode)
        m.load_state_dict(source['model'], strict=True)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        opt.load_state_dict(source['optimizer'])
        count = 0
        if args.stage == 'timing' and mode == 'gap':
            assert all(bool((p == 0).all()) for p in m.extra[-1].parameters())
            for p in m.extra.parameters():
                count += int(p in opt.state)
                opt.state.pop(p, None)
        models[key], opts[key], initial[key] = m, opt, boundary
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        reset[key] = count

    # The policy has no stochastic forward operations (also checked below).
    # Restore the archived RNG nevertheless, so saved continuation state is useful
    # if stochastic operations are introduced in a separately versioned experiment.
    torch.set_rng_state(source['torch_rng'])
    random.setstate(source['python_rng'])

    metadata = dict(stage=args.stage, protocol='protocols/state_score_followup_20261027.md',
                    code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    source_sha256=hashes, initial_global_updates=initial, aux_optimizer_states_reset=reset,
                    parameter_counts={k: sum(p.numel() for p in m.parameters()) for k, m in models.items()},
                    data_manifest_sha256=MANIFEST, batch=16, TEST_evaluated=False, RL=False,
                    torch_threads=4, source_rng_restored=True,
                    evaluation='known862VAL + fixed128TRAIN', optimizer='saved Adam; no weight decay; clip1')
    save('metadata.json', metadata)
    best, history, exposure = {}, [], []
    presentations = {k: 0 for k in models}
    updates = {k: 0 for k in models}
    active, started = 0., time.perf_counter()
    source_evals = {}

    def checkpoint(key, label, stage_step):
        global_step = max(stage_step, initial[key]) if args.stage == 'timing' else 1079 + stage_step
        torch.save(dict(model=models[key].state_dict(), optimizer=opts[key].state_dict(), mode=models[key].mode,
                        stage=args.stage, stage_step=stage_step, global_step=global_step,
                        stage_presentations=presentations[key], stage_updates=updates[key],
                        torch_rng=torch.get_rng_state(), python_rng=random.getstate(), source_sha256=hashes[key]),
                   out / f'{key}_{label}.pt')

    def source_values(step, source_mode):
        if step not in source_evals:
            source_evals[step] = json.loads((SOURCE / f'eval_{step:04d}.json').read_text())
        return source_evals[step]['VAL']['scratch_' + source_mode]

    def evaluate(step, boundary=False):
        begin = time.perf_counter()
        values = {k: [] for k in models}
        probes = {k: [] for k in models}
        for m in models.values():
            m.eval()
        with torch.no_grad():
            for cmd, count, target in [('val', 862, values), ('probe', 128, probes)]:
                for offset in range(0, count, 12):
                    rows = call(cmd=cmd, offset=offset, count=12)
                    items, stats = [], {}
                    for row in rows:
                        g = tensors(row['graph'], 'cpu')
                        for key, m in models.items():
                            a, b = m.forward_calls, m.base.forward_calls
                            x = m(g)
                            assert m.forward_calls - a == m.base.forward_calls - b == 1
                            items.append(dict(id=row['id'], arm=key, x=x.tolist()))
                            if cmd == 'probe':
                                logs, _ = distributions(x, row['decisions'])
                                labels = torch.tensor([d['selected'] for d in row['decisions']])
                                mask = torch.tensor([len(d['available']) > 1 for d in row['decisions']])
                                nll = -logs.gather(1, labels[:, None]).squeeze(1)
                                stats[key, row['id']] = dict(NLL=float(nll.mean()), active=int(mask.sum()),
                                    active_nll=float(nll[mask].sum()), correct=int(((logs.argmax(-1) == labels) & mask).sum()))
                    lookup = {r['id']: r for r in rows}
                    for value in call(cmd='roll', items=items):
                        row, key = lookup[value['id']], value['arm']
                        target[key].append(dict(id=row['id'], base=row['base_id'], n=row['n'], C=value['C'],
                                               HEFT=row['reference'], teacher=row['teacher'], **stats.get((key, row['id']), {})))
        metrics = {}
        for key in models:
            ratios = sorted(v['C'] / v['HEFT'] for v in values[key])
            ds = probes[key]
            na = sum(v['active'] for v in ds)
            metrics[key] = dict(C_HEFT=sum(ratios) / len(ratios), HEFT_losses=sum(z > 1 + 1e-12 for z in ratios),
                                p95=ratios[int(.95 * (len(ratios) - 1))],
                                TRAIN_NLL=sum(v['NLL'] for v in ds) / len(ds),
                                TRAIN_active_NLL=sum(v['active_nll'] for v in ds) / na,
                                TRAIN_active_top1=sum(v['correct'] for v in ds) / na,
                                TRAIN_C_teacher=sum(v['C'] / v['teacher'] for v in ds) / len(ds))
            if boundary:
                expected = {v['id']: v['C'] for v in source_values(initial[key], specs[key][0])}
                assert all(v['C'] == expected[v['id']] for v in values[key]), key
            if key not in best or metrics[key]['C_HEFT'] < best[key]['C_HEFT']:
                best[key] = dict(stage_step=step, global_step=max(step, initial[key]) if args.stage == 'timing' else 1079 + step,
                                 **metrics[key])
                checkpoint(key, 'best', step)
            checkpoint(key, 'last', step)
            models[key].train()
        save(f'eval_{step:04d}.json', dict(stage_step=step, initial_boundary=boundary, metrics=metrics,
                                        VAL=values, TRAIN_probe=probes, seconds=time.perf_counter() - begin))
        history.append(dict(stage_step=step, metrics=metrics))
        save('progress.json', dict(best=best, history=history, updates=updates, presentations=presentations))
        print('VAL', step, json.dumps({k: round(v['C_HEFT'], 7) for k, v in metrics.items()}), flush=True)
        return values

    def replay_audit(step):
        key = 'resume_original540'
        name = 'last' if step == 1079 else f'stage{step}'
        ref = torch.load(SOURCE / f'scratch_original_{name}.pt', map_location='cpu', weights_only=False)
        full, meaningful, key_bias = 0., 0., 0.
        for k, value in models[key].state_dict().items():
            difference = (value - ref['model'][k]).abs()
            full = max(full, float(difference.max()))
            if k.endswith('in_proj_bias') and ('task_attn' in k or 'pool_attn' in k):
                q, kb, v = difference.chunk(3)
                meaningful = max(meaningful, float(q.max()), float(v.max()))
                key_bias = max(key_bias, float(kb.max()))
            else:
                meaningful = max(meaningful, float(difference.max()))
        assert meaningful <= 1e-5, (step, full, meaningful)
        rng = torch.get_rng_state().clone()
        reference = StateActor('original')
        reference.load_state_dict(ref['model'], strict=True)
        reference.eval()
        torch.set_rng_state(rng)
        models[key].eval()
        max_output_error = 0.
        observed = []
        with torch.no_grad():
            for offset in range(0, 862, 12):
                rows = call(cmd='val', offset=offset, count=12)
                items = []
                for row in rows:
                    g = tensors(row['graph'], 'cpu')
                    x, y = models[key](g), reference(g)
                    max_output_error = max(max_output_error, float((x-y).abs().max()))
                    items.append(dict(id=row['id'], arm=key, x=x.tolist()))
                observed.extend(call(cmd='roll', items=items))
        expected = {v['id']: v['C'] for v in source_values(step, 'original')}
        mismatches = [v['id'] for v in observed if v['C'] != expected[v['id']]]
        result = dict(step=step, full_weight_error=full, key_bias_error=key_bias,
                      meaningful_weight_error=meaningful, max_output_error=max_output_error,
                      makespan_mismatches=mismatches, exact_makespans=len(observed)-len(mismatches))
        save(f'resume_audit_{step:04d}.json', result)
        assert max_output_error <= 5e-5 and not mismatches, result
        models[key].train()
        return result

    try:
        identity = call(cmd='init')
        assert identity == dict(VAL=862, TRAIN_probe=128, manifest=MANIFEST)
        save('identity.json', identity)
        checks = []
        for row in call(cmd='checks'):
            g = tensors(row['graph'], 'cpu')
            items = []
            for key, m in models.items():
                m.zero_grad(set_to_none=True)
                rng = torch.get_rng_state().clone()
                x = m(g)
                assert torch.equal(torch.get_rng_state(), rng)
                loss = bc_loss(x, row['decisions'])
                loss.backward()
                assert bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in m.parameters())
                items.append(dict(id=row['id'], arm=key, x=x.detach().tolist(), sample=True, seed=28))
            for value in call(cmd='roll', items=items, keep_decisions=True):
                logs, _ = distributions(models[value['arm']](g).detach(), value['decisions'])
                labels = torch.tensor([d['selected'] for d in value['decisions']])
                selected = logs.gather(1, labels[:, None]).squeeze(1)
                reference = torch.tensor([d['selected_log_probability'] for d in value['decisions']])
                error = float((selected - reference).abs().max())
                assert error < 5e-5
                checks.append(dict(id=row['id'], arm=value['arm'], probability_error=error))
        for m in models.values():
            m.zero_grad(set_to_none=True)
        save('checks.json', dict(probability_checks=checks, stochastic_forward=False, optimizer_steps=0))
        evaluate(0, boundary=True)
        audits = []
        log = (out / 'curves.jsonl').open('w')
        if args.stage == 'timing':
            epochs = [(28, 270, 1)]
        else:
            epochs = [(29, 0, 2), (30, 0, 3)]
        for seed, skip_updates, epoch in epochs:
            call(cmd='start', seed=seed, skip=skip_updates * 16)
            local_step = skip_updates
            while True:
                begin = time.perf_counter()
                rows = call(cmd='next', count=16)
                if not rows:
                    break
                local_step += 1
                step = local_step if args.stage == 'timing' else (epoch - 2) * 1079 + local_step
                enabled = [k for k in models if args.stage != 'timing' or step > initial[k]]
                for k in enabled:
                    opts[k].zero_grad(set_to_none=True)
                    lr = 1e-3
                    if args.stage == 'duration' and k.endswith('cosine'):
                        lr = 1e-4 + .5 * 9e-4 * (1 + math.cos(math.pi * (step - 1) / 2157))
                    for group in opts[k].param_groups:
                        group['lr'] = lr
                losses = {k: 0. for k in enabled}
                for row in rows:
                    g = tensors(row['graph'], 'cpu')
                    for key in enabled:
                        loss = bc_loss(models[key](g), row['decisions'])
                        assert bool(torch.isfinite(loss))
                        (loss / len(rows)).backward()
                        losses[key] += float(loss.detach()) / len(rows)
                norms = {}
                for key in enabled:
                    norms[key] = float(torch.nn.utils.clip_grad_norm_(models[key].parameters(), 1.))
                    assert math.isfinite(norms[key])
                    opts[key].step()
                    updates[key] += 1
                    presentations[key] += len(rows)
                active += time.perf_counter() - begin
                log.write(json.dumps(dict(step=step, epoch=epoch, local_step=local_step, losses=losses, norms=norms,
                                          lr={k: opts[k].param_groups[0]['lr'] for k in enabled}, active_seconds=active)) + '\n')
                log.flush()
                nodes = [540, 810, 1079] if args.stage == 'timing' else [540, 1079, 1619, 2158]
                if step in nodes:
                    if args.stage == 'timing' and step in [810, 1079]:
                        audits.append(replay_audit(step))
                    values = evaluate(step)
                    if args.stage == 'timing' and step == 1079:
                        expected = {v['id']: v['C'] for v in source_values(1079, 'original')}
                        assert all(v['C'] == expected[v['id']] for v in values['resume_original540'])
                if step % 100 == 0:
                    print('TRAIN', step, round(active, 2), flush=True)
            assert local_step == 1079
            exp = call(cmd='exposure')
            assert exp['configs'] == 17261 and exp['actions'] == 1122860
            if args.stage == 'timing':
                assert exp['ordered_keys_sha256'] == 'd359b4bd1aae2fcca1c14317c09f8b674bb7ef32ff5638c05b96b95f83ceaecf'
            exposure.append(exp)
        log.close()
        expected_updates = dict(gap_after270=809, gap_after540=539, resume_original540=539) if args.stage == 'timing' else {k: 2158 for k in models}
        assert updates == expected_updates
        if args.stage == 'duration':
            assert set(presentations.values()) == {34522}
        save('summary.json', dict(status='complete', stage=args.stage, best=best, history=history,
                                 exposure=exposure, updates=updates, presentations=presentations,
                                 resume_audit=dict(applicable=args.stage == 'timing',
                                                   passed=True if args.stage == 'timing' else None,
                                                   checkpoints=audits),
                                 active_seconds_joint=active, total_seconds=time.perf_counter() - started,
                                 TEST_evaluated=False))
        print('COMPLETE', json.dumps(best), flush=True)
    finally:
        if process.poll() is None:
            call(cmd='close')
            process.wait(timeout=20)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--worker', action='store_true')
    p.add_argument('--stage', choices=['timing', 'duration'])
    p.add_argument('--sim-python')
    p.add_argument('--output', type=Path)
    p.add_argument('--timing-run', type=Path)
    args = p.parse_args()
    worker() if args.worker else train(args)
