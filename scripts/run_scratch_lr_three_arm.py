#!/usr/bin/env python3
"""Matched full-TRAIN BC branches; fixed two-pass budget and automatic curves."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'runs/scratch_both_seven_epoch_20261028/best.pt'
OUT = ROOT / 'runs/scratch_lr_three_arm_20261028'
PROTOCOL = 'protocols/scratch_lr_three_arm_20261028.md'
MANIFEST = '8ef25bfef98975e8a79999c44df263f9650124abd4cac3773d4dce75a2b9a4ab'
ARMS = ('A_constant_1e3', 'B_constant_3e4', 'C_cosine_3e4_3e5')
TOTAL = 2158


def rate(arm, update):
    assert arm in ARMS and 1 <= update <= TOTAL
    if arm == ARMS[0]:
        return 1e-3
    if arm == ARMS[1]:
        return 3e-4
    return 3e-5 + .5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * (update - 1) / (TOTAL - 1)))


def save_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def same(a, b, torch):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], torch) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y, torch) for x, y in zip(a, b))
    return a == b


def train(out):
    import torch
    from daglab.module_lab.models import tensors
    from daglab.state_score_torch import StateActor, bc_loss, distributions

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out.mkdir(exist_ok=False)
    save = lambda name, value: save_json(out / name, value)
    started = time.perf_counter()
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    assert (source['step'], source['presentations'], source['actions'], source['mode']) == (5500, 87985, 5722820, 'both')
    models, opts = {}, {}
    for arm in ARMS:
        model = StateActor('both')
        model.load_state_dict(source['model'], strict=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt.load_state_dict(copy.deepcopy(source['optimizer']))
        assert same(opt.state_dict(), source['optimizer'], torch)
        assert all(float(v['step']) == 5500 for v in opt.state.values())
        assert all(g['betas'] == (.9, .999) and g['eps'] == 1e-8 and g['weight_decay'] == 0 for g in opt.param_groups)
        for group in opt.param_groups:
            group['lr'] = rate(arm, 1)
        models[arm], opts[arm] = model, opt
    # Reject cross-arm parameter or moment aliasing before any optimizer step.
    storages = []
    for arm in ARMS:
        pointers = {p.data_ptr() for p in models[arm].parameters()}
        for value in opts[arm].state.values():
            pointers.update(v.data_ptr() for v in value.values() if torch.is_tensor(v))
        assert all(pointers.isdisjoint(previous) for previous in storages)
        storages.append(pointers)
        assert same(models[arm].state_dict(), source['model'], torch)
    torch.set_rng_state(source['torch_rng'])
    random.setstate(source['python_rng'])
    data = ROOT / 'data/structured_scheduling_v1'
    assert hashlib.sha256((data / 'MANIFEST.json').read_bytes()).hexdigest() == MANIFEST
    shard_hashes = []
    for shard in json.loads((data / 'MANIFEST.json').read_text())['shards']:
        if shard['split'] in ('train', 'val'):
            digest = hashlib.sha256((data / shard['path']).read_bytes()).hexdigest()
            assert digest == shard['sha256']
            shard_hashes.append(dict(path=shard['path'], sha256=digest))
    assert len(shard_hashes) == 84
    save('metadata.json', dict(protocol=PROTOCOL, code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source=str(SOURCE.relative_to(ROOT)), source_sha256=source_sha, source_step=5500,
        source_presentations=87985, source_actions=5722820, source_historical_passes=87985/17261,
        manifest_sha256=MANIFEST, arms=list(ARMS), initial_lr={k: rate(k, 1) for k in ARMS},
        final_lr={k: rate(k, TOTAL) for k in ARMS}, schedule_formula='protocol; cosine progress=(new_update-1)/2157',
        additional_epochs=2, data_seeds=[35, 36], data_order='two fresh complete buffer128-shuffled traversals; same batches all arms',
        branch_not_uninterrupted_epoch_continuation=True, batch=16, updates_per_pass=1079,
        additional_updates_per_arm=TOTAL, additional_presentations_per_arm=34522, additional_actions_per_arm=2245720,
        optimizer='Adam', betas=[.9, .999], eps=1e-8, weight_decay=0, gradient_clip=1,
        source_Adam_moments_and_steps_restored=True, parameters=32612, threads=1, deterministic=True,
        evaluate_every_new_updates=50, extra_eval_updates=[0, 1079, 2158],
        VAL=862, TRAIN_probe=128, TRAIN_unique=17261, RL=False, TEST_evaluated=False))
    process = subprocess.Popen(['/opt/miniconda3/envs/dag-research/bin/python', str(Path(__file__).resolve()), '--worker'],
        cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def call(**query):
        process.stdin.write(json.dumps(query, allow_nan=False) + '\n')
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError('physical worker terminated')
        return json.loads(line)

    def forward(model, row):
        before = model.forward_calls, model.base.forward_calls
        x = model(tensors(row['graph'], 'cpu'))
        assert (model.forward_calls - before[0], model.base.forward_calls - before[1]) == (1, 1)
        return x

    update = 0
    pass_index = 0
    pass_step = 0
    added_presentations = 0
    added_actions = 0
    best, history, exposures = {}, [], []
    active_seconds = 0.
    evaluation_seconds = 0.
    valmeta = {r['id']: r for r in csv.DictReader((ROOT / 'runs/policy_failure_localization_20261027/records.csv').open())}

    def checkpoint(arm, tag):
        torch.save(dict(model=models[arm].state_dict(), optimizer=opts[arm].state_dict(), mode='both',
            arm=arm, source=str(SOURCE.relative_to(ROOT)), source_sha256=source_sha,
            step=5500+update, new_update=update, pass_index=pass_index, pass_step=pass_step,
            presentations=87985+added_presentations, actions=5722820+added_actions,
            torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
            data_seed=34+pass_index if pass_index else None,
            data_skip=min(pass_step*16, 17261)), out / f'{arm}_{tag}.pt')

    def plot():
        env = dict(os.environ, MPLCONFIGDIR='/private/tmp/dag-policy-matplotlib')
        subprocess.run(['/opt/miniconda3/envs/dag-research/bin/python', str(ROOT/'scripts/plot_scratch_lr.py'), '--run', str(out)],
            cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)

    def evaluate():
        nonlocal evaluation_seconds
        begin = time.perf_counter()
        values, probes = {k: [] for k in ARMS}, {k: [] for k in ARMS}
        for model in models.values():
            model.eval()
        with torch.no_grad():
            for cmd, count, target in [('val', 862, values), ('probe', 128, probes)]:
                for offset in range(0, count, 16):
                    rows = call(cmd=cmd, offset=offset, count=16)
                    items, stats = [], {}
                    for row in rows:
                        for arm, model in models.items():
                            x = forward(model, row)
                            items.append(dict(id=row['id'], arm=arm, x=x.tolist()))
                            if cmd == 'probe':
                                lp, _ = distributions(x, row['decisions'])
                                labels = torch.tensor([d['selected'] for d in row['decisions']])
                                mask = torch.tensor([len(d['available']) > 1 for d in row['decisions']])
                                nll = -lp.gather(1, labels[:, None]).squeeze(1)
                                stats[arm, row['id']] = dict(NLL=float(nll.mean()), active=int(mask.sum()),
                                    active_NLL_sum=float(nll[mask].sum()), correct=int(((lp.argmax(-1) == labels) & mask).sum()))
                    lookup = {r['id']: r for r in rows}
                    for result in call(cmd='roll', items=items):
                        row, arm = lookup[result['id']], result['arm']
                        target[arm].append(dict(id=row['id'], base=row['base_id'], n=row['n'], C=result['C'],
                            HEFT=row['reference'], teacher=row['teacher'], **stats.get((arm, row['id']), {})))
        metrics = {}
        for arm in ARMS:
            vals, ds = values[arm], probes[arm]
            ratios = sorted(r['C']/r['HEFT'] for r in vals)
            nactive = sum(r['active'] for r in ds)
            def avg(pred):
                selected = [r for r in vals if pred(valmeta[r['id']])]
                return sum(r['C']/r['HEFT'] for r in selected)/len(selected)
            metrics[arm] = dict(C_HEFT=sum(ratios)/862, HEFT_losses=sum(r > 1+1e-12 for r in ratios),
                p95=ratios[int(.95*861)], groups=dict(layered=avg(lambda r: r['family']=='layered'),
                CCR5=avg(lambda r: float(r['target_ccr'])==5),
                tight=avg(lambda r: r['eta'] not in ['', 'None'] and float(r['eta'])<=.25)),
                TRAIN_NLL=sum(r['NLL'] for r in ds)/128,
                TRAIN_active_NLL=sum(r['active_NLL_sum'] for r in ds)/nactive,
                TRAIN_active_top1=sum(r['correct'] for r in ds)/nactive,
                TRAIN_C_teacher=sum(r['C']/r['teacher'] for r in ds)/128)
            if update == 0:
                archived = json.loads((SOURCE.parent/'eval_5500.json').read_text())
                expected = {r['id']: r for r in archived['VAL']['scratch_both']}
                assert all(r['C']==expected[r['id']]['C'] and r['HEFT']==expected[r['id']]['HEFT'] for r in vals)
                assert abs(metrics[arm]['C_HEFT']-archived['metrics']['scratch_both']['C_HEFT'])<1e-12
            if arm not in best or metrics[arm]['C_HEFT'] < best[arm]['C_HEFT']:
                best[arm] = dict(new_update=update, step=5500+update, pass_index=pass_index, pass_step=pass_step, **metrics[arm])
                checkpoint(arm, 'best')
            checkpoint(arm, f'update{update:04d}')
            checkpoint(arm, 'last')
            models[arm].train()
        elapsed = time.perf_counter()-begin
        evaluation_seconds += elapsed
        save(f'eval_{update:04d}.json', dict(new_update=update, step=5500+update, metrics=metrics, VAL=values, TRAIN_probe=probes))
        history.append(dict(new_update=update, step=5500+update, pass_index=pass_index, pass_step=pass_step, metrics=metrics))
        save('progress.json', dict(status='running', best=best, history=history, active_seconds=active_seconds,
            evaluation_seconds=evaluation_seconds, total_seconds=time.perf_counter()-started))
        if update == 0:
            save('resume_audit.json', dict(source_VAL_exact_all_arms=True, source_model_exact=True,
                source_Adam_moments_steps_exact=True, no_cross_arm_storage_alias=True,
                source_sha256=source_sha, C_HEFT={k: metrics[k]['C_HEFT'] for k in ARMS}))
        plot()
        print('VAL', update, 'global', 5500+update, {k: round(v['C_HEFT'], 8) for k, v in metrics.items()}, flush=True)

    def optimize(model, opt, rows):
        opt.zero_grad(set_to_none=True)
        mean = 0.
        for row in rows:
            loss = bc_loss(forward(model, row), row['decisions'])
            assert bool(torch.isfinite(loss))
            (loss/len(rows)).backward()
            mean += float(loss.detach())/len(rows)
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
        assert math.isfinite(norm)
        opt.step()
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
        return mean, norm

    try:
        assert call(cmd='init') == dict(VAL=862, TRAIN_probe=128, manifest=MANIFEST)
        checks = []
        for row in call(cmd='checks'):
            items, xs = [], {}
            for arm, model in models.items():
                model.zero_grad(set_to_none=True)
                rng = torch.get_rng_state().clone()
                x = forward(model, row)
                assert torch.equal(rng, torch.get_rng_state())
                loss = bc_loss(x, row['decisions'])
                loss.backward()
                assert bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                model.zero_grad(set_to_none=True)
                xs[arm] = x.detach()
                items.append(dict(id=row['id'], arm=arm, x=x.detach().tolist(), sample=True, seed=28))
            for result in call(cmd='roll', items=items, keep_decisions=True):
                ds = result['decisions']
                lp, _ = distributions(xs[result['arm']], ds)
                actual = lp.gather(1, torch.tensor([d['selected'] for d in ds])[:, None]).squeeze(1)
                err = float((actual-torch.tensor([d['selected_log_probability'] for d in ds])).abs().max())
                assert err < 5e-5
                checks.append(dict(id=row['id'], arm=result['arm'], probability_error=err))
        save('preflight.json', dict(data_hashes=shard_hashes, likelihood_checks=checks, gradients_finite=True,
            one_forward=True, stochastic_forward=False, no_cross_arm_storage_alias=True))
        evaluate()
        with (out/'training_curves.jsonl').open('w') as log:
            for pass_index, seed in enumerate([35, 36], 1):
                pass_step = 0
                call(cmd='start', seed=seed)
                while True:
                    begin = time.perf_counter()
                    rows = call(cmd='next', count=16)
                    if not rows:
                        break
                    update += 1
                    pass_step += 1
                    added_presentations += len(rows)
                    added_actions += sum(len(r['decisions']) for r in rows)
                    record = {}
                    for arm, model in models.items():
                        for group in opts[arm].param_groups:
                            group['lr'] = rate(arm, update)
                        loss, norm = optimize(model, opts[arm], rows)
                        record[arm] = dict(NLL=loss, gradient_norm_before_clip=norm, LR=[g['lr'] for g in opts[arm].param_groups])
                        if update == 1:
                            checkpoint(arm, 'first_update')
                    active_seconds += time.perf_counter()-begin
                    keys = [r['id'] for r in rows]
                    log.write(json.dumps(dict(new_update=update, step=5500+update, pass_index=pass_index,
                        pass_step=pass_step, batch=len(rows), added_presentations=added_presentations,
                        added_actions=added_actions, batch_keys_sha256=hashlib.sha256(('\n'.join(keys)+'\n').encode()).hexdigest(),
                        arms=record, active_seconds=active_seconds), allow_nan=False)+'\n')
                    log.flush()
                    if update == 1:
                        replay_checks = []
                        for arm in ARMS:
                            model = StateActor('both')
                            model.load_state_dict(source['model'])
                            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                            opt.load_state_dict(copy.deepcopy(source['optimizer']))
                            for group in opt.param_groups:
                                group['lr'] = rate(arm, 1)
                            loss, _ = optimize(model, opt, rows)
                            assert loss == record[arm]['NLL']
                            assert same(model.state_dict(), models[arm].state_dict(), torch)
                            assert same(opt.state_dict(), opts[arm].state_dict(), torch)
                            replay_checks.append(dict(arm=arm, LR=rate(arm, 1), model_and_Adam_exact=True))
                        torch.set_rng_state(source['torch_rng'])
                        random.setstate(source['python_rng'])
                        save('first_batch.json', dict(keys=keys, records=record))
                        save('startup_verified.json', dict(actual_update=1, global_step=5501,
                            source_VAL_exact=True, independently_replayed_first_updates=replay_checks,
                            parameters_finite=True, curves_exist=(out/'training_curves.png').exists()))
                        print('STARTUP VERIFIED: all three arms independently replayed update5501; actual LR',
                            {k: record[k]['LR'] for k in ARMS}, flush=True)
                    if update % 50 == 0 or pass_step == 1079:
                        evaluate()
                exposure = call(cmd='exposure')
                assert pass_step == 1079 and exposure['configs']==17261 and exposure['actions']==1122860
                exposures.append(exposure)
                save('exposure.json', exposures)
        assert (update, added_presentations, added_actions) == (2158, 34522, 2245720)
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_sha
        assert len(history) == 46
        for arm in ARMS:
            assert all(float(v['step'])==7658 for v in opts[arm].state.values())
        save('summary.json', dict(status='complete', best=best, history=history,
            added_updates_per_arm=update, cumulative_updates_per_arm=5500+update,
            added_presentations_per_arm=added_presentations, cumulative_presentations_per_arm=87985+added_presentations,
            added_actions_per_arm=added_actions, cumulative_actions_per_arm=5722820+added_actions,
            exposure_by_pass=exposures, active_seconds=active_seconds, evaluation_seconds=evaluation_seconds,
            total_seconds=time.perf_counter()-started, source_unchanged=True, TEST_evaluated=False,
            final_independent_result_audit='pending; do not promote automatically'))
        plot()
        print('COMPLETE', best, flush=True)
    except BaseException:
        save('failure.json', dict(new_update=update, traceback=traceback.format_exc()))
        raise
    finally:
        if process.poll() is None:
            try:
                call(cmd='close')
                process.wait(timeout=30)
            except Exception:
                process.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    if args.worker:
        from scripts.run_state_score_followup import worker
        worker()
    else:
        train(args.output.resolve())
