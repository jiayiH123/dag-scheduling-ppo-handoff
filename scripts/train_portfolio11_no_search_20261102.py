#!/usr/bin/env python3
"""Bounded original17K/Portfolio11-no-search experiment; original VAL862 only."""
import argparse
import collections
import fcntl
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from daglab.sft_three_panel import connect, unpack, to_instance, write_json
from daglab.structured_dataset import sha256
from daglab.offline_data import digest
from daglab.epoch_training_plan import make_epoch_plan
from scripts.train_sft_three_panel import ThreePanelTrial
from scripts.check_direct_score import Bridge

CONFIG = ROOT / 'configs/sft_portfolio11_no_search_20261102.json'
SCRIPT = Path(__file__).resolve()


def config():
    return json.loads(CONFIG.read_text())


class Store:
    def __init__(self):
        cfg = config()
        self.train = connect(ROOT / cfg['teacher_cache'] / 'public.sqlite3')
        self.extra = connect(ROOT / cfg['validation_cache'])
        self.panel_keys = {'synthetic_val862': [r[0] for r in self.extra.execute(
            'SELECT key FROM records WHERE panel="synthetic_val862" ORDER BY key')]}
        assert len(self.panel_keys['synthetic_val862']) == 862

    def row(self, key):
        if key in self.panel_keys['synthetic_val862']:
            r = self.extra.execute('SELECT payload FROM records WHERE key=?', (key,)).fetchone()
        else:
            r = self.train.execute('SELECT payload FROM records WHERE key=? AND source="synthetic" AND role="train"', (key,)).fetchone()
        if r is None:
            raise KeyError(key)
        return unpack(r[0])

    def raw(self, key):
        if key in self.panel_keys['synthetic_val862']:
            r = self.extra.execute('SELECT raw FROM records WHERE key=?', (key,)).fetchone()
        else:
            r = self.train.execute('SELECT raw FROM records WHERE key=? AND source="synthetic" AND role="train"', (key,)).fetchone()
        if r is None:
            raise KeyError(key)
        return unpack(r[0])

    def close(self):
        self.train.close()
        self.extra.close()


class Trial(ThreePanelTrial):
    def __init__(self, resume=False, out_override=None):
        import torch
        from daglab.state_score_torch import StateActor, bc_loss, distributions
        from daglab.module_lab.models import tensors
        self.cfg = config()
        self.cfg.update(lr=self.cfg['optimizer']['lr'], sources=['synthetic'])
        self.epochs = self.cfg['epochs']
        self.arm = dict(id='portfolio11_no_search_original17K', sources=['synthetic'], data_stage='base')
        self.torch, self.bc_loss, self.distributions, self.tensors = torch, bc_loss, distributions, tensors
        self.out = Path(out_override) if out_override else ROOT / self.cfg['run']
        self.out.mkdir(parents=True, exist_ok=True)
        self.data = ROOT / self.cfg['teacher_cache']
        self.store = Store()
        self.con = self.store.train
        self.batches, self.source_for_key, self.plan_sha = make_epoch_plan(self.con, ['synthetic'], self.cfg['seed'], self.epochs, self.cfg['batch_size'])
        self.corpus_size = len(self.source_for_key)
        self.updates_per_epoch = math.ceil(self.corpus_size / self.cfg['batch_size'])
        self.target = len(self.batches)
        assert (self.corpus_size, self.updates_per_epoch, self.target) == (17261, 1079, 7553)
        self.observation_steps = {0, self.target, *range(100, self.target + 1, 100)}
        self.diagnostic_steps = set()
        self.observation = dict(eval_sources=['synthetic_val862'], eval_every=100)
        paths = dict(config=CONFIG, teacher_manifest=self.data/'MANIFEST.json',
            teacher_database=self.data/'public.sqlite3', validation_database=ROOT/self.cfg['validation_cache'],
            implementation=SCRIPT, physical_training_helpers=ROOT/'scripts/train_synthetic_transfer.py',
            epoch_update=ROOT/'scripts/train_synthetic_transfer_epochs.py',
            inherited_trainer=ROOT/'scripts/train_sft_three_panel.py',
            helpers=ROOT/'daglab/sft_three_panel.py', sampler=ROOT/'daglab/epoch_training_plan.py',
            model=ROOT/'daglab/state_score_torch.py', decoder=ROOT/'daglab/state_score.py',
            plotting=ROOT/'figures/gen_fig_teacher_search_ablation_20261102.py')
        self.binding = {k: sha256(p) for k, p in paths.items()}
        manifest = json.loads((self.data/'MANIFEST.json').read_text())
        assert manifest['status'] == 'complete' and manifest['database_sha256'] == self.binding['teacher_database']
        audit = json.loads((ROOT/self.cfg['run']/'INPUT_AUDIT.json').read_text())
        assert audit['status'] == 'PASS' and audit['teacher_database_sha256'] == self.binding['teacher_database']
        assert audit['validation_database_sha256'] == self.binding['validation_database']
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        random.seed(self.cfg['seed'])
        torch.manual_seed(self.cfg['seed'])
        self.model = StateActor('both')
        opt = self.cfg['optimizer']
        self.opt = torch.optim.Adam(self.model.parameters(), lr=opt['lr'], betas=tuple(opt['betas']), eps=opt['eps'], weight_decay=opt['weight_decay'])
        assert sum(p.numel() for p in self.model.parameters()) == 32612
        old = json.loads((ROOT/'runs/synthetic_transfer_20261031/S0_original/metadata.json').read_text())
        init_sha = digest({k:v.tolist() for k,v in self.model.state_dict().items()})
        assert self.plan_sha == old['plan_sha256'] and init_sha == old['initial_parameters_sha256']
        self.worker = Bridge('scripts/train_portfolio11_no_search_20261102.py', '--worker')
        self.analysis_metadata = {}
        self.step = self.actions = 0
        self.exposures = collections.Counter()
        self.unique = collections.defaultdict(set)
        self.training_seconds = self.eval_seconds = 0.
        self.history = []
        self.best = None
        self.best_by_panel = {}
        self.started = time.perf_counter()
        self.resumed_from = None
        if resume:
            self.restore()
            self.best_by_panel = torch.load(self.out/'last.pt', map_location='cpu', weights_only=False).get('best_by_panel', {})
            self.trim_auxiliary_logs()
        else:
            assert not (self.out/'metadata.json').exists(), 'Refuse to overwrite a trial'
            write_json(self.out/'config.json', self.cfg)
            write_json(self.out/'metadata.json', dict(binding=self.binding, seed=28, parameters=32612,
                architecture=self.cfg['architecture'], optimizer=opt, score_bounds=[8,8,2,2], epochs=7,
                batch_size=16, corpus_configs=17261, corpus_bases=10000, actions_per_epoch=1122860,
                updates_per_epoch=1079, total_updates=7553, total_graph_presentations=120827,
                plan_sha256=self.plan_sha, initial_parameters_sha256=init_sha,
                initial_and_epoch_plan_exactly_match_old_S0=True,
                validation_counts={'synthetic_val862':862}, observation_steps=sorted(self.observation_steps),
                train_probe=0, from_scratch=True, teacher_version='Portfolio11-no-search',
                validation_teacher_reference='legacy Teacher64; not recomputed',
                primary_metric='base-macro C/HEFT', evaluation_used_for_tuning=True,
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        self.model.train()

    def observe(self):
        marker = self.out/f'observation_complete_{self.step:05d}.json'
        if marker.exists():
            return
        if not any(p['step'] == self.step for p in self.history):
            self.evaluate()
        self.plot()
        self.checkpoint('last')
        write_json(marker, dict(step=self.step, complete=True))

    def plot(self):
        subprocess.run([self.cfg['sim_python'], str(ROOT/'figures/gen_fig_teacher_search_ablation_20261102.py')], cwd=ROOT, check=True)


def audit_inputs():
    cfg = config()
    out = ROOT/cfg['run']
    out.mkdir(parents=True, exist_ok=True)
    store = Store()
    old = connect(ROOT/cfg['training_cache']/'public.sqlite3')
    try:
        cols = 'key,base_id,n,k,physical_sha'
        query = f'SELECT {cols} FROM records WHERE source="synthetic" AND role="train" ORDER BY key'
        rows = list(store.train.execute(query))
        assert rows == list(old.execute(query)) and len(rows) == 17261
        bases = {r[1] for r in rows}
        assert len(bases) == 10000
        graph_checks = actions = 0
        for key, *_ in rows:
            new_row = store.row(key)
            old_row = unpack(old.execute('SELECT payload FROM records WHERE key=?',(key,)).fetchone()[0])
            assert new_row['graph'] == old_row['graph'] and new_row['reference'] == old_row['reference']
            assert new_row['teacher_version'] == 'Portfolio11-no-search'
            actions += len(new_row['decisions'])
            graph_checks += 1
        assert actions == 1122860
        from daglab.structured_unique import iter_unique_rows
        original_val = {r['key']:r for r in iter_unique_rows(ROOT/'data/structured_scheduling_v1','val')}
        expected = sorted(original_val)
        assert expected == store.panel_keys['synthetic_val862'] and len(expected) == 862
        for key in expected:
            v = store.row(key)
            assert store.raw(key) == original_val[key]
            assert v['reference'] == original_val[key]['quality']['reference_HEFT']
            assert v['base_id'] not in bases and key not in dict((r[0],1) for r in rows)
        write_json(out/'INPUT_AUDIT.json',dict(status='PASS', configs=17261, bases=10000,
            graph_features_unchanged=graph_checks, actions_per_epoch=actions, validation_configs=862,
            validation_bases=len({store.row(k)['base_id'] for k in expected}),
            validation_membership_and_physical_inputs_match_original=True,
            validation_feature_cache_reused_from_old17K_run=True,
            no_train_val_key_or_base_overlap=True, augmented_training_configs_used=0,
            teacher_database_sha256=sha256(ROOT/cfg['teacher_cache']/'public.sqlite3'),
            validation_database_sha256=sha256(ROOT/cfg['validation_cache'])))
    finally:
        store.close()
        old.close()


def engineering():
    trial = Trial(out_override=ROOT/config()['run']/'engineering')
    try:
        trial.gate()
        trial.update()
        trial.update()
        trial.checkpoint('last')
        trial.update()
        expected = {k:v.detach().clone() for k,v in trial.model.state_dict().items()}
        opt = trial.opt.state_dict()
        trial.close()
        trial = Trial(resume=True, out_override=ROOT/config()['run']/'engineering')
        assert trial.step == 2
        trial.update()
        for k,v in trial.model.state_dict().items():
            assert trial.torch.equal(v, expected[k]), k
        for p,state in opt['state'].items():
            for k,v in state.items():
                actual = trial.opt.state_dict()['state'][p][k]
                assert trial.torch.equal(v,actual) if isinstance(v,trial.torch.Tensor) else v == actual
        write_json(ROOT/config()['run']/'ENGINEERING_CHECKS.json', dict(status='PASS',
            binding=trial.binding, exact_resume=True, independent_CE_gradient_and_teacher_replay=True,
            initial_weights_and_epoch_plan_match_old_S0=True, quality_conclusion=False))
    finally:
        trial.close()


def worker():
    from daglab.state_score import StateScoreDecoder
    store = Store()
    @lru_cache(maxsize=24)
    def decoder(key):
        return StateScoreDecoder(to_instance(store.raw(key)))
    try:
        for line in sys.stdin:
            q = json.loads(line)
            if q['cmd'] == 'close':
                print('{}', flush=True)
                return
            assert q['cmd'] == 'roll'
            results = []
            for item in q['items']:
                r = decoder(item['key']).rollout(item['coefficients'], sample=item.get('sample',False),
                    seed=item.get('seed',0), forced_mapping=item.get('forced_mapping'))
                row = dict(key=item['key'], C=r['makespan'], cost=r['cost'], feasible=True, decoder_seconds=r['decoder_seconds'])
                if q.get('keep_decisions'):
                    row['decisions'] = r['decisions']
                results.append(row)
            print(json.dumps(results, allow_nan=False), flush=True)
    finally:
        store.close()


def controller():
    cfg = config()
    out = ROOT/cfg['run']
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child = None
    awake = subprocess.Popen(['caffeinate','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    stage = 'input_audit'
    def status(**extra):
        write_json(out/'controller_status.json',dict(status=stage,pid=os.getpid(),updated_unix=time.time(),
            child_pid=child.pid if child else None,**extra))
    try:
        status()
        audit_inputs()
        stage = 'engineering_checks'
        status()
        if not (out/'ENGINEERING_CHECKS.json').exists():
            with (out/'engineering.log').open('a') as log:
                subprocess.run([cfg['torch_python'],'-u',str(SCRIPT),'--check'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        checks = json.loads((out/'ENGINEERING_CHECKS.json').read_text())
        assert checks['status'] == 'PASS'
        for restart in range(3):
            stage = 'training'
            cmd = [cfg['torch_python'],'-u',str(SCRIPT),'--train']
            if (out/'last.pt').exists():
                cmd.append('--resume')
            with (out/'training.log').open('a') as log:
                child = subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            with (out/'commands.jsonl').open('a') as log:
                log.write(json.dumps(dict(pid=child.pid,command=cmd,time=time.time()))+'\n')
            while child.poll() is None:
                status(restarts=restart)
                time.sleep(10)
            summary = out/'summary.json'
            if child.returncode == 0 and summary.exists() and json.loads(summary.read_text())['status'] == 'complete':
                stage = 'complete'
                status()
                return
            if not (out/'last.pt').exists() or restart == 2:
                raise RuntimeError(f'Training exited {child.returncode}; see training.log')
        raise RuntimeError('Restart budget exhausted')
    except BaseException:
        stage = 'failed'
        status(error=traceback.format_exc())
        if child and child.poll() is None:
            child.terminate()
        raise
    finally:
        awake.terminate()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker',action='store_true')
    ap.add_argument('--check',action='store_true')
    ap.add_argument('--train',action='store_true')
    ap.add_argument('--resume',action='store_true')
    args = ap.parse_args()
    if args.worker:
        worker()
    elif args.check:
        engineering()
    elif args.train:
        trial = Trial(args.resume)
        try:
            trial.run()
        finally:
            trial.close()
    else:
        controller()
