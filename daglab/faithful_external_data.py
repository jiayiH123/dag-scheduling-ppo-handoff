"""Read-only original TRAIN/VAL interfaces; TRAIN supervision is never used."""
import collections
import hashlib
import json
from pathlib import Path
import random
import sqlite3

from .faithful_external import ROOT, RUN
from .sft_three_panel import unpack, to_instance, write_json
from .structured_dataset import sha256

CONFIG = ROOT / 'configs/faithful_external_20261102.json'


def config():
    cfg=json.loads(CONFIG.read_text())
    import sys
    cfg.update(python=sys.executable,plot_python=sys.executable)
    return cfg


class Inputs:
    def __init__(self):
        cfg = config()
        self.train_path = ROOT / cfg['train_database']; self.val_path = ROOT / cfg['validation_database']
        self.train = sqlite3.connect('file:' + str(self.train_path) + '?mode=ro', uri=True)
        self.val = sqlite3.connect('file:' + str(self.val_path) + '?mode=ro', uri=True)
        metadata = self.train.execute('select key,base_id,n,k,physical_sha from records where source="synthetic" and role="train" order by key').fetchall()
        self.keys = [r[0] for r in metadata]
        assert len(self.keys) == 17261 and len({r[1] for r in metadata}) == 10000
        self.validation = []
        for key, raw, payload in self.val.execute('select key,raw,payload from records where panel="synthetic_val862" order by key'):
            raw, payload = unpack(raw), unpack(payload)
            self.validation.append(dict(key=key, base_id=raw['base_id'], n=raw['input']['n'],
                family=raw['meta'].get('family', raw['meta'].get('topology')), inst=to_instance(raw),
                references=payload['references']))
        assert len(self.validation) == 862 and len({r['base_id'] for r in self.validation}) == 500
        assert not set(self.keys) & {r['key'] for r in self.validation}
        assert not {r[1] for r in metadata} & {r['base_id'] for r in self.validation}
        self.metadata = metadata

    def instance(self, key):
        # Only raw is queried. Existing teacher supervision is discarded explicitly.
        raw = unpack(self.train.execute('select raw from records where key=? and source="synthetic" and role="train"', (key,)).fetchone()[0])
        return to_instance(dict(schema=raw['schema'], input=raw['input'], meta=raw['meta'],
                                key=raw['key'], base_id=raw['base_id'], split='train'))

    def epoch_keys(self, seed, epoch):
        keys = list(self.keys); random.Random(seed * 100003 + epoch).shuffle(keys)
        return keys

    def audit(self):
        scope = dict(train=17261, train_bases=10000, val=862, val_bases=500,
            train_database_sha256=sha256(self.train_path), val_database_sha256=sha256(self.val_path),
            train_key_sha256=hashlib.sha256(('\n'.join(self.keys) + '\n').encode()).hexdigest(),
            val_key_sha256=hashlib.sha256(('\n'.join(r['key'] for r in self.validation) + '\n').encode()).hexdigest(),
            train_shapes={f'{n}x{k}': sum(r[2:4] == (n, k) for r in self.metadata)
                          for n in [20, 40, 80, 120] for k in [3, 4, 6]},
            supervision_used=False, TEST_queried=False)
        write_json(RUN / 'INPUT_AUDIT.json', scope)
        return scope


def summarize(rows, column='C'):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r['base_id']].append(r[column] / r['references']['HEFT']['C'])
    return dict(configs=len(rows), bases=len(groups),
        base_macro=sum(sum(g) / len(g) for g in groups.values()) / len(groups),
        config_mean=sum(sum(g) for g in groups.values()) / len(rows),
        wins=sum(r[column] < r['references']['HEFT']['C'] for r in rows),
        ties=sum(r[column] == r['references']['HEFT']['C'] for r in rows),
        losses=sum(r[column] > r['references']['HEFT']['C'] for r in rows),
        all_feasible=all(r['feasible'] for r in rows))
