"""Immutable training cache plus a small overlay for new three-panel VAL."""
import collections
import gzip
import json
import sqlite3
import zlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/sft_three_panel_run_20261101.json'


def config():return json.loads(CONFIG.read_text())
def unpack(blob):return json.loads(zlib.decompress(blob))
def pack(value):return zlib.compress(json.dumps(value,separators=(',',':'),allow_nan=False).encode(),3)
def write_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def connect(path):return sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True)


def to_instance(raw):
    if 'input' in raw:
        from .structured_dataset import to_instance as convert
        return convert(raw)
    from .offline_data import dense_instance
    return dense_instance(raw,max_tasks=300)


class Store:
    def __init__(self,cfg=None):
        self.cfg=cfg or config()
        self.train=connect(ROOT/self.cfg['training_cache']/'public.sqlite3')
        self.extra=connect(ROOT/self.cfg['data']/'validation.sqlite3')
        self.wf_keys=json.loads((ROOT/self.cfg['scope_lock']).read_text())['validation']['wf_by_application']['keys']
        self.panel_keys={name:[r[0] for r in self.extra.execute('SELECT key FROM records WHERE panel=? ORDER BY key',(name,))]
                         for name in ('synthetic_val862','parrot_pattern_templates')}
        self.panel_keys['wf_by_application']=sorted(self.wf_keys)
    def row(self,key):
        r=self.extra.execute('SELECT payload FROM records WHERE key=?',(key,)).fetchone()
        if r:return unpack(r[0])
        r=self.train.execute('SELECT payload,role FROM records WHERE key=?',(key,)).fetchone()
        if r is None:raise KeyError(key)
        if r[1] not in ('train','eval'):raise ValueError('unknown role')
        return unpack(r[0])
    def raw(self,key):
        r=self.extra.execute('SELECT raw FROM records WHERE key=?',(key,)).fetchone()
        if r:return unpack(r[0])
        r=self.train.execute('SELECT raw FROM records WHERE key=?',(key,)).fetchone()
        if r is None:raise KeyError(key)
        return unpack(r[0])
    def close(self):self.train.close();self.extra.close()


def simple_summary(rows,group_key='base_id'):
    """Equal independent unit mean of paired ratios, not ratio of totals."""
    groups=collections.defaultdict(list)
    for r in rows:groups[r[group_key]].append(r)
    refs=sorted(set.intersection(*(set(r['references']) for r in rows)))
    values={name:sum(sum(r['C']/r['references'][name]['C'] for r in rs)/len(rs)
                    for rs in groups.values())/len(groups) for name in refs}
    return dict(configs=len(rows),bases=len({r['base_id'] for r in rows}),units=len(groups),
                C_over=values,feasible=all(r['feasible'] for r in rows))


def panel_summary(panel,rows):
    if panel=='wf_by_application':
        from .wf_panels import report
        grouped=report(rows)
        return dict(grouped['all'],groups=grouped['groups'])
    if panel=='parrot_pattern_templates':
        result=simple_summary(rows,'public_plan_id');families=collections.defaultdict(list);templates=collections.defaultdict(list)
        for r in rows:families[r['family']].append(r);templates[r['public_plan_id']].append(r)
        result['groups']=dict(family={k:simple_summary(rs,'public_plan_id') for k,rs in families.items()},
                              template={k:simple_summary(rs,'public_plan_id') for k,rs in templates.items()})
        return result
    return simple_summary(rows)
