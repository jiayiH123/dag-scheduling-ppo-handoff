"""Shared immutable inputs and simulator RPC for the matched Teacher/PPO study."""
import collections
import hashlib
import json
from pathlib import Path
import random
import sys
from .sft_three_panel import connect,unpack,to_instance,write_json
from .structured_dataset import sha256

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/teacher_search_ablation_ppo_20261102.json'


def config():
    cfg=json.loads(CONFIG.read_text())
    cfg.update(torch_python=sys.executable,sim_python=sys.executable)
    return cfg


def train_keys():
    path=ROOT/'runs/original_teacher_sources_20261102/PER_CONFIGURATION.jsonl'
    keys=[json.loads(line)['key'] for line in path.read_text().splitlines()]
    assert len(keys)==len(set(keys))==17261
    raw_digest=hashlib.sha256(('\n'.join(keys)+'\n').encode()).hexdigest()
    assert raw_digest=='fed4ad273f9639d77146cf06b4d91d7745721dd49332b4fd52c290b21ce694c8'
    return keys


class Store:
    def __init__(self):
        cfg=config()
        self.train=connect(ROOT/cfg['original_cache']/'public.sqlite3')
        self.val=connect(ROOT/cfg['validation_cache'])
        self.keys=train_keys()
        actual={r[0] for r in self.train.execute('SELECT key FROM records WHERE source="synthetic" AND role="train"')}
        assert set(self.keys)==actual
        self.val_keys=[r[0] for r in self.val.execute('SELECT key FROM records WHERE panel="synthetic_val862" ORDER BY key')]
        assert len(self.val_keys)==862 and not actual&set(self.val_keys)
        self.val_set=set(self.val_keys)

    def record(self,key):
        con=self.val if key in self.val_set else self.train
        raw,payload=con.execute('SELECT raw,payload FROM records WHERE key=?',(key,)).fetchone()
        raw,payload=unpack(raw),unpack(payload)
        from .portable_runtime import enrich_payload
        return raw,enrich_payload(raw,payload,'legacy',False)

    def row(self,key):
        return self.record(key)[1]

    def close(self):
        self.train.close()
        self.val.close()


def serve():
    from .state_score import StateScoreDecoder
    from .classic_ppo import InstrumentedDecoder
    store=Store()
    cache={}
    try:
        for line in sys.stdin:
            q=json.loads(line)
            if q['cmd']=='close':
                print('{}',flush=True)
                return
            if q['cmd']=='rows':
                cache.clear()
                answer=[]
                for key in q['keys']:
                    raw,row=store.record(key)
                    inst=to_instance(raw)
                    cache[key]=inst
                    answer.append(dict(id=key,base=row['base_id'],n=row['n'],graph=row['graph'],
                        HEFT=row['reference'],teacher=row.get('teacher'),
                        references=row.get('references'),meta=row['meta']))
            elif q['cmd'] in ('roll','rlroll'):
                answer=[]
                for item in q['items']:
                    cls=InstrumentedDecoder if q['cmd']=='rlroll' else StateScoreDecoder
                    r=cls(cache[item['id']]).rollout(item['x'],sample=item.get('sample',False),
                        seed=item.get('seed',0),temperature=item.get('temperature',1.))
                    value=dict(id=item['id'],C=r['makespan'],cost=r['cost'],feasible=True)
                    if q['cmd']=='rlroll' or q.get('decisions'):
                        value['decisions']=r['decisions']
                    if q.get('schedule'):
                        value['schedule']=r['schedule']
                    answer.append(value)
            else:
                raise ValueError(q['cmd'])
            print(json.dumps(answer,allow_nan=False),flush=True)
    finally:
        store.close()


def macro(rows,field='C'):
    groups=collections.defaultdict(list)
    for r in rows:
        groups[r['base']].append(r[field]/r['HEFT'])
    return sum(sum(v)/len(v) for v in groups.values())/len(groups)


def code_binding(paths):
    return {p:sha256(ROOT/p) for p in paths}
