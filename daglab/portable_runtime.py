"""Portable cache reconstruction; unchanged graph builder and physical decoder."""
import json
from pathlib import Path
import sqlite3
from .sft_three_panel import pack, unpack, to_instance

ROOT=Path(__file__).resolve().parents[1]
_CACHE=None

def cache():
    global _CACHE
    if _CACHE is None:
        path=ROOT/'.cache/features.sqlite3';path.parent.mkdir(exist_ok=True)
        _CACHE=sqlite3.connect(path,timeout=120)
        _CACHE.execute('PRAGMA journal_mode=WAL')
        _CACHE.execute('CREATE TABLE IF NOT EXISTS graphs(key TEXT PRIMARY KEY,payload BLOB)')
        _CACHE.execute('CREATE TABLE IF NOT EXISTS traces(key TEXT,variant TEXT,payload BLOB,PRIMARY KEY(key,variant))')
    return _CACHE

def enrich_payload(raw,payload,variant='legacy',need_trace=True):
    con=cache(); key=raw['key']
    if 'graph' not in payload:
        saved=con.execute('SELECT payload FROM graphs WHERE key=?',(key,)).fetchone()
        if saved:payload['graph']=unpack(saved[0])
        else:
            from .reference import graph_features
            from .module_lab.features import enrich
            from .state_score import StateScoreDecoder
            instance=to_instance(raw);decoder=StateScoreDecoder(instance)
            graph,_=graph_features(instance)
            payload['graph']=enrich(dict(instance=instance,graph=graph),decoder)
            con.execute('INSERT OR IGNORE INTO graphs VALUES(?,?)',(key,pack(payload['graph'])));con.commit()
    if need_trace and 'supervision' in payload and 'decisions' not in payload:
        saved=con.execute('SELECT payload FROM traces WHERE key=? AND variant=?',(key,variant)).fetchone()
        if saved:decisions=unpack(saved[0])
        else:
            from .state_score import StateScoreDecoder
            decoder=StateScoreDecoder(to_instance(raw)); target=payload['supervision']
            replay=decoder.rollout([[0.]*4]*(decoder.n*decoder.k),forced_mapping=target['task_to_pool'])
            assert replay['makespan']==target['makespan'] and replay['cost']==target['cost']
            assert [r['start'] for r in replay['schedule']['tasks']]==target['starts']
            assert [r['end'] for r in replay['schedule']['tasks']]==target['ends']
            decisions=replay['decisions']
            con.execute('INSERT OR IGNORE INTO traces VALUES(?,?,?)',(key,variant,pack(decisions)));con.commit()
        payload['decisions']=decisions;payload['supervision']['decisions']=decisions
    return payload

class Data:
    def __init__(self,variant='strong',panel='synthetic_val862'):
        cfg=json.loads((ROOT/'configs/teacher_search_ablation_ppo_20261102.json').read_text())
        paths={'legacy':cfg['original_cache'],'strong':cfg['strong_teacher_cache'],'no_search':cfg['no_search_cache']}
        self.variant=variant
        self.train=sqlite3.connect('file:'+str(ROOT/paths[variant]/'public.sqlite3')+'?mode=ro',uri=True)
        valpath=ROOT/cfg['validation_cache'] if panel in ['synthetic_val862','parrot498'] else ROOT/f'data/runtime/{panel}.sqlite3'
        self.val=sqlite3.connect('file:'+str(valpath)+'?mode=ro',uri=True)
        actual_panel='parrot_pattern_templates' if panel=='parrot498' else panel if panel=='synthetic_val862' else 'wf_by_application'
        self.val_keys=[r[0] for r in self.val.execute('SELECT key FROM records WHERE panel=? ORDER BY key',(actual_panel,))]
        self.val_set=set(self.val_keys);self.keys=[r[0] for r in self.train.execute('SELECT key FROM records WHERE role="train" AND source="synthetic" ORDER BY key')]
        assert len(self.keys)==17261 and not set(self.keys)&self.val_set
    def record(self,key,need_trace=True):
        source=self.val if key in self.val_set else self.train
        raw,payload=source.execute('SELECT raw,payload FROM records WHERE key=?',(key,)).fetchone()
        raw,payload=unpack(raw),unpack(payload)
        return raw,enrich_payload(raw,payload,self.variant,need_trace)
    def close(self): self.train.close();self.val.close()

def evaluation(model,panel='synthetic_val862',limit=None):
    import torch
    from .state_score import StateScoreDecoder
    from .module_lab.models import tensors
    from .sft_three_panel import panel_summary
    data=Data(panel=panel);rows=[];was_training=model.training;model.eval()
    try:
        with torch.no_grad():
            for key in data.val_keys[:limit]:
                raw,payload=data.record(key,False)
                coefficients=model(tensors(payload['graph'],'cpu'))
                result=StateScoreDecoder(to_instance(raw)).rollout(coefficients.tolist())
                rows.append(dict(**payload['meta'],C=result['makespan'],cost=result['cost'],
                                 feasible=True,references=payload['references']))
        grouping='wf_by_application' if panel.startswith('wf') else 'parrot_pattern_templates' if panel=='parrot498' else 'synthetic_val862'
        return dict(panel=panel,summary=panel_summary(grouping,rows),rows=rows)
    finally:model.train(was_training);data.close()
