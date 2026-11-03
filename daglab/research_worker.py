"""Bounded-memory research RPC, derived indexed TRAIN cache, immutable sources."""
import hashlib
import json
import random
import sqlite3
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/structured_scheduling_v1'
PROBE = ROOT / 'runs/structured_gnn_failure_audit_20261026/payload.json'
MANIFEST = '8ef25bfef98975e8a79999c44df263f9650124abd4cac3773d4dce75a2b9a4ab'


def serve(db_path):
    from .structured_unique import iter_unique_rows
    from .structured_dataset import to_instance
    from .structured_gnn_simulator import shuffled
    from .reference import graph_features
    from .module_lab.features import enrich
    from .state_score import StateScoreDecoder

    assert hashlib.sha256((DATA / 'MANIFEST.json').read_bytes()).hexdigest() == MANIFEST
    db_path = Path(db_path)
    fresh = not db_path.exists()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    if fresh:
        db.execute('create table rows (idx integer primary key, key text unique, row blob)')
        db.execute('create table meta (key text primary key, value text)')
        digest = hashlib.sha256()
        for idx, row in enumerate(iter_unique_rows(DATA, 'train')):
            blob = zlib.compress(json.dumps(row, separators=(',', ':')).encode(), 1)
            db.execute('insert into rows values (?,?,?)', (idx, row['key'], blob))
            digest.update((row['key'] + '\n').encode())
        db.executemany('insert into meta values (?,?)', [('manifest', MANIFEST),
                           ('raw_keys', digest.hexdigest()), ('complete', 'true')])
        db.commit()
    meta = dict(db.execute('select key,value from meta'))
    assert meta['manifest'] == MANIFEST and meta['complete'] == 'true'
    count = db.execute('select count(*) from rows').fetchone()[0]
    assert count == 17261
    val = list(iter_unique_rows(DATA, 'val'))
    probes = [v['row'] for v in json.loads(PROBE.read_text()) if v['row']['split'] == 'train']
    cache = {}
    raw_keys = [v[0] for v in db.execute('select key from rows order by idx')]

    def getrow(idx):
        return json.loads(zlib.decompress(db.execute('select row from rows where idx=?', (idx,)).fetchone()[0]))

    def material(row, teacher=True):
        inst = to_instance(row)
        dec = StateScoreDecoder(inst)
        cache[row['key']] = dec
        graph, _ = graph_features(inst)
        graph = enrich(dict(instance=inst, graph=graph), dec)
        heft = row['quality'].get('reference_HEFT')
        if heft is None:
            heft = row['quality']['native_makespans'][row['quality']['rule_names'].index('HEFT')]
        item = dict(id=row['key'], base=row['base_id'], graph=graph, n=inst['n'],
                    HEFT=heft, teacher=None,
                    tags={k: v for k, v in row['input'].items() if k in ['n', 'k']},
                    metadata=row.get('metadata', {}))
        if teacher and 'supervision' in row:
            target = row['supervision']
            replay = dec.rollout([[0., 0., 0., 0.]] * (dec.n * dec.k), forced_mapping=target['task_to_pool'])
            assert replay['makespan'] == target['makespan'] and replay['cost'] == target['cost']
            item.update(decisions=replay['decisions'], teacher=target['makespan'])
        return item

    for line in sys.stdin:
        q = json.loads(line)
        cmd = q['cmd']
        if cmd == 'init':
            answer = dict(TRAIN=count, VAL=len(val), probe=len(probes), manifest=MANIFEST, raw_keys=meta['raw_keys'])
        elif cmd == 'order':
            mode, seed = q['mode'], q['seed']
            indices = list(range(count))
            if mode == 'buffer':
                indices = list(shuffled(indices, seed))
            elif mode == 'full':
                random.Random(seed).shuffle(indices)
            else:
                raise ValueError(mode)
            assert len(set(indices)) == count
            keys = [raw_keys[i] for i in indices]
            answer = dict(indices=indices, digest=hashlib.sha256(('\n'.join(keys) + '\n').encode()).hexdigest())
        elif cmd in ['train', 'val', 'probe', 'checks']:
            cache.clear()
            if cmd == 'train':
                rows = [getrow(i) for i in q['indices']]
            elif cmd == 'checks':
                shapes, rows = set(), []
                for i in range(count):
                    row = getrow(i)
                    shape = row['input']['n'], row['input']['k']
                    if shape not in shapes:
                        rows.append(row)
                        shapes.add(shape)
                    if len(shapes) == 12:
                        break
            else:
                rows = (val if cmd == 'val' else probes)[q['offset']:q['offset']+q['count']]
            answer = [material(r, q.get('teacher', True)) for r in rows]
        elif cmd in ['roll', 'rlroll']:
            answer = []
            for item in q['items']:
                dec = cache[item['id']]
                if cmd == 'rlroll':
                    from .classic_ppo import InstrumentedDecoder
                    dec = InstrumentedDecoder(dec.inst)
                result = dec.rollout(item['x'], sample=item.get('sample', False), seed=item.get('seed', 0),
                                     temperature=item.get('temperature',1.))
                entry = dict(id=item['id'], arm=item.get('arm', ''), C=result['makespan'], cost=result['cost'])
                if q.get('decisions') or cmd == 'rlroll':
                    entry['decisions'] = result['decisions']
                if q.get('schedule'):
                    entry['schedule'] = result['schedule']
                answer.append(entry)
        elif cmd == 'close':
            print('{}', flush=True)
            db.close()
            return
        else:
            raise ValueError(cmd)
        print(json.dumps(answer, allow_nan=False), flush=True)
