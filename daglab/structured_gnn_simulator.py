"""Bounded-memory physics RPC for the immutable structured-v1 CPU pilot.

Runs in the NumPy simulator environment, independently of the Torch interpreter.
No TEST/OOD reads, teacher generation, or modification of frozen records.
"""
import hashlib
import json
from pathlib import Path
import random
import sys
import time

from .structured_dataset import to_instance
from .structured_unique import iter_unique_rows
from .reference import graph_features, core, portfolio, mapping_of
from .module_lab.decoder import NormalizedEFTDecoder
from .module_lab.features import enrich


def shuffled(rows, seed, capacity=128):
    rng = random.Random(seed)
    buffer = []
    for row in rows:
        if len(buffer) < capacity:
            buffer.append(row)
            continue
        i = rng.randrange(len(buffer))
        yield buffer[i]
        buffer[i] = row
    rng.shuffle(buffer)
    yield from buffer


def prepare(row, check_teacher=False):
    start = time.perf_counter()
    inst = to_instance(row)
    decoder = NormalizedEFTDecoder(inst)
    if list(decoder.order) != row['input']['heft_order']:
        raise AssertionError('HEFT order identity')
    graph, _ = graph_features(inst)
    graph = enrich(dict(instance=inst, graph=graph), decoder)
    item = dict(id=row['key'], base_id=row['base_id'], graph=graph,
                reference=row['quality'].get('reference_HEFT'),
                n=inst['n'], pools=decoder.k, family=inst['family'], ccr=inst['target_ccr'])
    if 'supervision' in row:
        item['teacher_decisions'] = row['supervision']['decisions']
        if check_teacher:
            replay = decoder.rollout([[0., 0.]]*(decoder.n*decoder.k), 'local',
                                     forced_mapping=row['supervision']['task_to_pool'], materialize=True)
            for key in ['makespan', 'cost']:
                if replay[key] != row['supervision'][key]:
                    raise AssertionError(('teacher identity', key))
            tasks = replay['schedule']['tasks']
            for field, label in [('start', 'starts'), ('end', 'ends')]:
                if [t[field] for t in tasks] != row['supervision'][label]:
                    raise AssertionError(('teacher times', field))
            for a, b in zip(replay['decisions'], item['teacher_decisions']):
                for field in ['available', 'selected', 'action', 'offsets', 'budget_local']:
                    if a[field] != b[field]:
                        raise AssertionError(('teacher prefix', field))
    item['preparation_seconds'] = time.perf_counter()-start
    return item, decoder


def main():
    root = None
    val = []
    cache = {}
    train_iter = None
    train_count = 0
    action_count = 0
    key_hash = hashlib.sha256()
    for line in sys.stdin:
        q = json.loads(line)
        command = q['command']
        if command == 'init':
            root = Path(q['data'])
            counts = {}
            split_bases = {}
            for split in ['train', 'val']:
                bases = set()
                total = actions = 0
                for row in iter_unique_rows(root, split):
                    total += 1
                    bases.add(row['base_id'])
                    actions += len(row.get('supervision', {}).get('decisions', []))
                    if split == 'val':
                        if 'supervision' in row:
                            raise AssertionError('VAL labels must not be used for training')
                        val.append(row)
                counts[split] = dict(configs=total, bases=len(bases), actions=actions)
                split_bases[split] = bases
            if split_bases['train'] & split_bases['val']:
                raise AssertionError('base split leakage')
            result = dict(counts=counts, manifest_sha256=hashlib.sha256(
                (root/'MANIFEST.json').read_bytes()).hexdigest(), TEST_evaluated=False)
        elif command == 'checks':
            examples = {}
            for row in iter_unique_rows(root, 'train'):
                combo = row['input']['n'], row['input']['k']
                if combo not in examples:
                    examples[combo] = prepare(row, check_teacher=True)[0]
                if len(examples) == 12:
                    break
            result = dict(items=[examples[k] for k in sorted(examples)], teacher_replay_passed=len(examples))
        elif command == 'baselines':
            records = []
            start = time.perf_counter()
            for row in val[q['start']:q['start']+q['count']]:
                inst = to_instance(row)
                schedules, _ = portfolio(inst)
                decoder = NormalizedEFTDecoder(inst)
                for schedule in schedules:
                    core.validate(inst, schedule)
                reference = row['quality']['reference_HEFT']
                if schedules[0]['makespan'] != reference:
                    raise AssertionError('stored HEFT mismatch')
                replayed = [decoder.engine.evaluate(mapping_of(s), decoder.order)[0] for s in schedules]
                records.append(dict(id=row['key'], base_id=row['base_id'], n=inst['n'],
                                    pools=decoder.k, reference=reference,
                                    MSLBL=schedules[4]['makespan'],
                                    Portfolio_native=min(s['makespan'] for s in schedules),
                                    Portfolio_fixed_HEFT=min(replayed)))
            result = dict(records=records, seconds=time.perf_counter()-start)
        elif command == 'train_start':
            train_iter = iter(shuffled(iter_unique_rows(root, 'train'), q['seed']))
            train_count = action_count = 0
            key_hash = hashlib.sha256()
            result = dict(started=True)
        elif command == 'train_next':
            items = []
            for _ in range(q['count']):
                row = next(train_iter, None)
                if row is None:
                    break
                item, _ = prepare(row)
                items.append(item)
                train_count += 1
                action_count += len(item['teacher_decisions'])
                key_hash.update((row['key']+'\n').encode())
            result = dict(items=items, train_configs_seen=train_count, train_actions_seen=action_count,
                          ordered_key_sha256=key_hash.hexdigest())
        elif command == 'val_get':
            cache.clear()
            items = []
            for row in val[q['start']:q['start']+q['count']]:
                item, decoder = prepare(row)
                cache[item['id']] = decoder
                items.append(item)
            result = items
        elif command == 'val_rollout':
            result = []
            for item in q['items']:
                decoder = cache[item['id']]
                value = decoder.rollout(item['coefficients'], 'local', materialize=True)
                result.append(dict(id=item['id'], arm=item['arm'], makespan=value['makespan'],
                                   cost=value['cost'], decoder_seconds=value['decoder_seconds'],
                                   physically_validated=True))
        elif command == 'close':
            print(json.dumps(dict(closed=True)), flush=True)
            return
        else:
            raise ValueError(command)
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
