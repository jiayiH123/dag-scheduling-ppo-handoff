"""JSON RPC for module screens. TEST records are excluded at initialization.

Own-prefix decisions and feasibility use the unchanged authoritative decoder.
No Torch dependency here, so separate legacy simulation/Torch envs can be used.
"""
import hashlib
import json
import sys
from pathlib import Path
from daglab.io import read_records
from daglab.reference import CachedBudgetDecoder, core, mslbl, portfolio, teacher, random_policy_rollout
from .features import enrich
from .decoder import NormalizedEFTDecoder


def main():
    records, decoders, graphs, replayed = {}, {}, {}, {}
    decoder_mode = 'legacy'
    counts = {'complete_decodes': 0, 'validated_schedules': 0, 'teacher_reference_calls': 0,
              'forced_TRAIN_teacher_replays': 0, 'zero_head_HEFT_identity_checks': 0}
    def decoder(key):
        if key not in decoders:
            cls = NormalizedEFTDecoder if decoder_mode == 'normalized_eft' else CachedBudgetDecoder
            decoders[key] = cls(records[key]['instance'])
            if decoder_mode == 'normalized_eft':
                d = decoders[key]
                zero = d.rollout([[0., 0.]] * (d.n * d.k), 'local', materialize=True)
                original = core.schedule(records[key]['instance'], 'HEFT')
                if zero['schedule']['tasks'] != original['tasks']:
                    raise AssertionError('normalized zero-head HEFT task identity')
                counts['zero_head_HEFT_identity_checks'] += 1
        return decoders[key]
    for line in sys.stdin:
        q = json.loads(line)
        command = q['command']
        if command == 'init':
            decoder_mode = q.get('decoder', 'legacy')
            if decoder_mode not in ['legacy', 'normalized_eft']:
                raise ValueError('unknown decoder mode')
            rows = read_records(q['data'])
            records = {r['key']: r for r in rows if r['key'].startswith(('train', 'val'))}
            train = {r['instance']['latent_id'] for r in records.values() if r['key'].startswith('train')}
            val = {r['instance']['latent_id'] for r in records.values() if r['key'].startswith('val')}
            if train & val:
                raise AssertionError('TRAIN/VAL latent graph overlap')
            if not train or not val:
                raise ValueError('need TRAIN labels and separate VAL records')
            result = {'data_sha256': hashlib.sha256(Path(q['data']).read_bytes()).hexdigest(),
                      'records': [{'id': k, 'latent_id': r['instance']['latent_id'],
                                   'family': r['instance']['family'], 'n': r['instance']['n']}
                                  for k, r in records.items()], 'test_loaded_for_use': False}
        elif command == 'get':
            result = []
            for key in q['ids']:
                r = records[key]
                if key not in graphs:
                    graphs[key] = enrich(r, decoder(key))
                decisions = r.get('teacher_decisions') if key.startswith('train') else None
                if decisions is not None and decoder_mode == 'normalized_eft':
                    if key not in replayed:
                        d = decoder(key)
                        mapping = [-1] * d.n
                        for decision in decisions:
                            mapping[decision['action'] // d.k] = decision['action'] % d.k
                        forced = d.rollout([[0., 0.]] * (d.n * d.k), 'local', forced_mapping=mapping, materialize=True)
                        if forced['schedule']['tasks'] != r['teacher']['tasks']:
                            raise AssertionError('physical teacher schedule changed')
                        replayed[key] = forced['decisions']
                        counts['forced_TRAIN_teacher_replays'] += 1
                    decisions = replayed[key]
                result.append({'id': key, 'graph': graphs[key], 'reference': r['reference'],
                               'latent_id': r['instance']['latent_id'],
                               'teacher_decisions': decisions})
        elif command == 'rollout':
            result = []
            for item in q['items']:
                d = decoder(item['id'])
                r = d.rollout(item['logits'], 'local', sample=q.get('sample', False),
                              seed=item.get('seed', 0), materialize=True)
                counts['complete_decodes'] += 1
                counts['validated_schedules'] += 1  # materialize invokes core.validate
                if not q.get('keep_schedule', False):
                    r.pop('schedule')
                r['id'], r['scale'] = item['id'], d.scale
                result.append(r)
        elif command == 'references':
            result = []
            for key in q['ids']:
                r, d = records[key], decoder(key)
                inst = r['instance']
                row = {'id': key, 'latent_id': inst['latent_id'], 'n': inst['n'],
                       'family': inst['family'], 'ccr': inst['target_ccr'],
                       'budget_mode': inst['budget_mode'], 'resource_mode': inst['resource_mode']}
                for name in ['Greedy', 'HEFT', 'PEFT']:
                    row[name] = core.schedule(inst, name)['makespan']
                row['RandomMean5'] = sum(random_policy_rollout(inst, [0.] * (d.n * d.k), 'joint', True,
                                        inst['seed'] * 17 + j)['makespan'] for j in range(5)) / 5
                row['MSLBL'] = mslbl(inst)['makespan']
                schedules, _ = portfolio(inst)
                row['Portfolio'] = min(s['makespan'] for s in schedules)
                row['Teacher64'] = teacher(inst, d)[0]['makespan']
                counts['teacher_reference_calls'] += 1
                result.append(row)
        elif command == 'counts':
            result = dict(counts)
        elif command == 'close':
            print(json.dumps(counts), flush=True)
            return
        else:
            raise ValueError(command)
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
