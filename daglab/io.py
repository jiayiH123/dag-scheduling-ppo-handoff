from pathlib import Path
import gzip,json
ROOT=Path(__file__).resolve().parents[1]
DEFAULT_DATA=ROOT/'data/reference_dataset.jsonl.gz'
def read_records(path=DEFAULT_DATA,split=None,limit=None):
    path=Path(path);opener=gzip.open if path.suffix=='.gz' else open
    records=[]
    with opener(path,'rt') as f:
        for line in f:
            r=json.loads(line)
            if split and not r['key'].startswith(split):continue
            records.append(r)
            if limit and len(records)>=limit:break
    if not records:raise ValueError('no matching records')
    return records

def model_input(inst):
    from .reference import core
    info=core.prepare(inst)
    return {'physics':{'nonpreemptive':True,'communication':'fixed ceil((latency+data/bandwidth)*ticks_per_unit-1e-10); same-pool zero','overlap':True,'link_contention':False,'cost':'duration*pool_cost_rate'},
        'tasks':[{'id':v,'workload':inst['workload'][v],'demand':inst['demands'][v],'duration_by_pool':info['duration'][v],'cost_by_pool':info['costs'][v],'eligible_pools':info['eligible'][v],'heft_rank':info['ranks'][v]} for v in range(info['n'])],
        'edges':[{'parent':u,'child':v,'data_mb':inst['data_mb'][u][v]} for u,v in inst['edges']],
        'pools':[{'id':p,'speed':inst['speeds'][p],'capacity':inst['capacities'][p],'cost_rate':inst['cost_rates'][p]} for p in range(info['k'])],
        'bandwidth':inst['bandwidth'],'budget':inst['budget'],'bmin':inst['bmin'],'bfast':inst['bfast'],'ticks_per_unit':inst.get('ticks_per_unit',1),'latency':inst.get('latency',0)}
