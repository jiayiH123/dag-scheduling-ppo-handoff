"""Line-delimited JSON RPC; never generates labels on validation/test."""
import json,sys
from .io import read_records
from .reference import CachedBudgetDecoder

def main():
    records={};decoders={}
    for line in sys.stdin:
        q=json.loads(line);cmd=q['command']
        if cmd=='init':
            records={r['key']:r for r in read_records(q['data'])}
            result={s+'_ids':[k for k in records if k.startswith(s)] for s in ['train','val','test']}
        elif cmd=='get':
            result=[{'id':k,'graph':records[k]['graph'],'reference':records[k]['reference'],'feature_seconds':0.,'teacher_decisions':records[k].get('teacher_decisions') if k.startswith('train') else None} for k in q['ids']]
        elif cmd=='rollout':
            result=[]
            for item in q['items']:
                k=item['id']
                if k not in decoders:decoders[k]=CachedBudgetDecoder(records[k]['instance'])
                r=decoders[k].rollout(item['logits'],q['mode'],sample=q.get('sample',False),seed=item.get('seed',0),materialize=q.get('include_schedule',False))
                r['id']=k;result.append(r)
        elif cmd=='close':
            print(json.dumps({'closed':True}),flush=True);return
        else:raise ValueError(cmd)
        print(json.dumps(result,allow_nan=False),flush=True)
if __name__=='__main__':main()
