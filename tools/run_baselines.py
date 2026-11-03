#!/usr/bin/env python3
"""Recompute native heuristics / fixed-order teachers on any shipped VAL panel."""
import argparse,gzip,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from daglab.portable_runtime import Data
from daglab.sft_three_panel import unpack,to_instance,panel_summary
from daglab.reference import core,portfolio,mapping_of
from daglab.state_score import StateScoreDecoder
from daglab.external_baseline_sim import Problem
from daglab.diagnostic_decoder import DiagnosticDecoder

NAMES=['HEFT','PEFT','JoinLookahead','CostFirst','MSLBL-adapted',
       'PEFT-price0.1','PEFT-price0.3','PEFT-price1','PEFT-price3','PEFT-price10']
ALL=NAMES+['Greedy','BHEFT','BUDA','RandomMean5','Portfolio10-native','Portfolio11','Teacher64','Teacher64v2']

def methods(instance,names):
    decoder=StateScoreDecoder(instance);engine=decoder.engine;problem=Problem(instance);output={}
    seeds,_=portfolio(instance)
    for name,solution in zip(NAMES,seeds):
        if name in names:
            core.validate(instance,solution);output[name]=dict(C=solution['makespan'],cost=solution['cost'],feasible=True)
    for name in ['Greedy','BHEFT','BUDA']:
        if name in names or name=='BHEFT' and any(n in names for n in ['Portfolio11','Teacher64v2']):
            if name=='Greedy':
                s=core.schedule(instance,name);core.validate(instance,s)
                output[name]=dict(C=s['makespan'],cost=s['cost'],feasible=True)
            else:
                s=problem.schedule(rule=name);checked=problem.finish(s['mapping'],s['order'])
                assert (checked['C'],checked['cost'])==(s['C'],s['cost'])
                output[name]=dict(C=s['C'],cost=s['cost'],feasible=True,mapping=s['mapping'],order=s['order'])
    if 'RandomMean5' in names:
        dec=DiagnosticDecoder(instance);parts=[]
        for j in range(5):
            r=dec.rollout([[0.]*4]*(dec.n*dec.k),uniform=True,seed=131+instance['seed']*17+j)
            parts.append(dict(C=r['makespan'],cost=r['cost'],feasible=True))
        output['RandomMean5']=dict(C=sum(r['C'] for r in parts)/5,cost=sum(r['cost'] for r in parts)/5,feasible=True,components=parts)
    if 'Portfolio10-native' in names:
        s=min(seeds,key=lambda s:(s['makespan'],s['cost']));output['Portfolio10-native']=dict(C=s['makespan'],cost=s['cost'],feasible=True)
    if any(n in names for n in ['Portfolio11','Teacher64','Teacher64v2']):
        mappings=[mapping_of(s) for s in seeds];values=[engine.evaluate(m,decoder.order) for m in mappings]
        def search(mapping,value):
            mapping=list(mapping);rng=random.Random(instance['seed']+3917)
            for _ in range(64):
                changed=list(mapping)
                for v in rng.sample(range(engine.n),rng.choice([1,2])):changed[v]=rng.choice(engine.info['eligible'][v])
                if not engine.feasible(changed):continue
                candidate=engine.evaluate(changed,decoder.order)
                if candidate<value:mapping,value=changed,candidate
            return mapping,value
        old_index=min(range(10),key=lambda i:values[i]);old_mapping,old_value=search(mappings[old_index],values[old_index])
        if 'Teacher64' in names:output['Teacher64']=dict(C=old_value[0],cost=old_value[1],mapping=old_mapping,order=decoder.order,feasible=True)
        if 'Portfolio11' in names or 'Teacher64v2' in names:
            bheft=problem.schedule(rule='BHEFT');mappings.append(bheft['mapping']);values.append(engine.evaluate(bheft['mapping'],decoder.order))
            index=min(range(11),key=lambda i:values[i])
            if 'Portfolio11' in names:output['Portfolio11']=dict(C=values[index][0],cost=values[index][1],mapping=mappings[index],order=decoder.order,feasible=True)
            if 'Teacher64v2' in names:
                mapping,value=search(mappings[index],values[index])
                if old_value<=value:mapping,value=old_mapping,old_value
                output['Teacher64v2']=dict(C=value[0],cost=value[1],mapping=mapping,order=decoder.order,feasible=True)
        for name in ['Teacher64','Teacher64v2','Portfolio11']:
            if name in output:
                r=output[name];actual=problem.finish(r['mapping'],r['order']);assert (actual['C'],actual['cost'])==(r['C'],r['cost'])
    return {name:output[name] for name in names}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--panel',choices=['synthetic_val862','wf960','wf400','parrot498'],default='synthetic_val862')
    parser.add_argument('--methods',nargs='+',default=['HEFT','PEFT','MSLBL-adapted','BHEFT','BUDA'])
    parser.add_argument('--out',required=True);parser.add_argument('--limit',type=int);args=parser.parse_args()
    names=ALL if args.methods==['all'] else args.methods;assert set(names)<=set(ALL)
    data=Data(panel=args.panel);rows=[];started=time.perf_counter()
    try:
        for key in data.val_keys[:args.limit]:
            raw,payload=[unpack(v) for v in data.val.execute('SELECT raw,payload FROM records WHERE key=?',(key,)).fetchone()]
            results=methods(to_instance(raw),names)
            rows.append(dict(**payload['meta'],methods=results,feasible=True,references=payload['references']))
            if len(rows)%25==0:print('BASELINES',len(rows),flush=True)
        panel='wf_by_application' if args.panel.startswith('wf') else 'parrot_pattern_templates' if args.panel=='parrot498' else 'synthetic_val862'
        summaries={name:panel_summary(panel,[dict(r,C=r['methods'][name]['C']) for r in rows]) for name in names}
        path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(path,'wt') as stream:json.dump(dict(panel=args.panel,methods=names,summaries=summaries,rows=rows,seconds=time.perf_counter()-started),stream)
        print(json.dumps({name:r['C_over']['HEFT'] for name,r in summaries.items()},indent=2))
    finally:data.close()

if __name__=='__main__':main()
