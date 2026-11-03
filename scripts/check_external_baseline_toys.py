#!/usr/bin/env python3
"""Hand-derived budget allocation and exhaustive tiny scheduling checks."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,'/private/tmp/dag-baseline-deps-20261028')
from daglab.external_baseline_sim import Problem
from scripts.run_external_baselines import save

def main():
    inst=dict(n=2,model='fixed',overlap=True,ticks_per_unit=1,edges=[[0,1]],workload=[10,10],demands=[1,1],
        capacities=[1,1],speeds=[1,2],bandwidth=[[0,5],[5,0]],data_mb=[[0,10],[0,0]],latency=0,cost_rates=[1,4],budget=30)
    results=[]
    for budget,expected in [(20,20),(30,17),(40,10)]:
        inst['budget']=budget;p=Problem(inst);r=p.schedule(rule='BHEFT');f=p.finish(r['mapping'],r['order'])
        assert r['C']==f['C']==expected
        values=[p.engine.evaluate([a,b],p.order) for a in range(2) for b in range(2) if p.engine.feasible([a,b])]
        assert min(values)[0]==expected;results.append(dict(budget=budget,BHEFT_C=expected,tiny_exhaustive_opt=expected))
    inst['budget']=30;p=Problem(inst)
    for rule,expected in [('BUDA',20),('BUDA_printedledger',20),('BUDA_unitbalanced',17)]:
        r=p.schedule(rule=rule);assert r['C']==expected;p.finish(r['mapping'],r['order'])
    save(ROOT/'runs/external_baselines_20261028/hand_toy_checks.json',dict(status='PASS',BHEFT_hand_and_exhaustive=results,
        BUDA_hand_HTCM=[[41,51],[20,25]],BUDA_C=20,unitbalanced_BUDA_C=17))
    print('PASS: BHEFT exact fractional CTB/affordability, hand HTCM, exhaustive tiny OPT')

if __name__=='__main__':main()
