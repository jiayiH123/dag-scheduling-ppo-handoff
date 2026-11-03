#!/usr/bin/env python3
"""Current corrected baseline curves; old normalization attempts are separate."""
import json,os,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/external_baselines_budget_visible_20261028'

def logs(p):
    if not p.exists():return []
    text=p.read_text();ls=text.splitlines()
    if text and not text.endswith('\n'):ls=ls[:-1]
    return [json.loads(l) for l in ls]

def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(16,8));panels=['synthetic','wfinstances','tpch_composite'];status={}
    for method,label,color in [('cai','CAI-DR BiGAT','#1664a5'),('graphormer','CAI-DR Graphormer','#dc6c22')]:
        hist=json.loads((RUN/method/'history.json').read_text())['history'] if (RUN/method/'history.json').exists() else []
        for ax,panel in zip(axes[0],panels):
            if hist:ax.plot([r['epoch'] for r in hist],[r['panels'][panel]['C_HEFT'] for r in hist],label=label,color=color,marker='o')
        records=logs(RUN/method/'training.jsonl');ys=[r['sampled_C_HEFT'] for r in records]
        smooth=[sum(ys[max(0,i-99):i+1])/min(100,i+1) for i in range(len(ys))]
        axes[1,0].plot([r['graphs']/17261 for r in records],smooth,label=label,color=color)
    control=json.loads((ROOT/'runs/score_ranges_multival_20261028/summary.json').read_text())
    heuristic=json.loads((ROOT/'runs/external_baselines_20261028/heuristics/summary.json').read_text())
    for ax,panel in zip(axes[0],panels):
        ax.axhline(control['panels'][panel]['methods']['control']['C_HEFT'],color='#666',ls='--',label='current GNN B')
        ax.axhline(heuristic['panels'][panel]['methods']['BHEFT']['C_HEFT'],color='#138543',ls=':',label='BHEFT adapted')
        sp=RUN/'cai_sparse/summary.json'
        if sp.exists():
            s=json.loads(sp.read_text())
            if not s.get('intermediate',True):ax.scatter([1],[s['panels'][panel]['methods']['CAI_noDR_BiGAT_visible']['C_HEFT']],marker='x',color='#994499',label='CAI-noDR onepass')
        ax.set_title(panel+' full DEV C/HEFT');ax.set_xlabel('Complete TRAIN passes');ax.set_xlim(0,7.2)
    rec=[r for r in logs(RUN/'giph/training.jsonl') if r['update']<=300]+logs(RUN/'giph_fast_r1/training.jsonl')
    ys=[r['batch_return'] for r in rec];axes[1,1].plot([r['graphs']/17261 for r in rec],[sum(ys[max(0,i-24):i+1])/min(25,i+1) for i in range(len(ys))],color='#138543')
    axes[1,2].plot([r['graphs'] for r in rec],[r['actions'] for r in rec],color='#138543')
    axes[1,0].set_title('CAI sampled TRAIN C/HEFT: trailing100');axes[1,0].set_xlabel('TRAIN passes')
    axes[1,1].set_title('GiPH TRAIN normalized return: trailing25');axes[1,1].set_xlabel('TRAIN passes')
    axes[1,2].set_title('GiPH accepted-lineage relocation decisions');axes[1,2].set_xlabel('TRAIN configurations')
    for ax in axes.flat:ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=7)
    for method in ['cai','graphormer','cai_sparse','giph_fast_r1']:
        p=RUN/method;complete=False
        if (p/'summary.json').exists():
            s=json.loads((p/'summary.json').read_text());complete=s.get('status')=='complete' and not s.get('intermediate',True)
            if method=='giph_fast_r1':complete=s.get('status')=='complete' and 'training' in s
        status[method]=dict(status='failed' if (p/'failure.json').exists() else 'complete' if complete else 'running',
            progress=json.loads((p/'progress.json').read_text()) if (p/'progress.json').exists() else None)
    fig.suptitle('Corrected budget-visible external adaptations; '+', '.join(k+':'+r['status'] for k,r in status.items()),fontsize=10);fig.tight_layout()
    for suffix in ['png','pdf']:
        tmp=RUN/f'training_curves.tmp.{suffix}';fig.savefig(tmp,format=suffix,dpi=155);os.replace(tmp,RUN/f'training_curves.{suffix}')
    plt.close(fig);tmp=RUN/'status.tmp.json';tmp.write_text(json.dumps(status,indent=2)+'\n');os.replace(tmp,RUN/'status.json')
    return all(s['status'] in ['complete','failed'] for s in status.values())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    for i in range(480 if args.watch else 1):
        if render():break
        if args.watch:time.sleep(30)
