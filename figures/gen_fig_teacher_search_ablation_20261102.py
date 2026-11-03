#!/usr/bin/env python3
"""Observed-only SFT and paired-seed PPO dashboards; shared writer lock."""
import fcntl
import html
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'runs/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from daglab.teacher_search_ablation import config


def read(path,default):
    return json.loads(path.read_text()) if path.exists() else default


def jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines(keepends=True) if l.endswith('\n')]


def main():
    cfg=config()
    root=ROOT/cfg['run']
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'plot.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42,'axes.grid':True,'grid.alpha':.15})
    def save(fig,name):
        for ext in ['png','pdf']:
            tmp=root/f'{name}.tmp.{ext}'
            fig.savefig(tmp,dpi=300,bbox_inches='tight',format=ext)
            tmp.replace(root/f'{name}.{ext}')
        plt.close(fig)
    def finish(ax,title,xlabel,ylabel):
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
        if ax.lines:
            ax.legend(fontsize=8)
    def baselines(ax):
        values=[('HEFT',1,'#999999'),('BHEFT',.9679010441074981,'#CC79A7'),
            ('Teacher64 (legacy)',.8426115732787387,'#444444'),('Teacher64+BHEFT-v2',.831359167419895,'#009E73')]
        for name,value,color in values:
            ax.axhline(value,color=color,ls='--',lw=.9,label=name)
    colors={'strong':'#D55E00','no_search':'#0072B2'}
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for arm,path in [('strong',ROOT/cfg['strong_SFT_run']),('no_search',ROOT/cfg['no_search_SFT_run'])]:
        rows=jsonl(path/'training.jsonl')
        blocks=[rows[i:i+50] for i in range(0,len(rows),50)]
        if blocks:
            ax.plot([b[-1]['step'] for b in blocks],[sum(r['NLL'] for r in b)/len(b) for b in blocks],color=colors[arm],label=arm)
    finish(ax,'SFT training loss (50-update mean)','SFT optimizer updates','Teacher-prefix cross entropy')
    save(fig,'sft_loss')
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    for arm,path in [('strong',ROOT/cfg['strong_SFT_run']),('no_search',ROOT/cfg['no_search_SFT_run'])]:
        rows=read(path/'val_synthetic_val862_curve.json',[])
        if rows:
            ax.plot([r['step'] for r in rows],[r['C_over']['HEFT'] for r in rows],color=colors[arm],label=arm)
    baselines(ax)
    finish(ax,'SFT: original VAL862 / 500 DAGs','SFT optimizer updates','Makespan / HEFT (base-macro)')
    save(fig,'sft_validation')
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for arm in ['strong','no_search']:
        for seed,style in zip(cfg['PPO']['seeds'],['-',':','--']):
            rows=read(root/f'ppo_{arm}_seed{seed}'/'curve.json',[])
            if not rows:
                continue
            for ax,field in zip(axs,['C_HEFT_base_macro','C_HEFT_config_mean']):
                ax.plot([r['actor_updates'] for r in rows],[r[field] for r in rows],color=colors[arm],ls=style,marker='o',ms=2,label=f'{arm} / seed{seed}')
    baselines(axs[0])
    finish(axs[0],'PPO: primary base-macro','Actual PPO actor updates','Makespan / HEFT')
    finish(axs[1],'PPO: configuration mean (historical metric)','Actual PPO actor updates','Makespan / HEFT')
    save(fig,'ppo_validation')
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for arm in ['strong','no_search']:
        for seed,style in zip(cfg['PPO']['seeds'],['-',':','--']):
            rows=jsonl(root/f'ppo_{arm}_seed{seed}'/'updates.jsonl')
            blocks=[rows[i:i+32] for i in range(0,len(rows),32)]
            for ax,field in zip(axs,['actor_loss','value_loss']):
                points=[(b[-1]['actor_updates'],sum(r[field] for r in b if field in r)/sum(field in r for r in b)) for b in blocks if any(field in r for r in b)]
                if points:
                    ax.plot([p[0] for p in points],[p[1] for p in points],color=colors[arm],ls=style,label=f'{arm} / seed{seed}')
    finish(axs[0],'PPO actor loss (32-slot mean)','Actual PPO actor updates','Clipped loss - entropy bonus')
    finish(axs[1],'PPO value loss (32-slot mean)','Actual PPO actor updates','Half mean squared error')
    save(fig,'ppo_losses')
    body=['<html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><h1>Teacher search ablation: SFT + repeat512 PPO</h1>',
        '<p>Original17261 TRAIN, original862 VAL only. Main PPO: T=.5, actorLR3e-5, 512configs x4passes, 128collections. Three RL seeds share one SFT seed per arm.</p>',
        '<p>Primary metric is base-macro; configuration mean is reported separately for historical comparison. Old/new Teacher lines are evaluation references.</p>']
    for name in ['sft_loss','sft_validation','ppo_losses','ppo_validation']:
        body.append(f'<p><a href="{name}.pdf">{name} PDF</a></p><img src="{name}.png" style="max-width:100%;width:1200px">')
    state=read(root/'controller_status.json',{})
    body.append('<pre>'+html.escape(json.dumps(state,indent=2))+'</pre></html>')
    tmp=root/'report.tmp.html'
    tmp.write_text('\n'.join(body))
    tmp.replace(root/'report.html')


if __name__=='__main__':
    main()
