#!/usr/bin/env python3
"""Small progress report for terminal-only, original-budget GiPH replicas."""
import collections
import gzip
import html
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/giph_paper200_20261102'
OLD = ROOT / 'runs/faithful_external_20261102'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runs/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    cfg = read(ROOT / 'configs/giph_paper200_20261102.json')
    results = []; progress = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for seed in cfg['seeds']:
        out = RUN / f'seed{seed}'
        records = [json.loads(line) for line in (out / 'training.jsonl').read_text().splitlines()] if (out / 'training.jsonl').exists() else []
        if records:
            axes[0].plot([r['episode'] for r in records], [r['loss'] for r in records], label=f'seed{seed}', lw=.8)
            axes[1].plot([r['episode'] for r in records], [r['best_C']/r['initial_C'] for r in records], label=f'seed{seed}', lw=.8)
        p = read(out / 'progress.json', {})
        progress.append(dict(seed=seed, progress=p, trained=read(out / 'TRAINING_COMPLETE.json')))
        if (out / 'COMPLETE.json').exists(): results.append(read(out / 'COMPLETE.json'))
    axes[0].set_yscale('symlog', linthresh=1.)
    axes[0].set_ylabel('REINFORCE loss (symlog)')
    axes[1].set_ylabel('Best visited C / initial C (TRAIN sample)')
    for ax in axes:
        ax.set_xlabel('Training episode'); ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]: ax.legend()
    for ext in ['png', 'pdf']:
        temp = RUN / f'training_loss.tmp.{ext}'; fig.savefig(temp, dpi=180); temp.replace(RUN / f'training_loss.{ext}')
    plt.close(fig)
    lines = ['# GiPH：原文200 episode预算', '',
        '替换原先每个学习率17261 episode的过重方案。旧GiPH任务已停止，日志保留；CAI独立队列继续。',
        'TRAIN17261为抽样池；每个种子均匀有放回抽200题，seed11/22/33独立训练；Adam0.01、gamma0.97、每题最多2n步。',
        '沿用作者DGL核心和精确物理适配；原论文4节点/4边特征，外加费用/容量/预算3节点特征。没有额外学习率搜索。',
        '训练完成才对终点checkpoint全量评价VAL862，随机起点和HEFT起点分开；不按VAL选checkpoint、不取种子最优。',
        '', '|种子|进度|已训练episode|', '|---|---|---:|']
    for row in progress:
        p = row['progress']; trained = row['trained']
        lines.append(f"|{row['seed']}|{p.get('stage','starting')}|{trained['episodes'] if trained else p.get('episode',0)} / 200|")
    lines += ['', '|种子|随机起点 C/HEFT|HEFT起点 C/HEFT|训练秒数|', '|---|---:|---:|---:|']
    for row in results:
        values = row['final']['variants']
        lines.append(f"|{row['seed']}|{values['random']['base_macro']:.8f}|{values['HEFT']['base_macro']:.8f}|{row['training_seconds']:.1f}|")
    if len(results) == 3:
        means = {v: {m: float(np.mean([r['final']['variants'][v][m] for r in results]))
                     for m in ['base_macro', 'config_mean']} for v in ['random', 'HEFT']}
        subgroups = {}
        for variant in ['random', 'HEFT']:
            by_n = collections.defaultdict(lambda: collections.defaultdict(list))
            by_family = collections.defaultdict(lambda: collections.defaultdict(list))
            for row in results:
                with gzip.open(RUN / f"seed{row['seed']}" / 'val_terminal.json.gz', 'rt') as stream: cases = json.load(stream)['rows']
                assert len(cases) == 862 and all(c['feasible'] for c in cases)
                for case in cases:
                    ratio = case['methods'][variant]['C'] / case['references']['HEFT']['C']
                    by_n[str(case['n'])][case['base_id']].append(ratio)
                    by_family[case['family']][case['base_id']].append(ratio)
            aggregate = lambda g: float(np.mean([np.mean(x) for x in g.values()]))
            subgroups[variant] = dict(by_n={k:aggregate(g) for k,g in by_n.items()},
                                     by_family={k:aggregate(g) for k,g in by_family.items()})
        final = dict(status='complete', final_three_seed_mean=means, runs=results, subgroups=subgroups,
                     configuration=cfg, checkpoint_selection='fixed final episode200; no VAL selection')
        (RUN / 'FINAL_RESULTS.json').write_text(json.dumps(final, indent=2) + '\n')
        for variant in ['random', 'HEFT']:
            scores = [r['final']['variants'][variant]['base_macro'] for r in results]
            lines += ['', f"{variant} 三训练种子均值：{means[variant]['base_macro']:.8f}；样本标准差：{np.std(scores,ddof=1):.8f}。"]
    else: final = None
    lines += ['', '此处所有指标来自已知VAL862/500底图；主指标是底图宏平均，配置均值另存JSON。',
              '详细逐例映射/顺序/费用/动作数：seed*/val_terminal.json.gz；训练曲线：training_loss.png/pdf。']
    report = '\n'.join(lines) + '\n'
    (RUN / 'REPORT.md').write_text(report)
    (ROOT / 'docs/GIPH_PAPER200_RESULTS_20261102.md').write_text(report)
    (RUN / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>GiPH 200 episodes</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{width:100%}pre{white-space:pre-wrap}</style>'
        '<img src="training_loss.png"><pre>' + html.escape(report) + '</pre>')
    # Existing renderer is not bound into training; keep CAI curves current.
    subprocess.run(['/opt/miniconda3/envs/dag-research/bin/python',
                    str(ROOT / 'scripts/report_faithful_external_20261102.py')], cwd=ROOT, check=True)
    cai = read(OLD / 'cai_FINAL.json')
    status = 'complete' if final is not None and cai is not None else 'running_split_queues'
    info = read(RUN / 'controller_status.json', {})
    write = lambda path,value:path.write_text(json.dumps(value,indent=2)+'\n')
    write(OLD / 'controller_status.json', dict(status=status, pid=info.get('pid'),
        old_supervisor_stopped=True, cai_queue=read(OLD / 'cai_QUEUE_PROCESS.json'),
        replacement_giph_run=str(RUN.relative_to(ROOT)), canceled_GiPH_budget=17261, current_GiPH_budget=200))
    if status == 'complete':
        write(OLD / 'FINAL_RESULTS.json', dict(status='complete', methods={'cai':cai,'giph':final},
            giph_run=str(RUN.relative_to(ROOT)), heavy_GiPH_queue_canceled=True))


if __name__ == '__main__': main()
