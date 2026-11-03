#!/usr/bin/env python3
"""Human-readable progress and final results; no training or model selection changes."""
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/faithful_external_20261102'
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'runs/matplotlib-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def paired_analysis(summary,method):
    totals={}; meta={}; count=0
    for run in summary['runs']:
        directory=RUN/run['arm']
        seeds=[101,102,103,104,105] if method=='giph' else [101]
        for seed in seeds:
            with gzip.open(directory/f'val_final_seed{seed}.json.gz','rt') as stream:
                rows=json.load(stream)['rows']
            assert len(rows)==862
            for row in rows:
                key=row['key']; meta[key]=row
                for variant,value in row['methods'].items():
                    totals.setdefault(variant,collections.defaultdict(float))[key]+=value['C']
            count+=1
    sft_path=ROOT/'runs/sft_teacher64_bheft_original_20261102/val_synthetic_val862_07553.json.gz'
    with gzip.open(sft_path,'rt') as stream:
        sft={r['key']:r for r in json.load(stream)['rows']}
    assert set(sft)==set(meta)
    results={}
    for variant,values in totals.items():
        groups=collections.defaultdict(list); by_n=collections.defaultdict(lambda:collections.defaultdict(list))
        by_family=collections.defaultdict(lambda:collections.defaultdict(list))
        for key,value in values.items():
            row=meta[key]; C=value/count; ref=row['references']['HEFT']['C']; base=row['base_id']
            assert row['references']['HEFT']==sft[key]['references']['HEFT']
            groups[base].append((C-sft[key]['C'])/ref)
            by_n[str(row['n'])][base].append(C/ref)
            by_family[str(row['family'])][base].append(C/ref)
        differences=np.array([sum(g)/len(g) for g in groups.values()])
        rng=np.random.default_rng(20261102)
        boot=differences[rng.integers(0,len(differences),(2000,len(differences)))].mean(1)
        aggregate=lambda g:sum(sum(v)/len(v) for v in g.values())/len(g)
        results[variant]=dict(mean_C_HEFT_difference_vs_current_SFT=float(differences.mean()),
            conditional_500_base_bootstrap_CI95=[float(v) for v in np.quantile(boot,[.025,.975])],
            CI_scope='fixed VAL-selected policies and averaged sampling; not training/selection uncertainty',
            by_n={n:aggregate(g) for n,g in by_n.items()},by_family={n:aggregate(g) for n,g in by_family.items()})
    return dict(variants=results,SFT_reference_sha256=hashlib.sha256(sft_path.read_bytes()).hexdigest())


def main():
    giph_replaced = (RUN/'GIPH_HEAVY_BUDGET_STOP.json').exists()
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'font.size':10})
    colors = ['#0072B2','#009E73','#E69F00','#CC79A7','#56B4E9']
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    lossfig, lossaxes = plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    table = []; results = {}
    for ai,method in enumerate(['cai','giph']):
        directories = [] if method=='giph' and giph_replaced else sorted(p for p in RUN.glob(method+'_s*') if p.is_dir())
        for index,directory in enumerate(directories):
            curve = load(directory/'curve.json',[]); complete = load(directory/'COMPLETE.json')
            if curve:
                xs=[p['unit'] for p in curve]; ys=[p['primary'] for p in curve]
                axes[ai].plot(xs,ys,marker='o',ms=3,lw=1.3,label=directory.name,color=colors[index%len(colors)])
            if (directory/'training.jsonl').exists():
                lines=(directory/'training.jsonl').read_text().splitlines()
                records=[]
                for line in lines:
                    try:records.append(json.loads(line))
                    except json.JSONDecodeError:pass
                stride=max(1,len(records)//3000); shown=records[::stride]
                if shown:
                    lossaxes[ai].plot([r['update'] for r in shown],[r['loss'] for r in shown],lw=.65,
                        alpha=.7,label=directory.name,color=colors[index%len(colors)])
            if complete:
                best=complete['best']; variants=best['variants']
                table.append([directory.name, f"{best['primary']:.8f}",
                    f"{variants['HEFT']['base_macro']:.8f}" if 'HEFT' in variants else '—',
                    str(best['unit']),str(complete['presentations']),str(complete['moves'])])
        for ax in [axes[ai],lossaxes[ai]]:
            ax.set_title('DAG-CAI-DR (paper Graphormer)' if method=='cai' else 'GiPH (author DGL; random start primary)')
            ax.set_xlabel('Optimizer updates' if method=='cai' else 'Training episodes (update if moves feasible)')
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=6)
        for value,label,color in [(1.,'HEFT','#777777'),(.8661899379000059,'Current SFT','#D55E00'),
                                  (.831359167419895,'Teacher64+BHEFT-v2','#009E73')]:
            axes[ai].axhline(value,ls='--',lw=.9,label=label,color=color)
        axes[ai].set_ylabel('VAL862 C/HEFT (500-base macro; lower is better)')
        axes[ai].legend(fontsize=6,ncol=2)
        lossaxes[ai].set_ylabel('Policy loss (symlog; display downsample <=3,000 points)')
        lossaxes[ai].set_yscale('symlog',linthresh=.1)
    for name,figure in [('validation',fig),('training_loss',lossfig)]:
        for ext in ['png','pdf']:
            temporary=RUN/f'{name}.tmp.{ext}'; figure.savefig(temporary,dpi=300); temporary.replace(RUN/f'{name}.{ext}')
        plt.close(figure)
    controller=load(RUN/'controller_status.json',{})
    lines=['# CAI / GiPH 论文忠实适配复现实验','',f"当前状态：{controller.get('status','engineering')}。",'',
        'TRAIN：原始17261配置／10000底图；VAL：原始862配置／500底图。没有新增数据、TEST或教师标签训练。',
        '主指标先算逐例C/HEFT，再在底图内平均、最后500底图等权。所有选模使用已知VAL，不是独立测试。','',
        'CAI使用作者Graphormer核心模块、论文4层4头64维配置、CAI采样与密集奖励。'
        'GiPH使用作者DGL网络、2n轨迹、Categorical采样、原在线REINFORCE更新与关键性统计。'
        '必要的预算/累计资源/通信输入与模拟器适配详见协议；不声称原场景发表数值复现。','',
        '|运行|最佳VAL主指标|同一策略HEFT起点|选中更新|训练配置呈现|迁移动作|',
        '|---|---:|---:|---:|---:|---:|']
    lines += ['|'+'|'.join(row)+'|' for row in table]
    lines += ['','GiPH主指标为原生随机起点；HEFT起点单列。学习率和checkpoint由随机起点VAL选择，'
        '没有为每实例挑选策略或采样种子。最终5次采样取均值，不取best-of-5。', '',
        'CAI三种学习率各7epoch，GiPH三种学习率各完整17261配置、每题2n迁移；'
        '各自选定学习率后补seed22/33。主阶段seed11；最终3个训练种子分别报告。', '',
        '图：validation.png/pdf、training_loss.png/pdf；逐例映射、顺序、费用与物理可行性位于各arm/val_*.json.gz。',
        '数据与来源：INPUT_AUDIT.json、sources/MANIFEST.json；数值/梯度/特征/恢复检查：engineering/CHECKS.json。']
    for method in ['cai','giph']:
        summary=load(RUN/f'{method}_FINAL.json')
        if summary:
            results[method]=summary
            analysis=paired_analysis(summary,method)
            (RUN/f'{method}_PAIRED_ANALYSIS.json').write_text(json.dumps(analysis,indent=2)+'\n')
            lines += ['',f"## {method.upper()} 最终结果",'',f"选中学习率：{summary['selected_rate']}。",'',
                '|训练运行|选中更新|原生起点C/HEFT|HEFT起点C/HEFT|','|---|---:|---:|---:|']
            primary='CAI' if method=='cai' else 'random'
            for row in summary['runs']:
                lines += [f"|{row['arm']}|{row['selected_unit']}|{row['metrics'][primary]['base_macro']:.8f}|"
                          +(f"{row['metrics']['HEFT']['base_macro']:.8f}|" if method=='giph' else '—|')]
            for variant,values in summary['final_three_seed_mean'].items():
                scores=[r['metrics'][variant]['base_macro'] for r in summary['runs']]
                lines += ['',f"{variant}：3训练种子均值 **{values['base_macro']:.8f}**，"
                    f"种子间样本标准差{np.std(scores,ddof=1):.8f}；配置均值{values['config_mean']:.8f}。"]
            lines += ['', '分任务数/家族及相对固定SFT的条件底图bootstrap见'+f'`{method}_PAIRED_ANALYSIS.json`。']
    report='\n'.join(lines)+'\n'
    if giph_replaced:
        report = ('# GiPH预算变更（用户要求）\n\n旧的GiPH完整17261遍历及学习率搜索已停止，'
            '本页GiPH旧曲线不再作为活动实验展示。新GiPH按原文每种子200episode、LR0.01，'
            '仅训练终点全量评价，三种子均值；原论文4/4特征加必要物理输入。'
            '新版进度/结果：[GiPH200报告](../giph_paper200_20261102/report.html)。'
            'CAI按原计划继续，本页以下旧GiPH计划文字仅为历史记录。\n\n') + report
    (RUN/'REPORT.md').write_text(report)
    (ROOT/'docs/FAITHFUL_EXTERNAL_RESULTS_20261102.md').write_text(report)
    import html
    (RUN/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Faithful CAI/GiPH</title>'
        '<style>body{font:16px system-ui;max-width:1300px;margin:32px auto}img{width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>CAI / GiPH faithful adaptation</h1><img src="validation.png"><img src="training_loss.png">'
        '<pre>'+html.escape(report)+'</pre>')
    if controller.get('status')=='complete' and not giph_replaced:
        (RUN/'FINAL_RESULTS.json').write_text(json.dumps(dict(status='complete',methods=results),indent=2)+'\n')


if __name__=='__main__':main()
