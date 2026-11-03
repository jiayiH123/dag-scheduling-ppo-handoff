#!/usr/bin/env python3
"""Replot saved full-VAL evaluations; no model inference or training."""
import collections
import csv
import gzip
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runs/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from daglab.structured_dataset import sha256

RUN = ROOT / 'runs/sft_teacher64_bheft_original_20261102'
ABLATION = ROOT / 'runs/teacher_search_ablation_ppo_20261102'
OUT = RUN / 'all_baselines_val862'
NAMES = ['HEFT', 'PEFT', 'JoinLookahead', 'CostFirst', 'MSLBL-adapted',
         'PEFT-price0.1', 'PEFT-price0.3', 'PEFT-price1', 'PEFT-price3', 'PEFT-price10']
PALETTE = ['#0072B2', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#333333', '#8A7810']
STYLES = ['--', ':', '-.', (0, (5, 2, 1, 2)), (0, (2, 2))]


def main():
    OUT.mkdir(exist_ok=True)
    sources = {}

    def load(path):
        path = Path(path)
        sources[str(path.relative_to(ROOT))] = sha256(path)
        if path.suffix == '.gz':
            with gzip.open(path, 'rt') as stream:
                return json.load(stream)
        return json.loads(path.read_text())

    curve = load(RUN / 'val_synthetic_val862_curve.json')
    steps = [p['step'] for p in curve]
    assert steps == [0] + list(range(100, 7501, 100)) + [7553]
    checkpoints = [load(RUN / f'val_synthetic_val862_{step:05d}.json.gz')['rows'] for step in steps]
    current = {r['key']: r for r in checkpoints[-1]}
    assert len(current) == 862 and len({r['base_id'] for r in current.values()}) == 500

    def metrics(values):
        assert set(values) == set(current)
        groups = collections.defaultdict(list)
        for key, row in current.items():
            groups[row['base_id']].append(values[key] / row['references']['HEFT']['C'])
        return {
            'base_macro': sum(sum(g) / len(g) for g in groups.values()) / 500,
            'config_mean': sum(sum(g) for g in groups.values()) / 862,
        }

    evaluations = []
    for point, rows in zip(curve, checkpoints):
        data = {r['key']: r for r in rows}
        assert len(rows) == 862 and set(data) == set(current)
        for key, row in data.items():
            assert row['feasible'] and row['base_id'] == current[key]['base_id']
            assert row['references'] == current[key]['references']
        value = metrics({key: row['C'] for key, row in data.items()})
        assert abs(value['base_macro'] - point['C_over']['HEFT']) < 1e-12
        evaluations.append(dict(step=point['step'], **value))

    baselines = []

    def add(name, values, description):
        baselines.append(dict(method=name, **metrics(values), definition=description))

    native = load(ABLATION / 'final_analysis/TEN_RULES_NATIVE_AND_FIXED_VAL862.json')
    raw = {r['key']: r for r in native['rows']}
    assert len(raw) == 862 and set(raw) == set(current)
    for key, row in raw.items():
        assert row['base_id'] == current[key]['base_id']
        assert row['HEFT'] == current[key]['references']['HEFT']['C']
    for index, name in enumerate(NAMES):
        add(name, {key: r['native_C'][index] for key, r in raw.items()},
            'Native rule: own task ordering and pool allocation, with shared physical constraints.')
        stored = next(r for r in native['native_ranking'] if r['method'] == name)
        assert abs(baselines[-1]['base_macro'] - stored['base_macro_C_HEFT']) < 1e-12

    for name in ['BHEFT', 'Greedy', 'RandomMean5', 'BUDA']:
        add(name, {key: r['references'][name]['C'] for key, r in current.items()},
            'Existing same-VAL reference; local physical-model adaptation where applicable.')

    portfolio = load(ABLATION / 'no_search_teacher_val862/RESULTS.json')
    portfolio_rows = {r['key']: r for r in portfolio['rows']}
    assert len(portfolio_rows) == 862 and set(portfolio_rows) == set(current)
    for key, row in portfolio_rows.items():
        assert row['feasible'] and row['base_id'] == current[key]['base_id']
        assert row['HEFT'] == current[key]['references']['HEFT']['C']
        assert row['C'] == min(v[0] for v in row['candidate_values'])
    add('Portfolio11 (no search)', {key: r['C'] for key, r in portfolio_rows.items()},
        'Best of eleven candidate mappings replayed in fixed HEFT order; zero search proposals.')
    add('Teacher64 (legacy)', {key: r['references']['Teacher64']['C'] for key, r in current.items()},
        'Historical ten-rule, fixed-HEFT replay plus 64 local proposals.')
    teacher = load(RUN / 'new_teacher_val862/RESULTS.json')
    teacher_rows = {r['key']: r for r in teacher['rows']}
    assert len(teacher_rows) == 862 and set(teacher_rows) == set(current)
    for key, row in teacher_rows.items():
        assert row['feasible'] and row['base_id'] == current[key]['base_id']
        assert row['HEFT_C'] == current[key]['references']['HEFT']['C']
        assert row['C'] <= portfolio_rows[key]['C']
    add('Teacher64+BHEFT-v2', {key: r['C'] for key, r in teacher_rows.items()},
        'Eleven fixed-HEFT seeds, 64 fresh proposals, retain legacy final if better/equal; SFT teacher.')

    external = load(ABLATION / 'final_analysis/EXTERNAL_BASELINES_VAL862.json')
    assert external['status'] == 'PASS' and external['exact_keys_and_baseline_references_match']
    assert external['source_current_rows_sha256'] == sha256(RUN / 'val_synthetic_val862_07553.json.gz')
    for artifact in external['artifacts']:
        path = ROOT / artifact['path']
        assert sha256(path) == artifact['sha256']
        sources[artifact['path']] = artifact['sha256']
    for key, label in [
        ('CAI_DR_BiGAT_visible_selected_epoch1', 'DAG-CAI-DR / BiGAT'),
        ('CAI_DR_Graphormer_visible_selected_epoch7', 'DAG-CAI-DR / Graphormer'),
        ('GiPH_HEFT_2n', 'GiPH / HEFT start / 2n'),
        ('GiPH_random_2n', 'GiPH / random start / 2n'),
    ]:
        m = external['metrics'][key]
        assert m['configs'] == 862 and m['bases'] == 500
        baselines.append(dict(method=label, base_macro=m['C_HEFT_base_macro'],
                              config_mean=m['C_HEFT_config_mean'],
                              definition=f'Existing adapted learned control: {key}; historical selection preserved.'))

    assert len(baselines) == 21 and len({b['method'] for b in baselines}) == 21
    style = {}
    for i, b in enumerate(baselines):
        style[b['method']] = dict(color=PALETTE[i % len(PALETTE)], ls=STYLES[i // len(PALETTE)], lw=1.15)
    style['Teacher64+BHEFT-v2'] = dict(color='#009E73', ls='-', lw=1.6)
    style['Portfolio11 (no search)'] = dict(color='#CC79A7', ls='-.', lw=1.5)
    style['Teacher64 (legacy)'] = dict(color='#333333', ls=':', lw=1.5)

    with (OUT / 'sft_curve.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'base_macro', 'config_mean'])
        writer.writeheader(); writer.writerows(evaluations)
    with (OUT / 'baseline_values.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'base_macro', 'config_mean', 'definition'])
        writer.writeheader(); writer.writerows(sorted(baselines, key=lambda b: b['base_macro']))

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': .13, 'pdf.fonttype': 42,
        'savefig.dpi': 300,
    })
    for metric, suffix, description in [
        ('base_macro', '', 'Primary: equal weight per base DAG'),
        ('config_mean', '_config_mean', 'Supplementary: equal weight per configuration'),
    ]:
        fig = plt.figure(figsize=(17.8, 8.2))
        full = fig.add_axes([.055, .25, .295, .63])
        zoom = fig.add_axes([.405, .25, .295, .63])
        legend = fig.add_axes([.725, .16, .27, .73])
        legend.set_axis_off(); legend.set_xlim(0, 1); legend.set_ylim(0, 1)
        values = [r[metric] for r in evaluations]
        best_index = min(range(len(values)), key=values.__getitem__)
        assert best_index == len(values) - 1
        for ax in [full, zoom]:
            for b in baselines:
                ax.axhline(b[metric], zorder=1, **style[b['method']])
            ax.plot(steps, values, color='#D55E00', lw=2.4, marker='o', ms=2.6, zorder=4)
            ax.plot(steps[best_index], values[best_index], color='#D55E00', marker='*', ms=12, zorder=5)
            ax.set_xlim(-100, 7900)
            ax.set_xlabel('SFT optimizer updates')
            ax.set_ylabel('Makespan / HEFT (lower is better)')
            ax.xaxis.set_major_locator(MultipleLocator(2000))
        lower = min([b[metric] for b in baselines] + values) - .012
        upper_zoom = 1.11
        full.set_ylim(lower, max(b[metric] for b in baselines) + .10)
        zoom.set_ylim(lower, upper_zoom)
        full.axhspan(lower, upper_zoom, color='#0072B2', alpha=.055, zorder=0)
        full.set_title('(a) All methods: full range', loc='left')
        zoom.set_title('(b) Same data: zoom near SFT', loc='left')
        zoom.yaxis.set_major_locator(MultipleLocator(.05))
        zoom.annotate(f'Best / final: {values[best_index]:.5f}\nupdate {steps[best_index]}',
                      xy=(steps[best_index], values[best_index]), xytext=(4400, .949),
                      fontsize=9, color='#D55E00', bbox=dict(facecolor='white', edgecolor='none', alpha=.9),
                      arrowprops=dict(arrowstyle='-', color='#D55E00', lw=.8))

        legend.text(0, 1.025, 'Method / line key', weight='bold', fontsize=10)
        legend.text(.99, 1.025, 'C / HEFT', ha='right', weight='bold', fontsize=10)
        entries = [dict(method='SFT (ours; final)', value=values[-1],
                        style=dict(color='#D55E00', ls='-', lw=2.4))]
        entries += [dict(method=b['method'], value=b[metric], style=style[b['method']])
                    for b in sorted(baselines, key=lambda b: b[metric])]
        for i, entry in enumerate(entries):
            y = .98 - i * .0435
            legend.plot([0, .09], [y, y], **entry['style'])
            legend.text(.115, y, entry['method'], va='center', fontsize=8.4,
                        weight='bold' if i == 0 else 'normal')
            legend.text(.99, y, f"{entry['value']:.5f}", ha='right', va='center', fontsize=8.4,
                        weight='bold' if i == 0 else 'normal')
        fig.suptitle('Original synthetic VAL862: SFT versus all baseline references',
                     x=.055, ha='left', y=.965, fontsize=16, weight='bold')
        fig.text(.055, .915,
                 f'{description}  |  862 configurations / 500 DAGs  |  77 full-panel evaluations; no smoothing',
                 fontsize=10, color='#444444')
        fig.text(.055, .145,
                 'SFT: original 17,261 TRAIN configurations; Teacher64+BHEFT-v2 labels; seed 28; 7 epochs / 7,553 updates.',
                 fontsize=9)
        fig.text(.055, .112,
                 'Evaluations: update 0, every 100 updates, and final update 7,553. Each baseline is a fixed horizontal reference.',
                 fontsize=9)
        fig.text(.055, .079,
                 'Rules use native ordering; Portfolio11 selects fixed-HEFT mapping replays. Learned controls are local adaptations (seed 11).',
                 fontsize=9)
        fig.text(.055, .046,
                 'CAI: 7 training epochs, historical selected epoch; GiPH: 1 training epoch / 16 moves per graph, up to 2n evaluation moves.',
                 fontsize=9, color='#444444')
        for ext in ['png', 'pdf']:
            fig.savefig(OUT / f'val_all_baselines{suffix}.{ext}', dpi=300)
        plt.close(fig)

    manifest = dict(status='PASS', configs=862, bases=500, evaluations=len(evaluations),
                    steps=steps, baselines=baselines, sft_curve=evaluations,
                    metric='Per-case C/HEFT; average within each base DAG, then average over 500 DAGs.',
                    no_smoothing=True, no_new_training=True, no_new_inference=True,
                    source_sha256=sources, script_sha256=sha256(Path(__file__)))
    (OUT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    report = [
        '# VAL862：SFT 与全部 baseline 曲线', '',
        '绘图使用已保存的全部逐例结果，没有启动训练或重新推理，也没有改变原始图。', '',
        '主图：`runs/sft_teacher64_bheft_original_20261102/all_baselines_val862/val_all_baselines.png/pdf`。',
        '左图显示完整范围，右图放大相同数据；右侧线型表列出全部数值。', '',
        'SFT 为新 Teacher64+BHEFT-v2 标签训练的原始17261配置、seed28、7epoch版本。'
        '使用0、100、200、…、7500、7553更新的77次全量评价，逐次核验同一862 key、500 base_id、HEFT引用及可行性标记；没有平滑或插值。', '',
        '主指标为先对每配置计算C/HEFT，再在底图内平均，最后对500张底图等权平均。'
        '补充图 `val_all_baselines_config_mean.png/pdf` 使用862配置等权，和历史配置均值口径对应，勿交叉比较数值。', '',
        '十种单独规则使用各自原生顺序；Portfolio11是11个候选映射按固定HEFT重放后逐例取优、没有搜索。'
        '当前Teacher为Portfolio11起点+64次新提案，并保留旧Teacher最终解以防退化；旧Teacher另列。', '',
        '按用户所列方法加入全部13条参照，另外保留旧Teacher，以及已有Greedy、RandomMean5、BUDA和CAI/GiPH两种版本，共21条水平参照。'
        'CAI历史选模不变；GiPH同时保留HEFT及随机起点。外部方法是本地适配，训练量并不完全相等，详见EXTERNAL_BASELINES_CURRENT_VAL862_20261102.md。', '',
        '|方法|底图等权 C/HEFT|配置均值 C/HEFT|', '|---|---:|---:|',
        f"|SFT末轮/最佳 step7553|{evaluations[-1]['base_macro']:.8f}|{evaluations[-1]['config_mean']:.8f}|",
    ]
    report += [f"|{b['method']}|{b['base_macro']:.8f}|{b['config_mean']:.8f}|"
               for b in sorted(baselines, key=lambda b: b['base_macro'])]
    report += ['', '绘图脚本保存于`figures/gen_fig_val862_all_baselines_20261102.py`；'
               '输出目录保存baseline_values.csv、sft_curve.csv和带输入SHA256的MANIFEST.json。']
    (ROOT / 'docs/VAL862_ALL_BASELINES_SFT_PLOT_20261102.md').write_text('\n'.join(report) + '\n')
    print(json.dumps(dict(status='PASS', output=str(OUT), evaluations=len(evaluations),
                          baselines=len(baselines), final=evaluations[-1]), indent=2))


if __name__ == '__main__':
    main()
