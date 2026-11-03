#!/usr/bin/env python3
"""Read-only final paired analysis; no model inference or new training."""
import collections
import gzip
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from daglab.teacher_search_ablation import config
from daglab.sft_three_panel import connect,unpack,write_json


def load(path):
    return json.loads(path.read_text())


def compressed(path):
    with gzip.open(path,'rt') as f:
        return json.load(f)


def main():
    cfg=config()
    root=ROOT/cfg['run']
    out=root/'final_analysis'
    out.mkdir(exist_ok=True)
    assert load(root/'controller_status.json')['status']=='complete'
    assert load(root/'SUMMARY.json')['status']=='complete'
    val=connect(ROOT/cfg['validation_cache'])
    metas={key:unpack(blob)['meta'] for key,blob in val.execute('SELECT key,payload FROM records WHERE panel="synthetic_val862"')}
    val.close()
    assert len(metas)==862
    reference=None
    results={}
    selected={}
    seeds=cfg['PPO']['seeds']
    initial_critic={}
    main_keys={}
    warm_keys={}
    for arm in ['strong','no_search']:
        folder=ROOT/cfg['strong_SFT_run'] if arm=='strong' else ROOT/cfg['no_search_SFT_run']
        sft=load(folder/'summary.json')
        assert sft['status']=='complete' and sft['steps']==7553 and sft['presentations']==120827
        md=load(folder/'metadata.json')
        runs=[]
        for seed in seeds:
            p=root/f'ppo_{arm}_seed{seed}'
            s=load(p/'summary.json')
            m=load(p/'metadata.json')
            assert s['status']=='complete' and s['actor_updates']==1024 and s['critic_updates']==1024 and s['prewarm_updates']==200
            assert s['main_trajectories']==2048 and s['main_unique_configs']==512
            assert s['binding']==m['binding']
            if seed in initial_critic:
                assert m['initial_critic_sha256']==initial_critic[seed]
                assert m['main_keys']==main_keys[seed] and m['warm_keys']==warm_keys[seed]
            else:
                initial_critic[seed]=m['initial_critic_sha256']
                main_keys[seed],warm_keys[seed]=m['main_keys'],m['warm_keys']
            assert len(set(m['main_keys']))==512 and not set(m['main_keys'])&set(m['warm_keys'])
            best=s['best']
            path=p/f'eval_u{best["actor_updates"]:04d}_c{best["collection"]:03d}.json.gz'
            ev=compressed(path)
            rows={r['key']:r for r in ev['rows']}
            assert set(rows)==set(metas) and all(r['feasible'] for r in rows.values())
            refs={key:r['HEFT'] for key,r in rows.items()}
            if reference is None:
                reference=refs
            else:
                assert reference==refs
            selected[arm,seed]=rows
            start=s['history'][0]
            assert abs(start['C_HEFT_base_macro']-sft['best']['objective'])<1e-12
            last=s['history'][-1]
            runs.append(dict(seed=seed,best_update=best['actor_updates'],best_C_HEFT=best['C_HEFT_base_macro'],
                config_mean_at_primary_best=best['C_HEFT_config_mean'],last_C_HEFT=last['C_HEFT_base_macro'],
                relative_improvement_over_SFT=1-best['C_HEFT_base_macro']/start['C_HEFT_base_macro']))
        values=[r['best_C_HEFT'] for r in runs]
        results[arm]=dict(SFT_best=sft['best']['objective'],SFT_best_step=sft['best']['step'],
            SFT_last=sft['history'][-1]['objective'],SFT_plan_sha256=md['plan_sha256'],
            SFT_initial_sha256=md['initial_parameters_sha256'],runs=runs,
            PPO_best_mean=statistics.mean(values),PPO_best_sample_std=statistics.stdev(values),
            PPO_last_mean=statistics.mean(r['last_C_HEFT'] for r in runs),
            relative_mean_best_gain_over_SFT=1-statistics.mean(values)/sft['best']['objective'])
    for k in ['SFT_plan_sha256','SFT_initial_sha256']:
        assert results['strong'][k]==results['no_search'][k]
    groups=collections.defaultdict(list)
    for key,m in metas.items():
        groups[m['base_id']].append(key)
    assert len(groups)==500
    base_diffs=[]
    for keys in groups.values():
        base_diffs.append(statistics.mean(
            statistics.mean((selected['no_search',seed][k]['C']-selected['strong',seed][k]['C'])/reference[k] for k in keys)
            for seed in seeds))
    diffs=np.array(base_diffs)
    rng=np.random.default_rng(20261102)
    boot=diffs[rng.integers(0,500,size=(5000,500))].mean(1)
    paired=[]
    for seed in seeds:
        a,b=selected['strong',seed],selected['no_search',seed]
        paired.append(dict(seed=seed,no_search_faster_configs=sum(b[k]['C']<a[k]['C'] for k in metas),
            equal_configs=sum(b[k]['C']==a[k]['C'] for k in metas),
            no_search_slower_configs=sum(b[k]['C']>a[k]['C'] for k in metas),
            config_mean_relative_slowdown_no_search=statistics.mean(b[k]['C']/a[k]['C']-1 for k in metas)))
    by_n=[]
    for n in [20,40,80,120]:
        subset=[ks for ks in groups.values() if metas[ks[0]]['n']==n]
        row=dict(n=n,bases=len(subset),configs=sum(len(ks) for ks in subset))
        for arm in ['strong','no_search']:
            row[arm]=statistics.mean(statistics.mean(
                statistics.mean(selected[arm,seed][k]['C']/reference[k] for k in ks) for ks in subset) for seed in seeds)
        by_n.append(row)
    summary=dict(status='PASS',scope='known VAL862, single SFT seed28 per arm, three paired RL seeds; per-seed VAL-selected best checkpoints',
        arms=results,paired=paired,by_n=by_n,
        no_search_vs_strong_SFT_ratio=results['no_search']['SFT_best']/results['strong']['SFT_best']-1,
        no_search_vs_strong_PPO_best_mean_ratio=results['no_search']['PPO_best_mean']/results['strong']['PPO_best_mean']-1,
        no_search_minus_strong_PPO_base_mean=float(diffs.mean()),
        conditional_base_bootstrap95_no_search_minus_strong=np.quantile(boot,[.025,.975]).tolist(),
        CI_scope='500-base bootstrap conditional on fixed VAL-selected models; does not account for checkpoint selection or independent SFT training uncertainty',
        matching_SFT_initialization_and_sampler=True,matching_PPO_critic_initializations_and_keys=True,
        actual_actor_and_critic_updates_per_PPO_arm=1024,source_or_model_mutations=False)
    write_json(out/'SUMMARY.json',summary)
    report=['# Teacher搜索消融＋PPO：最终结果','','实验完整结束：两组SFT均7epoch/7553更新，六个PPO臂各128collections/2048主轨迹/1024actor更新/1024critic更新/200critic预热。TRAIN17261、原始VAL862；未评价WF/TPC/Parrot。全部已选观测可行；两SFT初始化/采样计划相同，配对RLseed的critic初值/512主配置/256预热配置相同。','','主口径：500底图等权C/HEFT，越低越好。PPO按每seed独立VAL最佳点，允许更新0；不是三个seed中只挑一个。','','|阶段|强Teacher|无搜索Teacher|','|---|---:|---:|',
        '|Teacher参考|0.83135917|0.84362941|',
        f"|SFT最佳|{results['strong']['SFT_best']:.8f}|{results['no_search']['SFT_best']:.8f}|",
        f"|SFT末轮|{results['strong']['SFT_last']:.8f}|{results['no_search']['SFT_last']:.8f}|"]
    for i,seed in enumerate(seeds):
        report.append(f"|PPO seed{seed}最佳|{results['strong']['runs'][i]['best_C_HEFT']:.8f}|{results['no_search']['runs'][i]['best_C_HEFT']:.8f}|")
    report+=[f"|PPO最佳三seed均值|{results['strong']['PPO_best_mean']:.8f}|{results['no_search']['PPO_best_mean']:.8f}|",
        f"|PPO末轮三seed均值|{results['strong']['PPO_last_mean']:.8f}|{results['no_search']['PPO_last_mean']:.8f}|",'',
        '无搜索差距在SFT阶段已存在，PPO没有补回，三seed均值差进一步扩大。相对各自SFT，强Teacher组最佳PPO均值小幅改善，无搜索组更小；无搜索seed22最佳点是更新0。强Teacherseed33末轮略差于SFT起点，不能把最佳点优势说成稳定单调提升。',
        '',f"无搜索SFT相对强组汇总值高{summary['no_search_vs_strong_SFT_ratio']*100:.3f}%；PPO最佳均值高{summary['no_search_vs_strong_PPO_best_mean_ratio']*100:.3f}%。这里是汇总值之比，不是逐配置相对差均值。",'',
        '本轮不支持为给RL留空间而删除教师搜索；默认保留强Teacher。搜索复杂度/标签一致性是否影响学习机制未单独诊断，不能把可能原因写成已证实。强Teacher还含旧最终解保留，因此是当前完整搜索版与纯11候选版的对照。三RLseed共享每组一份SFTseed28，不是三个独立端到端种子；这些VAL已反复用于调参。',
        '',f"500底图条件bootstrap：无搜索减强组C/HEFT均值差={float(diffs.mean()):.8f}，95%区间={summary['conditional_base_bootstrap95_no_search_minus_strong']}。固定已选模型，不覆盖checkpoint选择和SFT随机性。",'',
        '|任务数|配置数|强组PPO最佳三seed均值|无搜索组PPO最佳三seed均值|','|---|---:|---:|---:|']
    report.extend(f"|{r['n']}|{r['configs']}|{r['strong']:.8f}|{r['no_search']:.8f}|" for r in by_n)
    (ROOT/'docs/TEACHER_SEARCH_ABLATION_PPO_RESULTS_20261102.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['arms']},indent=2))
    print('arms',json.dumps(results,indent=2))


if __name__=='__main__':
    main()
