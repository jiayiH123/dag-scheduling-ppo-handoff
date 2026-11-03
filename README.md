# DAG Scheduling：SFT → PPO 迁移包

本仓库是 2026-11-03 为换电脑继续 PPO 优化整理的独立快照。它保留实际实验输入、两版教师标签、SFT 权重、baseline 实现和历史对比数据，去掉重复的多 GB SQLite 缓存、本机依赖环境和大多数中间权重。原电脑上的实验不会因拉取或运行本仓库而停止。

## 问题与当前方法

已知 DAG、异构设备池、通信时延、有限累计资源容量、硬费用预算，最小化全局 makespan。当前模型为 32 宽、2 层的 GNN + Attention，32,612 个参数；一次完整 actor 前向为每个任务–池输出 `a,w,c_gap,c_wait`，随后按固定 HEFT 构造顺序，用 `a + w*phi + delta + c_gap*gap + c_wait*wait` 选池。前两项参数范围 ±8，后两项 ±2。物理模拟器负责预算、通信、依赖和资源可行性。

训练数据仅含原始 17,261 条去重合成配置，来自 10,000 张底图，单轮 1,122,860 个动作。没有加入此前 4,000 条增强训练数据，也没有 TPC 或 TEST。所有随附 VAL 都曾用于分析/调参。

## 在另一台电脑开始

建议 Python 3.10，先使用 CPU 路径；当前 PPO/物理模拟器及 GiPH 适配代码均以 CPU 为基准，尚无可直接启用的 CUDA 训练入口。

```bash
git clone https://github.com/jiayiH123/dag-scheduling-ppo-handoff.git
cd dag-scheduling-ppo-handoff
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0
python -m pip install -r requirements.txt
python tools/prepare.py
python tools/verify.py
```

`prepare.py` 校验压缩分片并重建本地 SQLite。图特征和教师逐动作前缀由原来的确定性特征函数、`StateScoreDecoder` 按需生成并缓存到 `.cache/`；每次教师回放都核对 makespan、费用、各任务起止时间。打包时保存了 12 种真实规模×三版标签的特征/前缀校验值。训练、评价脚本不会依赖 `/Users/...` 或本机 conda 路径。

`verify.py` 检查 36 组图特征/标签回放、两版 SFT 各 862 条完整评价复现，以及真实 PPO 更新和精确断点恢复。它只做工程验证，不是新一轮正式调参；结果保存在 `evidence/PORTABLE_VERIFICATION.json`。

## 继续 PPO

默认复用之前 repeat512 配置：T=0.5，actor Adam 3e-5，critic Adam 1e-3，独立 9,153 参数 critic；256 条轨迹预热 critic 200 次更新；512 条主训练配置重复四遍，每批 16 条轨迹，4 遍优化、轨迹 minibatch=8，128 次收集，最多 1,024 次 actor 更新；gamma=lambda=1、clip=0.2、entropy=0.001、KL stop=0.02，无教师 anchor。

```bash
# 新建 PPO：从 Teacher64v2 的 SFT best 开始
python tools/train_ppo.py --arm strong --seed 41 --out runs/portable_ppo_strong_s41

# 从无搜索教师的 SFT best 开始
python tools/train_ppo.py --arm no_search --seed 41 --out runs/portable_ppo_nosearch_s41

# 独立的参数探索示例（另开输出目录，保留对照）
python tools/train_ppo.py --arm strong --seed 22 --actor-lr 1e-5 --entropy 0.002 --out runs/portable_ppo_lr1e5_s22

# 新电脑上自己启动的 PPO 可用同一组参数恢复
python tools/train_ppo.py --arm strong --seed 41 --out runs/portable_ppo_strong_s41 --resume
```

可调参数见 `--help`，包括 source checkpoint、actor/critic LR、温度、entropy、clip、KL 门限及评价间隔；默认每 100 次实际 actor 更新评价 VAL862。`--collections` 允许 1～128，显式缩短工程试验；默认仍是完整 128。每次保存实际参数、样本顺序、loss、VAL 曲线、训练/评价耗时及模型/优化器/RNG 状态。默认从 SFT **只加载模型权重**，新建 PPO 优化器；`--resume` 恢复本机新 PPO 的完整状态。

## SFT 与现成 checkpoint

|版本|教师定义|best 位置|best step|VAL862 底图宏平均 C/HEFT|
|---|---|---|---:|---:|
|`strong`|11 个固定 HEFT 回放候选（含 BHEFT）+64 次局部提案；必要时保留旧教师最终结果|`runs/sft_teacher64_bheft_original_20261102/best.pt`|7553|0.86618994|
|`no_search`|同样 11 个候选，0 次局部搜索，不保留旧搜索结果|`runs/teacher_search_ablation_ppo_20261102/no_search_sft/best.pt`|7500|0.88196727|

两目录均包含 `best.pt`、`last.pt`、原始配置、训练记录和每 100 次更新的完整 VAL 评价记录；没有携带全部中间模型权重。`last.pt` 与 `best.pt` 可能不属于同一步，尤其无搜索版本。历史两版 SFT 都是同一个训练 seed28，不能称为三个独立 SFT 种子。

```bash
python tools/train_sft.py --teacher strong --out runs/portable_sft_strong
python tools/train_sft.py --teacher no_search --out runs/portable_sft_nosearch
```

默认均为从零开始、Adam0.001、batch16、7epoch/7,553次更新、每100次更新评价。`--resume` 恢复新实验，`--max-updates` 可显式标记短工程运行。两版教师底图和物理条件完全相同，只改变标签。

## 评价数据

|面板|规模|用途|
|---|---|---|
|`synthetic_val862`|862 配置、500 底图|原始合成 VAL；20/40/80/120任务|
|`wf960`|960 配置、120 次公开执行、83 个拓扑组、27 个应用/引擎配对|完整旧 WF 评价范围，全部保留|
|`wf400`|400 配置、25 次执行、14 个应用/引擎配对|后来聚焦的 WF 面板，与 WF960 分开|
|`parrot498`|498 配置、8 个模板|参考公开应用模式构造的工作流，**不是公开原始执行轨迹**|

WF 底图来自公开执行记录；资源池、网络、费用和预算采用本研究的模拟适配。`data/wf_application_panels_20261101/REGISTRY.json` 保留应用名单和完整旧面板成员。WF 报告按应用/引擎、任务数、应用×任务数分别汇总。没有按单个模型失败样例静默删数据。

```bash
python tools/evaluate.py --checkpoint runs/sft_teacher64_bheft_original_20261102/best.pt --panel wf960 --out runs/portable_eval/wf960.json.gz
python tools/evaluate.py --checkpoint runs/sft_teacher64_bheft_original_20261102/best.pt --panel wf400 --out runs/portable_eval/wf400.json.gz
```

合成主指标先算逐例 C/HEFT，再在底图内平均、最后底图等权；配置均值另列，不能与主指标混用。WF 额外按执行→拓扑→应用等权，逐应用结果在输出 JSON 的 `summary.groups`。

## Baseline 与画图

包含 RandomMean5、Greedy、HEFT、普通 PEFT、PEFT-price0.1/0.3/1/3/10、JoinLookahead、CostFirst、MSLBL适配、BHEFT、BUDA适配、Portfolio11、旧Teacher64、新Teacher64v2，以及旧适配版 DAG-CAI-DR/BiGAT、Graphormer、GiPH 随机/HEFT起点的代码及同一 VAL862 对比结果。BUDA 当前适配存在已记录的尺度/账本问题，不作为已忠实复现的原文结果。

```bash
# 原来那张 21 条 baseline + 每100步SFT 的完整图：直接读取保存的逐例结果，无需训练
python figures/gen_fig_val862_all_baselines_20261102.py

# 在同一物理模型下重跑启发式/教师（all含18种，可指定方法减少耗时）
python tools/run_baselines.py --panel synthetic_val862 --methods all --out runs/portable_baselines/val862.json.gz

# 把新 PPO 曲线加入同一组 baseline 对比
python tools/plot_comparison.py --curve runs/portable_ppo_strong_s41/curve.json --out runs/portable_comparison/all_methods
```

旧图中的学习型 baseline 是旧适配结果，不能冒充下面正在进行的作者核心复现。GOODRL/WeCAN 尚无本场景已完成的测量值，不会虚构它们的横线。

## GiPH / DAG-CAI-DR 未完成复现的迁移

打包时 GiPH 三个种子均已训练满200episode，正在做随机/HEFT起点的全量VAL；DAG-CAI-DR已完成第一个学习率七轮，第二个学习率进行中。**这不是两项复现已完成的声明。** 精确快照和文件哈希在 `evidence/BASELINE_SNAPSHOT.json`，各目录 `resume.pt` 是可恢复进度的依据；复制的进度文本可能比 checkpoint 更新。

作者核心模块固定版本、必要物理适配、特征/梯度/更新等价性与恢复检查、best/resume 权重均已携带；不携带 Mac 的 DGL/NumPy 二进制。GiPH 按论文200episode、LR0.01、gamma0.97、每题2n步、原生4/4特征，另加预算/费用/容量输入；旧的17261episode×学习率搜索已取消。

```bash
python -m pip install -r requirements-baselines.txt
# 在新机器显式迁移数据/路径绑定，保留权重、Adam、RNG与计数，再完成所选种子的评价
python tools/resume_baseline.py --method giph --seed 11
python tools/resume_baseline.py --method giph --seed 22
python tools/resume_baseline.py --method giph --seed 33

# 继续 CAI 未完成的学习率/种子队列
python tools/resume_baseline.py --method cai
```

重建 SQLite 的文件字节会改变，因此必须用这个入口记录绑定迁移，不能直接绕过原检查。迁移记录在 `evidence/portable_baseline_migration/`。未完成的半次 VAL 不视为缓存结果；换电脑会从该次评价起点重测，**不会因此重新训练已完成的200episode**。不要在两台机器同时写同一个同步目录。

## GOODRL 与 WeCAN 资料

`papers/` 包含 GOODRL、WeCAN v2、GiPH、DAG-CAI-DR、BUDA、MSLBL 论文及来源校验；WeCAN 还保存论文 LaTeX 源码。

`third_party/GOODRL/` 是作者仓库固定 commit `ff4b66c9e18aff1ffb3f156201a1b55093287641` 的代码、许可证、工作流文件及小型验证/权重文件；为控制仓库体积，四份大体积生成 HEFT memory 缓存用固定版本下载清单代替。若要运行作者原始训练流程，先执行 `python tools/fetch_goodrl_cache.py`，再按该目录 README 安装 PyG/Gym 等独立依赖并运行 step1/step2/step3。它原生研究动态多工作流平均 flowtime，尚未适配本研究的单DAG硬预算makespan；`docs/GOODRL_CRITIC_TRANSFER_PLAN_20261027.md` 解释区别及已有迁移尝试。

`third_party/WeCAN/README.md` 明确记录：当前没有找到可核实的官方代码入口，本仓库**不含伪称官方的 WeCAN 实现**；提供论文/源码/超参数与适配清单，供后续复现。论文材料不是算法实现。

## 文件结构

```text
data/packed/                物理输入与教师映射的压缩分片（真正随Git上传）
data/MANIFEST.json          配置数、成员/分片校验和
data/runtime/               prepare生成，本地缓存，不上传
daglab/ vendor/             模型、PPO组件、可行性模拟器和启发式
tools/                     可迁移的prepare/verify/SFT/PPO/eval/resume/plot入口
scripts/ figures/           原实验相关代码与历史图重建脚本
configs/ protocols/        原参数及实验方案；实际新参数写入新run
runs/...                   选定SFT权重/曲线、逐例baseline、未完成外部模型快照
third_party/ papers/       固定版本作者代码、论文、许可证及来源
docs/ evidence/            实验结论、范围说明、迁移改动和验证报告
```

`evidence/PORTABILITY_CHANGES.json` 列出为迁移修改过的旧代码；网络结构、物理约束及默认 PPO 数值配置保持原设置。依赖版本、源仓库 commit、权重和数据文件校验供另一台电脑核对；新实验请使用独立 `runs/portable_*` 目录，保留历史结果。
