# 外部学习baseline接入当前原始VAL862

本次仅汇总已有逐例结果，不新训练/推理，不更改学习基线权重或旧输入。已逐例核验862 key、500底图identity、HEFT和旧Teacher reference一致，并核查已存可行性标记。当前全量面板不是历史WF200/TPC800；以下只谈当前相同synthetic VAL862。

前期缺失原因：本轮SFT/PPO出图读取启发式参考缓存，CAI/GiPH的独立外部结果没有接入该缓存/报表。不是不存在结果，也不应由结果较差而排除。此次接入统一汇总，训练曲线及其源码/绑定保持不变。

|方法|当前底图等权C/HEFT|配置均值C/HEFT|
|---|---:|---:|
|Current_strong_SFT|0.86618994|0.84694829|
|Current_strong_PPO_mean3|0.86190263|0.84126028|
|MSLBL|0.96188216|0.95577977|
|BHEFT|0.96790104|0.94449194|
|HEFT|1.00000000|1.00000000|
|CAI_DR_BiGAT_visible_selected_epoch1|1.25328121|1.27003596|
|CAI_DR_Graphormer_visible_selected_epoch7|1.25568923|1.27093919|
|GiPH_HEFT_2n|0.92060078|0.91132178|
|GiPH_random_2n|1.19411657|1.13294437|

## 已有适配与限制

DAG-CAI-DR适配学习任务优先级，按就绪任务选择后做预算可行EFT分配；BiGAT与Graphormer两版本都完成原始17261合成TRAIN的7epoch，各训练图4份采样方案，训练seed11。保留历史以配置均值选出的统一epoch（BiGAT1、Graphormer7），没有因改为底图均值而重新选epoch。BiGAT末轮及noDR补充结果同时存JSON。

GiPH适配学习对当前任务→池映射做迁移，训练只1epoch、每图16步，评价可2n步并保留路径上最佳解，包含HEFT起点本身。主对照必须保留较强HEFT起点，随机起点另列。该方法评价需多次网络调用，与本模型一次forward构造不同，后续需单独列推理成本。确定性迁移及训练截断是本地适配选择。

它们是统一预算/通信/累计资源物理模型下的适配版本，不是原论文原场景未经修改复现；外部学习方法目前只有seed11，GiPH训练预算也小于当前7epochSFT，尚不是充分调优、多seed、等预算的最终论文比较。这些初版分数只能说明当前实现/配方表现，不能推出原方法无效。

当前WF400、编程320未完成这两类模型的对应评价；若恢复这些面板，应重评相同完整成员，不把旧WF200/TPC800表直接搬入。此处没有启动任何补训或新面板评价。

全部方法当前统一口径与逐例来源SHA见runs/teacher_search_ablation_ppo_20261102/final_analysis/EXTERNAL_BASELINES_VAL862.json。
