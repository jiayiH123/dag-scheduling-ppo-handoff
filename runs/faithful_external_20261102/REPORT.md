# GiPH预算变更（用户要求）

旧的GiPH完整17261遍历及学习率搜索已停止，本页GiPH旧曲线不再作为活动实验展示。新GiPH按原文每种子200episode、LR0.01，仅训练终点全量评价，三种子均值；原论文4/4特征加必要物理输入。新版进度/结果：[GiPH200报告](../giph_paper200_20261102/report.html)。CAI按原计划继续，本页以下旧GiPH计划文字仅为历史记录。

# CAI / GiPH 论文忠实适配复现实验

当前状态：running_split_queues。

TRAIN：原始17261配置／10000底图；VAL：原始862配置／500底图。没有新增数据、TEST或教师标签训练。
主指标先算逐例C/HEFT，再在底图内平均、最后500底图等权。所有选模使用已知VAL，不是独立测试。

CAI使用作者Graphormer核心模块、论文4层4头64维配置、CAI采样与密集奖励。GiPH使用作者DGL网络、2n轨迹、Categorical采样、原在线REINFORCE更新与关键性统计。必要的预算/累计资源/通信输入与模拟器适配详见协议；不声称原场景发表数值复现。

|运行|最佳VAL主指标|同一策略HEFT起点|选中更新|训练配置呈现|迁移动作|
|---|---:|---:|---:|---:|---:|
|cai_s11_lr0.0005|1.16629645|—|0|120827|0|

GiPH主指标为原生随机起点；HEFT起点单列。学习率和checkpoint由随机起点VAL选择，没有为每实例挑选策略或采样种子。最终5次采样取均值，不取best-of-5。

CAI三种学习率各7epoch，GiPH三种学习率各完整17261配置、每题2n迁移；各自选定学习率后补seed22/33。主阶段seed11；最终3个训练种子分别报告。

图：validation.png/pdf、training_loss.png/pdf；逐例映射、顺序、费用与物理可行性位于各arm/val_*.json.gz。
数据与来源：INPUT_AUDIT.json、sources/MANIFEST.json；数值/梯度/特征/恢复检查：engineering/CHECKS.json。
