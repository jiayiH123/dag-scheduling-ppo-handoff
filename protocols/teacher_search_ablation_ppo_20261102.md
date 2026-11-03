# Teacher局部搜索消融：同配方SFT＋repeat512 PPO

2026-11-02，用户明确同意启动两种Teacher的SFT/RL对照，并指定参考policy_alignment_coverage_20261027/repeat512。当前强TeacherSFT完整7epoch已完成，主指标最佳step7553，C/HEFT底图均值.8661899379；旧历史repeat512报告为配置均值，禁止直接混比。

## 固定研究问题与数据

是否不使用教师局部搜索能在相同PPO预算下获得更好的最终确定性策略，而非仅更大的相对增量。A=no-search:11启发式固定HEFT候选中(C,cost)最优，0修改提案，不允许保留任何旧Teacher64最终解；B=当前Teacher64+BHEFT-v2（含旧最终解保留）。两者同17261TRAIN/10000底图/1122860动作，图/物理/特征一致；VAL仅原862/500底图。无新增数据、无WF/TPC/Parrot、无VAL标签训练、无推理后搜索。

A从头seed28，32612参数StateActor both32×2/4heads，范围±8/±8/±2/±2，Adam.001、batch16、clip1、7epoch/7553更新，全部同当前B。旧10种子固定(C,cost)可缓存重建并校验，独立BHEFT回放，11选最优后生成自己的StateScore前缀监督。保留独立新缓存，原缓存不变。A/B分别按同网格底图等权VAL选择SFT起点，B复用已完成best.pt。SFT最好点差异是预声明选模流程的一部分，另留两者末轮以查对照。

## PPO主设置

repeat512，T=.5，actorAdam3e-5，独立SummaryCritic64（9153参数），criticAdam1e-3，gamma=lambda=1，clip.2，entropy.001，clipnorm1，KLstop.02；每收集16配置各1完整采样轨迹，512配置循环4遍，128collections=2048完整主轨迹，4优化epoch×2轨迹minibatch8，最多1024actor/1024critic更新。无BC锚/教师奖励/组内baseline。复用原经典PPO/温度对齐代码和解析76维前缀critic观测，自己的可行集/前缀/旧概率必须匹配。

A/B各做相同RLseed41/22/33，合计6个PPO臂；每个seed两臂同TRAIN索引顺序、采样种子、critic初始权重与minibatch RNG。三RL种子共享每臂seed28的SFT起点，不声称3个独立端到端SFT种子。按历史shuffle源顺序索引，明确冻结512主配置，另用原顺序2048:2304的256配置预热critic，224拟合/32留出，200次更新；仅在各自SFT策略T.5采样，critic预热不得改变actor。

奖励每步−ΔpartialMakespan/HEFT，完整回报=−C/HEFT。保持旧训练按实际动作平均PPO surrogate，不暗改成底图重加权；主评价底图等权，另配置均值用于历史对齐，明确二者不同。RL可能退化，best允许选更新0，保存last和完整负结果。

确定性VAL862在actor实际更新0/100/200…及最终，并额外collections32/64/96/128（复核历史观测点）完整评价；多epoch中途评价不改变rollout/adv/oldlogprob。每次保存逐实例值、base/config两个均值、全部传统参考及新Teacher线。主best按底图均值选择，另外记录配置均值best，不拼接实例最优。横轴用实际actor更新，另记collection/更新机会/实际更新，critic预热单列。

## 工程门槛与有限执行

开始前提交协议/实现。No-search12形状校验：10候选缓存精确重建、BHEFT同HEFT顺序和独立物理重放、自己的轨迹/预算/容量/通信正确、输入图不变；全17261键/动作数/物理哈希与原一一对应。PPO12形状×T1/.5概率跨框架、仪表观测不改排程、reward telescoping、critic梯度不流actor、GAE MC一致、原起点862物理输出重现；短实际更新重复和断点恢复（模型/optimizer/RNG）精确一致。正式训练使用全量设定，不用小试排除设计。

有限本地controller：强SFT→PPO与no-search标签生成并行；A数据完毕自动完整SFT，再自动A→PPO，全部6臂完成后只做汇总、停止，不自动调参或扩数据。最多2工作进程组并行；单个标签任务2CPU worker。每阶段日志/心跳/错误保存；PPO按完整collection边界原子resume，若中途失败重放最多该collection8次更新、丢弃未提交尾部指标，采样完全确定。至多2次恢复。没有聊天自动推理定时器，不声称有/loop或cron工具。

## 记录与判断

记录四格SFT_A、SFT_B、A+RL、B+RL，配对相对起点增益、最终绝对C/HEFT、三RL种子分布和500底图条件区间，明确已知VAL探索。A+RL更优才支持采用无搜索教师；仅RL增量更大不支持；若B+RL更优保留强教师。此次不自动增加grouped4/BC续训等新臂；历史有预算匹配BC控制，但不能用其代替本起点的匹配控制作强RL归因。

动态图：A的SFT loss、A/B SFT VAL轨迹、六臂PPO policy/value loss和VAL轨迹（实际100更新），HEFT/BHEFT/旧Teacher/新Teacher参考；PNG300DPI+PDF以及HTML。来源和预处理hash、采样计划、参数、开销与所有尝试均落盘。
