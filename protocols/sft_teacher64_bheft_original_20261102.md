# Teacher64+BHEFT-v2：原始17K单面板对照训练

用户于2026-11-02明确授权现在启动。仅替换原始17261配置的TRAIN监督，不使用4000扩充配置。

- 输入：已完成且哈希绑定的新Teacher64+BHEFT-v2缓存；原始10000底图、17261配置、1122860动作/epoch。物理图、特征、配置键与旧版逐一一致。新教师含旧10候选+BHEFT、64次新提案、旧最终解不退化保留，历史标注计算复用另记。
- 从零初始化seed28；StateActor both/edge_attention32×2/4heads/32612参数；四输出范围±8、±8、±2、±2；固定HEFT顺序。
- Adam lr=.001 betas=.9/.999 eps=1e-8 weight_decay=0 clip=1；固定学习率，无EMA/RL；图内动作CE平均，再按实际batch图数平均。
- batch16，每epoch1079更新，尾批13；7epoch共7553更新、120827配置呈现、7860020监督动作。采样计划和初始参数须与旧S0逐字节哈希一致。
- 仅原始synthetic VAL862/500底图；step0、100、200…7500、7553，77次完整评价。无WF/TPC/Parrot、无TRAIN rollout probe。保存逐实例物理验证、所有旧baseline和旧Teacher64线；未为VAL重做新Teacher，不能把旧参考标成v2。
- 主选模：底图等权C/HEFT；同时报告旧配置均值C/HEFT供历史对齐。保留所有11参考的逐例结果。best.pt据主指标选择，不能拼接不同checkpoint。
- 两张动态曲线：training_loss和val_synthetic（PNG300DPI/PDF），后者叠加旧17K相同评价成员的曲线及baseline。每50步断点和loss图，100步全量VAL，额外初始/最终。
- 启动门槛：完整新缓存SHA、准确TRAIN键/物理哈希/图特征、原始VAL键/图/参考一致、TRAIN/VAL键与底图不交叉；新教师轨迹可复现；独立CE/梯度/概率校验；精确断点恢复。工程更新隔离，正式训练从零。
- 有限本地controller，独占锁、日志、心跳、最多2次精确恢复。完成7epoch即停，不自动扩实验。单seed探索；VAL已被反复用于调参，不称未见确认测试。
