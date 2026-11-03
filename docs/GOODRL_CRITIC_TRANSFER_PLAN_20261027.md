# 借鉴GOODRL改善一次forward调度策略的RL训练

2026-10-27。论文与作者源码再审阅；以下为可执行设计草案，尚未训练，没有复制原版GOODRL实验或更改当前模型。

## 来源与核对范围

GOODRL，Yifan Yang等，ICLR2025。用户提供29页PDF，SHA256：
`e365900dfc777d6cefd5ef375d324dc3cd3e3b1aa82fe5d2bf7e983697dd6dbe`。
重新阅读主文4.1/4.2.2/4.3、附录H.1/H.2/J/L；目视核对PDF第6、23、24页，表9、10、13及图5。

- 作者仓库：https://github.com/YifanYang1995/GOODRL
- 离线PPO：https://github.com/YifanYang1995/GOODRL/blob/master/step2.py
- 在线PPO：https://github.com/YifanYang1995/GOODRL/blob/master/step3.py
- 网络和更新：https://github.com/YifanYang1995/GOODRL/blob/master/policy/actor3.py
- 参数：https://github.com/YifanYang1995/GOODRL/blob/master/config/Params.py

本次读取master源码，未成功取得commit固定版本，未执行作者代码；不能据此声称数值复现。官方OpenReview链接由作者仓库指向`4PlbIfmX9o`，本轮网页遇验证页，论文依据来自本地PDF。

## 先分清可迁移训练与不可直接套用的部署

GOODRL原版是在动态到达的多工作流环境中，每个决策点为当前focused task选择VM；随环境变化，重新构图并调用actor，目标为平均flowtime。源码step2的validation循环内先model(...)再env.step(...)，不是整张DAG只一次forward。

本项目保留：固定HEFT构造顺序、已知单DAG、累计容量池、固定通信延迟和硬总预算，以全局makespan为目标。部署为一次actor输出全部task-pool系数，然后一次无神经调用的物理构造。GOODRL的动态actor不能直接替换进来后仍声称满足该条件。

真正优先迁移的是**训练用critic与更新组织**。critic在训练中可以读取逐步状态、反复前向；部署移除critic即可，不增加actor部署调用次数。

## 论文设计与我们的差异

| GOODRL的具体设计 | 本项目现状 | 迁移判断 |
|---|---|---|
| 离线HEFT模仿→critic预热200迭代→PPO1000迭代，表9 | 强启发式组合教师BC→随机critic立刻共同学习 | 先冻结actor预热critic；200的单位不能照搬 |
| critic为独立GAT＋Self-Attention＋pooling＋MLP，图5 | 76维摘要MLP，9153参数 | 优先比较更完整的图状态value模型 |
| 在线actor5、critic20个epochs，表10 | 每批actor/critic均4遍，约1:1；优势在收集后固定 | 比较critic优先/更高频训练及用更新后的V重算MC优势 |
| 异常梯度拒绝更新，主文4.3.2 | 全局梯度norm裁剪1＋更新前局部KL早停 | 两者不同；拒绝更新作为后续独立消融 |

论文的离线预热200迭代每次采集4个独立episode，并非固定缓存上200个minibatch；这些episode包含多工作流，不能与我们的2048个单DAG配置直接按数字比较。

更高频critic与梯度控制是论文**在线**训练组件，不应说成作者离线PPO每处均如此。表13的`w/o freq`还改为actor/critic聚合loss更新，不能当成只改变更新次数的单变量证据。它支持值得试验，不证明在我们环境必然有效。

## 怎样设计适合我们资源池的图critic

建议单独构建训练状态图`G_critic(G,prefix)`：

- 任务节点：原始任务属性、DAG依赖、是否已排程、已选pool、已知start/end、剩余任务/关键路径信息。
- pool节点：算力、累计容量、单价；已排程任务到pool的分配边携带需求和预留时间区间。
- 通信信息：依赖边数据量、池间链路参数；当前任务到各可行pool的候选边携带EFT、data-ready、等待和费用。
- 全局预算信息：总预算、已花费用、未来最低保留、剩余升级余量。
- 独立GNN编码任务/pool关系，以Attention/池化和value head输出一个状态价值；value网络不共享actor梯度。

有两点必须适配，而不能照抄原文：

1. 原文VM串行FIFO，用上一任务到当前任务的机器顺序边。我们的pool允许累计容量并行，单条“最后任务”边不够，需要保留多个资源预留区间及其分配关系。
2. 原文删除物理上已完成任务。我们的HEFT前缀表示已排程，可能只是未来预留，并不等于已经执行完成；不能简单删除所有前缀节点，否则会丢失父任务通信和容量上下文。

最小版本可以先用独立DAG图编码＋动态前缀属性，保留已有候选物理特征；复杂日历事件节点作为后续扩展。与相近参数量的加宽76维MLP比较，避免把参数量增加全部归因于图结构。

## critic优先训练的具体含义

当前实现每批先用旧critic计算GAE并detach，后续4次优化使用固定优势；这是常见PPO配置，不是已经发现的bug。仅增加当批critic优化次数，不会自动更新已经缓存的优势。

GOODRL在线step3先`train_critic`，再`update_advantages(new_values)`，然后更新actor。因此迁移时需明确：保持完整MC returns固定，用新的`V(s)`计算`A=MC_return−V(s)`，detach后用于actor；旧动作logprob仍保留采样时版本，不随actor更新改写。

优先以gamma1/lambda1验证该组织方式，避免同时改变makespan目标、GAE方式和图网络。GOODRL的gamma.99用于它的动态任务环境，不能无条件复制到我们有限单DAG的完整工期目标。

## 最小实验顺序草案，尚未执行

先统一以历史PPO共同BC0.875386为起点做机制对照，胜者再迁移到新BC0.871449；避免起点变化冒充模块收益。保留BC续训、Portfolio、Teacher64质量对照，TEST继续冻结。

| 阶段 | 改动 | 要回答的问题 |
|---|---|---|
| A | 原MLP critic、无预热的经典PPO复核 | 原负结果是否复现？ |
| B | 同actor/MLP critic，新增critic预热 | 冷启动是否是关键问题？ |
| C | 在B基础上单独换图critic，并增加参数匹配MLP控制 | 结构信息是否比单纯扩大value网络有效？ |
| D | 在胜出critic上比较1:1与4:1 critic/actor训练，并显式重算优势 | 更准确且及时的value是否改善部署工期？ |
| E | 之后单独加入异常梯度拒绝 | 是否减少灾难性更新，又是否过度忽略难样本？ |

建议B的小规模起点：共同冻结BC采样256个TRAIN配置的完整轨迹，使用真实MC return预热critic200个minibatch更新；随后各臂用相同2048个TRAIN配置做128次收集。这个200是我们的新试验设置，不等价论文200个重新收集数据的迭代。所有臂获得相同预热样本，报告额外value训练的计算量；后续增加计算匹配控制。预热数据只能来自TRAIN和学生自身轨迹，不能用VAL教师轨迹。

所有新臂统一补齐：更新后整批KL、相对固定BC的KL、固定TRAIN进度分组残差EV、完整确定性VAL工期、采样均值与部署argmax区别，以及actor部署前向计数。评估网格保持一致，不把多样本最优当部署成绩。

梯度拒绝阈值不能照抄0.075：论文奖励除1000，我们除实例HEFT；参数数量、loss reduction和batch也不同。当前作者源码使用逐参数梯度范数的均值，与我们全局norm统计量不同。若尝试，需记录接受率并按任务规模/CCR/预算切片，避免只剩容易实例获得更新。

## 可发表性与结果边界

以上组件均有GOODRL或其他actor-critic方法先例，不能直接当作原创贡献。可能的适配价值是：在硬预算、通信及累计容量条件下，让训练用value网络更好地指导一次forward actor；它是否产生稳定RL增量、超过Teacher64或相关baseline，必须由本任务实验回答。

GOODRL的教师是HEFT，我们是组合启发式＋局部改进，教师强度不同。它能在自身场景改进HEFT，不保证我们一定超过Teacher64。当前actor部署形式、硬约束和数据冻结均保持原状；本文是迁移方案，没有新训练结果。
