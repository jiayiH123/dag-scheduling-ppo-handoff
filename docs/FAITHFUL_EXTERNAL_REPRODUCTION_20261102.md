# CAI / GiPH：论文忠实适配复现启动记录

用户指定：忠于原论文核心算法并适配当前问题，TRAIN17261、VAL862，允许调参后汇报最终VAL结果。旧外部方法记录不覆盖、不冒充本轮结果。

实验根目录：`runs/faithful_external_20261102/`。
协议：`protocols/faithful_external_20261102.md`，初始提交8e1412b；
初始工程实现及并行安排提交77579ea；预算输入尺度修正提交35295ef。保留版本控制器启动PID7477。
所有PID仅为启动时记录，实时状态以controller_status.json及各arm进度为准。

## 方法与来源

|项目|DAG-CAI-DR|GiPH|
|---|---|---|
|作者版本|1a2aacc3fc44d8d8f1ef779488a39cab80abf991|427caced16988f0f7c22b7dcceea59fa472dbc34|
|本轮网络|实际作者Graphormer encoder/layer/attention模块；论文4层4头64维与64-64-1头|实际作者FNN、GiPHEmbedding、MPLayer、SoftmaxActor，DGL1.1.3|
|决策|作者GumbelTopoSortAdj/CAI概率，学习任务优先级，预算可行EFT选池|学习任务迁移，训练和评价均2n步、Categorical采样|
|训练|Adam、batch16、每图4条采样顺序、7epoch|Adam、gamma=.97、逐实例原在线REINFORCE更新、17261完整实例|
|奖励|作者代码的逐任务HEFT完成时刻差，原批标准化/截断/正则/熵项|原始相邻工期改善、折扣return和前缀平均reward baseline；不额外clip或按初始C归一化|
|参数数|126353|640（作者CLI默认embedding10、head hidden16，输入因物理约束从5扩到8）|

作者原Graphormer调用分支与附带模块输入/返回不匹配，且该分支的维度与论文附录不一致。
本轮以论文附录C.2为网络配置准据，直接载入实际核心模块，接入连续数值输入/图字典。
仅替代q_noise=0、layerdrop=0所需的Fairseq包装器为等价Torch运算；不是旧自写Transformer。
作者源码原封不动保存在sources/中，修复/输入适配在daglab/faithful_external.py中。
AST提取执行作者的实际类/函数定义，以绕过未调用的旧环境导入；未改其定义内容。

必要场景适配：共同累计资源SGS、硬预算与mask/reserve、实际通信与整数费用；添加可见费用/容量/预算特征。
GiPH继承原5节点/5边特征与[-1,1]缩放、原数据加权初始化及历史关键性更新。
常数列置0（含边）、零通信关键性初始化取均匀有限极限；ID0特判推广到真正根节点。
这些是已披露修复，不能称完全未经修改作者程序或原论文场景发表分数复现。

## 工程证据

`engineering/CHECKS.json`：全部12种20/40/80/120任务×3/4/6池实图上，
CAI ready概率/梯度、实际DGL网络完整2n轨迹、梯度有限、HEFT身份与最终物理可行性通过。
作者Graphormer attention与独立QK-softmax-V前向及梯度对照通过。
作者GiPH原始5节点/5边特征在120任务6池样例逐项对照通过（只加声明的根/有限值修复）。
GiPH线上更新的权重与随机数恢复逐位相同；另执行作者原finish_episode，对应更新权重最大差异为0，见GIPH_UPDATE_PARITY.json。CAI的权重、Adam状态及随机数恢复逐位相同，见CAI_RESUME.json。
工程状态没有成为正式初始化或模型检查点。

依赖全部装在run/deps，不改现有环境：Torch2.11.0现有CPU环境、NumPy1.26.4、DGL1.1.3，及共享模拟器依赖。
原作者Graphormer和GiPH来源zip SHA在sources/MANIFEST.json；输入SHA在INPUT_AUDIT.json；
每个arm metadata/resume/COMPLETE均绑定输入审计、工程证据、模型/物理模块/协议源码SHA。

## 本轮由AI选择、已预先固定的调参安排

CAI seed11学习率5e-4、1e-4、1e-3，各完整7epoch，再选定学习率做seed22/33各7epoch。
GiPH seed11学习率.01、.003、.001，各完整17261实例，再选定学习率做seed22/33。
三档GiPH可并行，随后两个确认种子并行；CAI顺序运行；最多4个实际trainer各1CPU线程。
没有按小图或短16步轨迹筛掉设计，也没有根据表现删除任何VAL成员。
每种方法一共5个有限训练arm，自动结束，不扩展SFT或额外数据。

CAI每500更新及每epoch末全VAL；GiPH每4096实例及末尾全VAL；包含训练前评价。
主指标是500底图等权的配对C/HEFT，配置均值另列。
GiPH按随机起点主指标选共同checkpoint；HEFT起点另报同一策略。
最终GiPH每个选中策略再做5组固定采样种子，取平均、不取逐例best-of-5。
3训练种子分别报告，SFT参照仍只有seed28；已知VAL选模不是独立确认TEST。

原生DGL全2n工程探针每实例约.36–17.73秒，12种规模平均4.11秒。
这只能提供初步量级，不能当全训练耗时；恢复完整轨迹和在线更新后，GiPH会显著慢于旧16步/批16版本。
记录实际累计训练秒数、episodes、迁移数和推理调用，按持续吞吐更新估计。

## 查进度与最终结果

- `controller_status.json`：总队列状态；`*_QUEUE_PROCESS.json`：方法队列PID。
- 各arm的`progress.json`、`training.jsonl`、`worker.log`、`curve.json`。
- `validation.png/pdf`和`training_loss.png/pdf`：控制器约每2分钟刷新。
- `report.html`和`REPORT.md`：进度与结果；同步文档为`docs/FAITHFUL_EXTERNAL_RESULTS_20261102.md`。
- `cai_FINAL.json`、`giph_FINAL.json`：各方法完成后的3种子结果；最终`FINAL_RESULTS.json`只在全部完成后生成。
- 逐例调度映射/顺序/费用/可行性均保存在`val_*.json.gz`。
- `*_PAIRED_ANALYSIS.json`：分任务数/家族、对固定SFT的条件500底图bootstrap；不含训练/选模不确定性。

这是有限本地程序，不是可在聊天中主动推送消息的定时agent。
完成时会自动生成最终结果文件和报告；在这些文件出现之前不得将旧适配结果当作本轮最终数值。

## 输入尺度修正与归档计算

额外全TRAIN审计发现5条新增预算特征(B-min)/(fast-min)>10，最大128373，4条fast=min。
这是输入尺度问题，不是因VAL成绩删样本。改为(B-min)/(max-min)，全17261TRAIN与862VAL均在[0,1]，所有输入成员保持。
初始控制器6451及关联任务已停；全部初版结果/源码快照保存在archive_unbounded_budget_ratio。
其中CAI已正式执行2000更新、31997次配置呈现、722.53秒训练，单独计入调参计算，绝不重新标成工程试跑。
GiPH初版尚在初始VAL阶段，正式更新0。全部工程门槛重验后，统一新版本由7477重启；两版曲线不拼接。
