# GiPH：原文200 episode预算

替换原先每个学习率17261 episode的过重方案。旧GiPH任务已停止，日志保留；CAI独立队列继续。
TRAIN17261为抽样池；每个种子均匀有放回抽200题，seed11/22/33独立训练；Adam0.01、gamma0.97、每题最多2n步。
沿用作者DGL核心和精确物理适配；原论文4节点/4边特征，外加费用/容量/预算3节点特征。没有额外学习率搜索。
训练完成才对终点checkpoint全量评价VAL862，随机起点和HEFT起点分开；不按VAL选checkpoint、不取种子最优。

|种子|进度|已训练episode|
|---|---|---:|
|11|evaluation|200 / 200|
|22|evaluation|200 / 200|
|33|evaluation|200 / 200|

|种子|随机起点 C/HEFT|HEFT起点 C/HEFT|训练秒数|
|---|---:|---:|---:|

此处所有指标来自已知VAL862/500底图；主指标是底图宏平均，配置均值另存JSON。
详细逐例映射/顺序/费用/动作数：seed*/val_terminal.json.gz；训练曲线：training_loss.png/pdf。
