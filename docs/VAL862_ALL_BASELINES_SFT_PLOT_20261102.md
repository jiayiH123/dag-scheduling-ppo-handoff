# VAL862：SFT 与全部 baseline 曲线

绘图使用已保存的全部逐例结果，没有启动训练或重新推理，也没有改变原始图。

主图：`runs/sft_teacher64_bheft_original_20261102/all_baselines_val862/val_all_baselines.png/pdf`。
左图显示完整范围，右图放大相同数据；右侧线型表列出全部数值。

SFT 为新 Teacher64+BHEFT-v2 标签训练的原始17261配置、seed28、7epoch版本。使用0、100、200、…、7500、7553更新的77次全量评价，逐次核验同一862 key、500 base_id、HEFT引用及可行性标记；没有平滑或插值。

主指标为先对每配置计算C/HEFT，再在底图内平均，最后对500张底图等权平均。补充图 `val_all_baselines_config_mean.png/pdf` 使用862配置等权，和历史配置均值口径对应，勿交叉比较数值。

十种单独规则使用各自原生顺序；Portfolio11是11个候选映射按固定HEFT重放后逐例取优、没有搜索。当前Teacher为Portfolio11起点+64次新提案，并保留旧Teacher最终解以防退化；旧Teacher另列。

按用户所列方法加入全部13条参照，另外保留旧Teacher，以及已有Greedy、RandomMean5、BUDA和CAI/GiPH两种版本，共21条水平参照。CAI历史选模不变；GiPH同时保留HEFT及随机起点。外部方法是本地适配，训练量并不完全相等，详见EXTERNAL_BASELINES_CURRENT_VAL862_20261102.md。

|方法|底图等权 C/HEFT|配置均值 C/HEFT|
|---|---:|---:|
|SFT末轮/最佳 step7553|0.86618994|0.84694829|
|Teacher64+BHEFT-v2|0.83135917|0.80908471|
|Teacher64 (legacy)|0.84261157|0.82209772|
|Portfolio11 (no search)|0.84362941|0.82322835|
|GiPH / HEFT start / 2n|0.92060078|0.91132178|
|MSLBL-adapted|0.96188216|0.95577977|
|BHEFT|0.96790104|0.94449194|
|JoinLookahead|0.99308143|0.99525446|
|HEFT|1.00000000|1.00000000|
|PEFT-price0.1|1.03585728|1.04016707|
|PEFT|1.03781877|1.04307249|
|PEFT-price0.3|1.04110353|1.04079938|
|PEFT-price1|1.08635992|1.05912136|
|GiPH / random start / 2n|1.19411657|1.13294437|
|Greedy|1.22414046|1.23781527|
|DAG-CAI-DR / BiGAT|1.25328121|1.27003596|
|DAG-CAI-DR / Graphormer|1.25568923|1.27093919|
|PEFT-price3|1.33119875|1.22225039|
|PEFT-price10|1.82440231|1.60828863|
|BUDA|2.80300196|2.38449332|
|RandomMean5|2.95529026|2.56667297|
|CostFirst|3.24283467|2.75993191|

绘图脚本保存于`figures/gen_fig_val862_all_baselines_20261102.py`；输出目录保存baseline_values.csv、sft_curve.csv和带输入SHA256的MANIFEST.json。
