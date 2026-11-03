# WF 按应用拆分的固定评价面板：2026-11-01

沿用现有 WF960，只拆分统计，不改变物理参数、实例或历史总分。
120 次执行、83 个 WL16 拓扑组、26 个应用名称、27 个应用/引擎配对，来自 5 种引擎。
应用/引擎配对是可解释的来源分组，不等于独立图形家族；具体形状统计保存在 REGISTRY.json。
下表的“小样本”表示少于 3 次执行，并非按模型成绩决定。所有组都保留在记录中。

|应用/引擎|执行数|拓扑组|配置数|任务数|
|---|---:|---:|---:|---|
|makeflow/blast|10|2|80|43–103|
|makeflow/bwa|5|1|40|104–104|
|nextflow/airrflow|1|1|8|212–212|
|nextflow/atacseq|1|1|8|265–265|
|nextflow/chipseq|1|1|8|210–210|
|nextflow/cutandrun|1|1|8|120–120|
|nextflow/fetchngs|1|1|8|43–43|
|nextflow/hic|1|1|8|38–38|
|nextflow/mag|1|1|8|157–157|
|nextflow/methylseq|1|1|8|36–36|
|nextflow/rnaseq|2|2|16|101–197|
|nextflow/sarek|1|1|8|26–26|
|nextflow/sarek_somatic|5|1|40|164–164|
|nextflow/smrnaseq|1|1|8|197–197|
|nextflow/taxprofiler|1|1|8|127–127|
|nextflow/viralrecon|1|1|8|203–203|
|pegasus/1000genome|8|8|64|52–260|
|pegasus/cycles|6|6|48|67–266|
|pegasus/epigenomics|8|8|64|41–263|
|pegasus/montage|4|3|32|58–178|
|pegasus/seismology|2|2|16|101–201|
|pegasus/soykb|5|5|40|96–286|
|pegasus/srasearch|25|5|200|22–104|
|snakemake/RASflow|4|4|32|31–201|
|snakemake/rna-seq-star-deseq2|19|19|152|45–251|
|snakemake/varlociraptor|4|4|32|91–271|
|streamflow/1000genome|1|1|8|26–26|

后续每次 VAL 的逐例结果可调用 `daglab.wf_panels.report(rows)`，同时得到按应用/引擎、应用名称、规模、应用×规模分组的全部 baseline 比值。
调用不增加模型 forward 或模拟器 rollout。主比值小于 1 表示模型快于相应 baseline；改进百分比为 100×(1−比值)。
按配置→执行→配对内拓扑→应用/引擎等权；另保留配置均值。
论文可选择聚焦明确应用范围，但需记录范围如何确定，完整结果留档，不能把看过并用于选择的范围称为未见测试。
对一组应用内全部既定配置进行比较，避免逐实例丢弃失败案例；不同 baseline 使用同一成员集合。
