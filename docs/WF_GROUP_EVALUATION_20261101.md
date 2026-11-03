# WF 逐应用评价及后续实验要求

2026-11-01。用户同意“合成 DAG＋公开科学工作流＋agent 应用案例”的证据路线，并要求今后按 WF 子类查看效果，再探索论文聚焦范围。此次只拆分既有输入与保存结果，没有训练、推理、重新构造调度或改变现有总分。

## 当前 WF 覆盖

现有 WF960 是 WfInstances 来源的 120 次执行、83 个 WL16 拓扑组、26 个应用/流程名称、27 个应用/引擎配对、5 种引擎。任务数 22–286；每次执行 8 套既定物理配置。示例：Pegasus 的 Montage、Epigenomics、Cycles、1000genome；Makeflow 的 BLAST/BWA；Snakemake 的 RNA-seq 等。

“引擎”是运行工作流的软件，例如 Pegasus/Nextflow，不是调度设备池；“应用”是具体计算流程。一个应用可能在多个引擎中实现，不同执行也可能共享同一拓扑。应用分组便于论文解释；图形模式通过深度、层宽、汇聚/分叉、边密度等统计说明，不能把 27 个配对称为 27 种独立图形家族。

完整清单：[面板说明](../data/wf_application_panels_20261101/README.md)。机器清单：[REGISTRY.json](../data/wf_application_panels_20261101/REGISTRY.json)。各组 source_path、base_id、拓扑指纹、任务数、深度、层宽、最大入/出度均已保留。各组互不重叠并完整覆盖原 960 配置。

## 已交付的历史结果

`runs/wf_application_reports_20261101/`：

- `S0_original_group_curves.json`
- `S1_structure_group_curves.json`
- `S2_attributes_group_curves.json`
- 每组实验对应 `*_application_curves.csv`：每次观测×应用/引擎×全部已有 baseline 的比值。
- `VERIFICATION.json`：三个实验各 36 个观测点，分组比值和总分全部与原文件中的 strata/metrics 数值一致，误差 < 1e-12。模型调用和新 rollout 均为 0。

JSON 同时包含 application_engine、application、task_count、application_engine_task_count 四种切片。所有组均保留，没有自动按成绩筛选。少于 3 次执行的组在输入清单标为证据较稀疏；这是样本覆盖标记，不是排除条件。

## 下一轮模型优化必须接入

在新 trainer 每次完成 WF 全量 VAL pass 后，对已经得到的逐例 rows 调用：

```python
from daglab.wf_panels import report

wf_groups = report(rows)
# 保存本次 step 和 wf_groups，并累计每组 C_over 随 step 的变化。
```

报告函数不读取 checkpoint，也不调用模型或模拟器，所以分组不增加推理次数。每个子组同时报告配置数、执行数、拓扑组数、节点范围、可行性、C/HEFT、C/BHEFT、C/BUDA、C/MSLBL、C/Teacher64 等所有实际已有的 baseline；小于 1 表示模型更快。缺失或不可行的 baseline 结果会明确标记，不能悄悄丢弃相应病例。

主统计沿用配置→执行→应用配对内拓扑→应用/引擎等权，并另外保留配置均值和胜率。固定规模档 20–40、41–80、81–120、121–200、201–300。旧入口中的总分/选模逻辑没有改；分组报告应在新实验启动前接入，不能声称旧训练器已自动增加新文件。

## 论文聚焦范围

可以探索模型在哪些具体应用或规模范围表现更好，并据此收敛论文所研究的场景。范围固定后，对这些应用的完整既定配置、所有 baseline 和多个训练种子使用相同成员集合；继续保留其余组的完整记录。研究日志记录范围选择时间和依据，论文结论限制在实际展示的范围。已经看过并用于选范围的数据属于已知评价，不称为完全未见的确认测试。

例如最终聚焦 Montage 和 Cycles，可以明确写“在 Montage、Cycles 两类公开科学工作流上进行评价”，并列出具体执行数、拓扑数、配置数和规模。该表述描述选定范围，不外推为所有 WF 应用普遍获益。

本次未决定最终应用名单，也未删除差例；后续以逐组、跨种子且与强 baseline 的实际结果讨论范围。
