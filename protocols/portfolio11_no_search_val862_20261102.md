# 原始VAL862无搜索Teacher参考

2026-11-02，用户要求在teacher_search_ablation_ppo_20261102/sft_validation.pdf加入无搜索Teacher。仅补评价参考，不改变训练/选模/历史结果。

在完全相同862VAL/500底图上运行原10启发式候选加BHEFT，均固定HEFT顺序回放，按(C,cost)选最优，恰好0次局部修改，不保留旧Teacher最终解。12种n×k的TRAIN工程实例要求与已完成Portfolio11-no-search缓存逐例mapping/C/cost精确一致。全部VAL独立物理回放可行，BHEFT结果与旧参考相等，并要求强Teacher最终(C,cost)逐例不差于本无搜索结果。只读原VAL、TRAIN及已完成强Teacher参考；保存逐例候选值、选中映射、输入/code哈希和汇总。

图中新增紫红点划线Portfolio11 (no search)，保留全部现有线；主值是底图等权C/HEFT，配置均值单列。原训练绑定的绘图源码保持不变，独立有限刷新进程共用plot.lock，更新sft_validation.pdf/png及稳定with_no_search_teacher副本；本轮父controller完成后最终渲染退出，无额外训练或推理策略改变。
