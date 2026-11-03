# 新Teacher64+BHEFT-v2：原始VAL862参考线

用户2026-11-02明确要求重新计算原始862VAL的新Teacher并加入现有图。

固定同862配置/500底图、不修改训练数据、VAL输入/历史模型逐例结果/选模指标。按训练新Teacher同定义：10种旧规则的固定HEFT映射候选+BHEFT，初始11者取(C,cost)最小，seed+3917做64次一/二任务提案，再与重建旧Teacher最终解比较保留更优者。旧Teacher从10候选+原64次搜索确定性重建，并要求(C,cost)与缓存旧参考逐例完全一致；BHEFT要求同HEFT顺序、独立物理回放与缓存一致。新解物理验证且不劣于旧Teacher/BHEFT。不是最优解保证。

先以原TRAIN各n×k共12例核对本实现的新Teacher最终mapping/C/cost与已完成训练标签完全一致，然后双worker计算全部862，仅写独立VAL参考产物、不导入TRAIN。逐例checkpoint、可恢复、不改历史参考。汇总同时给出500底图等权C/HEFT和862配置均值及相对旧Teacher改善覆盖。

原训练源码与哈希绑定绘图文件保持不变。用独立渲染器读取新增参考与全部已记录曲线，绘制绿色Teacher64+BHEFT-v2虚线、保留黑色旧Teacher线。渲染输出同目录val_synthetic.pdf/png及带with_new_teacher后缀的稳定副本。有限辅助刷新进程监测本次训练出图，在原训练完成后最终渲染并退出；不修改任何模型/优化器/训练断点绑定。所有图像原子替换，避免与原出图进程临时文件冲突。

输出runs/sft_teacher64_bheft_original_20261102/new_teacher_val862/，保存binding、ENGINEERING_CHECKS、逐例rows、progress、SUMMARY、MANIFEST。主图参考线口径与其他线一致：先配置C/HEFT，底图内平均，再500底图等权平均。
