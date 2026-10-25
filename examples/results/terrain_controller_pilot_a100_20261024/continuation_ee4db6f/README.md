# ee4db6f：A100预检续跑

**84项测试与resume2通过；双卡检查未通过，pilot未启动。** 已通过的数据迁移和单卡冒烟直接复用。

resume日志为`resumed_at_update=10, phase_index=1, phase_update=5`，结束update=12、exposures=528。双卡两个rank均完成5次更新、退出0；rank0保存了`phase0_teacher.pt`与`latest.pt`两个文件，rank1没有保存文件。两rank的控制器初始化有72/134个参数张量不同，critic为6/6，结束时仍不一致。

- [追加执行记录](../../../../docs/records/terrain_controller_pilot_a100_20261024.md#9-ee4db6f-续跑)
- [结构化报告、rank证据与来源](report.json)
- [逐次评估表](evaluation_table.tsv)
- [预检训练曲线](preflight_curves.png)

本次新增7轮控制器全局迭代、2次critic更新、672窗口曝光、128训练闭环生成帧；正式pilot均为0。显存检查和六窗口接触图未执行。checkpoint与原始日志留在A100，旧失败记录及旧结果文件保留。
