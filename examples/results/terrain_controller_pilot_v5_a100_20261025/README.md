# Pilot v5：预检历史与最终结果

2026-10-28 更新：[e88523d 修复后完整结果](continuation_e88523d/README.md)已完成全部预检及 1500 次新增更新，八项验收通过七项，一步误差未通过。正式训练耗时 63 分 8 秒；报告包含独立账本、GPU-hours、分组结果和九张图。下面保留 2026-10-26 的原始失败记录。

2026-10-26 在 A100 服务器核对批准提交 `2c41613`。84 项测试通过，补充的归一化 critic 反向检查因尺度张量被原地更新而失败。单卡 smoke、双卡故障退出预检和正式 pilot 均未启动，新增训练更新为 0。

- [执行记录、复现命令与解释](../../../docs/records/terrain_controller_pilot_v5_a100_20261025.md)
- [report.json](report.json)：阶段状态、零消耗账本、测试计数及输入摘要。
- [tests.log](tests.log)：84 项测试完整输出。
- [normalized_backward_check.log](normalized_backward_check.log)：CPU 复现完整输出，退出码 1。
- [input_proof.json](input_proof.json)：16 个历史检查点及数据身份。
- [final_integrity.json](final_integrity.json)：历史检查点零差异，未创建 v5 训练目录。

公开日志中的私人环境目录用 `${PFNN_PROJECT}` 表示。原日志和教师副本留在 A100 的 `runs/tmt_v5_coord/`；没有发布权重或动作数据。
