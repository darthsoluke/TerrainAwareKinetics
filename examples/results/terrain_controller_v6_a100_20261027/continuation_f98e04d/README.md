# v6 修复后重跑：A 的长窗口评估中断

2026-10-29，源码 `e1793dc8429dfe343db180937f4923ada955c08e`，包含 `f98e04d` 测试修复。**90 项 CPU 测试全部通过，0 跳过**；FK 最大误差为 0m。

A 完成 v1 教师检查点的 16 帧 raw/display/feedback 三种模式后，在 64 帧基线评估中失败：评估器将 16 帧历史缓冲区作为 64 帧生成序列，逐帧指标出现维度错误。B 未启动，A 使用 0.004115851 GPU-hours，训练更新为 0。

- [执行记录、命令和原因](../../../../docs/records/terrain_controller_v6_a100_20261027.md#5-2026-10-29-修复后重跑a-的-64-帧评估中断)。
- [report.json](report.json)：停止位置、测试结果、消耗和产物状态。
- [partial_eval_report.json](partial_eval_report.json)：异常时保留下来的三组完整指标。
- [table.tsv](table.tsv) / [by_group.tsv](by_group.tsv)：仅 v1t、16 帧的 3 行 / 21 行部分结果。
- [batch_digests.json](batch_digests.json)：新脚本与历史训练器两种摘要计算口径。
- [checkpoint_hashes.tsv](checkpoint_hashes.tsv) / [input_proof.json](input_proof.json)：五个评估输入的 update 与哈希，以及全部旧检查点身份。
- [tests.log](tests.log) / [tests.xml](tests.xml)：90 项测试日志与 JUnit。
- [eval_a.log](eval_a.log) / [failure_traceback.txt](failure_traceback.txt)：完整评估错误。
- [gpu_hours.tsv](gpu_hours.tsv)：A、B 与预检分别计账。

v2+lock 判定、单角色加锁耗时、v2 图片、B 训练及联合表均未执行。没有重新跑模型来填补缺失项。15 个空掩码统计值在 JSON 中为 `null`，TSV 中为空；报告列出字段位置。

公开日志用 `${TAK_ROOT}` / `${PFNN_PROJECT}` 替换私人目录，JUnit 主机名写为 `A100`。没有发布权重或动作数据。
