# v6 rerun 2: A 完成，B 等待资源

2026-10-29，实验源码固定为 `60fd119534e739c4f70bfc1fd82bcc21e85db810`。本目录先发布 A 的完整结果；B 的双卡预检和训练尚未开始，当前交接保持 `running`。

90 项 CPU 测试通过，0 失败、0 跳过，包含 20 帧回归。A 完成五个检查点在 16/64 帧下的 raw/display/feedback 共 30 行、210 行分组统计，FK 误差和旧 v1t16 确定性最大差异均为 0。20 个旧检查点的前后哈希一致。

v2 的 display/feedback 在两个长度下均通过漂移、进度、穿透增量和 IK 残差四项，但源接触脚速为 0.146779–0.174438m/s，高于候选门槛 0.03m/s，四组均未通过候选判读。六个接触图和六个足底曲线已逐图审阅。

| 文件 | 内容 |
| --- | --- |
| [report.json](report.json) | A 的执行回执、输入身份、全部指标、候选判定和累计账本 |
| [eval_report.json](eval_report.json) | 原评估报告的公开副本 |
| [table.tsv](table.tsv) | 五个检查点、两种长度、三种模式的全部总体指标 |
| [by_group.tsv](by_group.tsv) | 七组的漂移、进度、两种脚速与着地比例；空掩码字段留空 |
| [v2_candidate_gates.tsv](v2_candidate_gates.tsv) | 计划 §3 的四组逐项判读 |
| [visual_review.md](visual_review.md) | 十二张 PNG 的逐图观察及六窗口末帧漂移 |
| [review_receipt.json](review_receipt.json) | 固定窗口选择、CPU 渲染、源动作及历史 v2 一致性证据 |
| [batch_digests.json](batch_digests.json) | 16/64 帧的脚本摘要和训练器格式摘要 |
| [determinism.json](determinism.json) | 与旧 v1t16 三模式 61 个数值逐项比较，最大差异 0 |
| [checkpoint_hashes.tsv](checkpoint_hashes.tsv) | 五个 A 检查点的版本与 SHA-256；全部 20 个哈希见 report.json |
| [gpu_hours.tsv](gpu_hours.tsv) | 两次早期失败与成功 A 独立计账，累计 0.031167811 GPU-hours |
| [tests.log](tests.log)、[tests.xml](tests.xml) | 已完成测试的日志和逐项结果 |
| [eval_a.log](eval_a.log) | A 的成功执行日志 |
| [sparse_reference_setup_failure.log](sparse_reference_setup_failure.log) | 稀疏检出缺参考 TSV 的启动前错误，零模型帧和训练更新 |

单角色脚锁耗时 P50/P99 为 10.393651/29.977724ms。A 本次进程壁钟为 95.214735s，分配一张 A100，消耗 0.026448538 GPU-hours；累计计入前两次失败后为 112.204120s，仍计入原 30 分钟预算。B 及其两个预检在此快照的消耗均为 0，后续仍按原合计 4.3 GPU-hours 上限执行。

JSON 的空掩码统计保留 `null`，TSV 留空。图片为固定源窗口和模型输出的诊断渲染。公开文件由私有 Drive 取回并校验，原始归档 SHA-256 为 `642babe91bc099d8c7dbb964eef72267a4a3b50a457c7afabb5b5ea0ea089d0d`。目录只含聚合结果、审查图和已隐去私人路径的日志。

[详细执行记录](../../../../docs/records/terrain_controller_v6_a100_20261027.md)；[当前任务](../../../../handoff/tasks/2026-10-29_v6_rerun2.md)。
