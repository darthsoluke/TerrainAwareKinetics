# 历史路线与实验账本

第一次阅读请看 [项目全景的路线表](../START_HERE.md#4-项目经历过哪些路线)。下面是需要追溯实验时的入口。

| 路线 | 记录 |
| --- | --- |
| BONES转换、压缩与接触小试 | [BONES执行](../records/bones_seed_training_execution_20261004.md) |
| LAFAN原生MotionBricks | [LAFAN执行](../records/lafan_mb_training_20261006.md) |
| PARC扩增及长路线混合 | [长路线混合](../records/long_route_mixed_training_20261010.md) |
| 完整混合模型及质量评估 | [最终候选](../records/pfnn_final_candidate_review_20261016.md) |
| 近远地形与运行响应 | [响应/调参](../records/pfnn_response_and_tuning_20261017.md) |
| 场景控制接口原型 | [v2](../records/scene_control_v2_execution_20261018.md)、[v3](../records/scene_control_v3_execution_20261018.md) |
| 人体参考与G1执行 | [跟踪器比较](../records/mb_tracker_comparison_20261019.md) |
| 高频重规划与历史反馈 | [prefix实验](../records/reactive_prefix_training_20261021.md) |
| PFNN逐帧P1 | [P1复核](../records/pfnn_frame_actor_p1_review_20261022.md) |
| A/B/C架构与评测修正 | [最新结果](../records/pfnn_physics_findings_20261022.md) |

## 几本独立账本

| 运行 | 已记录的规模 | 保存/质量状态 |
| --- | --- | --- |
| 正式混合MB |74,125实际更新，38,400,000样本曝光 | 三阶段完成，最终质量失败 |
| 逐帧P1 |1,774次已记录更新，453,262窗口曝光 | checkpoint保存1,652次更新/422,030窗口；时间上限已停止 |
| A/B/C v3 |每臂12次更新/24,576窗口；合计36次/73,728窗口 | 三臂完成，支撑稳定性未通过 |
| P2 |部分目标代码与测试完成 | 未启动完整训练 |

早期其他小试、探针与技术验证分别保留各自记录；上表不充当整个项目的累计总账。

历史报告中的进程号、端口和“正在运行”属于原记录时刻。当前结果从首页与最新交接进入，避免启动已经完成的作业。
