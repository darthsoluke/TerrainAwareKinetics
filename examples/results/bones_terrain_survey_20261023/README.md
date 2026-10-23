# BONES-SEED 地形候选调查结果

2026-10-23 在 PRO6000 上以 `f2d4c5c` 运行，处理 142,220 条 timeline。去重候选 19,935 条，其中原始 9,969 条、镜像 9,966 条。

完整解释、实际命令、汇总表和误命中实例见 [执行报告](../../../docs/records/bones_terrain_survey_pro6000_20261023.md)。

| 文件 | 用途 |
| --- | --- |
| [survey_report.json](survey_report.json) | 完整统计、规则、命中词和输入哈希 |
| [sample_candidates_seed0.txt](sample_candidates_seed0.txt) | 每类 30 条，固定随机种子 0，共 90 条待人工审查的候选 |
| [pkl_describe.json](pkl_describe.json) | 131,454 行 pickle 的全部列、示例和缺失字段 |
| [execution_receipt.json](execution_receipt.json) | 执行环境、代码版本、双端校验和条件步骤状态 |
| [timeline_summary.txt](timeline_summary.txt) | 调查打印的完整汇总表 |
| [tests.txt](tests.txt) | PRO6000 测试输出：7 passed, 1 skipped |
| [SHA256SUMS](SHA256SUMS) | 六个公开结果文件的 SHA256 |
| [publication_manifest.json](publication_manifest.json) | 原始文件与公开副本的哈希对应 |

私人机器路径已替换为角色占位符。数值、规则和 90 条抽样未改。完整源标注和候选动作文件保留在私有制品存储；本目录提供此次结果与人工审阅样本。

本轮没有 BVH 几何匹配，`inputs.bvh_root` 为 null；`with_bvh: 0` 对应这项未执行的检查。pickle 缺少 `content_vertical_move` 和 `content_body_position`，所以按计划跳过带 metadata 的第二次调查。
