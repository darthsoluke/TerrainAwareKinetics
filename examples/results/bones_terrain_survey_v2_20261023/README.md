# BONES-SEED survey v2：动作词干清单

2026-10-23 在 PRO6000 上运行 `d313b65`。测试最后一行：`9 passed, 1 skipped in 2.04s`。

执行说明和迁移进度见 [v2记录](../../../docs/records/bones_terrain_survey_v2_pro6000_20261023.md)。

```text
stems: {'unique': 2863, 'with_any_hit': 812, 'with_dimension_tokens': 17}
```

| 文件 | 内容 |
| --- | --- |
| [stems.tsv](stems.tsv) | 全部2,863个动作词干，包括未命中关键词的词干 |
| [survey_report.json](survey_report.json) | 完整分桶统计、规则、词干数量与输入哈希 |
| [stdout.txt](stdout.txt) | 原样保存的汇总表和stems行 |
| [tests.txt](tests.txt) | PRO6000测试输出 |
| [execution_receipt.json](execution_receipt.json) | survey完成时的环境、迁移授权和磁盘余量快照 |
| [SHA256SUMS](SHA256SUMS) | 五个公开结果文件的SHA256 |
| [publication_manifest.json](publication_manifest.json) | 原始结果与公开副本的哈希对应 |
| [transfer_source_summary.json](transfer_source_summary.json) | 全部原始BVH源清单的数量、大小和哈希；目标端验证完成后另附报告 |

每个词干列出各桶命中比例、原始与镜像数量、家族和演员数量、尺寸词，以及一条示例描述。812个词干有至少一次命中，2,051个没有命中；17个含尺寸词。

v2保留v1的关键词规则，候选总量仍为19,935条。汇总行的`event_s`已改为`-`。候选进入地形拟合时，需要几何证据；词干清单用于逐组审核。

`candidates.jsonl`保留在PRO6000，没有发布。公开文件只替换私人机器路径，数值和全部词干行保持原样。
