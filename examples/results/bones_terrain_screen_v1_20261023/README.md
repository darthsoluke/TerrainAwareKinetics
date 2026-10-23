# BONES 几何筛查 v1

2026-10-23 在 PRO6000 的 CPU 上执行，源码 `0f97409`，16 个 worker。

| 运行 | 处理 | 成功 | 拒绝 | 耗时 |
| --- | ---: | ---: | ---: | ---: |
| 每词干两条试运行 | 196 | 196 | 0 | 20.4 秒 |
| 配置范围内全量 | 6,795 | 6,788 | 7 | 422.8 秒 |

测试：`19 passed in 4.46s`。七条拒绝均为源骨架偏差超过 2 mm。执行步骤、完整错误和结果解释见 [执行记录](../../../docs/records/bones_terrain_screen_pro6000_20261023.md)。

## 查看结果

| 内容 | 试运行 | 全量 |
| --- | --- | --- |
| 完整报告、阈值和源码哈希 | [report.json](pilot/report.json) | [report.json](full/report.json) |
| 打印的分组 JSON | [by_group.json](pilot/by_group.json) | [by_group.json](full/by_group.json) |
| 98 个词干的统计 | [by_stem.json](pilot/by_stem.json) | [by_stem.json](full/by_stem.json) |
| 完整逐片段表 | [clips_compact.tsv](pilot/clips_compact.tsv) | [clips_compact.tsv](full/clips_compact.tsv) |
| stdout 最后 60 行 | [stdout_tail.txt](pilot/stdout_tail.txt) | [stdout_tail.txt](full/stdout_tail.txt) |

TSV 分别有 196 和 6,795 条数据行，另有一行表头。最后一列为错误信息；成功片段该列为空。

## 执行证据

- [dry/report.json](dry/report.json)：从 71,132 条原始 BVH 中按配置选中 6,795 条，词干缺失列表为空。
- [tests_fixed.txt](execution/tests_fixed.txt)、[imports_fixed.txt](execution/imports_fixed.txt)：导入路径修复后的测试与 Kimodo/SciPy 导入输出。
- [tests.txt](execution/tests.txt)、[imports.txt](execution/imports.txt)：首次运行的测试和 Kimodo 未找到错误。
- [environment.json](execution/environment.json)、[worker_init.txt](execution/worker_init.txt)：独立 Python 环境、依赖和蒙皮哈希、CPU 初始化结果。
- [resources_host.txt](execution/resources_host.txt)：72 个可用 CPU、498 GiB 内存；[resources.txt](execution/resources.txt) 保留受 OMP 单线程设置影响的首次查询。
- [dry.stdout.txt](execution/dry.stdout.txt)、[pilot.stdout.txt](execution/pilot.stdout.txt)、[full.stdout.txt](execution/full.stdout.txt)：完整执行输出，全量日志末行记录退出码 0。
- [input_lineage.json](execution/input_lineage.json)：每条输入哈希与原始迁移清单的核对结果。
- [publication_manifest.json](publication_manifest.json)、[SHA256SUMS](SHA256SUMS)：原始与公开文件的哈希对应，以及本目录文件校验清单。

`selection.jsonl` 和 `clips.jsonl` 留在 PRO6000。这里的报告和紧凑表只替换私人机器路径，保留全部科学数值及失败记录。
