# BONES动作—场景训练集v1：构建与审计结果

2026-10-25在PRO6000完成，目录日期沿用任务约定。源码`4ef8af1`，CPU构建、物理GPU3审计，**没有训练或优化器更新**。

| 数据集 | 动作 | 时长秒 | 场景变体 | 演员 | 用途 |
| --- | ---: | ---: | ---: | ---: | --- |
| 默认E集 | 5,650 | 35,523.299 | 22,597 | 415 | 默认训练数据；本轮审计对象 |
| 单独E+G扩展集 | 6,543 | 41,049.305 | 26,167 | 419 | 包含默认E集和893条G；单独保留 |
| 开发集E | 388 | 2,685.266 | 1,552 | 158 | 全量构建前的门槛检查 |

三个构建均零错误，最大FK往返误差0.0048mm，最多14个箱体；演员划分无交叉。81项测试通过。默认集8,192窗口审计的头顶净空顺序、入体率与有限损失检查通过。单角色一步P50/P99为12.902/16.991ms。

| 内容 | 文件 |
| --- | --- |
| 命令、测试、分组/划分表、全部审计数值 | [执行记录](../../../docs/records/bones_env_v1_pro6000_20261024.md) |
| 默认构建原始报告 | [build/report.json](build/report.json) |
| 默认集审计原始报告 | [audit/report.json](audit/report.json) |
| 射线入体贡献，按组 | [audit/inside_by_group.json](audit/inside_by_group.json) |
| 扩展集、开发集原始报告 | [geometry_only/report.json](geometry_only/report.json)、[dev/report.json](dev/report.json) |
| 分组、划分及交叉计数 | [groups.tsv](groups.tsv)、[splits.tsv](splits.tsv)、[group_splits.tsv](group_splits.tsv)、[counts.json](counts.json) |
| 源码、命令、耗时、GPU和数据身份 | [execution.json](execution.json) |
| 测试输出 | [数据集](tests/dataset.txt)、[controller](tests/controller.txt)、[射线](tests/ray.txt)、[地形](tests/terrain.txt) |

时长按动作计一次，不乘场景变体数。演员在每行内去重，同一演员可能出现在多个动作组。`geometry_only`目录包含E和G，不作为默认集。manifest与动作分片留在PRO6000，公共目录只含报告和汇总。
