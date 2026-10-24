# BONES terrain v9：全量结果

实际运行于2026-10-25，目录日期沿用任务约定。源码为 `9ad7edddb585cc9588d3909ebdfb9f5e39393db8`，PRO6000 CPU、单池68 workers，无训练或调参。

**6,788条拟合全部成功，耗时829秒。** 最大支撑蒙皮超2cm为零，匍匐/手膝/跪姿等指定目标通过；箱体组 `body_rests_lower` 丢弃比例为6/115（5.22%），超过1%门槛。Controller测试8通过、1失败，完整错误已保留。

| 内容 | 入口 |
| --- | --- |
| 测试、命令、目标判定、20条F样本与75条图片审查 | [完整执行记录](../../../docs/records/bones_terrain_fit_v9_pro6000_20261024.md) |
| v7/v9全字段逐组对照 | [Markdown](comparison.md)、[TSV](comparison.tsv)、[JSON](comparison.json) |
| 6,788条紧凑指标 | [fits_compact.tsv](fits_compact.tsv) |
| 分组汇总、阶段耗时和审查记录 | [by_group.json](by_group.json)、[report.json](report.json) |
| 箱体组6块被丢弃证据的详情 | [box_body_drop_details.json](box_body_drop_details.json) |
| 三组测试原始输出 | [地形](tests/terrain_tests.txt)、[射线](tests/ray_tests.txt)、[controller完整traceback](tests/controller_tests.txt) |
| 75条审阅动画与几何 | [viewer/index.json](viewer/index.json) |

`viz/`保留75张原始接触图，共10,967,917 bytes；完整记录逐条链接。选取11组各5张确定性抽样，再加入已知尺寸组F最高的20张。没有最小支撑超限片段，因此本轮没有额外worst图。20张补图直接使用已保存fit和缓存关节，没有重新拟合。

完整fit和6,788条motion cache留在PRO6000的私有运行目录。发布的75条viewer均有对应PNG，索引只列实际存在的文件。所有文字为UTF-8，并检查无U+FFFD。
