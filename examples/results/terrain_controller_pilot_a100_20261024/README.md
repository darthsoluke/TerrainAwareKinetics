# A100地形控制器pilot：预检历史与续跑

最新的[`be5fc1f`双卡pilot](pilot_be5fc1f/README.md)已完成3,000次更新，耗时47分12.48秒。84项测试、两卡参数一致性和运行计数通过；六项质量验收中一步误差改善、穿透帧比例和滑步条件未通过。完整命令和历史见[执行记录第10节](../../../docs/records/terrain_controller_pilot_a100_20261024.md#10-be5fc1f-pilot)。

此前的[ee4db6f续跑](continuation_ee4db6f/README.md)已通过84项测试和resume2；当时双卡保存数量与模型参数一致性检查未通过，正式pilot尚未启动。以下保留9785c01的早期结果。

执行源码为`9785c01`。依赖已补齐，**82项测试全部通过**；默认E集5,652个文件逐一SHA-256核对一致。单卡冒烟完成10次更新，但resume没有增加预期的2次更新，因此停止后续GPU阶段，正式pilot未启动。

- [完整执行记录、阶段耗时与账本](../../../docs/records/terrain_controller_pilot_a100_20261024.md)
- [当前结构化报告](report.json)
- [逐次评估表](evaluation_table.tsv)
- [预检曲线](preflight_curves.png)
- [最初的依赖失败记录，已解决](preflight_report.json)

预检合计**控制器10次更新、critic 5次更新、160窗口曝光、320训练闭环生成帧**；正式pilot均为0。恢复前后model和critic权重逐tensor相等。原始日志、JUnit和checkpoint留在A100；这里保留小型报告与图表。
