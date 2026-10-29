# 实验结果：直接阅读与复现

## BONES 地形候选调查

[2026-10-23 PRO6000 结果](bones_terrain_survey_20261023/README.md) 包含完整 JSON 报告、每类 30 条固定抽样、metadata 列清单与运行记录。142,220 条标注中选出 19,935 条去重候选，人工审查与几何筛选是下一步。

[survey v2](bones_terrain_survey_v2_20261023/README.md) 增加全部2,863个动作词干的汇总表，并修正汇总行的事件时长显示。

[几何筛查 v1](bones_terrain_screen_v1_20261023/README.md) 提供每词干两条试运行与配置范围内全量筛查的报告、分组统计、完整紧凑 TSV 和执行日志。

[支撑几何拟合 v1](bones_terrain_fit_v1_20261023/README.md) 提供 6,788 条拟合结果、221 张接触图、38 条 viewer 数据和演员尺度统计。

[地形 v3](bones_terrain_fit_v3_20261023/README.md) 完成支撑 v2、顶棚、侧墙与场景变体：6,788 条主样本和 120 条 edge 样本，发布 249 张 PNG、68 条主 viewer 和 11 条 edge viewer。侧墙和变体质量门尚未通过，失败指标与样例一并保留。

[地形 v4](bones_terrain_fit_v4_20261023/README.md) 提供 6,788 条同口径对照、严格资格分类、目标判定与 63 条图片/viewer 审查集合。平地与保留墙体通过目标，低姿态支撑仍未通过。

[地形 v5](bones_terrain_fit_v5_20261023/README.md) 提供 41 条定向对照、6,788 条全量统计与 77 张审查图。最大支撑超限清零，crouch 墙体显著减少；匍匐和手膝的缺支撑目标仍未通过。

[地形 v6](bones_terrain_fit_v6_20261023/README.md) 提供 71 条定向对照、v4/v5/v6 全量比较、抬高足底证据与沿墙滑动统计，以及 113 张审查图。正常支撑保护和墙数量门槛通过，低姿态碎片支撑目标仍未通过。

[地形 v7](bones_terrain_fit_v7_20261023/README.md) 提供相同 71 条定向对照和 v5/v6/v7 的 6,788 条全量比较。低姿态的缺支撑拒绝减少，但多箱片段与正常支撑保护目标未通过；86 条正常组受影响片段附有完整 patch 数值和 PNG。梯子/越障全部排除，原资格单独保留。

[地形 v9 全量](bones_terrain_fit_v9_20261024/README.md) 提供6,788条v7/v9对照、各阶段耗时、75张审查图及75条viewer。匍匐和手膝目标通过，箱体组证据丢弃比例仍超标；controller测试的1项失败保留完整traceback。

[第6步训练集构建与审计](bones_env_v1_20261024/README.md) 提供默认E集5,650条及单独E+G扩展集6,543条的分组/划分统计。81项测试通过，包含修复后的controller SDF对照；GPU3审计8,192个窗口，未进行训练。

[第7步A100 pilot预检](terrain_controller_pilot_a100_20261024/README.md)已完成依赖修复、82项测试、5,652文件迁移校验和10次单卡更新。resume未增加预期的两次更新，后续阶段停止；提供评估表、预检曲线和独立账本，正式pilot尚未启动。

[ee4db6f续跑](terrain_controller_pilot_a100_20261024/continuation_ee4db6f/README.md)通过84项测试，resume正确达到12次更新。双卡完成运行，但2个checkpoint文件和两rank参数不一致使检查未通过；旧记录保留，新证据追加发布。

随后完成的[v1正式pilot](terrain_controller_pilot_a100_20261024/pilot_be5fc1f/README.md)包含3,000次更新、14行评估和六窗口对照图，质量验收通过3/6。

[v2正式pilot](terrain_controller_pilot_v2_a100_20261025/README.md)新增1,500次闭环更新，引入混合一步监督及更强穿透代价，验收通过5/6。提供源动作基线、最终分组表和匹配更新对照；着地脚速度仍未通过，模型未部署。

[v3正式pilot](terrain_controller_pilot_v3_a100_20261025/README.md)新增1,500次几何滑步训练，验收通过4/6。穿透归零，几何着地比例降至2.12%，一步误差高于保持基线；包含v1/v2新口径补测、六窗口对照及两张三方足底曲线。

[v4从零pilot](terrain_controller_pilot_v4_a100_20261025/README.md)通过教师阶段额外门，随后闭环严重退化并在2500次评估断言退出。提供2250次保存状态的诊断图、全部可用评估、四版对照、完整异常及包含未保存250次更新的账本。

[v5预检](terrain_controller_pilot_v5_a100_20261025/README.md)通过84项测试，补充CPU检查发现归一化critic尺度原地更新导致反向传播失败。正式训练未启动；发布完整错误、输入身份与零消耗账本。

[e88523d修复后完整结果](terrain_controller_pilot_v5_a100_20261025/continuation_e88523d/README.md)包含全部预检及1500次新增更新，八项验收通过七项：无发散保护退出，末250次critic loss中位数0.2477；一步误差1.136cm仍未通过。提供五版对照、分组结果、九张图及GPU-hours，保留全仓库失败清单，未部署。

[v6测试停止记录](terrain_controller_v6_a100_20261027/README.md)包含89通过、1失败、0跳过的完整日志。脚锁测试的旋转正交性断言失败，A/B均未启动，新增训练和GPU-hours为0。

[v6修复后重跑](terrain_controller_v6_a100_20261027/continuation_f98e04d/README.md)通过90项测试及FK检查，A完成v1教师的16帧三模式后，在64帧基线评估发生维度错误。部分表格与完整异常已保留，A消耗0.004116 GPU-hours，B未启动。

[v6 rerun 2 的 A](terrain_controller_v6_a100_20261027/continuation_60fd119/README.md)发布30行总体、210行分组结果、十二张审查图与逐门判读。FK及确定性差异为0；v2加锁的源接触脚速仍未达到候选门槛。A累计0.031168 GPU-hours，B等待双卡，任务保持running。

## A/B/C 架构对照

这些JSON只包含小规模A/B/C实验的聚合结果和版本信息，没有动作轨迹、场景或权重。

```bash
python scripts/compare_arms.py --root examples/results/ablation_v3 --fixed-review examples/results/fixed_masks_summary.json
```

- `ablation_v3/` 保留原17帧评分的聚合数值。
- `fixed_masks_summary.json` 使用14个有效指令帧和三臂统一的源支撑区间，内含重新计算的持久性基线。
- 48增强行实际来自12个原始窗口、3个录制家族。
- A/B未锁定相同公共层初值，C容量较小，因此不作架构优劣定论。
- 最新解释与撤回的物理指标见 [复核报告](../../docs/records/pfnn_physics_findings_20261022.md)。

这些是已完成研究运行的记录，当前模型质量状态仍为未批准交付。
