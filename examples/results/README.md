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

[第7步A100 pilot准备](terrain_controller_pilot_a100_20261024/README.md)停在82项测试门槛：81通过、1项因缺少pytz失败，保留完整traceback。数据迁移和GPU训练尚未启动，更新账本为零。

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
