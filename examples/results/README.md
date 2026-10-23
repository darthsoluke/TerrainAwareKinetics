# 实验结果：直接阅读与复现

## BONES 地形候选调查

[2026-10-23 PRO6000 结果](bones_terrain_survey_20261023/README.md) 包含完整 JSON 报告、每类 30 条固定抽样、metadata 列清单与运行记录。142,220 条标注中选出 19,935 条去重候选，人工审查与几何筛选是下一步。

[survey v2](bones_terrain_survey_v2_20261023/README.md) 增加全部2,863个动作词干的汇总表，并修正汇总行的事件时长显示。

[几何筛查 v1](bones_terrain_screen_v1_20261023/README.md) 提供每词干两条试运行与配置范围内全量筛查的报告、分组统计、完整紧凑 TSV 和执行日志。

[支撑几何拟合 v1](bones_terrain_fit_v1_20261023/README.md) 提供 6,788 条拟合结果、221 张接触图、38 条 viewer 数据和演员尺度统计。

[地形 v3](bones_terrain_fit_v3_20261023/README.md) 完成支撑 v2、顶棚、侧墙与场景变体：6,788 条主样本和 120 条 edge 样本，发布 249 张 PNG、68 条主 viewer 和 11 条 edge viewer。侧墙和变体质量门尚未通过，失败指标与样例一并保留。

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
