# TerrainAwareKinetics

面向三维环境的实时人体动作生成研究：用手柄控制移动方向与身体朝向，让角色连续地走、跑、停、转向，并适应地形。

本仓库汇集了基于 MotionBricks 的改造、逐帧 GRU / Transformer 实验、人体与 G1 物理执行的接入代码，以及完整研究记录。源码基础来自 [NVIDIA GR00T / MotionBricks](https://github.com/NVlabs/GR00T-WholeBodyControl)。

**第一次接触这个项目，请从 [零基础项目全景](docs/START_HERE.md) 开始。** 它解释我们要做什么、有哪些数据、各代模型怎样运行、实验进展到哪里。

## 当前进展

最近模型结果复核：**2026-10-22**；BONES 数据调查：**2026-10-23**。

- 已处理 BONES、LAFAN、PARC、PFNN 等动作来源，已有动作与地形配对、SOMA30 人体表示和 15 张 PFNN 演示地图。
- 地形感知 MotionBricks、逐帧 P1 GRU、A/B/C 小规模结构对照均有实际训练记录。
- 浏览器控制、WASD、Xbox 双摇杆、快速切图和同窗口调参已有实现与实测记录。
- 最新评估显示：单步动作拟合改善，连续生成的支撑稳定性与穿透质量仍未通过交付标准。
- P2 闭环训练目标已有部分实现；更完整的轻量 Transformer 和物理关系建模仍在研究中。
- BONES 标注调查已在 PRO6000 完成：142,220 条标注中，19,935 条命中地形候选类别。每类 30 条抽样已发布，包含需要人工核查的关键词误命中。

最新结论从 [A/B/C 复核](docs/records/pfnn_physics_findings_20261022.md) 和 [最新交接](docs/records/pfnn_physics_audit_status_20261022.md) 查看。

最新数据工作见 [BONES 调查报告](docs/records/bones_terrain_survey_pro6000_20261023.md)；[JSON、90 条抽样与运行记录](examples/results/bones_terrain_survey_20261023/README.md) 可直接读取。

随后完成的 [survey v2](docs/records/bones_terrain_survey_v2_pro6000_20261023.md) 汇总了 [2,863 个动作词干](examples/results/bones_terrain_survey_v2_20261023/stems.tsv)，包含812个有关键词命中的词干和17个带尺寸词的词干。

全部 **71,132 条原始 BONES BVH** 已迁移到 PRO6000，保留日期目录并逐文件验证通过，差异为0；见 [迁移完成记录](examples/results/bones_terrain_survey_v2_20261023/transfer/README.md)。

后续 [BONES 几何筛查](docs/records/bones_terrain_screen_pro6000_20261023.md) 的试运行、全量结果和逐片段紧凑表见 [结果目录](examples/results/bones_terrain_screen_v1_20261023/README.md)。

[支撑几何拟合与可视化](examples/results/bones_terrain_fit_v1_20261023/README.md) 已完成 6,788 条拟合，发布 221 张接触图、38 条 viewer 数据及完整分组指标。

2026-10-24 完成 [地形 v3 四路运行](examples/results/bones_terrain_fit_v3_20261023/README.md)：增加顶棚、侧墙和场景变体，发布 249 张审查图与 79 条 viewer 数据。变体接受率 86.88%，侧墙仍有超限，详见 [执行与质量报告](docs/records/bones_terrain_fit_v3_pro6000_20261023.md)。

随后完成 [v4 三路全量与点检](examples/results/bones_terrain_fit_v4_20261023/README.md)：平地配墙降为 4/900，保留墙体无超限；低姿态支撑出现退步。报告按新规则列出 4,814 条环境可解释、639 条仅几何一致及 1,335 条拒绝，发布 63 条审查图/viewer。

此前的 [地形 v7 全量与审查](examples/results/bones_terrain_fit_v7_20261023/README.md) 已完成 6,788 条 CPU 拟合：最大支撑超限保持零，匍匐/手膝的 E+G 比例升至 84.33% / 86.88%；残留多箱片段与正常支撑保护目标仍未通过。报告保留 86 条正常组受影响片段的数值和图片，以及 v5/v6/v7 完整对照。

2026-10-25 完成 [地形 v9 全量与 controller 测试](examples/results/bones_terrain_fit_v9_20261024/README.md)：6,788条CPU拟合零失败，耗时829秒。匍匐/手膝目标通过，箱体组证据丢弃比例5.22%仍超标；controller有1项SDF距离对照失败。发布75张图、75条viewer、全量对照及完整错误记录。

随后完成 [第6步动作—场景训练集构建与审计](examples/results/bones_env_v1_20261024/README.md)：默认E集5,650条、单独E+G扩展集6,543条，演员划分无交叉。SDF修复后81项测试全部通过；GPU3完成8,192窗口审计，单角色一步P50/P99为12.902/16.991ms。本轮没有训练。

[第7步A100双卡pilot](examples/results/terrain_controller_pilot_a100_20261024/pilot_be5fc1f/README.md)完成3,000次更新，耗时47分12.48秒。84项测试及两卡同步通过，根漂移和骨盆高度达标；一步误差改善、穿透和滑步条件仍未通过。已发布完整评估表、训练曲线和六个固定窗口的源动作/闭环对照图，模型未部署。

[pilot v2](examples/results/terrain_controller_pilot_v2_a100_20261025/README.md)从同一教师checkpoint新增1,500次更新，耗时26分48秒，六项验收通过5项。一步误差改善45.95%，穿透超限比例降至0.195%；着地脚速度仍未达标，wall_contact分组出现穿透退步。源动作基线、分组结果和v1/v2对照已发布，模型未部署。

[pilot v3](examples/results/terrain_controller_pilot_v3_a100_20261025/README.md)使用几何滑步代价，新增1,500次更新后穿透超限降至0，但几何着地比例仅2.12%，一步拟合也退步，验收通过4/6。两张源动作/v2/v3足底曲线显示抬脚和支撑流失；完整对照已发布，未追加训练或部署。

[pilot v4](examples/results/terrain_controller_pilot_v4_a100_20261025/README.md)从零加入支撑脚教师项和源接触约束，1500次教师滑步门通过；闭环随后严重退化，在2500次评估中断言退出。已计入2500次消耗，checkpoint保存到2250次，完整失败记录和诊断图已发布，未恢复或部署。

[pilot v5预检](examples/results/terrain_controller_pilot_v5_a100_20261025/README.md)的84项测试通过；补充CPU检查发现归一化critic的尺度原地更新破坏反向传播，后续预检和正式训练停止。新增训练消耗为0，复现命令与完整错误已发布。

## 按你的问题阅读

| 你想知道什么 | 入口 |
| --- | --- |
| 我没有相关经验，项目到底在做什么？ | [从零开始](docs/START_HERE.md) |
| 我们拥有哪些动作、场景和训练样本？ | [数据说明](docs/data/README.md) |
| MotionBricks、GRU、Transformer 有什么关系？ | [架构说明](docs/architecture/README.md) |
| 历史上尝试过哪些路线？ | [历史与账本](docs/history/README.md) |
| 当前重点研究什么？ | [研究方向](docs/research/README.md) |
| 我想运行源码或复现结果表 | [开发与复现](docs/development/README.md) |
| 我需要原始详细执行记录 | [历史记录索引](docs/records/README.md) |

## 目录

```text
TerrainAwareKinetics/
  docs/                 入门、数据、架构、历史、研究和详细记录
  motionbricks/         Python 模块：表示、模型、地形感知、控制器接入
  scripts/              数据准备、训练、评估、浏览器服务与历史实验入口
  configs/              版本化实验配置
  tests/                模型合同、数学、时钟、协议与评估检查
  web/                  浏览器界面源码
  examples/results/     可以直接读取的精简数值结果
  tools/                仓库检查和旧工作区导入工具
  patches/              外部依赖的已有适配补丁
  setup.py              Python 包安装入口
  LICENSE               上游源码与模型许可说明
```

`scripts/` 内保留已有模块名，使 `scripts.<name>` 导入仍然成立。每个顶层目录都提供 README，列出常用入口；详细实验按 `docs/records/` 追溯。

## 先复现一张表

有 Python 就可以读取仓库内的精简结果，无需下载模型和动捕：

```bash
python scripts/compare_arms.py --root examples/results/ablation_v3 --fixed-review examples/results/fixed_masks_summary.json
```

输出包括 A/B/C 三臂和持久性基线。各指标的单位、评分时序与限制见 [结果说明](examples/results/README.md)。

运行模型和训练需要另外准备相应权重、动作数据、场景与官方人体资源；说明见 [开发与复现](docs/development/README.md)。

## 来源与许可

项目保留了上游源码、版权与许可信息，见 [LICENSE](LICENSE)、[来源说明](NOTICE.md) 和 [原始 MotionBricks 说明](docs/upstream/MotionBricks.md)。数据集、人体资源、第三方跟踪器和模型权重按各自来源管理。

公开仓库同步源码、文档与精简统计结果。模型权重、动捕、场景数据、训练缓存和私人运行配置位于独立制品存储。
