# 2026-10-23 源码与文档发布检查

本次将旧MotionBricks工作区的源码和研究文档迁入TerrainAwareKinetics，新增中文入门文档、主题入口、结果示例和仓库检查工具。没有启动训练或替换运行模型。

## 检查结果

| 检查 | 结果 |
| --- | --- |
| Python语法、JSON、本地文档链接与公开内容 | `python tools/check_repository.py` 通过 |
| A/B/C原口径与统一源支撑结果表 | 仓库内JSON直接读取，与最新复核表一致 |
| P1训练runner的CPU合同 | `tests/test_pfnn_frame_training.py`：62项通过 |
| 逐帧actor、闭环目标、评分修正 | 相关三个测试文件：27项通过 |

测试环境：Windows，Python 3.10.20，PyTorch 2.11.0+cu128，NumPy 2.2.6，pytest 9.1.1。上述测试在CPU运行；人体依赖使用本机已准备的Kimodo参考版本。

```bash
python tools/check_repository.py
python scripts/compare_arms.py --root examples/results/ablation_v3 --fixed-review examples/results/fixed_masks_summary.json
python -m pytest tests/test_pfnn_frame_training.py -q
python -m pytest tests/test_pfnn_frame_actor.py tests/test_pfnn_frame_closed_loop.py tests/test_pfnn_ablation_review.py -q
```

后两个命令需开发依赖；人体测试还需按 [开发说明](README.md) 准备Kimodo并加入Python模块搜索路径。

## 来源与整理

初始导入覆盖1,539份源码、配置、测试、网页与历史文档，映射和内容哈希记录在 [source_snapshot.json](source_snapshot.json)。目录导航、CI、检查工具和精简结果另行新增。公开副本保留模型结构与数学计算，机器地址、私人路径和制品链接改为模板。

检查中修正了将四段软件版本号误识别为机器地址的导入规则，保留NVIDIA依赖版本变化测试。导入工具的额外私人替换项从仓库外配置读取。

完整训练、场景评估和浏览器30Hz质量验收沿用各实验的记录，本次发布没有重跑这些长时实验。
