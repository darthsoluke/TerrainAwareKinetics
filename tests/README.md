# 测试与验证

这些测试覆盖不同阶段的模型、数据、时钟和协议，依赖范围不同。

| 检查 | 运行需求 |
| --- | --- |
| `python tools/check_repository.py` | Python标准库；检查公开仓库结构与文档 |
| `pytest tests/test_bones_terrain_survey.py -q` | pytest、pandas；完整动作依赖不可用时跳过一项家族兼容性检查 |
| `pytest tests/test_bones_terrain_screen.py -q` | NumPy、SciPy、pytest；合成数据检查筛查与选择，不需要Kimodo |
| `pytest tests/test_bones_transfer.py -q` | pytest和标准库；用临时文件检查只读源、镜像排除、校验与空间门槛 |
| `pytest tests/test_pfnn_frame_training.py -q` | PyTorch、NumPy、pytest；用CPU替代训练边界的runner检查 |
| `pytest tests/test_pfnn_frame_closed_loop.py -q` | 人体表示/骨架依赖；P2目标和时间梯度检查 |
| `pytest tests/test_pfnn_ablation_review.py -q` | PyTorch及模型导入依赖；评分mask、误差与汇总检查 |
| 真实场景、CUDA、协议和渲染测试 | 依各测试说明准备模型、场景、设备或本机服务 |

代码测试通过表示对应实现合同被检查。长期动作控制、完整皮肤几何与本地实时交互有独立实验门，见 [项目全景](../docs/START_HERE.md#13-我们怎样判断模型真的变好了)。
