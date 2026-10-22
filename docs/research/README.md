# 正在研究的问题

## 目前最需要解释的现象

单步姿态预测改善了，但模型使用自身历史后出现支撑滑动、Root漂移和穿透。当前证据要求把动作幅度、接触时序、实际帧间速度与长期控制分别检查。

## 方向与状态

| 方向 | 实际进展 | 阅读入口 |
| --- | --- | --- |
| 生成历史训练 | P2位移、速度、路径与混合历史目标已写入代码并做CPU检查 | [P2准备](../records/pfnn_frame_actor_p2_preparation_20261022.md) |
| 小型Transformer | C臂小试已完成；完整模型设计还有未实现部分 | [设计](../records/pfnn_lightweight_transformer_design_20261022.md)、[结果](../records/pfnn_physics_findings_20261022.md) |
| 全身三维感知 | 观测接口与受控几何已实现；全身交互数据和训练仍需扩展 | [全数据3D计划](../records/pfnn_all_data_3d_plan_20261021.md) |
| 接触与物理关系 | 调研人体/物体/场景联合建模、物理监督和约束解码 | [物理关系调研](../records/physical_relation_motion_survey_20261022.md) |
| 数据扩充 | 已有PARC等数据，调研LINGO/TRUMANS等来源 | [数据与方法](../records/whole_body_scene_methods_datasets_review_20261018.md) |

## 论文阅读入口

- PFNN：逐帧控制、运动状态、沿轨迹看地形。
- MotionBricks：层次路径/动作建模与时间token。
- DartControl：真实历史与自生成历史的课程。
- PARC：地形动作、几何目标与物理执行反馈。
- PhysPT、InterPhys：身体物理参数、接触和受力如何进入学习。
- InterDiff、ContactNets：接触相对表示与结构化动力学。

原文链接、阅读范围与本项目的判断见 [PFNN论文复核](../records/pfnn_3d_literature_review_20261021.md) 和 [物理关系调研](../records/physical_relation_motion_survey_20261022.md)。
