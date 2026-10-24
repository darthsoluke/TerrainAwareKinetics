# 架构：按实际分支理解模型

入门解释和流程图见 [项目全景第7～11节](../START_HERE.md#7-架构一我们改造过的地形感知-motionbricks)。

| 分支 | 实际结构 | 状态与入口 |
| --- | --- | --- |
| 地形感知MotionBricks | Codec/VQ-VAE＋Root Transformer＋Pose Transformer＋地形适配 | [实际配置审阅](../records/training_inference_architecture_review_20261018.md) |
| P1单帧模型 | 4帧历史＋256维GRU；预测路径后重采样环境，再生成下一帧 | [P1报告](../records/pfnn_frame_actor_p1_review_20261022.md) |
| A/B/C对照 | 16帧窗口；A为GRU均值，B加空间注意力，C为小型因果Transformer | [最新复核](../records/pfnn_physics_findings_20261022.md) |
| G1物理执行 | MB人体参考→G1参考→SONIC/ProtoMotions→MuJoCo | [控制器比较](../records/mb_tracker_comparison_20261019.md) |
| scene_control_v3 | 统一GRU输出Root、身体、15路接触/释放与路径预览 | [未训练原型](../records/scene_control_v3_model_20261018.md) |
| 完整轻量Transformer | 运动时序、24空间token、身体区域查询和因果输出 | [设计方案](../records/pfnn_lightweight_transformer_design_20261022.md) |
| 地形感知动作Transformer | 16帧历史、部位射线token、路径站点token、6部位query；可微闭环训练目标 | [代码说明](../records/terrain_motion_transformer_20261024.md)，未训练 |

## 各分支共有的环境观测

在当前PFNN逐帧分支，三个512点区域分别观察身体、期望方向和实际运动方向。共享编码器产生24个空间token。Root预测的路径会触发走廊精细查询，未知区域通过有效性标记表达。

## 核心源码

| 内容 | 文件 |
| --- | --- |
| P1单帧模型 | [pfnn_frame_actor.py](../../motionbricks/data/pfnn_frame_actor.py) |
| A/B/C模型 | [pfnn_frame_actor_v2.py](../../motionbricks/data/pfnn_frame_actor_v2.py) |
| 地形编码与旧MB适配 | [unified_terrain.py](../../motionbricks/data/unified_terrain.py) |
| 三维采样 | [unified_scene_field.py](../../motionbricks/data/unified_scene_field.py) |
| 场景组合、批量查询、路径观察 | [pfnn_3d_scene.py](../../motionbricks/data/pfnn_3d_scene.py) |
| P2闭环目标 | [pfnn_frame_closed_loop.py](../../motionbricks/data/pfnn_frame_closed_loop.py) |

完整Transformer设计中的六身体查询与显式时间编码，不全部等同于已经训练的C臂。模型状态以实际代码和结果报告为准。
