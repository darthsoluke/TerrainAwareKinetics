# Python核心模块

| 子目录 | 内容 |
| --- | --- |
| `data/` | 数据读取、动作合同、场景查询、模型适配、逐帧actor和评估目标 |
| `motion_backbone/` | 原生MotionBricks的Root/Pose网络和推理流程 |
| `vqvae/` | 动作压缩器、量化器和解码器 |
| `motionlib/` | 骨架、旋转、动作表示和FK工具 |
| `mb_sonic/` | 人体参考、G1重定向、物理控制器与MuJoCo接入 |
| `scene_control_v2/` | 真实provider及原生模型接线实验 |
| `scene_control_v3/` | 未训练因果控制架构原型 |

从 [架构导航](../docs/architecture/README.md) 进入当前模型，而不是按文件数量判断哪个分支正在使用。
