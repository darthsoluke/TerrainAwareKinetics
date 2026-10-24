# 地形感知动作 Transformer 与可微闭环训练：代码说明

2026-10-24。本文说明新加入的 `motionbricks/controller/` 代码。它实现 [轻量 Transformer 设计](pfnn_lightweight_transformer_design_20261022.md) 的输入与结构部分，以及受 SuperMimic（CoRL 2026）启发的可微闭环训练目标。**代码尚未训练，也未在本机运行 torch 测试**（本机无 torch），由 PRO6000 执行 [测试](../../tests/test_terrain_controller.py) 后才算可用。启动训练仍需单独的有界计划和预检门。

## 1. 模块

| 文件 | 内容 |
| --- | --- |
| [scene_sdf.py](../../motionbricks/controller/scene_sdf.py) | 有向长方体集合的带符号距离，可对点求导；穿透代价、支撑代价；float64 参考实现 |
| [tokens.py](../../motionbricks/controller/tokens.py) | 按关节名分 6 个身体部位；把骨骼射线整理成部位 token；从预测路径站点打射线的前瞻 token；傅里叶位置编码 |
| [terrain_motion_transformer.py](../../motionbricks/controller/terrain_motion_transformer.py) | 模型本身，`plan` / `act` 两阶段，输出与 P1 `FrameActor` 兼容 |
| [closed_loop.py](../../motionbricks/controller/closed_loop.py) | 用模型自己的输出连续生成 H 帧，代价可微地回传；critic 估计 H 帧之后的代价 |
| [returns.py](../../motionbricks/controller/returns.py) | TD(λ) 目标及参考实现 |

射线求交函数从 [skeleton_ray_sampler.py](../../motionbricks/data/skeleton_ray_sampler.py) 抽出共用，并支持每个样本使用不同场景（训练批次混合场景时需要）。

## 2. 每帧输入

全部在当前朝向坐标系中表达，根节点水平位置相对最后一帧。

| token | 数量 | 来源 |
| --- | --- | --- |
| 历史帧 | 16 | 原生全局 SOMA30（366 维），可学习的帧龄嵌入；开头不足时用掩码 |
| 指令 | 1 | 左摇杆速度、右摇杆朝向差，换到朝向坐标系 |
| 身体部位 | 6 | 根、躯干、左右臂、左右腿；各自的骨骼射线（距离、法向、命中、在实体内） |
| 路径站点 | 8 | 模型先预测 0.1–0.8 秒的 8 个站点，再从每个站点向下、上、左、右、前打射线 |
| 体积（可选） | 24 | 现有 SharedTerrainEncoder 的空间 token |

## 3. 结构

1. **plan**：历史帧经 3 层自注意力编码；指令 query 读取历史、指令、部位 token，输出 8 个路径站点。
2. 调用方沿路径查询场景（`StationProber`）。
3. **act**：6 个部位 query 彼此自注意力，并交叉注意力读取全部上下文和站点 token。每个关节的旋转增量只取自它所属部位的 token；根部位输出位移，两条腿输出 4 个脚接触。

旋转和位移输出层初始化为零，未训练的模型保持上一帧姿态。宽度 128、4 头，每帧约 40 个 token。

## 4. 可微闭环训练

SuperMimic 需要学习一个世界模型，因为物理仿真不可微。运动学控制器的下一状态就是模型输出经官方正向运动学（`advance`），场景代价是解析距离，整条链本身可微，不需要世界模型。

每次从真实片段取 16 帧历史，模型用自己的输出连续生成 H 帧（默认 16），每帧的代价包括：

| 代价 | 含义 |
| --- | --- |
| 模仿 | 只在源动作的下一帧仍然有效时（同一指令、同一场景）比较位移和局部旋转 |
| 指令 | 水平速度跟左摇杆，朝向跟右摇杆 |
| 穿透 | 关节（按半径视为球体）进入任何实体；脚和脚趾半径为零 |
| 支撑 | 模型判为着地的脚底离最近表面的距离 |
| 滑步 | 着地脚底的水平速度 |
| 平滑 | 关节加速度 |

策略最小化 Σγᵗcₜ + γᴴV(s_H)，梯度穿过每一个生成帧和末状态的价值。末状态价值用冻结权重计算，只更新策略；critic 用 TD(λ) 目标单独训练。脚底位置用官方脚底偏移近似（脚平放时准确）。

## 5. 待确认

- torch 测试在 PRO6000 上的结果，包括平移不变性、掩码帧不影响输出、端到端梯度。
- 3080 上的单帧耗时（模型 + 射线 + 站点），目标是整帧 60fps。
- 与旧 P2 目标 [pfnn_frame_closed_loop.py](../../motionbricks/data/pfnn_frame_closed_loop.py) 的关系：旧目标只用源动作做标签，新目标在源动作不再适用时用任务代价，两者可以组合。
- 历史帧只做了平移归一，没有旋转到朝向坐标系；朝向不变性目前依靠数据增强和朝向坐标系下的其他输入。
