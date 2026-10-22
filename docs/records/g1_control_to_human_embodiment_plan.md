# G1 控制链路到人体化 Embodiment 实施基准

本文用于把当前三路调研结果沉淀成后续实现基准。结论以仓库源码为准，重点区分三件事：

- MotionBricks 负责生成/转换运动表示，当前 Python demo 主要面向 MuJoCo 预览和下游消费。
- Unitree G1 真实控制链路在 C++ deploy 栈内完成，最终输出 `LowCmd_`。
- 人体化 embodiment 不能只靠换皮肤完成；视觉预览、物理仿真和真实机器人控制是三层不同目标。

## 1. 官方 Unitree G1 控制链路

源码中真实 G1 部署链路可以概括为：

`Input -> MotionBricks/Planner -> MotionSequence/qpos -> Policy/WBC -> MotorCommand -> LowCmd`

这里的 `Policy/WBC` 指 C++ 部署栈中的控制策略/全身控制输出路径；仓库里的 Decoupled WBC 是另一路组件，不等价于 MotionBricks 直接发布 DDS。

### 1.1 Input

G1 deploy 主入口在 `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`。核心类是 `G1Deploy`：

- `G1Deploy::Input()` 周期性调用 `input_interface_->update()` 和 `input_interface_->handle_input(...)`。
- `input_interface/zmq_endpoint_interface.hpp` 的 `ZMQEndpointInterface` 可把外部流式输入解码为 `MotionSequence`，也支持 token-only 协议把 `token_state` 写入外部 token 状态。
- `input_interface/streamed_motion_merger.hpp` 的 `StreamedMotionMerger` 负责把分片流式 motion 合并进 `MotionSequence`。

### 1.2 MotionBricks / Planner

MotionBricks Python demo 路径在 `motionbricks/scripts/interactive_demo_g1.py`：

- `main(args)` 创建 `navigation_demo(args)`。
- 每帧调用 `demo_agent.full_agent.get_next_frame()` 得到当前 `qpos`，写入 MuJoCo `mj_data.qpos`。
- `demo_agent.controller.generate_control_signals(...)` 生成控制条件。
- `demo_agent.full_agent.generate_new_frames(control_signals, ...)` 生成后续 motion frames。

对应实现分布在：

- `motionbricks/motionbricks/motion_backbone/demo/utils.py`：`navigation_demo` 初始化模型、控制器和 MuJoCo simulator。
- `motionbricks/motionbricks/motion_backbone/demo/controllers.py`：`base_controller`、`WASD_controller.generate_control_signals(...)`、`random_controller.generate_control_signals(...)`。
- `motionbricks/motionbricks/motion_backbone/demo/full_agent.py`：`full_navigation_agent.generate_new_frames(...)`、`get_next_frame()`、`get_context_motion_features()`、`get_context_mujoco_qpos()`。

C++ 部署侧 planner 在：

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner.hpp`：`LocalMotionPlannerBase`、`ResampleGeneratedSequence50Hz()`、`UpdateContextFromMotion(...)`。
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner_tensorrt.hpp`：`LocalMotionPlannerTensorRT` 是当前 TensorRT planner backend。
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`：`G1Deploy::Planner()` 发起/更新规划，`G1Deploy::CurrentFrameAdvancement()` 把可用 planner motion 接入 `current_motion_`。

### 1.3 MotionSequence / qpos

`MotionSequence` 定义在 `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/motion_data_reader.hpp`，用于 C++ deploy、planner 和流式输入之间传递 motion：

- `MotionSequence::JointPositions(frame)` / `JointVelocities(frame)` 存 G1 关节位置和速度。
- `MotionSequence::BodyPositions(frame)` / `BodyQuaternions(frame)` 存 root/body 位姿。
- 可选字段 `SmplJoints(frame)` / `SmplPoses(frame)` 存人体/SMPL 相关数据。
- `MotionDataReader::ReadFromCSV(...)` 可从 `joint_pos.csv`、`joint_vel.csv`、`body_pos.csv`、`body_quat.csv`、`smpl_joint.csv`、`smpl_pose.csv` 等文件读入 motion。

MotionBricks Python 侧的 `mujoco_qpos` 是 36 维：

- `root 7`：root translation 3 + root quaternion 4。
- `29`：G1 MuJoCo 关节自由度。
- 源码依据：`motionbricks/docs/motion_representation.md`、`motionbricks/motionbricks/helper/mujoco_helper.py` 的 `mujoco_qpos_converter`。

注意：Python demo 的 `mujoco_qpos` 缓存在 `full_navigation_agent.frames['mujoco_qpos']` 中；C++ deploy 的 `MotionSequence` 是另一个明确的数据容器，二者可以互相适配，但不是同一个对象。

### 1.4 Policy / WBC

C++ 控制周期在 `G1Deploy::Control()`：

- 收集机器人状态和输入。
- `GatherObservations()` 组装 policy observation。
- `CreatePolicyCommand()` 调用 `PolicyEngine::Infer()`，读取 policy 输出 action。
- action 经 `isaaclab_to_mujoco` 映射、`g1_action_scale` 缩放，再叠加 `default_angles`，写入 `MotorCommand`。

相关文件：

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/control_policy.hpp`：`PolicyEngine`，输入 tensor 名为 `obs_dict`，输出 tensor 名为 `action`，action 维度为 `G1_NUM_MOTOR=29`。
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/encoder.hpp`：`EncoderEngine`，输入 `obs_dict`，输出 `encoded_tokens`，供 policy 的 `token_state` 使用。
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/observation_config.hpp`：observation 名称、维度和组装规则来自 YAML 配置，并在初始化时校验总维度。

### 1.5 LowCmd

真实硬件输出在 `G1Deploy::LowCommandWriter()`：

- 从 `motor_command_buffer_` 读取 `MotorCommand`。
- 填充 Unitree `LowCmd_` 的每个 `motor_cmd()[i]`：`tau`、`q`、`dq`、`kp`、`kd`。
- 设置 `mode_pr`、`mode_machine`、CRC。
- 通过 `lowcmd_publisher_->Write(dds_low_command)` 发布到 `rt/lowcmd`。

相关硬接口定义在 `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/robot_parameters.hpp`：

- `HG_CMD_TOPIC = "rt/lowcmd"`。
- `HG_STATE_TOPIC = "rt/lowstate"`。
- `G1_NUM_MOTOR = 29`。
- `MotorCommand` 的 `q_target`、`dq_target`、`kp`、`kd`、`tau_ff` 都是 29 维。

## 2. MotionBricks 产物和边界

### 2.1 `control_signals`

`control_signals` 是 MotionBricks 生成的条件输入，不是机器人底层命令。来源包括：

- `motionbricks/motionbricks/motion_backbone/demo/controllers.py` 的 `WASD_controller.generate_control_signals(...)` 和 `random_controller.generate_control_signals(...)`。
- 常见字段包括 `movement_direction`、`facing_direction`、`mode`、`allowed_pred_num_tokens`。
- `motionbricks/scripts/interactive_demo_g1.py` 会按模式追加 `context_mujoco_qpos` 或 `context_motion_features`。

这些字段给 `full_navigation_agent.generate_new_frames(...)` 使用，用来约束下一段生成 motion。

### 2.2 `model_features`

`model_features` 是 MotionBricks 内部 motion representation：

- 缓存在 `motionbricks/motionbricks/motion_backbone/demo/full_agent.py` 的 `self.frames['model_features']`。
- 初始值来自 clip holder 的 `motion_feature`。
- 新帧由 inferencer 预测后写回 `self.frames['model_features']`。
- 表示细节见 `motionbricks/docs/motion_representation.md`，其中 G1Skeleton34 有 34 个 joint，MuJoCo hinge joints 为 29 个。

`model_features` 可以转换到 `mujoco_qpos`，但它本身不是 `LowCmd`，也不是 C++ `MotionSequence`。

### 2.3 `mujoco_qpos` `[root 7 + 29]`

`mujoco_qpos` 是 MotionBricks 面向 MuJoCo/G1 的 qpos 表示：

- 维度为 36。
- 前 7 维是 root translation + root quaternion。
- 后 29 维是 G1 MuJoCo joints。
- 转换实现是 `motionbricks/motionbricks/helper/mujoco_helper.py` 的 `mujoco_qpos_converter.convert_motion_features_to_mujoco_qpos(...)` 和 `convert_mujoco_qpos_to_motion_transforms(...)`。

坐标/姿态约定不能随意混用：

- `motionbricks/docs/motion_representation.md` 记录 MotionBricks motion 表示和 MuJoCo 表示之间存在坐标轴差异。
- root quaternion 在文档中按 `(w, x, y, z)` 记录；消费端要确认自己的图形/物理引擎是否需要转换顺序。

### 2.4 `MotionSequence`

`MotionSequence` 是 C++ 部署/stream/planner 的 motion 容器，不是 Python demo 里的帧缓存：

- 定义在 `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/motion_data_reader.hpp`。
- planner 输出在 `LocalMotionPlannerBase::ResampleGeneratedSequence50Hz()` 中从 30Hz planner qpos 重采样为 50Hz `MotionSequence`。
- `G1Deploy::CurrentFrameAdvancement()` 使用 planner 产物更新 `current_motion_`，再由 control loop 使用当前 motion frame 组装 observation。

### 2.5 边界结论

MotionBricks 当前不是直接 DDS 控制模块：

- Python demo 把 `qpos` 写入 MuJoCo `mj_data.qpos` 并渲染，不发布 `LowCmd_`。
- raylib/Three.js viewer 消费 MotionBricks 输出用于可视化，不参与 `LowCommandWriter()`。
- 真实 G1 控制必须经过 deploy 栈的 observation、policy action、`MotorCommand` 和 `LowCmd_` 路径，或者通过一个明确适配层接入 `MotionSequence`/ZMQ/token 合约。

因此不能把 `mujoco_qpos`、`model_features` 或 viewer 里的 retarget frame 直接当成硬件命令。

## 3. 不可随便改的硬接口

以下接口是后续实现的边界。任何改动都应有兼容性说明、回放验证和真实硬件安全审查。

### 3.1 `LowCmd_` / DDS 输出

来源：

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/robot_parameters.hpp`

约束：

- `LowCommandWriter()` 以 500Hz 写 `LowCmd_`。
- DDS topic 是 `rt/lowcmd`。
- 每个 motor command 的 `q`、`dq`、`kp`、`kd`、`tau` 字段和 CRC 都属于硬件协议边界。
- viewer 或 motion 生成脚本不应绕过这一层直接控制真实机器人。

### 3.2 G1 29 DOF

来源：

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/robot_parameters.hpp`
- `motionbricks/docs/motion_representation.md`
- `motionbricks/motionbricks/helper/mujoco_helper.py`

约束：

- `G1_NUM_MOTOR = 29`。
- policy action、`MotorCommand`、关节 gains、action scale 都按 29 维定义。
- MotionBricks `mujoco_qpos` 是 `[root 7 + 29]`，合计 36 维。
- 人体 skeleton、SMPL-X 或其它 humanoid 的自由度不能直接塞进 G1 29 DOF 接口。

### 3.3 Joint maps

来源：

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp`

关键映射：

- `isaaclab_to_mujoco`
- `mujoco_to_isaaclab`
- upper/lower body maps
- wrist maps

这些映射连接 IsaacLab 训练顺序、MuJoCo qpos 顺序和硬件/策略顺序。任何重排都会影响 observation、policy action、planner context 和 LowCmd 目标关节。

### 3.4 Planner tensor names

来源：

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner_tensorrt.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner_onnx.hpp`

TensorRT planner 当前硬编码的输入/输出名包括：

- 输入：`context_mujoco_qpos`
- 输入：`mode`
- 输入：`target_vel`
- 输入：`height`
- 输入：`has_specific_target`
- 输入：`specific_target_positions`
- 输入：`specific_target_headings`
- 输入：`allowed_pred_num_tokens`
- 输入：`movement_direction`
- 输入：`facing_direction`
- 输入：`random_seed`
- 输出：`mujoco_qpos`
- 输出：`num_pred_frames`

版本差异也属于接口：

- version 0 期望 6 个输入。
- version 1/2 期望 11 个输入。
- 输出 buffer 按 `64 * 36` qpos 和 1 个 `num_pred_frames` 配置。

### 3.5 Policy / encoder tensor names

来源：

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/control_policy.hpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/encoder.hpp`

约束：

- policy 输入 tensor 名是 `obs_dict`。
- policy 输出 tensor 名是 `action`。
- encoder 输入 tensor 名是 `obs_dict`。
- encoder 输出 tensor 名是 `encoded_tokens`。
- observation 配置中的字段名和总维度必须与模型一致。

### 3.6 频率关系

来源：

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner.hpp`
- `motionbricks/motionbricks/motion_backbone/demo/controllers.py`

当前频率边界：

- Input thread：100Hz。
- Control thread：50Hz。
- Planner thread：10Hz。
- Low command writer：500Hz。
- C++ planner 模型输出按 30Hz 理解，再由 `ResampleGeneratedSequence50Hz()` 重采样到 50Hz `MotionSequence`。
- MotionBricks Python demo 默认 `_FPS = 30`，`base_controller._CONTROLLER_DT = 8 / _FPS`。

改 viewer 或 retarget 时，不要把 30Hz 生成频率、50Hz 控制频率和 500Hz LowCmd 发布频率混成一个时间轴。

## 4. 人体化三层路线

### A. 视觉级高精蒙皮预览

目标：把 MotionBricks 生成的 G1 motion 以更像人的形态展示出来。

允许做的事：

- 消费 `mujoco_qpos` 或转换后的 joint transforms。
- 做 G1 skeleton 到 human-like skeleton 的 retarget。
- 用 GLB/SMPL-X/MetaHuman/其它高精度 rigged human 资产做蒙皮预览。
- 用 Three.js、Unreal 或其它成熟 viewer 做高清显示。

已有基础：

- `motionbricks/scripts/render_human_skeleton_demo.py`：把 MotionBricks/G1 joint 方向投影到固定 human-like limb length，适合作为快速 skeleton/capsule preview。
- `motionbricks/scripts/export_skinned_human_viewer.py`：导出 Three.js browser viewer，加载 rigged humanoid GLB，并在运行时 retarget 到 GLB bones。
- `motionbricks/scripts/raylib_motionbricks_viewer.py`：raylib 本地窗口消费 MotionBricks 输出并播放 human preview skeleton。

边界：

- 这一层不表示真实人体控制已经实现。
- mesh/skin 换得更真实，不等于物理约束、控制策略或硬件接口已经适配。

### B. 物理仿真人体 / 更灵活 humanoid

目标：在物理仿真中支持人体或更灵活 humanoid 的动力学、关节限制、碰撞和控制。

需要新增或明确的内容：

- 新的 human/humanoid skeleton 定义，而不是复用 G1 29 DOF。
- 新 qpos / motion representation 的 root、joint、quaternion、坐标轴约定。
- 仿真模型的 body、joint limit、actuator、contact、mass/inertia。
- 从 MotionBricks 输出到该物理模型的 retarget 或生成路径。
- 如果要闭环控制，需要对应 policy/WBC 或重新训练后的控制器。

边界：

- 这层不是当前 raylib/Three.js viewer 能自动完成的。
- 仅把 G1 motion retarget 到人体骨架，仍然可能违反人体关节限制或物理可行性。

### C. 真实机器人控制 / 类人机器人

目标：把 motion/intent 安全地接入真实硬件。

G1 路线：

- 必须保留 `LowCmd_`、G1 29 DOF、joint maps、policy observation/action 合约。
- MotionBricks 或外部 planner 只能通过明确的 `MotionSequence`、ZMQ stream、token 或 planner tensor 入口接入。
- 最终动作仍由 `PolicyEngine`/控制栈生成 `MotorCommand`，再由 `LowCommandWriter()` 发布。

非 G1 类人机器人路线：

- 需要新的机器人参数、joint map、action scale、gains、observation 配置和安全限制。
- 需要重新训练或微调 planner/policy/WBC。
- 需要单独的仿真验证、回放验证和硬件保护策略。

边界：

- 当前仓库不能被描述为已经实现“真实人体控制”。
- 当前高精人体 viewer 只能作为 embodiment 可视化和 retarget 研究基础。

## 5. 当前 raylib / Three.js 脚本定位

### 5.1 raylib

`motionbricks/scripts/raylib_motionbricks_viewer.py` 当前应定位为 MotionBricks 输出消费端：

- 它预计算短段 MotionBricks rollout。
- 使用 `full_agent.get_next_frame()` 得到 qpos。
- 用 `convert_mujoco_qpos_to_motion_transforms(...)` 转成 transforms。
- 再 retarget 到 human preview skeleton，并用 raylib 播放。

raylib 不应被提升为 canonical 控制层：

- 不发布 DDS。
- 不写 `LowCmd_`。
- 不决定 G1 joint map 或 policy tensor 合约。
- 不承担高清蒙皮主 viewer 的长期职责。

短期可以在 raylib 中完善平台边界、窗口交互、调试叠加和离线预览，但它应继续保持“消费 MotionBricks 输出”的角色。

### 5.2 Three.js / Unreal 等高清 viewer

`motionbricks/scripts/export_skinned_human_viewer.py` 更接近高精蒙皮预览路线：

- 它导出 `animation.json`。
- 加载 rigged humanoid `avatar.glb`。
- 在 Three.js 中做运行时 retarget 和 playback。

后续高清 viewer 建议优先走 Three.js/Unreal 等资产生态更完整的路径：

- 更容易接入 GLB、SMPL-X、MetaHuman 或 DCC 导出的 rig。
- 更适合做材质、灯光、相机、动画检查和产品级展示。
- raylib 保持轻量 native/debug viewer 即可。

## 6. 下一步执行清单

### 短期：raylib 平台边界

- 明确 `raylib_motionbricks_viewer.py` 只消费 MotionBricks 输出，不触碰 DDS、`LowCmd_`、policy tensor 合约。
- 给 raylib viewer 的输入/输出写清楚：输入是 `mujoco_qpos` 或 transforms，输出是屏幕预览。
- 固化坐标轴、quaternion 顺序、帧率插值和 skeleton bone length 的假设。
- 避免把 raylib retarget 代码扩展成真实控制逻辑。

### 中期：完善 skinned human retarget

- 以 `export_skinned_human_viewer.py` 的 Three.js/GLB 路径为主线，明确 humanoid rig 的骨骼命名和 bone map。
- 把 G1 joints 到 human bones 的 retarget 规则拆成可配置映射。
- 增加对 root motion、foot contact、肩/髋/脊柱朝向的校验。
- 建立 `animation.json` 或等价中间格式的 schema，记录坐标系、帧率、joint names、quaternion convention。
- 使用少量固定 motion clips 做回归检查，确保 viewer 改动不会破坏现有 MotionBricks 输出消费。

### 长期：HumanSkeleton / SMPL-X / motion rep / retrain

- 新增 `HumanSkeleton` 或 SMPL-X 风格 skeleton 定义，独立于 G1 29 DOF。
- 设计人体 motion representation，明确 root、body joints、hands、face/shape 参数是否进入模型。
- 扩展数据预处理，把 AMASS/SMPL-X/自有人体动作数据转成新 representation。
- 如需物理仿真，新增人体或 humanoid MuJoCo/Isaac/Unreal Physics 模型，并定义 actuator/control 语义。
- 如需真实机器人控制，为目标机器人新增 `robot_parameters`、joint maps、observation config、action scale、gains 和安全限制。
- 对 planner/policy/WBC 做重新训练或微调，不能假设 G1 policy 可直接控制人体 skeleton 或其它机器人。

## 7. 实施守则

- 文档和 viewer 可以消费 MotionBricks 输出；真实硬件命令只能走 deploy 栈定义的安全接口。
- 改视觉层时，不改 `LowCmd_`、G1 29 DOF、joint maps、planner tensor names、policy tensor names。
- 改 motion representation 时，必须同步说明与 `model_features`、`mujoco_qpos`、`MotionSequence` 的转换关系。
- 改频率或插值时，必须说明 30Hz generation、50Hz control、500Hz command writer 之间的关系。
- 对外描述时只说“人体化预览”“skinned human viewer”“retarget research path”，不要写成“已经实现真实人体控制”。
