# MotionBricks -> UE Bridge 最小数据规格

本文档定义 Step 1 数据契约：MotionBricks 侧只导出可校验的 qpos 与 34-bone 全局 transform 样本，供 UE 侧最小 debug playback 验证。本文不声明 UE 移植完成，也不改变 MotionBricks 模型逻辑。

## 文件

- 主序列：`out/ue_bridge_test/motionbricks_g1_sequence.json`
- 轴向测试：`out/ue_bridge_test/axis_tests.json`
- 导出脚本：`scripts/export_ue_bridge_test_sequence.py`

## 推荐 JSON 结构

主序列和 `axis_tests.json` 都推荐导出同一套 runtime-readable 契约字段。`schema` object 保留为丰富 metadata 和人类审阅入口；UE runtime 的 canonical skeleton 读取位置是 `skeleton.bone_names` 和 `skeleton.parents`。顶层 `bone_names` / `parents` 仅作为兼容旧检查脚本的别名，内容必须由 `skeleton` 派生，不能手写成另一份。

推荐结构：

```json
{
  "schema_version": "motionbricks_ue_bridge.v1",
  "source_hz": 30.0,
  "sample_rate_hz": 30.0,
  "root_quat_order": "wxyz",
  "qpos_length": 36,
  "joint_order": ["root_x_m", "..."],
  "dof_names": ["left_hip_pitch_joint", "..."],
  "schema_hash": "<sha256>",
  "schema_header": {
    "schema_version": "motionbricks_ue_bridge.v1",
    "source_hz": 30.0,
    "sample_rate_hz": 30.0,
    "root_quat_order": "wxyz",
    "qpos_length": 36,
    "joint_order": ["root_x_m", "..."],
    "dof_names": ["left_hip_pitch_joint", "..."],
    "schema_hash": "<sha256>"
  },
  "skeleton": {
    "bone_count": 34,
    "bone_names": ["pelvis_skel", "..."],
    "parents": [-1, "..."]
  },
  "bone_count": 34,
  "bone_names": ["pelvis_skel", "..."],
  "parents": [-1, "..."],
  "schema": {
    "schema_version": "motionbricks_ue_bridge.v1",
    "source_hz": 30.0,
    "bone_names": ["pelvis_skel", "..."],
    "parents": [-1, "..."]
  },
  "frames": []
}
```

`axis_tests.json` 使用相同 header / skeleton / schema 布局，但把 `frames` 换成 `tests`。UE parser 应优先读取 `skeleton.bone_names` / `skeleton.parents`，再用顶层别名做兼容 fallback。

## qpos36 layout

`qpos` 长度必须为 36，来自 G1 29-DOF MuJoCo free-root 表示：

| index | 字段 | 单位 | 说明 |
| --- | --- | --- | --- |
| 0..2 | `root_xyz_m` | meters | MuJoCo world root position，顺序 `[x, y, z]` |
| 3..6 | `root_quat_wxyz` | unit quat | MuJoCo world root rotation，顺序必须是 `[w, x, y, z]` |
| 7..35 | `joint_dofs` | radians | 29 个 hinge DOF，顺序见 runtime header 的 `dof_names` |

`root_quat_wxyz` 必须归一化，容差建议 `abs(norm - 1) <= 1e-3`。UE 侧不要假设输入是 xyzw。

默认 29 hinge DOF 顺序是 MuJoCo XML / qpos 顺序：

1. `left_hip_pitch_joint`
2. `left_hip_roll_joint`
3. `left_hip_yaw_joint`
4. `left_knee_joint`
5. `left_ankle_pitch_joint`
6. `left_ankle_roll_joint`
7. `right_hip_pitch_joint`
8. `right_hip_roll_joint`
9. `right_hip_yaw_joint`
10. `right_knee_joint`
11. `right_ankle_pitch_joint`
12. `right_ankle_roll_joint`
13. `waist_yaw_joint`
14. `waist_roll_joint`
15. `waist_pitch_joint`
16. `left_shoulder_pitch_joint`
17. `left_shoulder_roll_joint`
18. `left_shoulder_yaw_joint`
19. `left_elbow_joint`
20. `left_wrist_roll_joint`
21. `left_wrist_pitch_joint`
22. `left_wrist_yaw_joint`
23. `right_shoulder_pitch_joint`
24. `right_shoulder_roll_joint`
25. `right_shoulder_yaw_joint`
26. `right_elbow_joint`
27. `right_wrist_roll_joint`
28. `right_wrist_pitch_joint`
29. `right_wrist_yaw_joint`

实际消费时必须以 JSON 顶层或 `schema_header` 的 `dof_names` 和 `joint_order` 为准，并校验 `schema_hash`。

## 34-bone transform array

每帧导出 `bone_global_positions_m`，长度必须等于 `skeleton.bone_names` 的长度，当前为 34。数组顺序与 `skeleton.bone_names` 一一对应，`skeleton.parents[i]` 是第 `i` 个 bone 的父节点 index；root 的 parent 为 `-1`。

当前导出字段：

- `skeleton.bone_names`: 34 个 MotionBricks G1 skeleton bone 名称，UE runtime canonical 读取位置。
- `skeleton.parents`: 34 个 parent index，UE runtime canonical 读取位置。
- `bone_names` / `parents`: 顶层兼容别名，必须与 `skeleton` 内容一致。
- `bone_global_positions_m`: shape `[34][3]`，单位 meters，坐标为 MuJoCo basis。
- `bone_global_rotations_wxyz`: shape `[34][4]`，可选但当前脚本会导出；顺序为 `[w, x, y, z]`，坐标为 MuJoCo basis。

如果后续只导出 positions 而不导出 rotations，UE 侧只能做 joint cloud / bone segment debug playback，不能用这些数据直接驱动完整骨骼姿态。旋转缺失时，UE 侧必须从 qpos 自己做 FK 或显式标记为 position-only preview。

当前 34-bone 名称顺序：

1. `pelvis_skel`
2. `left_hip_pitch_skel`
3. `left_hip_roll_skel`
4. `left_hip_yaw_skel`
5. `left_knee_skel`
6. `left_ankle_pitch_skel`
7. `left_ankle_roll_skel`
8. `left_toe_base`
9. `right_hip_pitch_skel`
10. `right_hip_roll_skel`
11. `right_hip_yaw_skel`
12. `right_knee_skel`
13. `right_ankle_pitch_skel`
14. `right_ankle_roll_skel`
15. `right_toe_base`
16. `waist_yaw_skel`
17. `waist_roll_skel`
18. `waist_pitch_skel`
19. `left_shoulder_pitch_skel`
20. `left_shoulder_roll_skel`
21. `left_shoulder_yaw_skel`
22. `left_elbow_skel`
23. `left_wrist_roll_skel`
24. `left_wrist_pitch_skel`
25. `left_wrist_yaw_skel`
26. `left_hand_roll_skel`
27. `right_shoulder_pitch_skel`
28. `right_shoulder_roll_skel`
29. `right_shoulder_yaw_skel`
30. `right_elbow_skel`
31. `right_wrist_roll_skel`
32. `right_wrist_pitch_skel`
33. `right_wrist_yaw_skel`
34. `right_hand_roll_skel`

## 坐标候选

MotionBricks qpos 和本桥接导出的 bone transforms 使用 MuJoCo world basis：

- MuJoCo: `X forward / Y left / Z up`
- UE: `X forward / Y right / Z up`

位置候选转换：

```text
UE_cm = (x, -y, z) * 100
```

四元数必须做 basis change，不能只做 `wxyz` 到 `xyzw` 重排。对任意 MuJoCo rotation matrix `R_mj`，候选 UE basis change 可写成：

```text
B = diag(1, -1, 1)
R_ue = B * R_mj * B
```

然后再把 `R_ue` 转为 UE 需要的 quaternion 存储顺序。因为 `B` 包含镜像，直接对 quaternion 分量换符号容易出错；UE 侧应以矩阵或经过验证的 basis-change 函数实现，并用本规格的 axis tests 校验。

## Schema header

每个 JSON 必须包含 UE parser 可直接读取的 runtime header。当前导出在顶层和 `schema_header` 中都提供同一组字段：

- `schema_version`: 字符串，例如 `motionbricks_ue_bridge.v1`。
- `source_hz`: MotionBricks 原始采样率，单位 Hz。
- `sample_rate_hz`: `source_hz` 的兼容别名。
- `root_quat_order`: 必须是 `wxyz`。
- `qpos_length`: 固定为 36。
- `joint_order`: qpos 36 槽位顺序，长度 36。
- `dof_names`: 29 hinge DOF 名称数组。
- `schema_hash`: 对除 `schema_hash` 之外的 `schema` 关键字段做稳定 JSON 后的 SHA-256，用于 UE 侧快速拒绝不匹配数据。

每个 JSON 还必须包含 `skeleton` object：

- `skeleton.bone_count`: 当前固定为 34。
- `skeleton.bone_names`: 34-bone 名称数组，UE runtime canonical 读取位置。
- `skeleton.parents`: parent index 数组，长度必须等于 `skeleton.bone_names`。

`schema` object 仍必须保留，用于 richer metadata、debug 与 hash 复算，建议至少包含：

- `unit`: 单位说明，至少包含 `root_xyz`, `bone_positions`, `angles`。
- `coordinate_system`: 明确 source basis 与 UE candidate。
- `bone_position_space`: 当前为 `mujoco_world_meters`。
- `bone_rotation_space`: 当前为 `mujoco_world_wxyz`。
- `has_bone_global_rotations`: 当前为 `true`。
- `frame_stride`: 主序列抽帧步长。

## 必须通过的测试

UE 侧读取 `axis_tests.json` 后至少验证：

1. `neutral`: qpos 长度为 36，root quat 为 wxyz 且归一化，bone 数为 34。
2. `mujoco_x_plus_1m`: root 的 MuJoCo `+X` 位移在候选 UE 转换后应表现为 UE `+X` 方向 `+100cm`。
3. `mujoco_y_plus_1m`: root 的 MuJoCo `+Y` 位移在候选 UE 转换后应表现为 UE `-Y` 方向 `-100cm`。
4. `root_yaw_plus_90deg`: root 绕 MuJoCo `+Z` yaw `+90deg` 后，UE 侧必须通过 basis-changed rotation 匹配；不能只重排 quaternion 分量。
5. `single_joint`: 单个 hinge DOF 改变时，UE debug playback 应只看到对应运动链发生预期变化；其他 root/兄弟链不应整体漂移。
6. `same_frame_compare`: 使用主序列第 0 帧 qpos 在 UE 侧重算 FK，结果应与同帧 `bone_global_positions_m` 在坐标转换后逐 bone 对齐；建议位置误差阈值先设为 `1e-3m` 量级，再按 UE debug 可视化调整。

只有这些测试通过，才进入下一步 UE playback 或 retarget 工作。
