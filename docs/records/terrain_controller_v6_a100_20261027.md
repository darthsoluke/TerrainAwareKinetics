# 第 7 步 v6：测试停止记录

执行日期：2026-10-28（Asia/Singapore）。[v6 计划](terrain_controller_v6_plan_20261027.md)中的 A、B 均已获用户批准；本次源码为 `eafb71b5df2512d667fd4f6a6b4cc4db1a811b4d`。

**结果：89 项通过、1 项失败、0 跳过。失败发生在脚锁测试的旋转正交性断言；按用户规定停止，A、v6a、v6b 均未启动。新增训练更新和 GPU-hours 均为 0。**

## 1. 环境与命令

在 A100 checkout 执行 `git pull --ff-only origin main`，更新到上述提交。复用项目独立 Python 和 Kimodo 环境，测试固定在 CPU：

| 项目 | 实际值 |
| --- | --- |
| Python | 3.11.17 |
| PyTorch | 2.5.1+cu124 |
| CUDA runtime 版本 | 12.4 |
| 测试设备 | CPU，`CUDA_VISIBLE_DEVICES=''` |
| OMP / OpenBLAS / MKL 线程数 | 各 1 |
| 测试退出码 | 1 |
| pytest 输出末行 | `1 failed, 89 passed, 15 warnings in 11.56s` |

完整测试命令：

```bash
CUDA_VISIBLE_DEVICES='' "$PY" -m pytest \
  tests/test_bones_env_dataset.py tests/test_terrain_controller.py \
  tests/test_skeleton_ray_sampler.py tests/test_bones_terrain_fit.py \
  tests/test_bones_terrain_scene.py tests/test_bones_terrain_screen.py \
  tests/test_bones_terrain_survey.py tests/test_train_terrain_controller.py \
  tests/test_foot_lock.py -q --junitxml=runs/tmt_v6_coord/tests.xml \
  > runs/tmt_v6_coord/tests.log 2>&1
```

此前八个文件共 85 项全部通过；新增 `test_foot_lock.py` 的 5 项全部执行，4 项通过、1 项失败：

| 脚锁测试 | 结果 |
| --- | --- |
| `test_two_bone_ik_reaches_reachable_targets_and_keeps_bone_lengths` | **失败** |
| `test_foot_lock_fk_matches_the_representation` | 通过 |
| `test_foot_lock_pins_a_sliding_stance_foot_and_releases` | 通过 |
| `test_rollout_frames_modes_and_progress_ratio` | 通过 |
| `test_contact_rate_anchor_is_phase_free_and_reaches_the_contact_head` | 通过 |

## 2. 精确失败位置

失败位于 [tests/test_foot_lock.py 第 57 行](../../tests/test_foot_lock.py#L57)：

```python
assert torch.allclose(
    new_a @ new_a.transpose(-1, -2),
    torch.eye(3, dtype=torch.float64).expand(n, 3, 3),
    atol=1e-9,
)
```

输出为 `AssertionError: assert False`。该断言检查输出髋关节旋转矩阵乘以自身转置后是否接近单位矩阵。前面的 FK 计算与返回位置一致性断言、可达目标误差 `< 1e-6` 断言均已通过；后面的直腿退化情形因本断言中止而未执行。

静态检查中可见，测试的 `_random_rotations` 没有指定生成 dtype，调用方在旋转矩阵生成后才 `.double()`，随后使用 `atol=1e-9` 检查正交性。日志没有记录最大正交误差，具体是输入精度还是 IK 计算引入误差，仍需定位。本次保持源代码和断言不变，没有绕过测试启动实验。

完整 pytest 失败展开、断言、警告和末行见 [tests.log](../../examples/results/terrain_controller_v6_a100_20261027/tests.log)。公开副本仅将私人环境目录替换为 `${PFNN_PROJECT}`；原始日志和 JUnit 保留在 A100 的 `runs/tmt_v6_coord/`。

## 3. 停止位置与账本

| 阶段 | 状态 | 新增训练更新 | GPU-hours |
| --- | --- | ---: | ---: |
| CPU 测试 | 完成，触发停止条件 | 0 | 0 |
| A：五检查点脚锁评估 | 未启动 | 0 | 0 |
| B：v6a 双卡两步预检 | 未启动 | 0 | 0 |
| B：v6a 正式训练 | 未启动 | 0 | 0 |
| B：v6b 双卡两步预检 | 未启动 | 0 | 0 |
| B：v6b 正式训练 | 未启动 | 0 | 0 |
| B 后的脚锁复评 | 未启动 | 0 | 0 |

`runs/foot_lock_eval_v6`、`runs/tmt_v6a`、`runs/tmt_v6b` 均不存在，没有创建等待队列或训练进程。CPU 单元测试的合成输入与梯度检查单列为测试，不计作正式训练数据曝光。

A 的实数据 `fk_max_error_m`、16 帧批次摘要、`raw_matches_trainer_rollout_cm`、单角色加锁耗时、结果表和图片尚未生成；脚锁的 FK 单元测试通过值不替代 A 的实数据测量。B 的阶段统计、加权代价、bootstrap 占比和第 3000 次验收均未执行。

## 4. 检查点与发布

停止后仅计算计划引用的五个既有检查点文件的 SHA-256，与 v5 已发布记录一致；本轮没有加载它们执行 A 或 B。v2/v3/v5 的 update-3000 状态来自此前记录，本轮尚未执行 A 的加载核验。

| 检查点 | SHA-256 |
| --- | --- |
| v1 teacher | `0c75f4429789d619a5dfd30b86fde4fec3ff7a02ceeb76b63a15d068bd68045b` |
| v2 latest | `794d4ce660a4883b1b2d154be088b940c8995ec7d32a6fcac4d6aedf38474eba` |
| v3 latest | `fc4afd691fc8d7e1c2645de5185c80d0c75b1dbc68a677d413a5ba05c2bb7220` |
| v4 teacher | `5fd0023fa92935c648e027e567746c05c9ddcd55c9235328ade318862d274423` |
| v5 latest | `594882aee0927fdf7a76e9af8a9ef9019a7140cc4e16a0d9a5e3d3b707570a15` |

[精简结果目录](../../examples/results/terrain_controller_v6_a100_20261027/README.md)包含测试日志、JUnit、机器可读状态、源码/检查点哈希及独立 GPU-hours 账本。本次只发布测试停止记录；没有权重、动作数据、新模型或部署。
