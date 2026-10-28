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

## 5. 2026-10-29 修复后重跑：A 的 64 帧评估中断

第 1～4 节保留 2026-10-28 的原始测试停止记录。本次执行的是 [handoff 当前任务](../../handoff/tasks/2026-10-28_v6_rerun.md)，源码固定为 `e1793dc8429dfe343db180937f4923ada955c08e`，包含 `f98e04d` 的测试输入精度修复。执行前将交接状态标为 running，以 `2257ed9` 推送；A100 使用独立、干净的固定源码 checkout。

**结果：90 项 CPU 测试全部通过，FK 检查误差为 0；A 在第一个检查点的 64 帧基线评估处发生 16/64 帧维度不匹配，执行停止。B 的两个训练臂均未开始，新增训练更新为 0。**

### 测试门与检查点

使用第 1 节同样的九个测试文件，输出改为 `runs/tmt_v6_rerun_coord/tests.log` 和 `tests.xml`，CPU 环境及线程数保持不变。输出末行：

```text
90 passed, 15 warnings in 14.47s
```

0 失败、0 错误、0 跳过。`test_foot_lock.py` 的五项测试均实际运行，包含两骨骼测试末尾的直腿分支；`foot_lock.py` 与上次停止时完全一致。本次没有修改模型、IK、指标或测试代码。

在 CPU 上逐一加载只读检查点，确认 v1t/v4t 为 update 1500，v2/v3/v5 的 `latest.pt` 均为 **update 3000**。五个文件的哈希与第 4 节一致；另对 v1～v5 共 20 个既有检查点保存前后哈希，差异为 0。结果见[检查点表](../../examples/results/terrain_controller_v6_a100_20261027/continuation_f98e04d/checkpoint_hashes.tsv)和[输入快照](../../examples/results/terrain_controller_v6_a100_20261027/continuation_f98e04d/input_proof.json)。

### A 的命令、资源与停止点

扫描全部 A100 的 index、UUID、显存、利用率和计算 PID 后，选定空闲的物理 **GPU 3** 并按 UUID 固定分配。执行时间为 **2026-10-29 02:10:10.394～02:10:25.211**（Asia/Singapore），外部壁钟上限 1800 秒。

原评估脚本接收的参数如下；测量包装器只补充批次摘要、异常捕获及耗时记录，未替换模型计算或指标实现：

```bash
$PY scripts/evaluate_foot_lock.py --dataset runs/bones_env_v1 --output runs/foot_lock_eval_v6 \
  --checkpoint v1t=runs/tmt_pilot_v1/phase0_teacher.pt \
  --checkpoint v2=runs/tmt_pilot_v2/latest.pt \
  --checkpoint v3=runs/tmt_pilot_v3/latest.pt \
  --checkpoint v4t=runs/tmt_pilot_v4/phase0_teacher.pt \
  --checkpoint v5=runs/tmt_pilot_v5/latest.pt
```

原程序首先通过 FK 核对，**`fk_max_error_m=0.0`**。随后完成 v1 教师检查点的 16 帧 raw、display、feedback 三种模式。进入同一检查点的 64 帧 `base = evaluate(...)` 时退出码为 1，尚未进入该长度的模式循环：

```text
File "${TAK_ROOT}/motionbricks/controller/training.py", line 138, in motion_metrics
    pelvis = (generated_joints[:, :, root, 1] - source_joints[:, :, root, 1]).abs().mean(-1)
RuntimeError: The size of tensor a (16) must match the size of tensor b (64) at non-singleton dimension 1
```

完整[评估日志](../../examples/results/terrain_controller_v6_a100_20261027/continuation_f98e04d/eval_a.log)和[traceback](../../examples/results/terrain_controller_v6_a100_20261027/continuation_f98e04d/failure_traceback.txt)保留在结果目录。原程序只在全部检查点结束后写输出，因此 `runs/foot_lock_eval_v6` 没有生成；异常捕获保留了当时已完成的内存报告，并据此导出下面的部分表格，没有重新跑模型。

### 原因定位

[ClosedLoop.step 第 169 行](../../motionbricks/controller/closed_loop.py#L169)每步删除最早一帧，再追加新帧，历史长度始终为输入的 16 帧。[rollout 第 286 行](../../motionbricks/controller/closed_loop.py#L286)返回的是这份历史缓冲区。

[evaluate 第 219 行](../../motionbricks/controller/training.py#L219)使用 `result["history"][:, -horizon:]` 作为生成序列；当 `horizon=64` 时，切片仍只有 16 帧。[motion_metrics 第 129 行](../../motionbricks/controller/training.py#L129)取出 64 帧源关节，随后逐帧相减触发上述错误。当前通过的 rollout 单元测试只检查 4 帧，没有覆盖 64 帧与固定历史缓冲区的区别。

本次停止于原评估流程的运行错误，没有缩短 horizon、跳过基线、补帧或更换生成序列来继续。后续需要修复评估函数如何取得完整生成序列，并覆盖长于历史缓冲区的评估长度。

### 验证批次身份

16 帧 batch 为 256 个窗口，validation 可用窗口共 9746；64 帧 batch 同为 256，可用窗口共 6782。两个摘要算法分别列出，避免把不同计算方法的摘要直接比较：

| 长度 | 评估脚本摘要：四个张量字节串 | 历史训练器摘要格式：全部字段、名称、类型与形状 |
| --- | --- | --- |
| 16 帧 | `d7970772737eeb3a9c43ae37997dcb5016eb57da797d3f31592e8104e50ffc72` | `29c47a5525e2e0f2d97d715daee5e933387b79fb62b55363d57f009788c486da` |
| 64 帧 | `6279cb57a0cbc466add4424be8b5ba291c2e4209b7a060f544e609efd266f4e8` | `ee04aebdd00af4aaab1b73d19dc2f2f9959bd9285c36efe73dbcc2fcf3bd908c` |

16 帧的训练器格式摘要与 v1～v5 固定验证批次一致。v1t、16 帧 raw 的 **`raw_matches_trainer_rollout_cm=0.0`**；其他检查点尚未进入评估，没有对应值。

### 已完成的部分结果：v1 教师，16 帧

| 模式 | 根漂移 P50 cm | 进度比 P50 / P10 | 穿透超 2cm 帧 % | 源接触脚速 m/s | 几何着地脚速 m/s | 锁激活比例 % | IK 残差 P95 / 最大 m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 1.2491 | 0.9910 / 0.8900 | 11.1084 | 0.17896 | 0.17008 | — | — |
| display | 1.2491 | 0.9910 / 0.8900 | 8.6914 | 0.03491 | 0.03249 | 34.4360 | 0.03051 / 0.23526 |
| feedback | 1.1426 | 0.9902 / 0.8980 | 9.0332 | 0.02593 | 0.02678 | 35.8154 | 0.05499 / 0.21731 |

[table.tsv](../../examples/results/terrain_controller_v6_a100_20261027/continuation_f98e04d/table.tsv)保留这三行的全部数值，[by_group.tsv](../../examples/results/terrain_controller_v6_a100_20261027/continuation_f98e04d/by_group.tsv)保留 21 行分组结果。源接触掩码为空的 prone/quadruped 脚速等 15 个统计值没有定义：JSON 写 `null`，TSV 留空，并在报告中列出具体字段位置，没有写成 0。

v2 尚未评估，因此计划 §3 的 **v2+lock 候选判定未执行**。单角色加锁耗时在脚本末尾，尚未测量；六个 v2 窗口的接触图与足底曲线也未生成。v6a/v6b、各自双卡预检、125 次更新项统计、bootstrap 占比以及最终联合表全部未执行。

### 独立消耗与文件

| 阶段 | 状态 | GPU 数 | 壁钟秒 | GPU-hours | 新增训练更新 |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU 测试 | 90 项通过 | 0 | 14.47 | 0 | 0 |
| A | 64 帧基线评估中断 | 1 | 14.817064 | 0.004115851 | 0 |
| v6a 预检 / 正式训练 | 未启动 | 0 | 0 | 0 | 0 |
| v6b 预检 / 正式训练 | 未启动 | 0 | 0 | 0 | 0 |
| B 后脚锁复评 | 未启动 | 0 | 0 | 0 | 0 |

A 的峰值已分配显存为 **1,902,115,328 字节**；GPU-hours 按“进程壁钟 × 分配卡数 ÷ 3600”计算。未触及 30 分钟上限，没有训练、恢复、额外重跑或部署。A 的进程已退出，B 输出目录均不存在。

原始测试、输入快照、批次摘要和异常保留在 A100 `runs/tmt_v6_rerun_coord/`；通过私有 Drive 取回的九个文件逐个核对 SHA-256。公开[修复后停止结果](../../examples/results/terrain_controller_v6_a100_20261027/continuation_f98e04d/README.md)只含日志、聚合 JSON/TSV 和检查点哈希，旧的测试失败产物原样保留。
