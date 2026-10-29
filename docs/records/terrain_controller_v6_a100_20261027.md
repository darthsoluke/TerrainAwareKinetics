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

## 6. 2026-10-29 rerun 2：A 完成，B 等待双卡

执行 [rerun 2 交接](../../handoff/tasks/2026-10-29_v6_rerun2.md)，实验源码固定为 `60fd119534e739c4f70bfc1fd82bcc21e85db810`，running 状态已在 `603303d` 推送。本节发布 A 完成时的结果快照；B 尚未进入预检，唯一队列等待两张无其他计算进程的 A100，交接保持 running。

**A 完成全部 30 行和 210 行分组结果，FK 误差、raw 与训练器 rollout 差异、旧 v1t16 行确定性差异全部为 0。v2 显示加锁和回灌加锁在两个长度下均未通过 0.03m/s 的源接触脚速门槛，其他四项候选条件通过。**

### 测试、输入身份与环境修复

九文件 CPU 测试已完成一次：`90 passed, 15 warnings in 12.63s`，0 失败、0 跳过；包含长于 16 帧历史缓冲区的 20 帧回归。v1t/v4t 的 checkpoint 为 update 1500，v2/v3/v5 为 update 3000，20 个旧 checkpoint 的前后 SHA-256 全部一致。数据清单 SHA-256 为 `67808b6813a42b676b58607b47bedc0bda9566819bee258f6ac4416c5a33c74d`。完整身份与源码文件摘要见 [report.json](../../examples/results/terrain_controller_v6_a100_20261027/continuation_60fd119/report.json)。

本轮首次启动在进入模型计算前发现稀疏检出缺少旧的 `continuation_f98e04d/table.tsv`，包装器退出；没有执行 FK、采样、模型帧或训练更新。按已获批准的环境修复范围，从同一固定提交补齐该参考目录，文件 SHA-256 为 `32e97e1c95c6ae08a5f7087bc6a8a1f5d4ed5c16565d24d280b0766525c9d76b`，与 Git blob 和已发布副本一致。源码未改变，失败日志和 2.172321s 分配时间保留；新队列只启动一次。

### A 的执行与确定性

A 使用交接指定的五个 checkpoint 和原评估参数，在一张空闲 A100 上于 08:21:15.412–08:22:50.626（Asia/Singapore）完成。进程壁钟 95.214735s，峰值已分配显存 1,933,306,368 字节，GPU-hours 为 0.026448538。CPU 图片渲染随后完成六个窗口、三种生成模式共 288 个诊断帧。

| 核对项 | 结果 |
| --- | --- |
| `fk_max_error_m` | 0.0m，门槛 0.001m |
| `raw_matches_trainer_rollout_cm` | 五检查点 × 两种长度共 10 项，全部 0.0cm |
| 旧 v1t16 三模式确定性 | 61 个数值逐项比较，最大绝对差 0.0；只排除运行耗时 `seconds` |
| 单角色加锁耗时 | P50 10.393651ms，P99 29.977724ms |
| 固定图片一致性 | 源关节与旧 v2 相同；raw 与旧 v2 保存数组最大误差 0.0m |

| 长度 | 脚本四张量 SHA-256 | 训练器格式 SHA-256 |
| --- | --- | --- |
| 16 | `d7970772737eeb3a9c43ae37997dcb5016eb57da797d3f31592e8104e50ffc72` | `29c47a5525e2e0f2d97d715daee5e933387b79fb62b55363d57f009788c486da` |
| 64 | `6279cb57a0cbc466add4424be8b5ba291c2e4209b7a060f544e609efd266f4e8` | `ee04aebdd00af4aaab1b73d19dc2f2f9959bd9285c36efe73dbcc2fcf3bd908c` |

每种长度均取 validation、seed 1 的首批 256 个窗口；16/64 帧可用窗口分别为 9746/6782。全部总体指标见 [table.tsv](../../examples/results/terrain_controller_v6_a100_20261027/continuation_60fd119/table.tsv)，七组结果见 [by_group.tsv](../../examples/results/terrain_controller_v6_a100_20261027/continuation_60fd119/by_group.tsv)。空源接触掩码和空移动组的统计仍为 JSON `null`、TSV 空白。

### v2 的候选判读

| 帧数 | 模式 | 漂移 P50 cm | 进度 P50 | 源接触脚速 m/s | 穿透超 2cm 帧 % | IK 残差 P95 m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | raw | 0.887700 | 0.964771 | 0.174437 | 0.195313 | 无锁 |
| 16 | display | 0.887700 | 0.964771 | 0.164129 | 0.195313 | 0.001177978 |
| 16 | feedback | 0.904848 | 0.963210 | 0.146779 | 0.195313 | 0.005406463 |
| 64 | raw | 2.405901 | 0.953449 | 0.178846 | 2.520752 | 无锁 |
| 64 | display | 2.405901 | 0.953449 | 0.174438 | 2.520752 | 0.000000498 |
| 64 | feedback | 2.382458 | 0.955033 | 0.167904 | 2.484131 | 0.000453108 |

按计划 §3 在各长度内与 raw v2 比较：display 的根漂移和进度不变；feedback 的最大漂移变化为 1.932%，最大进度变化为 0.166%。穿透增量最大为 0，IK 残差 P95 均小于 1cm。这四项全部通过，但四组源接触脚速仍为门槛的 4.89–5.81 倍，**v2 + lock 均未列为候选**。逐门结果见 [v2_candidate_gates.tsv](../../examples/results/terrain_controller_v6_a100_20261027/continuation_60fd119/v2_candidate_gates.tsv)。

16 帧的锁激活比例为 10.2295%，64 帧为 1.9501%；IK 残差最大值仍达 4.23–5.78cm。64 帧的穿透比例虽未因加锁增加，绝对值仍约 2.5%。

### 十二张图的审阅

[逐图审阅](../../examples/results/terrain_controller_v6_a100_20261027/continuation_60fd119/visual_review.md)逐一链接六个接触图和六个足底曲线，保留与 v1–v5 相同的窗口、场景和源动作。三种生成模式整体上大体重合：转向慢跑、侧移、蹲行和匍匐中的足部相位偏差大多保留；两段上箱的支撑脚在加锁后仍滑动。A164 的 display 右踝/脚趾在约 0.2s 出现约 0.31/0.35m/s 速度峰，feedback 降低早期速度后，后段仍高于源支撑脚。

### 账本与 B 的当前状态

| 阶段 | 分配 GPU 数 | 壁钟秒 | GPU-hours | 新增训练更新 |
| --- | ---: | ---: | ---: | ---: |
| 第 5 节长序列评估失败，计入 A 总预算 | 1 | 14.817064 | 0.004115851 | 0 |
| 本轮缺参考 TSV 的启动前失败 | 1 | 2.172321 | 0.000603422 | 0 |
| 本轮 A 完成 | 1 | 95.214735 | 0.026448538 | 0 |
| A 累计 | 1 | 112.204120 | 0.031167811 | 0 |
| v6a / v6b 预检，各自独立 | 0 | 0 | 0 | 0 |
| v6a / v6b 正式训练，各自独立 | 0 | 0 | 0 | 0 |
| B 后脚锁复评 | 0 | 0 | 0 | 0 |

GPU-hours 按进程壁钟 × 分配卡数计算；等待空闲卡不分配 GPU。CPU 测试及图片渲染不消耗 GPU-hours。A 的剩余预算仍供 B 后复评使用，合计不超过原 1800s。

B 队列等待两张空闲 A100；每臂仍为从 v1 teacher1500 开始的 1500 次新增更新，先双卡两步预检，再顺序执行 v6a、v6b，每臂最多 2h，预检和两臂合计最多 4.3 GPU-hours。当前尚无 B 的训练项统计或验收值；最终联合表将在原队列完成或停止后追加。A 的候选质量结论不改变已批准的 B 参数和预算。

本次 [结果目录](../../examples/results/terrain_controller_v6_a100_20261027/continuation_60fd119/README.md)发布小 JSON/TSV、12 张 PNG 和日志；只从私有 Drive 取回文件，传输归档 SHA-256 已校验。第 1–5 节及旧结果原样保留。
