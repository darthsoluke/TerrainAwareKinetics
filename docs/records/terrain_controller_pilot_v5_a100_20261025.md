# 第 7 步 pilot v5：反向修复、完整训练与验收

最新结果（2026-10-28）：修复后完成全部 **1500 次新增更新**，正式训练耗时 **63 分 8 秒**，八项验收通过七项。训练全程有限、没有触发发散保护，末 250 次归一化 critic loss 中位数为 0.2477；一步误差 1.136cm，仍高于保持上一帧的 0.938cm。模型未部署。完整结果从[第 7 节](#7-正式运行与训练曲线)开始，[小型结果目录](../../examples/results/terrain_controller_pilot_v5_a100_20261025/continuation_e88523d/README.md)包含表格和九张图。

第 1～5 节保留首次预检失败记录，第 6 节记录修复后的测试与双卡预检。

执行日期：2026-10-26（Asia/Singapore）。[批准的计划](terrain_controller_pilot_v5_plan_20261025.md)规定从 v4 的第 1500 次教师检查点继续，新增最多 1500 次更新、2 小时。执行源码为 `2c41613f0a96bcb4cc1bf8f773b9520848b5476f`。

**结果：84 项测试通过；补充的 CPU 零更新反向检查失败，后续预检和正式训练停止。新增训练更新、训练窗口及生成帧均为 0。**

## 1. 已完成的准备

在 A100 checkout 执行 `git pull --ff-only origin main`，更新到批准的提交。复用原独立环境和已有 `runs/bones_env_v1`，没有重新传输训练集。

| 项目 | 实测结果 |
| --- | --- |
| Python | 3.11.17 |
| PyTorch / CUDA runtime | 2.5.1+cu124 / 12.4 |
| 测试 | 84 passed，0 failed，0 skipped；11 warnings，9.79 秒 |
| 运行设备 | 测试与补充检查均为 CPU，`CUDA_VISIBLE_DEVICES=''` |
| A100 资源检查 | 扫描全部 8 卡的 index、UUID、显存、利用率及计算 PID；当时无空闲卡 |
| 历史检查点 | v1–v4 共 16 个文件，前后 SHA-256 差异为 0 |
| v4 教师副本 | 复制至新协调目录，SHA-256 与源文件相同 |
| 正式输出目录 | `runs/tmt_pilot_v5` 未创建；没有启动训练或排队进程 |

测试命令：

```bash
$PY -m pytest tests/test_bones_env_dataset.py tests/test_terrain_controller.py \
  tests/test_skeleton_ray_sampler.py tests/test_bones_terrain_fit.py \
  tests/test_bones_terrain_scene.py tests/test_bones_terrain_screen.py \
  tests/test_bones_terrain_survey.py tests/test_train_terrain_controller.py \
  -q --junitxml=runs/tmt_v5_coord/tests.xml
```

输出末行：

```text
84 passed, 11 warnings in 9.79s
```

教师源文件是 `runs/tmt_pilot_v4/phase0_teacher.pt`，保存状态为 `update=1500, exposures=1536000, phase_index=0, phase_update=1500`。这次只读取并复制了它，没有恢复训练。

| 身份 | SHA-256 |
| --- | --- |
| v4 教师检查点 | `5fd0023fa92935c648e027e567746c05c9ddcd55c9235328ade318862d274423` |
| 数据集 manifest | `67808b6813a42b676b58607b47bedc0bda9566819bee258f6ac4416c5a33c74d` |
| 沿用的固定验证批次 | `29c47a5525e2e0f2d97d715daee5e933387b79fb62b55363d57f009788c486da` |

验证批次摘要来自前轮记录，本轮尚未构造 GPU 验证批次，也没有新的逐评估摘要。

## 2. 为什么补做反向检查

源码核对发现，归一化 critic 的两条路径共同使用一个可变的 `scale`：

1. [Critic.frozen](../../motionbricks/controller/closed_loop.py#L117) 将策略输出乘以 `self.scale.detach()`。`detach()` 切断梯度，但仍共享原张量的存储与版本计数。
2. [Critic.loss](../../motionbricks/controller/closed_loop.py#L124) 随后执行 `self.scale.mul_(...).add_(...)`，原地更新尺度。
3. [ClosedLoop.rollout](../../motionbricks/controller/closed_loop.py#L269) 先构造 bootstrap 目标，再计算 critic loss。训练器接着调用策略的 `backward()`，此时前向计算保存的尺度已经被改动。

反向传播需要前向时的数值；PyTorch 检测到该值的版本变化，因此拒绝计算梯度。现有 normalized 测试只检查前向损失有限及尺度更新，没有对这组配置调用策略反向传播，所以 84 项测试通过仍留下了这个缺口。

在等待 GPU 资源时，用相同 A100 Python 环境、CPU 和仓库原始类执行下面的零更新复现。它不读动作、不加载训练模型、不创建优化器。

## 3. 复现命令与完整错误

实际执行命令（`$PY` 为上述独立 Python，工作目录为仓库根目录，环境中 `CUDA_VISIBLE_DEVICES=''`）：

```bash
$PY -c 'import torch; from motionbricks.controller.closed_loop import Critic; torch.manual_seed(0); critic=Critic(4, normalized=True); state=torch.randn(2,4,requires_grad=True); policy=critic.frozen(state).clamp(0.0,50.0).sum(); critic_loss=critic.loss([state],torch.ones(2,1)); print("device=cpu optimizer_updates=0 scale_version="+str(critic.scale._version),flush=True); policy.backward()' > runs/tmt_v5_coord/normalized_backward_check.log 2>&1
```

退出码为 **1**。完整 stdout/stderr 见 [normalized_backward_check.log](../../examples/results/terrain_controller_pilot_v5_a100_20261025/normalized_backward_check.log)；公开副本仅把私人环境目录替换为 `${PFNN_PROJECT}`。完整 traceback：

```text
device=cpu optimizer_updates=0 scale_version=2
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "${PFNN_PROJECT}/env/lib/python3.11/site-packages/torch/_tensor.py", line 581, in backward
    torch.autograd.backward(
  File "${PFNN_PROJECT}/env/lib/python3.11/site-packages/torch/autograd/__init__.py", line 347, in backward
    _engine_run_backward(
  File "${PFNN_PROJECT}/env/lib/python3.11/site-packages/torch/autograd/graph.py", line 825, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor []] is at version 2; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).
```

按训练器的调用顺序，这条冲突路径会在启用 bootstrap 后进入；给定的 smoke 是预热两次后启用。这里实际测到的是 CPU 零更新失败，尚未执行指定的单卡 smoke。

## 4. 停止位置与验收状态

| 阶段或判据 | 状态 |
| --- | --- |
| 84 项指定测试 | 通过，0 跳过 |
| 补充 normalized + bootstrap 反向检查 | 失败，原地更新引发张量版本冲突 |
| 单卡恢复 smoke：1500 起点、只缺 scale、有限值 | 未运行；已核对源检查点元数据与副本哈希 |
| 双卡强制评估失败、同次停止及 diverged.pt | 未运行；未注入异常 |
| 正式双卡 pilot | 未启动 |
| 一步误差、根漂移、穿透、骨盆、着地脚速度与比例 | 未评估 |
| 新门：全程不触发发散保护 | 未评估，没有训练过程 |
| 新门：末 250 次 critic loss 中位数 < 10 | 未评估，没有训练更新 |

本次错误来自补充的 CPU 检查，没有进入训练器的发散保护；因此不存在 `diverged.pt`。训练、模型与损失代码保持批准的提交内容，未调参、重启或部署。

继续执行前，需要让 bootstrap 的反向计算持有本次前向所用的尺度快照，并补上 normalized critic 与 bootstrap 同时启用时的反向测试。

## 5. 独立账本与产物

| 计数 | 本轮新增 |
| --- | ---: |
| 单卡 / 双卡预检优化器更新 | 0 / 0 |
| 正式策略 / critic 更新 | 0 / 0 |
| 训练窗口 / 训练生成帧 | 0 / 0 |
| 验证与六窗口复核生成帧 | 0 |
| 已用正式训练壁钟 | 0 秒 |
| 未使用批准预算 | 1500 次更新，2 小时 |

CPU 合成张量检查不构成训练数据曝光。已读取的 v4 教师状态仍属于原账本，不计作 v5 的新增更新。

[精简结果目录](../../examples/results/terrain_controller_pilot_v5_a100_20261025/README.md) 包含机器可读报告、84 项测试日志、完整反向错误和输入身份记录。服务器 `runs/tmt_v5_coord/` 保留原始 JUnit、日志、输入快照和教师副本；16 个旧检查点逐文件校验未变。

正式训练未启动，所以本轮没有 v5 评估表、梯度/critic 曲线、匹配更新行、最终分组结果、六张接触图或两张足底图。既有 v1–v4 结果保留原样。

## 6. 2026-10-27 修复后续跑

以上第 1～5 节保留 2026-10-26 的失败记录。本节对应用户批准的修复后执行，源码为 `e88523d5088d34effb2a79dccbb3fcaeab150b3f`，训练计划、1500 次新增更新和 2 小时上限均保持不变。

### 测试与原始复现

在 A100 的独立 Python 3.11.17 / PyTorch 2.5.1+cu124 环境，以 `CUDA_VISIBLE_DEVICES=''` 执行整个测试目录：

```bash
$PY -m pytest tests -q --continue-on-collection-errors \
  --junitxml=runs/tmt_v5_fixed_coord/tests_full.xml
```

完整结果：**3124 passed、280 failed、275 errors、33 skipped，482 warnings，142.45 秒**。失败主要来自旧 MotionBricks 模型配置、PARC、SONIC 等私有依赖或资产缺失；另有旧模块的 `PRIVATE_STORAGE` 未定义、`governor_state` 属性缺失等代码错误。完整仓库测试没有全部通过，原始日志及 JUnit 保留于服务器协调目录。

本轮相关的八个测试文件另行执行，结果为 **85 passed，0 skipped，11 warnings，8.08 秒**。下面两个测试在全仓库运行和相关文件运行中都实际执行且通过：

- `test_normalized_critic_bootstrap_backward_survives_the_scale_update`。
- `test_teacher_closed_loop_and_evaluation_steps_run_on_a_batch`，包含 normalized critic、bootstrap cap，以及连续的 `policy.backward()` 和 `critic_loss.backward()`。

第 3 节的零更新 CPU 复现也重跑成功，退出码 0：

```text
device=cpu optimizer_updates=0 scale_version=2
policy_backward=passed
```

新日志为 `runs/tmt_v5_coord/normalized_backward_check_e88523d.log`，与旧的 `normalized_backward_check.log` 同目录保存。源教师检查点仍为第 1500 次、SHA-256 `5fd0023fa92935c648e027e567746c05c9ddcd55c9235328ade318862d274423`，16 个历史检查点哈希一致。

### 双卡两步预检

2026-10-27 22:39:37～22:40:01（Asia/Singapore），确认物理 GPU 0、1 空闲后按 UUID 固定分配。每卡 512，恢复原 v4 教师副本，执行两次 `loop:4` 更新；归一化、250 次预热、bootstrap cap 50、三项代价封顶及梯度退出阈值均与正式 v5 相同。两步仍处于预热期；启用 bootstrap 的反向另由已通过的单元测试覆盖。

| 检查 | 结果 |
| --- | --- |
| 每个 rank 的策略 / critic 更新 | 2 / 2，两个反向均完成 |
| 各卡参数 | 逐张量摘要一致；评估中的 `rank_parameter_disagreement=0` |
| critic 恢复 | 仅缺新增的 `scale` 缓冲区 |
| `critic.scale` | rank 0：1.03441155；rank 1：1.03611860，均从 1.0 改变 |
| 全部逐次记录与评估值 | 有限 |
| 每卡峰值已分配显存 | 3,475,231,744 字节 |
| 壁钟 / GPU-hours | 23.7516 秒 / 0.0131953 |
| 保存状态 | update 1502；exposures 1,544,192 |
| 输出处理 | 通过后删除 `runs/tmt_v5_fixed_gpu_preflight/`；测量日志与检查结果保留于协调目录 |

这两次更新单列为预检消耗：2048 个窗口、8192 个闭环帧和 2048 个混合教师预测。正式训练仍从原 v4 第 1500 次教师检查点开始，不继承这两次更新。GPU-hours 按“进程壁钟 × 分配卡数 ÷ 3600”计算，不含等待资源的时间。

修复后的测试、输入身份和失败清单见 [continuation_e88523d 结果目录](../../examples/results/terrain_controller_pilot_v5_a100_20261025/continuation_e88523d/README.md)。

### 原单卡预检与双卡故障退出检查

单卡 smoke 从同一教师副本恢复到 update 1504，新增 4 次更新，耗时 20.5496 秒。critic 仅缺新增的 `scale`，全部训练和评估值有限；warmup=2，后两次更新启用 bootstrap，两个反向均完成。

双卡故障检查在 update 1502 的评估入口主动抛出 `RuntimeError: v5 preflight: intentional evaluation failure after update 1502`。两张卡都在 update 1502、exposures 1,536,512 停止，各 rank 退出码为 2；torchrun 汇总退出码为 1，符合本次故障注入预期。只有 rank 0 写入 `diverged.pt` 和 `latest.pt`，各卡参数摘要与保存的检查点一致。`eval.jsonl` 记录了阶段内及结束时两次相同错误，参数差异均为 0。

这个检查耗时 15.2279 秒。`diverged.pt` 保存模型、critic、两个优化器、学习率调度器、计数状态和配置。错误由测量脚本注入，仓库训练源码未改；正式 pilot 从原教师文件重新恢复。

## 7. 正式运行与训练曲线

源码固定为 `e88523d5088d34effb2a79dccbb3fcaeab150b3f`。2026-10-28 **00:35:11.791～01:38:14.397**（Asia/Singapore），使用确认空闲的物理 A100 0、1，按 UUID 固定分配，每卡 batch 512，全局 batch 1024。

执行参数如下。实际由一次性 nohup 监督进程通过测量包装器调用同一训练器，并加外部 `timeout 7200`；包装器记录每次更新、参数摘要、显存和验证批次摘要。

```bash
torchrun --standalone --nnodes 1 --nproc_per_node 2 scripts/train_terrain_controller.py \
  --dataset runs/bones_env_v1 --output runs/tmt_pilot_v5 \
  --resume runs/tmt_pilot_v4/phase0_teacher.pt \
  --phases teacher:1500 loop:4:500 loop:8:500 loop:16:500 \
  --teacher-mix 1.0 --foot-weight 1.0 --contact-source source \
  --support-weight 2500 --skating-mode contact --skating-weight 100 --penetration-weight 2500 \
  --support-gap-clip 0.1 --penetration-depth-clip 0.1 --skating-speed-clip 1.0 \
  --critic-normalized --critic-warmup 250 --bootstrap-cap 50 \
  --abort-grad-norm 1e4 --abort-patience 25 \
  --batch 512 --max-hours 2 --eval-every 250 --eval-batch 256
```

恢复位置为 update 1500、phase 1、phase_update 0；没有重跑教师阶段。正式进程退出码 0、`stopped=complete`，保存到 update 3000，全部新增更新均已保存。7 条评估记录的固定批次 SHA-256 均为 `29c47a5525e2e0f2d97d715daee5e933387b79fb62b55363d57f009788c486da`，`rank_parameter_disagreement` 全为 0；最终检查点的每个参数张量与两张卡都一致。16 个旧检查点哈希未变。

正式壁钟 **3788.0778 秒**；两卡峰值已分配显存分别为 **12,745,025,024 / 12,763,436,544 字节**。每阶段恰好 500 次更新，rank 0 测得的每次更新平均时间为：H4 **1.2634 秒**、H8 **2.2641 秒**、H16 **3.9443 秒**，包括闭环、混合教师、两个反向和优化器，不含数据加载与评估。

[训练曲线](../../examples/results/terrain_controller_pilot_v5_a100_20261025/continuation_e88523d/training_curves.png)和[1500 条逐次记录](../../examples/results/terrain_controller_pilot_v5_a100_20261025/continuation_e88523d/training_updates.tsv)显示：预热结束和课程切换附近有波动，但裁剪前梯度最大值为 **1533.90625**，全程没有达到 1e4。末 250 次 critic loss 中位数：rank 0 **0.2476825**，rank 1 **0.2427863**；先按每次更新对两卡取均值、再求中位数为 **0.2549127**，三种统计均低于 10。

## 8. 评估与验收

评估固定使用 256 个验证窗口，每个闭环 16 帧。保持基线为 0.938170cm；源动作的穿透超限比例为 0，几何着地脚速度为 0.0136703m/s，着地比例为 32.2632%。以下“速度”和“着地比例”均按各自几何着地脚计算。

| 累计更新 | 一步误差 cm | 根漂移 P50 / P90 cm | 穿透超限帧 % | 骨盆误差 cm | 着地脚速度 m/s | 着地比例 % |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1750 | 0.7763 | 4.286 / 19.481 | 7.3975 | 2.889 | 0.02915 | 53.5889 |
| 2000 | 1.1374 | 3.325 / 19.051 | 1.2207 | 3.428 | 0.03456 | 39.1113 |
| 2250 | 1.2216 | 3.698 / 30.466 | 0.6592 | 2.595 | 0.02969 | 46.0449 |
| 2500 | 1.1039 | 2.830 / 30.041 | 0.8545 | 2.691 | 0.02396 | 39.2212 |
| 2750 | 1.1945 | 3.201 / 38.278 | 0.6348 | 2.748 | 0.02080 | 46.2585 |
| 3000 | 1.1358 | 3.654 / 38.957 | 0.4883 | 2.860 | 0.02118 | 45.6482 |

结束评估与第 3000 次定期评估完全一致，原始[评估表](../../examples/results/terrain_controller_pilot_v5_a100_20261025/continuation_e88523d/evaluation_table.tsv)保留两行，也包含全部闭环损失项。按源接触标记计算的最终脚速为 0.0142179m/s，源动作为 0.0120614m/s。

| 验收项 | 要求 | 最终结果 | 判定 |
| --- | --- | --- | --- |
| 数值与各卡参数 | 全程有限，卡间差异 0 | 通过全部逐次记录与 7 次评估检查 | 通过 |
| 一步全身误差 | 比保持基线低至少 30%，即 ≤ 0.656719cm | 1.135788cm，比基线高 21.06% | **失败** |
| 16 帧根漂移 P50 | < 10cm | 3.654092cm | 通过 |
| 穿透超 2cm 帧比例 | ≤ 源 + 0.5 个百分点 | 20/4096 帧，0.488281%；门槛 0.5% | 通过 |
| 骨盆高度误差 | < 5cm | 2.860146cm | 通过 |
| 几何着地脚 | 速度 ≤ 0.0705055m/s；比例 ≥ 16.1316% | 0.0211785m/s；45.6482% | 通过 |
| 新门：不触发发散保护 | 全程无保护退出 | 完成，无 `diverged.pt` | 通过 |
| 新门：末 250 次 critic loss | 归一化中位数 < 10 | 两卡分别 0.2477 / 0.2428 | 通过 |

**八项通过七项，整体验收未通过。** v5 完成了稳定的闭环训练，但一步姿态拟合仍退步。没有追加训练、调整参数或部署。

[v1～v5 同更新位置表](../../examples/results/terrain_controller_pilot_v5_a100_20261025/continuation_e88523d/matching_updates.tsv)包含 1750、2000、2250、2500、2750、3000 六个位置，共 30 行。v4 在 2500 评估失败，后续位置明确标记 unavailable；旧版当时未记录的新增指标也保留 unavailable。v5 的低滑步与较高着地比例同时成立，移动动作的跟随仍需单独看下面的分组与图片。

## 9. 最终分组

| 组 | 窗口 | 根漂移 P50 cm | 穿透超限帧 % | 源穿透 % | 骨盆误差 cm |
| --- | ---: | ---: | ---: | ---: | ---: |
| crouch | 48 | 20.450 | 1.0417 | 0 | 4.349 |
| flat_control | 43 | 39.071 | 0 | 0 | 2.487 |
| kneel | 35 | 1.413 | 0 | 0 | 1.451 |
| known_dimension_support | 20 | 5.970 | 2.1875 | 0 | 5.423 |
| prone | 13 | 1.648 | 0 | 0 | 0.609 |
| quadruped | 90 | 2.930 | 0 | 0 | 2.543 |
| wall_contact | 7 | 2.056 | 4.4643 | 0 | 2.913 |

总 P50 为 3.65cm，但 flat_control 为 39.07cm、crouch 为 20.45cm，移动片段的跟随问题很明显。20 个穿透超限帧来自 crouch 8 帧、known_dimension_support 7 帧和 wall_contact 5 帧；wall_contact 的 7 个窗口共有 112 帧，其中 5 帧超限。精确值见[分组表](../../examples/results/terrain_controller_pilot_v5_a100_20261025/continuation_e88523d/by_group.tsv)。

## 10. 固定窗口视觉复核

六个窗口沿用 v1 的文件、起点、场景变体与相机约定。蓝色是源动作，绿色是第 3000 次检查点生成的 16 帧；全部 96 帧生成完成，源 FK 数组与 v1～v4 保存的数组逐项相同。

| 窗口及图片 | 逐图观察 | 末帧根漂移 cm |
| --- | --- | ---: |
| [转向慢跑 A255](../../examples/results/terrain_controller_pilot_v5_a100_20261025/continuation_e88523d/contact_sheets/turn_jog_270_R_003__A255__s40.png) | 绿色腿部摆动幅度较小，后半段与蓝色源动作的位置和朝向分离。 | 41.84 |
| [侧移停止 A030](../../examples/results/terrain_controller_pilot_v5_a100_20261025/continuation_e88523d/contact_sheets/walk_sideway_045_stop_002__A030__s60.png) | 身体姿态保持完整，但绿色足部移动明显减少，末段未复现源动作的右脚抬起。 | 37.22 |
| [蹲行拾物 A412](../../examples/results/terrain_controller_pilot_v5_a100_20261025/continuation_e88523d/contact_sheets/item_pick_up_crouch_walk_R_003__A412__s32.png) | 低姿态保留，源动作继续向前、降低身体，生成动作逐渐落后且躯干偏高。 | 54.32 |
| [匍匐转换 A129](../../examples/results/terrain_controller_pilot_v5_a100_20261025/continuation_e88523d/contact_sheets/change_idle_crawl_to_idle_crawl_right_001__A129__s48.png) | 蓝绿骨架大体重合，保持低姿态，手脚末端有小幅差异。 | 1.27 |
| [上箱 A224](../../examples/results/terrain_controller_pilot_v5_a100_20261025/continuation_e88523d/contact_sheets/come_up_50cm_box_R_002__A224__s24.png) | 支撑侧较稳定，摆动脚抬升不足，后段躯干低于源动作；骨盆误差 6.91cm。 | 4.83 |
| [上箱 A164](../../examples/results/terrain_controller_pilot_v5_a100_20261025/continuation_e88523d/contact_sheets/come_up_50cm_box_R_003__A164__s24.png) | 上箱趋势保留，腿部轨迹与源动作未重合，末段身体偏低；骨盆误差 5.51cm。 | 8.05 |

两张足底曲线同时画出源动作、v2、v3、v4 与 v5。v4 明确使用**保存于 2250 次**的数组；曲线使用对称对数纵轴，保留其失败值。

- [侧移停止足底曲线](../../examples/results/terrain_controller_pilot_v5_a100_20261025/continuation_e88523d/sole_curves/walk_sideway_045_stop_002__A030__s60__soles.png)：v5 左脚提前落下，右脚高度一直在约 -1.36～-0.13cm，未复现源动作后半段的抬脚。右脚全窗口速度 P50 为 0.01182m/s，源动作为 0.26750m/s；这个窗口显示移动不足与低脚速同时出现。
- [上箱足底曲线](../../examples/results/terrain_controller_pilot_v5_a100_20261025/continuation_e88523d/sole_curves/come_up_50cm_box_R_002__A224__s24__soles.png)：v5 右支撑脚高度约 0.571～0.575m，接近源动作；左摆动脚最高约 0.459m，低于源动作的 0.744m，水平移动也较少。

## 11. 账本、GPU-hours 与文件

| 阶段 | 新增策略更新 | 训练窗口 | 闭环生成帧 | 混合教师预测 | GPU-hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| 双卡两步预检 | 2 | 2048 | 8192 | 2048 | 0.0131953 |
| 单卡 smoke | 4 | 128 | 512 | 128 | 0.0057082 |
| 双卡故障退出检查 | 2 | 128 | 512 | 128 | 0.0084600 |
| **正式 pilot** | **1500** | **1,536,000** | **14,336,000** | **1,536,000** | **2.1044877** |
| 合计 | 1508 | 1,538,304 | 14,345,216 | 1,538,304 | **2.1318512** |

每次策略更新对应一次 critic 更新；正式 pilot 每张卡各执行 1500 次策略和 1500 次 critic 优化器步骤。表中“新增策略更新”按全局同步更新计数，不把两卡重复计为两次。正式 H4/H8/H16 各 500 次；继承的 1500 次教师更新和 1,536,000 次旧教师曝光保留原账本。

原生 checkpoint 的 `exposures=15,872,000` 是继承教师曝光加本轮闭环帧数，未包含另列的混合教师预测。没有消耗但未保存的正式更新。评估另生成 39,936 个闭环帧、2496 个一步诊断帧；六窗口复核生成 96 帧，均不计入训练窗口，见[双卡统计与诊断帧账本](../../examples/results/terrain_controller_pilot_v5_a100_20261025/continuation_e88523d/rank_and_frame_summary.json)。

最终 `runs/tmt_pilot_v5/latest.pt` 的 SHA-256 为 `594882aee0927fdf7a76e9af8a9ef9019a7140cc4e16a0d9a5e3d3b707570a15`。各阶段检查点、原始日志和私有动作数组留在 A100；公开目录只含聚合 JSON/TSV、九张 PNG、测试日志及本文。未进行新一轮训练、模型替换、长时控制或本地部署。
