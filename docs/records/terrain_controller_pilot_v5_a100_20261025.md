# 第 7 步 pilot v5：归一化 critic 的反向预检失败

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
