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
