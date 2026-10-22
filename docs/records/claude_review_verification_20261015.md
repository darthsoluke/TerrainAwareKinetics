# Claude 审查意见复核：当前 28942

复核对象是用户提交的《MotionBricks 28942 实现审查报告》，依据当前本地代码、完整模型元数据、训练 checkpoint 继承记录、现有推理证据和本次 CPU 探针。未修改训练/推理实现、权重或运行服务，未启动训练。报告提出的两个“已确认 P0”均不能按其描述成立，其推荐补丁不宜直接应用。

## 逐项结论

| 原报告判断 | 复核结果 |
|---|---|
| codec 训练时 `_adapting=False`，decoder 环境 hook 不参与 | **错误。** `source_decode` 进入 `source_codec_context`，显式设 `_adapting=True`。decoder 环境/预测高度条件的既有适配模块参与前向，但参数被冻结；可训练参数只有原生 decoder。冻结与不参与计算是两回事。 |
| 一个采样 region 全未知，就会在 attention 抛错 | **错误。** 三个 region 编成同一行的 24 个 memory tokens；attention 要求每个 batch row 至少有一个有效 token，不要求每个 region 都有效。两区全未知、身体仍有有效点时正常执行。 |
| 全未知时跳过 `_stage_hook` 就能修复边界 | **不应采用。** `sampler.observe` 在进入 hook 前已经拒绝全部未知输入。跳过注入既绕不过该检查，也会引入未验证的缺环境生成路径。全部未知的产品处理可以单独设计，不能把它冒充已知空旷平地。 |
| 边界 ValueError 会使整个服务中断 | **未成立。** WebSocket handler 捕获 ValueError 并返回错误消息。该次生成会失败；这不等于 Python 服务退出。没有对正在使用的 live 页面做越界操作。 |
| SDF 采样读到已归一化速度 | **错误。** `UnifiedTerrainController._predict` 是包在原生 `predict` 外的 wrapper；它先读取物理 VX/VZ 并采样，之后调用 `_native_predict`，其内部才归一化。 |
| Pose 同时接收旧高度场与新体积 SDF | **正确。** 实际配置和完整权重确认两路存在，训推同样激活。这是架构复杂性/条件依赖风险，当前证据不能判定为 P0 或唯一病因。 |
| 从 `fresh_kwargs` 删除 `terrain_observation` 可以修复双注入 | **不应直接采用。** 这只改变训练 helper；live 原生推理仍传入旧观测。会打破当前训练/推理条件一致性，而且旧 Pose 地形编码器是最后一轮可训练主干的一部分。移除一路需要显式架构版本、两端一致及消融验证。 |
| L-BFGS 不在 `fit_native_root_fullbank.py`，继承关系待查 | **已核实。** Adam Root → Adam continuation → L-BFGS polish → Pose。Pose parent SHA 与 L-BFGS checkpoint SHA 完全相等，L-BFGS 不是 Adam 脚本的父阶段。100 次更新、209 次损失/梯度计算、18392 窗口曝光分别记录。 |
| `atol=1e-5` 导致预取拒绝，应改为 `1e-4` | **缺少证据，不采用。** 当前 live 实际使用 `CommittedUnifiedBackend` 的覆盖方法。既有 up/down 对照各 5 次预取，均与串行输出差 0、deferred=0；这些证据不能覆盖任意长程，但不能据旧基类猜测放宽历史一致性检查。 |
| `fresh_inference_requires_played_boundary` 是容差失败证据 | **错误。** 这是旧基类因剩余 buffer/新生成策略返回的正常 deferred 原因；不能据此推导 allclose 失败。当前覆盖方法与旧基类策略也不同。 |
| Root gather 的 `-1/-2` 是当前 velocity 路径问题 | **不在当前路径。** 有 `local_root_component_mask` 时，代码执行 `apply_terminal_velocity` 后提前返回，该 gather 不执行。无需往用户服务临时插入断言验证此静态分支。 |
| 源历史训练与生成历史推理有分布差异 | **成立。** 最近训练使用源历史，连续控制使用生成历史。现有 BONES 包含 idle 和 start，不能说训练“完全没有 idle 起步”；是否覆盖目标起步命令/地形/相位组合是另一问题。 |
| 末端速度标签不是记录的玩家意图 | **成立。** `source_terminal_goal` 使用 31→32 帧速度与第 31 帧朝向。单间隔标签的稳定性、相位与持续命令的关系值得量化。不能从单帧差分本身断定其错误，也不能未经统计猜测训练速度集中在 0.3–0.8m/s。 |
| 直接混合生成历史和原源未来可修闭环 | **不充分。** 生成状态偏离后，原时间索引的源未来可能不可达；需要身体、地形、前缀一致的真实恢复监督。不能拼假过渡、平移源动作或把任意最近帧当 expert。 |

## 代码依据

路径相对于 `${PROJECT_ROOT}`，行号对应本次读取的版本。

- codec 条件上下文：`scripts/probe_unified_native_codec.py:26`、`:37`；decoder hook：`motionbricks/data/unified_terrain.py:178`；冻结范围：`scripts/fit_native_codec_fullbank.py:86`。
- raw velocity 在 native 调用之前采样：`motionbricks/data/unified_terrain.py:195`、`:206`、`:218`；原生归一化：`motionbricks/motion_backbone/inference/motion_inference.py:141`。
- region 分组/有效 token：`motionbricks/data/unified_terrain.py:43`、`:53`、`:77`；全部未知拒绝：`motionbricks/data/unified_scene_field.py:152`；服务 ValueError 处理：`scripts/soma_lab_server.py:447`。
- 旧 Pose 高度场注入：`motionbricks/motion_backbone/neural_modules/pose_backbone.py:198`；新 Pose 注入：`motionbricks/data/unified_terrain.py:136`；训练上下文：`motionbricks/data/unified_pose_training.py:62`；最后一轮解冻：`scripts/fit_native_pose_fullbank.py:104`。
- 当前 backend 构建：`scripts/soma_native_bank_server.py:75`；实际预取和提交：`motionbricks/data/unified_committed_playback.py:13`、`:34`。
- velocity 分支提前返回：`motionbricks/motion_backbone/inference/motion_inference.py:302`；训练命令来源：`motionbricks/data/unified_training_goals.py:12`。

## 本次验证与范围

已执行 36 项测试，全部通过：

```powershell
Set-Location ${PROJECT_ROOT}
$env:PYTHONPATH = '${PROJECT_ROOT};${PROJECT_ROOT}/out/vendor/kimodo'
& 'D:/.anaconda/envs/motionbricks/python.exe' -m pytest tests/test_unified_terrain.py tests/test_native_bank_live_contract.py tests/test_native_fullbank_parity.py tests/test_soma_terrain_prefetch.py -q
```

额外 CPU 探针实际调用现有 sampler、encoder、residual、controller wrapper 和 codec 上下文：

- 有界平面近边缘，三组有效点数 `[64, 0, 0]`，24 个 memory tokens 中 2 个有效；设置非零 attention 残差权重后输出有限。全部采样未知时，由 sampler 明确拒绝。
- raw 速度 `[0,0]`、`[0,1.6]`、`[1.5,-0.75]` 通过 controller wrapper 后，采样速度逐值保持不变，native 回调收到同值。回调是轻量替身；该探针验证 wrapper 输入链，未跑完整模型。调用顺序另由真实原生源码核对。
- `source_codec_context` 内 `_adapting=True`、32 帧，退出后恢复先前状态。
- CPU 只读 checkpoint 确认 `root_config.model.backbone_network.args.terrain_conditioning=True`、Pose 同样为 True；完整权重具有旧 Pose encoder/attention 和新 shared encoder/Pose attention。最后 Pose scope 包含旧 Pose encoder，共享体积编码器冻结。

结构化输出：`out/parc_rugged32_20261013_v2/claude_review_verification_20261015.json`。这些是有限范围的 CPU 链路验证，不是新增整模型 rollout、地图边界覆盖或地形质量认证。

只读核对 `/api/config`，28942 仍为已认可的完整 native 模型，紧凑权重 SHA256 `631d760c59d5f59bab3b795c11867f69e91c4704df0aa75270ba4c18443bed23`，使用 verified CUDA 和 committed-buffer-v3。引用的旧预取对照为 `native_bank_live_cuda_prefetch_20261015_v1/complete.json`；旧全地图浏览器快照为 `native_bank_live_20261015_v1/full-world-browser.json`，记录 193 个预取块、0 deferred、0 underrun、16 累积 commit-wait ticks，不能说完全无等待。

未重新统计全部 88 个实际训练窗口的速度分布。已有 `unified_pose_speed_coverage_20261014_v1-report.json` 属于较早的窗口/路线集合，不能冒充最新 bank 统计。live 包的 `up-scene.json` 入口命令为约 3.726m/s，只是一个入口值，也不是全分布。

## 应保留的审查重点

1. 量化最新 88 个窗口与实际控制的速度、朝向、步态相位和起步组合覆盖，检查真实源标签与持续意图之间的关系。
2. 在相同历史/命令/场景下，审计两路地形输入及模型预测高度的贡献；消融用于识别依赖，不以删一路后单次效果作为修复证据。
3. 对生成历史偏移构造有真实可达目标的受控恢复问题，再决定是否训练；保留原固定时间 baseline 和失败路线。
4. 对已经报告的动作突变实测跨发布块和跨生成块连续性、命令年龄及等待时长；由实际误差决定修复，维持当前历史一致性检查。

报告对小样本闭环泛化和 hindsight 命令标签的提醒有价值；它尚未证明可直接修掉当前动作问题的代码 bug。前向依赖、冻结参数、归一化时机和当前 backend 身份必须先核实，不能按严重度标签直接套补丁。
