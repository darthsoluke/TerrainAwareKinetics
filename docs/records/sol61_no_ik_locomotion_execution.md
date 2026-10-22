# sol6.1 执行交接：无 IK 的 SOMA 连续 locomotion

日期：2026-10-05。执行目标不是继续改善一组重建视频，而是得到可接 UE 的、由历史和控制目标驱动的连续动作后端。

## 1. 不变的边界

- SOMA30 运动表示，官方 SOMA77 补齐、FK 与蒙皮，平地；先做 idle、walk、run、turn、backward、lateral 及其过渡。
- 不使用足部 IK 锁定、脚锚点后处理、解码后 root 补偿或逐帧贴地来消除滑步。验收原始生成结果。
- 不重新研究骨架、重造接触标签、扩大质量/惯量诊断、加入尚未验证的动力学真值。本轮不用地形、强化学习或物理跟踪器。
- 保留 MB 的关键帧/轨迹可控架构，但普通操控测试不能输入录制动作的未来身体姿态。编辑器主动提供的未来关键帧是另一种合法模式，单独验收。
- 保留旧数据、运行和用户改动。实验许可与 UE 部署通过分开记录，不能因部署尚未通过而循环阻止训练。
- 本文取代旧路线中“接触修正兜底”的建议；已有 UE 骨架重定向与坐标转换不等于允许用足部锁定掩盖生成误差。

## 2. 本轮已知事实

V3 Region 使用 152 条训练片段，在固定 57 个 validation 窗口上，原始重建 MPJPE 35.69 mm、实测稳定区间滑步 56.54 mm/s。它不是玩家控制结果，编码器看过完整源窗口。

本轮机制实验已经实现并运行：

- 同一模型绕过量化后，MPJPE 43.73 mm、滑步 65.06 mm/s，反而更差。连续 latent 是分布外探针，不能据此证明量化永远不是瓶颈，但没有证据支持现在扩大码本。
- 源 FK 与速度目标差约 1.55e-5 m/s，未发现该处坐标/时间差分错误。
- 解码的 XYZ 与旋转 FK 相差约 13.93 mm，值得检验表示一致性。
- 接触速度误差的共同平移分量约占平方误差 77.23%；这只说明 root 改动可以补偿部分误差，不证明 root 是原因。当前重建已输入正确源 root。
- 增加预测 XYZ/FK/速度一致性的 Control/Consistent 两组各 10,000 次更新已完成。两组从同一 V3 Region 权重开始，旧损失相同；实验组仅增加一致性项，不加推理后处理。
- 本地完整自动测试为 198 passed。测试通过只证明已测契约，不证明动作质量。

本次最终实验结论和下一阶段 checkpoint 见文末“收尾结果”；以该处为准，不复用历史错误评分表。

## 3. 从近期方法学习什么

### MotionBricks

[论文](https://arxiv.org/html/2604.24833v1) 的 tokenizer、Root、Pose 分工与轨迹/关键帧控制继续作为主体。完整源动作压缩重建不等于历史条件下生成；下一步必须补齐 Pose 和连续执行链路。

### ARDY，SIGGRAPH 2026

[论文](https://research.nvidia.com/labs/sil/projects/ardy/assets/ardy_paper.pdf) 和 [两阶段源码](https://github.com/nv-tlabs/ardy/blob/main/ardy/model/auto_latent_twostage_denoiser.py) 的可迁移机制是：先预测全局 root，再以该预测的局部根运动条件生成身体，两者在生成过程中协调；不要把玩家期望速度直接当作人体已实现的位移。

源码训练时将 root 转换结果 detach，因此不能把它描述为身体损失直接贯穿所有 root 参数。它的解码器用局部根速度减少滑步；我们的 MB 已有这种条件，不重复发明。论文仍报告滑步/抖动，官方发布当前为 Core/G1，SOMA 权重尚未发布，不能直接当作 SOMA 替换品。

### DART / DartControl，ICLR 2025

[论文](https://arxiv.org/html/2410.05260v3) 和 [训练源码](https://github.com/zkf1997/DART/blob/main/mld/train_mvae.py) 可借鉴的是：冗余运动表示保持一致，以及逐渐使用模型生成的历史进行训练。只用真实历史训练，然后无限接续自己的输出，会产生训练和运行差异。

上述机制是参考，不整套搬入扩散模型，不采用旧系统足部 IK，也不声称论文已经保证零滑步。

## 4. 执行顺序

### A. 冻结 tokenizer，接通官方 Pose/Root 路径

1. 使用文末指定的 tokenizer；冻结参数和码本统计，记录 checkpoint/骨架/stats/dataset 哈希。后续 Pose 训练期间绝不能更新码本。
2. 复用 `motionbricks/motion_backbone/models/pose_model.py`、对应 `pose_backbone`、官方 `motion_inference.py`；在人类数据适配层增加 Pose 阶段，不新写一个无关控制器。现有 `train_bones_locomotion.py` 只有 tokenizer/root 两种阶段，不能把它描述为 Pose 已接通。
3. 处理真实冻结缺陷：上游有 `requires_grad_ = False` 的属性赋值，它不等于调用冻结方法。用自动测试验证 tokenizer 参数/EMA 不变、device 正确、train/eval 切换不误开码本更新，修复尽量局限在人类适配路径。
4. 最初只用真实历史 + 根轨迹训练 Pose 的官方 token 预测任务，建立可学习基线；不得把 argmax 后接触误差伪装成可反传的训练目标，也不凭空平均 token embedding 当作有效动作。
5. Root 学习从历史和稀疏目标产生可执行的轨迹；Pose 在实际 Root 输出上运行。保留 source-root 条件只作诊断上限，两种结果分别记录。
6. 运行时唯一运动来源是模型输出 root。`provided_velocity` 在 decoder 中可以表示“上游 Root 已预测的轨迹”，不能表示外部按键另行拖动 root。先不删除这一已训练契约，也不同时重训整个 tokenizer。

Pose 先使用现有端到端路线提出的紧凑官方 Transformer 配置（宽度 512、8 层、8 个 attention heads），而不是直接沿用预览权重的 1024/16 大模型。token 头数、码本大小和特征维度从冻结 tokenizer/表示读取，不硬编码 G1 配置。该容量只是固定初始方案，RTX 3080 是否达标仍须实测。

现有 `validate_training_scope` 只给 tokenizer 提供长于烟测的受限试验入口。应新增有独立协议、步数上限、数据/权重身份和自动前置检查的 Root/Pose 试验范围；不能将 `full_training_gate_passed` 改成 true，也不能因此又停在 1,000 步烟测交接。本文预算是待执行实验的边界，不是对完整数据或部署质量的认证。

A 的交付：一次完整 `历史 + 目标 -> Root -> Pose tokens -> decoder -> SOMA` 生成可以重复运行、保存 checkpoint 并恢复，绝非从未来源动作编码得到 tokens。

### B. 建立真正的连续生成并训练接续能力

1. 实现单一 rollout 入口，训练评估和未来 UE 后端复用。只保留已实际执行的历史；已发布帧不能因下一次重规划被改写。
2. 初始测试计划 64 帧、提交 4 帧再重规划，使用官方允许的历史/上下文长度。这个长度是起点而非结论；记录有效历史、token 对齐和边界速度。不要在同一轮同时扫描历史长度/网络宽度/损失权重。
3. 从训练片段构造连贯的相邻窗口，逐渐引入模型生成历史。教师历史、生成历史分别统计；状态缓存按序列隔离，不把随机片段当连续时间。
4. 先只替换真实相邻片段的历史，保留相容目标。所有坐标重定位必须同时变换历史、目标、身体和 root。不要大幅改变轨迹却仍用原来的足部坐标作正确答案，也不要把历史漂移后的突跳强行监督成正常过渡。
5. 训练条件和线上条件一致：目标速度/朝向是意图，不是完整未来动捕；未来姿态、接触标签、骨盆高度均不能偷看。监督损失仍可使用训练目标的源标签，但它们不是运行时输入。
6. 接触状态由运动生成路径预测/表达，稳定接触、滚动换点、离地不能被“一律锁脚”代替。已有源证据蒙版继续用于配对监督，禁止用预测接触关闭自己的惩罚。

B 的交付：连续 60 秒的 start/walk/stop、walk/run、90/180 度转向、倒退、侧移脚本，至少 3 个固定采样 seed。无重置、无真实未来动作、无 IK，保存逐帧输入、输出和计时。

### C. 训练和数据预算，不再检查完一轮就停止

- 先运行短集成烟测并自动阻断确切缺陷，然后直接启动训练。不要把烟测或过拟合几个片段当最终交付。
- 起始预算：Root 10,000、Pose 20,000 次有效更新；生成历史训练作为另一明确阶段，先 10,000 次。第一阶段 tokenizer 固定，后续阶段不偷偷换码本。
- 教师历史与生成历史微调的比较必须从相同 Pose checkpoint 出发、同预算、同数据。另保留完全未微调 checkpoint，避免把多训练的收益当作接续策略的收益。
- 本轮先用现有 152 条 train、57 个固定 validation 窗口，test 保持不动。自动统计六类行为及起停/转向过渡覆盖；若某行为/过渡缺失，明确列出，不声称学会。
- 若主要失败对应缺失数据，再从已有 BONES 原始数据增补一个版本化训练集。按原始动作及其派生/增广家族隔离划分，排除现有 val/test 及近重复；不能只按文件名避漏。不要重新下载已有全集。
- 启动前重新查 GPU，独立同预算对照可用不同空闲卡；不要占用他人的卡，也不要为了多卡而更改有效 batch、更新次数或数据量。

### D. 验收原始生成，不只看平均滑步

- 配对的 57 窗口继续使用固定源蒙版：MPJPE、局部旋转误差、滑步均值/分位/峰值、穿地/离地、关节加速度误差。
- 自由操控没有一份时间对齐的真实未来，不得沿用源蒙版或把 MPJPE 当生成质量。单列固定几何判据下的近地物质点速度、接触持续段漂移、鞋底穿地/悬空分布；报告覆盖率和未决率，不能让模型用“全不接触”作弊。
- 同时报 root 目标跟踪、实际速度/朝向、加减速/转向响应、位移累计误差、窗口边界跳变。站着不动、持续悬空或走错方向，即使滑步小也不通过。
- 保留全部脚本与 seed 的汇总、最坏片段和完整视频，不只交付挑选的两秒 GIF。
- 先证明无 IK 的 SOMA 连续生成，再迁移 UE；不以更换渲染器或重定向掩盖源模型问题。

## 5. UE 与 RTX 3080 的后续接口

后端输入：已应用的人体历史、玩家意图、可选作者关键帧；输出：有时间戳的 root、SOMA 局部旋转、接触预测及模型身份。只允许一个推理任务在途，采用最新输入重规划，拒绝旧结果，不堆积输入队列。

UE 沿用既有 Enhanced Input / Character / Movement / AnimInstance 分工。移动组件只应用模型 root 一次；实际碰撞结果反馈给下一次历史。重定向后的误差与原始 SOMA 误差分别测量，不加足部修正来通过验收。

RTX 3080 必须实测 GPU 生成耗时、输入至首个受影响姿态的延迟、帧预算及缓冲耗尽次数。分块生成可以输出 60 Hz 姿态流，但不能把渲染插值 60 FPS 说成模型每秒推理 60 次。先验收原生 60 Hz 输出与 UE 帧时间，再单列模型更新频率。具体延迟阈值在设备烟测后冻结，不承诺论文设备上的数值能直接复现。

## 6. 环境与现有资产

- 本地项目：`${PROJECT_ROOT}`
- SSH：`ssh -i ${SSH_KEY_FILE} -p 12303 -o BatchMode=yes -o StrictHostKeyChecking=yes private-user@remote-host.example`
- 远端项目：`${PRIVATE_STORAGE}/motionbricks-preparation/GR00T-WholeBodyControl/motionbricks`
- Python：`${PRIVATE_STORAGE}/motionbricks-preparation/envs/mb-bones/bin/python`
- PYTHONPATH：`${PRIVATE_STORAGE}/motionbricks-preparation/kimodo:${PRIVATE_STORAGE}/motionbricks-preparation/GR00T-WholeBodyControl/motionbricks`
- 数据：`${PRIVATE_STORAGE}/motionbricks-bones/pilot-v7`
- 完整训练标签：远端项目 `out/sole_region_repair_v3/labels_train`
- 原始 BONES：`${PRIVATE_STORAGE}/bones-seed/uniform_orig/soma_uniform/bvh`
- 本轮训练：`${PRIVATE_STORAGE}/motionbricks-preparation/runs/locomotion-consistency-20261005/{control,consistent}`
- 本轮评估：远端项目 `out/locomotion_consistency_v1/review_all`
- 最初 V3 Region：`${PRIVATE_STORAGE}/motionbricks-preparation/runs/sole-region-repair-v3-20261005-r4/region`

## 7. 必须自动执行的防回退检查

1. tokenizer/codebook 全程冻结且恢复一致；新阶段参数确实更新。
2. 同一 history/command/seed 下，改变未提供的未来真值不影响生成输出。
3. 普通操控模式无未来身体条件；source-root 上限实验不能混入操控成绩。
4. 全程同一 root 权威，坐标重定位、序列切换、窗口交接不重复积分、不串状态。
5. 从生成历史连续运行，不能暗中周期性重置为真实姿态。
6. 无 NaN、无空评分伪通过、无预测标签改变配对监督蒙版；零覆盖记不可判断。
7. 日志和可视化都确认 IK、足部锁定、贴地、离线平滑关闭。

检查嵌入训练和评估入口自动运行；只在发现确切缺陷时修复重试，不另开无限诊断阶段。

特别检查 tokenizer 的非因果卷积：运行时若需要历史 tokens，必须仅由已经可用的历史计算，不能先编码完整真实未来再截取前缀。目标 tokens 可以在训练时编码未来，但与运行时输入严格分离。

## 8. 下轮交付清单

- 可恢复的冻结 tokenizer + Root/Pose 训练产物及精确配置/数据/代码身份。
- 可运行的连续生成入口、实际命令序列、3-seed 汇总、完整原始 SOMA 视频和最坏失败片段。
- teacher-history 与 generated-history 的等预算比较，说明哪些问题仍来自 tokenizer、哪些来自预测/接续。
- 尚未达标如实标注；不能把流程跑通写成自然动作已解决。下一步由失败证据决定，不默认继续加损失。

## 收尾结果

两组训练和最终评估均已完成，进程退出码为 0。57 个窗口、7,615 个实测稳定区间完全一致，初始加载状态/seed/数据/标签清单一致。指标均为未经 IK/锁脚的模型原始重建：

| 指标 | Control | Consistent |
|---|---:|---:|
| XYZ/FK 差异 | 13.12 mm | 6.17 mm |
| 滑步均值 | 50.24 mm/s | 49.49 mm/s |
| 最坏滑步 | 988.51 mm/s | 1,226.97 mm/s |
| MPJPE | 35.50 mm | 36.54 mm |
| 各窗口 P95 滑步的均值，非合并 P95 | 164.00 mm/s | 170.61 mm/s |
| 关节加速度误差 | 4.065 m/s² | 3.572 m/s² |
| 鞋底穿地均值 | 0.883 mm | 0.816 mm |
| 最深穿地 | 27.48 mm | 26.98 mm |
| 训练耗时 / 峰值分配显存 | 616.0 s / 865.1 MB | 632.6 s / 866.4 MB |

预设预测：一致性差异降至对照的 70% 以下、滑步降至 80% 以下、MPJPE 不超过 105%。实际比值为 47.02%、98.50%、102.93%。**滑步预测失败，不接受为滑步解决方案。** 表示更一致、加速度更平滑是独立收益，但不足以解释或消除主要接触误差；单 seed 也不能为 1.5% 的滑步差异作统计结论。

**下一阶段固定使用 Control**，原因是相同预算下姿态误差更低、滑步尾部更小，且不引入未证实有效的额外目标。它只获准作为 Root/Pose 实验底座，不是合格 UE 动作模型，不意味着问题已解决。

- checkpoint：`${PRIVATE_STORAGE}/motionbricks-preparation/runs/locomotion-consistency-20261005/control/checkpoints/final.ckpt`
- checkpoint SHA256：`134ed76c3bc2fcdcbe087c945aac15d42ba4735f7ec785cb1e4f71c8415435f5`
- 完整评估：本地项目 `out/locomotion_consistency_v1/review_all/sole_region_ab_metrics.json`
- 初始状态核对：本地项目 `out/locomotion_consistency_v1/paired_initialization.json`
- 机制探针：本地项目 `out/locomotion_mechanisms_v1/region_probe_exact_windows.json`

下一步直接进入 A，不扩大一致性损失扫描，不继续以重建视频冒充连续生成。本轮未启动 Root/Pose 新训练，未改 UE，未生成新的操控视频。
