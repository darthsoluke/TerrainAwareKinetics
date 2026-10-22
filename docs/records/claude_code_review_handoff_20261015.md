# Terrain-Aware MotionBricks：Claude Code 审查交接

目标：让 SOMA 人物在 WASD 指令下，根据周身及前进方向的环境自然上下楼，并保留正常平地运动。先审查现有实现与证据，不默认开始新训练。**以实际代码、当前权重和可复现推理为准；本说明及历史报告只提供索引，不作为正确性证明。**

工作区：`${UPSTREAM_ROOT}`。本任务主要在其 `motionbricks/` 子项目，以下路径均相对此子项目。外层 `gear_sonic*`、`decoupled_wbc/` 等机器人项目不是当前网页控制器的入口。

## 1. 目录地图

```text
motionbricks/                          # 子项目根目录
├─ motionbricks/                       # Python 包，与外层目录同名
│  ├─ motion_backbone/
│  │  ├─ models/                       # Root / Pose 模型与采样
│  │  ├─ inference/                    # MB 原生生成与解码编排
│  │  └─ demo/                         # 官方风格 WASD 控制、agent 和运动缓冲
│  ├─ vqvae/                           # 多头量化 tokenizer / codec
│  ├─ motionlib/                       # 骨架、动作表征及训练基础设施
│  ├─ data/                            # 本项目主要扩展：数据、地形、控制、评估
│  ├─ geometry/                        # 几何工具
│  └─ exp_setup/                       # 实验配置装配
├─ scripts/                            # 数据准备、训练、诊断、导出及本地服务
├─ tests/                              # 协议、坐标、损失、推理及播放契约测试
├─ web/soma_lab/                       # 当前 WASD 页面
├─ web/terrain_route_review/           # 源动作/模型输出离线对照
├─ web/decoder_pilot/                  # 较早的离线对照页面
├─ assets/                            # 骨架、模型及显示资源
├─ out/                               # 权重、数据、报告、动画、运行日志
│  └─ vendor/                         # PARC、GMR、kimodo 等依赖源码
├─ docs/                              # 设计、实验记录和交接；按日期核对时效
└─ patches/                           # 保存的依赖补丁；仍需核对实际加载源码
```

`data/`、`scripts/`、`out/` 内有大量历史实验，文件名存在不等于当前调用。当前工作树有已修改及未跟踪代码，不要 reset/clean，也不要假定 `motion_backbone/`、`vqvae/` 完全等于官方版本；例如本地 `motion_inference.py` 和 `motion_vqvae.py` 已有修改。

## 2. 当前有效入口

| 入口 | 审查重点 |
|---|---|
| `scripts/soma_native_bank_server.py` | 28942：校验导出身份，完整加载 Root/Pose/codec/adaptation 权重，绑定地图与执行器 |
| `scripts/soma_a100_trial_server.py`、`scripts/soma_lab_server.py` | 地图/出生点、SOMA 蒙皮、WebSocket 会话、发布与播放确认 |
| `web/soma_lab/app.js` | 键盘与相机方向、命令发送、帧缓冲、播放时钟、诊断显示 |
| `motionbricks/data/native_mb_online_conditions.py`、`native_mb_online_live.py` | 命令语义、历史/坐标转换、调用原生生成、截取实际发布帧 |
| `motionbricks/data/unified_terrain.py` | `UnifiedTerrainController`：单条原生生成链上的共享环境编码和分阶段条件注入 |
| `motionbricks/data/unified_scene_field.py`、`joint_sdf_observation.py` | 体积采样、SDF 查询及有效性；重点检查世界坐标和边界 |
| `motionbricks/data/unified_committed_playback.py`、`soma_terrain_execution.py` | 不可变已发布块的预取/提交，以及经过一致性验证的 CUDA 执行 |
| `scripts/prepare_unified_pose_cases.py`、`train_unified_native_routes.py`、`fit_native_{codec,root,pose}_fullbank.py` | 样本选择、真实窗口、分阶段训练、损失与冻结范围；花括号表示三个实际脚本 |

对照 MB 的基础入口：`motion_backbone/demo/controllers.py` 中的 `WASD_controller`、`demo/full_agent.py`、`inference/motion_inference.py`、`models/root_model.py`、`models/pose_model.py`，以上均位于内部 `motionbricks/` 包。论文：[MotionBricks](https://arxiv.org/html/2604.24833v1)，官方项目：[NVlabs MotionBricks](https://nvlabs.github.io/motionbricks/)。论文、官方发布代码、本地改动有差别时分别列明，不把本地协议说成论文必然要求。

## 3. 当前设计思路及边界

1. **保留 MB 的生成主干。** 输入可观测的四帧历史、水平移动命令和实际环境，由 Root 预测运动，再由 Pose 预测离散姿态 tokens，codec 解码为动作。当前命令采用世界 XZ 末端速度及期望朝向协议；必须与官方目标位置接口、训练源片段末端速度标签分别比较，不能默认语义完全相同。
2. **平地也是环境。** BONES 配显式平面，PARC 配对应场景，共用条件生成路径。`field_from_heightfield` 的平面快捷查询是几何算法选择，不应变成切换动作模型或抵消平地条件的分支。旧 adapter 和旧地形输入可能仍在底座中，须沿实际调用检查，不能只读新的 wrapper。
3. **固定预算的三组体积采样。** 当前默认身体周围、输入方向、历史实际速度方向各 `8×8×8` 点，共 1536 点；共享编码一次，供 Root 两阶段、Pose 和 decoder 使用。decoder 还读模型预测 Root 的相对高度变化。当前不是沿每个未来骨骼重新查询整片场景，也不是已实现完整“15 骨骼接触控制器”。
4. **三维查询不等于任意三维场景。** 当前 live 地图由高度场棱柱构造 SDF，支持查询身体上下左右空间，但不能表达悬空顶棚/洞穴。显示的全身骨骼 SDF 是诊断；实际模型输入点、骨骼中心距离与完整蒙皮碰撞指标须区分。
5. **训练与部署分层验证。** 最近采用 codec 自重建→Root→Pose 的分阶段能力修复；Pose 训练有源 Root teacher，实际 actor 必须只用自己预测的 Root/tokens。源重建、训练窗口通过、连续生成历史下的 WASD、未见地形泛化是不同证据。
6. **延迟通过执行与缓冲处理。** 预取只依赖已发布且不会再修改的模型块，在对应播放确认后提交；要验证命令时效和 stale-history 拒绝。禁止借 IK、锁脚、碰撞投影、Root 替换或输出平滑掩盖模型错误，也禁止输入未来真值高度/姿态。

## 4. 当前模型与证据

当前服务为 `http://127.0.0.1:28942/`，同模型接入 256×256 米地图、37 个出生点；地图是测试场景，未用于这轮训练。28941 为旧对照，28936 为离线查看服务，审查时保留。

- 模型包：`out/parc_rugged32_20261013_v2/native_bank_live_20261015_v1/`；先读 `export.json`，再核对 `weights.pt` 和在线 `/api/config`。当前紧凑权重 SHA256：`631d760c59d5f59bab3b795c11867f69e91c4704df0aa75270ba4c18443bed23`。旧 `soma_unified_terrain_server.py` 只加载其旧 adaptation，不等于此完整模型。
- 地图：`out/a100_live_20261010/terrain-world.npz` 及同名 JSON；JSON 指向渲染 mesh 和出生点。检查渲染与模型查询的尺寸、原点、轴向、哈希一致。
- 最近训练实际使用 1 段 PARC 上楼、1 段下楼、6 段 BONES，共 88 个重叠的 33 帧窗口。PARC/BONES 损失来源权重各 50%。最终 Pose 阶段 50 次真实更新、4400 次窗口曝光；不是 4400 条独立动作，也不是基础模型全部训练量。
- 完整记录：`out/parc_rugged32_20261013_v2/unified_native_pose_fullbank_20261015_v2-report.json`。数据准备池有 6 上楼/4 下楼/6 BONES，不能把准备池总数当作这轮实际训练数。
- 已有本地/A100、eager/CUDA、串行/预取对照及浏览器证据，索引见 `docs/native_bank_wasd_20261015.md`。用户认可视觉效果并授权试用；恒定命令、idle 起步、部分朝向平地等失败仍保留在 `docs/native_stage_repair_20261014.md` 及对应 JSON，不能因视觉认可而忽略。

## 5. 审查基本流程：先推理复现，再判断训练

1. **固定身份。** 读取正在运行的 cmdline、`/api/config`、模型 receipt、实际加载权重和代码 diff。记录源码/权重/场景/运行参数；历史端口/PID/报告仅作线索，不据此重启服务。
2. **追踪一次真实输入。** 从 WASD→世界方向与朝向→条件归一化→SDF→Root→Pose→解码→世界坐标→蒙皮→发布/ACK，逐层记录形状、单位、帧号、时间与坐标系。检查松键、反向、斜向、变速是否进入模型，前端是否加载保存的非默认滑条设置。
3. **验证数值与因果一致性。** 同四帧历史、场景、命令、权重下比较原生推理/训练验证/本地执行，优先核对离散 tokens，再比较 Root、FK、实际发布帧。默认 8 tokens、4 次 Pose argmax、4 历史帧、24 个可播放新帧、每包 8 帧、30Hz、0 path refinements；生成跨度、缓冲跨度、实际命令响应延迟分别计量。训练 future target 的变化不能泄漏到 actor 条件。
4. **复现连续闭环。** 同初态同命令比较串行与预取，检查实际已播放历史、跨块突变、输入延迟、buffer underrun 和 commit wait；平均 30fps 不能证明不卡顿。`scripts/check_native_bank_live.py` 现有对照针对包内 up/down 场景，不是完整地图覆盖证明。
5. **再验证地形与动作。** 先平地起步/行走/跑步/转向/松键，再上下楼，不同距离、朝向和速度，保留失败。记录真实前进、完整路线、左右脚交替、滑步、关节速度突变、全蒙皮穿入峰值/P99与未知几何比例；不能用停住、绕开、删帧或只报 P99 代替通过。相同历史/命令更换几何的控制实验可验证地形响应，但不能把楼梯历史称为干净平地标签。
6. **最后判断训练缺口。** 按实际索引检查源/生成历史分布、命令标签语义、步态相位、坐标转换、冻结模块和原生损失。每个修改先写可验证预测，再作最小对照。推理对齐并不证明模型泛化；发现能力不足时提出数据/训练修复，勿用运行时补丁掩盖。

本地 Python：`D:/.anaconda/envs/motionbricks/python.exe`。从子项目目录执行的轻量入口如下；本说明编写时未重新运行这些测试，也未启动训练：

```powershell
Set-Location ${PROJECT_ROOT}
$env:PYTHONPATH = '${PROJECT_ROOT};${PROJECT_ROOT}/out/vendor/kimodo'
& 'D:/.anaconda/envs/motionbricks/python.exe' -m pytest tests/test_native_bank_live_contract.py tests/test_unified_terrain.py tests/test_soma_terrain_prefetch.py tests/test_terrain_execution.py -q
& 'D:/.anaconda/envs/motionbricks/python.exe' -m scripts.check_native_bank_live --help
```

完整模型推理检查会占用本机 GPU，先检查现有负载；不要与用户正在操作的页面争抢资源。远端 A100 的不可变训练快照和本地当前源码需独立核验；不要覆盖旧快照或重启已结束的完整训练。数据、模型、动画传输继续使用已有私有 Drive/NAS/内网方式。

审查输出请保持简洁：按严重度列问题，每项附文件/行号、复现输入、观测结果、因果推导和最小修复/验证方案；明确区分已证实问题、待验证假设与训练覆盖不足。**不因已有报告或测试通过而跳过代码审查，也不以静态推理代替真实执行验证。**
