# BONES-SEED 地形拟合：阶段计划与第一步

2026-10-22。用户决定：当前阶段只用 BONES-SEED，目标是给源动作配上丰富的地形与三维环境。
本文件是计划和第一步工具的说明。本轮没有读取或修改任何动作文件，没有训练。

2026-10-23 更新：第一步已在 PRO6000 完成，结果见 [调查报告](bones_terrain_survey_pro6000_20261023.md)。当前下一步是审查抽样误报，并准备通过筛选的 BVH 子集。

## 运行位置

本阶段的开发和运行都在 PRO6000，不使用 A100 或 g1/g2/g3。g3 只是现有 BONES 源数据和 `metadata.pkl` 的存放位置；
需要的文件按项目已有的私有传输方式复制到 PRO6000 项目目录，源文件保持不变。

## 原则

1. **动作 GT 不变。** 只拟合环境，不编辑、不物理清洗、不重定向源动作。
   允许的变换只有已有流程中的整段刚体平移/朝向规范化和时间重采样；镜像是否计入单独决定。
2. **地板只能来自源动作里真实存在的支撑。** 平地动作只能配平地，或在两次落脚之间配沟、坑这类更低的几何；
   台阶、坡道、平台必须来自本身有支撑高度变化的片段。
3. **头顶和两侧几何可以覆盖所有片段。** 在身体扫掠体之外放顶棚、侧墙、横梁，不碰动作。
4. **环境必须能解释每个姿态。** 蹲、爬帧要有紧贴的遮挡；直立帧的路径上方不能有低矮几何。
   前视走廊里不能放"本该改变行为却没有改变"的几何。
5. **拟合结果是构造的环境，不是采集现场的扫描。** 记录来源、阈值和失败，不把弱证据写成测量结果。

## 已有可复用部分

| 环节 | 位置 |
| --- | --- |
| BVH 解析、SOMA77→SOMA30、FK/roundtrip 检查、标定姿态裁剪 | `scripts/prepare_bones_locomotion.py` |
| 官方蒙皮 sole 顶点 | `motionbricks/data/bones_seed.py`（`official_sole_surfaces`） |
| 平地固定支撑平面与接触标签 | `bones_seed.estimate_contacts` |
| 由静止 sole 顶点反推水平支撑面、分层、源经过空间不能是实体 | `motionbricks/data/stair_support.py`（`reconstruct_support`） |
| 场景表示与查询 | `SceneManifest`（`scene_geometry_field.py`）与 `CompositeSceneField`（`pfnn_3d_scene.py`） |
| 录制家族划分 | `bones_seed.recording_group` 加 annotation 传播并查集 |

已知限制：`reconstruct_support` 在楼梯模式下要求至少 3 个高度层，没有坡道，只用脚底；
此前只处理了 A300/A301 的 34 条 `stairs_climbing`，21 条通过（见 [Native MB Stair Skill](native_mb_stair_skill_20261007.md)）。

## 阶段步骤

| 步骤 | 内容 | 进入下一步的依据 |
| --- | --- | --- |
| 1. 候选摸底 | 只读标注，按"支撑高度变化 / 手或障碍支撑 / 低姿态"分桶，统计条数、时长、家族、演员 | 报告各桶规模，人工抽查关键词误报 |
| 2. 几何筛查 | 对候选 BVH 算支撑高度跨度、手膝着地比例、骨盆高度分布，核对第 1 步的标签 | 每桶有可信的实际几何证据量 |
| 3. 地板拟合 | 平面 / 参数化图元（台阶、平台、坡道、箱体）/ 不规则时退回 patch+RBF；推广 `reconstruct_support` 到单级台阶、坡道和手膝支撑 | 接触点到支撑面距离、源经过空间零穿透 |
| 4. 头顶与侧面几何 | 身体扫掠体之外的顶棚、侧墙、横梁；解释性约束 | 蹲爬帧解释率、直立帧零低矮遮挡 |
| 5. 变体与观测 | 每个窗口多份可行几何变体，输出 `SceneManifest`，用运行时同一查询生成观测 | 硬约束校验、分布统计、覆盖矩阵 |

## 第一步：候选摸底工具

- 模块：[bones_terrain_survey.py](../../motionbricks/data/bones_terrain_survey.py)，只依赖标准库；读 parquet/pickle 时需要 pandas。
- 入口：[survey_bones_terrain_candidates.py](../../scripts/survey_bones_terrain_candidates.py)。
- 测试：[test_bones_terrain_survey.py](../../tests/test_bones_terrain_survey.py)。

输入是官方时间轴 jsonl，可选官方元数据表和 BVH 根目录（只读文件名）。
有元数据时，`content_vertical_move=1` 计入支撑高度变化，`content_body_position` 为低姿态值时计入低姿态；
没有元数据时只用文件名、概述和分段描述的关键词。每个命中都记录来源字段，便于审查。

输出目录必须是新目录：

- `survey_report.json`：各桶条数、原始/镜像分开、原始时长、命中分段时长、家族数、演员数、
  人工/传播标注比例、高频命中词、各桶组合，以及输入文件哈希和全部规则。
- `candidates.jsonl`：每条候选一行，含命中证据、分段时长、上下文标记（坐姿、舞蹈/格斗、手持物、跌倒）。

状态：已在 142,220 条真实 timeline 上运行，得到 19,935 条去重候选（9,969 原始、9,966 镜像）。
PRO6000 测试为 `7 passed, 1 skipped in 2.16s`；[90 条固定抽样](../../examples/results/bones_terrain_survey_20261023/sample_candidates_seed0.txt) 已发布。
抽样暴露了搬箱子、挠头和揉肚子等误命中，下一步先审核规则。几何筛选尚未执行。

### 元数据现状

官方 `seed_metadata_v003.parquet` 目前在 PRO6000、g3 的 BONES 源库和项目目录都没有找到，NAS 尚未查完。
g3 的 SMPL 过滤版 `metadata.pkl`（约 8.37 MiB）已经私有 Drive 复制到 PRO6000，双端哈希一致。
实际为 131,454 行，只有 `filename`、`duration`、`fps`、`length` 四列；缺少所需内容标签，已按条件跳过带 metadata 的调查。
完整列清单见 [pkl_describe.json](../../examples/results/bones_terrain_survey_20261023/pkl_describe.json)。
pickle 加载会执行代码，只用于自己信任的文件。没有元数据时，第一步只依赖时间轴关键词。

PRO6000 可访问的数据目录中尚未找到 BONES 动作 BVH，本轮省略 `--bvh-root`。几何筛选需要迁入确认的候选子集。

运行示例：

```bash
python scripts/survey_bones_terrain_candidates.py --describe-metadata --metadata /path/to/metadata.pkl
python scripts/survey_bones_terrain_candidates.py \
    --timelines /path/to/seed_timelines.jsonl \
    --bvh-root /path/to/soma_uniform/bvh \
    --output runs/bones_terrain_survey_v1
```

## 第一步审查（2026-10-23）

依据 [PRO6000 调查结果](bones_terrain_survey_pro6000_20261023.md) 和其中每类 30 条的固定种子抽样，人工逐条判读：

| 类别 | 抽样中相关 | 主要误报 | 说明 |
| --- | ---: | --- | --- |
| 支撑高度变化 | 12/30 | `crate`（搬箱行走 15 条）、放物到 platform、跛行 uneven、撞到 obstacle | `crate` 在该桶占约 2,216/5,430 条；去掉搬箱行走后抽样 12/15 相关 |
| 手或障碍支撑 | 14/30 | `hand(s) on the`（挠头、背手行走、坐桌边） | `climbs` 主要来自上 50cm 箱子；手是否承重需要几何判断 |
| 低姿态 | 24/30 | `belly`（揉肚子）、`hunched`（受伤含胸）、`bent over`、跌倒 `lying` | `mohak_*` 是手膝着地爬行家族，带前进/后退/起停 |

以上是 90 条抽样的判读，不是全量精度。结论：

1. **关键词已经完成估计规模的作用，不再继续调关键词。** 哪些片段进入拟合，由第二步的几何证据决定。
2. **BONES 文件名是结构化的**（`<动作>_<方向/起停/角度>_<take>__A<演员>[_M]`）。按动作词干汇总比抽样更可审：
   v2 增加 `stems.tsv`，每个词干一行，含各类命中比例、原始/镜像条数、家族、演员和示例描述，
   未命中任何类别的词干也列出，用于发现标注没写出的支撑变化。
3. **文件名里有已知尺寸**，如 `come_up_50cm_box`、`jump_on_50cm`、`jump_over_obstacle_0_75m`。
   这些片段可作为拟合的外部校验：拟合出的支撑高度应接近文件名尺寸。v2 把尺寸词单独列出。
4. **第二步需要动作文件在 PRO6000 上。** 顶棚和侧面几何适用于所有移动片段，几何筛查也应覆盖未命中关键词的片段，
   因此建议迁移全部原始（非镜像）SOMA Uniform BVH，而不是只迁候选子集。

## 第二步：几何筛查（2026-10-23）

### v2 词干审查

[stems.tsv](../../examples/results/bones_terrain_survey_v2_20261023/stems.tsv) 共 2,863 个词干，812 个有命中。
按词干名和描述逐个审查与地形、姿态相关的词干后，挑出下面的组，写入
[bones_terrain_screen_v1.json](../../configs/bones_terrain_screen_v1.json)。条数是原始片段数（不含镜像）。

| 组 | 原始片段 | 代表词干 | 用途 |
| --- | ---: | --- | --- |
| 已知尺寸支撑 | 1,760 | `come_up_50cm_box`（984）、`come_down_50cm_box`（597）、`jump_on_1m`、`jump_on_box_230cm` | 拟合支撑高度，并与文件名尺寸对照 |
| 箱体支撑（无尺寸） | 48 | `box_jump`、`box_jump_down` | 支撑高度变化 |
| 楼梯 | 49 | `stairs_climbing_up/down/jog_*` | 多级支撑；数量少 |
| 越障 | 86 | `jump_over_obstacle_0.5m…2m`、`vault_over`、`monkey_jump` | 障碍高度对照最低身体净空；手可能撑在障碍上 |
| 梯子 | 64 | `ladder_climbing_*` | 只筛查，不进第一版拟合 |
| 蹲 | 1,145 | `crouch`（817，多方向蹲走）、`turn_crouch`、`idle_crouch*` | 顶棚候选 |
| 手膝着地 | 833 | `mohak_*`（前进、后退、转向、起停）、`on_all_fours_*` | 顶棚候选，手膝支撑 |
| 匍匐 | 919 | `crawl`（600）、`turn_crawl`、`idle_crawl*` | 最低顶棚候选 |
| 跪 | 638 | `kneeling` | 静态低姿态 |
| 靠墙 | 353 | `wall_leaning_*`、`h_b_w_*`（墙后掩护蹲起，手扶墙） | 侧墙候选，有身体接触证据 |
| 平地对照 | 900 | `walk`、`jog`、各方向与弧线，每个词干最多 60 条 | 检查筛查把平地误判成多级的比例 |

明确不选：搬箱行走 `lift_crate_walk`、撞小障碍 `bump_into_small_obstacle_*`（是踢到物体，不是支撑）、
推障碍 `push_obstacle`（物体在动）、阀门/拉杆、舞蹈、原地跳、健身动作、晕倒/跌倒。

### 筛查内容

模块 [bones_terrain_screen.py](../../motionbricks/data/bones_terrain_screen.py)，入口
[screen_bones_terrain_geometry.py](../../scripts/screen_bones_terrain_geometry.py)。转换沿用楼梯数据的官方路径：
Kimodo BVH、30Hz SLERP、SOMA77 FK、官方蒙皮每只脚 98 个 sole 顶点。动作不改，结果都在源世界坐标。

| 输出 | 来源 |
| --- | --- |
| 支撑层 | `stair_support.reconstruct_support` 的候选面（静止、局部最低、源动作没穿过），按 3cm 聚类成层；不使用它的"至少三层"成对判定 |
| 手、膝、肘支撑 | 掌心（腕与中指末端中点）、膝（Shin）、肘（ForeArm）静止段，是否落在某个支撑层上 |
| 姿态 | 每帧按"脚所在支撑层"之上的髋高比例、躯干俯仰、胸高和手是否着地分为站、蹲、手膝、匍匐、其他低姿态 |
| 头顶净空 | 每种姿态下头关节到支撑层的高度分位数，后续顶棚的下界依据 |
| 尺寸校验 | 有尺寸的词干：支撑层跨度减尺寸；最低身体净空减尺寸（越障） |

站立髋高参考与 `bones_seed.validate_planar_posture` 相同（Uniform 骨架，全体共用）。所有阈值是候选值，写入报告。

输出目录：`selection.jsonl`、`clips.jsonl`（完整逐条结果，留在 PRO6000）、`clips_compact.tsv`（每条一行，可公开）、
`by_stem.json`、`by_group.json`、`report.json`（阈值、选择、失败原因、源码哈希）。

状态：已用合成数据单元测试；尚未在真实 BVH 上运行。先每个词干 2 条做试跑，检查失败原因后再全量。

## 待决定

1. **Uniform 或 Proportional。** 已有 sole 顶点来自官方 `skin_standard.npz`，与 Uniform 骨架一致。
   建议第一版沿用 Uniform；Proportional 需要按演员重新蒙皮并标定 sole，作为后续对照。
2. **帧率。** 建议地形拟合用 30Hz，与楼梯数据、PFNN 和逐帧 actor 一致。
3. **镜像。** 镜像是否算改动 GT；摸底报告已把原始和镜像分开统计。
