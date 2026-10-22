# BONES-SEED 地形拟合：阶段计划与第一步

2026-10-22。用户决定：当前阶段只用 BONES-SEED，目标是给源动作配上丰富的地形与三维环境。
本文件是计划和第一步工具的说明。本轮没有读取或修改任何动作文件，没有训练。

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

状态：已用合成标注做单元测试，**尚未在真实 BONES-SEED 标注上运行**。关键词是候选规则，
命中不等于有支撑变化，未命中也不等于平地。

### 元数据现状

官方 `seed_metadata_v003.parquet` 目前在 PRO6000、g3 的 BONES 源库和项目目录都没有找到，NAS 尚未查完。
g3 有一份 SMPL 过滤版本的 `metadata.pkl`（约 8.37 MiB），结构未知。先用
`--describe-metadata --metadata <pkl>` 查看列名，确认是否含官方列，再决定能否作为元数据输入。
pickle 加载会执行代码，只用于自己信任的文件。没有元数据时，第一步只依赖时间轴关键词。

运行示例：

```bash
python scripts/survey_bones_terrain_candidates.py --describe-metadata --metadata /path/to/metadata.pkl
python scripts/survey_bones_terrain_candidates.py \
    --timelines /path/to/seed_timelines.jsonl \
    --bvh-root /path/to/soma_uniform/bvh \
    --output runs/bones_terrain_survey_v1
```

## 待决定

1. **Uniform 或 Proportional。** 已有 sole 顶点来自官方 `skin_standard.npz`，与 Uniform 骨架一致。
   建议第一版沿用 Uniform；Proportional 需要按演员重新蒙皮并标定 sole，作为后续对照。
2. **帧率。** 建议地形拟合用 30Hz，与楼梯数据、PFNN 和逐帧 actor 一致。
3. **镜像。** 镜像是否算改动 GT；摸底报告已把原始和镜像分开统计。
