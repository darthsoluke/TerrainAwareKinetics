# BONES-SEED 地形候选调查：PRO6000 执行结果

日期：2026-10-23，Asia/Singapore。调查在 PRO6000 上完成，源码版本 `f2d4c5caed62c6efc56a9efb93c65382f18183fe`。

代码入口：[调查脚本](../../scripts/survey_bones_terrain_candidates.py)、[分类模块](../../motionbricks/data/bones_terrain_survey.py)、[已有测试](../../tests/test_bones_terrain_survey.py)。下一阶段的范围见 [地形拟合计划](bones_terrain_fitting_plan_20261022.md)。

本次读取 142,220 条 timeline，选出 19,935 条至少命中一个类别的候选：9,969 条原始动作、9,966 条镜像，涉及 927 个家族。原始候选片段的 timeline 时长合计 77,145.009 秒，约 21.43 小时。

这是标注调查的产物。下一步需要人工审查下方抽样，再对确认的候选做 BVH 几何筛选。

## 1. 执行环境与测试

| 项目 | 实际值 |
| --- | --- |
| 执行主机 | PRO6000 |
| 工作目录 | `${TAK_ROOT}` |
| Python | `${PFNN_PROJECT}/env/bin/python` |
| Python / pandas / pytest | 3.11.16 / 2.3.3 / 9.1.1 |
| GPU | `CUDA_VISIBLE_DEVICES` 为空；没有启动 GPU 作业或训练 |
| 调查运行时源码状态 | `git status --porcelain` 为空，没有修改调查代码；本报告随后按用户要求发布 |

执行命令：

```bash
python -m pytest tests/test_bones_terrain_survey.py -q
```

实际最后一行：

```text
7 passed, 1 skipped in 2.16s
```

完整输出：[tests.txt](../../examples/results/bones_terrain_survey_20261023/tests.txt)。一项需要完整动作处理依赖的兼容性测试被跳过，符合任务包允许的情况。

## 2. 输入位置与双端校验

两份输入都从 g3 经项目私有 Google Drive 复制到 PRO6000。没有复制 BVH 语料。

公开文档使用路径占位符：`${TAK_ROOT}` 是项目根目录，`${BONES_INPUTS}` 是本次输入目录，`${PFNN_PROJECT}` 是既有独立环境所在项目；`${G3_TIMELINE_SOURCE}` 和 `${G3_METADATA_SOURCE}` 指各自的源文件目录。实际路径保存在私有交付包，输入文件身份由下方哈希确定。

| 文件 | 大小 | g3 源路径 | PRO6000 使用路径 |
| --- | ---: | --- | --- |
| timeline | 80,373,523 bytes | `${G3_TIMELINE_SOURCE}/timelines.jsonl` | `${BONES_INPUTS}/timelines.jsonl` |
| pickle metadata | 8,771,887 bytes | `${G3_METADATA_SOURCE}/metadata.pkl` | `${BONES_INPUTS}/metadata.pkl` |

SHA256：

```text
timelines.jsonl
g3:      379d6a5b86cea06b7201d485d19ee53512cc58449352b3cf113a95d1d27603d8
PRO6000: 379d6a5b86cea06b7201d485d19ee53512cc58449352b3cf113a95d1d27603d8

metadata.pkl
g3:      8104cce0cbd4387c129b595551c4e85e210e9db13f6c9275eec65a29abb08bbc
PRO6000: 8104cce0cbd4387c129b595551c4e85e210e9db13f6c9275eec65a29abb08bbc
```

两个文件均匹配。timeline 开头已检查，包含 `filename`、`overview_description`、`events`。

PRO6000 的 `${USER_HOME}` 和 `/data` 可访问目录中没有找到 BONES SOMA Uniform 动作 BVH。找到的七个 BVH 是 SOMA 骨架模板；另有 35 个目录返回权限不足。因此本轮没有可用的 BVH 根目录与动作计数，运行时省略 `--bvh-root`。报告中的 `with_bvh: 0` 对应未做动作文件匹配。

扫描记录仍保存在 PRO6000：

```text
${BONES_INPUTS}/discovery_20261023.RuzzM6/paths.tsv
${BONES_INPUTS}/discovery_20261023.RuzzM6/errors.txt
```

## 3. Pickle 检查的完整输出

执行命令：

```bash
python scripts/survey_bones_terrain_candidates.py --describe-metadata \
  --metadata ${BONES_INPUTS}/metadata.pkl \
  > ${BONES_INPUTS}/pkl_describe.json
```

完整原文件：[pkl_describe.json](../../examples/results/bones_terrain_survey_20261023/pkl_describe.json)。内容如下：

```json
{
  "rows": 131454,
  "columns": {
    "duration": 131454,
    "filename": 131454,
    "fps": 131454,
    "length": 131454
  },
  "examples": {
    "duration": "12.18",
    "filename": "jump_and_land_heavy_001__A001",
    "fps": "50.0",
    "length": "609"
  },
  "official_columns_present": [
    "filename"
  ],
  "official_columns_missing": [
    "move_name",
    "move_duration_frames",
    "package",
    "category",
    "is_mirror",
    "take_actor",
    "actor_uid",
    "content_name",
    "content_type_of_movement",
    "content_body_position",
    "content_vertical_move",
    "content_horizontal_move",
    "content_props",
    "content_uniform_style",
    "content_natural_desc_1",
    "content_natural_desc_2",
    "content_natural_desc_3",
    "content_natural_desc_4",
    "content_technical_description",
    "content_short_description",
    "content_short_description_2"
  ]
}
```

Step 7 已跳过：虽然存在 `filename`，但 `content_vertical_move` 和 `content_body_position` 都缺失。没有生成带 metadata 的第二份 survey。

## 4. Timeline 调查与原始汇总表

执行命令：

```bash
python scripts/survey_bones_terrain_candidates.py \
  --timelines ${BONES_INPUTS}/timelines.jsonl \
  --output runs/bones_terrain_survey_v1_timeline
```

退出码为 0。完整标准输出：[timeline_summary.txt](../../examples/results/bones_terrain_survey_20261023/timeline_summary.txt)。打印的表格：

```text
bones-terrain-survey-v1: wrote runs/bones_terrain_survey_v1_timeline
clips: {'total': 142220, 'timeline_only': 142220, 'metadata_only': 0, 'both': 0}
bucket                         clips    orig  mirror      orig_s   event_s  families  actors
support_height_change           5430    2715    2715     20664.8   14237.1       310     346
hand_or_obstacle_support        2730    1365    1365     10561.8    4600.8       135     348
low_posture                    12509    6256    6253     50095.7   37445.1       586     380
any_candidate                  19935    9969    9966     77145.0       0.0       927     496
flat_remainder                122285   61163   61122    435575.1       0.0      3411     518
```

`orig_s` 是非镜像候选片段的总时长，按 timeline 最后一个事件的结束时间计。`event_s` 是该类别内命中事件的时间总和；`any_candidate` 和 `flat_remainder` 两行没有计算这个字段，打印器显示为 0.0。各类别可以重叠，去重后的合计使用 `any_candidate`。

原始完整 JSON：[survey_report.json](../../examples/results/bones_terrain_survey_20261023/survey_report.json)。PRO6000 原始位置：

```text
${TAK_ROOT}/runs/bones_terrain_survey_v1_timeline/survey_report.json
${TAK_ROOT}/runs/bones_terrain_survey_v1_timeline/candidates.jsonl
```

## 5. 固定种子抽样：每类 30 条

完整抽样文件：[sample_candidates_seed0.txt](../../examples/results/bones_terrain_survey_20261023/sample_candidates_seed0.txt)。文件共 93 行：三个类别标题，各跟随 30 条候选。保留了用户指定的 `filename | evidence | overview[:120]` 格式，按每类分别执行 `random.Random(0).shuffle(sel)`。

抽样中已有具体的关键词误命中，供 Claude 和人工审核时优先检查：

| 类别 | 抽样中的例子 | 触发原因与需要区分的含义 |
| --- | --- | --- |
| 支撑高度变化 | `lift_crate_walk_ff_loop_180_R_003__A143_M` | `crate` 命中搬箱子行走，文本没有描述脚下支撑高度变化 |
| 支撑高度变化 | `injured_R_leg_walk_ff_stop_270_003__A173_M` | `uneven` 描述跛行的步子，文本没有描述崎岖地面 |
| 手部或障碍支撑 | `itching_head_R_003__A047_M` | `hand on the` 命中挠头 |
| 手部或障碍支撑 | `walk_hands_on_back_stop_004__A095_M` | `hands on the` 命中双手放在背后 |
| 低姿态 | `tasty_R_003__A270` | `belly` 命中揉肚子 |

调查产物保留原规则和全部命中；本次没有修改规则、删样本或运行几何筛选。

## 6. 关于数据完整性，给 Claude 的补充

PRO6000 已有经迁移哈希校验的 PFNN 数据、缓存和运行制品，位置是：

```text
${PFNN_PROJECT}/transfers/assets_v4/data
```

本次又补齐了运行 BONES 标注调查所需的 timeline 与这份 pickle。timeline 记录了 71,132 条原始名称和 71,088 条镜像名称，合计 142,220 条；BONES 动作 BVH 没有在本次 PRO6000 可访问目录扫描中找到。两者的下一步是：先审查 90 条样本和候选规则，再将确认的候选 BVH 子集从现有来源迁到 PRO6000，以便几何筛选。

## 7. 文件与可追溯性

| 文件 | 内容 |
| --- | --- |
| [survey_report.json](../../examples/results/bones_terrain_survey_20261023/survey_report.json) | 调查原始完整报告，含计数、时长、关键词、规则和输入哈希 |
| [sample_candidates_seed0.txt](../../examples/results/bones_terrain_survey_20261023/sample_candidates_seed0.txt) | 每类 30 条，共 90 条抽样 |
| [pkl_describe.json](../../examples/results/bones_terrain_survey_20261023/pkl_describe.json) | metadata 全部列、示例及缺失字段 |
| [execution_receipt.json](../../examples/results/bones_terrain_survey_20261023/execution_receipt.json) | PRO6000 路径、环境、提交、双端 SHA256、条件步骤状态 |
| [timeline_summary.txt](../../examples/results/bones_terrain_survey_20261023/timeline_summary.txt) | 调查标准输出 |
| [tests.txt](../../examples/results/bones_terrain_survey_20261023/tests.txt) | 本轮 PRO6000 测试标准输出 |
| [SHA256SUMS](../../examples/results/bones_terrain_survey_20261023/SHA256SUMS) | 上述六个公开结果文件的哈希 |
| [publication_manifest.json](../../examples/results/bones_terrain_survey_20261023/publication_manifest.json) | 原始文件与公开副本的哈希对应 |

这些结果由 PRO6000 经私有 Drive 取回，六个文件的本地 SHA256 全部匹配。公开副本只替换私人机器路径，数值、输入哈希、规则和抽样内容保持原样。报告包的 PRO6000 目录是 `${TAK_ROOT}/runs/bones_terrain_survey_delivery_20261023`。

调查执行时解决了 rclone 不在 PATH 和旧共享 OAuth 客户端额度耗尽的问题；任务私有配置使用已有独立客户端。调查代码未改，未安装或修改 cloth 环境，未训练、未占用 GPU。运行结束后关闭了 SSH；报告和源码按用户后续要求同步到 GitHub。
