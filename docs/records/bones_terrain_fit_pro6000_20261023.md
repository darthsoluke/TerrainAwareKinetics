# BONES 支撑几何拟合与可视化：PRO6000 执行记录

2026-10-23，运行源码 `50c1bf175626383a449835a01ab1022e36d72631`。

**试运行和全量执行均已完成。** 试运行拟合 196 条、渲染 196 条，耗时 161.7 秒；全量拟合 6,788 条、渲染 221 条，耗时 585.5 秒。两轮的处理失败数和渲染失败数均为 0。几何指标中，全量有 2,520 条触发穿透报告阈值，分组结果与最严重样例一起发布，供下一步审查。

入口：[结果与示例图](../../examples/results/bones_terrain_fit_v1_20261023/README.md)、[全部接触图](../../examples/results/bones_terrain_fit_v1_20261023/full/viz/README.md)、[6,788 行紧凑表](../../examples/results/bones_terrain_fit_v1_20261023/full/fits_compact.tsv)。

## 1. 环境与输入

全部拟合和渲染在 PRO6000 上执行，CPU only，16 个 worker；沿用几何筛查时的独立 Python 环境、Kimodo 快照和线程设置：

```bash
export PY=${PFNN_PROJECT}/env/bin/python
export PYTHONPATH="${KIMODO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "${TAK_ROOT}"
```

没有安装新包、修改源 BVH 或执行训练。环境版本及输入文件 SHA256 见 [environment.json](../../examples/results/bones_terrain_fit_v1_20261023/execution/environment.json)。

输入是已完成的 `runs/bones_screen_v1_full`，筛查源码为 `0f97409`。该运行共有 6,795 条结果，其中 7 条因源骨架偏差超过 2 mm 被拒绝；拟合入口按规则只处理剩余 6,788 条。试运行在这些成功条目中按文件名 SHA256 排序，每词干取两条，共 98 个词干、196 条。

## 2. 测试与导入检查

```bash
$PY -m pytest tests/test_bones_terrain_fit.py tests/test_bones_terrain_screen.py \
  tests/test_bones_terrain_survey.py -q
$PY -c "import matplotlib, PIL; print(matplotlib.__version__, PIL.__version__)"
```

完整输出：

```text
.........................                                                [100%]
25 passed in 5.01s

3.11.2 12.3.0
```

对应 Matplotlib 3.11.2、Pillow 12.3.0。原始文件：[tests.txt](../../examples/results/bones_terrain_fit_v1_20261023/execution/tests.txt)、[imports.txt](../../examples/results/bones_terrain_fit_v1_20261023/execution/imports.txt)。

## 3. 实际执行命令

试运行：

```bash
$PY -u scripts/fit_bones_terrain_support.py \
  --screen-run runs/bones_screen_v1_full \
  --output runs/bones_fit_v1_pilot \
  --limit-per-stem 2 --viz-per-stem 2 --gif --workers 16
```

试运行报告：`failed=0`、`failures=[]`，`rendered={"clips":196,"ok":196,"failed":[]}`，耗时 `161.7` 秒。完整 `rendered.reasons` 见 [pilot/report.json](../../examples/results/bones_terrain_fit_v1_20261023/pilot/report.json)。处理失败率为 0%，通过 5% 的阶段门后启动全量。

全量：

```bash
nohup "$PY" -u scripts/fit_bones_terrain_support.py \
  --screen-run runs/bones_screen_v1_full \
  --output runs/bones_fit_v1_full \
  --viz-per-stem 2 --worst-per-group 3 --gif --workers 16 \
  > runs/bones_fit_v1_execution/full.stdout.txt 2>&1 < /dev/null &
```

实际命令外层使用 `bash -c` 记录退出状态，两轮均为 `FIT_EXIT=0`。后台任务已结束。

全量报告：`failed=0`、`failures=[]`，`rendered={"clips":221,"ok":221,"failed":[]}`，总耗时 `585.5` 秒。日志显示拟合阶段约 425 秒，其余时间用于重新读取审查片段并生成图、GIF 和 viewer 数据。

| 运行 | 拟合 | 处理失败 | 渲染 | 渲染失败 | 总耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 试运行 | 196 | 0 | 196 | 0 | 161.7 秒 |
| 全量 | 6,788 | 0 | 221 | 0 | 585.5 秒 |

打印的完整分组 JSON：[试运行](../../examples/results/bones_terrain_fit_v1_20261023/pilot/by_group.json)、[全量](../../examples/results/bones_terrain_fit_v1_20261023/full/by_group.json)。

## 4. 几何指标

拟合保留源动作的世界坐标，为最低支撑层建立地面板，为更高支撑层建立实心箱体。`minimal` 包住支撑证据并留出余量，`maximal` 在身体活动空间允许的范围内扩大边界。

穿透检查使用每只脚 98 个蒙皮点和 12 个关节球。代码先从深度中扣除 2 cm 容差，再统计剩余深度超过 2 cm 的帧；下表沿用这一口径。P95 是各片段最大深度的 95% 分位数。本轮 minimal 与 maximal 在表中这两项统计上相同。

| 组 | 试运行触发片段 / 拟合数 | 全量触发片段 / 拟合数 | 全量最大深度 P95（m） |
| --- | ---: | ---: | ---: |
| 已知尺寸支撑 `known_dimension_support` | 4 / 20 | 32 / 1,760 | 0.0004 |
| 箱体支撑 `box_support` | 4 / 8 | 22 / 48 | 0.5966 |
| 楼梯 `stairs` | 4 / 12 | 21 / 49 | 0.7385 |
| 越障 `obstacle_clearance` | 1 / 18 | 4 / 86 | 0.0156 |
| 梯子 `ladder` | 7 / 20 | 16 / 64 | 0.7135 |
| 蹲姿 `crouch` | 3 / 32 | 36 / 1,143 | 0.0091 |
| 手膝着地 `quadruped` | 24 / 24 | 830 / 831 | 0.1298 |
| 匍匐 `prone` | 14 / 14 | 904 / 919 | 0.1261 |
| 跪姿 `kneel` | 2 / 2 | 619 / 638 | 0.0962 |
| 靠墙 `wall_contact` | 4 / 16 | 36 / 350 | 0.0694 |
| 平地对照 `flat_control` | 0 / 30 | 0 / 900 | 0.0020 |

各组的 `unsupported_evidence_points` 均为 0。低姿态组的大量穿透标记，以及箱体、楼梯、梯子的严重个例，是下一步人工审查的重点。图片同时显示源骨架、最小与最大箱体，便于逐例检查支撑证据和身体空间是否相容。

已抽看的最严重样例 `jump_to_shoulder_roll_R_003__A305`，最大穿透指标为 `0.8124 m`，127 帧超过报告阈值。[接触图](../../examples/results/bones_terrain_fit_v1_20261023/full/viz/box_support/jump_to_shoulder_roll_R_003__A305.png) 中，拟合箱体覆盖了落地翻滚所占空间；该片段也包含在公开 viewer 中。

## 5. 演员尺度

[actor_scale.json](../../examples/results/bones_terrain_fit_v1_20261023/full/actor_scale.json) 用上/下 50 cm 箱体片段估计每个演员在 SOMA Uniform 坐标中的尺度，再检查其他带尺寸词干。

- 取得 359 个演员的尺度估计；P5 / P50 / P95 为 `1.04031 / 1.14434 / 1.23580`。
- 至少四条参考片段的演员，其组内高度标准差中位数为 `0.0011243 m`，约 1.12 mm。
- 其他 50 cm 跳上/跳下词干，校准后的尺寸比值中位数约为 `0.9981–0.9996`。
- `jump_on_1m` 为 `0.96835`，`jump_off_1m` 为 `0.91953`，`jump_on_box_230cm` 为 `0.92627`。这些词干在演员尺度校准后仍有偏差，报告保留逐词干统计。

比值定义为“检测到的最高支撑层 /（文件名尺寸 × 演员尺度）”。越障动作的支撑层通常来自起落地，障碍高度应结合净空检查；`actor_scale.json` 中这类词干的数值也原样保留。

## 6. 审查图与 viewer 选样

全量生成 196 条按词干抽取的样本，加上 25 条额外的 `worst_penetration` 样例，共 221 条。每条均有 PNG、GIF 和完整 viewer 数据。

PNG 共 `33,887,479 bytes`（33.89 MB），低于 40 MB，因此全部 221 张发布，保留 11 个组目录，没有删图。目录和数量见 [viz/README.md](../../examples/results/bones_terrain_fit_v1_20261023/full/viz/README.md)。

公开 viewer 共 38 条，包含全部 25 条 `worst_penetration`，每个组至少一条来自 sample 的片段，以及以下最低数量要求：

| 词干 | 要求条数 |
| --- | ---: |
| `come_up_50cm_box` | 2 |
| `jump_on_box_230cm` | 1 |
| `stairs_climbing_up` | 2 |
| `stairs_climbing_down` | 1 |
| `mohak_forward` | 2 |
| `crawl` | 2 |
| `crouch` | 2 |
| `kneeling` | 1 |
| `h_b_w_stand_to_crouch` | 1 |

选样先保留全部 worst，再补足指定词干，最后补足每组 sample；候选按文件名 SHA256 排序，已有条目复用。完整名单、选择原因及数量见 [publication_selection.json](../../examples/results/bones_terrain_fit_v1_20261023/publication_selection.json)。[viewer/index.json](../../examples/results/bones_terrain_fit_v1_20261023/full/viewer/index.json) 只列这 38 个 JSON。

已在 PRO6000 解码检查全部 221 张 PNG、检查全部 221 个多帧 GIF，核对 38 个 viewer 的索引、关节数、整数毫米坐标和骨段索引。PNG 均为 1320×572。另抽看了上箱、楼梯、手膝及匍匐的三维和侧视图。检查记录见 [artifact_checks.json](../../examples/results/bones_terrain_fit_v1_20261023/execution/artifact_checks.json)。

## 7. 服务器产物和发布

`${TAK_ROOT}` 为 PRO6000 上的仓库目录。完整产物保留在：

| 产物 | PRO6000 路径 |
| --- | --- |
| 试运行报告与全部拟合 | `${TAK_ROOT}/runs/bones_fit_v1_pilot/` |
| 全量 6,788 个拟合 JSON | `${TAK_ROOT}/runs/bones_fit_v1_full/fits/<filename>.json` |
| 全量 221 个 GIF | `${TAK_ROOT}/runs/bones_fit_v1_full/viz/<group>/<filename>.gif` |
| 全量 PNG | `${TAK_ROOT}/runs/bones_fit_v1_full/viz/<group>/<filename>.png` |
| 完整 viewer 集合 | `${TAK_ROOT}/runs/bones_fit_v1_full/viewer/` |
| 原始 stdout 与测试日志 | `${TAK_ROOT}/runs/bones_fit_v1_execution/` |

GitHub 发布报告、紧凑 TSV、全部 PNG、指定 38 条 viewer、测试与环境记录。GIF 和 `fits/*.json` 保留在 PRO6000。公开包经私有 Drive 传输，逐文件 SHA256 核对通过后再写入仓库；私人路径替换为角色占位符，PNG、viewer 坐标与科学数值保持原样。

仓库检查器增加了结果目录 PNG 的格式检查、单文件 3 MiB 上限及每个结果目录 PNG 合计 40 MB 上限。拟合和渲染算法保持运行版本；本次发布没有调整动作或阈值。
