# BONES 地形方法 v2：v3 分片执行结果

实际运行日期：**2026-10-24，Asia/Singapore**。文件名沿用任务指定的 `20261023`。源码固定为 `f49e5abae96a29edf8249751405392ba2740d624`，方法见 [BONES 动作—环境配对方法 v2](bones_terrain_method_v2_20261023.md)。

**运行和发布已完成，方法质量门尚未通过。** 主任务成功处理 6,788 条动作，生成 27,152 个场景变体，其中 23,590 个通过非地面穿透门，通过率 86.8813%。成功拟合的 2,904 条顶棚在本次蒙皮采样中穿透为零；777 条拟合墙体出现超过 2 cm 的穿透。另完成 120 条 `on_the_edge` 检查。

结果入口：[报告、图片与 viewer](../../examples/results/bones_terrain_fit_v3_20261023/README.md)。全部运算使用 PRO6000 CPU，源 BVH 和此前运行产物保持不变，没有训练。

## 1. 测试、站立高度与运行环境

沿用筛查的独立 Python 和 Kimodo 快照：

```bash
export PY=${PFNN_PROJECT}/env/bin/python
export PYTHONPATH="${KIMODO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "${TAK_ROOT}"

$PY -m pytest tests/test_bones_terrain_fit.py tests/test_bones_terrain_scene.py \
  tests/test_bones_terrain_screen.py tests/test_bones_terrain_survey.py -q
```

结果为 **`38 passed in 5.54s`**。实际命令另加 `--junitxml` 保存逐测试记录，确认 `test_skin_vertices_uses_official_skinner_interface` 已运行且未跳过。日志：[tests_updated.txt](../../examples/results/bones_terrain_fit_v3_20261023/execution/tests_updated.txt)、[skinner_test.txt](../../examples/results/bones_terrain_fit_v3_20261023/execution/skinner_test.txt)。

站立头顶检查原样输出：

```text
1.760514736175537
```

单位为米，位于 1.6–1.95 m 范围内。[环境记录](../../examples/results/bones_terrain_fit_v3_20261023/execution/environment.json) 是首轮 16-worker 运行前的快照，保存包版本、输入文件及蒙皮资源哈希；后续实际 worker 数见各任务回执。首次远端拉取遇到 TLS 中断，旧代码下新增测试文件未找到；重试更新成功后才执行上述 38 项测试及后续拟合。

## 2. 试运行和四路执行

用户切换为四路并行前，16-worker 试运行已经完成：196 条拟合、196 条渲染，零失败，140.7 秒，保留在 `runs/bones_fit_v3_pilot`。按新要求另建 `runs/bones_fit_v3_pilot_48`，执行 48-worker 预检并采样内存，没有覆盖前一次结果。

```bash
$PY scripts/fit_bones_terrain_support.py --screen-run runs/bones_screen_v1_full \
  --output runs/bones_fit_v3_pilot_48 --limit-per-stem 2 --viz-per-stem 2 \
  --variants 4 --skin-stride 2 --workers 48
```

48-worker 试运行：196 条拟合和渲染全部成功，`failures=[]`，耗时 **86.2 秒**。失败率 0%，低于 5%；耗时低于 `8 × 161.7 = 1293.6` 秒。v1 的 161.7 秒包含 GIF；本次试运行按指令没有开启 GIF，因此耗时用于阶段门检查。

通过后，主线程通过唯一持久 SSH 启动四个独立后台任务，各子代理负责对应结果的只读审查。共同拟合参数：

```bash
--viz-per-stem 2 --worst-per-group 3 --gif --variants 4 --skin-stride 2
```

| 分片 | Workers | 组 | 输出目录 |
| --- | ---: | --- | --- |
| S | 20 | known_dimension_support、box_support、stairs、obstacle_clearance、ladder、flat_control | `runs/bones_fit_v3_S` |
| L1 | 20 | crouch、kneel、wall_contact | `runs/bones_fit_v3_L1` |
| L2 | 20 | quadruped、prone | `runs/bones_fit_v3_L2` |
| E | 4 | on_the_edge / edge_balance | `runs/bones_screen_edge_v1`、`runs/bones_fit_edge_v1` |

S/L1/L2 各自执行以下命令，替换表中的分片名和组列表：

```bash
$PY -u scripts/fit_bones_terrain_support.py --screen-run runs/bones_screen_v1_full \
  --output runs/bones_fit_v3_S --groups known_dimension_support box_support stairs \
  obstacle_clearance ladder flat_control --viz-per-stem 2 --worst-per-group 3 \
  --gif --variants 4 --skin-stride 2 --workers 20
```

E 顺序执行：

```bash
$PY scripts/screen_bones_terrain_geometry.py --bvh-root "${BONES_BVH_ROOT}" \
  --config configs/bones_terrain_screen_edge_v1.json --output runs/bones_screen_edge_v1 --workers 4
$PY scripts/fit_bones_terrain_support.py --screen-run runs/bones_screen_edge_v1 \
  --output runs/bones_fit_edge_v1 --viz-per-stem 8 --variants 2 --workers 4
```

每路由 `nohup` 监督进程运行，日志位于 `runs/bones_fit_v3_coord/<name>.log`；监督器记录命令、退出码、结果计数及 `DONE_<name>`。完整命令见公开的 [S](../../examples/results/bones_terrain_fit_v3_20261023/execution/S.json)、[L1](../../examples/results/bones_terrain_fit_v3_20261023/execution/L1.json)、[L2](../../examples/results/bones_terrain_fit_v3_20261023/execution/L2.json)、[E](../../examples/results/bones_terrain_fit_v3_20261023/execution/E.json) 回执。

| 运行 | 拟合成功 | 渲染成功 | 处理 / 渲染失败 | 脚本耗时（秒） | 外层 wall-clock（秒） |
| --- | ---: | ---: | --- | ---: | ---: |
| 48-worker pilot | 196 | 196 | 0 / 0 | 86.2 | 87.1 |
| S | 2,907 | 123 | 0 / 0 | 1,446.4 | 1,447.3 |
| L1 | 2,131 | 58 | 0 / 0 | 1,394.1 | 1,395.2 |
| L2 | 1,750 | 44 | 0 / 0 | 1,434.1 | 1,435.0 |
| E | 120 | 11 | 0 / 0 | 344.0（拟合） | 415.6（含筛查） |

合计最多 64 个计算 worker。每次启动前检查 `free -g`；监督器每 30 秒从 `/proc/meminfo` 记录与 `free` 同源的全机内存数据及负载。

| 运行 | 最低可用内存（GiB） | 采样内存压力峰值（GiB） |
| --- | ---: | ---: |
| 48-worker pilot | 378.126 | 120.800 |
| S | 355.668 | 143.258 |
| L1 | 355.668 | 143.258 |
| L2 | 355.657 | 143.269 |
| E | 365.348 | 133.578 |

压力定义为 `MemTotal - MemAvailable`，包含全机其他工作，不是某一分片独占内存。预检起点可用内存约 422.9 GiB，最低约 378.1 GiB；按 worker 数从 48 扩到 64 比例估算后仍高于 100 GiB。实际四路最低可用内存为 355.657 GiB。**重启次数为 0，没有中途重新分配 worker。** 原始采样保存在 `execution/*.memory.txt`。

## 3. v1 与 v3 支撑指标

下表保留两版实际定义：v1 使用脚底点和关节球，先扣除 2 cm 容差；v3 主指标使用完整 **18,056 个蒙皮顶点、零容差**。v3 对 30 Hz 动作每两帧取一帧，即 15 Hz 采样。两版测量对象和容差不同，本轮表格记录各自数值，同口径改善仍需另外比较。

`clips_with_frames_over_2cm` 是至少一个检查帧超过 2 cm 的片段数。`penetration_max_p95_m` 是各片段峰值的 P95。此处深度为箱体水平投影内低于箱顶的垂直距离，包含地面。

### 最小支撑范围 minimal

| 组 | 片段数 | v1 超限片段 | v3 超限片段 | v1 峰值 P95（m） | v3 峰值 P95（m） |
| --- | ---: | ---: | ---: | ---: | ---: |
| box_support | 48 | 22 | 23 | 0.5966 | 0.1035 |
| crouch | 1,143 | 36 | 157 | 0.0091 | 0.0274 |
| flat_control | 900 | 0 | 80 | 0.0020 | 0.0217 |
| kneel | 638 | 619 | 601 | 0.0962 | 0.0924 |
| known_dimension_support | 1,760 | 32 | 39 | 0.0004 | 0.0176 |
| ladder | 64 | 16 | 9 | 0.7135 | 0.1695 |
| obstacle_clearance | 86 | 4 | 17 | 0.0156 | 0.0235 |
| prone | 919 | 904 | 919 | 0.1261 | 0.1259 |
| quadruped | 831 | 830 | 830 | 0.1298 | 0.1295 |
| stairs | 49 | 21 | 16 | 0.7385 | 0.1607 |
| wall_contact | 350 | 36 | 47 | 0.0694 | 0.0676 |

### 最大支撑范围 maximal

| 组 | v1 超限片段 | v3 超限片段 | v1 峰值 P95（m） | v3 峰值 P95（m） |
| --- | ---: | ---: | ---: | ---: |
| box_support | 22 | 23 | 0.5966 | 0.1090 |
| crouch | 36 | 157 | 0.0091 | 0.0274 |
| flat_control | 0 | 80 | 0.0020 | 0.0217 |
| kneel | 619 | 602 | 0.0962 | 0.0924 |
| known_dimension_support | 32 | 718 | 0.0004 | 0.1580 |
| ladder | 16 | 15 | 0.7135 | 0.3928 |
| obstacle_clearance | 4 | 19 | 0.0156 | 0.0245 |
| prone | 904 | 919 | 0.1261 | 0.1259 |
| quadruped | 830 | 830 | 0.1298 | 0.1295 |
| stairs | 21 | 33 | 0.7385 | 0.3712 |
| wall_contact | 36 | 47 | 0.0694 | 0.0676 |

平地额外支撑箱从 v1 的 31/900 条降到 v3 的 0/900。另一方面，已知尺寸组的 maximal 有 718/1,760 条皮肤超限，仍需修正扩展范围。各组 `unsupported_evidence_points=0` 只表示纳入统计的证据点下方存在表面；地面也算表面，支撑间隙需单独看。

例如 `stairs_climbing_down_loop_R_001__A300` 的支撑间隙 P95 为 0.992220 m，四个变体仍被接受；`jump_to_shoulder_roll_R_003__A304` 的支撑箱全部被拒绝后，间隙 P95 为 1.709351 m，四个变体也被接受。当前变体接受条件未包含支撑间隙。

## 4. Scene 分组结果

顶棚状态缩写：F = `fitted`；N = `no_sustained_low_posture`；R = `no_conflict_free_rectangle`；T = `unexplainable_tight_above_bound`；B = `unexplainable_body_reaches_bound`。解释率分位数只统计成功拟合的顶棚，空白用 `-` 表示。

| 组 | F | N | R | T | B | 紧顶棚解释率 P50 | P10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| box_support | 1 | 9 | 19 | 19 | 0 | 0.0000 | 0.0000 |
| crouch | 1,009 | 122 | 12 | 0 | 0 | 1.0000 | 1.0000 |
| flat_control | 0 | 900 | 0 | 0 | 0 | - | - |
| kneel | 274 | 0 | 363 | 1 | 0 | 1.0000 | 1.0000 |
| known_dimension_support | 2 | 1,736 | 12 | 10 | 0 | 0.5625 | 0.2125 |
| ladder | 3 | 50 | 7 | 1 | 3 | 0.6854 | 0.1371 |
| obstacle_clearance | 0 | 77 | 9 | 0 | 0 | - | - |
| prone | 831 | 0 | 88 | 0 | 0 | 1.0000 | 1.0000 |
| quadruped | 699 | 0 | 132 | 0 | 0 | 1.0000 | 1.0000 |
| stairs | 0 | 41 | 7 | 1 | 0 | - | - |
| wall_contact | 85 | 246 | 19 | 0 | 0 | 1.0000 | 1.0000 |

`fitted_overhead_clips_with_frames_over_2cm` 的实现按顶棚/墙体检查项计数。逐条审计确认本轮全部超限项都是 `wall_1`，因此本轮也等于唯一墙体超限片段数，表中据此标为“墙体超限”。

| 组 | 拟合墙体片段 | 墙体超限 | 接受变体 / 全部 | 直立高顶棚变体 | 杂物均数 | 地面皮肤穿透 P95（m） |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| box_support | 43 | 0 | 188 / 192 | 14 | 1.984 | 0.1000 |
| crouch | 667 | 28 | 4,460 / 4,572 | 223 | 2.062 | 0.0274 |
| flat_control | 574 | 43 | 3,428 / 3,600 | 1,828 | 2.000 | 0.0217 |
| kneel | 623 | 8 | 2,433 / 2,552 | 0 | 1.989 | 0.0926 |
| known_dimension_support | 1,562 | 1 | 6,717 / 7,040 | 3,344 | 1.985 | 0.0150 |
| ladder | 51 | 13 | 192 / 256 | 42 | 2.023 | 0.1000 |
| obstacle_clearance | 59 | 1 | 340 / 344 | 149 | 1.977 | 0.0235 |
| prone | 514 | 420 | 1,992 / 3,676 | 0 | 1.967 | 0.1000 |
| quadruped | 520 | 187 | 2,576 / 3,324 | 0 | 2.038 | 0.1000 |
| stairs | 22 | 1 | 164 / 196 | 23 | 1.883 | 0.0811 |
| wall_contact | 339 | 75 | 1,100 / 1,400 | 479 | 1.979 | 0.0677 |

完整原始块见 [merged/by_group.json](../../examples/results/bones_terrain_fit_v3_20261023/merged/by_group.json)。场景检查测量皮肤点在有限箱体内部到最近面的距离，地面单独记录。0.2 m 厚地面板的这一深度最大为 0.1 m，因而与第 3 节的垂直下陷深度数值不同。

**顶棚：** 主线程读取全部 fit 的 `fitted_checks.max_m`，确认 2,904 条成功顶棚的紧、松检查均为 0，详见 [geometry_gate_audit.json](../../examples/results/bones_terrain_fit_v3_20261023/execution/geometry_gate_audit.json)。蹲、手膝、匍匐组的成功子集解释率中位数达到 0.9 门槛，但还有未拟合片段和低解释个例：`mohak_idle_stop_001__A125` 的解释率为 0，四个变体仍被接受。

**侧墙：** 777 条墙体超过 2 cm，零穿透门未通过。墙体定位使用地面以上 0.2–2.0 m 的皮肤高度带，验收使用完整皮肤；此外，墙面可大幅外移。`walk_backward_loop_004__A026` 的墙外移 10.7777 m，`mohak_loop_003__A032` 外移 7.8391 m 后仍有 0.054048 m 穿透。接触得分来自初始候选平面，最终平面的接触没有重新测量。

**变体：** 3,562 个被拒绝，总接受率 86.8813%，尚未达到接近 100% 的目标。`accepted` 当前只检查非地面实体是否有采样帧穿透超过 2 cm，解释率、地面穿透和支撑间隙另列。逐片段接受数保存在 [merged/fits_compact.tsv](../../examples/results/bones_terrain_fit_v3_20261023/merged/fits_compact.tsv)，各分片审查和选例在 `execution/*_audit.json`、`execution/*_review.md`。

## 5. 紧顶棚高度校准统计

从合并 TSV 中仅取 `ceiling_status=fitted` 的片段，解析 `ceiling_range_m` 的第一个值，按组计算 P5/P50/P95。输入是 TSV 保留到 0.001 m 的高度，坐标为 SOMA Uniform 世界中相对拟合地板的高度；这里没有更新原来的 0.72/0.45 姿态比例参数。

| 组 | 成功片段 | P5（m） | P50（m） | P95（m） |
| --- | ---: | ---: | ---: | ---: |
| box_support | 1 | 0.7420 | 0.7420 | 0.7420 |
| crouch | 1,009 | 1.0776 | 1.2600 | 1.4680 |
| flat_control | 0 | - | - | - |
| kneel | 274 | 1.2390 | 1.2900 | 1.6980 |
| known_dimension_support | 2 | 1.6824 | 1.6855 | 1.6886 |
| ladder | 3 | 1.4938 | 1.5460 | 1.6918 |
| obstacle_clearance | 0 | - | - | - |
| prone | 831 | 0.3350 | 0.5580 | 0.6465 |
| quadruped | 699 | 0.7150 | 0.8220 | 0.9270 |
| stairs | 0 | - | - | - |
| wall_contact | 85 | 1.1880 | 1.5790 | 1.7078 |

机器可读结果：[ceiling_calibration.json](../../examples/results/bones_terrain_fit_v3_20261023/ceiling_calibration.json)。演员尺度另见 [S/actor_scale.json](../../examples/results/bones_terrain_fit_v3_20261023/shards/S/actor_scale.json)，尺度 P5/P50/P95 为 1.04031/1.14434/1.23580。

## 6. on_the_edge

120 条筛查和拟合全部成功。筛查与拟合均为 **119 条单层、1 条无可用层**，没有两层及以上片段；120 条都没有生成抬高的支撑箱体。无层片段为 `on_the_edge_003__A414`。

11 条渲染全部成功。皮肤支撑指标有 5 条超过 2 cm，片段峰值 P95 为 0.0187 m；240/240 个变体通过非地面门。111 条拟合了墙；唯一拟合顶棚的 `on_the_edge_002__A414` 对其 13 个低姿态采样帧的解释率为 0。当前方法没有从这批动作恢复出抬高的窄支撑面。

三条图片观察：

1. [on_the_edge_001__A236](../../examples/results/bones_terrain_fit_v3_20261023/edge/viz/edge_balance/on_the_edge_001__A236.png) 显示角色在单一地面高度上直立、伸展手臂，图中没有墙、顶棚或抬高的支撑箱体。
2. [on_the_edge_001__A098](../../examples/results/bones_terrain_fit_v3_20261023/edge/viz/edge_balance/on_the_edge_001__A098.png) 显示多帧姿态位于紫色墙旁，标注的皮肤峰值为 0.026 m，32 个采样帧超过 2 cm。
3. [on_the_edge_002__A414](../../examples/results/bones_terrain_fit_v3_20261023/edge/viz/edge_balance/on_the_edge_002__A414.png) 显示紫色墙和一小块青色顶棚，标注的低姿态解释率为 0%，21 个采样帧穿透超过 2 cm。

原始数据：[edge/by_group.json](../../examples/results/bones_terrain_fit_v3_20261023/edge/by_group.json)、[edge/fits_compact.tsv](../../examples/results/bones_terrain_fit_v3_20261023/edge/fits_compact.tsv)、[E 审计](../../examples/results/bones_terrain_fit_v3_20261023/execution/E_audit.json)。

## 7. 发布与可追溯性

发布目录为 `examples/results/bones_terrain_fit_v3_20261023/`：

- `shards/S、L1、L2/` 保留独立报告、分组 JSON、紧凑 TSV 和 stdout 尾部；演员尺度位于 S。
- `merged/` 是 11 个不重叠组的并集，6,788 个片段名唯一，TSV 只有一行表头。
- `viz/` 保留分片的 225 张图及补充审查的 13 张图；`edge/viz/` 为 11 张图。共 249 张、33,982,087 bytes，低于 40 MB，未删图。
- `viewer/` 共 68 条，含旧版全部 38 条、指定五组各 3 条额外样本及全部 29 条标为 `worst_penetration` 的片段；`edge/viewer/` 包含全部 11 条 edge 渲染。
- `pilot/` 发布新的 48-worker 预检报告；执行、内存、审查及校验记录位于 `execution/`。

旧 38 条中本轮未自动渲染的条目，以及 kneel 额外样本，统一从已保存的 fit JSON 生成审查图和 viewer，写入新的 `runs/bones_fit_v3_review_supplement`。总共补渲染 13 条，拟合结果未重算。选样名单与原因见 [publication_selection.json](../../examples/results/bones_terrain_fit_v3_20261023/publication_selection.json)。

完整私有产物仍在 PRO6000：`${TAK_ROOT}/runs/bones_fit_v3_{S,L1,L2}/fits/`、各分片 `viz/<group>/*.gif`、`${TAK_ROOT}/runs/bones_fit_edge_v1/` 和 `${TAK_ROOT}/runs/bones_fit_v3_coord/`。GitHub 没有提交 GIF、`fits/*.json` 或源 BVH。

全部 249 张 PNG 已解码并检查非空，225 个 GIF 已确认多帧。主 viewer 68 条、edge viewer 11 条的索引和骨架结构已核对；旧 38 条的 `frames_mm`、关节名、连线、帧率与 v1 完全一致。原有 8 个输入、资源和转换代码文件哈希未变化，所有 6,788 个拟合 JSON 均通过有限数值检查。细节见 [artifact_checks.json](../../examples/results/bones_terrain_fit_v3_20261023/execution/artifact_checks.json)。

侧墙扫描在 48-worker 预检日志中出现 13 次、S 中出现 12 次 `RuntimeWarning: invalid value encountered in subtract`。代码在高度带为空时会出现 `-inf - -inf`，该方向随后被排除；警告保留在日志中，没有压制或修改算法。上述日志次数受每进程警告去重影响，不作为受影响片段数。

结果通过私有 Drive 取回，原始与公开副本逐文件 SHA256 对应见 [publication_manifest.json](../../examples/results/bones_terrain_fit_v3_20261023/publication_manifest.json)。公开副本仅替换私人路径，PNG、动作帧和科学数值保持原样。
