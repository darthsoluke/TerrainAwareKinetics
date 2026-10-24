# BONES 地形拟合 v4：PRO6000 执行与审查

实际执行日期：**2026-10-24，Asia/Singapore**。文件名沿用任务指定日期。运行源码：`c9039da705426c9304876c1cf7955fa7fa2afce0`，方法见 [方法文档第 9 节](bones_terrain_method_v2_20261023.md#9-v4-修订根据-v3-执行结果)。

**执行完成，部分质量目标未通过。** 三路共处理 6,788 条，拟合和渲染失败均为 0。平地误配墙、已知尺寸箱体和保留墙体的穿透有改善；匍匐、手膝与跪姿的支撑超限仍未接近零，其中前两组相对同口径 v3 出现明显退步。没有修改参数追求达标，也没有训练。

结果入口：[报告与审查图](../../examples/results/bones_terrain_fit_v4_20261023/README.md)、[每组 v3/v4 完整对照表](../../examples/results/bones_terrain_fit_v4_20261023/comparison.md)、[机器可读 TSV](../../examples/results/bones_terrain_fit_v4_20261023/comparison.tsv)。

## 1. 环境、测试和部位标签

沿用独立 Python、Kimodo 和线程设置：

```bash
export PY=${PFNN_PROJECT}/env/bin/python
export PYTHONPATH="${KIMODO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "${TAK_ROOT}"
$PY -m pytest tests/test_bones_terrain_fit.py tests/test_bones_terrain_scene.py \
  tests/test_bones_terrain_screen.py tests/test_bones_terrain_survey.py -q --junitxml=/tmp/v4_tests.xml
```

实际结果：**`46 passed in 5.80s`**，全部运行，无跳过。`test_skin_vertices_uses_official_skinner_interface` 已按 JUnit 记录核实通过，见 [测试日志](../../examples/results/bones_terrain_fit_v4_20261023/execution/tests.txt) 与 [接口测试记录](../../examples/results/bones_terrain_fit_v4_20261023/execution/skinner_test.txt)。

蒙皮绑定姿态检查输出：

```text
1.760514736175537 {0: 7557, 1: 2289, 2: 5694, 3: 2516}
```

高度单位为米；代码 0 为其他部位，1 为躯干，2 为手，3 为脚，总计 18,056 个顶点。绑定姿态检查未报错。[part_labels.txt](../../examples/results/bones_terrain_fit_v4_20261023/execution/part_labels.txt) 保留原输出。

全部数值计算在 PRO6000 的 CPU 上运行。服务器原代理的 TLS 连接失败后，通过同一条任务 SSH 转发到已有 M78 链路，成功拉取代码并恢复私有 Drive 传输；机器配置保留在私有记录。源 BVH、此前运行目录和 cloth 环境保持不变。

## 2. 试运行与三路全量

48-worker 试运行：

```bash
$PY scripts/fit_bones_terrain_support.py --screen-run runs/bones_screen_v1_full \
  --output runs/bones_fit_v4_pilot_48 --limit-per-stem 2 --viz-per-stem 2 \
  --variants 4 --skin-stride 2 --workers 48
```

196 条拟合、196 条渲染全部成功，耗时 **87.1 秒**。通过失败率低于 5%、耗时不超过 `2 × 86.2 = 172.4` 秒的门槛后，才启动三路全量。

共同命令参数：

```bash
$PY -u scripts/fit_bones_terrain_support.py --screen-run runs/bones_screen_v1_full \
  --output runs/bones_fit_v4_S --groups known_dimension_support box_support stairs \
  obstacle_clearance ladder flat_control --viz-per-stem 2 --worst-per-group 3 \
  --gif --variants 4 --skin-stride 2 --workers 20
```

L1 将输出改为 `runs/bones_fit_v4_L1`，组改为 `crouch kneel wall_contact`；L2 输出为 `runs/bones_fit_v4_L2`，组为 `quadruped prone`。每路由 nohup 监督器保存命令、退出码、内存采样及 `DONE_<name>`。三位子代理在运行期间只接收状态，完成后审查各自 JSON、TSV 和 PNG。没有运行 `on_the_edge`。

| 任务 | Workers | 拟合 | 渲染 | 处理/渲染失败 | 脚本耗时（秒） | 外层耗时（秒） |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Pilot | 48 | 196 | 196 | 0/0 | 87.1 | 87.9 |
| S | 20 | 2,907 | 116 | 0/0 | 1,538.8 | 1,539.8 |
| L1 | 20 | 2,131 | 56 | 0/0 | 1,496.0 | 1,497.0 |
| L2 | 20 | 1,750 | 44 | 0/0 | 1,796.8 | 1,797.8 |

所有退出码为 0，没有重启。每次启动前可用内存均超过 100 GiB；三路运行的采样最低可用内存约 **368.66 GiB**，全机 `MemTotal-MemAvailable` 采样峰值约 130.27 GiB。完整回执和每 30 秒内存采样在 [execution/](../../examples/results/bones_terrain_fit_v4_20261023/execution/)。

## 3. 怎样与 v3 比较

v3 原报告的 `skin` 同时包含地面和支撑箱，v4 则分别记录 `skin_support` 与 `skin_ground`。本轮额外用 4 个 CPU worker 只读复测 v3 已保存的几何：有支撑箱的 1,885 条重新计算官方蒙皮，另外 4,903 条直接复用原来的地面单独指标。没有重新拟合或改动 v3 场景。

两版都使用 18,056 个顶点、30 Hz 动作每两帧取一帧、零容差，统计垂直低于箱顶的深度。重新合并地面与支撑后，全部 v3 峰值和超限帧数与原记录一致，最大数值误差为 **0.0 m**。复测耗时 1,607.2 秒，与全量并行，总 worker 数最高 64。见 [复测回执](../../examples/results/bones_terrain_fit_v4_20261023/execution/v3_normalization_receipt.json)。

完整对照表每组一行、v3/v4 各一列，包含支撑、地面、来源、标记、墙体状态和类型、墙体拒绝原因、顶棚、解释率、变体、资格片段数与秒数。v3 没有保存 `ground_source`、承重墙类别、墙体细分拒绝原因和新资格分类，这些格子标为未记录；v3 变体拒绝原因按其唯一的穿透门派生。v4 资格时长使用脚本记录的 `frames/30` 秒。

### 支撑箱体超限片段

以下均为至少一个采样帧的 `skin_support` 深度超过 2 cm 的片段数，已排除地面。minimal/maximal 分别表示最小/最大支撑范围。

| 组 | 片段数 | v3 minimal | v4 minimal | v3 maximal | v4 maximal |
| --- | ---: | ---: | ---: | ---: | ---: |
| box_support | 48 | 2 | 3 | 4 | 7 |
| crouch | 1,143 | 0 | 2 | 0 | 4 |
| flat_control | 900 | 0 | 0 | 0 | 0 |
| kneel | 638 | 20 | 11 | 28 | 35 |
| known_dimension_support | 1,760 | 21 | 7 | 705 | 177 |
| ladder | 64 | 1 | 2 | 7 | 2 |
| obstacle_clearance | 86 | 0 | 0 | 2 | 0 |
| prone | 919 | 1 | 402 | 1 | 595 |
| quadruped | 831 | 1 | 118 | 1 | 311 |
| stairs | 49 | 12 | 4 | 30 | 17 |
| wall_contact | 350 | 0 | 1 | 0 | 4 |

已知尺寸组 v3 原报告含地面时为 718 条，分离支撑后为 705 条；v4 为 177 条。匍匐与手膝的地面高度改善，同时出现大量新的低矮支撑几何，导致支撑超限退步。

### 地面峰值深度分布

每个片段先取蒙皮低于地面的最大垂直深度，再计算片段间分位数。单位为米，表中依次为 P50/P95/P99。

| 组 | v3 | v4 |
| --- | --- | --- |
| box_support | 0.0166 / 0.1035 / 0.1840 | 0.0100 / 0.0347 / 0.0393 |
| crouch | 0.0121 / 0.0274 / 0.0511 | 0.0119 / 0.0250 / 0.0300 |
| flat_control | 0.0128 / 0.0217 / 0.0281 | 0.0128 / 0.0216 / 0.0275 |
| kneel | 0.0439 / 0.0924 / 0.1110 | 0.0006 / 0.0284 / 0.0312 |
| known_dimension_support | 0.0093 / 0.0150 / 0.0199 | 0.0093 / 0.0150 / 0.0198 |
| ladder | 0.0100 / 0.1695 / 0.5495 | 0.0100 / 0.0418 / 0.0994 |
| obstacle_clearance | 0.0162 / 0.0235 / 0.0261 | 0.0162 / 0.0235 / 0.0261 |
| prone | 0.0760 / 0.1259 / 0.1562 | 0.0024 / 0.0145 / 0.0290 |
| quadruped | 0.0833 / 0.1295 / 0.1414 | 0.0013 / 0.0134 / 0.0296 |
| stairs | 0.0086 / 0.0577 / 0.3619 | 0.0086 / 0.0303 / 0.0671 |
| wall_contact | 0.0102 / 0.0676 / 0.1236 | 0.0095 / 0.0169 / 0.0218 |

### 地面来源与标记

L = `lowest_foot_level`，S = `skin_lowest_contacts`。S 的片段数也就是 `ground_from_skin` 的次数。v3 的来源字段未记录，`ground_from_skin` 和 `inverted_foot_level_dropped` 两项是 v4 新增。

| 组 | v4 地面来源 L / S | ground_from_skin | inverted_foot_level_dropped | support_rejected v3 → v4 |
| --- | --- | ---: | ---: | --- |
| box_support | 28 / 20 | 20 | 0 | 6 → 11 |
| crouch | 1,116 / 27 | 27 | 0 | 0 → 2 |
| flat_control | 898 / 2 | 2 | 0 | 0 → 0 |
| kneel | 143 / 495 | 495 | 2 | 7 → 81 |
| known_dimension_support | 1,758 / 2 | 2 | 0 | 12 → 11 |
| ladder | 47 / 17 | 17 | 1 | 6 → 7 |
| obstacle_clearance | 86 / 0 | 0 | 1 | 9 → 9 |
| prone | 14 / 905 | 905 | 1 | 3 → 302 |
| quadruped | 30 / 801 | 801 | 6 | 4 → 478 |
| stairs | 47 / 2 | 2 | 0 | 14 → 15 |
| wall_contact | 313 / 37 | 37 | 0 | 0 → 5 |

## 4. 墙体、顶棚与资格

完整 `wall_status`、`wall_kind`、候选墙拒绝原因和顶棚状态计数保存在 [对照表](../../examples/results/bones_terrain_fit_v4_20261023/comparison.md) 及 [by_group.json](../../examples/results/bones_terrain_fit_v4_20261023/by_group.json)。候选墙原因按方向/候选计数，同一片段可以拒绝多个候选。

| 组 | v3 保留墙 | v4 保留墙 | v4 lean / brace | v4 原地转换顶棚状态 | v4 顶棚解释率 P50 / P10 |
| --- | ---: | ---: | --- | ---: | --- |
| box_support | 43 | 0 | 0 / 0 | 13 | 0 / 0 |
| crouch | 667 | 327 | 8 / 319 | 4 | 1 / 1 |
| flat_control | 574 | 4 | 0 / 4 | 0 | - |
| kneel | 623 | 7 | 3 / 4 | 243 | 1 / 1 |
| known_dimension_support | 1,562 | 56 | 0 / 56 | 13 | 0.5625 / 0.2125 |
| ladder | 51 | 6 | 6 / 0 | 7 | 0.6854 / 0.1371 |
| obstacle_clearance | 59 | 0 | 0 / 0 | 8 | 0 / 0 |
| prone | 514 | 1 | 0 / 1 | 119 | 1 / 1 |
| quadruped | 520 | 37 | 37 / 0 | 118 | 1 / 1 |
| stairs | 22 | 0 | 0 / 0 | 7 | - |
| wall_contact | 339 | 115 | 101 / 14 | 9 | 1 / 0.9236 |

所有保留墙和成功拟合顶棚的采样帧均未超过 2 cm。被最终蒙皮检查删除的墙不计入“保留墙”：例如 quadruped 删除了 40 条，crouch 删除了 1 条。顶棚解释率分位数只统计成功拟合子集。

v4 共接受 **19,225/27,152 个变体（70.81%）**；v3 为 23,590/27,152（86.88%）。v4 同时检查非地面实体穿透、脚底证据支撑高度和持续低姿态的环境解释，v3 只检查前一项；因此两版接受率代表不同准入条件。

资格分三类：E = `environment_explained`，G = `geometry_only_low_posture_unexplained`，R = `rejected`。以下每格为“片段数 / 秒数”。v3 未记录这一分类，完整对照中保留为空。

| 组 | E | G | R |
| --- | --- | --- | --- |
| box_support | 9 / 41.97 | 26 / 112.34 | 13 / 64.90 |
| crouch | 1,108 / 6,626.44 | 30 / 274.09 | 5 / 43.07 |
| flat_control | 900 / 6,277.69 | 0 / 0 | 0 / 0 |
| kneel | 210 / 1,805.81 | 315 / 1,406.07 | 113 / 916.89 |
| known_dimension_support | 1,683 / 6,415.73 | 17 / 83.01 | 60 / 287.54 |
| ladder | 47 / 246.52 | 10 / 59.51 | 7 / 26.76 |
| obstacle_clearance | 73 / 353.07 | 3 / 20.24 | 10 / 84.57 |
| prone | 238 / 2,171.26 | 130 / 1,109.68 | 551 / 4,976.76 |
| quadruped | 205 / 1,737.69 | 74 / 433.56 | 552 / 4,255.02 |
| stairs | 26 / 104.99 | 7 / 39.00 | 16 / 92.33 |
| wall_contact | 315 / 2,440.58 | 27 / 104.72 | 8 / 29.16 |
| 合计 | **4,814 / 28,221.75** | **639 / 3,642.22** | **1,335 / 10,777.00** |

各组前三位拒绝原因如下，数字是变体层面的命中次数；同一变体可能同时有多个原因。F = `foot_evidence_not_supported`，U = `low_posture_unexplained`，P = `low_posture_partly_explained`，S = `solid_penetrates_skin`。

| 组 | 第一 | 第二 | 第三 |
| --- | --- | --- | --- |
| box_support | U 152 | F 52 | P 4（S 同为 4） |
| crouch | P 76 | U 52 | F 18 |
| flat_control | 无 | 无 | 无 |
| kneel | U 1,464 | F 432 | S 32 |
| known_dimension_support | F 200 | U 88 | S 40 |
| ladder | U 44 | F 24 | P 8 |
| obstacle_clearance | F 40 | U 32 | P 4 |
| prone | S 1,923 | F 1,637 | U 1,088 |
| quadruped | F 2,184 | U 556 | S 428 |
| stairs | F 60 | U 32 | S 4 |
| wall_contact | U 108 | F 29 | P 28 |

## 5. 目标判定

| 目标 | 结果 | 判断 |
| --- | --- | --- |
| flat_control 墙比例 ≤1% | 4/900 = 0.44% | **通过** |
| flat_control 环境可解释 ≥98% | 900/900 = 100% | **通过** |
| 已知尺寸 maximal 超限远低于 v3 718 | 原口径 718、同口径 705，v4 为 177 | **降低目标通过**，仍有 10.06% 片段超限 |
| 已知尺寸 inverted 标记接近零 | 0/1,760 | **通过** |
| prone 支撑超限接近零 | minimal 402/919，maximal 595/919（64.74%） | **未通过** |
| quadruped 支撑超限接近零 | minimal 118/831，maximal 311/831（37.42%） | **未通过** |
| kneel 支撑超限接近零 | minimal 11/638，maximal 35/638（5.49%） | **未通过** |
| 保留墙和拟合顶棚无 >2cm 帧 | 所有组均为零 | **通过本次隔帧检查** |
| wall_contact 墙覆盖与十图审查 | 115/350 = 32.86%，审查 10 张该组图 | 覆盖超过 30%，仍有未拟合片段 |

`ground_from_skin` 的比例：prone 905/919（98.48%），quadruped 801/831（96.39%），kneel 495/638（77.59%）。地面深度下降很明显，但它没有解决新增低矮支撑箱的冲突。eligibility 是当前准入程序的输出；本轮没有把这些片段加入训练缓存或启动训练。

## 6. 全部 inverted 标记与点检

已知尺寸组没有触发。全库 11 条如下；低姿态三组之外仅有前两条，所以点检覆盖这两条，未补造第三条。完整丢弃高度与所属组见 [inverted_foot_level_dropped.json](../../examples/results/bones_terrain_fit_v4_20261023/inverted_foot_level_dropped.json)。

| 片段 | 组 |
| --- | --- |
| `jump_over_obstacle_2m_L_003__A415` | obstacle_clearance |
| `ladder_slide_down_R_303__A302` | ladder |
| `kneeling_start_003__A167` | kneel |
| `kneeling_stop_003__A167` | kneel |
| `idle_crawl_stop_002__A232` | prone |
| `mohak_backward_loop_001__A030` | quadruped |
| `mohak_backward_start_001__A029` | quadruped |
| `mohak_backward_start_004__A030` | quadruped |
| `mohak_loop_002__A029` | quadruped |
| `mohak_loop_003__A029` | quadruped |
| `mohak_turn_000_004__A030` | quadruped |

三位子代理分别查看 S 的 10 张、L1 的 14 张（含 10 张 wall_contact）、L2 的 10 张样例，再检查各自所属的指定点检图。每条点检结论如下，支撑数值均为 minimal/maximal 峰值：

| 片段 | 图像与数据判断 |
| --- | --- |
| `crawl_ff_loop_315_005__A131` | **合理**：地面上持续匍匐，顶棚 0.642–0.654 m；支撑穿透 0，地面 4.2 mm，4/4 接受。 |
| `come_up_50cm_box_R_002__A053` | **支撑位置合理**：箱顶约 0.6949 m 与落脚对应，支撑 5.5/5.5 mm；Uniform 尺度下高于名义 50 cm，4/4 接受。 |
| `stairs_climbing_up_start_R_101__A301` | **最大范围可疑**：七个最小台阶跟随上行，但最大扩展穿透 131.3 mm；最小为 15.2 mm，抽样变体仍为 4/4 接受。 |
| `jog_backward_loop_002__A036` | **合理**：平地倒退慢跑，没有虚构台阶或墙；支撑穿透 0，地面 14.5 mm，4/4 接受。 |
| `idle_to_wall_leaning_idle_360_R_002__A289` | **合理**：墙面与倚靠姿态对应，无支撑穿透，地面 9.0 mm，4/4 接受。 |
| `drinking_wall_leaning_mug_090_R_003__A282` | **合理**：侧墙位于伸手位置，符合扶撑；支撑穿透 0，地面 9.9 mm，4/4 接受。 |
| `drinking_wall_leaning_mug_360_R_003__A282` | **合理**：墙面与向后倚靠对应；支撑穿透 0，地面 6.2 mm，4/4 接受。 |
| `kneeling_start_R_003__A408` | **地面仍可疑**：地面穿透 27.6 mm；站立与跪姿占用同一区域，不加顶棚合理，无支撑箱，0/4 接受。 |
| `kneeling_stop_003__A051` | **合理地保留无环境解释分类**：同一位置起身，不加顶棚；支撑穿透 0，地面 0.8 mm，0/4 接受。 |
| `idle_crawl_stop_001__A128` | **支撑可疑**：新增四块箱体，支撑穿透 23.0/26.6 mm；原地起身不加顶棚合理，0/4 接受。 |
| `jump_over_obstacle_2m_L_003__A415` | **重建可疑**：大幅跳跃没有对应高障碍，2.3134 m 脚底层被丢弃；小顶棚解释率 0，支撑 0/17.7 mm，0/4 接受。 |
| `ladder_slide_down_R_303__A302` | **缺失几何可疑**：1.9387 m 脚底层被丢弃后，垂直下降没有梯子或抬高支撑；支撑穿透 0，地面 12.4 mm，0/4 接受。 |

最小/最大范围与随机变体分别测量。例如楼梯最大范围存在冲突，但四个采样变体可以没有触发其实体穿透门；不能把采样通过数替代最大范围质量。

## 7. 交付与来源

远端合并结果位于 `${TAK_ROOT}/runs/bones_fit_v4_merged`，保留统一 TSV、分组统计、对照表、viewer 索引和点检图。各分片完整 `fits/*.json`、GIF 仍在 `${TAK_ROOT}/runs/bones_fit_v4_{S,L1,L2}`，日志与监督回执在 `runs/bones_fit_v4_coord`。

公开目录为 `examples/results/bones_terrain_fit_v4_20261023/`，包含小型统计、紧凑 TSV、回执、选定 PNG 和不超过 80 条 viewer。图片/数据选择清单见 [publication_selection.json](../../examples/results/bones_terrain_fit_v4_20261023/publication_selection.json)。没有提交完整拟合 JSON、GIF、BVH、缓存或机器连接信息。

公开包通过私有 Drive 传输并逐文件核验 SHA256；原始与公开路径替换的对应在 [publication_manifest.json](../../examples/results/bones_terrain_fit_v4_20261023/publication_manifest.json)。此前输入文件及全部 6,788 份 v3 fit 的哈希在任务结束时再次核验，结果见 [artifact_checks.json](../../examples/results/bones_terrain_fit_v4_20261023/execution/artifact_checks.json)。
