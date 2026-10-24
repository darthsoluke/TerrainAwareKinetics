# BONES 地形拟合 v5：PRO6000 执行与审查

实际执行日期：**2026-10-24，Asia/Singapore**。文件名沿用任务指定日期。运行源码：`981157235447cf416627f373b7557bbae17f2c04`，方法见 [第 10 节](bones_terrain_method_v2_20261023.md#10-v5-修订根据-v4-执行结果)。

**全量执行完成，匍匐和手膝两组的联合质量目标未通过。** 6,788 条拟合全部成功，196 条全量渲染全部成功。最大支撑的蒙皮穿透超限片段降到零，crouch 墙体从 327 降到 58；但匍匐和手膝的缺支撑拒绝次数上升。所有参数保持提交中的默认值，没有训练。

结果入口：[报告目录](../../examples/results/bones_terrain_fit_v5_20261023/README.md)、[完整 v3/v4/v5 对照](../../examples/results/bones_terrain_fit_v5_20261023/comparison.md)、[机器可读表](../../examples/results/bones_terrain_fit_v5_20261023/comparison.tsv)、[目标判定数据](../../examples/results/bones_terrain_fit_v5_20261023/target_results.json)。

## 1. 环境和测试

在 PRO6000 的独立 PFNN Python 环境运行，设置与 v4 一致：

```bash
export PY=${PFNN_PROJECT}/env/bin/python
export PYTHONPATH="${KIMODO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "${TAK_ROOT}"
git pull --ff-only
$PY -m pytest tests/test_bones_terrain_fit.py tests/test_bones_terrain_scene.py \
  tests/test_bones_terrain_screen.py tests/test_bones_terrain_survey.py \
  -q --junitxml=runs/bones_fit_v5_coord/tests.xml
```

实际结果为 **`49 passed in 5.82s`，0 skipped**。这个源码提交共收集 49 项：fit 18、scene 12、screen 9、survey 10；任务预期写的是 50。官方 skinner 接口测试实际执行并通过，见 [JUnit](../../examples/results/bones_terrain_fit_v5_20261023/execution/tests.xml) 和 [收集清单](../../examples/results/bones_terrain_fit_v5_20261023/execution/test_inventory.txt)。

绑定姿态检查：

```text
1.760514736175537 {0: 7557, 1: 2289, 2: 5694, 3: 2516}
```

高度为米；代码 0/1/2/3 分别为其他、躯干、手、脚，共 18,056 个顶点。环境为 Python 3.11.16、NumPy 1.26.4、SciPy 1.17.1、PyTorch 2.7.1+cu128、Matplotlib 3.11.2、Pillow 12.3.0。计算在 CPU 上完成；[环境回执](../../examples/results/bones_terrain_fit_v5_20261023/execution/environment.json) 保存版本与实际变量。

全程由主线程使用一条持久 SSH；断线后确认旧连接结束再重连。源 BVH、旧运行目录和 cloth 环境未改动。只有主线程操作 Git。

## 2. 定向检查和试跑

从原 screen 的记录筛出以下集合的并集，不重新筛查或改写原记录：12 条 v4 点检、11 条 inverted-foot 标记、v4 maximal 峰值最高的 10 条 prone 和 10 条 quadruped。去重后为 **41 条**。排名使用原始浮点指标，文件名用于打破并列。

```bash
$PY scripts/fit_bones_terrain_support.py --screen-run runs/bones_screen_v5_targets \
  --output runs/bones_fit_v5_targets --viz-per-stem 50 --variants 4 --skin-stride 2 --workers 16
```

41 条拟合、41 张 PNG 全部成功，耗时 **98.6 秒**。主线程逐张检查了全部 PNG，并写了 [41 行图像判断](../../examples/results/bones_terrain_fit_v5_20261023/targets/target_visual_review.md)。[逐片段 v4/v5 表](../../examples/results/bones_terrain_fit_v5_20261023/targets/target_comparison.md) 同时列出最小/最大支撑深度与超限帧数、层高、身体承托足部的证据块、三项标记、墙体状态/类型、资格和拒绝原因。

两个进入试跑的条件均通过：最大支撑超限清单为空；`come_up_50cm_box_R_002__A053` 保留 1 个支撑，`stairs_climbing_up_start_R_101__A301` 保留 7 个支撑。两者的 maximal 峰值分别为 5.5 mm、15.2 mm。

定向样本的 maximal 超限片段由 v4 的 22 条降到零，但 E+G 资格由 19 条降到 14 条。图中仍可见爬行的碎片状低支撑，指定梯子和越障片段仍缺少对应的高处几何；这些问题在全量审查中继续保留。

随后执行每词干 2 条、48 worker 的试跑：

```bash
$PY scripts/fit_bones_terrain_support.py --screen-run runs/bones_screen_v1_full \
  --output runs/bones_fit_v5_pilot_48 --limit-per-stem 2 --viz-per-stem 2 \
  --variants 4 --skin-stride 2 --workers 48
```

196 条拟合和渲染全部成功，脚本耗时 **90.0 秒**，外层 91.1 秒。失败率为零，耗时低于任务规定的 261 秒；v4 为 87.1 秒。通过后才启动全量。外层计时包装仅记录原函数耗时，未改变拟合或渲染参数；[最慢 10 条](../../examples/results/bones_terrain_fit_v5_20261023/execution/P48_slowest10.json) 分别保留拟合、渲染时间与源帧数。

## 3. 三路全量和合并

每路 20 个 CPU worker，并行运行。共同命令如下，L1/L2 仅替换组和输出名：

```bash
$PY -u scripts/fit_bones_terrain_support.py --screen-run runs/bones_screen_v1_full \
  --output runs/bones_fit_v5_S --groups known_dimension_support box_support stairs \
  obstacle_clearance ladder flat_control --viz-per-stem 2 --worst-per-group 3 \
  --gif --variants 4 --skin-stride 2 --workers 20
```

| 任务 | 组 | 拟合 | 渲染 | 拟合/渲染失败 | 脚本秒数 | 外层秒数 |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| S | 已知尺寸、箱体、楼梯、越障、梯子、平地 | 2,907 | 108 | 0/0 | 1,571.1 | 1,573.0 |
| L1 | crouch、kneel、wall_contact | 2,131 | 50 | 0/0 | 1,493.1 | 1,494.7 |
| L2 | quadruped、prone | 1,750 | 38 | 0/0 | 1,818.0 | 1,819.2 |

nohup 监督器记录了实际命令、退出码、每 30 秒内存采样和完成标记；均退出 0，没有重启。最低采样可用内存约 **353.19 GiB**。三位子代理在对应任务完成后只读审查结果和 PNG。额外 10 张 crouch 墙体图用已保存拟合渲染，未重新拟合。

合并目录为 `runs/bones_fit_v5_merged`：三份紧凑表拼接为 6,788 行，11 个组无重叠，viewer 索引合并保留在服务器。三路源码提交和五个模块的 SHA256 一致，见 [合并回执](../../examples/results/bones_terrain_fit_v5_20261023/merge_receipt.json)。

## 4. v3 / v4 / v5 对照

完整表每组一行，保留 v4 的所有字段，并加入 `floating_slab`、`foot_patches_rest_on_body`、`patches_resting_on_body` 总数、悬空平台最小厚度和最小离地底高。v3/v4 没有记录的新字段为 null。

**穿透口径发生了变化。** v3/v4 表使用之前保存的箱顶垂直深度；v5 排除箱底以下的点，对箱内点计算到顶面与底面较近的垂直距离。三版都使用官方 18,056 顶点、30Hz 动作隔两帧采样。此表比较各版实际报告；几何是否合理另见逐图审查。

下表为至少有一个采样帧超过 2 cm 的片段数，每格依次为 minimal / maximal：

| 组 | 片段 | v3 | v4 | v5 |
| --- | ---: | --- | --- | --- |
| box_support | 48 | 2 / 4 | 3 / 7 | 0 / 0 |
| crouch | 1,143 | 0 / 0 | 2 / 4 | 0 / 0 |
| flat_control | 900 | 0 / 0 | 0 / 0 | 0 / 0 |
| kneel | 638 | 20 / 28 | 11 / 35 | 0 / 0 |
| known_dimension_support | 1,760 | 21 / 705 | 7 / 177 | 0 / 0 |
| ladder | 64 | 1 / 7 | 2 / 2 | 0 / 0 |
| obstacle_clearance | 86 | 0 / 2 | 0 / 0 | 0 / 0 |
| prone | 919 | 1 / 1 | 402 / 595 | 0 / 0 |
| quadruped | 831 | 1 / 1 | 118 / 311 | 0 / 0 |
| stairs | 49 | 12 / 30 | 4 / 17 | 0 / 0 |
| wall_contact | 350 | 0 / 0 | 1 / 4 | 0 / 0 |

v5 maximal 超限的完整 [bug 清单](../../examples/results/bones_terrain_fit_v5_20261023/maximal_over_2cm_bug_clips.json) 为空。保留墙体和拟合顶棚的超限片段也均为零，见 [独立汇总](../../examples/results/bones_terrain_fit_v5_20261023/retained_surface_checks.json)。

### 新标记与悬空平台

标记按片段计；证据块总数按 patch 计。厚度与底高取 minimal 和 maximal 中所有悬空支撑的最小值；底高相对各自地面，单位为米。没有悬空支撑的格子为“无”。

| 组 | floating_slab | foot_patches_rest_on_body | patches 总数 | 最小厚度 | 最小底高 |
| --- | ---: | ---: | ---: | ---: | ---: |
| box_support | 0 | 1 | 1 | 无 | 无 |
| crouch | 0 | 1 | 1 | 无 | 无 |
| flat_control | 0 | 0 | 0 | 无 | 无 |
| kneel | 0 | 51 | 61 | 无 | 无 |
| known_dimension_support | 0 | 3 | 3 | 无 | 无 |
| ladder | 1 | 0 | 0 | 0.072916 | 0.290916 |
| obstacle_clearance | 0 | 0 | 0 | 无 | 无 |
| prone | 0 | 33 | 39 | 无 | 无 |
| quadruped | 0 | 230 | 755 | 无 | 无 |
| stairs | 0 | 0 | 0 | 无 | 无 |
| wall_contact | 0 | 1 | 1 | 无 | 无 |
| 合计 | **1** | **320** | **861** | **0.072916** | **0.290916** |

已知尺寸组的三个身体承托足部标记均保留在 [完整清单](../../examples/results/bones_terrain_fit_v5_20261023/known_size_flagged_clips.json)：

- `come_down_50cm_box_R_002__A049`
- `jump_off_1m_R_102__A301`
- `jump_off_1m_R_103__A301`

该组三条各剔除 1 个 patch，比例 **3/1,760 = 0.17%**；没有悬空平台标记。

### 低姿态资格与拒绝原因

E 为 `environment_explained`，G 为 `geometry_only_low_posture_unexplained`。S/F 分别为 `solid_penetrates_skin` / `foot_evidence_not_supported`，按变体计数，同一变体可以有多个拒绝原因。

| 组 | E+G v4 | E+G v5 | S v4 → v5 | F v4 → v5 | 联合目标 |
| --- | --- | --- | --- | --- | --- |
| prone | 368/919，40.04% | 464/919，50.49% | 1,923 → 0 | 1,637 → 1,834 | **未通过：F 上升** |
| quadruped | 279/831，33.57% | 279/831，33.57% | 428 → 0 | 2,184 → 2,211 | **未通过：资格持平、F 上升** |
| kneel | 525/638，82.29% | 532/638，83.39% | 32 → 0 | 432 → 424 | **通过** |

全库资格为 **E 4,893 / G 676 / rejected 1,219**，对应 28,849.69 / 3,928.80 / 9,862.48 秒。接受变体 **19,554/27,152，72.02%**，v4 为 70.81%。完整表保留每组资格、时长、全部拒绝原因、地面来源/深度、墙体和顶棚统计。

## 5. 逐项目标判定

| 目标 | 实际结果 | 判定 |
| --- | --- | --- |
| 每组 maximal 支撑超限为零 | 全部 11 组为零，bug 清单为空 | **通过** |
| prone 可用比例上升且 S/F 均下降 | 可用比例上升；S 降到 0；F 上升 197 | **未通过** |
| quadruped 可用比例上升且 S/F 均下降 | 可用比例持平；S 降到 0；F 上升 27 | **未通过** |
| kneel 可用比例上升且 S/F 均下降 | 82.29% → 83.39%；S/F 均下降 | **通过** |
| 已知尺寸两项新标记各 ≤1% | floating 0%，body-rest 0.17% | **通过** |
| crouch 墙体远少于 327 | 58，减少 82.26%；审查 10 张保留墙体图 | **数量目标通过** |
| wall_contact 墙覆盖 ≥30% | 112/350 = 32.00%，v4 为 32.86% | **通过** |
| 平地 100% 环境可解释、墙 ≤1% | 900/900，墙 3/900 = 0.33% | **通过** |

## 6. 图片审查

主线程审查 41 张定向图。三个子代理分别审查 S 的 11 张、L1 的 15 张和 L2 的 10 张，并逐图写下判断：

- [S：箱体、楼梯、梯子、越障、平地](../../examples/results/bones_terrain_fit_v5_20261023/execution/S_review.md)
- [L1：10 张 crouch 墙体与 kneel / wall_contact](../../examples/results/bones_terrain_fit_v5_20261023/execution/L1_review.md)
- [L2：手膝与匍匐](../../examples/results/bones_terrain_fit_v5_20261023/execution/L2_review.md)

梯子和越障组全部悬空平台只有 `ladder_slip_R_104__A301`，对应厚度 **7.29 cm**、底部离地 **29.09 cm**；越障组没有悬空平台。图中平台靠近脚部下降轨迹，明显低于高举双手，作为局部脚踏板的厚度合理。其接触证据高度差 P95 仍为 36.38 cm，4 个变体全部因缺支撑和低姿态未解释而拒绝。定向图中的 `ladder_slide_down_R_303__A302` 和 `jump_over_obstacle_2m_L_003__A415` 仍缺少相应的高处几何。

10 张 crouch 保留墙体图中，2 张有较清楚的前伸手与墙位对应，8 张仍可疑，主要是收臂、转身外缘或路径侧缘被配成墙。数量目标通过，但残留墙体的动作依据仍需改善。S 的搬箱样例也出现了把搬物手势解释为 brace 墙的情况。

本轮把支撑穿透压到门槛以内，同时暴露了剩余问题：部分脚底证据没有得到有效支撑，低姿态图里还存在碎片状矮箱。下一步应逐项对齐被保留的接触证据、被拒绝的支撑和缺支撑原因，尤其检查匍匐、手膝和梯子；本轮没有修改参数追求通过。

## 7. 文件位置与发布

服务器上相对 `${TAK_ROOT}` 的位置：

| 内容 | 路径 |
| --- | --- |
| 定向筛选输入 | `runs/bones_screen_v5_targets/` |
| 定向拟合和 PNG | `runs/bones_fit_v5_targets/` |
| 48-worker 试跑 | `runs/bones_fit_v5_pilot_48/` |
| 全量拟合、GIF、viewer | `runs/bones_fit_v5_S/`、`runs/bones_fit_v5_L1/`、`runs/bones_fit_v5_L2/` |
| 合并表与完整 viewer 索引 | `runs/bones_fit_v5_merged/` |
| 监督日志、计时、审查包 | `runs/bones_fit_v5_coord/` |

公开结果只包含小型统计、紧凑表、测试/执行记录及已审查 PNG。完整 fit JSON、GIF、viewer 动作数据和源 BVH 留在服务器。通过私有 Drive 传回结果，逐文件核对 SHA256 后将机器路径替换为角色变量；数值和 PNG 保持原样。校验与文件清单见 [publication_manifest.json](../../examples/results/bones_terrain_fit_v5_20261023/publication_manifest.json) 和 [SHA256SUMS](../../examples/results/bones_terrain_fit_v5_20261023/SHA256SUMS)。
