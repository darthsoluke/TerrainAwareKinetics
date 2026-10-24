# BONES 地形拟合 v6：PRO6000 全量对照

实际执行日期：**2026-10-24，Asia/Singapore**。文件名沿用任务指定日期。运行源码：`5a015448fe02321a75cb49d26a31445e10728c26`，方法见 [第 11 节](bones_terrain_method_v2_20261023.md#11-v6-修订根据-v5-执行结果)。

**全量执行完成，匍匐和手膝的核心目标仍未通过。** 6,788 条拟合、196 条全量渲染全部成功，支撑穿透超限保持零。正常箱体和楼梯未出现 `foot_lower_than_sole` 丢弃，墙体数量目标通过；但两组低姿态的缺支撑拒绝只下降 1.04% / 0.18%，至少四个支撑箱的片段数完全没有下降。

[结果入口](../../examples/results/bones_terrain_fit_v6_20261023/README.md) · [v4/v5/v6 完整表](../../examples/results/bones_terrain_fit_v6_20261023/comparison.md) · [机器可读 TSV](../../examples/results/bones_terrain_fit_v6_20261023/comparison.tsv) · [目标数值](../../examples/results/bones_terrain_fit_v6_20261023/target_results.json)

## 1. 环境、源码和测试

使用 PRO6000 的独立 PFNN Python 环境和原 Kimodo，全部计算在 CPU 上完成。没有训练、调参或修改源 BVH、此前运行目录、cloth 环境；只有主线程操作 Git。

```bash
export PY=${PFNN_PROJECT}/env/bin/python
export PYTHONPATH="${KIMODO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "${TAK_ROOT}"
$PY -m pytest tests/test_bones_terrain_fit.py tests/test_bones_terrain_scene.py \
  tests/test_bones_terrain_screen.py tests/test_bones_terrain_survey.py \
  -q --junitxml=runs/bones_fit_v6_coord/terrain_tests.xml
$PY -m pytest tests/test_skeleton_ray_sampler.py -q \
  --junitxml=runs/bones_fit_v6_coord/ray_tests.xml
```

| 测试 | 实际输出 | 跳过 |
| --- | --- | ---: |
| 四个地形文件 | **53 passed in 5.75s** | 0 |
| 骨骼射线 | **6 passed in 1.44s** | 0 |

JUnit 确认 `test_skin_vertices_uses_official_skinner_interface` 和 `test_torch_matches_numpy_reference` 均实际运行通过。见 [测试回执](../../examples/results/bones_terrain_fit_v6_20261023/execution/test_receipt.json)、[地形 JUnit](../../examples/results/bones_terrain_fit_v6_20261023/execution/terrain_tests.xml)、[射线 JUnit](../../examples/results/bones_terrain_fit_v6_20261023/execution/ray_tests.xml)。

站立高度为 1.760514736 m；18,056 个蒙皮顶点的部位计数为其他 7,557、躯干 2,289、手 5,694、脚 2,516。新接口实际得到 **左脚 1,258、右脚 1,258 个顶点**。完整版本和环境变量见 [environment.json](../../examples/results/bones_terrain_fit_v6_20261023/execution/environment.json)。

首次网络更新连接失去响应后，确认旧连接结束再重连，全程保持单一任务 SSH。复用服务器已有的 127 个 v5 Git 对象，校验后只传缺少的代码/文档和 Git 元数据，压缩包为 175,232 bytes；再执行 `git pull --ff-only` 从校验后的 bundle 快进到上述提交。运行期间源码保持该版本。

## 2. 71 条定向检查

按任务的四个来源取并集：41 条 v5 target、10 条 L2 审查、10 条 v5 crouch 保留墙体，以及随机 10 条 v5 `environment_explained` 且至少四箱的 prone/quadruped。合并后 **71 条，没有重叠**。随机池为 56 条，按文件名排序后使用 `Random(0).sample`，抽到 9 条 prone 和 1 条 quadruped。清单、原因和原 screen 文件哈希见 [target_selection.json](../../examples/results/bones_terrain_fit_v6_20261023/execution/target_selection.json)。

```bash
$PY scripts/fit_bones_terrain_support.py --screen-run runs/bones_screen_v6_targets \
  --output runs/bones_fit_v6_targets --viz-per-stem 50 --variants 4 --skin-stride 2 --workers 16
```

71 条拟合、71 张 PNG 全部成功，耗时 **113.8 秒**。主线程打开了每张 PNG，并写下 [71 条图像判断](../../examples/results/bones_terrain_fit_v6_20261023/targets/target_visual_review.md)，文件名覆盖与选择清单一致。

三项停止条件均未触发：

- maximal `skin_support` 超过 2 cm 的片段为零。
- `come_up_50cm_box_R_002__A053` 保留 1 个支撑，`stairs_climbing_up_start_R_101__A301` 保留 7 个支撑。
- 定向集的 known_dimension_support / stairs 没有 raised patch 被 `foot_lower_than_sole` 丢弃。

[v5/v6 逐片段表](../../examples/results/bones_terrain_fit_v6_20261023/targets/target_comparison.md) 包含最小/最大支撑数、patch 检查/保留/丢弃原因、各 patch 的 `foot_drop_median_m` 的中位数、墙状态/类型/漂移、资格和拒绝原因。v5 未保存的新字段为 null。

定向集的箱数在全部 71 条中都未改变。E/G/R 从 **28/8/35 → 29/8/34**，F 从 **141 → 137**；变化主要来自 `mohak_loop_002__A029` 转为 E。10 条旧 crouch 墙体全部保留，图像上仍有 8 条可疑。正常箱子和楼梯保留，部分爬行样例仍有碎片状矮支撑。

## 3. 试跑、三路全量与合并

试跑每词干 2 条，48 worker、4 variants、skin stride 2：

```bash
$PY scripts/fit_bones_terrain_support.py --screen-run runs/bones_screen_v1_full \
  --output runs/bones_fit_v6_pilot_48 --limit-per-stem 2 --viz-per-stem 2 \
  --variants 4 --skin-stride 2 --workers 48
```

196 条拟合和渲染全部成功，**92.9 秒**，外层 94.0 秒，满足零失败、至多 180 秒。通过后才启动三路，各 20 worker：

```bash
$PY -u scripts/fit_bones_terrain_support.py --screen-run runs/bones_screen_v1_full \
  --output runs/bones_fit_v6_S --groups known_dimension_support box_support stairs \
  obstacle_clearance ladder flat_control --viz-per-stem 2 --worst-per-group 3 \
  --gif --variants 4 --skin-stride 2 --workers 20
```

L1 的组为 `crouch kneel wall_contact`，L2 为 `quadruped prone`，输出分别改为对应分片名。

| 任务 | 拟合 | 渲染 | 拟合/渲染失败 | 脚本秒数 | 外层秒数 |
| --- | ---: | ---: | --- | ---: | ---: |
| S | 2,907 | 108 | 0/0 | 1,645.9 | 1,647.2 |
| L1 | 2,131 | 50 | 0/0 | 1,586.7 | 1,588.8 |
| L2 | 1,750 | 38 | 0/0 | 1,892.1 | 1,893.2 |

nohup 监督器保存了实际命令、退出码、每 30 秒内存采样和完成标记；所有退出码为 0，没有重启。最低采样可用内存 **360.40 GiB**。计时包装只记录原函数耗时。为补足专项图评，另从保存的拟合生成 26 张 PNG，其中 L1 16 张、L2 10 张，没有重新拟合。

合并为 6,788 行、11 个组；与 v4/v5 的文件名集合逐组一致，三路源码与五个模块 SHA256 一致。见 [合并回执](../../examples/results/bones_terrain_fit_v6_20261023/merge_receipt.json)。

## 4. 支撑、证据块与墙体

完整比较保留 v5 的全部列，并新增 raised patch 的检查/保留/分原因丢弃、至少四箱片段数、单片最大箱数和沿墙滑动拒绝次数。支撑箱数不含地面。

v4 使用箱顶垂直深度；v5/v6 考虑底面，并采用到上下表面较近的垂直距离。三版均用官方 18,056 蒙皮顶点，30Hz 动作隔两帧采样。

### 支撑超限

每格为至少一个采样帧超过 2 cm 的片段数，依次为 minimal / maximal：

| 组 | 片段数 | v4 | v5 | v6 |
| --- | ---: | --- | --- | --- |
| box_support | 48 | 3 / 7 | 0 / 0 | 0 / 0 |
| crouch | 1,143 | 2 / 4 | 0 / 0 | 0 / 0 |
| flat_control | 900 | 0 / 0 | 0 / 0 | 0 / 0 |
| kneel | 638 | 11 / 35 | 0 / 0 | 0 / 0 |
| known_dimension_support | 1,760 | 7 / 177 | 0 / 0 | 0 / 0 |
| ladder | 64 | 2 / 2 | 0 / 0 | 0 / 0 |
| obstacle_clearance | 86 | 0 / 0 | 0 / 0 | 0 / 0 |
| prone | 919 | 402 / 595 | 0 / 0 | 0 / 0 |
| quadruped | 831 | 118 / 311 | 0 / 0 | 0 / 0 |
| stairs | 49 | 4 / 17 | 0 / 0 | 0 / 0 |
| wall_contact | 350 | 1 / 4 | 0 / 0 | 0 / 0 |

[maximal bug 清单](../../examples/results/bones_terrain_fit_v6_20261023/maximal_over_2cm_bug_clips.json) 为空，保留墙体和拟合顶棚的超限片段也均为零。各组地面深度分位数与 v5 相同，完整值见比较表。

### Raised patch 与箱数

raised patch 指高于估计地面的脚底接触证据。表中 B = `body_under_sole`，F = `foot_lower_than_sole`；同一 patch 可同时命中两项，不能直接把原因次数相加当作唯一丢弃数。v6 的旧字段 `patches_resting_on_body` 现在保存两项规则的并集。

| 组 | 检查 | 保留 | B 丢弃 | F 丢弃 | 至少四箱 v4/v5/v6 | 最大箱数 v4/v5/v6 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| box_support | 161 | 159 | 1 | 1 | 4 / 4 / 4 | 5 / 5 / 5 |
| crouch | 53 | 52 | 1 | 0 | 4 / 4 / 4 | 8 / 8 / 8 |
| flat_control | 0 | 0 | 0 | 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| kneel | 569 | 506 | 61 | 33 | 0 / 0 / 0 | 3 / 3 / 3 |
| known_dimension_support | 3,615 | 3,612 | 3 | 0 | 0 / 0 / 0 | 3 / 3 / 3 |
| ladder | 36 | 36 | 0 | 0 | 0 / 0 / 0 | 2 / 2 / 2 |
| obstacle_clearance | 25 | 25 | 0 | 0 | 0 / 0 / 0 | 3 / 3 / 3 |
| prone | 7,197 | 7,115 | 39 | 56 | 392 / 381 / 381 | 19 / 18 / 18 |
| quadruped | 4,306 | 3,544 | 755 | 270 | 75 / 82 / 82 | 13 / 21 / 21 |
| stairs | 242 | 242 | 0 | 0 | 25 / 25 / 25 | 10 / 10 / 10 |
| wall_contact | 95 | 94 | 1 | 0 | 0 / 0 / 0 | 3 / 3 / 3 |
| 合计 | **16,299** | **15,385** | **861** | **360** | 不适用 | 不适用 |

唯一丢弃数为 **914**。同脚判据的 360 次命中中，**307 个也被旧规则命中，新增剔除 53 个**；其中 prone 新增 43 个、quadruped 7 个、kneel 2 个、box_support 1 个。紧凑表中，11 条片段的箱数变化，21 条的最小/最大尺寸变化，5 条的层高列表变化；两组低姿态的“四箱以上”分布和最大箱数没有下降。

### 墙体

最后一列是候选方向被 `contact_slides_along_plane` 拒绝的次数，同一片段可贡献多次。

| 组 | 保留墙 v4 | v5 | v6 | 沿墙滑动拒绝次数 |
| --- | ---: | ---: | ---: | ---: |
| box_support | 0 | 0 | 0 | 0 |
| crouch | 327 | 58 | 55 | 18 |
| flat_control | 4 | 3 | 3 | 7 |
| kneel | 7 | 3 | 2 | 2 |
| known_dimension_support | 56 | 55 | 50 | 206 |
| ladder | 6 | 6 | 3 | 35 |
| obstacle_clearance | 0 | 0 | 0 | 0 |
| prone | 1 | 0 | 0 | 3 |
| quadruped | 37 | 37 | 33 | 21 |
| stairs | 0 | 0 | 0 | 0 |
| wall_contact | 115 | 112 | 106 | 20 |

wall_contact 覆盖 **106/350 = 30.29%**，超过 25%，没有触发“五条丢失墙体”的条件诊断。保留墙体的 `contact_drift_m` 与逐图判断一起记录在 L1 审查中。

## 5. 目标判定

E 为环境可解释，G 为仅几何一致、低姿态未获环境解释；任务中的可用比例按 E+G。下表的 F 为变体级 `foot_evidence_not_supported` 次数，与上一节的 patch 原因缩写分开。

| 目标 | v5 → v6 | 判定 |
| --- | --- | --- |
| 每组 maximal 支撑超限为零 | 全部 11 组保持零 | **通过** |
| prone：F 大幅下降、E+G 上升、四箱片段大幅下降 | F 1,834 → 1,815（−1.04%）；E+G 464 → 469，50.49% → 51.03%；四箱片段 381 → 381 | **未通过** |
| quadruped：同上 | F 2,211 → 2,207（−0.18%）；E+G 279 → 280，33.57% → 33.69%；四箱片段 82 → 82 | **未通过** |
| known_dimension_support 的新丢弃 ≤1% | 0/3,615 checked patches；受影响 0/1,760 clips | **通过** |
| stairs 的新丢弃 ≤1% | 0/242 checked patches；受影响 0/49 clips | **通过** |
| crouch 保留墙少于 58 | 58 → 55 | **通过数量目标** |
| wall_contact 配墙 ≥25% | 32.00% → 30.29% | **通过** |
| flat_control：100% E，墙 ≤1% | E 900/900；墙 3/900 = 0.33% | **通过** |

[正常组受影响片段清单](../../examples/results/bones_terrain_fit_v6_20261023/normal_group_foot_lower_drops.json) 为空，没有需补列的受影响 PNG。两组低姿态的 E+G 略增，但 F 的降幅很小且四箱分布完全未降，因此联合目标未通过。

全库资格为 **E 4,899 / G 678 / rejected 1,211**，对应 28,905.80 / 3,940.23 / 9,794.94 秒。接受变体 **19,578/27,152 = 72.11%**，v5 为 72.02%。每组资格、时长、所有拒绝原因、地面来源、顶棚和场景变体均在完整比较表中。

## 6. 图片与残留问题

主线程逐张审查 71 张定向 PNG。全量后，S 审查 10 张；L1 审查 10 张 crouch 保留墙、10 张 wall_contact 保留墙及 2 张 kneel；L2 审查每组 5 条 E 片段，优先选残留箱数多的片段，再按文件名哈希选取。全量审查避开定向集，共 **113 张不同 PNG**。

- [71 条定向图评](../../examples/results/bones_terrain_fit_v6_20261023/targets/target_visual_review.md)
- [S：正常支撑与其他组](../../examples/results/bones_terrain_fit_v6_20261023/execution/S_review.md)
- [L1：20 面保留墙与跪姿](../../examples/results/bones_terrain_fit_v6_20261023/execution/L1_review.md)
- [L2：10 条环境可解释的低姿态](../../examples/results/bones_terrain_fit_v6_20261023/execution/L2_review.md)

定向样本 `crawl_ff_loop_225_004__A131` 的 11 个 raised patch 全部保留，`foot_drop_median_m` 的中位数为 **7.38 mm**，低于 30 mm 的同脚判据阈值，8 个小箱仍在。正常箱子和楼梯样例的对应中位数为 4.16 / 5.19 mm。结合全量 307 个重叠命中和仅 53 个新增剔除，新判据对主要碎片支撑问题的覆盖仍很有限。

10 条原有 crouch 墙体在定向阶段全部保留，8 条仍存在收臂、身体外缘或路径侧缘被配墙的问题。接触漂移小的几何条件仍可能与缺少明确压墙动作的样例同时出现；新增的 20 张墙体图评单独记录了手/背与墙的对应。

新增 20 条墙体图中，14 条的手撑/背靠位置相符，6 条因收臂姿态或叠帧遮挡仍无法确认压墙关系；其 `contact_drift_m` 都在 0.003735–0.073286 m，低于 0.10 m。10 条 E 低姿态图全部还有局部小箱：7 条至少四箱，另 3 条各两箱；prone 样本可见成排 8–11 cm 宽的小块，静止手膝样本仍有两根孤立小柱。

梯子和越障的高处几何仍是未解决项。方法第 11 节写明这两组保持拒绝，但本次 `eligibility` 字段仍有 **47/64 条梯子、73/86 条越障归为 E**，与该约定不一致；本报告保留实际程序输出。

## 7. 产物位置与发布

服务器相对 `${TAK_ROOT}` 的目录：

| 内容 | 路径 |
| --- | --- |
| 定向筛选与拟合 | `runs/bones_screen_v6_targets/`、`runs/bones_fit_v6_targets/` |
| 48-worker 试跑 | `runs/bones_fit_v6_pilot_48/` |
| 三路完整 fit、GIF、viewer | `runs/bones_fit_v6_S/`、`runs/bones_fit_v6_L1/`、`runs/bones_fit_v6_L2/` |
| 合并表、完整 viewer 索引 | `runs/bones_fit_v6_merged/` |
| 监督日志、计时、审查证据 | `runs/bones_fit_v6_coord/` |

公开目录保存统计、紧凑表、测试和执行回执、逐图判断及审查 PNG。完整 fit JSON、GIF、viewer 动作数据、源 BVH 保留在服务器。私有 Drive 传输后逐文件验证 SHA256，再替换机器路径；数值与 PNG 保持原样。见 [发布清单](../../examples/results/bones_terrain_fit_v6_20261023/publication_manifest.json)、[文件哈希](../../examples/results/bones_terrain_fit_v6_20261023/SHA256SUMS)。
