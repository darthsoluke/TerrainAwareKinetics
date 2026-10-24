# BONES terrain v7：PRO6000 执行与几何审查

实际执行日期：**2026-10-24**。目录日期 `20261023` 沿用任务约定。拟合源码固定为 `c4b776a920d69fc3f997f8e38821f1300837f177`，方法见 [第 12 节](bones_terrain_method_v2_20261023.md#12-v7-修订根据-v6-执行结果)。

**6,788 条全部完成，但本轮总体目标未通过。** 低姿态的缺支撑拒绝大幅减少，剩余多箱片段仍超出门槛；正常支撑三组有 86 条受到新规则影响，丢弃比例均超过 1%。测试、定向检查与 pilot 通过，完整结果和逐图审查在下文保留。

## 1. 环境、代码与测试

全部拟合在 PRO6000 的项目独立 Python 环境中以 CPU 执行，`CUDA_VISIBLE_DEVICES=''`，OMP、OpenBLAS、MKL 均为单线程。沿用 v6 的 Kimodo、PYTHONPATH、4 个场景变体和 skin stride 2，使用官方 **18,056 个蒙皮顶点**。主线程负责唯一 SSH 会话和 Git；S、L1、L2 子线程分别准备统计与专项审查。没有训练或参数调优。

代码更新使用已校验的缓存 Git 对象和差量 bundle，`git pull --ff-only` 到上述版本；复用 104 个缓存 blob。源 BVH、旧运行目录、cloth 环境保持原样。

```bash
$PY -m pytest tests/test_bones_terrain_fit.py tests/test_bones_terrain_scene.py \
  tests/test_bones_terrain_screen.py tests/test_bones_terrain_survey.py \
  -q --junitxml=runs/bones_fit_v7_coord/terrain_tests.xml
$PY -m pytest tests/test_skeleton_ray_sampler.py \
  -q --junitxml=runs/bones_fit_v7_coord/ray_tests.xml
```

- 地形测试：**55 passed in 5.81s**。
- 骨骼射线测试：**6 passed in 1.45s**。
- 两组均无失败或跳过；JUnit 核实官方 skinner 接口及 `test_torch_matches_numpy_reference` 实际运行并通过。

## 2. 71 条定向对照

v6 的 71 条定向输入逐字节复制到新的 `runs/bones_screen_v7_targets/`；两份 JSONL 的 SHA256 与 v6 一致。执行：

```bash
$PY scripts/fit_bones_terrain_support.py --screen-run runs/bones_screen_v7_targets \
  --output runs/bones_fit_v7_targets --viz-per-stem 50 --variants 4 --skin-stride 2 --workers 16
```

**71 条拟合、71 张 PNG 全部成功，耗时 91.0 秒。** 主线程查看每张 PNG 的 3D 和侧视图，并记录一条判断。为了按顺序检查，完整图片每四张缩放排成一页，没有裁剪；结果目录保留原图。

三项停止条件均未触发：maximal 支撑超过 2 cm 的片段为零；指定箱子和楼梯分别保留 **1 / 7 个支撑**；known_dimension_support / stairs 没有 patch 被 `body_rests_lower` 丢弃。

逐片段表保存 v6/v7 的 patch 检查数、保留数、分原因丢弃数、`body_drop_median_m` 中位数、最小/最大箱数、资格与拒绝原因。v6 没有记录的身体高度字段为 null。

| 定向组 | 条数 | 支撑箱总数 v6→v7 | 至少四箱片段 v6→v7 | 缺支撑拒绝 F v6→v7 | E / G / R v6→v7 |
| --- | ---: | --- | --- | --- | --- |
| 匍匐 prone | 27 | 161 → 73 | 22 → 11 | 49 → 4 | 12/3/12 → 18/8/1 |
| 手膝 quadruped | 22 | 41 → 0 | 4 → 0 | 76 → 0 | 3/0/19 → 21/1/0 |

E 表示环境可解释，G 表示仅几何一致、低姿态未获环境解释，R 表示拒绝。F 为四个场景变体中 `foot_evidence_not_supported` 的总命中次数。箱数不含地面。

71 条中，35 条箱数下降、33 条不变、3 条增加。手膝样本的脚下小箱全部清除，匍匐样本仍有 13 条带支撑箱；旧的 10 条蹲行侧墙全部保留，多数在图中仍缺少明确压墙关系。指定的上箱与七级楼梯几何完整保留。

## 3. Pilot 与三路全量

```bash
$PY scripts/fit_bones_terrain_support.py --screen-run runs/bones_screen_v1_full \
  --output runs/bones_fit_v7_pilot_48 --limit-per-stem 2 --viz-per-stem 2 \
  --variants 4 --skin-stride 2 --workers 48
```

Pilot **196 条拟合、196 张渲染、零失败，92.4 秒**，外层 93.4 秒。满足零失败且不超过 v6 的两倍，即 **185.8 秒**。

通过定向与 pilot 门槛后启动 S、L1、L2，各 20 worker，使用 nohup 监督器保存命令、退出码、每 30 秒的内存采样与完成标记。计时包装只记录原拟合和渲染函数的耗时。

```bash
$PY -u scripts/fit_bones_terrain_support.py --screen-run runs/bones_screen_v1_full \
  --output runs/bones_fit_v7_S --groups known_dimension_support box_support stairs \
  obstacle_clearance ladder flat_control --viz-per-stem 2 --worst-per-group 3 \
  --gif --variants 4 --skin-stride 2 --workers 20
```

L1 的组为 `crouch kneel wall_contact`，L2 为 `quadruped prone`，输出目录分别为 `runs/bones_fit_v7_L1`、`runs/bones_fit_v7_L2`。源版本及所有拟合参数在本轮保持固定。

| 分片 | 拟合 | 渲染 | 拟合/渲染失败 | 脚本秒数 | 外层秒数 |
| --- | ---: | ---: | --- | ---: | ---: |
| S | 2,907 | 108 | 0/0 | 1,615.4 | 1,616.6 |
| L1 | 2,131 | 50 | 0/0 | 1,549.2 | 1,550.7 |
| L2 | 1,750 | 38 | 0/0 | 1,673.7 | 1,674.7 |

合并得到 **6,788 条、11 个组**。文件名集合与 v6 一致，v5/v6 历史指标取自已保存的比较结果。三路源码与五个模块 SHA256 一致，最低采样可用内存 **369.94 GiB**。

收尾阶段 SSH 两次被远端断开。每次确认旧会话退出后才重建唯一连接，并核对完成标记及待写文件；后台拟合没有中断，两个断线时发出的命令均尚未执行，没有重复启动任务。

## 4. 匍匐残留小箱的实测原因

定向图中，部分片段的高处足底证据全部被丢弃，却仍有多个小箱。通过读取四条已保存的 v6/v7 fit，核对了足底证据的高度、归属层和最终聚类；没有重新拟合。

以 `crawl_ff_loop_225_004__A131` 为例：

1. 新规则检查的 **11 个**高处足底 patch 全部因 `body_rests_lower` 被丢弃。
2. 另有两块足底证据，距估计地面分别 **5.956 cm / 4.596 cm**，低于 6 cm，因此在 [检查入口](../../motionbricks/data/bones_terrain_fit.py#L426) 直接保留。
3. [支撑层归属](../../motionbricks/data/bones_terrain_fit.py#L452) 允许向上匹配 3 cm；这两块证据都匹配到 **6.668 cm** 高的旧层，分别上调 **0.711 cm / 2.071 cm**。
4. 该层随后加入 **22 个膝部接触事件**，形成 9 个空间簇，其中 8 个生成箱子：**1 个来自足底，7 个来自膝部**，另一个簇因身体冲突被拒绝。

v6 的同一层包含 13 个足底 patch 和同样 22 个膝部事件，得到 8 个箱子，其中 3 个仅有足底证据、4 个为足底与膝部混合、1 个仅有膝部。v7 移除了连接部分证据的足底 patch，膝部事件仍能在被保留的高度层上形成离散支撑。

另外三条对照也显示，删 patch 后箱数不一定单调下降：`crawl_ff_start_225_002__A133` 的簇数和箱数从 5 增到 6；`crawl_ff_stop_225_006__A131` 的簇数从 11 增到 13，成功箱数从 9 增到 10；`crawl_ff_start_360_002__A130` 从 6 箱增到 7 箱。

因此，剩余碎片的来源已经具体定位到 **低于 6 cm 的保留证据、向上归层，以及后加入的膝部接触聚类**。地面本身仍是一整块连续实体。场景资格检查包含穿透、保留证据的支撑覆盖和顶棚解释率，没有“小箱数量”这一项，因此 E 片段仍可能带有碎片支撑。


## 5. v5 / v6 / v7 全量比较

[完整逐组表](../../examples/results/bones_terrain_fit_v7_20261023/comparison.md) 保留全部 v6 字段，包含地面深度、顶棚、墙拒绝原因、场景变体、悬空板尺寸和资格时长；JSON 与 TSV 同步提供。以下表格展示本轮的主要变化。

11 个组的 minimal / maximal 支撑超限均为零，保留墙与拟合顶棚的超限也为零。检查使用全部蒙皮顶点、隔两帧采样，阈值为 2 cm。地面深度单独报告，见完整表。

### 全组支撑与资格

| 组 | 条数 | minimal/maximal 超限 v5→v6→v7 | E+G 比例 v5/v6/v7 | F v5/v6/v7 | 至少四箱 v5/v6/v7 | 最大箱数 v5/v6/v7 |
| --- | --- | --- | --- | --- | --- | --- |
| box_support | 48 | 0/0 → 0/0 → 0/0 | 77.08% / 79.17% / 83.33% | 44 / 40 / 32 | 4 / 4 / 3 | 5 / 5 / 4 |
| crouch | 1143 | 0/0 → 0/0 → 0/0 | 99.65% / 99.65% / 99.65% | 19 / 19 / 19 | 4 / 4 / 4 | 8 / 8 / 8 |
| flat_control | 900 | 0/0 → 0/0 → 0/0 | 100.00% / 100.00% / 100.00% | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| kneel | 638 | 0/0 → 0/0 → 0/0 | 83.39% / 83.54% / 94.51% | 424 / 420 / 140 | 0 / 0 / 0 | 3 / 3 / 2 |
| known_dimension_support | 1760 | 0/0 → 0/0 → 0/0 | 97.22% / 97.22% / 97.56% | 196 / 196 / 172 | 0 / 0 / 0 | 3 / 3 / 3 |
| ladder | 64 | 0/0 → 0/0 → 0/0 | 84.38% / 84.38% / 0.00% | 40 / 40 / 32 | 0 / 0 / 0 | 2 / 2 / 1 |
| obstacle_clearance | 86 | 0/0 → 0/0 → 0/0 | 88.37% / 88.37% / 0.00% | 40 / 40 / 36 | 0 / 0 / 0 | 3 / 3 / 3 |
| prone | 919 | 0/0 → 0/0 → 0/0 | 50.49% / 51.03% / 84.33% | 1834 / 1815 / 582 | 381 / 381 / 150 | 18 / 18 / 16 |
| quadruped | 831 | 0/0 → 0/0 → 0/0 | 33.57% / 33.69% / 86.88% | 2211 / 2207 / 436 | 82 / 82 / 21 | 21 / 21 / 21 |
| stairs | 49 | 0/0 → 0/0 → 0/0 | 69.39% / 69.39% / 69.39% | 60 / 60 / 60 | 25 / 25 / 25 | 10 / 10 / 10 |
| wall_contact | 350 | 0/0 → 0/0 → 0/0 | 98.00% / 98.00% / 99.43% | 28 / 28 / 8 | 0 / 0 / 0 | 3 / 3 / 3 |

### v7 抬高足底证据

| 组 | 检查 | 保留 | 唯一丢弃 | body_under_sole | foot_lower_than_sole | body_rests_lower | body_drop 中位数 mm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| box_support | 161 | 146 | 15 | 1 | 1 | 14 | -76.752 |
| crouch | 53 | 50 | 3 | 1 | 0 | 3 | -73.607 |
| flat_control | 0 | 0 | 0 | 0 | 0 | 0 | - |
| kneel | 569 | 99 | 470 | 61 | 33 | 468 | 74.768 |
| known_dimension_support | 3615 | 3531 | 84 | 3 | 0 | 82 | -74.219 |
| ladder | 36 | 24 | 12 | 0 | 0 | 12 | -67.975 |
| obstacle_clearance | 25 | 22 | 3 | 0 | 0 | 3 | -76.554 |
| prone | 7197 | 443 | 6754 | 39 | 56 | 6750 | 60.856 |
| quadruped | 4306 | 222 | 4084 | 755 | 270 | 4075 | 82.279 |
| stairs | 242 | 239 | 3 | 0 | 0 | 3 | -83.768 |
| wall_contact | 95 | 54 | 41 | 1 | 0 | 41 | -40.473 |

### 标记、悬空板与墙体

| 组 | floating_slab v5/v6/v7 | foot_patches_rest_on_body v5/v6/v7 | patches_resting_on_body v5/v6/v7 | v7 最薄悬空板 m | v7 最低板底距地 m | 墙 v5/v6/v7 | v7 沿墙滑动拒绝 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| box_support | 0 / 0 / 0 | 1 / 2 / 8 | 1 / 2 / 15 | - | - | 0 / 0 / 0 | 0 |
| crouch | 0 / 0 / 0 | 1 / 1 / 3 | 1 / 1 / 3 | - | - | 58 / 55 / 55 | 18 |
| flat_control | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | - | - | 3 / 3 / 3 | 7 |
| kneel | 0 / 0 / 0 | 51 / 52 / 292 | 61 / 63 / 470 | - | - | 3 / 2 / 2 | 2 |
| known_dimension_support | 0 / 0 / 0 | 3 / 3 / 78 | 3 / 3 / 84 | - | - | 55 / 50 / 50 | 206 |
| ladder | 1 / 1 / 0 | 0 / 0 / 11 | 0 / 0 / 12 | - | - | 6 / 3 / 3 | 35 |
| obstacle_clearance | 0 / 0 / 0 | 0 / 0 / 3 | 0 / 0 / 3 | - | - | 0 / 0 / 0 | 0 |
| prone | 0 / 0 / 0 | 33 / 53 / 796 | 39 / 82 / 6754 | - | - | 0 / 0 / 0 | 3 |
| quadruped | 0 / 0 / 0 | 230 / 231 / 665 | 755 / 762 / 4084 | - | - | 37 / 33 / 33 | 21 |
| stairs | 0 / 0 / 0 | 0 / 0 / 3 | 0 / 0 / 3 | - | - | 0 / 0 / 0 | 0 |
| wall_contact | 0 / 0 / 0 | 1 / 1 / 24 | 1 / 1 / 41 | - | - | 112 / 106 / 106 | 20 |

### 未建模组的实际排除前判定

| 组 | 排除数 | 排除前 E | 排除前 G | 排除前 R | 排除片段秒数 | training_eligible |
| --- | --- | --- | --- | --- | --- | --- |
| ladder | 64 | 47 | 9 | 8 | 332.769 | 0 |
| obstacle_clearance | 86 | 73 | 4 | 9 | 457.838 | 0 |

### 正常支撑的新判据丢弃

| 组 | 丢弃 / 检查 patch | patch 比例 | 受影响片段 | 片段比例 | ≤1% |
| --- | --- | --- | --- | --- | --- |
| known_dimension_support | 82 / 3615 | 2.268% | 76 | 4.318% | 未通过 |
| box_support | 14 / 161 | 8.696% | 7 | 14.583% | 未通过 |
| stairs | 3 / 242 | 1.240% | 3 | 6.122% | 未通过 |

丢弃原因可以重叠，“唯一丢弃”按被删 patch 计一次。旧字段 `patches_resting_on_body` 在 v7 中包含三条丢弃规则的并集。身体高度中位数以脚底减去非足部身体最低点计算，正值表示身体更低，负值表示身体更高。v7 没有悬空板，因此厚度和离地板底的最小值为空。

全库共检查 **16,299** 个抬高足底 patch，保留 **4,830** 个，唯一丢弃 **11,469** 个。

正常组共 **99 个 patch / 86 条片段**被 `body_rests_lower` 影响。每条的 `body_below_fraction`、`body_drop_median_m` 和 PNG 均列于 [完整受影响清单](../../examples/results/bones_terrain_fit_v7_20261023/normal_group_body_rests_lower_drops.md)。定向集未触发停止条件，全量审计才发现这些影响；实际箱体变化另见逐图审查。

全库最终资格：**E 5,506 / G 772 / R 360 / X 150**。X 为 `excluded_unmodelled_geometry`；梯子和越障的 E+G 降为零是按组排除的结果。排除前的实际 v7 判定在上表保留。原始变体接受数为 **22,496/27,152 = 82.85%**，该数字仍包含排除组的拟合结果；这两组的最终 `training_eligible` 均为 false。

## 6. 目标逐项判定

| 目标 | 实测 | 判定 |
| --- | --- | --- |
| 每组 maximal 支撑超限为零 | 0/6,788，[bug 清单](../../examples/results/bones_terrain_fit_v7_20261023/maximal_over_2cm_bug_clips.json) 为空 | **通过** |
| prone：F 低于 v6 的 30% | 1,815 → 582，要求至多 544 | **未通过** |
| prone：至少四箱低于 v6 的 10% | 381 → 150，要求至多 38 | **未通过** |
| prone：E+G 比例上升 | 51.03% → 84.33% | **通过** |
| quadruped：F 低于 v6 的 30% | 2,207 → 436，要求至多 662 | **通过** |
| quadruped：至少四箱低于 v6 的 10% | 82 → 21，要求至多 8 | **未通过** |
| quadruped：E+G 比例上升 | 33.69% → 86.88% | **通过** |
| kneel：F 低于 420 | 420 → 140 | **通过** |
| 正常三组 body_rests_lower 丢弃 ≤1% | 已知尺寸 2.268%；箱体 8.696%；楼梯 1.240% | **三组均未通过** |
| 梯子与越障 100% 排除 | 64/64、86/86，training_eligible 均为 false | **通过** |
| crouch 墙 ≤58 | 55 | **通过数量目标** |
| wall_contact 覆盖 ≥25% | 106/350 = 30.29% | **通过** |
| flat_control：100% E，墙 ≤1% | 900/900 E，墙 3/900 = 0.33% | **通过** |

两组低姿态的联合目标均未通过，正常支撑保护门也未通过。所有结果使用同一固定版本，保留失败指标与图片，没有根据这些数值调整参数或重跑拟合。

## 7. 专项图片审查

主线程逐张查看 71 条定向 PNG；全量专项审查另选 10 条正常箱体/楼梯做 v6/v7 对照、5 条旧版 G/R 跪姿、10 条本版新转为 E 的匍匐/手膝。正常组被 `body_rests_lower` 影响的 86 条全部列入 PNG 集合，其中 1 条与正常对照样本重叠。

- [71 条定向图评](../../examples/results/bones_terrain_fit_v7_20261023/targets/target_visual_review.md)
- [S：正常支撑对照与全部受影响片段](../../examples/results/bones_terrain_fit_v7_20261023/execution/S_review.md)
- [L1：5 条旧版 G/R 跪姿](../../examples/results/bones_terrain_fit_v7_20261023/execution/L1_review.md)
- [L2：10 条新转为 E 的低姿态](../../examples/results/bones_terrain_fit_v7_20261023/execution/L2_review.md)

S 查看了全部 95 张当前 PNG 和 10 张旧版对照。**10 对正常样本的支撑箱图形均保持一致**：9 对整图像素相同，`stairs_slip_R_004__A300` 的箱体图形也相同，只删去 1.357 m 的高度层标线并改变标题标记。其被删 patch 的 `body_below_fraction=0.5`、`body_drop_median_m=-0.017264843 m`，实际七箱布局保持不变。

扩展到全部 95 条审查样本，有 9 条箱数变化（4 增、5 减）；86 条箱数相同，其中 70 条的紧凑表支撑尺寸改变。正常组丢弃超标并不等同于每条都丢失箱体。图评分别记录证据、箱数、尺寸和当前姿态；例如 `stairs_climbing_up_start_R_001__A300` 与 `stairs_slip_R_007__A300` 均从 5 箱减到 4 箱，仍可见稀疏支撑和间隙。另有部分楼梯在 v6 就缺台阶，这些旧问题也保留了相同的对照图。

L1 的 5 条样本中，`kneeling_loop_001__A409` 从 R 转为 E，另外 4 条保持 G。五条均没有支撑箱，已有几何保持相同；改善来自足底证据筛选。A097 起身片段仍有最大 3.05 cm 的地面深度，和 v6 一致。

L2 按“v7 箱数优先、再按文件名哈希”选取新转为 E 的片段，避开定向集。10 条中，全部 5 条匍匐和 3 条手膝仍有碎片支撑，2 条手膝停止片段已清除。`crawl_ff_loop_225_001__A133` 从 11 箱增到 13 箱，`on_all_fours_forward_start_R_003__A199` 从 4 箱增到 10 箱；二者都从 R 转为 E。每条均有图像判断和对应数值。

## 8. 产物与复现

服务器相对 `${TAK_ROOT}` 的目录：

| 内容 | 路径 |
| --- | --- |
| 定向输入与结果 | `runs/bones_screen_v7_targets/`、`runs/bones_fit_v7_targets/` |
| 48-worker pilot | `runs/bones_fit_v7_pilot_48/` |
| 三路完整 fit、GIF、viewer | `runs/bones_fit_v7_S/`、`runs/bones_fit_v7_L1/`、`runs/bones_fit_v7_L2/` |
| 全量比较与合并表 | `runs/bones_fit_v7_merged/` |
| 日志、计时、测试与审查 | `runs/bones_fit_v7_coord/` |

公开 [结果目录](../../examples/results/bones_terrain_fit_v7_20261023/README.md) 保存聚合数值、紧凑表、JUnit、执行回执和审查 PNG。完整 fit、GIF、viewer 动作和源 BVH 留在服务器。全部 PNG 共 **191 张、29,329,934 bytes**，其中 181 张为 v7、10 张为正常对照的 v6。

审查补图从已保存 fit 渲染，没有重新拟合：S 83 张、L1 4 张、L2 8 张，共 95 张；各使用 4 个 CPU worker。两份源筛选 JSONL 在发布前再次核对 SHA256。私有 Drive 传输包先校验哈希，再将机器路径替换为角色占位符；数值和 PNG 字节保持原样。

核查入口：[完整比较](../../examples/results/bones_terrain_fit_v7_20261023/comparison.json)、[目标回执](../../examples/results/bones_terrain_fit_v7_20261023/target_results.json)、[四条残留诊断](../../examples/results/bones_terrain_fit_v7_20261023/residual_support_diagnostic.json)、[发布清单](../../examples/results/bones_terrain_fit_v7_20261023/publication_manifest.json)、[SHA256SUMS](../../examples/results/bones_terrain_fit_v7_20261023/SHA256SUMS)。
