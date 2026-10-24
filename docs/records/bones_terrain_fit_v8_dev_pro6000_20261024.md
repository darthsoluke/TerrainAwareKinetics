# BONES terrain v8 dev：PRO6000 快速迭代记录

执行日期：2026-10-24。源码固定为 `4f153aa73a3295125ceaa4abdcc4c0453cf67f3d`，CPU 单池、68 workers、593 条 dev 片段，首次构建缓存后立即重跑。**停止门未通过：27 条标记片段仍有 `body_rests_lower` 丢弃，共 29 个 patch。** 未继续额外渲染或全量运行，参数保持源码默认值。

## 环境与命令

地形测试 **59 passed in 4.51s**；骨骼射线测试 **6 passed in 1.10s**，均零失败、零跳过。初始化后的 `limb_vertex_counts`：

| foot_left | foot_right | leg_left | leg_right |
| ---: | ---: | ---: | ---: |
| 1258 | 1258 | 789 | 794 |

`nproc` 测得 **72**，worker 数为 72−4=68。测量使用 `env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc`；进程内 `OMP_NUM_THREADS=1` 会让直接调用的 `nproc` 输出 1。每个 worker 的 OMP/OpenBLAS/MKL 仍为单线程，`CUDA_VISIBLE_DEVICES=''`。

```bash
$PY -u scripts/fit_bones_terrain_support.py \
  --screen-run runs/bones_screen_v1_full --output runs/bones_fit_v8_dev \
  --clips configs/bones_terrain_devset_v1.txt --motion-cache runs/bones_motion_cache_s2 \
  --viz-per-stem 0 --worst-per-group 5 --variants 4 --skin-stride 2 --workers 68
```

第二次只将输出改为 `runs/bones_fit_v8_dev_cached`，其余参数相同；没有使用近似的 `--cache-skin`。旧 BVH 与旧运行结果保持只读。

## 时间与缓存一致性

两次均完成 **593 条，拟合/渲染失败为零**。墙钟时间 **108.4 → 100.8 秒**，下降 7.01%。下表每格为 `累计 / P50 / P95` 秒，累计值是并行片段阶段耗时之和。

| 阶段 | 首次 | 缓存重跑 |
| --- | --- | --- |
| 动作转换 / 缓存读取（含蒙皮） | 2105.4 / 3.1983 / 6.8818 | 1282.3 / 1.8543 / 4.3881 |
| screen | 25.0 / 0.0379 / 0.0788 | 26.2 / 0.0391 / 0.0828 |
| 支撑拟合 | 552.8 / 0.7204 / 2.1695 | 677.3 / 0.9062 / 2.5933 |
| 场景 | 2363.2 / 3.3343 / 8.5276 | 2755.6 / 3.9299 / 10.0564 |

每个阶段均统计 593 条。首次全部为 `motion_converted`，第二次全部为 `motion_cached`。逐条去除唯一的 `timing_s` 字段后，按键排序的 JSON **593/593 完全相同，差异 0**，没有使用数值容差。完整 `stage_seconds` 和命令在 [report.json](../../examples/results/bones_terrain_fit_v8_dev_20261024/report.json)。

## 同一 593 条的 v7 / v8 对照

E=环境可解释，G=仅几何一致，R=拒绝，X=未建模几何排除。F 按场景变体的 `foot_evidence_not_supported` 次数计；“四箱”按片段计；最后一列按 patch 计。v7 片段、箱数、穿透与资格取自已发布紧凑 TSV。该 TSV 的拒绝原因已去重且没有 patch 详情，因此 F 和 body-drop 数从同名已保存 fit 补齐，没有重新拟合 v7。

| 组 | 条数 | maximal >2cm v7→v8 | F v7→v8 | 至少四箱 v7→v8 | E/G/R/X v7→v8 | body_rests_lower v7→v8 |
| --- | ---: | --- | --- | --- | --- | --- |
| box_support | 42 | 0 → 0 | 32 → 24 | 3 → 0 | 9/25/8/0 → 9/27/6/0 | 14 → 8 |
| crouch | 49 | 0 → 0 | 0 → 0 | 0 → 0 | 47/2/0/0 → 47/2/0/0 | - |
| flat_control | 41 | 0 → 0 | 0 → 0 | 0 → 0 | 41/0/0/0 → 41/0/0/0 | - |
| kneel | 42 | 0 → 0 | 20 → 0 | 0 → 0 | 15/22/5/0 → 15/27/0/0 | - |
| known_dimension_support | 116 | 0 → 0 | 4 → 24 | 0 → 0 | 108/7/1/0 → 105/5/6/0 | 82 → 28 |
| ladder | 41 | 0 → 0 | 20 → 28 | 0 → 0 | 0/0/0/41 → 0/0/0/41 | - |
| obstacle_clearance | 40 | 0 → 0 | 16 → 12 | 0 → 0 | 0/0/0/40 → 0/0/0/40 | - |
| prone | 81 | 0 → 0 | 86 → 12 | 39 → 3 | 46/14/21/0 → 56/22/3/0 | - |
| quadruped | 61 | 0 → 0 | 24 → 0 | 6 → 0 | 51/4/6/0 → 56/5/0/0 | - |
| stairs | 40 | 0 → 0 | 48 → 48 | 22 → 22 | 24/4/12/0 → 24/4/12/0 | 3 → 0 |
| wall_contact | 40 | 0 → 0 | 0 → 0 | 0 → 0 | 34/6/0/0 → 35/5/0/0 | - |

## 停止条件

最大支撑超限 **0/593**。两个保护样本 `come_up_50cm_box_R_002__A053`、`stairs_climbing_up_start_R_101__A301` 分别保留 **1 / 7 个支撑**。

配置 TSV 有 **77 条**标记为 `v7_normal_drop`；其中 **27 条、29 个 patch**仍被同名规则丢弃，触发用户规定的停止门。各 patch 的实际值如下，单位为米；完整字段见 report.json 的 `dev_validation.audit.stop_conditions`。

| 片段 | 侧 | 离地高度 | body_below_fraction | body_drop_median_m |
| --- | --- | ---: | ---: | ---: |
| come_down_50cm_box_R_001__A330 | left | 0.613770 | 0.666667 | 0.072884053 |
| come_down_50cm_box_R_001__A359 | left | 0.574798 | 0.750000 | 0.053687841 |
| come_down_50cm_box_R_002__A049 | left | 0.657553 | 0.541667 | 0.040036321 |
| come_down_50cm_box_R_002__A107 | left | 0.634651 | 1.000000 | 0.065475941 |
| come_down_50cm_box_R_002__A219 | left | 0.593972 | 1.000000 | 0.111973554 |
| come_down_50cm_box_R_002__A249 | left | 0.620550 | 0.750000 | 0.071651131 |
| come_down_50cm_box_R_002__A320 | left | 0.621093 | 0.500000 | 0.064366043 |
| come_down_50cm_box_R_002__A341 | left | 0.602893 | 1.000000 | 0.162863195 |
| come_down_50cm_box_R_002__A342 | left | 0.575506 | 0.666667 | 0.114740372 |
| come_down_50cm_box_R_002__A359 | left | 0.578990 | 0.500000 | 0.045627266 |
| come_down_50cm_box_R_003__A088 | left | 0.630848 | 0.666667 | 0.068761081 |
| come_down_50cm_box_R_003__A144 | left | 0.620250 | 0.666667 | 0.074584275 |
| come_down_50cm_box_R_003__A150 | left | 0.608067 | 0.750000 | 0.081440419 |
| come_down_50cm_box_R_003__A202 | left | 0.574538 | 1.000000 | 0.082058400 |
| come_down_50cm_box_R_003__A220 | left | 0.613953 | 0.500000 | 0.034947425 |
| come_down_50cm_box_R_003__A242 | left | 0.565244 | 0.666667 | 0.070463538 |
| come_down_50cm_box_R_003__A322 | left | 0.610252 | 1.000000 | 0.154208750 |
| come_down_50cm_box_R_003__A359 | left | 0.582001 | 0.666667 | 0.065652668 |
| come_down_50cm_box_R_003__A378 | left | 0.608984 | 1.000000 | 0.156851411 |
| come_down_50cm_box_R_003__A393 | left | 0.582323 | 0.666667 | 0.092991829 |
| come_down_50cm_box_R_003__A481 | left | 0.545568 | 1.000000 | 0.132785976 |
| come_down_50cm_box_R_004__A222 | left | 0.583633 | 1.000000 | 0.086887598 |
| come_down_50cm_box_R_005__A056 | left | 0.612020 | 0.666667 | 0.084472537 |
| come_up_50cm_box_R_002__A065 | left | 0.683808 | 1.000000 | 0.044537127 |
| come_up_50cm_box_R_002__A065 | right | 0.686884 | 0.937500 | 0.058812708 |
| come_up_50cm_box_R_003__A065 | right | 0.685665 | 0.743243 | 0.051333278 |
| come_up_50cm_box_R_004__A065 | right | 0.685269 | 1.000000 | 0.057097971 |
| jump_form_box_to_safety_roll_360_R_001__A502 | left | 0.198478 | 0.500000 | -0.000457678 |
| jump_form_box_to_safety_roll_360_R_001__A502 | right | 0.206149 | 0.500000 | 0.007213958 |

## 图片审查与产物

本轮命令生成 **0 张 worst PNG**：当前 `choose_renders` 只在最小支撑超限时选择 worst，`--worst-per-group 5` 不会强制每组输出五张。停止门触发后，没有进入额外 5 条 prone/quadruped E 的渲染与图审。

仅发布本记录及 [compact TSV](../../examples/results/bones_terrain_fit_v8_dev_20261024/fits_compact.tsv)、[by_group.json](../../examples/results/bones_terrain_fit_v8_dev_20261024/by_group.json)、[report.json](../../examples/results/bones_terrain_fit_v8_dev_20261024/report.json)。完整两次 fit、测试日志、缓存和审计保留在 PRO6000 的上述运行目录及 `runs/bones_fit_v8_dev_coord/`。使用唯一 SSH 连接传送小代码与聚合结果，没有 Drive 传输或新增哈希清单。
