# BONES terrain v9 dev：停止门与场景检查加速

执行日期：**2026-10-25（Asia/Singapore）**；文件名沿用指定的 `20261024`。源码固定为 `8c1d2f8fd877c45cb9d98d8f617a788a95cc2d05`，只运行同一批 593 条 dev 片段，CPU 单池、68 workers，使用已有精确动作缓存，未训练或调参。

**全部规定的停止条件通过。** 77 条 `v7_normal_drop` 标记片段均无同名丢弃，prone F=12、quadruped F=0；564 条 patch 决定未变的片段中，2,256 个变体的 `penetration_m` 和 `accepted` 完全一致。593 条拟合和 60 张渲染均无失败。全量 6,788 条留待下一步，本轮没有运行。

已知尺寸支撑的 F **24→28**，其中 `come_up_50cm_box_R_002__A065` 从 G 转为 R；图中仍是较窄的最小支撑。图审也记录了缺台阶、缺墙和疑似将搬物手势配成墙的情况。

## 测试与执行

- 地形四文件：**60 passed in 4.55s**；射线测试：**6 passed in 1.07s**；零失败、零跳过。
- 实际肢体顶点数：左右脚 **1258 / 1258**，左右腿 **789 / 794**。可用核数 72，每个 worker 的 OMP/OpenBLAS/MKL 为单线程，`CUDA_VISIBLE_DEVICES=''`。
- 从已发布 v8 dev 报告提取 27 个唯一停止样本，写入 `runs/bones_fit_v9_dev_coord/render_list.txt`，与 593 条配置清单逐名核对。

```bash
$PY -u scripts/fit_bones_terrain_support.py \
  --screen-run runs/bones_screen_v1_full --output runs/bones_fit_v9_dev \
  --clips configs/bones_terrain_devset_v1.txt --motion-cache runs/bones_motion_cache_s2 \
  --viz-per-stem 0 --worst-per-group 5 --sample-per-group 3 \
  --render-list runs/bones_fit_v9_dev_coord/render_list.txt \
  --variants 4 --skin-stride 2 --workers 68
```

## 时间

脚本墙钟 **73.6 秒**，外层 **74.276 秒**。v8 首轮为 108.4 秒，缓存重跑为 100.8 秒；v9 比首轮减少 32.10%。v9 墙钟包含 60 张 PNG 的生成，v8 两次都没有渲染。下表累计值是 593 条初次拟合各阶段耗时之和；渲染不在 `stage_seconds` 内。

| 阶段 | v8 首轮累计 s | v8 缓存累计 s | v9 累计 s | v9 P50 / P95 s |
| --- | ---: | ---: | ---: | --- |
| 动作转换 / 缓存读取（含蒙皮） | 2105.4 | 1282.3 | 1433.1 | 2.1094 / 4.8698 |
| screen | 25.0 | 26.2 | 25.0 | 0.0373 / 0.0827 |
| 支撑拟合 | 552.8 | 677.3 | 572.1 | 0.7429 / 2.2744 |
| 场景 | 2363.2 | 2755.6 | 941.8 | 1.3058 / 3.8664 |

全部 593 条命中 `motion_cached`。场景累计耗时相对 v8 首轮减少 **60.15%**。完整计时、参数和模块版本在 [report.json](../../examples/results/bones_terrain_fit_v9_dev_20261024/report.json)。

## 同一 593 条的三版对照

每个三联值依次为 **v7 / v8 / v9**。E=环境可解释，G=仅几何一致，R=拒绝，X=未建模几何排除。F 按场景变体的缺支撑拒绝次数计，四箱按片段计，body-drop 按 patch 计。箱数、资格和穿透按紧凑表逐名对齐；F 与 patch 计数读取各版已保存 fit，没有重新拟合旧版本。

| 组 | 条数 | maximal >2cm | F | 至少四箱 | E/G/R/X（三版） | body_rests_lower |
| --- | ---: | --- | --- | --- | --- | --- |
| box_support | 42 | 0 / 0 / 0 | 32 / 24 / 24 | 3 / 0 / 0 | 9/25/8/0 → 9/27/6/0 → 9/27/6/0 | 14 / 8 / 6 |
| crouch | 49 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 47/2/0/0 → 47/2/0/0 → 47/2/0/0 | 0 / 0 / 0 |
| flat_control | 41 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 41/0/0/0 → 41/0/0/0 → 41/0/0/0 | 0 / 0 / 0 |
| kneel | 42 | 0 / 0 / 0 | 20 / 0 / 0 | 0 / 0 / 0 | 15/22/5/0 → 15/27/0/0 → 15/27/0/0 | 31 / 11 / 11 |
| known_dimension_support | 116 | 0 / 0 / 0 | 4 / 24 / 28 | 0 / 0 / 0 | 108/7/1/0 → 105/5/6/0 → 105/4/7/0 | 82 / 28 / 0 |
| ladder | 41 | 0 / 0 / 0 | 20 / 28 / 28 | 0 / 0 / 0 | 0/0/0/41 → 0/0/0/41 → 0/0/0/41 | 7 / 0 / 0 |
| obstacle_clearance | 40 | 0 / 0 / 0 | 16 / 12 / 16 | 0 / 0 / 0 | 0/0/0/40 → 0/0/0/40 → 0/0/0/40 | 2 / 2 / 0 |
| prone | 81 | 0 / 0 / 0 | 86 / 12 / 12 | 39 / 3 / 3 | 46/14/21/0 → 56/22/3/0 → 56/22/3/0 | 853 / 121 / 121 |
| quadruped | 61 | 0 / 0 / 0 | 24 / 0 / 0 | 6 / 0 / 0 | 51/4/6/0 → 56/5/0/0 → 56/5/0/0 | 423 / 194 / 193 |
| stairs | 40 | 0 / 0 / 0 | 48 / 48 / 48 | 22 / 22 / 22 | 24/4/12/0 → 24/4/12/0 → 24/4/12/0 | 3 / 0 / 0 |
| wall_contact | 40 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 34/6/0/0 → 35/5/0/0 → 35/5/0/0 | 6 / 1 / 1 |

## 一致性与停止门

patch 决定按原顺序比较 `(height_above_ground_m, side, area_m2, bool(dropped))`。诊断比例或冗余丢弃理由改变、而保留/丢弃决定相同的片段仍纳入核对。**564 条决定相同、29 条改变**；相同组的支撑箱完整几何也没有变化。对其全部四个变体逐项比较 `penetration_m` 和 `accepted`，**2,256/2,256 完全一致，零容差、零差异**。

| 停止条件 | 结果 |
| --- | --- |
| 任意 maximal 支撑超限 | 0/593，通过 |
| 指定上箱 / 楼梯丢失支撑 | 分别保持 1 / 7 个，minimal 与 maximal 均保持，通过 |
| 配置中 `v7_normal_drop` 仍触发同名丢弃 | 0/77；v8 的 27 条停止样本全部解除此原因，通过 |
| prone F >24 | 12，通过 |
| quadruped F >12 | 0，通过 |
| 上述逐变体一致性失败 | 0，通过 |

本轮停止门针对配置中的 77 条标记片段；组表中的 `body_rests_lower` 列统计各组所有片段，因此仍保留了其他片段的丢弃计数。

## 60 张 PNG 逐图记录

33 张为每组 3 条固定哈希样本，另加 27 条指定旧停止样本，两集合无重叠；无 worst 追加图。主线程查看了全部原图的 3D 与侧视面板，以 15 页无裁剪四联图逐张检查。下表每条链接到发布的原始 PNG。E/G/R/X 为程序分类，文字记录实际可见的几何。

| # | 片段与 PNG | 图像判断 |
| ---: | --- | --- |
| 1 | [box_jump_down_360_002__A359](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/box_support/box_jump_down_360_002__A359.png) | �����˵������ʼ�㲿�߶ȶ�Ӧ�����������գ�δ��϶��G�� |
| 2 | [come_down_50cm_box_R_001__A330](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_001__A330.png) | ��̨֧�źб������䶥��Ӧ�ߴ���λ���������λ�����⡣ |
| 3 | [come_down_50cm_box_R_001__A359](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_001__A359.png) | ������̨����Ų��߶ȶ�Ӧ���������û�и���С�䡣 |
| 4 | [come_down_50cm_box_R_002__A049](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A049.png) | ������̨֧�ţ�����Сʵ���խ������̬ת��δ������ͣ�G�� |
| 5 | [come_down_50cm_box_R_002__A107](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A107.png) | ��С֧��������ϸ�������Χ���߽Ͽ���խ֧����״���踴�ˡ� |
| 6 | [come_down_50cm_box_R_002__A219](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A219.png) | ��̨�䱣������������䶥ʱǰ�ſ����䵽�����档 |
| 7 | [come_down_50cm_box_R_002__A249](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A249.png) | �ߴ�֧������̨��λ��Ӧ����ض��޶�����Ƭ֧�š� |
| 8 | [come_down_50cm_box_R_002__A320](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A320.png) | ���串����̨λ�ã�ǰ��������򱣳��������档 |
| 9 | [come_down_50cm_box_R_002__A341](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A341.png) | ��ʼ��λ��Ӧ�䶥��ĩ����ؽ�λ�����⣬û������С�顣 |
| 10 | [come_down_50cm_box_R_002__A342](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A342.png) | ̧����̨���������������䣬������ؿռ������� |
| 11 | [come_down_50cm_box_R_002__A359](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_002__A359.png) | ������̨�䱣������ع���δ������Ͱ�֧�ŷָ |
| 12 | [come_down_50cm_box_R_003__A088](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A088.png) | �䶥����̨�㲿�߶ȶ�Ӧ���ʹ�վ��λ�����ڵ����ϡ� |
| 13 | [come_down_50cm_box_R_003__A144](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A144.png) | ��̨ʵ��λ�ڸ߽��·�������̧�Ⱥ����������ա� |
| 14 | [come_down_50cm_box_R_003__A150](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A150.png) | ����֧�ſ���ߴ���λ��Ӧ����ز��޶������塣 |
| 15 | [come_down_50cm_box_R_003__A202](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A202.png) | �����ϵ͵���̨�飬�ߵ����ν�λ�ֱ��Ӧ�䶥�͵��档 |
| 16 | [come_down_50cm_box_R_003__A220](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A220.png) | ��̨֧�ű��ֵ��䣬ǰ����̽ʱû�ж�ʧ����·�֧�š� |
| 17 | [come_down_50cm_box_R_003__A242](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A242.png) | ��������ʼ�㲿��Ӧ����ǰ���û�и���̨�ס� |
| 18 | [come_down_50cm_box_R_003__A322](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A322.png) | ��̨�鱣�����ߴ��㲿���䶥���룬������������� |
| 19 | [come_down_50cm_box_R_003__A359](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A359.png) | ��̨֧�ųʵ������ο飬�����Ƶ������û����Ƭ�䡣 |
| 20 | [come_down_50cm_box_R_003__A378](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A378.png) | �䶥������ʼ��λ��ǰ��վ�����򱣳����ϰ��� |
| 21 | [come_down_50cm_box_R_003__A393](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A393.png) | ��̨�����ڸߴ��㲿�·������·��δ����С֧�š� |
| 22 | [come_down_50cm_box_R_003__A481](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_003__A481.png) | ���䱣�����ʼ��λ��Ӧ����غ���������� |
| 23 | [come_down_50cm_box_R_004__A222](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_004__A222.png) | �ߵͽ�λ�ֱ��Ӧ֧�ſ�����棬δ���ɶ�������̨�ס� |
| 24 | [come_down_50cm_box_R_005__A056](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_down_50cm_box_R_005__A056.png) | ��̨�䱣����̧����ʹ��������û�ж���֧�š� |
| 25 | [come_up_50cm_box_R_001__A206](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_up_50cm_box_R_001__A206.png) | ����֧�Ÿ������ոߴ���λ����ʼվ�����򱣳ֵ��档 |
| 26 | [come_up_50cm_box_R_002__A065](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_up_50cm_box_R_002__A065.png) | ��Сʵ������խ�������ָߴ���λ����ʵ�ķ�Χ����ǰΪR�� |
| 27 | [come_up_50cm_box_R_002__A282](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_up_50cm_box_R_002__A282.png) | �����Ľ�λ��Ӧ���䶥�棬��ǰû�ж���֧�ſ顣 |
| 28 | [come_up_50cm_box_R_003__A065](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_up_50cm_box_R_003__A065.png) | ���串�Ǹߴ�������λ��վ���͵���̬���δ��϶��G�� |
| 29 | [come_up_50cm_box_R_004__A065](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/come_up_50cm_box_R_004__A065.png) | �߲㵥�䱣����0.49m�м��ֻ�б��ߣ�û�ж�Ӧ̨�ף���ǰΪR�� |
| 30 | [crawl_ff_loop_270_R_001__A216](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/prone/crawl_ff_loop_270_R_001__A216.png) | ����·�����޽Ż�ϥ��С�䣬�����Ͷ��︲���˶���Χ�� |
| 31 | [crawl_ff_loop_315_005__A131](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/prone/crawl_ff_loop_315_005__A131.png) | �����·�ֻ���������棬�Ͷ��︲������·��������Ƭ֧�š� |
| 32 | [crouch_ff_start_315_003__A148](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/crouch/crouch_ff_start_315_003__A148.png) | ����·���Ϸ��������������û������С�䣬Ҳû�в�ǽ�� |
| 33 | [crouch_ff_start_360_R_003__A197](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/crouch/crouch_ff_start_360_R_003__A197.png) | ������̬�ص���ǰ������δ��϶�����ǽ��E�Ļ����������踴�ˡ� |
| 34 | [h_b_w_crouch_to_stand_270_start_001__A344](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/wall_contact/h_b_w_crouch_to_stand_270_start_001__A344.png) | �������û��֧��С�䣬δ���ǽ�������G���ࡣ |
| 35 | [h_b_w_stand_to_crouch_270_start_003__A344](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/wall_contact/h_b_w_stand_to_crouch_270_start_003__A344.png) | ����·���Ϸ�����˶����û��ǽ�棬��ǽ��ϵδ���֡� |
| 36 | [idle_crawl_to_idle_crawl_right_R_003__A232](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/prone/idle_crawl_to_idle_crawl_right_R_003__A232.png) | ����ת���·���С�䣬�Ͷ��︲��������Χ�� |
| 37 | [idle_turn_360_001__A066](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/flat_control/idle_turn_360_001__A066.png) | ƽ��ԭ��ת��ֻ�е��棬û�ж����䡢ǽ��� |
| 38 | [item_pick_up_crouch_walk_R_003__A412](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/crouch/item_pick_up_crouch_walk_R_003__A412.png) | ����·���������Ͷ��︲�ǣ���;����û����Ƭ֧�š� |
| 39 | [jog_sideway_135_start_001__A031](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/flat_control/jog_sideway_135_start_001__A031.png) | �����ܶ�����ƽ��չ����û������ǽ�塢�����̨�ס� |
| 40 | [jump_form_box_to_safety_roll_360_R_001__A502](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/box_support/jump_form_box_to_safety_roll_360_R_001__A502.png) | ��������֮�⣬�жκ�ĩ�����������Ͱ��飻����������ΪR�� |
| 41 | [jump_over_obstacle_0_75m_R_102__A416](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/obstacle_clearance/jump_over_obstacle_0_75m_R_102__A416.png) | �ڿտ�Խ����û�ж�Ӧ�ϰ�ʵ�壬����ֻ�е��棻X���ౣ���� |
| 42 | [jump_over_obstacle_1_5m_L_101__A416](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/obstacle_clearance/jump_over_obstacle_1_5m_L_101__A416.png) | �ϸ߿�Խ�켣��û���ϰ��壬δ����֧�ţ�����X�� |
| 43 | [jump_over_obstacle_1m_R_102__A305](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/obstacle_clearance/jump_over_obstacle_1m_R_102__A305.png) | ��Խ�켣������������ȱ��Ҫ��������壬����X�� |
| 44 | [jump_to_shoulder_roll_R_001__A305](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/box_support/jump_to_shoulder_roll_R_001__A305.png) | ����֧��������խ��������·��û�е���飬����̬δ�񻷾����ͣ�G�� |
| 45 | [jump_to_shoulder_roll_R_102__A416](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/box_support/jump_to_shoulder_roll_R_102__A416.png) | �ߴ����������ʼ�㲿��Ӧ�����������䵽�������棬δ��϶�� |
| 46 | [kneeling_loop_001__A101](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/kneel/kneeling_loop_001__A101.png) | ��ֹ�����·�û��С�䣬���︲�ǵ���̬��Χ�� |
| 47 | [kneeling_start_R_003__A408](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/kneel/kneeling_start_R_003__A408.png) | վ��ת���˱��ֿտ����棬û�н���С���ǿ�ӵĶ��G�� |
| 48 | [kneeling_stop_003__A097](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/kneel/kneeling_stop_003__A097.png) | ���������·�û��С�䣬ͷ���ռ���գ�����G���ࡣ |
| 49 | [ladder_slide_down_R_002__A300](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/ladder/ladder_slide_down_R_002__A300.png) | �ߴ��»�����û���ݵ����ݼܼ��Σ�����X�ų��� |
| 50 | [ladder_slide_down_R_004__A300](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/ladder/ladder_slide_down_R_004__A300.png) | ��ֱ�»��켣ֻ����棬û��ʵ���ʸ��ṹ������X�� |
| 51 | [ladder_slip_R_002__A300](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/ladder/ladder_slip_R_002__A300.png) | ���䶯��ȱ�����Ӽ��Σ�δ�ù���ƽ̨���棬����X�� |
| 52 | [lift_crate_come_up_50cm_box_R_005__A249](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/known_dimension_support/lift_crate_come_up_50cm_box_R_005__A249.png) | ����֧�Ŵ��ڣ�����ǰ�����Ƹ�����������ǽ����ǽ���Ϳ��ɡ� |
| 53 | [mohak_backward_stop_002__A031](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/quadruped/mohak_backward_stop_002__A031.png) | ��ϥ����ֹͣ�·�û����Ƭ�䣬�������︲��ֹͣ��Χ�� |
| 54 | [mohak_start_001__A031](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/quadruped/mohak_start_001__A031.png) | ��ϥ��·��ֻ����������Ͷ��û�нŻ�ϥ�µ�С���� |
| 55 | [on_all_fours_turn_270_R_003__A202](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/quadruped/on_all_fours_turn_270_R_003__A202.png) | ��ϥԭ��ת���·�û��֧��С�䣬���︲��ת������ |
| 56 | [stairs_climbing_down_loop_R_102__A301](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/stairs/stairs_climbing_down_loop_R_102__A301.png) | �������֧��֮�����м�϶�������½��߶�ǫ̇̄�ף���ǰΪR�� |
| 57 | [stairs_climbing_jog_down_stop_R_003__A300](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/stairs/stairs_climbing_jog_down_stop_R_003__A300.png) | ��ĩ�˱���һ��С�ף�ǰ���½�û�ж�Ӧ���ݣ���Ƭ���Ա���ΪE�� |
| 58 | [stairs_slip_R_003__A300](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/stairs/stairs_slip_R_003__A300.png) | �߸�֧����ɢ���У���Ⱥ͸߶ȱ仯�ϴ�̤��߶��붯����Ӧ�� |
| 59 | [walk_backward_start_001__A029](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/flat_control/walk_backward_start_001__A029.png) | ƽ�غ�����û�ж���̨�ס�ǽ���� |
| 60 | [wall_leaning_idle_360_R_003__A285](../../examples/results/bones_terrain_fit_v9_dev_20261024/viz/wall_contact/wall_leaning_idle_360_R_003__A285.png) | ��ǽidleû������ǽ�棬ͼ��ֻ�е�������壬��ǰΪE�� |

发布内容为本记录、[紧凑 TSV](../../examples/results/bones_terrain_fit_v9_dev_20261024/fits_compact.tsv)、[分组 JSON](../../examples/results/bones_terrain_fit_v9_dev_20261024/by_group.json)、report.json 和 60 张 PNG（9,152,446 bytes）。完整 fit、缓存和 viewer 留在 PRO6000；没有发布动作、模型、GIF 或哈希清单。
