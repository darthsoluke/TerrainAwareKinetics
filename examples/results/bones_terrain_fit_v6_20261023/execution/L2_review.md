# BONES terrain v6 L2 复核

结论：prone、quadruped 的联合目标均未通过。F 和任务可用率仅小幅改善，4plus 数量完全不变；所审 10 张 E 样本均保留局部支撑块，其中 prone 五条及 quadruped 的 A129 后退片段仍有明显的成排小箱。

源码版本 `5a015448fe02321a75cb49d26a31445e10728c26`。[分片报告](../shards/L2/report.json)记录拟合 1750/1750、渲染 38/38、失败 0、耗时 1892.1 秒、20 workers。本次按主线程提供的[选择清单](L2_review_selection.json)另审每组 5 张 E 图，清单已排除 main 的 71 targets；逐条核对了 [TSV](../fits_compact.tsv) 与[详细证据](L2_review_evidence.json)。

F 为 `foot_evidence_not_supported` 的变体拒绝计数；E=`environment_explained`，G=`geometry_only_low_posture_unexplained`，任务可用率按 `(E+G)/组内 clips` 计算；4plus 为至少 4 个支撑箱的 clip 数。数据见 [v5 基线](../../bones_terrain_fit_v5_20261023/comparison.json)、[v6 对照](../comparison.json)及 [L2 分组统计](../shards/L2/by_group.json)。

| 组 | clips | F：v5 -> v6 | E：v5 -> v6 | G：v5 -> v6 | E+G 数量及占比：v5 -> v6 | 4plus：v5 -> v6 |
| --- | ---: | --- | --- | --- | --- | --- |
| prone | 919 | 1834 -> 1815，减少 19（1.04%） | 301 -> 306 | 163 -> 163 | 464/919（50.49%）-> 469/919（51.03%），+0.54 个百分点 | 381 -> 381（41.46%） |
| quadruped | 831 | 2211 -> 2207，减少 4（0.18%） | 206 -> 207 | 73 -> 73 | 279/831（33.57%）-> 280/831（33.69%），+0.12 个百分点 | 82 -> 82（9.87%） |

两组 minimal/maximal 支撑穿透超过 2 cm 的 clip 数均为 0，已拟合顶板超过 2 cm 的 clip 数均为 0，unsupported evidence points 均为 0。最大支撑箱数仍分别为 18、21。有效场景变体分别从 1189/3676 增至 1209/3676、从 824/3324 增至 828/3324。

Raised patches 核对如下。删除总数按 `checked-kept` 去重；两类原因可同时命中同一 patch。

| 组 | checked | kept | body_under_sole | foot_lower_than_sole | 两原因重叠 | 去重删除：v5 -> v6 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| prone | 7197 | 7115 | 39 | 56 | 13 | 39 -> 82，新增 43 |
| quadruped | 4306 | 3544 | 755 | 270 | 263 | 755 -> 762，新增 7 |

v5 的 `patches_resting_on_body` 分别为 39、755，与 v6 的 `body_under_sole` 计数一致；新增足部下探判据在 quadruped 中有 263 次与旧原因重叠，独立新增删除仅 7 个。两组命中删除的 clip 数分别从 33 增至 53、从 230 增至 231。删除增加后，4plus 仍未下降。

本次 10 条均为 E、场景变体通过 4/4、tight/loose 顶板解释率均为 1.0；箱数在 minimal/maximal 中一致。共检查 102 个 raised patches、保留 99 个、删除 3 个，全部删除来自 `crawl_ff_stop_225_R_001__A233` 的 `foot_lower_than_sole`。这三个 patch 的 `foot_below_fraction=1.0`，下探中位数为 3.62–3.93 cm；该条仍保留一个 `body_points_under=5`、`foot_below_fraction=0.25`、下探中位数 2.92 cm 的 patch，未达到报告配置的 10 点及 0.5 比例门槛。

下表层高取 `levels_m` 的非零支撑层，相对地面；patch 栏为检查/保留/删除。每行对应一张实际查看的 PNG。

| 组 / 图 | 箱数 | 层高 cm | patches | 小箱残留判断 |
| --- | ---: | ---: | --- | --- |
| prone / [crawl_ff_loop_225_R_003__A214](../viz/prone/crawl_ff_loop_225_R_003__A214.png) | 13 | 6.25 | 11/11/0 | 爬行路径上短条与方块交替排列，含 0.11×0.11 m 小箱，足部支撑碎片化明显。 |
| prone / [crawl_ff_loop_360_R_003__A231](../viz/prone/crawl_ff_loop_360_R_003__A231.png) | 13 | 6.43 | 15/15/0 | 路径后半段呈密集分段条带，夹有多枚 0.11 m 方箱及更窄小块；小箱残留明显。 |
| prone / [crawl_ff_stop_225_R_001__A233](../viz/prone/crawl_ff_stop_225_R_001__A233.png) | 13 | 7.17 | 24/21/3 | 删除 3 个 patch 后仍成排留箱，包含 0.08×0.08 m 小块；停止动作周围的细碎支撑仍明显。 |
| prone / [crawl_ff_stop_315_R_003__A232](../viz/prone/crawl_ff_stop_315_R_003__A232.png) | 13 | 6.39 | 11/11/0 | 侧视中多段窄箱密集相接，夹有 0.08×0.094 m 小块；支撑仍按轨迹切碎。 |
| prone / [crawl_ff_stop_360_R_002__A231](../viz/prone/crawl_ff_stop_360_R_002__A231.png) | 12 | 7.62 | 14/14/0 | 中后段连续出现短箱，末端仍有独立小块，最小一块为 0.091×0.094 m；残留明显。 |
| quadruped / [mohak_backward_loop_001__A121](../viz/quadruped/mohak_backward_loop_001__A121.png) | 4 | 9.01 | 6/6/0 | 两段约 0.5 m 长的窄条加两枚小垫块沿后退路径排列；数量较少，但局部垫高仍突兀。 |
| quadruped / [mohak_backward_loop_001__A129](../viz/quadruped/mohak_backward_loop_001__A129.png) | 10 | 6.92 | 10/10/0 | 足部轨迹下反复排列窄短箱，最窄 0.081 m；是本组最明显的成排碎片化残留。 |
| quadruped / [mohak_idle_loop_001__A129](../viz/quadruped/mohak_idle_loop_001__A129.png) | 2 | 6.19 | 2/2/0 | 静止屈腿后方保留两枚 0.11×0.11 m 方箱，孤立小柱十分清楚；对应 patch 面积仅约 1.21、1.82 cm²，视觉上仍是细小足部证据生成的垫块。 |
| quadruped / [mohak_turn_270_003__A123](../viz/quadruped/mohak_turn_270_003__A123.png) | 2 | 6.25 | 3/3/0 | 转身区域留两块分离支撑，分别为 0.281×0.149 m、0.163×0.11 m；残留局部、数量少，仍有小垫块感。 |
| quadruped / [mohak_turn_270_003__A129](../viz/quadruped/mohak_turn_270_003__A129.png) | 2 | 7.65 | 6/6/0 | 一块 0.366×0.271 m 支撑和一条 0.501×0.11 m 窄条集中在转身足迹内；碎片化较轻，局部垫高仍可见。 |

视觉复核判定：小箱残留问题仍在。7/10 样本至少 4 箱，另外 3 条各 2 箱；E 样本中的成排足迹垫块和 idle 双小柱，与两组 4plus 未下降的结果一致。
