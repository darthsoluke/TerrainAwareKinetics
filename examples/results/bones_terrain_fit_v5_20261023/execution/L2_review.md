# BONES terrain v5 L2 审阅

2026-10-24。已只读核对[本分片报告](../shards/L2/report.json)、[分组统计](../shards/L2/by_group.json)、[紧凑表](../fits_compact.tsv)中的全部 1,750 行 L2 记录、[逐图证据](L2_review_evidence.json)、[点检清单](L2_review_selection.json)和全部 10 张 PNG。10 个文件名与 main 的 target_visual_review.md 无重叠。本路 20 CPU workers、4 variants、skin_stride=2，拟合 1,750/1,750，失败 0；生成可视化 38 份，本次目检包内 10 份。

**判定：最大支撑穿透超 2 cm 的片段清零；prone、quadruped 的联合目标均未通过。** prone 的可用片段（E+G）增加 96 条，但 F 增加 197 次；quadruped 的可用片段（E+G）持平，F 增加 27 次。

本文“可用”按任务口径计 E+G。E 单列称“环境可解释”（environment_explained），G 为“仅几何一致、低姿态缺少环境解释”（geometry_only_low_posture_unexplained），R 为“拒绝”（rejected）。F = foot_evidence_not_supported，S = solid_penetrates_skin，U = low_posture_unexplained。E/G/R 按片段计；F/S/U 按变体原因命中计，同一变体可命中多个原因。以下对照来自 [v4 基线](../../bones_terrain_fit_v4_20261023/comparison.json)。

| 指标，v4 -> v5 | prone，919 条 | quadruped，831 条 |
| --- | ---: | ---: |
| 环境可解释（E） | 238 -> 301 | 205 -> 206 |
| 仅几何一致、低姿态缺少环境解释（G） | 130 -> 163 | 74 -> 73 |
| 拒绝（R） | 551 -> 455 | 552 -> 552 |
| 可用（E+G） | 368 -> 464 | 279 -> 279 |
| F | 1,637 -> 1,834 | 2,184 -> 2,211 |
| S | 1,923 -> 0 | 428 -> 0 |
| 最小 / 最大支撑超 2 cm 片段 | 402 / 595 -> 0 / 0 | 118 / 311 -> 0 / 0 |
| 接受变体 / 全部变体 | 931/3,676 -> 1,189/3,676 | 811/3,324 -> 824/3,324 |

TSV 重算的组数、资格数、支撑箱分布、flags 和接受变体数与 by_group 一致。F 涉及 prone 461 条、quadruped 554 条，其中分别有 6、2 条仍归入可用（E+G）；R 分别为 455、552 条。

## 逐图判断

各文件名链接到对应 PNG。箱数不含地面；p/g 分别为最大支撑布局的皮肤支撑穿透峰值、支撑间隙绝对值 P95，单位 cm，数值取自[逐图证据](L2_review_evidence.json)；“-”表示无支撑证据点。

| 文件名 | 资格；接受变体 | 箱数；p / g | 逐图判断 |
| --- | --- | --- | --- |
| [crawl_ff_loop_270_R_001__A216](../viz/prone/crawl_ff_loop_270_R_001__A216.png) | 环境可解释（E）；4/4 | 0；0 / - | 匍匐轨迹沿地面展开，顶棚解释率 100%，无额外低箱或身体接触 patch 标记；本组清晰的环境可解释例子。地面皮肤深度峰值 1.1172 cm。 |
| [idle_crawl_right_to_idle_crawl_R_003__A229](../viz/prone/idle_crawl_right_to_idle_crawl_R_003__A229.png) | 环境可解释（E）；4/4 | 4；1.9935 / 1.87 | 仍保留 4 块离散的 7.01 cm 低支撑，最小箱短边 8-11 cm，最大范围相互重叠。环境可解释，碎片化形态仍在，支撑穿透接近 2 cm 门槛。 |
| [idle_crawl_start_001__A233](../viz/prone/idle_crawl_start_001__A233.png) | R；0/4 | 4；1.8777 / 13.54 | 站立与匍匐转换下方残留 4 块 13.02 cm 低箱；记录身体冲突导致的证据削减与候选支撑拒绝。原地转换无法配解释顶棚，F=4、U=4。 |
| [idle_crawl_to_idle_crawl_right_R_003__A232](../viz/prone/idle_crawl_to_idle_crawl_right_R_003__A232.png) | R；0/4 | 1；1.6465 / 6.17 | 仅余一块 12.5 x 11 cm 小支撑，高 6.90 cm；其他接触覆盖仍缺失，记录证据削减与支撑拒绝。顶棚解释 100%，F=4。 |
| [mohak_backward_stop_002__A031](../viz/quadruped/mohak_backward_stop_002__A031.png) | R；0/4 | 1；1.4382 / 11.11 | 接触轨迹局部只留 11 x 10 cm 小箱，身体接触 patch 标记 4 个；记录证据削减与支撑拒绝，足部高度覆盖仍失败，F=4。 |
| [mohak_loop_002__A031](../viz/quadruped/mohak_loop_002__A031.png) | R；0/4 | 1；1.8276 / 11.14 | 约 8 m 的移动轨迹只在末端留 11 x 9.3 cm 小箱；身体接触 patch 标记 15 个。顶棚覆盖整段，但支撑证据覆盖不足，F=4。 |
| [mohak_start_001__A031](../viz/quadruped/mohak_start_001__A031.png) | R；0/4 | 1；1.6885 / 11.91 | 起始局部保留 11 x 11 cm 小箱，身体接触 patch 标记 3 个；记录证据削减与支撑拒绝，F=4。 |
| [on_all_fours_turn_270_R_003__A202](../viz/quadruped/on_all_fours_turn_270_R_003__A202.png) | R；0/4 | 1；1.6085 / 9.80 | 转身轨迹一侧保留 16.9 x 11 cm 支撑，最大范围向外扩展；记录证据削减，足部高度覆盖仍失败，F=4。 |
| [on_all_fours_turn_360_R_002__A201](../viz/quadruped/on_all_fours_turn_360_R_002__A201.png) | R；0/4 | 0；0 / 13.70 | 图中已无额外箱，但保留 12.45 cm 支撑证据层、103 个证据点；地面与目标接触高度仍失配，F=4。 |
| [turn_crawl_270_R_003__A229](../viz/prone/turn_crawl_270_R_003__A229.png) | R；0/4 | 4；1.9349 / 8.29 | 转向轨迹周围仍有 4 块离散的 8.18 cm 低支撑；记录证据削减与候选支撑拒绝。顶棚解释 100%，接触覆盖仍失败，F=4。 |

10 图合计可用（E+G）2/10 条，其中环境可解释（E）2 条、G 0 条、R 8 条；接受 8/40 个变体，F=32、U=4、S=0。9 图有解释率 100% 的顶棚，1 图为原地转换拒配。3 张 mohak 图合计标记 22 个身体接触 patch。证据清单的 floating_slab_clips、maximal_over_2cm_bug_clips 均为空，10 图的 slabs 也均为空。

## 主要残留问题

- **支撑碎片仍多。** prone 有 649/919 条保留额外支撑，其中 381 条至少 4 箱，单条最多 18 箱；quadruped 为 487/831 条、82 条至少 4 箱、最多 21 箱。逐图可见 8-11 cm 短边的局部小箱，既出现在 R，也出现在 E 样本中。
- **身体接触与足部覆盖仍是主要缺口。** prone/quadruped 的 foot_patches_rest_on_body 标记为 33/230 条，evidence_reduced_for_body_conflict 为 577/573 条，support_rejected 为 356/487 条。F 检查 patch 质心与目标支撑面的高度差是否在 3 cm 内；unsupported_evidence=0 只表示存在可查询表面，地面仍可留下很大的高度 gap。8 张拒绝图的 gap P95 为 6.17-13.70 cm。
- **地面深度另有残留。** TSV 的 ground_skin_max_m 列值超过 2 cm：prone 25 条，峰值 4.08 cm；quadruped 32 条，峰值 3.61 cm。其中 E/G/R 分别为 13/5/7、28/2/2。S 的实现只检查非地面实体，地面深度单独记录；该定义已与本次 report 中 SHA 一致的本地拟合、场景源码核对。

本次仅提交审阅结论，参数保持原值。
