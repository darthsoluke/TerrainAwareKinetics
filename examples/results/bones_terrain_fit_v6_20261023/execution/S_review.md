# S 分片审查

2026-10-24；源提交 `5a015448fe02321a75cb49d26a31445e10728c26`。已核对全部 2907 条 TSV、分组统计及 10 张选定 PNG。拟合 2907/2907、渲染 108/108 成功，失败均为 0，20 workers，报告耗时 1645.9 秒。任务可用 **2813/2907（96.77%）**；known_dimension_support 与 stairs 的 `foot_lower_than_sole` 丢弃均为 0。

视觉上需保留的观察：越障样本及两张梯子样本未生成对应障碍物或梯体；搬箱上台阶样本生成了手位旁的 `brace` 墙；下楼停止样本只保留末段一级踏面。具体记录见逐图表。

依据：[report.json](../shards/S/report.json)、[by_group.json](../shards/S/by_group.json)、[fits_compact.tsv](../fits_compact.tsv)、[审查证据](S_review_evidence.json)、[选图清单](S_review_selection.json)。

## 分类统计

E=`environment_explained`，G=`geometry_only_low_posture_unexplained`；任务可用按 E+G 计。

| 组 | 总片数 | E | G | rejected | E+G | 可用率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| known_dimension_support | 1760 | 1692 | 19 | 49 | 1711 | 97.22% |
| box_support | 48 | 9 | 29 | 10 | 38 | 79.17% |
| stairs | 49 | 27 | 7 | 15 | 34 | 69.39% |
| obstacle_clearance | 86 | 73 | 3 | 10 | 76 | 88.37% |
| ladder | 64 | 47 | 7 | 10 | 54 | 84.38% |
| flat_control | 900 | 900 | 0 | 0 | 900 | 100.00% |
| 合计 | 2907 | 2748 | 65 | 94 | 2813 | 96.77% |

TSV 与上述分组计数一致。65 条 G 中，63 条低姿态未解释、2 条部分解释。94 条 rejected 均含 `foot_evidence_not_supported`，其中 67 条仅此原因、3 条同时低姿态部分解释、24 条同时低姿态未解释。每片 4 个场景变体，报告 accepted 为 10992/11628；E 均为 4/4，G 均为 0/4，G 的几何结果仍计入任务可用。

## 抬高足底 Patch

| 组 | checked | kept | body_under_sole 丢弃 | foot_lower_than_sole 丢弃 | 受 foot_lower 影响片数/总片数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| known_dimension_support | 3615 | 3612 | 3 | 0 | 0/1760 |
| box_support | 161 | 159 | 1 | 1 | 1/48 |
| stairs | 242 | 242 | 0 | 0 | 0/49 |
| obstacle_clearance | 25 | 25 | 0 | 0 | 0/86 |
| ladder | 36 | 36 | 0 | 0 | 0/64 |
| flat_control | 0 | 0 | 0 | 0 | 0/900 |
| 合计 | 4079 | 4074 | 4 | 1 | 1/2907 |

全部原因的丢弃比例为 **5/4079（0.1226%）**。`foot_lower_than_sole` 为 **1/4079（0.0245%）**，受影响片段 **1/2907（0.0344%）**；箱体组内分别为 1/161（0.6211%）和 1/48（2.0833%）。TSV 的 `foot_patches_rest_on_body` 标志覆盖 5/2907 条（known 3 条、box 2 条）。

重点正常组 known+stairs：`foot_lower_than_sole` 为 **0/3857（0%）**，受影响 **0/1809（0%）**，无需补画此类片段；其他原因有 3/3857（0.0778%）的 `body_under_sole` 丢弃。10 张抽查图中，6 条含抬高 patch，合计 21/21 保留；任意原因丢弃 0/21（0%），受影响 0/10 条。

## 几何统计

2907 条均采用 `penetration_authority=skin`，minimal/maximal 支撑穿透超过 2 cm 的片数均为 0，未支撑证据点为 0。各组 P95 如下；地面采用独立的 `scene.ground_skin_penetration_p95_m` 字段。

| 组 | minimal 支撑 P95 (m) | maximal 支撑 P95 (m) | 场景地面 P95 (m) | 零支撑箱片数 |
| --- | ---: | ---: | ---: | ---: |
| known_dimension_support | 0.0152 | 0.0185 | 0.0150 | 12/1760 |
| box_support | 0.0192 | 0.0192 | 0.0367 | 1/48 |
| stairs | 0.0197 | 0.0198 | 0.0353 | 7/49 |
| obstacle_clearance | 0.0092 | 0.0113 | 0.0235 | 78/86 |
| ladder | 0.0146 | 0.0160 | 0.0451 | 48/64 |
| flat_control | 0.0000 | 0.0000 | 0.0216 | 900/900 |

拟合墙共 56 条（known 50、ladder 3、flat 3），拟合低顶棚共 7 条（known 2、box 1、ladder 3、obstacle 1），顶棚超过 2 cm 的片数为 0。证据中的 `maximal_over_2cm_bug_clips` 为空。唯一 `floating_slab` 为 `ladder_slip_R_104__A301`，TSV 已列 rejected；该片不在本次 10 张选图中。

## 逐图记录

以下 10 张均已查看；E 8 张、G 2 张。穿透数值取证据中的 `skin_support`，地面单独注明。

| PNG | 审查记录 |
| --- | --- |
| [box_jump_down_360_002__A359](../viz/box_support/box_jump_down_360_002__A359.png) | G；0.5788 m 台面与落地轨迹对应，1 个支撑箱、2/2 patch 保留；支撑最大穿透 10.486 mm，低姿态未拟合顶棚。 |
| [come_up_50cm_box_R_001__A206](../viz/known_dimension_support/come_up_50cm_box_R_001__A206.png) | E；0.6078 m 台面承接上台阶终点，2/2 patch 保留；minimal/maximal 支撑穿透 9.910/18.586 mm。证据原始 `metrics.maximal.penetration_max_m` 为 51.859 mm，裁决采用上述 skin_support，TSV 超阈值帧为 0。 |
| [idle_turn_360_001__A066](../viz/flat_control/idle_turn_360_001__A066.png) | E；原地转身保持地面层，0 支撑箱、0 墙、0 抬高 patch；支撑穿透 0，地面最大穿透 13.637 mm。 |
| [jump_over_obstacle_1m_R_102__A305](../viz/obstacle_clearance/jump_over_obstacle_1m_R_102__A305.png) | E；起跳、腾空与落地轨迹清晰，0 支撑箱、0 墙，图中未生成 1 m 障碍物；地面最大穿透 9.844 mm。 |
| [jump_to_shoulder_roll_R_102__A416](../viz/box_support/jump_to_shoulder_roll_R_102__A416.png) | G；1.1065 m 起始台面及近地 0.0690 m 层保留，4 个支撑箱、6/6 patch 保留；滚翻段有近地小支撑块，地面来自皮肤接触；支撑穿透 17.883 mm，未拟合低顶棚。 |
| [ladder_slide_down_R_002__A300](../viz/ladder/ladder_slide_down_R_002__A300.png) | E；可见从高处下滑至地面的轨迹，0 支撑箱、0 墙，未生成梯体；足底支撑证据 68 点，地面最大穿透 17.843 mm。 |
| [ladder_slip_R_002__A300](../viz/ladder/ladder_slip_R_002__A300.png) | E；滑落姿态位于地面上方，0 支撑箱、0 墙、足底支撑证据 0 点，未生成梯体；地面来自皮肤接触，最大穿透 6.842 mm。 |
| [lift_crate_come_up_50cm_box_R_005__A249](../viz/known_dimension_support/lift_crate_come_up_50cm_box_R_005__A249.png) | E；0.5931 m 台面、2/2 patch 保留；搬箱前伸手旁生成 1 面 brace 墙，其稳定段 1.6 秒、接触漂移 50.140 mm；墙穿透 0，支撑穿透 8.144 mm，保留搬运姿态被解释为墙接触这一语义疑点。 |
| [stairs_climbing_jog_down_stop_R_003__A300](../viz/stairs/stairs_climbing_jog_down_stop_R_003__A300.png) | E；只保留末段 0.1731 m 踏面，1 个支撑箱、1/1 patch 保留，前段下降轨迹下方没有完整阶梯；maximal 支撑穿透 13.338 mm。 |
| [stairs_slip_R_003__A300](../viz/stairs/stairs_slip_R_003__A300.png) | E；7 个高低分布的支撑箱覆盖 0.3570 至 2.1661 m 层，8/8 patch 保留；maximal 支撑穿透 19.894 mm，超过 20 mm 的帧为 0。 |
