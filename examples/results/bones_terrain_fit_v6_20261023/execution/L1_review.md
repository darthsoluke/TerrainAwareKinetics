# BONES terrain v6 L1 审查

拟合、渲染和墙数量门槛通过；墙接触语义仍有 6 条待复核。已逐张检查指定 22 PNG：crouch 10、wall_contact 10、kneel 2。20 条保留墙中，14 条位置与手撑/背靠姿态相符（crouch 6、wall_contact 8），6 条实际压墙未确认（crouch 4、wall_contact 2）；另 2 条 kneel 无墙。PNG 为骨架叠帧，未绘制掌面、皮肤接触点；下表逐条记录可见的接触部位和判读疑点。

来源：[report](../shards/L1/report.json)、[by_group](../shards/L1/by_group.json)、[TSV](../fits_compact.tsv)、[证据](L1_review_evidence.json)、[选图清单](L1_review_selection.json)。源码版本 `5a015448fe02321a75cb49d26a31445e10728c26`；拟合 2131/2131，失败 0；分片渲染 50/50，失败 0；耗时 1586.7 s，20 workers。本次审图使用主代理排除 71 targets 后的清单。

E=`environment_explained`，G=`geometry_only_low_posture_unexplained`，R=`rejected`；任务可用按 E+G 统计。

| 组 | 拟合 | E | G | R | 可用 E+G | 保留墙（brace / lean） |
|---|---:|---:|---:|---:|---:|---:|
| crouch | 1143 | 1109 | 30 | 4 | 1139 | 55（46 / 9） |
| kneel | 638 | 214 | 319 | 105 | 533 | 2（0 / 2） |
| wall_contact | 350 | 315 | 28 | 7 | 343 | 106（8 / 98） |
| 合计 | 2131 | 1638 | 377 | 116 | 2015（94.56%） | 163（54 / 109） |

crouch 墙数 55 < 58，较主代理提供的 [v5 基线](../../bones_terrain_fit_v5_20261023/comparison.json) 少 3 条；wall_contact 覆盖 106/350=30.29%，高于 25%，不触发额外 5 条丢墙诊断。墙状态累计：crouch `no_contact_plane` 679、`no_still_segment` 409；kneel `no_contact_plane` 636；wall_contact `no_contact_plane` 244。

`scene.rejection_reasons` 原始累计如下；R 列记录整条拒绝数，以下记录原因出现次数。

| 组 | foot_evidence_not_supported | low_posture_partly_explained | low_posture_unexplained |
|---|---:|---:|---:|
| crouch | 19 | 76 | 52 |
| kneel | 420 | 0 | 1464 |
| wall_contact | 28 | 28 | 108 |

抽查的 20 条保留墙 `contact_drift_m` 为 0.003735 至 0.073286，均低于 0.10 m；`wall_1` 皮肤最大穿透均为 0。全分片 minimal/maximal 支撑盒超过 2 cm 的片段数均为 0；证据中的 `floating_slab_clips`、`maximal_over_2cm_bug_clips` 均为空。`kneeling_stop_003__A097` 有独立的地面皮肤穿透：最大 0.030533 m，7/41 个采样帧超过 2 cm，仍保留原始 G 分类。

下表漂移单位 m，保留六位小数；N/A 表示没有拟合墙。候选拒绝码：N=`no_trunk_or_bracing_hand_contact`，F=`feet_at_the_plane`，M=`plane_must_move_to_clear_body`，D=`contact_slides_along_plane`。这些计数来自 `wall_report.rejected`，记录同条动作内淘汰的墙候选。场景原因 L=`low_posture_unexplained`；“无”表示该条 `rejection_reasons` 为空。

| 组 / PNG | 类型；contact_drift_m | 手/背压墙逐图判断 | 分类；场景原因；墙候选拒绝 |
|---|---|---|---|
| crouch / [crouch_ff_start_315_R_001__A195](../viz/crouch/crouch_ff_start_315_R_001__A195.png) | brace；0.007762 | 未确认：墙在起步端手侧，起始叠帧与斜视遮住手端贴墙关系。 | E；无；N22 |
| crouch / [crouch_ff_start_315_R_003__A246](../viz/crouch/crouch_ff_start_315_R_003__A246.png) | brace；0.043030 | 未确认：低蹲起步时双臂屈曲，手端朝墙撑压的姿态不清晰。 | E；无；N23 |
| crouch / [crouch_ff_stop_180_R_002__A195](../viz/crouch/crouch_ff_stop_180_R_002__A195.png) | brace；0.036261 | 手撑位置相符：停止末段屈肘前伸，墙位于前伸手端一侧。 | E；无；N16、F2 |
| crouch / [crouch_ff_stop_180_R_003__A195](../viz/crouch/crouch_ff_stop_180_R_003__A195.png) | brace；0.028154 | 手撑位置相符：末段保持屈膝、前伸手姿态，墙邻接手侧。 | E；无；N17、F2 |
| crouch / [crouch_ff_stop_270_R_001__A197](../viz/crouch/crouch_ff_stop_270_R_001__A197.png) | brace；0.045548 | 手撑位置相符：3D 末段手端朝向窄侧墙，躯干与脚位于墙内侧。 | E；无；N17、F2、M3 |
| crouch / [crouch_ff_stop_270_R_003__A196](../viz/crouch/crouch_ff_stop_270_R_003__A196.png) | brace；0.073286 | 手撑位置相符：末段手侧靠近墙；为本次抽查漂移最大条目。 | E；无；N19、F1 |
| crouch / [crouch_ff_stop_270_R_003__A246](../viz/crouch/crouch_ff_stop_270_R_003__A246.png) | brace；0.057974 | 手撑位置相符：停止处墙位于手端外侧，躯干没有呈背靠姿态。 | E；无；N18、F1 |
| crouch / [idle_crouch_loop_002__A148](../viz/crouch/idle_crouch_loop_002__A148.png) | lean；0.042066 | 未确认：低蹲侧视中墙与手/背轮廓未清晰接合，未见明确背靠姿态。 | E；无；N23 |
| crouch / [idle_crouch_right_loop_R_002__A197](../viz/crouch/idle_crouch_right_loop_R_002__A197.png) | brace；0.061432 | 手撑位置相符：3D 前伸手端接近墙内面，墙位于身体前侧。 | E；无；N20、F2 |
| crouch / [turn_crouch_360_R_003__A195](../viz/crouch/turn_crouch_360_R_003__A195.png) | brace；0.018467 | 未确认：转身叠帧覆盖多个朝向，压墙手及其稳定阶段无法分辨。 | E；无；N17 |
| wall_contact / [drinking_wall_leaning_mug_090_R_001__A285](../viz/wall_contact/drinking_wall_leaning_mug_090_R_001__A285.png) | lean；0.006257 | 背靠位置相符：墙位于后躯干侧，饮水手位于身前，靠墙部位应为背部。 | E；无；N20、F2 |
| wall_contact / [drinking_wall_leaning_mug_090_R_004__A291](../viz/wall_contact/drinking_wall_leaning_mug_090_R_004__A291.png) | lean；0.015412 | 背靠位置相符：侧视墙在后背方向，抬手饮水时躯干位置稳定。 | E；无；N22、F1 |
| wall_contact / [h_b_w_crouch_270_stop_001__A305](../viz/wall_contact/h_b_w_crouch_270_stop_001__A305.png) | brace；0.048801 | 未确认：末段双臂屈曲，墙、前臂与躯干投影重叠，brace 手端落点不清。 | G；L4，0/4 变体接受；N17、F4、M2 |
| wall_contact / [h_b_w_crouch_270_stop_002__A305](../viz/wall_contact/h_b_w_crouch_270_stop_002__A305.png) | brace；0.039710 | 未确认：末段手臂与躯干均叠在墙面投影中，缺少可辨认的手端贴墙关系。 | E；无；N16、F4、D1 |
| wall_contact / [idle_to_wall_leaning_idle_270_R_001__A291](../viz/wall_contact/idle_to_wall_leaning_idle_270_R_001__A291.png) | lean；0.012552 | 背靠位置相符：躯干向后墙靠拢后保持，双手垂下，无手撑姿态。 | E；无；N22、F1 |
| wall_contact / [idle_to_wall_leaning_idle_360_R_002__A282](../viz/wall_contact/idle_to_wall_leaning_idle_360_R_002__A282.png) | lean；0.010479 | 背靠位置相符：侧视上身向墙倾靠，足部留在前方，双臂下垂。 | E；无；N22 |
| wall_contact / [idle_to_wall_leaning_idle_360_R_003__A282](../viz/wall_contact/idle_to_wall_leaning_idle_360_R_003__A282.png) | lean；0.055274 | 背靠位置相符：接近墙后上背停在墙侧，双手未伸向墙。 | E；无；N19 |
| wall_contact / [wall_leaning_idle_270_R_003__A291](../viz/wall_contact/wall_leaning_idle_270_R_003__A291.png) | lean；0.003735 | 背靠位置相符：静止姿态上背靠墙、下肢前置，为抽查最小漂移。 | E；无；N22 |
| wall_contact / [wall_leaning_idle_to_idle_270_R_001__A287](../viz/wall_contact/wall_leaning_idle_to_idle_270_R_001__A287.png) | lean；0.023256 | 背靠位置相符：初始后肩背邻墙，随后离墙站直，双手自然下垂。 | E；无；N16、F4 |
| wall_contact / [wall_leaning_idle_to_idle_270_R_002__A290](../viz/wall_contact/wall_leaning_idle_to_idle_270_R_002__A290.png) | lean；0.039547 | 背靠位置相符：初始后躯干处于墙侧，离墙过渡可见，无手撑要求。 | E；无；N23 |
| kneel / [kneeling_loop_001__A101](../viz/kneel/kneeling_loop_001__A101.png) | 无墙；N/A | 无手/背压墙：双膝低位、手臂垂下，墙状态 no_contact_plane；低顶与跪姿相符。 | E；无；N24 |
| kneel / [kneeling_stop_003__A097](../viz/kneel/kneeling_stop_003__A097.png) | 无墙；N/A | 无手/背压墙：跪姿起身，墙状态 no_contact_plane；地面最大穿透 0.030533 m，7/41 帧 >2 cm。 | G；L4，0/4 变体接受；N6 |

本次样本 E20、G2、R0，任务可用 22/22。两个 G 条目的天花板均为 `no_conflict_free_rectangle`；保留墙判读待复核的 6 条已逐行标出，原始 E/G 标签和可用计数保持不变。
