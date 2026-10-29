# A：六个固定窗口的十二张图

2026-10-29 完成逐图审阅。Source 为蓝色，v2 raw 为绿色，display 为橙色虚线，feedback 为紫色点线。每个窗口都是同一段 16 帧、同一场景变体；四种序列在同一坐标系中叠加。足底图的第一帧速度使用最后一帧真实历史计算。渲染采用 CPU，源关节与旧 v2 记录相同，raw 与旧 v2 保存数组的最大误差为 0。

| 图片 | 观察 |
| --- | --- |
| [转向慢跑 A255 接触图](contact_sheets/turn_jog_270_R_003__A255__s40.png) | 三种生成模式大体重合，保持向前移动和转身；后半段脚部与源动作出现偏差。回灌后末帧根漂移从 2.653cm 增至 3.136cm。 |
| [转向慢跑 A255 足底曲线](sole_curves/turn_jog_270_R_003__A255__s40__soles.png) | raw/display 左脚曲线几乎相同，约 0.3s 左踝低于零高度；feedback 改变右脚高度。右脚速度仍明显高，display 在约 0.2s 有短时速度峰。 |
| [侧移停止 A030 接触图](contact_sheets/walk_sideway_045_stop_002__A030__s60.png) | 三种生成骨架大体重合，腿部仅有小幅差别，整体侧移动作保留。display 的根轨迹与 raw 一致。 |
| [侧移停止 A030 足底曲线](sole_curves/walk_sideway_045_stop_002__A030__s60__soles.png) | 生成左脚下降晚于源动作，raw/display 几乎相同。feedback 降低早期右脚速度，但后段摆动轨迹仍偏离源动作。 |
| [蹲行拾取 A412 接触图](contact_sheets/item_pick_up_crouch_walk_R_003__A412__s32.png) | 三种生成模式重合，保持低姿态和蹲行动作；加锁对本窗口的可见骨架影响很小。 |
| [蹲行拾取 A412 足底曲线](sole_curves/item_pick_up_crouch_walk_R_003__A412__s32__soles.png) | 三种生成曲线几乎一致。左脚落下较晚，右脚抬起高度明显低于源动作，后段脚趾高度降至零以下。 |
| [匍匐转换 A129 接触图](contact_sheets/change_idle_crawl_to_idle_crawl_right_001__A129__s48.png) | 三种生成骨架在俯卧低姿态中重合，仅末端有细小差别；根漂移约 0.314cm。 |
| [匍匐转换 A129 足底曲线](sole_curves/change_idle_crawl_to_idle_crawl_right_001__A129__s48__soles.png) | 三种生成曲线基本相同；源左脚下降时生成左脚上升，右脚也高于源轨迹。图中差异没有被脚锁修正。 |
| [上箱 A224 接触图](contact_sheets/come_up_50cm_box_R_002__A224__s24.png) | 三种模式保持上箱过程，支撑腿出现小幅差别；feedback 的末帧根漂移略低于 raw/display。 |
| [上箱 A224 足底曲线](sole_curves/come_up_50cm_box_R_002__A224__s24__soles.png) | 左脚摆动高度较低、相位较晚。右脚加锁降低早期滑移，display 约 0.2s 出现速度峰；随后仍约 0.05–0.18m/s，明显高于源支撑脚。 |
| [上箱 A164 接触图](contact_sheets/come_up_50cm_box_R_003__A164__s24.png) | 三种模式保留抬腿和上箱姿态，骨架大体重合；右支撑腿差别很小，末帧根漂移约 1.87cm。 |
| [上箱 A164 足底曲线](sole_curves/come_up_50cm_box_R_003__A164__s24__soles.png) | 左脚高度峰值低于源动作且较晚，脚速峰值反而提前。右源支撑脚速度接近零，display 的右踝/脚趾在约 0.2s 达到约 0.31/0.35m/s；feedback 减少早期速度，但后段仍有约 0.05–0.18m/s 的滑动。 |

| 窗口 | raw 末帧根漂移 cm | display cm | feedback cm |
| --- | ---: | ---: | ---: |
| turn_jog_270_R_003 A255，s40 | 2.652792 | 2.652792 | 3.135711 |
| walk_sideway_045_stop_002 A030，s60 | 2.165598 | 2.165598 | 2.263230 |
| item_pick_up_crouch_walk_R_003 A412，s32 | 0.743840 | 0.743840 | 0.743821 |
| change_idle_crawl_to_idle_crawl_right_001 A129，s48 | 0.313969 | 0.313969 | 0.313973 |
| come_up_50cm_box_R_002 A224，s24 | 1.716230 | 1.716230 | 1.704640 |
| come_up_50cm_box_R_003 A164，s24 | 1.872800 | 1.872800 | 1.876590 |

这些图显示脚锁主要改变少量支撑帧，不能消除源动作与生成动作的足部相位偏差；上箱窗口还出现显示加锁的短时脚速峰。定量候选判读见 [v2_candidate_gates.tsv](v2_candidate_gates.tsv)：四组均未通过源接触脚速门槛，其余四项通过。
