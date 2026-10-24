# v7 定向集逐图审查

主线程查看了 71 张完整 PNG 的 3D 与侧视图。为便于顺序检查，每四张无裁剪缩放到一页；原图随结果发布。每行给出图像判断，箱数来自同一片段的保存结果。E/G/R/X 分别为环境可解释、仅几何一致、拒绝、未建模几何排除。

| # | 片段与 PNG | v6→v7 箱数 | 图像判断 |
| --- | --- | --- | --- |
| 1 | [change_idle_crawl_to_idle_crawl_right_002__A129](viz/prone/change_idle_crawl_to_idle_crawl_right_002__A129.png) | 4 → 0 | 符合：侧卧转换下面的小箱已清除，地面与整片低顶棚连续。 |
| 2 | [come_up_50cm_box_R_002__A053](viz/known_dimension_support/come_up_50cm_box_R_002__A053.png) | 1 → 1 | 符合：单个箱子保留，箱顶对齐登高后的脚部位置，没有被新规则删掉。 |
| 3 | [crawl_ff_loop_225_004__A131](viz/prone/crawl_ff_loop_225_004__A131.png) | 8 → 8 | 有残留：仍有 8 个沿路径排列的窄小块，匍匐下方被切碎，片段仍为 rejected。 |
| 4 | [crawl_ff_loop_225_R_001__A215](viz/prone/crawl_ff_loop_225_R_001__A215.png) | 14 → 6 | 有残留：14 个箱减少为 6 个，但腹部及腿下仍分散着小块，未形成合理连续支撑。 |
| 5 | [crawl_ff_loop_225_R_003__A233](viz/prone/crawl_ff_loop_225_R_003__A233.png) | 10 → 0 | 符合：10 个脚下小箱清零，顶棚覆盖整条匍匐路径。 |
| 6 | [crawl_ff_loop_270_R_001__A216](viz/prone/crawl_ff_loop_270_R_001__A216.png) | 0 → 0 | 符合：保持无支撑箱，低顶棚与匍匐动作的高度和范围相符。 |
| 7 | [crawl_ff_loop_315_005__A131](viz/prone/crawl_ff_loop_315_005__A131.png) | 0 → 0 | 符合：地面连续，无孤立脚下箱，顶棚覆盖沿途低姿态。 |
| 8 | [crawl_ff_loop_360_R_002__A230](viz/prone/crawl_ff_loop_360_R_002__A230.png) | 9 → 5 | 有残留：仍有 5 个分离的小块，图中没有顶棚，支撑形状仍碎片化。 |
| 9 | [crawl_ff_loop_360_R_003__A230](viz/prone/crawl_ff_loop_360_R_003__A230.png) | 10 → 7 | 有残留：仍有 7 个沿脚部轨迹排列的小块，未拟合顶棚。 |
| 10 | [crawl_ff_start_180_R_001__A214](viz/prone/crawl_ff_start_180_R_001__A214.png) | 10 → 0 | 符合：10 个箱清零，起步路径保留连续地面和完整顶棚。 |
| 11 | [crawl_ff_start_180_R_004__A127](viz/prone/crawl_ff_start_180_R_004__A127.png) | 6 → 5 | 有残留：5 个小箱在身体下方分段排列，虽为 E，碎片支撑仍在。 |
| 12 | [crawl_ff_start_225_002__A133](viz/prone/crawl_ff_start_225_002__A133.png) | 5 → 6 | 有残留：箱数从 5 增至 6，侧视可见身体下方连续分裂的小块。 |
| 13 | [crawl_ff_start_225_R_001__A230](viz/prone/crawl_ff_start_225_R_001__A230.png) | 9 → 0 | 支撑形状改善：脚下箱清零；图中没有顶棚，保留 G 分类。 |
| 14 | [crawl_ff_start_270_R_002__A233](viz/prone/crawl_ff_start_270_R_002__A233.png) | 10 → 0 | 符合：10 个箱清零，匍匐路径与低顶棚连贯。 |
| 15 | [crawl_ff_start_315_001__A130](viz/prone/crawl_ff_start_315_001__A130.png) | 4 → 4 | 有残留：4 个箱全部仍在，集中于躯干和腿的下方。 |
| 16 | [crawl_ff_start_315_001__A132](viz/prone/crawl_ff_start_315_001__A132.png) | 5 → 3 | 有残留：减少到 3 个较宽小块，但仍在无登高动作的身体下方。 |
| 17 | [crawl_ff_start_315_R_003__A232](viz/prone/crawl_ff_start_315_R_003__A232.png) | 10 → 5 | 有残留：10 个箱减到 5 个，沿匍匐路径仍呈离散窄条。 |
| 18 | [crawl_ff_start_360_002__A130](viz/prone/crawl_ff_start_360_002__A130.png) | 6 → 7 | 有残留：6 个箱增到 7 个，侧视仍是一排脚下碎片。 |
| 19 | [crawl_ff_stop_180_R_001__A230](viz/prone/crawl_ff_stop_180_R_001__A230.png) | 3 → 0 | 符合：停步段脚下箱清零，顶棚覆盖低姿态停止范围。 |
| 20 | [crawl_ff_stop_225_006__A131](viz/prone/crawl_ff_stop_225_006__A131.png) | 9 → 10 | 有残留：9 个箱增到 10 个，停止路径下仍有密集小条块。 |
| 21 | [crawl_ff_stop_225_R_003__A215](viz/prone/crawl_ff_stop_225_R_003__A215.png) | 8 → 0 | 支撑形状改善：8 个箱清零；没有顶棚，低姿态保持 G。 |
| 22 | [crouch_ff_start_180_R_001__A147](viz/crouch/crouch_ff_start_180_R_001__A147.png) | 0 → 0 | 墙体可疑：墙在起步位置后侧，身体随后远离，图中缺少明确持续压墙姿态。 |
| 23 | [crouch_ff_start_270_R_001__A246](viz/crouch/crouch_ff_start_270_R_001__A246.png) | 0 → 0 | 墙体可疑：侧墙包围起步段侧缘，手臂呈收拢姿态，压墙关系不清楚。 |
| 24 | [crouch_ff_stop_180_R_001__A247](viz/crouch/crouch_ff_stop_180_R_001__A247.png) | 0 → 0 | 墙体可疑：末段附近配有墙，蹲行停止动作没有清楚显示靠墙承重。 |
| 25 | [crouch_ff_stop_225_R_001__A244](viz/crouch/crouch_ff_stop_225_R_001__A244.png) | 0 → 0 | 墙体可疑：墙跟随停止区外缘；顶棚只覆盖路径前部，低姿态解释不足。 |
| 26 | [crouch_ff_stop_225_R_003__A195](viz/crouch/crouch_ff_stop_225_R_003__A195.png) | 0 → 0 | 墙体可疑：末段手臂收拢而非明显撑墙，墙与动作的必要关系不清楚。 |
| 27 | [crouch_ff_stop_270_R_001__A247](viz/crouch/crouch_ff_stop_270_R_001__A247.png) | 0 → 0 | 墙体可疑：墙位于停止区侧边，动作主要是蹲行收步。 |
| 28 | [crouch_ff_stop_270_R_002__A197](viz/crouch/crouch_ff_stop_270_R_002__A197.png) | 0 → 0 | 墙体可疑：墙沿整段侧缘延展，未见明确手掌或背部持续压墙。 |
| 29 | [crouch_ff_stop_315_R_002__A194](viz/crouch/crouch_ff_stop_315_R_002__A194.png) | 0 → 0 | 墙体可疑：墙覆盖停止区侧边，收臂动作与持续压墙不能从该图对应起来。 |
| 30 | [drinking_wall_leaning_mug_090_R_003__A282](viz/wall_contact/drinking_wall_leaning_mug_090_R_003__A282.png) | 0 → 0 | 位置基本相符：保留墙靠近伸出的手及身体侧后缘，无脚下小箱。 |
| 31 | [drinking_wall_leaning_mug_360_R_003__A282](viz/wall_contact/drinking_wall_leaning_mug_360_R_003__A282.png) | 0 → 0 | 位置相符：墙在后背一侧，后倾姿态与墙面对应，脚下无多余支撑。 |
| 32 | [idle_crawl_right_to_idle_crawl_R_003__A229](viz/prone/idle_crawl_right_to_idle_crawl_R_003__A229.png) | 4 → 0 | 符合：侧卧转正的脚下箱清零，顶棚连续覆盖身体。 |
| 33 | [idle_crawl_start_001__A233](viz/prone/idle_crawl_start_001__A233.png) | 4 → 3 | 有残留：起卧转换下仍有 3 个小箱，站立与伏地之间没有真正登台过程。 |
| 34 | [idle_crawl_stop_001__A128](viz/prone/idle_crawl_stop_001__A128.png) | 4 → 4 | 有残留：4 个箱仍围绕起卧位置，零碎支撑问题保留。 |
| 35 | [idle_crawl_stop_002__A232](viz/prone/idle_crawl_stop_002__A232.png) | 0 → 0 | 符合：起卧转换保持无支撑箱，不强加顶棚。 |
| 36 | [idle_crawl_to_idle_crawl_right_R_003__A232](viz/prone/idle_crawl_to_idle_crawl_right_R_003__A232.png) | 1 → 0 | 符合：原来的孤立小箱清除，低顶棚与伏地转向范围对应。 |
| 37 | [idle_to_wall_leaning_idle_360_R_002__A289](viz/wall_contact/idle_to_wall_leaning_idle_360_R_002__A289.png) | 0 → 0 | 位置相符：靠墙后的背部与保留墙面对应，没有额外脚下箱。 |
| 38 | [jog_backward_loop_002__A036](viz/flat_control/jog_backward_loop_002__A036.png) | 0 → 0 | 符合：平地后退跑只有地面，没有墙、顶棚或孤立支撑。 |
| 39 | [jump_over_obstacle_2m_L_003__A415](viz/obstacle_clearance/jump_over_obstacle_2m_L_003__A415.png) | 1 → 1 | 未解决：跨越段仍缺真实障碍体，末端有小支撑和局部顶棚；排除分类与图中缺项一致。 |
| 40 | [kneeling_start_003__A167](viz/kneel/kneeling_start_003__A167.png) | 0 → 0 | 仍有问题：跪下段无箱，但估计地面高于部分脚部骨架位置，原拒绝状态保留。 |
| 41 | [kneeling_start_R_003__A408](viz/kneel/kneeling_start_R_003__A408.png) | 0 → 0 | 符合：站立转跪姿只有地面，无人为小箱或顶棚。 |
| 42 | [kneeling_stop_003__A051](viz/kneel/kneeling_stop_003__A051.png) | 0 → 0 | 符合：跪姿起身保留空旷空间，脚下没有小箱。 |
| 43 | [kneeling_stop_003__A167](viz/kneel/kneeling_stop_003__A167.png) | 0 → 0 | 符合：跪姿转站立没有额外支撑，未强加低顶棚。 |
| 44 | [ladder_slide_down_R_303__A302](viz/ladder/ladder_slide_down_R_303__A302.png) | 0 → 0 | 未解决：高处下滑动作没有梯档几何；排除分类合理反映该缺项。 |
| 45 | [mohak_backward_loop_001__A030](viz/quadruped/mohak_backward_loop_001__A030.png) | 5 → 0 | 支撑形状改善：5 个小箱清零，手膝路径连续；没有顶棚，保持 G。 |
| 46 | [mohak_backward_loop_001__A127](viz/quadruped/mohak_backward_loop_001__A127.png) | 3 → 0 | 符合：后退手膝爬行只保留地面与顶棚，脚下小箱清零。 |
| 47 | [mohak_backward_loop_002__A127](viz/quadruped/mohak_backward_loop_002__A127.png) | 3 → 0 | 符合：整段后退路径无脚下碎片，低顶棚覆盖完整路径。 |
| 48 | [mohak_backward_loop_003__A127](viz/quadruped/mohak_backward_loop_003__A127.png) | 4 → 0 | 符合：4 个小箱清零，长后退路径的支撑面连续。 |
| 49 | [mohak_backward_start_001__A029](viz/quadruped/mohak_backward_start_001__A029.png) | 0 → 0 | 符合：后退起步没有新增脚下箱，顶棚连续。 |
| 50 | [mohak_backward_start_003__A029](viz/quadruped/mohak_backward_start_003__A029.png) | 1 → 0 | 符合：原有孤立脚下箱清除，后退起步与顶棚范围相符。 |
| 51 | [mohak_backward_start_003__A035](viz/quadruped/mohak_backward_start_003__A035.png) | 0 → 0 | 符合：地面与手膝动作连续，没有把抬脚姿态做成小台阶。 |
| 52 | [mohak_backward_start_004__A030](viz/quadruped/mohak_backward_start_004__A030.png) | 4 → 0 | 符合：4 个脚下小箱清零，较高爬行姿态与顶棚留出空间。 |
| 53 | [mohak_backward_stop_002__A031](viz/quadruped/mohak_backward_stop_002__A031.png) | 1 → 0 | 符合：后退停止区没有孤立小箱，顶棚与停止动作范围对应。 |
| 54 | [mohak_forward_loop_003__A127](viz/quadruped/mohak_forward_loop_003__A127.png) | 3 → 0 | 符合：向前长路径不再有 3 个分散脚下箱，地面连续。 |
| 55 | [mohak_forward_start_001__A129](viz/quadruped/mohak_forward_start_001__A129.png) | 6 → 0 | 符合：前进起步的 6 个小箱清零，手膝动作下方平整。 |
| 56 | [mohak_forward_stop_001__A120](viz/quadruped/mohak_forward_stop_001__A120.png) | 2 → 0 | 符合：前进停止区 2 个小箱清零，地面与顶棚连续。 |
| 57 | [mohak_loop_002__A029](viz/quadruped/mohak_loop_002__A029.png) | 0 → 0 | 符合：长循环保持无小箱，整条路径只有地面和上方顶棚。 |
| 58 | [mohak_loop_002__A031](viz/quadruped/mohak_loop_002__A031.png) | 1 → 0 | 符合：长循环的最后一个小箱清除，路径下方连续。 |
| 59 | [mohak_loop_003__A029](viz/quadruped/mohak_loop_003__A029.png) | 0 → 0 | 符合：长循环无碎片支撑，顶棚范围与路径一致。 |
| 60 | [mohak_start_001__A031](viz/quadruped/mohak_start_001__A031.png) | 1 → 0 | 符合：起步段的孤立脚下箱消失，手膝下方地面连续。 |
| 61 | [mohak_turn_000_004__A030](viz/quadruped/mohak_turn_000_004__A030.png) | 1 → 0 | 符合：原地转向不再保留单个脚下箱，顶棚完整。 |
| 62 | [mohak_turn_090_003__A029](viz/quadruped/mohak_turn_090_003__A029.png) | 2 → 0 | 符合：转向段 2 个小箱清零，没有将脚部姿态转成台阶。 |
| 63 | [mohak_turn_360_001__A120](viz/quadruped/mohak_turn_360_001__A120.png) | 2 → 0 | 符合：整圈转向下面没有孤立支撑柱，顶棚覆盖转向范围。 |
| 64 | [on_all_fours_idle_loop_001__A202](viz/quadruped/on_all_fours_idle_loop_001__A202.png) | 1 → 0 | 符合：静止手膝姿态脚下小柱清除，膝和手对应连续地面。 |
| 65 | [on_all_fours_turn_270_R_003__A202](viz/quadruped/on_all_fours_turn_270_R_003__A202.png) | 1 → 0 | 符合：手膝转向保持连续地面，原来的单个脚下箱消失。 |
| 66 | [on_all_fours_turn_360_R_002__A201](viz/quadruped/on_all_fours_turn_360_R_002__A201.png) | 0 → 0 | 符合：整圈手膝转向没有脚下支撑箱，低顶棚位置连贯。 |
| 67 | [stairs_climbing_up_start_R_101__A301](viz/stairs/stairs_climbing_up_start_R_101__A301.png) | 7 → 7 | 符合：7 级支撑完整保留，逐级升高的箱顶与脚步高度相符。 |
| 68 | [turn_crawl_270_R_003__A217](viz/prone/turn_crawl_270_R_003__A217.png) | 4 → 0 | 符合：伏地转向的 4 个小箱清零，顶棚覆盖转向范围。 |
| 69 | [turn_crawl_270_R_003__A229](viz/prone/turn_crawl_270_R_003__A229.png) | 4 → 0 | 符合：伏地转向的 4 个小箱清零，低姿态下方只剩连续地面。 |
| 70 | [turn_crouch_270_R_002__A195](viz/crouch/turn_crouch_270_R_002__A195.png) | 0 → 0 | 墙体仍需核查：墙靠近前伸双手，但图中是转身收臂，持续压墙关系不明确。 |
| 71 | [turn_crouch_360_R_002__A197](viz/crouch/turn_crouch_360_R_002__A197.png) | 0 → 0 | 墙体可疑：原地蹲转配有整面侧墙，未见清楚的手或背靠墙动作。 |
