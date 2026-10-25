# A100双卡pilot：be5fc1f

2026-10-25完成，源码固定为`be5fc1fc23526200d9ee604a14fea317748499b3`。84项测试全部通过，正式训练完成3,000次更新，耗时47分12.48秒。两个rank正常退出，全部14次评估的参数差异为0.0，最终checkpoint与两个rank逐参数SHA-256一致。

**六项验收中3项通过、3项未通过。** 一步误差改善29.2243%，低于30%要求；穿透超2cm的帧比例为1.538086%，高于1%要求；滑步代价未低于初始化值。根漂移P50为0.852408cm，骨盆高度误差为1.277630cm，两项均通过。结果未部署，没有追加训练。

| 文件 | 内容 |
| --- | --- |
| [完整执行记录第10节](../../../../docs/records/terrain_controller_pilot_a100_20261024.md#10-be5fc1f-pilot) | 命令、预检、耗时、验收、逐图观察与精确账本；前面的失败历史保留 |
| [report.json](report.json) | 冻结源码和依赖、逐rank回执、参数摘要、预检及正式账本、六个预选窗口 |
| [evaluation_table.tsv](evaluation_table.tsv) | 全部14行固定256窗口评估，包括最后一次重复终点评估和各项闭环代价 |
| [training_curves.png](training_curves.png) | 训练损失与固定验证指标；竖线为教师监督/H4/H8/H16切换位置 |
| [plot_revision.json](plot_revision.json) | 使用对称对数轴保留两次负policy loss，以及修订图的SHA-256 |

## 训练账本

正式pilot：**3,072,000个窗口曝光、14,336,000个训练闭环生成帧、3,000次控制器全局更新、1,500次critic全局更新**。全局batch为1,024，两个rank各用512。

脚本的15,872,000 exposures由教师窗口和闭环生成帧相加得到。全部预检另计24次控制器更新、2,496个窗口和16,832个训练生成帧。正式评估57,344帧和展示96帧均为推理，不计入训练。

## 六个固定窗口

训练前按文件名SHA-256排序固定窗口，没有用模型表现选图。蓝色为源动作，橙色为最终模型的16帧闭环输出，灰色为对应场景。验证集中没有stairs或box_support组，箱体示例来自known_dimension_support。

| 类别 | 对照图 | 观察 |
| --- | --- | --- |
| 平地转跑 | [turn_jog A255](contact_sheets/turn_jog_270_R_003__A255__s40.png) | 行进方向一致，后半段手臂与摆动腿偏差增大。 |
| 平地侧移收步 | [walk_sideway A030](contact_sheets/walk_sideway_045_stop_002__A030__s60.png) | 保持直立，手脚有小幅持续偏移。 |
| 蹲行 | [item_pick_up A412](contact_sheets/item_pick_up_crouch_walk_R_003__A412__s32.png) | 保持低姿态，后半段生成腿更直、抬脚不足。 |
| 匍匐转换 | [change_idle_crawl A129](contact_sheets/change_idle_crawl_to_idle_crawl_right_001__A129__s48.png) | 躯干接近重合，手脚有小偏移。 |
| 50cm上箱 | [come_up A224](contact_sheets/come_up_50cm_box_R_002__A224__s24.png) | 抬身与收腿基本跟随，末段头部和腿端有小偏移。 |
| 50cm上箱 | [come_up A164](contact_sheets/come_up_50cm_box_R_003__A164__s24.png) | 上箱时头部逼近顶板，源与生成骨架大体重合。 |

![训练与固定验证曲线](training_curves.png)

原始日志、模型权重、运动数组及checkpoint保留在A100。此目录只发布精简数值结果与审查图。
