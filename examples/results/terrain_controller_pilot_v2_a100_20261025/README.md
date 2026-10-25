# A100 pilot v2：混合一步监督与更强穿透代价

源码`62e72d5`，2026-10-25执行。84项测试通过；从v1教师阶段checkpoint新增1,500次更新，耗时26分48.076秒，两个rank正常结束并与最终checkpoint逐参数一致。

**六项验收通过5项，着地脚速度仍未通过。** 一步误差0.507091cm，比保持基线改善45.95%；穿透超限比例0.1953125%，低于源动作0%＋0.5个百分点门槛。着地脚速度0.174437m/s，源动作0.012061m/s，门槛0.068092m/s。结果未部署。

| 文件 | 内容 |
| --- | --- |
| [完整执行记录](../../../docs/records/terrain_controller_pilot_v2_a100_20261025.md) | 预检、命令、耗时、显存、验收、分组退步及逐图观察 |
| [report.json](report.json) | 输入/参数哈希、两rank回执、v1新评估、计划判定和独立账本 |
| [evaluation_table.tsv](evaluation_table.tsv) | 教师起点参考和v2七行原生评估，含源基线与全部闭环代价 |
| [v1_new_evaluation.tsv](v1_new_evaluation.tsv) | v1最终checkpoint使用新口径的完整结果，零训练更新 |
| [by_group.tsv](by_group.tsv) | 同一256窗口上的v1/v2最终分组结果 |
| [matching_updates.tsv](matching_updates.tsv) | 累计1750、2000、2500、3000位置的v1/v2对照 |
| [training_curves.png](training_curves.png) | 损失、误差、穿透和着地脚速度；负损失点保留 |

v2剩余8个穿透超限帧来自crouch 3帧和wall_contact 5帧；wall_contact的7个窗口穿透比例从v1的0升至4.464286%。

正式v2账本：**1,536,000个新窗口、14,336,000个闭环生成帧、1,536,000个混合teacher一步预测、1,500次控制器全局更新**。继承的1,500次教师更新及本轮2次单卡预检分别记账。

## 同一组六个窗口

沿用v1训练前固定的演员、起始帧和场景变体；源关节数组与v1逐值相同。蓝色为源动作、橙色为v2闭环、灰色为对应场景。

| 类别 | 图 | 观察 |
| --- | --- | --- |
| 平地转跑 | [A255](contact_sheets/turn_jog_270_R_003__A255__s40.png) | 整体方向跟随，后半段肩臂和足端偏移。 |
| 侧移收步 | [A030](contact_sheets/walk_sideway_045_stop_002__A030__s60.png) | 保持直立，末段手臂幅度和双足位置仍有差异。 |
| 蹲行 | [A412](contact_sheets/item_pick_up_crouch_walk_R_003__A412__s32.png) | 保持低姿态，抬脚仍不足。 |
| 匍匐转换 | [A129](contact_sheets/change_idle_crawl_to_idle_crawl_right_001__A129__s48.png) | 躯干基本重合，手脚有小偏移。 |
| 50cm上箱 | [A224](contact_sheets/come_up_50cm_box_R_002__A224__s24.png) | 抬身和收腿基本跟随，末段头部及足端有偏移。 |
| 50cm上箱 | [A164](contact_sheets/come_up_50cm_box_R_003__A164__s24.png) | 结束姿态接近源动作，头部靠近顶板，足端有小偏移。 |

![v1与v2训练和固定验证对照](training_curves.png)

原始日志、checkpoint、模型权重和运动数组留在A100，此目录仅包含小型统计与审查图。
