# A100 pilot v4：教师滑步改善，闭环运行失败

源码`a78b610`，2026-10-26执行。84项测试和从零单卡预检通过，正式训练在update2500评估时触发根四元数一致性断言，退出码1，耗时17分32.304秒。

**实际消耗2,500次更新，最新checkpoint保存到2,250次。** 末250次未保存但计入消耗，H16未开始，剩余500次预算未使用。没有恢复、补跑、修改断言或部署。

update1500的额外门通过：几何着地脚速度从v1教师的0.170082降至v4的0.051630m/s，着地比例33.07%。随后闭环指标严重退化；以下图像及最后分组结果明确使用**保存的2250次checkpoint**。

| 文件 | 内容 |
| --- | --- |
| [完整执行记录](../../../docs/records/terrain_controller_pilot_v4_a100_20261025.md) | 预检、资源等待、命令、失败、可用指标及独立账本 |
| [report.json](report.json) | 输入/参数摘要、实际与保存更新、可用门槛判定及诊断图元数据 |
| [failure_traceback.txt](failure_traceback.txt) | 完整异常堆栈；私人目录替换为变量名 |
| [evaluation_table.tsv](evaluation_table.tsv) | 0至2250的10次完整评估，含源标签与几何速度、双方着地比例 |
| [teacher_comparison.tsv](teacher_comparison.tsv) | update1500时v1与v4的完整指标 |
| [matching_updates.tsv](matching_updates.tsv) | v1–v4共同更新位置；v4的失败及未运行位置明确标注 |
| [by_group.tsv](by_group.tsv) | 三个旧版本终点与v4保存2250的分组结果 |
| [training_curves.png](training_curves.png) | 训练与评估的急剧退化；包含负值和大数值，未截去失败阶段 |

正式训练消耗**2,560,000个窗口、6,144,000个闭环生成帧、1,536,000个教师阶段窗口和1,024,000个混合teacher一步预测**。预检另计5次更新、160窗口、256闭环帧。全部十三个旧checkpoint和前三版执行记录保持不变。

## 六个失败诊断窗口

蓝色为源动作，绿色为v4保存的2250次模型。六个原定窗口都生成了16帧；源数组与此前版本相同。为看清极大漂移，每帧的两台镜头分别跟踪角色，图内打印真实世界根坐标；动作和场景没有修正。

| 窗口 | 观察 |
| --- | --- |
| [转跑 A255](contact_sheets/turn_jog_270_R_003__A255__s40.png) | 身体逐渐落入地面以下，姿态偏离。 |
| [侧移 A030](contact_sheets/walk_sideway_045_stop_002__A030__s60.png) | 躯干扭曲并持续下沉。 |
| [蹲行 A412](contact_sheets/item_pick_up_crouch_walk_R_003__A412__s32.png) | 低姿态未保持，根节点下降至地面以下。 |
| [匍匐 A129](contact_sheets/change_idle_crawl_to_idle_crawl_right_001__A129__s48.png) | 第一帧已远离地面，随后位移增长到极大量级。 |
| [上箱 A224](contact_sheets/come_up_50cm_box_R_002__A224__s24.png) | 穿过箱体后翻倒、下沉。 |
| [上箱 A164](contact_sheets/come_up_50cm_box_R_003__A164__s24.png) | 上箱失败，生成姿态与根位置严重偏离。 |

## 足底曲线

[侧移收步](sole_curves/walk_sideway_045_stop_002__A030__s60__soles.png)和[50cm上箱](sole_curves/come_up_50cm_box_R_002__A224__s24__soles.png)均保留Source、v2、v3，并加入v4。左右脚底点、脚趾点的高度和水平速度分别显示。纵轴为对称对数刻度，以同时读出源动作的小变化和v4的大偏差。

侧移中的v4脚底下降到约-2至-2.7m；上箱中的右脚底最低约-3.30m。它们与完整失败评估相符。

原始训练日志、checkpoint和运动数组留在A100；此目录仅发布小型统计、异常证据和九张图。
