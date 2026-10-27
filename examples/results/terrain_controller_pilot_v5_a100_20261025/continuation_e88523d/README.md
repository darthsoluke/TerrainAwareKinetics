# e88523d 修复后的 v5 执行

执行日期：2026-10-27～28。详细过程追加于 [v5 执行记录](../../../../docs/records/terrain_controller_pilot_v5_a100_20261025.md#7-正式运行与训练曲线)。上一层保留此前的失败记录。

正式 pilot 完成 1500 次新增更新，耗时 63 分 8 秒；八项验收通过七项，一步误差未通过。没有发散保护退出，末 250 次 critic loss 中位数为 0.2477。正式训练消耗 2.10449 GPU-hours，含预检共 2.13185 GPU-hours。模型未部署。

| 检查 | 结果 |
| --- | --- |
| 全仓库测试 | 3124 passed、280 failed、275 errors、33 skipped |
| 本轮相关八个文件 | 85 passed、0 skipped |
| 两个指定 normalized critic 反向测试 | 全仓库及单独运行均通过 |
| 原 CPU 零更新复现 | 通过，退出码 0 |
| 双卡两步、每卡 512 预检 | 全部通过，输出目录已删除 |

- [全仓库计数及完整失败清单](full_suite_summary.json)。
- [85 项测试与指定回归测试状态](controller_tests_summary.json)。
- [相关测试完整日志](tests_controller.log)。
- [CPU 反向复现日志](normalized_backward_check_e88523d.log)。
- [本轮源码、教师检查点及数据身份](input_proof.json)。
- [双卡两步预检：各项判定、参数摘要与显存](gpu_preflight.json)。
- [完整机器可读报告](report.json)：最终指标、阶段结果、验收、账本和图片窗口身份。
- [七条原生评估记录](evaluation_table.tsv)：包含末次重复评估和源动作基线。
- [v1～v5 同更新位置对照](matching_updates.tsv)：旧版缺失值标记为 unavailable。
- [最终分组表](by_group.tsv)。
- [1500 次更新的梯度、critic 与训练项](training_updates.tsv)及[训练曲线](training_curves.png)。
- [两卡 critic 统计与诊断帧账本](rank_and_frame_summary.json)。

六个固定窗口的[接触图](contact_sheets/)与两张[足底曲线](sole_curves/)逐图说明见执行记录第 10 节。灰色 v4 曲线来自保存于 2250 次的检查点，v5 使用完成后的第 3000 次检查点。

全仓库失败包含旧模型/PARC/SONIC 资产缺失和旧模块代码错误；详细原因逐项保留。双卡预检耗时 23.7516 秒，消耗 0.0131953 GPU-hours，未计入正式训练的 1500 次更新。

执行已结束，一次性队列和训练进程均退出。全部新增更新已保存，未追加预算、调参或恢复训练。
