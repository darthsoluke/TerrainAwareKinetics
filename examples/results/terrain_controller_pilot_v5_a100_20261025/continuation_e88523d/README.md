# e88523d 修复后的 v5 执行

执行日期：2026-10-27。详细过程追加于 [v5 执行记录](../../../../docs/records/terrain_controller_pilot_v5_a100_20261025.md#6-2026-10-27-修复后续跑)。上一层保留此前的失败记录。

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

全仓库失败包含旧模型/PARC/SONIC 资产缺失和旧模块代码错误；详细原因逐项保留。双卡预检耗时 23.7516 秒，消耗 0.0131953 GPU-hours，未计入正式训练的 1500 次更新。

当前状态：原单卡 smoke、强制评估失败检查与正式 pilot 等待空闲 GPU。服务器上的一次性等待进程只在前一阶段通过后继续，失败即停止；正式 pilot 尚未启动。
