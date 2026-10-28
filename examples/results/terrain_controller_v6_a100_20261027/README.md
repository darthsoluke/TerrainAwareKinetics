# v6 测试停止记录

2026-10-29 更新：[修复后重跑](continuation_f98e04d/README.md)的 90 项测试全部通过、FK 误差为 0；A 在 64 帧基线评估发生 16/64 帧维度错误，B 未启动。已发布三行部分结果、完整错误和独立 GPU-hours 账本。下面保留首次测试停止记录。

2026-10-28，A100 服务器，源码 `eafb71b5df2512d667fd4f6a6b4cc4db1a811b4d`。全部指定测试以 CPU 执行：**89 passed、1 failed、0 skipped**。

`test_foot_lock.py` 的 5 项全部运行。唯一失败位于两骨骼 IK 测试第 57 行的旋转正交性断言；此前目标到达与 FK 一致性断言已通过。按停止条件，A、v6a、v6b 均未启动，GPU-hours 为 0。

- [执行记录与完整命令](../../../docs/records/terrain_controller_v6_a100_20261027.md)。
- [tests.log](tests.log)：完整 pytest 输出与失败展开。
- [tests.xml](tests.xml)：JUnit，逐项测试状态。
- [report.json](report.json)：环境、源码与检查点身份、停止位置和计数。
- [gpu_hours.tsv](gpu_hours.tsv)：A、B 与各预检分别计账。

日志中的私人环境目录替换为 `${PFNN_PROJECT}`，JUnit 的主机名替换为 `A100`。原件保留在服务器 `runs/tmt_v6_coord/`，公开文件没有权重或动作数据。
