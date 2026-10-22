# PFNN 物理一致性评测：最新交接

2026-10-22 19:53。**A/B/C训练和原自动评估均已完成，旧等待任务已退出。**
C行已经补齐，不要再运行 `/tmp/_wait_c.sh` 或 `bash /tmp/_arm_strat.sh C`。
远端GPU3已释放；本次唯一SSH会话62753已关闭。GPU0/1/2继续保留给其他程序。

## 本次复核

- 读取真实远端A/B/C结果、checkpoint hash、实际评测/训练源码和监视器。
- 完成零训练、冻结权重的统一源接触mask评估，三臂同一1160个支撑区间。
- 原17帧评估最后3帧保持末指令；新评估仅使用14个有合法指令的生成帧。
- 本地6项回归测试通过：常量偏差不等于持续速度、接触区间端点、损失项遗漏、
  时钟检查、缺少闭环结果/预算不一致时拒绝汇总。
- 原报告、日志、源码与权重hash归档，新原始轨迹保留。两包通过私有Drive回本机并校验。
- 没有追加训练、修改checkpoint或替换操控页面。

最新数值和解释以 [复核报告](pfnn_physics_findings_20261022.md) 为准。

## 核心修正

统一源支撑区间的足速P50为A27.938/B31.240/C31.778cm/s，参考2.533cm/s。
原37.3/46.8/33.5来自模型自预测的不同接触集合，不能作为同条件速度排名。
C仍比A/B运动更慢。支撑位置拟合差这一发现保留，唯一根因和架构排序结论撤回。

48增强行来自12原始source/start窗口、3录制家族；不得把1484个足部时刻当独立样本。
A/B同seed未锁定公共层初始化；C容量/状态结构也不匹配。
原1.12%仅计body_fk，漏了boundary项；25倍loss不等于25倍梯度。
W1的3.30cm/s是24个标量抽样口径，不是24窗口的功效门。
力旋量链存在旋转约定、双脚支持遗漏、边界导数、求解状态和量纲问题，暂停作物理结论。

## 复现入口

在本机motionbricks根目录：

```powershell
python scripts/compare_arms.py --root out/pfnn_ablation_review_20261022/original_snapshot/runs/ablation_v3 --fixed-review out/pfnn_ablation_review_20261022/fixed_masks_review/fixed_masks_v1/report.json
```

原远端工作目录：
`${PRIVATE_STORAGE}/motionbricks-pfnn-frame-20261022/code_pfnn_frame_gpu3_v1`。
本次复核目录：
`${PRIVATE_STORAGE}/motionbricks-pfnn-frame-20261022/reviews/ablation_v3_closeout`。
本机证据：
`out/pfnn_ablation_review_20261022/`。

原C权重与所有完成作业保持原样；本机compare_arms现在读取报告，
不再硬编码参考/持久性基线，也不再用未校验力旋量作排名。

## 后续工作

先修正物理评测链并锁定一致的接触评分、初始化、监督目标和时钟，再决定训练实验。
本次没有自动扩大预算、启动新的架构对比或支撑加权训练。
维持单GPU3、单持久SSH、私有Drive传输及完整skin/长时控制/本地30Hz交付门。
