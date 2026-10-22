# Codec 修复小试方案

状态：**待用户批准新增训练预算及 B 分支的附加训练损失。未启动训练。**

现有三阶段已经完成3840万曝光；该模型的完整路线和静态平地几何门均失败。384窗口抽样中，codec在训练集上也有约2.2至2.6cm腿部重建误差。这轮小试只回答一个问题：保持编码器和码本不动，解码端是否还能改善腿脚重建，以及直接FK监督是否比继续原生损失有效。它不承担Root/Pose泛化修复，也不保证完整路线通过。

## 范围与预算

| 项目 | A：native | B：native_fk |
| --- | --- | --- |
| 初始权重 | 同一最终v16模型 | 与A完全相同 |
| 训练目标 | 原生codec损失及原系数 | 原生损失加0.02倍FK辅助项 |
| 可训练范围 | codec.decoder、decoder地形适配器、height_encoder | 与A相同 |
| 冻结范围 | Root、Pose、共享地形编码器、codec编码器、完整量化器和EMA码本 | 与A相同 |
| 实际Adam次数 | 250 | 250 |
| global/micro batch | 1024/512 | 1024/512 |
| 新增真实训练曝光 | 256000 | 256000 |
| batch256参考步 | 1000 | 1000 |
| 执行 | 一张空闲A100 | 另一张空闲A100，同时运行 |

**申请总额：500次新Adam、512000真实曝光、2000参考步。** 分开记账，不混入已经完成的50000步阶段，不宣称与旧优化轨迹等价。两分支使用相同训练批次作为对照，双方实际处理的曝光都计入新预算，不补样本。每分支外层限时1800秒，达到250更新即结束；失败保留并停止，不自动延长、补跑或恢复。

只复用既有PFNN/PARC/BONES训练cache和原50/25/25配额。原batch256抽样链从参考位置50000走到51000，以原64样本顺序组合成1024批次；验证和测试数据不得进入训练。每分支新建Adam，LR从5e-5按新增曝光余弦降至5e-7，clip0.5。它是新的独立小试，不是恢复或覆盖原Adam。

每次启动重新扫描全部GPU UUID、显存、利用率与实际计算PID。每卡memory fraction0.9，使用有实际计算用途的显存；不为占满80GB分配无用工作区。两分支是并行实验，不宣称是同一个optimizer的多卡数据并行。

## B 分支的明确变化

原生重建、commitment0.02、skate0.01和joint_velocity2.0全部保留。附加项为原生FK计算的根节点相对关节位置SmoothL1误差，作用于第4至27帧，覆盖左右Leg/Shin/Foot及原骨架foot_joint_idx中的脚趾关节。位置除以0.02m，再求SmoothL1均值，乘0.02加入原生loss。

这是训练期的附加监督，标签来自原训练源动作。它不向actor输入未来真值，不修改Root输出、不加IK/锁脚/碰撞投影/平滑，不调整地形。它也不是完整蒙皮穿透损失，不能把FK误差改善当作2cm穿透门通过。

## 比较与停止标准

只比较固定250更新的终点，不从中间检查点挑选最好模型。每50更新保存完整模型、Adam、RNG、样本摘要和独立曝光账本，同时检查既定每domain64个validation窗口。这些窗口沿用已记录的基线，不使用test路线作checkpoint选择。

报告每domain的原生loss、腿MPJPE均值/P95、RootXYZ RMS均值/P95，以及每窗口原1.5cm腿误差和2cmRoot RMS子门计数。完整skin尚未计算时，始终标记geometry_evaluated=false。

后续值得继续评估B的证据是：三domain的腿误差均值和P95都比初始值及A更低，同时Root RMS均值/P95均不劣于初始值和A（仅容许原数值误差5e-6m）。若只平均值变好、某domain变坏或Root明显退化，不把B提升为修复候选。A也必须相对初始值改善才有继续价值。这些是研究筛选条件，不能替代原发布质量门。

任一非有限loss/梯度、冻结tensor变更、EMA进入train、样本配额/身份不符、预算越界都会停止。新权重采用独立pilot schema，禁止自动export或替换28943，更不得替换28942认可版及28941/28936。

即使小试改善，最终仍须原18056顶点/8substeps/峰穿透不超过2cm/零unknown-uncertain/stance门、完整路线、flat起停转向与本机WASD30fps验证。任何进一步的Root/Pose训练或额外预算不包含在本次申请内。

## 已完成的准备

可审阅配置：`configs/pfnn_codec_repair_pilot_v1.json`，文件SHA256 `ea09ecc3e65c66d4d4024f2685be01f8e5ed4953fef8497cc539012829420cf0`；规范化计划SHA256 `1a2d51ded2fd710fc1a852168d473f637bb17089f8d76d2c8983b3ca76439ce7`。

执行器：`scripts/run_codec_repair_pilot.py`，SHA256 `353d0bb71f7f9c708407cb82fb73a67551e58b94d8b88f4182f335ba852d2578`。默认preflight模式不创建optimizer；execute模式必须绑定该分支已通过的同计划/同源码预检。范围与目标模块：`motionbricks/data/codec_repair_pilot.py`，SHA256 `0263bafe434252c00390f257eea056e1db94f66e99894c2b671324a75f3180d3`。

10项本地预算/计划漂移/真实NativeTrainingStage模式切换检查通过，Python语法检查通过。远端不可变快照`code_codec_repair_pilot_v1`已在实查空闲GPU0/2完成两分支零更新预检：各执行1024样本前后向及192验证样本，**0Adam、0训练曝光计账、全部模型tensor未变**，可训练参数均12496237，样本摘要均`79139fdf88b64ac908dd353b31352e62c74f2bc7d02ed217e9098667ec06b509`。三domain原生validation loss及逐窗口物理指标通过原数值容差与已有基线比较。

A/B初始梯度范数分别0.05616/0.08011，均有限；每卡peak allocated约19.44GB、reserved约35.34GB。这是预检显存，含验证的10.8/11.0秒不是生产训练速度，也未证明训练改善或optimizer冷恢复。小试不支持未经证明的自动恢复。

原始预检、完整trainable名单及合同已下载到`out/pfnn_mixed_20261015/codec_repair_preflight_v1/`，小包SHA256 `e6b08b365b94c9edbd1fccd8c9c7b11c513fad6e156716f9fa8a0188019c709f`。在收到本轮预算和B分支目标的明确批准前，后续跟进只维持现有服务与记录，不启动`--mode execute`，不重复本次预检。
