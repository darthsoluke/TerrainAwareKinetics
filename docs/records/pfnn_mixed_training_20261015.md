# PFNN + PARC + BONES 全量训练执行记录

## Current Work: 2026-10-22

Latest19:53: [A/B/C closeout reviewed](pfnn_physics_findings_20261022.md).
C and its auto-evaluation complete. New zero-update fixed-source-mask evaluation
completed onGPU3; A/B/C foot speeds27.938/31.240/31.778cm/s, source2.533.
Corrected loss/initialization/sample-size conclusions; old wrench ranks withheld
for evaluator defects.6tests pass, evidence privateDrive/hash-verified. No new
training or deployment. GPU3released,soleSSH62753closed; do not rerun old waiter.

Latest13:29: [P2 preparation](pfnn_frame_actor_p2_preparation_20261022.md).
Saved-trajectory CPU foot review completed,0newinference/0updates. Teacher foot
span preserved but stance sole speed3.66->8.75cm/s; closed42.36cm/s. Source masks
unchanged. SeparateP2 displacement/velocity/path/history objectives implemented;
8CPU tests pass. Draft10epoch/602900window/12h GPU3 plan; runner+GPU proofs pending.
No P2 preflight/smoke/training launched. Gait archive privateDrive/hash-verified;
soleSSH45034closed. Do not repeat completed reviews or P1 training.

Latest13:15: [P1 review completed](pfnn_frame_actor_p1_review_20261022.md).
P1 stopped at8h cap: recorded1774updates/453262windows; saved1652updates/422030
windows,7epochs. Unsaved122updates remain in consumed ledger; no repeat/resume.
192validation teacher/closed comparison and92train terrain windows complete.
Teacher body error0.756cm,26-frame closed body error10.555cm;3long trials all
fail control/skin. Idle16s drift4.462m; remote computeP99 about3ms. No promotion.
Future local checkpoint ordering repaired,65tests pass; old remote source preserved.
P2 target/loss/foot-gait review and bounded source-bound preflight remain next;
P2 NOT STARTED. All evaluation jobs exited,GPU3 released,soleSSH7046 closed.
PrivateDrive archives downloaded/hash-verified; heartbeat updated. GPU3 only.

Latest04:50: **PRO6000 GPU3 only; GPU0/1/2 reserved.**
[Execution and route measurements](pfnn_pro6000_execution_20261022.md).
PrivateDrive migration/hash checks complete;198 CPU checks, GPU3 zero-update
preflight and2-update/512-window Adam smoke passed. P1 launched once, timeout
PID2482846, worker2482964, runs/pfnn_frame_gpu3_v1/p1. Verified4 actual updates,
1024 windows/8192 next-frame targets; matching bindings/initialization, no failures.
Global256/micro256;10epoch/602900-window/8h caps, no automatic resume/retry.
A100 P1 never started. P0 totals4updates/1024windows across two hosts; P1 separate.
Original38.4M ledger and28944/Xbox model unchanged. M78 routes tested; no measured
4--5x end-to-end gain. SSH session69845 closed normally; detached P1 continues.
Local28944/28950 have no listener at04:49; no local model/service was changed.

Latest01:09: [Two-A100 execution](pfnn_frame_actor_execution_20261022.md).
P0 complete:138local checks, remote107+31 checks, full80file/64780row audit,
three source-bound zero-update preflights and2Adam smoke passed (512windows,
4096next-frame targets;192validation before/after). LocalRTX3080 six-case
full sensing/Root/Pose/FK P99=7.02--10.91ms; parityFK<=4.89e-7m.
Finalsnapshot code_pfnn_frame_actor_v3, global256/micro128. P1 NOT STARTED:
all8A100s occupied by others at launch and final scan. Exact launch in execution doc;
reuse passedproofs, no repeatedpreflight. P1 cap10epochs/602900windows/8hwall.
New heartbeat pfnn/30min; singleSSHclosed at handoff. Original38.4M unchanged;
no model/page promotion. Smallreports downloaded+SHAverified.

Latest22:40 PLAN ONLY: user requests allPFNN with our environment perception,
clarifies full3D. [Full plan](pfnn_all_data_3d_plan_20261021.md) and
[fresh literature review](pfnn_3d_literature_review_20261021.md) completed.
40original+40mirror files; all48232augmented train examples/12058paired windows,
existing family splits retained. Main proposal: full3D scene field + our1536point
body/path/velocity sensing + compact framewise Root/Pose controller with persistent
motion state; MB4frame remains comparison/prior.7primary papers reviewed plus
author materials.64aggregateGPUh is a proposed stage budget only.0newtraining,
no remote connection or live model/service change in this planning turn.

Latest13:39: [First prefix training result](reactive_prefix_training_20261021.md).
BothA100arms completed64Adam each;128newupdates/1024windows/7168prefixes.
Generated-history forward0.846->1.195m/s(target1.2), but idle16s drift0.270->1.179m,
stop0.933->2.200s and peak skin2.65->3.55cm regress. All9long trials completed,
0/9combined pass,4500frames,18056vertices/8substeps,0bodyunknown/uncertain.
No candidate promoted. Original28944/28950 remainPID11492/originalweights.
All remoteworkers exited/GPU0-1free/singleSSHclosed; no training queued.
Formal38.4Mledger unchanged; small reports verified and local. Nextdesign:
explicit current-intent plus long idle/braking objectives, not yet trained.

Latest13:03: User authorizes training-side changes and A100.
[Prefix training pilot](reactive_prefix_training_20261021.md): source-history
vs generated-history,64Adam each/batch8/7four-frame prefixes, same96train and
24validation windows, frozen codec encoder/codebook/shared scene encoder.
v2preflight passed0updates/exactforward/Root-Pose-codec gradients. GPU0/1
timeouts1816310/1816311 launched; verify realstate before reporting updates.
v1both arms failed baseline sensor-range FP32 boundary at0updates; preserved.
Original28944 weights unchanged; no new candidate automatically exported.

Latest11:54: User rejects reactivev3 as unstable/unable to walk.
[Gait repair](reactive_gait_repair_20261021.md): analog input bypassed4frame
cadence;517/763adopted plans lasted1frame. Retain cadence-onlyv4;
origin correction and8/24execution fail long replay and are removed.
Original28944restored;28950experimental/sharedPID11492, old25772stopped.
46Python/32JSchecks,0training. v4live2310frames/30.0106fps still fails idle
drift; user page28944Urban/speed3paused at seed. No reactive quality promotion.

Latest03:56: [Reactive kinematic results](reactive_kinematic_results_20261021.md).
Shared28944legacy/28950experiment/PID25772.24-frame forecasts, independentXbox
intent, exact-boundary4/2/1updates and measured immutable prefix. Three trials
display30fps, actualupdates~7.5/15/27.4Hz; computeP5025-28ms.2878histories exact;
7GPUparitycases/FK9.08e-7m/skin0;21504querypoints identical.45backend/37frontend
checks;34new3Dgeometry checks.0training. Idle drift0.031/0.050/0.131m/s remains;
no quality promotion.2-frame selected/paused on28950. v1/v2failures retained;
old11952/21264/27512stopped.

Latest priority: user requests full3D perception first, followed by PFNN-quality
framewise kinematic locomotion. [Revised plan](kinematic_3d_reactive_plan_20261021.md)
records the current heightfield-derived3D input limitation and PFNN's framewise
regression plus default IK/contact locking. Keep trained28944/Xbox viewer active.
This review adds a plan only; no training or runtime model changes.

Latest23:52 USER SWITCH: trained pure kinematic SOMA control restored on28944/
PID11952, map007/speed3, all15PFNN maps and inline tuning. Same final0f5d8e2b
checkpoint; direct Root/Pose/codec output, no G1 retarget/tracker/physics.
Native W800ms moves the model, observed30.0fps/no visible error.28948server27336
and planner18108 stopped.0training. Details: `kinematic_return_20261020_v1/service.json`
under the local PFNN output. Current task is direct model control; physics repair
reports below are retained historical work, not the active user page.

Latest23:44: [Registration and execution review](mb_scene_registration_repair_20261020.md).
Retained fixes: terrain chart registration, native Foot banking, batched rotation
conversion and proactive genuine-reference renewal.373tests pass. v49flat83s input
passes physical checks but later coverage fails; v50/v51 foot-fit candidates fail
posture/stability and are removed. v52Urban fails33.859s with torso contact at
30.016fps/0drops, no continuation exception.15maps restored;28948/27336/18108,
normal Xbox/WASD Ready.0training/filter; stable terrain control remains FAILED.

Latest23:07: [Scene registration repair](mb_scene_registration_repair_20261020.md).
Fixed metre-scale terrain input corruption and discarded native foot banking.
CPU chart-invariance joint error0.7983m->1.634e-6m, tokens identical.371tests pass.
v49 restores learned Root, preparing fixed-flat comparison on28948/PID13988.
v48 stopped; its49.983s fall remains.0training; physical outcome pending.

Latest21:26: [Session Reset repair](mb_session_reset_fix_20261020.md). Disconnect
no longer lets old-episode diagnostics claim a reset episode. Real Reset/reload/
Start succeeds. Separate v44 coverage failure6.96s retained; v45 starts steady
renewal120ms earlier while retaining100ms rejection bounds.215targeted tests
pass. Live35132/planner11348/28948; final continuous-input check in progress.

Latest19:47: [Intent/stopping correction](mb_intent_errata_20261020.md). CPU
probes locate incorrect braking in Root prediction. New bounded planar Root
reference before Pose improves stop speed0.624->0.054m/s on the same flat trace;
reverse/rapid-strafe progress changes to the intended sign. Zero training/filter.
v42 flat123205substeps and v43 PFNN000136968substeps pass physical audits; full
83s analog traces,~30fps/0drops. PFNN peak penetration4.372mm.381Python tests pass.
v43/28948 server29536/planner19844,15maps, normal Xbox/WASD Ready, DevGPU scope.
Side/back speed remains weak; visualP95178ms. No full quality acceptance.

Latest: [Implemented online repair](mb_online_repair_20261020.md). Fixed native
world/time handoff, decoded-past planar integration error, and online packaging
that still selected the singular marker-plane limb mapping. v39 completes83s
simulated Xbox plus idle:187.46s/187360controlled substeps, physical audit PASS,
29.991fps/0drops. Reverse strafe and stopping still fail direction metrics.
376Python checks pass. Root-rate QP candidate v40 failed the input deadline and
was removed; source/evidence retained. v41 restores15PFNN maps/28948 with the
audited v39 runtime.0training/filters. No stable-Xbox or all-terrain acceptance.
PFNN000 v41 stops at28.279s on unknown ground after exiting the valid domain,
29.993fps/0drops. Failure retained; normal Xbox/WASD page reset to Ready.
Current server33428/planner11348; explicit DevGPU scope. See linked report.

Latest: [Code/literature review](mb_online_control_review_20261020.md) complete.
47saved plans show native0.582m/s becoming10.033m/s at a source join, then entering
history. New fragment and old source use inconsistent registration; generation
is not conditioned at its committed handoff. Root dynamics lack hard bounds.
2CPU ONNX counterfactuals confirm reference XYZ/velocity does not affect this
actor directly.8primary papers reviewed;0new training/physics/GPU/runtime edits.
Fresh process scan: no Python or28947/28948/28949listener; old22856 no longer live.

Latest16:09: [Simulated Xbox results](mb_offline_online_gap_20261020.md).
100Hz continuous raw axes reproduce falls:PFNN00035.326s, flat64 baseline4.885s,
preview-protected v36=6.935s; all~30fps/0drops. Already-observed target overwrites
are fixed and46actual adoptions now have0Root/joint changes. Remaining reference
Root speed52.072m/s/angular143.928rad/s; source re-registration/joins amplify
native16.055m/s to319.561m/s. Stable gamepad remains FAILED.0training/filters.
Failed GPU services stopped;28949/PID22856 comparison preserved.323Python/30JS
checks pass. Old142s discrete-input pass below is not continuous-gamepad approval.

Latest: [Continuous-reference implementation](mb_offline_online_gap_20261020.md).
28949 reproduces the original three-panel physics bitwise at30.016fps. v35 on
28948/PID30992, GPU planner204, completes142.811s ofPFNN000controlled physics:
0falls/joint/motor violations, scene contact penetration1.569cm. Browser29.9933fps,
0drops/errors,21independent/combined commands. Episode2 is Ready for Xbox.
Explicit DevGPU scope is enabled; normal-power v34 deadline failure is retained.
ChainP99=83.951ms/first visualP95=222.5ms/body onsetP95=919.6ms, so response targets
remain open. All15maps available; no all-terrain approval.0training/0filters.
Old PIDs below are historical; the previous v32 and SONIC services are absent.

Latest priority: [Online History Contract](mb_offline_online_gap_20261020.md).
New zero-execution analysis finds mixed native-prior histories and a command-
dependent observer: fixed observations change leg history by median22.243deg
when switching stop/move rules. Preserve immutable causal history and one
observer before adding heavier contact optimization. Product configuration,
training totals and28948 remain unchanged; reports v32_history_contract_review_v1
and v32_observer_mode_counterfactual_v1 are complete.

Latest: [Retarget Repair Comparison](mb_g1_retarget_repair_20261020.md).
Same saved MB source and exact initial G1 state; three frozen-Proto16s rollouts.
Native material axes remove34.558deg near-extension amplification; contact/support
reference has0penetration. Actual jitter energy improves8.60%, penetration
4.672->4.065mm, planar drift3.840->3.625m; lateral excursion worsens. SolverP99
289.979ms and a6cm reference-dip stress failure block deployment.70tests pass,
1strict known failure.0training/filter,28948 unchanged; no new30fps/Xbox approval.

Latest: [Continuous MB Reference Isolation](mb_continuous_reference_isolation_20261020.md)
completed16.8s MB -> fixed G1 ->16s Proto with0training/0online switching.
Root source mean1.150m/s; stage2 anatomical twist amplification1.280->33.057deg
and G1 reference ground penetration2.328cm found. Physical audit passes15900steps,
but Root drift3.840m and extra knee motion remain. Fix retargeting first; live
v32 baseline unchanged, no repaired-WASD release.

Latest: severe gait oscillation rejects v31 Root residual; removed online.
v32_reference_only109016/GPU150500/28948 is a research baseline, not fixed-gait
delivery. User rejects pure post-processing; no EMA, output clamp or new training.
New analysis and contact-aware planning/control direction:
[Robust Gait Repair](mb_g1_robust_gait_repair_20261020.md). Older entries are history.

2026-10-20: authorized zero-training reference Root/joint continuity and execution
COM/yaw task implemented. v30 completed300s baseline with full physical audit PASS.
v31_root_task parent183916/GPU45100/28948 completed299.66s at29.994Hz, no runtime
failure, but post-turn side movement still fails direction control. No stable
handoff, new training or formal exposure. Baseline55588/28947 preserved.
Latest implementation/results: [Current Repair](mb_tracker_comparison_20261019.md).
Older process IDs below are historical.

21:25 Xbox/XInput is detected by the active28948browser, left translation/right
yaw. Continuous deadzone/slot enumeration fixed,26frontend+42CPU checks pass.
New zero-training jitter forensics verify two recorded episodes and12fixed-sensor
ONNX counterfactuals: severe native/reference switches drive large target changes.
See [Jitter Review](mb_tracker_comparison_20261019.md). User is actively playing;
do not reset/restart for research. No physical-stability fix deployed or training.

21:05 latest service: v19_feedback/28948 parent136736/GPU68072;28947 baseline intact.
User controls: left movement, right facing; strafe/reverse and independent yaw remain.
UI layout/focus fixed,25frontend+37protocol tests pass, desktop/mobile checked.
Measured feedback is active, but strafe/braking remain poor; v19 left the known
map at20.921s/29.9933Hz. Receipt retained and manually reset to Ready. No new
training or stable-WASD approval. Current details: STATUS：`../out/mb_sonic_e0_e3_20261018/task_artifacts/STATUS.md`.

### Historical v13 Closeout

16:50 final zero-training tracker comparison: integration complete, stable WASD
FAILED. v13 experimental28948/parent135292/GPUchild101152 retained;28947baseline
unchanged. Flat000 fell37.200s and rough007 fell38.859s at30fps/~0.99RTF despite
zero actual joint crossings/motor overlimits. Full audits and loop-drift evidence:
[tracker review](mb_tracker_comparison_20261019.md), STATUS：`../out/mb_sonic_e0_e3_20261018/task_artifacts/STATUS.md`.
No stable delivery or terrain-quality approval; new training/exposures0.

### Historical v11 Preparation

16:25 ProtoMotions zero-training comparison is active; latest state and evidence:
STATUS：`../out/mb_sonic_e0_e3_20261018/task_artifacts/STATUS.md`. v11/28948 is
preparing; no new handoff yet. Same-reference SONIC/Proto matching-G1 replays both
complete22s without fall or actual joint crossing. v9 real flat WASD/QE reached
210.819s at30Hz/1.00RTF with a full physical audit; rough007 reached105.180s at
30Hz/0.83RTF, then recording budget exhaustion. v11 optimizes exact sensor reads
and lossless recording, not physics/model math. Training updates/exposures added0.

### Historical Reference-IK Validation

当前 MB + SONIC + MuJoCo G1 集成状态见 STATUS：`../out/mb_sonic_e0_e3_20261018/task_artifacts/STATUS.md`（Asia/Singapore）。下方28944端口、PID及“当前/在线”描述均为2026-10-17的历史记录，最新状态以该入口为准。

12:31零训练约束求解验证结束：离线越限修复与346项CPU检查通过，完整66帧转换约20ms；在线v14/v15/v16仍触发200ms输入超时，未通过交互门。最终画面约29.8Hz，实验服务全部停止，28947基线保留；0训练更新，未交付新WASD版本。报告与失败记录见STATUS。

最终5688实际状态复核还发现跌倒及实际踝关节最大越限0.189897rad；294个参考均合法，在线QP修正次数为0。后续必须同时定位生成/适配/SONIC跟踪和延迟长尾，不能只解决参考合法性。汇总：`out/mb_sonic_e0_e3_20261018/constrained_reference_final_validation_v1.json`。

此前用户episode0003在54.60s因未来G1腰部参考越限停止：原映射最大0.634952rad、G1上限0.52rad，实际机器人当时0.019736rad。胸腔固定标记候选修复了锁骨耦合，但失败包仍有8帧越限（候选最大0.638628rad），未部署。用户随后批准参考层约束求解，本轮结果如上；原故障现场保留。

2026-10-19：28948已更新v13，修复连续输入使有效计划反复作废而断供的问题；按真实执行推进输入账本，保留100ms状态/计划、200ms输入门及真实参考覆盖。平地WASD/QE/连续摇杆实测约29.8fps，0浏览器故障，313项CPU检查通过；0训练更新。当前PID与记录见STATUS。

## Historical Work: Dual-Horizon Gait Fix (2026-10-17)

父真实tab8最终交付已核：`dual_parent_delivery_v1.json/.png`，ONLINE/new schema、execute8/32、preview8、4Pose/4刷新，短暂idle30fps/0underrun。父恢复原暂停/出生点并保留实际3m/s、Shift1.6、govON、8前瞻；URLspeed6为旧值，不当实际设置。两文件已纳最终manifest，本轮结束。

2026-10-17：已定位长档起步/停跑是Root近期慢、远端快的闭环规划退迟，不是渲染停帧。007/实际3m/s/governorOFF原8/12/16均速2.415/.230/.085m/s；正式训练固定8tokens/32帧的事实已核。经父review批准，部署独立`motionbricks.dual_horizon_execution.v1`：前瞻8/12/16，近期Root/Pose固定8，32帧真实生成/24可用帧，4帧重规划/4Posepasses/8帧传输不变。两Root共用原8coarse，4近+4真实长Root站点进入Actor地形memory；不拼输出、不移动末端约束、不重训练。

新12/16前瞻冷起步前2秒2.348/2.358m、持续1m/s均.867s，均速2.350/2.289；热切均速2.948/3.022，无预声明停步/leg-still事件。8档64帧bitwise，candidate原生/图token同、FK<=1.276e-6、skin0、history<=4.77e-7；far消融实际改变memory/logits。长时域采样点有时在短Root终点后方，原坐标保留。未插桩整包12/16 P95239.46/243.84ms；额外Root每包6.59/7.53ms，coarse仍每包2次。旧带reference-check计时已单独标明，不当生产延迟。

当前28944/PID115796，`--initial-map 007 --planning-policy dual --dual-proof .../dual_horizon_latency_v1/proof.json`；28943/PID60524保留。旧全长模式显式`--planning-policy full`，不自动fallback。UI明确前瞻与近期执行；修复同viewer stale draft及切档整份apply把速度6改回3的问题，定向6→12→16仍6/Shift1.6/governorON。实际32执行、24Root/40或56preview回读；007↔005旧session拒绝通过。子页暂停待父刷新真实用户页。报告`pfnn_gait_horizon_review_20261017.md`，manifest`dual_horizon_delivery_v1.json`。0optimizer，原地形质量门未改。

## Previous Work: Shared-Weight 8/12/16 Acceleration

父tab8最终验收已写`horizon_parent_delivery_v1.json/.png`：005/12/40Root帧/4刷新/CUDA真实回读，frame731为30fps，稍后1184帧出现Confirming played frames及28.6/28.7fps，0空buffer。用户页保持运行/画布聚焦，子review页暂停。结论为接近30fps的可操控加速交付，严格稳定30fps仍有慢包/ACK等待余项，不隐藏。

2026-10-17：28944现PID166916，`--root-path-observation --initial-map 005`；原28943/PID60524保留。通用Root→原生沿预测Root采样→四轮Pose/decoder支持8/12/16预热图，共享权重常量，不重载三份模型；有效未来24/40/56帧（.80/1.33/1.87s），4帧重规划/8帧传输/30Hz不变。query为动态观察输入，decoder条件帧数32/48/64同步。固定Root有效prefix17/21/25在构图时检查mask；未改采样数学或数值门。三档rolling token全同，FK最大3.46e-6，own-history<4.62e-7，horizon+governor串行/预取/ACK bitwise，005↔007复用通过。

005移动整包P50/P95：8档193.00/220.96ms、12档200.77/209.55ms、16档206.99/212.33ms（各12样本、排除3warmup）。六stage唯一CUDA常量760.55MB，其中758.60MB直接共享原模型；含一次换图allocator峰allocated1.099GB/reserved1.418GB。浏览器governorOFF实际8/12/16为29.74/30.000/29.999fps，0空buffer，ACK等待6/0/0；ON旧组29.66/29.63/29.83与4/3/1等待保留。慢包已prefix0，compute峰280–317ms/arrival296–331ms，属通用耗时抖动，不能称严格无等待30。未降低刷新/精度或增训练。

同窗8/12/16分段选择与实际Root帧数/CUDA状态已上线，候选副本复制mutable配置，ACK后同步提交num_tokens/runtime/inference/governor。最终005/3/12/4；子review页暂停，父线程已刷新真实用户tab8做最后审查。详见`docs/pfnn_horizon_acceleration_review_20261017.md`、`horizon_delivery_v1.json`。0optimizer，原质量结论不变。

## Previous Work: Optional Anticipatory Speed Control

2026-10-17：前瞻地形目标速度调节已交付为同窗可切换实验，**默认关闭**。28944现PID86712，007/3m/s/4帧，原28943/PID60524保留。仅限制发给Root的水平目标速度；30Hz、8tokens/4Posepasses、8帧包、原权重/几何检查不变，0optimizer。Terrain页有开关、强度、最低目标速度、恢复加速度及请求/目标/实际三速度读数。15图保留。

两条预声明过渡、六组v2均到达相同纵向终点，完整skin/196足底补证完成：正向off/balanced/cautious足底峰25.77/26.58/29.36cm，耗时2.285/4.403/5.100s；斜向29.55/28.96/13.29cm，耗时2.485/4.617/5.895s。正向更慢且更深，因此未默认启用，原2cm质量门仍未通过。初版sole补证因provider忽略region选择而无效，原文件及invalidated.json保留；v2显式196顶点选择/分母断言通过，复用保存world，未重跑生成/全身。

开关引入的预取边界等待已修复：仅预取副本应用新governor配置，候选在前包ACK后提交配置/起始状态；候选自身ACK才推进8帧状态。不改已发布包、不扩大8-key模型参数规则。4项CPU边界检查通过；真实五包OFF/ON序列与serial完整载荷bitwise、源帧0/8/16/24/32；浏览器连续四次开关328帧/10.9289s=30.012fps，0新增underrun/0deferred。修复前1次等待证据保留。页面最终暂停、governor关闭。详见`docs/pfnn_anticipatory_speed_review_20261017.md`、`anticipatory_speed_delivery_v2.json`及toggle_browser_v2；无新增训练/路线搜索。

## Previous Work: Root-Path Inference Latency Delivery

2026-10-17：已完成本轮推理工程诊断与可操控交付，详见`docs/pfnn_inference_latency_review_20261017.md`。当前28944为**PID186648**，`--root-path-observation --initial-map 007`，原28943/PID60524未动。Root单次图计算→原生沿Root采样→Pose四轮/decoder第二段图，FK/skin/history仍编译执行；同次未变的身体/速度查询结果复用，全部1536输入点保留。实际Pose细观察诊断已校正。0optimizer，权重/原质量门未改，15地图与同窗调参保留；未跑全量路线或测试套件。

同007/8tokens/4Pose/4帧档，20个整包样本（另排除3warmup）：native P50/P95=281.20/294.73ms，split=137.97/144.54ms，含publication、FK/skin及JSON，不含网络。8帧档141.19/154.00→69.43/75.57ms。固定第64帧、完全相同起点与实际参数的浏览器24按下/松开事件对照：prefix0/2/4的input-event-to-render P95约495/494/400ms，最后一组24/24显示、0underrun；GL提交后CPU时刻，不是物理photon测量。生产按最近真实request-to-binary耗时+18ms余量，每包开始时固定选择0/2/4，不全图写死4。最终v6独立WASD窗口实测29.94fps/0underrun，观察到0/4/2三种选择且请求均发生在锁定边界。

接续证据`root_path_latency_v5/complete.json`：首包及预取共4次活跃token一致，FK最大2.83e-6m<3e-5，actual-frame history误差<4.5e-7，native与split各自serial/prefetch bitwise、ACK身份通过。v3跨候选history绝对比较失败保留；最大2.145767e-5来自local_vel/body feature336，区别于同一motion的history门。相机改动仅修正浏览器意图/计时噪声，实际24/8/4重规划率未变，不作为提速来源。真实pointer拖动后W方向更新已检查，自动化临时按钮映射已移除。v5诊断进程首次请求停滞记录保留，正式v6不再携带基准并释放启动临时缓存；最终证据`root_path_adaptive_live_v6.json/.png`及`root_path_latency_delivery_v6.json`。当前隔离验收页暂停供review，用户原页需刷新加载新JS。

## Previous Work: Predicted Root Path Observation Preview

2026-10-17：用户授权直接修改地形观察并交给人工操控，先不动测试。独立试玩端口`http://127.0.0.1:28944/?map=007&speed=3`，监督进程**36260**，入口`--root-path-observation --initial-map 007`。原28943/PID60524保持不变。新链路先用原粗观察计算一次Root，再沿模型预测的Root曲线补采三维SDF及二维高度网格，更新Pose/decoder地形条件；保留身体周围、实际运动方向的立体观察。模型、码本、数据、训练预算未改，0optimizer。神经网络原生执行，骨骼/蒙皮/事实history沿用编译路径，未把整个模型重复跑两次。

未修改或运行测试套件，未运行路线评估或全15图推理扫描。只进行了语法/导入、服务启动、骨骼/蒙皮启动检查及页面启用。初始007/原24帧档观察到269帧、30fps、0underrun；随后用户已自行WASD操控并切到4帧档，截图在902帧约19.9fps、1underrun，包计算约371.7ms，不改动用户选择。新策略功能已接入；高频档仍需执行性能优化。证据`root_path_live_v2-readiness.json`、`local_root_path_live_v2.stdout.log`/stderr及`root_path_live_v2-browser.png`。最初重复context和预取执行类型接线失败已修复，旧进程111824/194280停止；不恢复旧进程。

## Previous Work: Inline Tuning And Terrain Response Plan

2026-10-17最新人工要求：小调参页必须与demo同窗口，同时调查兼顾地形质量与快速响应的各种路线并制定计划。已完成右侧338px可收起调参面板（响应/地形/生成/镜头/约定分页）；窄屏上下分区，动态画布/相机viewport按真实可用尺寸调整。参数绑定本页会话，保留个人预设/导入导出和实际值对照。浏览器已真实应用4帧档并持续约30fps/0underrun，测试后恢复原24帧档、007地图/2m/s，当前IAB tab6，页面`http://127.0.0.1:28943/?map=007&speed=2`。证据`pfnn_inline_tuning_browser_v1.json`及desktop_final/mobile_v1截图。页面服务在中断后已退出，核实端口/PID后恢复为**60524**，只启动当前map server；新readiness SHA807029490b678c66fdb2315933d016c034a1bef463db31607a982cb9f7bf51dd。旧PID47180/195804不再有效，不重复启动健康60524。

完整新诊断、PFNN原理和12路线/6阶段计划见`docs/pfnn_response_and_tuning_20261017.md`。关键修正：MotionBricks原论文附录C本来就建议3-9帧重规划、指令变化触发，并报告过度提频会影响连贯性；不能把当前24帧缓存和本机56-65ms循环当作架构必然上限。优先工程调度/观察profile，再做步态与地形接触的闭环训练提案，必要时再考虑逐帧小模型/运动匹配/物理跟踪。新的384000曝光三臂研究预算只是计划，**未批准/未启动**；旧512000曝光codec小试也仍未批准。全部本轮0optimizer，原质量失败仍保留，不宣称穿插根治。失败查询graph v1/v2未部署，核心ReusableTerrainExecution source恢复为f7f98e10f7e7db80333b8eefb07b5a8e42ba170c97b14e3e653bfdc8523b97d6。

## Previous Work: Fast PFNN Map Switching

最终复核时用户已自行切到002（`?map=002&speed=4`），菜单显示本次0.41秒，1327帧/30.0fps/0underrun；保留用户当前地图，未再次切回012。最终截图`pfnn_fast_switch_desktop_final.jpg`。

2026-10-17：已完成最新人工请求“切换地图速度太慢了，解决这个问题”。28943现为`195804`，入口仍`scripts.soma_pfnn_maps_server`，新增`--readiness-report out/pfnn_mixed_20261015/pfnn_fast_switch_readiness_v1.json`。旧133468已核完整cmdline后仅停止一次；其他服务未动，不恢复旧PID。15张PFNN原图全部保留。用户页面恢复到`?map=012&speed=4`并继续播放。

切图不再重新trace/capture模型：`ReusableTerrainExecution`在图外执行同一原生高度/SDF查询，将实际观察tensor显式输入共用模型图；骨骼SDF亦使用当前场景。启动时预加载并验证全部地图，后续只重绑已验证的scene/field/factual seed和报告。前端原地更新地图，复用角色/渲染器，并缓存访问过的地图数据；不再整页导航或重复传输角色skin。保留scene_revision旧会话拒绝与原未知地形门。训练/权重/质量结论不变，0optimizer。

实测旧切图约32-34秒；新浏览器首次访问000/006/011/urban分别0.333/0.558/1.348/1.284秒，重复006/011/012分别0.128/0.212/0.130秒，包含HTTP、地图安装与WebSocket ready。后端已缓存切换低于原Windows monotonic计时粒度（报告0或约15ms，不代表物理0耗时）。正式新进程启动验证15图各5命令：预计算观察与原生逐值相同、活跃token一致、FK最大2.2492363314e-6m<3e-5、官方skin通过；15图缓存往返后全部输出逐值相同且CUDA graph对象未更换。`pfnn_fast_switch_v1/complete.json` SHA d3f3058b77317cc3320e26bd1fd4413bc3217b27b64ec93e88e062fdee989867；实际服务readiness SHA1f73af87c38fab36286df83b4662e1b2d9ea2df33c587f1d12fcfb58bf4fa7b7。核心source SHA f7f98e10f7e7db80333b8eefb07b5a8e42ba170c97b14e3e653bfdc8523b97d6。

74项相关测试通过（两次套件有5项重叠，已去重），JS语法通过。浏览器012连续超过2100帧仍30fps/0underrun，短W输入有效；桌面1348x956及窄屏390x844全页截图/画面像素非空检查通过，viewport已恢复。证据`pfnn_fast_switch_browser_v1.json`、`pfnn_fast_switch_desktop_v1.jpg`、`pfnn_fast_switch_mobile_v1.jpg`。首个窄屏viewport截图会被IAB缩放，实际全页截图已单独核验，不以缩略图作为布局证明。已完成工程验证不重复；512000曝光codec修复小试仍未批准/未执行。原`pfnn-wasd` automation.toml已不存在，不擅自重建用户可能移除的跟进。

## Historical Map Catalog Preparation

2026-10-17 01:28：最新人工请求先是“放到pfnn的地图”，随后明确“所有的地图都要有，全都加载进来”。这是当前工作，**不等于批准codec修复训练预算**。用户反馈当前操控质量挺好，保留这条主观反馈；原严格几何验收结果不改。原版城市图已在28943/PID113008运行，仍同一最终权重0f5d8e2b05b8ec518a104a4979cca70f23efa64080abfc843f874634de5a9646。用户正在地图中操作，不随意重置其会话。

已从作者原始ZIP中央目录核对全部高度图，原始资源在F:/dev/GASP_57/ai4animationpy/Demos/Dataset/PFNN，demo与data同名图去重后共15张（000-007、009-014、urban）。完整地图均已打包到`out/pfnn_mixed_20261015/pfnn_map_catalog_v1/`；catalog SHAe117d11dd10547b4ec5c2b26be32f5758eb57a6575405aa57efda12fad222eca。保持作者hscale3.937007874cm/vscale3cm、data[x][z]转SOMA[z][x]；只做一次固定地形Y基准平移及4帧事实seed的XZ刚性摆放，不改高度差、不运行IK/碰撞推挤/输出修正。演示图用作者reset位置；额外地形图预先选固定低起伏入口。模型查询保留完整原图，显示使用作者默认stride2网格并保留边界。

新增`bvh_heightfield.py`用Warp1.18.0的solid/air双BVH及逐步邻域搜索，保持原下向实体/上向空气prism距离、最早索引tie及unknown边界门，平地仍原AnalyticPlaneField。15张图每图62个查询与原算法距离/法线/valid逐值相同；`geometry_complete_v3.json`及各图geometry_query_review保留。核心SHA ccdb3c55616e76999e4a8821d4fe83c0e14888bd24c413ea848746483000ca00。011的1536点探针约1.68s→117ms，urban约44ms→11ms，仅查询计时，不当整链fps。最初两版结果与失败测试均保留，不用它们覆盖当前proof。无需重新训练、retarget或cache。

`soma_pfnn_maps_server.py`和页面地图下拉菜单已实现；切图复用同一模型、重新绑定场景并过该场景原生/compiled token/FK/skin门，旧会话有scene_revision防串图。`pfnn_all_map_runtime_v1`全模型重载验证在6图后因被共享模型方案替代而停止，superseded.json保留、0optimizer，不重启。当前**本机零更新逐图验证PID140836，exec会话95495**，输出`pfnn_all_map_runtime_v2`，先读complete/failure/progress/log，不重复启动。原城市图服务仍113008；待15图验证后再一次切换为地图目录服务并测试菜单。64现有服务/验收测试、BVH/CUDA图及失效会话等新检查通过；完整新界面测试尚在进行。旧28942/28941/28936保留。后续跟进优先完成地图交付，不继续旧的“只等待训练授权”循环，也不能擅自启动修复训练。

## Training And Pending Repair Proposal

2026-10-16 22:20：**修复小试代码/计划/零更新GPU预检已准备好，正在请求一次明确的新预算及B分支目标授权；尚未启动训练。** 先读`docs/pfnn_codec_repair_pilot_proposal_20261016.md`及`configs/pfnn_codec_repair_pilot_v1.json`。A保持原生codec loss，B增加0.02倍根节点相对腿脚FK SmoothL1（坐标按.02m缩放，源训练标签，仅4:28）；两支都只训练codec.decoder/adaptation.decoder/adaptation.height_encoder（12496237参数），冻结Root/Pose/shared encoder/codec encoder/完整quantizer及EMA。申请每支250Adam、global1024/micro512、256000曝光，两卡独立并行，总500Adam/512000曝光/2000参考步；原batch256链50000→51000、50/25/25、fresh Adam5e-5余弦至5e-7/clip.5/.9内存，独立pilot schema、无自动export/恢复/追加预算。只比较固定250终点，validation不进训练，不使用test挑模型，原完整质量门不变。

`code_codec_repair_pilot_v1`上两支预检`codec_repair_preflight_v1`均完成，timeout171632/171699已结束；新全GPU扫描后分配0/2。每支1024样本前后向+192验证样本，**0Adam/0正式训练曝光、所有tensor未变**，固定validation原native loss及逐窗口leg/Root RMS与既有基线在原数值容差内一致，scope/EMA冻结正确。两支sample_digest同79139fdf88b64ac908dd353b31352e62c74f2bc7d02ed217e9098667ec06b509，初始梯度norm .05616/.08011，peak allocated19.44GB/reserved35.34GB；含验证和hash的10.8/11.0秒不是训练吞吐或质量改善证据。10本地预算/漂移/真实NativeTrainingStage模式测试及语法通过。runner SHA353d0bb71f7f9c708407cb82fb73a67551e58b94d8b88f4182f335ba852d2578，helper0263bafe434252c00390f257eea056e1db94f66e99894c2b671324a75f3180d3，canonical plan1a2d51ded2fd710fc1a852168d473f637bb17089f8d76d2c8983b3ca76439ce7。预检小包SHAe6b08b365b94c9edbd1fccd8c9c7b11c513fad6e156716f9fa8a0188019c709f已下载。**批准前不得启动--mode execute；健康服务不重开、上述预检不重复，后续heartbeat也不是新预算批准，不反复询问。**

2026-10-16 21:35补充：**`final_codec_objective_v1`已在新扫描空闲0/2/3完成，timeout监督4171621/4171639/4171648均结束，0optimizer，不重跑。** 本轮复用上轮384个固定sample_id来检查新的loss分解/输出梯度问题，不重新抽样或重跑路线/skin。原native codec loss复现差0，腿部物理指标复现差<=1.37e-9m，所有模型tensor hash不变、参数无grad、输出未修改。真实系数reconstruction1/commitment.02/skate.01/joint_velocity2；原生loss源码SHAc5ee20e3bb56680c7fffa09dee6b03210bd986828f7609007141a089a76ea644和stage37c672b82473bb82ef3bfe50b3554f56795202f663207b9c488daa60a1fd7a9a与正式v16逐文件相同，没有训练目标漂移。六组的归一化recon输出梯度平方范数约96.6%至97.6%落在Root特征，与腿MPJPE梯度cos约.0110至.0155、与脚相对Y误差cos约.0045至.0092。此度量依赖输出坐标/归一化，仅是局部输出空间方向，**不是参数/Adam方向，不证明改loss能改善动作或大batch是原因**；commitment对脱离encoder的recon leaf为常数，不代表实际encoder梯度为0。原codec objective没有18056顶点/8substep静态terrain穿透项；维持原门与原训练目标，不据诊断擅改。独立脚本SHA20fd7cd38febf204980f94be17fb4182128ce1c8831b9b7e5c0df6999d06d69b；小包`final_codec_objective_v1-small.tar.gz` SHAdcdfbd7377edbd6b2343b6bb8335b5e1d763fccb03e2d4c554414eac9d9003b6已下载展开，summary另存。28943仍PID167808，原28942/28941/28936未动；训练与质量结论不变，未新增训练预算。

2026-10-16 21:17补充：**`final_domain_diagnosis_v1`三卡零更新检查全部完成并释放0/2/3，不重复。** 独立`code_final_domain_diagnosis_v1`由final acceptance快照构建，脚本SHA75d172bf16e9fc25a4a91c36595f680c0ee158ef6c3b7a7cb8b998890c3b165f，CPU导入/语法通过后每child复查全部GPU空闲并记录UUID/实际PID。PFNN/PARC/BONES各64train+64validation，共384窗口，quarter0、source SHA/start去重、验证排除原定期128样本对应source窗口；仍只是相关窗口抽样，不冒称独立录制或全量审计。冻结最终模型，原cached观察/原native targets，0optimizer，模型全tensor hash未变，不改actor。训练/验证的codec腿部MPJPE均值分别：PFNN **2.42/2.75cm**，PARC **2.64/2.77cm**，BONES **2.23/2.12cm**；Pose native loss分别**.449/1.819、.985/1.457、.627/.703**。PFNN Root native loss .00314/.04852，但RootY均误差约2.24/1.99cm，不能把整个Root loss差直接当高度误差。结果支持codec训练样本重建残差与PFNN/PARC预测泛化差距共同存在，未证明大batch因果或“加步数就能通过”。本诊断未重算skin，不能替代原失败几何门。报告小包`final_domain_diagnosis_v1-small.tar.gz` SHA29e9e736f08119cc0ce481c8da18f99233db63d4f8b610889600a8dbad937fde已下载展开。具体审阅与下一步边界见`docs/pfnn_final_candidate_review_20261016.md`。

2026-10-16 21:06 Asia/Singapore：**三阶段正式预算全部完成，当前模型不满足最终交付质量门。禁止重启任何正式训练或把浏览器30fps当质量通过。** Root仍`full_training_v8/root`：50000实际Adam/12800000曝光；codec仍`full_training_v12/codec`：20125实际Adam/50000参考步/12800000曝光；最终Pose在`full_training_v16/pose`：4000实际Adam/50000参考步/12800000曝光。合计**74125实际Adam、150000参考步、38400000曝光**。Root/codec SHA分别d3cdac4d0c260361f275a5ee8f940298486ecf201f00e2051494e93d7dce63e8、7f8ce3b2498168f9bfa2490bbec52a5f30bce5d0f8b271d8e7f76120aabd61fa；**最终Pose weights SHA b960fbe9f792cf46333c23762a8f047259132acd5f6b595bbc726dffa8b40c19**。保护hash均通过，v16监督3567132及child3570020均已退出。新旧batch优化轨迹不宣称相同。后续仅在现有授权内做零更新诊断/修复/验收；新增正式训练预算或目标改变必须单独提出有证据的具体方案。

v14真实loader下的global capture失败已完成隔离：global+pinning失败，global无pinning和thread-local+pinning均与native输出/梯度/RNG/样本逐值一致。随后`pose_actual_loader_optimizer_workflow_v1`完成真正`train_unified_full_corpus.main`的native4/graph4/fresh-process resume2对照，含4worker pinned loader、50GiB preload、定期验证和保存，全bitwise，0正式预算。诊断harness仅覆盖source/execution准入身份，不修改optimizer/math/data，报告明确声明。74项本地接线/迁移门检查通过，旧套件有重叠不相加。v16使用thread-local冻结label图，只用于训练冻结codec编码，不进actor；从v15完整**actual3750/reference44000/11264000曝光**checkpoint接续，handoff SHA32f55b914deebcbc3d9b6bd9bbaf24d629632364db6be3bea9717169790c5994。v16首个检查点审查已完成（actual3792/reference45008/11522048、228Adam、scheduler45008、2CUDA RNG和1008批原sample链），不重复。最终sample_digest445ae56f4d98e961151bb574e90192a90f56054a650a6dd4f69c54f17280defc。v14零新增更新失败、v15恢复及所有旧barrier都保留，禁止恢复v1-v15或重跑Root/codec。

**正式模型完整评估失败，不能批准交付。** `final_evaluation_v1/complete.json`已完成52条路线、source/idle两种起步共104rollout：全路线门**0/104**，source-history窗口门0/52，单独geometry门4/104，101/104到终点平面，3个runtime失败。最差穿透**0.5404780507m**、横偏1.2771065235m。原flat脚本在604帧离开约8.2m宽原scene后失败，记录保留。另建事先固定的64m平面（不移动地形、不改输出）`final_flat_arena_v1`完成1084帧/135包，0runtime失败/0unknown/0uncertain，但峰穿透**0.0305039566m>0.02m**，仍失败。所有报告weights_unchanged、0optimizer。不能删失败、放宽门、加IK/锁脚/平滑或把源Root/源token送给actor。

`final_codec_ceiling_v1`零更新诊断已完成，PID4019942退出/GPU0释放，不重复：5个去重窗口包含固定up/down/idle/walk及两条最差路线（其中一条与固定up重合）。原actor输入不变；source autoencoder只作明确离线诊断。三个楼梯窗口Root planner Y最大误差约0.06957/0.18232/0.10448m，actor相对腿MPJPE约0.09274/0.05037/0.15906m，源codec重建分别0.03209/0.02644/0.03661m；源重建的down穿透0.04245m，另一up为0.02010m。编码直解与索引解最大差2.86e-6至5.48e-6。模型预测与codec重建均存在误差，不能仅归因于UI或执行图；这5窗不能代表全量train/validation统计，也不证明大batch是原因。

**本机交互链路已实测，仍为未批准候选。** 最终紧凑权重804128550bytes、SHA0f5d8e2b05b8ec518a104a4979cca70f23efa64080abfc843f874634de5a9646，通过自有私有Drive下载并核hash。`final_protocol_v1`在最终模型上通过HTTP/WebSocket24帧、官方18056顶点rig、串行payload/RNG相同及profile/提前ack拒绝；compiled活跃token相同、FK最大8.79e-7m通过原3e-5门，0optimizer。独立本机**28943 PID167808**使用`final_acceptance_flat_v1`、verified-cuda、`--acceptance-test`；固定321x321/dx.2/[-32,32]平面，原4帧事实idle history，复用原权重硬链接。RTX3080 10GB浏览器已实际W/A/S/D各按3秒，约75秒连续到2243帧，DOM采样29.8/30.0fps、0underrun，放键后约47秒仍正常；截图和原DOM trace为`local_flat_wasd_v1.jpg`、`local_flat_wasd_v1-trace.json`。当前暂停，IAB tab12保留。原rugged浏览器第82帧越出已知地形失败也保留`local_rugged_acceptance_failed.jpg`/`local_wasd_rugged_trace.json`。这些不是全量几何验收或质量批准。**28942/28941/28936及用户页面未改。**

本轮21:00全8GPU index/UUID/显存/利用率/真实compute PID已复查：0/2/3无计算进程、约16-18MiB；1为他人66494占53110MiB，4/5/6/7为他人3662414各53656MiB。零利用率有PID仍占用。下一次重新扫描再分配有依据的零更新工作，不为填满显存做无效分配、不重复已完成评估。所有完成/迁移/实际loader proofs/路线/平地/协议/codec诊断小报告已下载`out/pfnn_mixed_20261015/final_results_v1-small.tar.gz`，SHA64a8734282f12dce546adf70347e3f69a9bd90aef12886518d7cece45c53dee5，并按原目录展开；大模型/动画仍只私有Drive。每30分钟跟进继续，正常无变化不通知，最终质量/本地交付真正通过才停用。

## Historical Work: v15 Verified Native Pose Recovered

2026-10-16 18:22 Asia/Singapore：**正式训练已在v15恢复，未丢任何Adam/样本更新；v14图执行禁止重启。** 当前监督PID3099868（task根`resume_verified_pose_after_graph_failure_v1.py`），Pose PID3100009，stage目录`full_training_v15/pose`，训练仍使用原不可变**code_full_workflow_v13**。GPU2/5、global6144/micro3072/.9、原生coalesced frozen encoding/cache/目标teacher RNG，**不使用label图**。18:22实际**actual2860 / reference22640 / 5795840曝光（45.28%）**，无failure；恢复段276次Adam均4.66330s/6144，非全历史均速。最终仍actual4000/reference50000/12800000曝光；Root仍v8/root、codec仍v12/codec。v15是同源码/同contract直接恢复，**没有新的v15/migration.json**，查看`v15/recovery.json`；Pose contract仍绑定原v13/migration.json身份，不能把这误判为回退样本或缺失迁移。

v14在实查空闲GPU0/4预加载后，原v13 Pose2816402于**actual2584/reference16016/4100096曝光**完整保存并按计划rename barrier退出，v13监督2815677已停止；`v13/pose_graph_handoff.json`、`pose/graph_handoff_barrier.json`及latest.pt/latest.tmp.pt/latest_before_graph_handoff.pt/graph_handoff.pt全部保留。handoff SHA**4c1e0cb2b7038d344ab860fabf932c6515634e9abe917f123b65be9aa6005578**。v14监督3042913/child3070676已退出；首个`FrozenGraphPoseMicrobatches._prepare_graph`在base backward/Adam前的CUDA capture失败，v14/pose只有contract/failure，没有任何progress或checkpoint，**0新增正式更新**。Graph错误为“operation failed due to a previous error during capture”，根因尚未确定；不能把早前准备好batch的数值/optimizer证明冒充实际DataLoader/preload环境已通过。GPU4在分配之后另有他人7378MiB进程，分配前扫描确实无进程；没有修改他人任务，也不能仅凭这点断言它是capture根因。`v14/production_rejected.json`明确撤回生产准入，保留所有失败和曾通过的受控证明，禁止凭旧proof直接再部署。

恢复仅消费上述完整2584checkpoint，v15重新全卡扫描选GPU2/5、复用v13原训练源码及contract；model/Adam/scheduler/RNG/sample原样接续。**首个恢复checkpoint审查已完成，不重复健康后续审查：** `first_recovery_checkpoint_review_v1.json`在actual2625/reference17000/4352000曝光核228组Adam均2625、scheduler17000、2CUDA RNG相同、与handoff完整contract逐字段相同、最终codec parent/保护hash一致，重算16016→17000的984原参考批sample_digest一致，文件inode/size/mtime稳定。0optimizer重放/0全tensor重hash，不代表质量；原JSON与`full_training_v15-recovery.json`已下载。失败/恢复小证据包`pose_graph_failure_recovery_v1-small.tar.gz`（54745bytes，SHA5a355a212ed9c62c853e2726202aff45fc1564df7e3d1528fba3dd5d01904d08）已下载。原v13/v14均不能恢复，Root/codec不重跑，所有先前checkpoint审查不重复。

本轮新增受控graph optimizer验证确实完成且仍保留：`pose_graph_optimizer_workflow_v1` PID2961474已退出，GPU0/6上的4连续+全新进程2续训共**6诊断Adam/0正式预算**，复用既有native6144 reference（未重跑4步）；output/loss/梯度/model/Adam/scheduler/RNG/validation及1536尾批逐值相同。wrapper `frozen_graph_pose_training.py` SHA72d3fcf700504fc9136fe9c4477e1532b546b21c18819a01dec410716dc444fb，probe538a88e25df183933c2fc5de39fa51938d8360832752faea23ba313f4e7c0032；每张卡在并发前串行预捕获，warmups3，最终signature[18,6]。连续/冷恢复报告SHA139a1eb6cf37c1e952d2c478f3fa3cfbf67fa55d4f58d48f02134e9b1efdca89、e53be19616f90b7ffbabe373a6c38e550742ca247fa7893fc8ce8a55fd45b242，midpoint0370d87086cc1f35f4a647aaf6d19e89a846c7e0c63577210ce7961cfc9c548c；小包`pose_graph_optimizer_proofs_v1.tar.gz` SHA42532a5d23e561d8e3d579e40a7687cc09abf5c7d929a72528b819c2107d01b0。106项本地迁移/预算/验收检查及远端CPU预检通过，但这些**未覆盖真实4worker/pinned DataLoader及预加载生命周期**，因此没有保证v14生产capture成功。热4.13/4.24秒含诊断tensor比较，不作为生产加速。所有这些已完成诊断勿重复。

当前唯一新增诊断队列：**`pose_graph_loader_isolation_v2` PID3179111**，不可变`code_pose_graph_loader_isolation_v2`；先读其state/complete/failure，不重复启动。v1/3150836在首case GPU复查不再空闲时0CUDA/0更新退出，已保留；v2使用两次间隔10秒的全GPU空闲扫描，每child仍重查，选可用卡中的较高编号，不固定等待某张卡。四个全新进程控制为global capture+pin on、global+pin off、thread_local+pin on、native+pin on；全部真实4worker/persistent DataLoader、原50GiB预加载生命周期、同一正式2584checkpoint/Adam驻留/样本起点，**0optimizer**。每case限600秒，捕获失败是明确的对照结果，其他错误停止；完成后与native的输出/梯度/RNG/样本逐值比较。18:22 `pose_graph_loader_global_pin_v2`已在GPU0/4复现生产capture失败，0更新；其他控制待空闲资源，尚无根因结论或修复准入。DataLoader pin线程和capture模式目前只是待检验变量，不可直接套thread_local/关闭pinning，更不能把0更新结果当生产optimizer证明。真正修复仍需实际DataLoader环境的真实Adam/验证/冷恢复及不丢正式进度的独立迁移。18:18全卡2/5本Pose，1/3/4/6他人，0/7当次空闲；须重新核全部卡，0%且有PID不空闲。最终路线/本地WASD/30fps仍待，正常健康进度不频繁通知。

2026-10-16 17:22 Asia/Singapore：**Root与codec均已完成；正式Pose已双卡无损接续。** 当前监督`full_training_v13` PID2815677，Pose PID2816402，不可变`code_full_workflow_v13/scripts/continue_pfnn_pose_exposure_workflow.py`，真实CUDA_VISIBLE_DEVICES=2,3。global6144/compute-micro3072/memory_fraction.9、冻结最终codec的新临时标签cache、coalesced frozen encoding、显式目标CUDA teacher RNG隔离；每Adam推进24个原batch256参考步。实读**actual2167 / reference6008 / 1538048曝光（12.016%）**，无failure；新167次实际Adam均4.47836s/6144，即.186598s/参考步，对原Pose前2000步均.617423s/256约**3.31倍真实样本吞吐**。这是接续段生产均值，非全历史或probe。最终Pose预算actual4000、reference50000、12800000曝光，总阶段仍38400000曝光/150000参考步；不宣称新旧batch的optimizer轨迹相同。每卡实际约64.3k/63.2kMiB，90%为上限，不做无效填充。17:22全8卡index/UUID/显存/利用率/真实计算PID已核：2/3本Pose，1/4/5他人，0/6/7空闲；下一次必须重查。

codec在`full_training_v12/codec`完成**20125实际Adam / 50000参考步 / 12800000曝光**，weights SHA**7f8ce3b2498168f9bfa2490bbec52a5f30bce5d0f8b271d8e7f76120aabd61fa**，protected_hash_passed=true。监督核codec曝光/hash后才启动正式Pose，startup2步/512曝光计入原Pose预算，并从同一optimizer接续。Root仍`full_training_v8/root`，50000实际更新/12800000曝光，weights d3cdac4d0c260361f275a5ee8f940298486ecf201f00e2051494e93d7dce63e8。禁止重跑Root/codec。原v12 Pose PID2754662在**2000实际更新/512000曝光**完整保存后按预期rename barrier退出，旧监督2246048已停止；`pose_exposure_handoff.json`、`pose/exposure_handoff_barrier.json`和latest.pt/latest.tmp.pt/latest_before_exposure_handoff.pt/exposure_handoff.pt全部保留。此IsADirectoryError是主动交接，不得恢复旧v1-v12监督或旧Pose。v13/migration.json绑定handoff SHA**4db8e2c507faf78d9e8949fef94c2ea90ddf8199f9dbcf5af2314cf07ad38083**、完整旧新源码/四proof/诊断midpoint，no_optimizer_updates_lost=true；本机`full_training_v13-migration.json` SHA8815c74bbf64ceb846ec34d5d796eb1a81adf7164adb91ae906b0a684f262dc7。2000交接使剩余48000参考步整除24，不补尾样本。验证/保存按曝光跨越周期触发（首批3008），不能用actual Adam计数或只检查整除1000。

**首个双卡Pose检查点已核完，不重复健康后续审查。** `first_pose_exposure_checkpoint_review_v1.json`核actual2042/reference3008/770048曝光，228组Adam step均2042，scheduler.last_epoch=3008，2CUDA RNG逐值相同，parent最终codec/protected hash/migration绑定正确；从旧2000参考步重算1008个原参考批的sample_digest一致，读取前后文件inode/size/mtime稳定。仅元数据/RNG/样本链检查，无optimizer重放、无全tensor重hash，不是质量批准。报告已下载`out/pfnn_mixed_20261015/first_pose_exposure_checkpoint_review_v1.json`，原mtime整数以此原始JSON为准。

本轮Pose证据全部已完成，不重跑：旧3072参考在GPU7完成4诊断Adam；旧resume_v1在启动前GPU不再满足空闲门，**0更新**退出，三个旧CPU监督2509671/2606024/2656925已退出，失败证据保留。新恢复监督2772101（`pose_execution_recovery_v2`）仅重分配未执行的resume，在实查空闲GPU2用同一reference/midpoint完成2步冷标签恢复，output/loss/全部梯度/样本/冻结tensor/model/Adam/scheduler/RNG逐值相同。随后GPU2/3的teacher隔离64/3072/1536及异常恢复0更新通过；global6144的serial4、dual4、fresh2-step resume含1536尾批和第三步后128/domain验证全部逐值通过，共10诊断Adam。总共16诊断Adam（4复用+2恢复+10并行）/0正式预算，全部临时标签未进正式训练。独立Philox增量3072=50339008、1536=25169600，不从oracle取RNG。核心实现仍c9d3e582149c621048d2a21ca126ba66ac793025474d120423cb54ab072928df。四准入proof（teacher/reference/continuous/resume）SHA依次60eebe3b2b2620a14a7da50eaa047e9ffcb97e9bf1db08435a7bfb1f29fba196、71ad84fa9002a183b6b6b72c1dc6fe8902239ae97696ec86f1cd5e6005fa6fce、188d4ba66f2fcc4848f32f2c863386dd46f064a989c4a6538330634c93993f1f、082b8b32aedc3cef83307d011931bc59e6e069eeaf6771530398ca9a75220034。105项本地预算/迁移/旧实现检查通过；后续78项接线/交接/验收检查通过，新增Pose6144最终账本验收用例后25项验收检查通过（有重叠，不相加冒称独立总数）。远端CPU预检先发现诊断快照缺完整训练依赖、继而修复Path参数类型；失败快照保留`code_full_workflow_v13_missing_dependency`，均0CUDA/0更新。正式v13由完整v12快照叠加已核Pose文件构建，CPU预检通过后才预加载。小报告包`pose_execution_proofs_v1-small.tar.gz`（4444bytes，SHA4df49da468ca62186a49b37844b7be82de2069adf46e4363cc31bb3ec03efca9）已下载，所有大tensor仍远端。

**新增冻结label图候选只有0更新证据，未部署。** 实查空闲GPU0执行`pose_frozen_graph_probe_v1`，不可变`code_pose_frozen_graph_probe_v1`，固定正式Pose2000 handoff（最终codec50000）后，每次强制清空标签cache，比较3个真实3072训练micro及64验证micro。只在冻结codec标签编码内重放完整dilated GEMM，Pose反向保持原生；tokens/原stride/logits/loss/全部Pose梯度/CPU-CUDA RNG逐值一致，全部模型tensor未变，0optimizer。热3072 forward/backward为3.3787/3.3846s→2.1964/2.1965s（约1.54倍该段速度），初次图9.629s及64首次1.344s包含capture，不当稳态。6训练/12含验证signature，peak reserved68656562176bytes。探针PID2908337已退出/GPU0释放。graph核心仍9c8750225ca9eae0220d6a3a9a40c192bab7d4dba4ab1a7720601faf1aa1fca1，脚本SHA9aea91b6d575ebec96e805ad1b45bd6a7518969df1cf896c26587162b288565d。**不得凭此改生产**：仍须每卡串行预捕获，避免并发临时CUDA capture；双卡真实Adam/定期验证/冷进程恢复、冻结码本与原数值门全部通过后再另建不可变迁移。可复用既有global6144 native参考tensor做新候选比较，不重跑已完成reference。当前正式v13不使用训练图，actor也不使用此标签图。最终完整路线/本地WASD/浏览器/30fps仍待；本机实查RTX3080 10GB，已有约6476MiB占用，后续只能实际测试，不把A100推理速度当本机帧率。

2026-10-16 16:25 Asia/Singapore：正式v12监督2246048/codec2246232仍健康，GPU0/2/3/6、global2048/micro512/EMA64/.9未变；现**actual19420 / reference44360 / 11356160曝光（88.72%）**，接续段均1.75877s/2048，无failure，Root完成/Pose未开始。全部8卡UUID/显存/利用率/计算PID再次核验，仍无空闲卡；GPU1/4/5/7均他人，4虽曾瞬时0%但仍有openpi PID2277407约53GB，不是空闲。两个已有Pose CPU监督2509671/2606024均健康等待，没有重复启动或GPU诊断更新。

**新增第三个有依赖的Pose验证队列，尚未部署生产。** 不可变`code_pose_parallel_exposure_v1`、CPU监督**PID2656925**，状态`pose_parallel_exposure_workflow_v1/state.json=waiting_for_teacher_rng_proof`。只有`pose_teacher_rng_v1`通过后才顺序启动`pose_parallel_exposure_reference_v1`（1空闲卡）、`parallel_v1`（2空闲卡）、`resume_v1`（2空闲卡），每child限900秒、启动前重新核全部GPU/计算PID，有失败即保留停止。候选`ParallelPoseMicrobatches`采用global6144/micro3072/.9、目标卡teacher RNG隔离、按micro顺序归并梯度、仅主卡Adam；首次每shape校准真实dropout Philox消耗后恢复原RNG/梯度，后续独立派发并检查偏移，不能从参考oracle取RNG。每step将2CUDA RNG规范为同一逻辑终态。对照为serial4/parallel4/全新进程2步checkpoint后续2，共预期10诊断Adam/0正式预算；当前仍0。每条4步序列的样本数为6144/6144/6144/**1536无补样本尾批**，总19968诊断曝光/78参考步，第三步后原128样本/domain/micro64验证；保存并比较logits/loss/全部梯度/样本、冻结tensor及model/Adam/scheduler/CPU与规范化2CUDA RNG终点。使用同一provisional codec7000，冷恢复缓存为空。新增23项本地预算/尾批/入口拒绝检查通过（含正式2步startup后global6144准确到12800000曝光、尾批1536），远端CPU导入/语法通过。**尚无GPU runtime或多卡optimizer通过证据，没有正式trainer接线/迁移许可**；通过后还须按实际正式Pose进度接入曝光账本、跨验证周期、source/proof/parent和无损迁移。不要直接改正在运行或将启动的v12 Pose。

候选实现SHA c9d3e582149c621048d2a21ca126ba66ac793025474d120423cb54ab072928df，probe8533ed02d2ef6047d014cfe829cb0862d62e95ef180547991c6587b7150002fd，workflow90d94b71d7b123cd476b5a77e4e2715ff9861144b782cfe45921a98fbf2643f4；绑定前置teacher/helper/stage源码SHA，拒绝源漂移。当前全GPU/真实cmdline及三队列证据`out/pfnn_mixed_20261015/pose_parallel_preparation_20261016_1625-status.json`。下一次先读三个队列state/complete/failure；排队有依赖，不要重复或并发抢前置验证的GPU。正式Pose仍必须基于最终完成codec冻结后新建标签/embedding；最终完整路线和本地WASD30fps尚未通过。

2026-10-16 16:14 Asia/Singapore：正式v12监督2246048/codec2246232仍在GPU0/2/3/6健康运行，实查cmdline/cwd/CUDA_VISIBLE_DEVICES未变。当前**actual19040 / reference41320 / 10577920曝光（82.64%）**，无failure；接续段均速1.76277s/2048、.220346s/参考步，剩余1085次实际Adam按此速度约32分钟，仅为codec估计，不含Pose和最终验收。Root已完成，Pose未开始。全部8卡index/UUID/显存/利用率/真实计算PID已复核，**此刻无空闲卡**：0/2/3/6本codec；1为66494，4/5为他人openpi及gipc进程，7为他人2507733（约8104MiB），不得因显存尚有余量或瞬时低利用率与其共用。未重复checkpoint结构审查或已完成诊断。

**Pose大批量真实optimizer/冷标签恢复已排队，不要重复启动。** 不可变`code_pose_exposure_resume_v1`、CPU监督PID2509671、`pose_exposure_resume_workflow_v1/state.json`为waiting_for_idle_gpu/reference，每15秒扫描全部GPU并由child在启动前复查。预期reference连续4次实际Adam、保存2步midpoint后全新进程续2次，共6诊断更新/0正式预算；当前尚未启动GPU child，实际诊断更新0。global/micro3072、memory.9，按曝光每Adam推进12参考步，第三步后插入原128样本/domain、micro64验证；核logits/loss/全部梯度/CPU-CUDA RNG/sample/model/Adam/scheduler及冻结tensor。复用`codec_pipeline_inputs_v1/checkpoint.pt`（provisional codec7000，SHA545ff06bade7308c71bb7f8a8bcb25fa717969b6637b1c569a96ca117b2e9c0c），临时标签不持久化、不进入正式Pose。每child限900秒，失败保留且不自动重试。该证明尚未通过，也不涵盖多卡或最终不满3072的尾批。

新增独立`code_pose_teacher_rng_v1`和CPU监督**PID2606024**，`pose_teacher_rng_workflow_v1/state.json`为waiting_for_dependency，0optimizer；只有上述两份warm/cold报告通过且实查两张卡空闲后才启动`pose_teacher_rng_v1`，不抢前置验证的卡。源码发现原`pose_teacher`在只恢复目标GPU的fork_rng内使用torch.manual_seed，后者会重设所有可见GPU，因此不宜直接套用Root并行器。新增显式opt-in `isolate_pose_teacher_rng`与`pose_teacher_rng.py`只保存/播种/恢复目标CUDA generator，默认路径不变；仅允许Pose和full factual history。新独立诊断将用真实64/3072/1536样本分别比较原单卡与双卡并发teacher的output/loss/全部梯度/目标RNG/共享CPU RNG，检查异常恢复及全部模型tensor不变，**0optimizer且不是并行optimizer准入**。候选尚无GPU通过证据，未接入正式v12或actor。19项本地教师/cache/stage/guard检查通过，1项双CUDA测试因本机不足两卡跳过；远端CPU导入及语法通过。helper SHA d76ebca9770ebb90a64fe9134728974eed1f2704a406fba22e74352a3f8d5a43，teacher44889afd98bbe26b978b131bb9a6fec43d2f6a9caf84091cdc9af84552e1f2f9，stage37c672b82473bb82ef3bfe50b3554f56795202f663207b9c488daa60a1fd7a9a，probe d68d3fb29bf882d39a571b90702ab7a246cae53cf279b136d1e0c20681957caa。当前状态证据`out/pfnn_mixed_20261015/pose_preparation_20261016_1614-status.json`。下一次先读两队列state/complete/failure，健康等待任务不重开；正式Pose仍必须等最终codec冻结后新建标签和输入embedding。

2026-10-16 15:13 Asia/Singapore：v12四卡生产保持健康，实际cmdline/cwd为code_full_workflow_v12、CUDA_VISIBLE_DEVICES=0,2,3,6，监督2246048/codec2246232未变。现**actual16950 / reference24600 / 6297600曝光**，无failure；新450次Adam均速1.91354s/2048，即.239192s/参考步，含初始化整体约原v9单卡**2.70倍**。最近actual16920→16950的30次真实更新（同一验证周期内）均速1.64850s/2048，即.206063s/参考步，约原v9单卡**3.139倍**；这是真实生产区间，不是probe或全历史均速。15:13全部GPU/UUID/显存/利用率/真实PID已复核：0/2/3/6均本codec且该次采样100%利用率、约37.4/36.4/36.4/48.0kMiB；1/4/5为他人，7空闲。显存上限90%仍有效，按吞吐选512微批，不做无效填充。

**四卡首次账本检查已经完成，不要重复健康后续审查。** `first_four_exposure_checkpoint_review_v1.json`在15:09检查当时最新reference23000 checkpoint：actual16750、5888000曝光、102组Adam step均16750、scheduler.last_epoch=23000、4CUDA RNG终态逐值相同、保护hash与parent匹配、migration绑定正确。按原batch256抽样器从handoff reference21000重算2000个参考批次，sample_digest与正式checkpoint一致；读取前后inode/size/mtime不变。只检查元数据/计数/RNG/参考样本链，未重hash全部模型tensor或重放optimizer，不作质量批准。reference22000与23000定期验证/保存已产出。报告本机out/pfnn_mixed_20261015/first_four_exposure_checkpoint_review_v1.json，v12/handoff_verified_v1.json已下载为full_training_v12-handoff_verified_v1.json。

最新迁移本机`out/pfnn_mixed_20261015/full_training_v12-migration.json` SHA33ceacdd5426c629c36a69aa8a5da0b2d74bc17c9169f043d3243277c451bf62。四卡proof/输入receipt/首检查点/Pose3072小报告包`codec_four_exposure_proofs_v1.tar.gz`（3476bytes，SHA947053558a5ad00fae8f783309f72a2d87e2eea87ca0af58dece74d473f68ae1）已下载；3072单独报告为pose_exposure_profile_3072_v1-report.json。所有本轮探针均已完成退出，不重跑。接下来继续v12四卡codec；利用真正空闲GPU为最终Pose的大batch、optimizer/冷标签恢复及并行准备做有依据的验证，正式Pose标签只能在最终codec冻结后新建。完整路线质量和本机WASD/30fps仍待最终模型验收。

2026-10-16 15:02 Asia/Singapore：**已正式四卡接续，full_training_v12监督PID2246048、codec PID2246232、GPU0/2/3/6**，不可变code_full_workflow_v12。global2048/compute-micro512/EMA64/.9显存上限；每次实际Adam处理8个原batch256参考步、32次原顺序EMA，LR/保存/验证按参考曝光推进。15:02已actual_updates16580、actual_exposures5539840、equivalent_reference_updates21640/50000；新80次实际Adam均速2.50044s、每参考步.312555s（含四卡初始化），对原v9单卡.646848s约2.07倍真实生产提速，不能把诊断3倍冒称此刻生产速度。无新failure。最终codec actual_updates预算**20125**，曝光仍12800000；实际Adam数和等效参考步必须分开读。Root已完成且仍v8/root，后续codec/Pose读v12。

原v11双卡codec在**actual16500 / reference21000 / 5376000曝光**完整保存后按预期exposure_four rename barrier退出；原监督2117202已停止。v12/migration.json绑定SHA02995709d6327787d942f05eb0739b2a4609ac8e15a90f5380f429325b0a04b1、旧新完整源码及三份四卡proof、no_optimizer_updates_lost=true、optimizer_trajectory_equivalent=false。v11/codec_exposure_four_handoff.json和codec/exposure_four_handoff_barrier.json保留，latest.pt/latest.tmp.pt/latest_before_exposure_four_handoff.pt/exposure_four_handoff.pt均保留；IsADirectoryError是主动交接，禁止恢复旧v1-v11监督或旧codec。

四卡证据：固定正式v11 reference18000/actual15750 checkpoint，`codec_four_exposure_inputs_v1/checkpoint.pt` SHAa555fbc2dfc4b84949fcf4a5f063bbc8cc3b2267e91417e659079343eff3e1cd。新global2048/micro512/EMA64/.9的serial4、四卡4、四卡2步checkpoint后全新进程续2，共10诊断Adam/0正式预算；均含原定期验证，output/loss/gradient/EMA/CPU与4CUDA RNG/model/Adam/scheduler全部逐值一致。参考sample起点用actual_exposures/256，不能用不再相等的Adam计数。热四卡1.677..1.713s/2048、恢复1.692s，serial5.85s/2048；这是新策略自身串并行/恢复证明，不是新旧global batch优化轨迹等价。三proof SHA11d3377ae725301f5ff56ba6e0a9c886e1aa5cbf13080f4c28f9b92008fc3933、ba1f557a4fc7fdb49ac87a9656d6f5acde0c957beaa0ed41143e9e0db20d6adc、e4fca0e9ecb307a8cb1310c4cc400684c88cb74e995451e927a3ab22bb60e2eb；pipeline SHA91bf28e9ef178c9a025928c800c691a281f3f0d39e128228702ac957134c3a66，graph/EMA helper仍9c8750.../d2fe9d...。诊断workflow2196324已全部完成勿重跑。四卡扩展后的80项本地迁移/预算/loader/acceptance测试通过，远端CPU迁移校验通过。每次四卡分配均核全部UUID/显存/利用率/计算PID，其他任务不变；临时50GiB/卡预加载工作区仅保住即将使用的资源，正式图构建前已释放，不作为吞吐或训练曝光。

此前双卡阶段已真实运行：v10在GPU3/6预加载期间GPU3被他人PID2072802占53196MiB，只有自有零更新预加载2072266和监督2071657被停止，resource_reassignment.json保留，正式旧v9不被回滚。v11监督2117202/codec2117351转用实查空闲GPU4/5，从v9完整15000Adam/3840000曝光接续，SHA287d93419a4cfbfa860e3038e1c335fc29a5e6fd3e7c53053c3c4b1e0140c807。v11首个检查点审查已完成：actual15250/reference16000/4096000曝光、102组Adam=15250、scheduler.last_epoch=16000、2CUDA RNG相同、原参考抽样链从15000重算1000参考批一致、保护hash与迁移绑定正确；报告out/pfnn_mixed_20261015/first_exposure_checkpoint_review_v1.json，不重复健康后续v11审查。v11新增1500次Adam均速1.66818s/1024，即.417045s/参考步，约原单卡1.55倍；现已主动交接给v12，不重启。v12四卡首个定期检查点须另做一次新账本/RNG拓扑检查。

GPU6与之后GPU7的必要Pose性能工作也已完成：`pose_exposure_profile_v1`用provisional codec7000和micro1024，冷热缓存loss/全梯度/CPU-CUDA RNG逐值一致，模型不变、0optimizer，热.648..650s、peak reserved22.52GB；`pose_exposure_profile_3072_v1`用micro3072，仍全部逐值一致、0optimizer，热1.843..1.856s、peak allocated59.97GB/reserved65.61GB。后者利用接近80GB卡容量的有效工作区，尚不是正式Pose策略或optimizer恢复证明。全部临时标签随探针退出，正式Pose仍必须等最终codec冻结后创建新标签。未来可据此做大batch/多卡Pose的真实optimizer与冷缓存恢复验证，不重跑这些0更新探针。1024报告本机out/pfnn_mixed_20261015/pose_exposure_profile_v1-report.json SHA2c257429ce8c67c71fd5601e783cfb208ad565c1d157fea23acdbc877f06cddb。

双卡proof小包已下载out/pfnn_mixed_20261015/codec_exposure_proofs_v1.tar.gz（4330bytes，SHAc0dd1feb103a6aef5b2fe7bf2c3796607fb58175a236afa6c04afe94cf5ecdd6），v11迁移/实进度/首检查点报告也已下载。SSH短暂连续关闭连接后自行恢复，严格主机验证和原key未改；未因监控连接中断重启健康训练。最终完整路线质量、本机WASD/浏览器/30fps仍未验收，跟进保持每30分钟直到真实交付。

2026-10-16 14:05 Asia/Singapore：用户最新明确要求“多卡加速啊，而且80G显存尽量跑满，等效50000步即可”。**此新授权覆盖旧的每stage必须50000次实际Adam限制**：每stage仍12800000样本曝光/50000个batch256参考步，允许大batch减少Adam次数；学习率按曝光进度，实际Adam计数独立保存，不宣称与旧优化轨迹逐值等效。数据源配额50/25/25、原生loss、Root冻结范围、全部数据及最终质量门不变。codec计算微批512但quantizer仍逐64样本顺序EMA，全局1024含16次原EMA，不能改成每512一次EMA。旧严格50000实际更新的记录保留为历史。

**新监督full_training_v10 PID2071657、预加载codec PID2072266、GPU3/6**，代码`code_full_workflow_v10/scripts/continue_pfnn_codec_graph_workflow.py --exposure-pipeline`。14:05状态preloading_codec，原v9监督1506039/codec1506519/GPU4仍健康，14310实际更新/3663360曝光/.64744s。新阶段将采用global1024/compute-micro512/EMA64/memory_fraction.9，等原进程在下一完整checkpoint（预计15000）保存后原子rename barrier无损接续；此刻尚无v10生产步，禁止按诊断速度冒称生产。新supervisor只有预加载ready后才停止旧v9监督；Root仍full_training_v8/root不重跑，后续codec/Pose写full_training_v10。后续必须查v10 state/migration/progress及v9 codec_exposure_handoff.json/exposure_handoff_barrier.json；不得恢复已交接的旧监督。Pose当前仍原已验证cache/coalesced策略，未来大batch/多卡须另有同配置数值和恢复证明，不能提前使用未冻结codec标签作为正式Pose数据。

本轮新显存策略实测`codec_large_micro128/256/512/1024_v1`各0optimizer，.9限额，固定7000checkpoint、每轮恢复初始tensor。热每微批约.379/.721/1.433/3.026s；reserved峰约22.0/25.8/49.6/67.0GB。1024微批虽更接近80GB容量但吞吐较慢，选512。全部扫描真实空闲后才启动，最初计划5/6时5已被他人占用，实际选3/6；无他人任务变更，显存不做无效占位。

`codec_exposure_reference_v1`用原生产单卡图核心、global1024/micro512/EMA64/.9执行4次新策略Adam（每次4个参考步）；`codec_exposure_parallel_v1`全新进程双卡4次、`codec_exposure_resume_v1`从双卡2步checkpoint全新进程续2步，均插入原128样本/domain的64微批定期验证。**所有output/gradient/loss/EMA/RNG/model/Adam/scheduler及终点逐值一致**，10诊断Adam/0正式预算；这是新大batch策略的串并行/恢复一致性，不是旧batch256优化轨迹等效。双卡热1.550..1.553s/1024，串行约2.91s/1024；相对旧生产.647s/256预计约1.67倍仅诊断，待v10真实吞吐。三proof SHA依次74ef56c803f2c83fc6f9b91c6bdf04e8a28a57625f2b82148e39b4042b0b34f4、0d00f50bf395deb675db755b892a791536e9ca7c5840dc250ad2b4e6800b52ac、07d4c412ed969266bc58d8c761064f91485f09b167536abf5f3483f1bac1c344。pipeline SHA d863e5a5172fce502f22194d915cab8bf9d7d07d72fd66dc44f45599a632603a、graph SHA9c8750225ca9eae0220d6a3a9a40c192bab7d4dba4ab1a7720601faf1aa1fca1、EMA helper SHAd2fe9df16b80561c8d55af0f896e236a6ba17c4ef078376c43b6765fec7d86b1。

114本地测试通过（预算/分组样本顺序/迁移拒绝/acceptance绑定/旧训练和cache迁移）；远端CPU validate-only通过。新ExposureBudget保留原batch256抽样链和曝光总量，sampler只组合原参考批次，数据/retarget/cache/审计没有重跑。导出与验收能绑定真实Adam数和曝光预算，旧模型仍严格按旧契约，所有quality/delivery标志仍false。预检首次仅运行器sys.path遗漏vendor导致CPU导入失败，补运行器路径后通过，未修改模型或训练源码来绕过依赖检查。SSH偶有连接关闭，仅重试读/传小源码；大模型全部留远端。

2026-10-16 13:28 Asia/Singapore：正式v9监督1506039/codec1506519/GPU4继续健康，**10810/50000更新、2767360曝光、0.64812s/update，无failure**；真实cmdline/cwd/CUDA_VISIBLE_DEVICES/TMPDIR已核。全部GPU/UUID/显存/利用率/真实计算进程检查后，5/6确实空闲，其余除4为本codec外均他人。新增`codec_pipeline_buffer_reference_v1/candidate_v1`已在重新核空闲的5/6完成并退出，均0optimizer更新。唯一候选改动是去掉微批返回梯度clone及首份累加器clone，利用每对完成并归并后才启动下一对的生命周期；CUDA stream、.125限额、EMA/RNG/样本策略不变。固定7002 midpoint先串行预热、第二卡eager验证、再算一组真实256/64梯度：原候选与去复制候选均与保留原串行output/loss/EMA/gradient/RNG/sample逐值相同。但此新零更新条件没有复现已知多步验证失败，且主卡peak allocated 8340550656→8340589568bytes几乎相同（reserved仅降低52MiB）；单次0.532/0.549s不构成速度比较。**没有支持扩大该候选Adam测试或生产迁移的新证据**，本轮到此停止该方向诊断，不盲跑旧失败；累计诊断Adam仍52，正式预算不变。新候选SHA bfb7bc730f09110f3cb734a7961eadb0a89a3eab8d266d55aa737289524b51df，仍只诊断使用。

小报告`out/pfnn_mixed_20261015/codec_pipeline_buffers_v1-summary.json`及`codec_pipeline_buffers_v1-reports.tar.gz`（3430bytes，SHA9d010fb801c42b7637da0b8c6381deb9abe1ace49399b6f91f60f5d0fcf4be7d）保留分配、实现hash和全部GPU快照。RNG helper源码检查未提供“当前设备意外漂移”的证据，不据此改RNG实现。Root仍已完成、Pose仍待最终codec；继续原v9生产与每30分钟检查，未重跑数据/审计、未修改正式optimizer行为、未批准WASD交付。

2026-10-16 13:20 Asia/Singapore：本轮双卡诊断全部结束，**仍无双卡生产准入**。最后`codec_pipeline_evaluation_v2`在重新核空闲后的GPU5/6完成4更新；第二卡eager验证的三domain loss与原生产验证逐值相同，但后续末层梯度及model/Adam三项仍不同，paired_endpoint_changes非空，passed=false。此前短程cold v1/v2通过不覆盖定期验证，不可凭它们迁移。全部新诊断合计52次明确Adam更新、0正式预算；所有CUDA探针退出，13:20全GPU/UUID/显存/利用率/真实进程复查5/6空闲，4为正式codec，其余均他人任务。以后只做有新依据的隔离，不盲重跑本轮已完成探针。

正式v9 codec PID1506519/GPU4、监督1506039均未变，现**10070/50000更新（20.14%）、2577920曝光、0.64878s/update，无failure**；Root已完成，Pose等待最终codec。未触碰生产源码/模型/optimizer或其他用户任务。已下载`out/pfnn_mixed_20261015/codec_pipeline_campaign_v1-summary.json`及小报告包`codec_pipeline_campaign_v1-reports.tar.gz`（54338bytes，SHA84ce71b024f136599fbdb2ac0beeb713f3560a0c907af5db15f3b58a4f0f3267），大模型/梯度/cuDNN日志仍留远端。双卡候选只存独立快照；本地dilated_graph_convolution.py的显式device stream改动也未在生产v9采用。纯冷进程2/4步虽全bitwise、热约.4s，不是正式加速或质量批准。仍保留每30分钟跟进及最终本地WASD验收目标。

2026-10-16 13:18 Asia/Singapore：响应用户“迅速利用空闲a100”，已实际使用逐次全卡扫描确认空闲的GPU3/6进行双卡codec流水线诊断；最新GPU3被他人占用，`codec_pipeline_evaluation_v2`已重新扫描后改用GPU5/6（timeout PID1840023，code_codec_pipeline_evaluation_v2）。正式codec始终为v9/PID1506519/GPU4，监督1506039，未暂停或重启；13:13已9430/50000更新、2414080曝光、0.64852s/update。**双卡候选尚未准入，不得迁移生产。** 固定输入`codec_pipeline_inputs_v1/checkpoint.pt`是7000步正式checkpoint硬链接，SHA545ff06bade7308c71bb7f8a8bcb25fa717969b6637b1c569a96ca117b2e9c0c，不重建数据或缓存。

候选`ordered_codec_pipeline.py`保留4次原顺序EMA及一条逻辑RNG，双GPU重叠编码/解码/反向，梯度按0/1/2/3归并、只有主卡Adam。v1导入失败0更新，v2因进程级默认CUDA图capture stream跨设备错误0更新；独立候选为每个图显式指定设备stream，生产v9源未改。v3/v4各10诊断更新均失败：首步仅末层weight梯度差1.490116e-8，后续model/Adam/EMA漂移。v4将预热回滚备份放CPU仍失败。`codec_pipeline_isolation_v1`零更新，四micro输入/upstream/stride完全相同，仅主卡micro2末层raw梯度差7.450581e-9；cuDNN日志记录该层9次engine18后第10次engine0，对应最后主卡micro2，证明该次workspace算法切换。v5提高候选限额至.15的10更新又增加两层梯度差，拒绝，不得用于生产。

新进程保留原.125限额后，`codec_pipeline_cold_v1`从原7002 midpoint恢复2更新、`codec_pipeline_cold_v2`从7000连续4更新均与保留的原生产串行参考output/gradient/EMA/loss/RNG/model/Adam/scheduler逐值一致，后者热0.405..0.419s，仅诊断，非生产提速。原参考来源ordered_codec_pipeline_v5的serial文件（该报告失败的是.15候选，不是serial参考）。但`codec_pipeline_cold_v3`从双卡midpoint恢复2更新并插入正式128样本/domain验证后，末步末层梯度仍差1.490116e-8、终点model/Adam三项不同；`codec_pipeline_evaluation_v1`将验证移到第二卡后4更新仍有同类差异。`codec_pipeline_workspace_v1`在末层反向前empty_cache的4更新从首步就失败，不采用。`codec_pipeline_validation_reference_v1`另2更新证明原生产实现插入同样验证仍逐值通过，且与双卡候选终点确有三项差异，排除不对称对照的误判。最新evaluation_v2只在第二卡验证期间使用原coalesced eager算子，训练仍原图/原.125，等待complete/failure；不是已通过方案。所有失败和大tensor留远端，无正式预算增加，无quality批准。18本地graph迁移/context测试通过，未把此测试当双卡准入。

2026-10-16 12:45 Asia/Singapore：用户询问进度后实查，codec仍PID1506519/GPU4、监督PID1506039，12:44已6740/50000更新（13.48%）/1725440曝光，接续段均速0.65374s/update，无failure；Root完成、Pose未开始。全部GPU/UUID/显存/利用率/真实进程已核，5/6空闲，其余除4为本codec外均他人占用。5000与6000定期验证已经产出，最新6000 checkpoint一次CPU mmap核验通过：102组Adam及scheduler均6000、1536000曝光、.125/graph/4EMA原预算正确、parent最终Root与migration SHA一致、文件inode/大小/mtime不变。报告`out/pfnn_mixed_20261015/first_graph_checkpoint_review_v1.json`，远端同名在full_training_v9根。仅metadata/accounting，未重hash全tensor或重放；此交接后保存检查已完成，健康后续checkpoint不重复。按当时吞吐codec剩余约7.9小时，不含Pose及最终质量/本地交互验收，不能当总交付ETA。

2026-10-16 12:16 Asia/Singapore：**codec加速已正式无损接续并持续更新**。v9监督PID1506039、codec PID1506519，实际cmdline/cwd为不可变code_full_workflow_v9，CUDA_VISIBLE_DEVICES=4、TMPDIR短路径均实核。原v8 codec PID1156626已在4000真实更新/1024000曝光完整保存后按预期barrier退出，旧v8监督PID697518已停止；禁止恢复旧v1-v8监督和旧codec。`full_training_v9/migration.json`绑定4000checkpoint SHA29271d76477ea6ec3ad98cb0d891cbd264c895a5f1cfd59d6f9d4b9bf2d7b677、旧新完整source及三份精确proof，no_optimizer_updates_lost=true。原v8/codec的latest.pt/latest.tmp.pt/latest_before_graph_handoff.pt/graph_handoff.pt均为有效保留文件，原failure的IsADirectoryError是主动交接，不得误修复重开。本机`out/pfnn_mixed_20261015/full_training_v9-migration.json`已保存。

当前codec4100/50000步、1049600曝光，无新failure，graph_dilated_convolutions=true、16图signature、memory_fraction0.125；新100次正式更新均速**0.7655s/update**（含冷图启动），对原codec阶段均速1.2504s约**1.63倍生产提速**，不夸称诊断2倍。峰当前nvidia-smi约9044MiB，仍为单GPU。Root完成结果继续只读full_training_v8/root；以后codec/Pose读full_training_v9，全部loss/4次顺序EMA/样本/optimizer/阶段预算不变。12:16已核全部8卡及计算进程：3/5空闲，4为codec，0/1/2/6/7他人占用；每次分配前重新检查。后续须在首个5000定期验证/checkpoint核健康保存，勿重复旧33kRoot结构审计或已完成诊断。最终模型质量、本机WASD/30fps仍未验收。

2026-10-16 12:05 Asia/Singapore：v9预加载已ready（PID1506519/GPU4、0新optimizer），旧v8监督PID697518已仅停止自身，`full_training_v8/codec_graph_handoff.json`保留；原codec PID1156626/GPU3在3490步后通过短时SIGSTOP/SIGCONT原子布置`codec/graph_handoff_barrier.json`并恢复运行，最新3530步。v9监督PID1506039状态waiting_for_complete_checkpoint，预计4000完整保存后接续。禁止恢复旧v1-v8监督或回滚3000检查点；Root已完成且位置仍v8/root。12:03段落的preloading状态已被此条更新。

2026-10-16 12:03 Asia/Singapore：新监督**full_training_v9 PID1506039、codec预加载child PID1506519、GPU4**，不可变`code_full_workflow_v9/scripts/continue_pfnn_codec_graph_workflow.py`。状态preloading_codec，原v8 codec PID1156626仍在GPU3健康继续，最新3420步/875520曝光；原v8监督PID697518须待新模型ready后才由v9停止自身。新codec需等待原训练在下一完整checkpoint（预计4000步）保存后无损接续，不回滚旧checkpoint。Root已完成并仍保留full_training_v8/root；交接后的codec/Pose写full_training_v9。当前尚未有新生产步，不把下述诊断速度当生产吞吐。新flag为`--graph-dilated-convolutions --coalesce-dilated-bias --cuda-memory-fraction 0.125`，仅codec/global256/micro64/单卡准入；图缓存不进入checkpoint、不进入actor，Pose仍原cache+coalesced/default memory，全部训练预算不变。

本轮进一步隔离成功：`codec_graph_isolation_v1`使用同一真实微批，原生repeat/仅前向图及保留前向缓存时raw梯度全部逐值相同；保留前后向图缓存后，无论图重放还是原生反向，唯一首发差异均为codec.decoder.model.6.weight最大1.117587e-8，其输入/upstream及stride完全相同，EMA/output/loss/RNG相同。该层input64x512x32、upstream64x365x32（stride11680,1,365），kernel365x512x3。`codec_terminal_workspace_v1`直接使用保存输入/weight/upstream，仅改变内存压力：800/1600MiB时梯度与原参考逐值相同，2400MiB时与失败候选梯度逐值相同；降回800MiB仍保留后者。cuDNN9.1日志明确backward-filter engine18（FFT）→engine0并缓存后者，证明该差异来自可用workspace导致算法切换，不再仅是warn-only随机性猜想。0optimizer。

`codec_graph_forward_optimizer_v1`（仅前向图/.1）完成10诊断更新后在下一次重建图时触发显存限额；保留的原4步/图4步终点已出现与旧both/.1类似EMA/output漂移，仍失败，不继续重跑。随后基于workspace证据测试**both图+memory_fraction0.125新配置**：`codec_graph_workspace_optimizer_v1`完成原4步/图4步/图cold2+2后2步/原2步checkpoint转图2步共12诊断更新，所有8组computation gate、模型/Adam/scheduler/EMA/CPU-CUDA RNG及所有终点全bitwise，passed=true。热前后向clip约1.20s→0.60s，单次冷capture约3.6..3.8s，峰reserved约9.48GB，不能与失败/.1的0.5s混称。旧.1候选仍拒绝，新证明只准入固定.125政策。

真实NativeTrainingStage新增显式guard和独立scoped模块forward接线。`codec_graph_integration_v1`再完成12诊断更新，接入后的model/Adam/EMA/loss/gradient/RNG及连续/恢复终点逐值一致；`codec_graph_cold_process_v1`用**全新进程、首次即.125且无预热原生算法cache**从保存的原2步checkpoint继续2步，仍逐值一致，排除依赖原探针算法缓存才通过。三份proof SHA依次4d60a24d9d9a1c8fb2e9397d107cdb01d39ea7085829939e9745c96da6ba7d0a、255bdca56b6615bb52976e65d91122d44e0cb52ef0c8e220cc815dcfd1ec0188、c26241c31bbae9baad63db47e7d1247bb1c3e2cf5930ececc305088e5b169a4a。实现核心仍c28531fa44a2966a7d9c6ad6d454f34a61c914ef0295ca8dcf899631ac1e8613，接线SHA b0bb0e2121248ee46d8e2e1879b4c90539f4bda9be9d39b87d5e7df82a9ace13，stage SHA895b2a7bc5b36c8e262159167118e7028d1af75a0d37d09718697c39af4d4d25。本轮共36诊断Adam更新、0正式预算；53项本地迁移/拒绝/上下文恢复/原训练/cache/barrier测试通过。

8份小报告已下载`out/pfnn_mixed_20261015/codec_graph_workspace_reports_v1.tar.gz`，SHA c82085cb069a055255d3948bbe81403783cdbaea15f9336fdbfbc9c669379c24（4403bytes）；大模型/梯度/cuDNN日志留远端。所有新诊断已结束，准入依据是新.125配置三份精确证明，旧失败全部保留。v9预加载后将仅停止v8监督，短时SIGSTOP/SIGCONT原子设置codec graph_handoff_barrier+55s watchdog，等完整torch.save后预期IsADirectoryError，再核实际计数/Adam/scheduler/sample/hash并接续；后续必须查v9 migration/progress和旧v8交接receipt，不能把预期failure当随机故障恢复。

2026-10-16 11:38 Asia/Singapore：正式codec仍PID1156626/GPU3、v8监督PID697518，2390/50000更新、611840曝光、约1.1997s/update，无failure；Root已完成/Pose未开始。本轮实际使用空闲GPU2/6进行两项有依据的执行候选验证，均**未准入生产**，正式训练始终未暂停/重启。最后全卡实查4/6空闲，2已被他人PID1392325使用；0/2/5/7为该他人进程、1为PID66494、3为codec，不可沿用旧空闲分配。

`batched_gemm_v1`将64次完整kernel addmm改为baddbmm，保留原生convolution_backward。6种实际shape（length8/16、dilation3/9/27）x2seed中前向最大差6.19888e-6，6/12组原atol1e-6/rtol1e-5门失败；线性upstream下三个梯度相同不能证明完整模型。虽然前向约5.2..6.4ms→1.3ms，结果被拒绝，0optimizer、不部署、不重复。

新增`dilated_graph_convolution.py`只在独立诊断中捕获/重放原逐样本full-kernel GEMM及原生反向，不更改EMA/样本/Adam，也不进actor。`dilated_graph_v1`同6shape x2seed的前向/全部梯度/RNG逐值相同，单层前后向约18..20ms→3.2..3.6ms、12图signature、peak reserved约0.94GB，0optimizer。这只是算子证明。`codec_execution_inputs_v1/checkpoint.pt`硬链接保留正式1000步codec，SHA503ea8fcad02ee3eb3c42120e4504f9d0bac08cc08cbca56adf77f8d386dd507。`codec_graph_reference_v1`（GPU2）/`codec_graph_candidate_v1`（GPU6）各2真实batch256/micro64、每batch重置同模型和RNG，`codec_graph_compare_v1`通过原门：输出/loss/EMA/CPU-CUDA RNG/sample相同，梯度分别1/102项有差异，最大9.31323e-9，不是bitwise。热前后向+clip1.1736s→0.5054s，候选16signature，首capture含启动4.35s，0optimizer；不直接作为正式加速。

**真实Adam证明失败，禁止部署图候选。** `codec_graph_optimizer_v1`因缺辅助模块在加载模型前失败、0更新；新不可变v2改用快照已有helper并预先核导入。`codec_graph_optimizer_v2`在GPU2完成原实现连续4步、图连续4步、图2+2冷cache恢复2步、原实现2步checkpoint→图恢复2步，共12明确诊断optimizer更新/0正式预算。3次model/Adam/scheduler/RNG恢复均逐值通过；图自身冷恢复最终全部tensor逐值一致。但原实现与图连续第3/4步、原→图恢复的第2步出现output原门失败及EMA embed_avg/embed差异，最大6.10352e-5；最终model/Adam亦有差异，passed=false。早前0更新梯度小差不能授权累积更新后的EMA漂移，未改任何门。热图约0.49..0.52s只保留为失败候选计时；不能再以0更新full-model通过声称已可迁移。梯度差异具体来自调度/allocator算法还是原warn-only算子的非确定性仍未隔离，不作根因定论。

候选实现SHA c28531fa44a2966a7d9c6ad6d454f34a61c914ef0295ca8dcf899631ac1e8613。8份小报告已下载`out/pfnn_mixed_20261015/codec_graph_reports_v1.tar.gz`（18198bytes，SHA c9ace44819c1bd9d0166b9cc62c3031f2fbf73214d025d97008b413e2b2e5644），大checkpoint/梯度/模型保持远端。所有探针已结束；失败快照和输出全部保留，不盲重跑。正式codec继续原coalesced/.1实现，后续优化仍需完整loss/EMA/gradient/RNG/optimizer接续证明与不丢更新交接。

2026-10-16 11:02 Asia/Singapore：codec同一PID1156626/GPU3已590/50000实际更新、151040曝光，续训均速1.1861s/update，loss/gradient有限、无failure。Root已完成，Pose尚未开始。最新全GPU/UUID/真实进程核验：2/6确实空闲；0/4/5/7为他人PID1149844，1为他人PID66494，均不能动。codec_accelerated_profile_v1已完成退出，报告SHA acd72b16df26cc4a2f48c056f13ca7eed738cf961f36eb97a91b77fdd9d81dfe；没有新的生产优化部署。后续codec加速应依据加速后117ms前向/178ms反向证据审查，必须保留4次有顺序依赖的EMA、原数值门和真实optimizer/checkpoint/sample连续性，不能直接使用Root的微批并行器。

2026-10-16 10:52 Asia/Singapore：**Root正式完成50000实际更新/12800000曝光**，PID697535正常退出，full_training_v8/root/complete.json的protected_hash_passed=true，最终weights SHA d3cdac4d0c260361f275a5ee8f940298486ecf201f00e2051494e93d7dce63e8；监督在放行codec前已复核权重SHA及预算。最后四卡接续段18000更新均速0.35795s，不能冒称整个历史Root均速。最终固定128样本/domain验证loss：PFNN0.04261640/PARC0.08348797/BONES0.00832671，非闭环质量批准。本机完成凭据`out/pfnn_mixed_20261015/full_training_v8-root-complete.json`。Root已经完成，不得重启或重跑。

v8监督仍PID697518，当前job=codec-resume，**正式codec PID1156626、GPU3**、cwd code_full_workflow_v8。startup PID1149779曾在GPU0完成2实际更新/512曝光并按计划退出，后续重新扫描全部GPU选择当时空闲GPU3，以同一latest.pt/Adam/scheduler/sample/RNG续训，2步计入同一50000预算。10:52已60步/15360曝光、均速1.2719s/update，无failure；parent_sha256严格绑定上述最终Root。显式coalesced bias、memory_fraction0.1、global256/micro64/50-25-25/Adam5e-5不变；每optimizer仍4次原codec EMA更新，保留原warn-only deterministic/cumsum限制，不能直接套Root微批并行。Root/共享scene encoder冻结，codec训练完成后才构建最终Pose标签。当前GPU以state.job_gpus及真实CUDA_VISIBLE_DEVICES/计算PID为准，state.gpus和gpu_snapshot_before仍保留初始Root分配，不是codec实时占用。10:52全卡检查：3是codec，0/1/4/5/7为他人任务（0%仍占约53GiB不能当空闲），2/6空闲。

10:57新增`codec_accelerated_profile_v1`已在实查空闲GPU2完成，使用最终Root模型及真实64样本50/25/25微批、与正式codec相同coalesced/.1政策，0optimizer更新、初始模型tensor恢复。一次热分层事件记录前向116.70ms、反向178.39ms；12个512通道dilation3/9/27短卷积前向各约5.14..5.45ms，全部Conv前向合计68.71ms。它测的是当前加速后codec，不是重跑旧未加速profile，且parent attention/layer事件有重叠不能混加。报告`out/pfnn_mixed_20261015/codec_accelerated_profile_v1-report.json`，不可变快照code_codec_accelerated_profile_v1，脚本SHA713c1e0aadf9c6b9497e6e5d7ee1e870c735bac23f28def337f9e167744c9de8。诊断已退出释放GPU2，不重复；本轮没有新的codec生产优化准入。

2026-10-16 09:11 Asia/Singapore：v8四卡Root健康，09:10实读33650/50000步、8614400曝光、均速0.3582s/update，无failure。已对**首次四卡生产checkpoint33000**做一次CPU mmap结构核对：183组Adam step及scheduler.last_epoch均33000、曝光8448000、4份CUDA RNG终态逐值相同、memory0.1/四卡/global256/micro64/50000预算及migration SHA绑定均正确，文件inode/大小/mtime读取前后不变，sample链存在。报告`out/pfnn_mixed_20261015/first_four_checkpoint_review_v1.json`，远端同名位于full_training_v8根；只检查元数据/计数/RNG拓扑，没有全tensor重hash、恢复重跑或质量批准。此项已完成，健康后续checkpoint不要重复核查。33000步每domain固定128样本验证loss为PFNN0.04187854、PARC0.08202951、BONES0.00849925，保留实际值，不作为闭环动作质量证据。全部8卡及真实进程已核：0/3/6/7仍原Root，1他人，2/4/5空闲；未启动重复诊断或未满足依赖的codec/Pose。

2026-10-16 09:01 Asia/Singapore：**四卡已正式无损接续并持续训练**。v8监督PID697518、Root PID697535，实际cmdline/cwd均为不可变code_full_workflow_v8，nvidia-smi确认Root实际占GPU0/3/6/7约2910/2550/2550/2550MiB。原v6 Root PID3413446已在32000真实更新/8192000曝光完整保存后按预期barrier退出，旧v7监督PID3549700已停止；禁止恢复旧v1-v7监督/旧Root。`full_training_v8/migration.json`绑定checkpoint SHA1d8e16f38183fb1ca551a374f311d068f9c100014a417814c1969d5d0e15f55a、旧新完整source、四卡proof及no_optimizer_updates_lost=true。原v6/root的latest.pt/latest.tmp.pt/latest_before_parallel_handoff.pt/parallel_handoff.pt均为有效保留文件，IsADirectoryError是有receipt的主动交接；本机迁移记录`out/pfnn_mixed_20261015/full_training_v8-migration.json`。

当前Root32240/50000步（64.48%）、8253440曝光，无新failure；从32000接续后的240次实际更新均速**0.3937s/update**，相对原双卡约0.5576s约1.42倍、相对最初7.431s约18.87倍。按现速Root剩余约1.94小时，仅Root吞吐估算，不含codec/Pose/质量验收。真实进度以后只读`full_training_v8/root/progress.json`；后续codec/Pose也在v8，全部原预算与v7已验证执行政策保持。检查全部8卡后，旧GPU4/5正常释放，GPU2也空闲，GPU1仍为他人任务；新增工作前必须重查。35项本地迁移/原训练/cache/barrier测试通过，已完成四卡诊断不重跑。

2026-10-16 08:51 Asia/Singapore：已实查GPU0/3/6/7全部空闲并完成新四卡验证，现**full_training_v8监督PID697518、新Root PID697535**，不可变`code_full_workflow_v8/scripts/run_pfnn_parallel_workflow.py`。新Root已完成原缓存hash复核/四卡模型预加载，`root-await-resume.json`确认0新optimizer更新，等待原v6 Root PID3413446完整32000步checkpoint。旧v7监督PID3549700仅停止自身，`full_training_v7/parallel_four_acceleration_handoff.json`保留。原Root在31010已通过短时SIGSTOP/SIGCONT原子布置`full_training_v6/root/parallel_handoff_barrier.json`，有55秒watchdog；仍在GPU4/5继续训练到下一完整torch.save，然后预期IsADirectoryError退出。**此刻尚未完成迁移，不能把四卡诊断速度当生产速度。** 禁止恢复旧v1-v7监督或从31000及更早checkpoint回滚。

`parallel_four_optimizer_v1`使用固定5000步checkpoint和原4个真实global256/micro64 batch，对照实际v6双卡实现SHA8ac4d64368f337d485d3b45b326038a014f6d294754ac741d2d1093df11e54d9，完成双卡连续4步、四卡连续4步、双卡2步checkpoint后四卡冷恢复2步，共10诊断更新、0正式预算。模型/Adam/scheduler/全部CPU和规范化CUDA RNG/loss逐值一致，serial_errors/resume_errors为空；首micro校准仍2168952 Philox increment，派发不取参考RNG，梯度仍按0→1→2→3原顺序相加、只有主卡Adam。四卡副本在每次完成后保存同一逻辑RNG终态，下一micro仍独立按原逻辑偏移设置。原双卡热0.492/0.477/0.472s，四卡0.413/0.346/0.353s，恢复热0.353s；这是diagnostic，尚未构成正式提速或质量结论。

新并行实现SHA0c4faa96d3a210a2372d83c462a35c04be5ee6c7fd73ec354658127f5f7ef62b，报告`out/pfnn_mixed_20261015/parallel_four_optimizer_v1-report.json` SHA6dd870ff69ef9e67f1cd372b052f8510d889cf61220717ff2750aaf1b5a5fdfd。迁移仅允许Root从2→4卡且memory_fraction0.1不变，必须绑定旧/新实现、独立RNG和双卡checkpoint恢复proof，其余数据/样本链/optimizer/预算契约保持；29项迁移拒绝测试及6项原训练/cache/barrier测试通过。v8后续Root/codec/Pose结果均在`full_training_v8`，三阶段各50000/global256/micro64/50-25-25/Adam5e-5，总150000/38400000不变。Codec沿用v7单卡coalesced+fraction0.1，Pose沿用单卡cache+coalesced frozen codec/default memory；5份已有执行proof绑定保存，新阶段每次扫描全部GPU。等migration.json及真实root/progress出现后再确认32000后连续计数与生产吞吐。

2026-10-16 06:57 Asia/Singapore：正式监督v7 PID3549700、Root v6 PID3413446均健康，Root仍在GPU4/5连续训练，现18910/50000更新（37.82%）、4840960曝光、0.555s/update；无正式failure，codec/Pose仍按原依赖等待。全部8卡再次检查UUID/显存/利用率/计算进程，GPU0/7空闲，其他4张为他人任务。本轮GPU0用于尚未验证的全模型CUDA图接入，全部诊断结束释放，未暂停/重启训练、未改optimizer、预算或旧服务器。

新增诊断`probe_full_corpus_inference_execution.py`复用原TerrainExecution与原3e-5m FK/活跃token完全相等门，加载已绑定全部组件的mechanics_world模型。`full_inference_execution_v1/v2`在motion索引处capture失败，v2的shape-only常量复制猜想未解决，`full_corpus_static_trace.py`仅保留为失败诊断候选，**不接入actor**。进一步核源码发现训练快照缺少本地viewer已存在的CUDA兼容改动：motion_features.py的device索引、torch_utils.py的固定索引FK、rotations.py的固定shape选择，以及官方Kimodo skeleton/kinematics.py相同的device/FK改动。v4补前三份后原始eager输出逐值一致，预测图已过首组门，但官方skin捕获仍失败；v5补第四份后完整通过。所有失败快照/日志保留；这是补齐独立推理快照的既有依赖，不是远端PyTorch2.5.1无法运行CUDA图，也没有修改任何训练快照。

`full_inference_execution_v3`保存原始5方向/idle参考，SHA e9e0f10706389a6f6309fe419e3972c65d49e13838b81575d9882206d1bd7b92。v5的eager feature及binary rig与之逐值一致；编译后5组活跃token全部一致，最大完整FK差9.08849e-7m，发布帧最大FK差6.16015e-7m，仍用原3e-5m门。**compiled/eager浮点输出不是bitwise**；官方18056顶点rig/LBS对同一发布motion误差0m。全模型tensor未变、0optimizer更新。远端10次热样本的完整8帧包中位126.10ms→22.47ms（约5.61倍），P95 127.81→22.87ms，初始化24.45s；这不是本机帧率。报告`out/pfnn_mixed_20261015/full_inference_execution_v5-report.json` SHA83c987195c7b27441d7ea29e2194b598691683e239f771594da8a13e8cea4853。

独立验收服务器现提供显式`--execution verified-cuda`（默认仍eager），每次加载实际候选后自行编译并过原token/FK/几何门。`full_acceptance_v3`在不可变`code_full_acceptance_v3`实际运行该选项，复用v2绑定artifact，无重复打包/训练：真实HTTP/WebSocket的3包24帧含stop预取和turn，与相同compiled引擎的serial binary/RNG逐值一致，错误profile/提前ack仍拒绝，模型未变。临时服务及PID4000225已退出，GPU0释放；报告`out/pfnn_mixed_20261015/full_acceptance_v3-report.json`。不能把compiled-vs-serial逐值一致误称compiled-vs-eager逐值一致。最终全量模型、本机设备、浏览器渲染/键盘WASD/30fps和完整闭环质量仍未验证；已完成诊断不重跑，正式模型结束后再按原验收要求执行。

2026-10-16 06:06 Asia/Singapore：v7监督PID3549700、v6 Root PID3413446健康，实际cmdline/输出路径核验不变；Root已13430/50000更新（26.86%）、3438080曝光，均速0.551s/update，约为最初7.431s的13.48倍。无Root/监督failure；codec/Pose仍待Root最终完成。按当前吞吐Root剩余约5.6小时，不含后续阶段及质量验收。再次检查全部8卡UUID/显存/利用率/计算进程；Root继续独占GPU4/5，GPU1/2/3/6为其他任务，GPU0/7空闲。新接口诊断实际使用GPU0，完成后已释放；没有暂停或重启正式训练。

新增独立`motionbricks/data/full_corpus_acceptance.py`、`scripts/package_full_corpus_acceptance.py`、`scripts/soma_full_corpus_acceptance_server.py`和`web/full_corpus_lab`。新schema为`motionbricks.full_corpus_acceptance.v1`，只允许显式loopback验收，不授权最终交付；正式候选必须绑定三阶段各50000更新、保护hash、完整Root/codec/Pose依赖链、全模型/最终codec/原bundle统计及场景SHA。场景只带静态terrain及4帧有限真实history，拒绝未来数组/越界路径/运行参数漂移；固定8tokens/4Posepass/4history/24playback/8packet/30Hz/0refinement。全18056顶点官方SOMA使用77骨rig传输。原offline receipt仍禁止交互，28942/28941/28936及旧web文件均保留。33项本地验收/离线loader测试、JS语法和Python编译已通过。

`full_acceptance_v2`在不可变`code_full_acceptance_v2`用修正后的`mechanics_world_v1/pose/weights.pt`完成真实HTTP/WebSocket集成检查：页面/JS/CSS/Three静态资源和config可读；3包共24真实生成帧（直行、stop预取、转向）与串行SomaLabSession的binary payload及CPU/CUDA RNG逐值一致，官方完整skin与rig重建最大误差0m。错误runtime profile和7/8帧提前ack均拒绝；全部模型tensor未变，0optimizer更新。临时loopback服务已关闭，PID3774179已退出。小报告`out/pfnn_mixed_20261015/full_acceptance_v2-report.json` SHA98e82527d45748a2c91b0f1cd69d5ef54c7858a0ae0a1cc92ff921d29f289374；大模型/动画均留远端。

`full_acceptance_v1`因训练快照缺失offline loader/committed backend依赖，在模型加载前失败；原快照/log/failure保留。v2补齐依赖及匹配viewer后先通过CPU导入再执行GPU检查，不是重跑已完成诊断。当前验收服务器仅eager；本次没有浏览器渲染、真实键盘WASD或30fps证据，更没有正式模型动作质量结论。后续须用最终三阶段候选重新绑定并执行完整路线/flat16秒/跨块/训推与本地交互验收，不能把此mechanics/API通过当作可试用交付。30分钟跟进继续，每轮仍检查所有GPU。

2026-10-16 05:17 Asia/Singapore：当前正式监督已为**full_training_v7 PID3549700**，不可变`code_full_workflow_v7/scripts/continue_pfnn_codec_pose_execution.py`；状态adopting_running_root。旧v6监督PID3413423仅停止自身，`full_training_v6/codec_pose_execution_handoff.json`绑定5份已通过的执行证明。**Root PID3413446未暂停、未重启**，仍在GPU4/5用原code_full_workflow_v6、原Adam/样本链，输出仍读`full_training_v6/root`；后续codec/Pose读`full_training_v7/codec`和`pose`。禁止恢复旧v1-v6监督。Root当前8070/50000实际更新、2065920曝光、均速0.548s/update，无failure；7000/8000定期checkpoint及验证均已产出，未重复结构审计。

本轮继续检查全部GPU及真实进程，空闲GPU3实际执行新增`pose_cold_bias_v1`，现已完成释放。使用固定5000步模型、3组真实64样本，每组cold miss及重新排序后的warm hit对照：token值、原生stride(8,1,512)、loss、全部Pose参数梯度、CPU/CUDA RNG全逐值一致，所有模型tensor未变、0optimizer更新。热算子条件下cold1.425..1.643s→0.123..0.139s；第一组含冷启动2.266s不作为稳态倍数，warm约0.09s两者一致。该证据针对冻结codec标签编码，不是最终模型动作质量。本机`out/pfnn_mixed_20261015/pose_cold_bias_v1-report.json`，SHA4a269f81ab5cacd74d8c104f8cfb31f2d3a238347d9a813c17347665a5b26748。

v7已把前轮codec memory compare及真实2步Adam fit证明落到未来codec命令：单卡`--coalesce-dilated-bias --cuda-memory-fraction 0.1`。Pose命令为`--cache-pose-tokens --coalesce-dilated-bias`（保留原默认memory policy）；只有显式memoization且codec为eval/frozen才允许Pose bias coalescing，错误训练态/可训练codec在准备输入前拒绝。23项本地guard/cache/迁移测试通过；Root现有训练模块和快照未变。Root到50000并验证completion/weights SHA后，v7才启动codec，再以最终冻结codec建立Pose标签；每阶段仍50000、256/64、50/25/25，codec/Pose的2次startup计入同一预算，总150000更新/38400000曝光不变。codec/Pose新政策尚未开始正式更新，真实吞吐到阶段开始后再核，不将cold-cache诊断倍数当总训练倍数。已完成对照不重复跑。

2026-10-16 04:59 Asia/Singapore：**正式双卡已无损接续并持续更新**。当前监督PID3413423、Root PID3413446，真实cwd为不可变`code_full_workflow_v6`，CUDA_VISIBLE_DEVICES=4,5、短TMPDIR均核；nvidia-smi确认同一Root进程实际占GPU4/5约2.85/2.51GB。旧v5 Root PID1721912已完整保存6000步/1536000曝光后按预期barrier退出，旧v5监督已停。`full_training_v6/migration.json`绑定handoff checkpoint SHA `2299196af04a56d25f98370686438afd5db027c3e634ad74db86f9f2781cb12e`、完整old/new source hashes、两项proof，no_optimizer_updates_lost=true。旧latest_before_parallel_handoff.pt/latest.tmp.pt/parallel_handoff.pt及主动failure全部保留，latest.pt恢复有效文件。不要重启旧v5或更早任务。

新Root已从6000连续推进至6160/50000步、1576960曝光，无failure，loss/gradient有限；160次正式新更新均速**0.603s/update**，相对v5完整均速2.579s约4.28倍，相对最初7.431s约12.32倍。按现速Root剩余约7.35小时，仅为当前吞吐估算，不含codec/Pose及质量验收。`root_parallel_devices=2`、`parallel_calibration_microbatches=1`明确记录；校准不更新optimizer。以v6/root/progress.json为唯一实时Root计数，后续codec/Pose也读v6；全局batch256/micro64/样本链/阶段各50000/总150000预算不变。

补充`codec_memory_fit_v1`在GPU7完成两个真实256/micro64 Adam诊断更新，实际state步1/2，峰reserved约7.58GB、初始模型恢复，0正式预算；完整报告证明0.1限额可以容纳这两步的Adam。通用诊断state writer原误标0，被`state_count_correction.json`保留原记录并改为2，complete.json原计数始终正确；仅修报告元数据，没有重跑。**v6正式codec仍使用原显存policy**，本证据供后续有版本记录的配置接入。8项执行/恢复小报告已取回`execution_scaling_reports_v1.tar.gz`并核SHA d4fa06af02bfcd40c31f727aa4b98a075b6eb4a624ceff09576e793dbc449827，大模型/梯度留远端。GPU3/7诊断结束释放，GPU0也已由旧Root正常释放；勿为占满GPU重复已完成验证。

2026-10-16 04:47 Asia/Singapore：本轮发现GPU3/4/5/7确实空闲（17..19MiB、0%、无计算进程）后已直接使用全部4卡。3/7完成默认/限额显存执行对照，4/5完成双卡微批与真实optimizer接续验证；正式v5 Root PID1721912始终在GPU0健康推进，没有重跑数据或正式预算。当前已预加载**v6监督PID3413423、新Root child PID3413446、GPU4/5**，不可变`code_full_workflow_v6`。新Root等待完整handoff，尚未执行正式新更新；旧v5 Root最新5750步/1472000曝光，继续至6000步边界。旧v5监督PID1717266在新模型ready后仅停止自身，`full_training_v5/parallel_acceleration_handoff.json`保留。禁止恢复旧v1-v5监督。

第二个实证瓶颈：`execution_memory_default_v1`与`execution_memory_capped_v1`使用同一5000步不可变硬链接`execution_inputs_v1/root.pt`（SHA da49e84132e3f8ac8105b28ba7c698b839e2413b5b556de657ec765d96991233）、两组真实batch256/micro64。默认allocator热前后向+clip2.542s，峰allocated36.57GB/reserved71.40GB；设置per-process fraction0.1后0.717s，峰allocated1.69GB/reserved1.83GB。输出/loss/CPU-CUDA RNG/样本相同，4个非膨胀卷积权重梯度有最多1.106e-9的差异，`execution_memory_compare_v1`通过**原有**loss/output atol1e-6 rtol1e-5、gradient atol5e-6 rtol1e-3门；不是跨配置bitwise相等，不得谎称差0或放宽门。0optimizer更新。显存预算约束了算法工作区选择，不能把它当成减少模型或batch。

`execution_parallel_v1`先证明两卡按原0→1→2→3顺序合并梯度可逐值一致，但它从串行参考取得RNG起点，仅诊断，不可部署。后续`parallel_root_training.py`仅Root、严格4x64微批：用当前首micro做一次不更新参数的RNG校准，保存/恢复原CPU/CUDA状态，实测每micro Philox increment2168952；各工作卡独立设置起点并逐次核偏移，按原顺序归并梯度，Adam只在主卡执行，logical CUDA RNG最终保持原流。每步先同步trainable参数，codec的EMA不参与并行。

`parallel_optimizer_v1`在GPU4/5用正式5000步模型及随后4个真实global256 batch：同一0.1显存配置下，串行4步 vs 独立RNG双卡4步 vs 保存2步后冷恢复剩余2步，model/Adam/scheduler/CPU-CUDA RNG/每micro loss全部逐值一致；无参考RNG调度，10个明确诊断更新，不计正式预算。热串行0.771..0.820s，双卡0.473..0.503s（测试含optimizer）；不能直接当生产速度或最终质量。实现SHA `8ac4d64368f337d485d3b45b326038a014f6d294754ac741d2d1093df11e54d9`，正式快照逐字相同。21项本地迁移/预算/缓存/barrier guard测试通过。

v6 Root已在GPU4/5各加载约1.29GB模型并完成原缓存SHA核验，`full_training_v6/root-await-resume.json`确认仅等待checkpoint；未重复优化。旧Root在5730步通过短暂停顿原子布置`full_training_v5/root/parallel_handoff_barrier.json`，保留`latest_before_parallel_handoff.pt`，下一完整torch.save后预期IsADirectoryError停止。v6核step/exposures/Adam/scheduler/sample digest后保存`parallel_handoff.pt`、恢复有效latest.pt并写migration，再同预算续训到50000。旧v5的这次预期failure也不得作为随机故障重启。新代码执行策略只有memory_fraction0.1及Root双卡，数据/loss/globalbatch256/micro64/配比/预算不变；后续codec仍单卡原policy+coalesced bias，Pose单卡已验证缓存。当前实际新进度仍须等handoff后读v6/root/progress.json。

额外`codec_memory_default_v1`/`codec_memory_capped_v1`已完成两真实batch0更新对照：1.428s→1.190s，码本/输出/loss/RNG相等，仅2个梯度差<=3.493e-10，原门通过。但0.1限额峰allocated约7.46GB，尚未验证完整Adam步骤的显存余量，所以**未给正式codec加此限额**；相关报告保留，不能直接套Root双卡。所有新任务已实查空闲并传短TMPDIR，GPU3/7诊断结束后释放，旧失败方案不复用。

2026-10-16 03:15 Asia/Singapore：v5监督PID1717266与Root PID1721912真实cmdline匹配，Root3570/50000实际更新、913920曝光、均速2.643s/update，约为旧7.431s的2.81倍，无failure。已检查全部8张GPU的UUID/显存/利用率/计算进程，均有任务占用，未启动重复或抢占工作。codec/Pose尚未进入正式阶段。

加速后的首个定期checkpoint（3000步）已CPU mmap重新打开核验：183组Adam step全部3000、scheduler.last_epoch3000、曝光768000，coalesced_dilated_bias=true、mechanics_only=false；样本digest和迁移绑定存在，读取前后文件inode/长度/mtime一致，大小1090264240bytes。仅核保存结构/计数，未重做全tensor hash或恢复测试；报告`out/pfnn_mixed_20261015/root_3000_checkpoint_review_v1.json`。固定每domain128条验证loss为PFNN0.04279772/PARC0.07331621/BONES0.00912854；相较2000步的0.04083003/0.07275240/0.00933754，前两者略升、BONES略降，三者仍低于基线。保留真实波动，不据此改loss或声称闭环质量通过。以后健康定期checkpoint不重复这次迁移后结构核验。

2026-10-16 02:50 Asia/Singapore：用户要求**每30分钟检查全部GPU，发现空闲即直接使用，无需再次确认**。每次查index/UUID、显存、利用率及真实计算进程，不再只查GPU0或固定等GPU6；空闲资源用于已授权的必要并行训练/加速等价验证/独立评估，优先缩短交付时间，保持原全局batch、样本顺序、预算和数值门。资源分配前再核空闲并记录，勿动他人进程、不重复健康或已完成任务。自动化已保存此规则并保持30分钟频率。本次8卡都有计算进程，1/3/4/5虽瞬时0%但仍占约53GB，不能当空闲。Root最新2990/50000步、765440曝光、均速2.685s/update，健康继续。

2026-10-16 02:08 Asia/Singapore：**正式加速已落地并持续更新**。v5监督PID1717266、新Root PID1721912真实cmdline核验通过；原Root PID2942425已在完整2000步/512000曝光退出，v5迁移检查通过后从同一checkpoint继续，现2050/50000步、524800曝光，无failure。2000步checkpoint SHA `e13150225b8695882bf8f3edd2d5d6778b31a0763de36897880eb48b37d96374`，migration明确no_optimizer_updates_lost=true；所有旧检查点与主动IsADirectoryError证据保留。最新contract已显式coalesced_dilated_bias=true、Pose memoization=false（Root阶段）。

新生产前50次实际更新均速2.699s/update，较旧7.431s/update约**2.75倍**；2010/2020/2030/2040/2050计数持续推进，loss/gradient均有限。单批诊断约7倍不冒充生产7倍，当前全流程还有其他开销。按现速Root剩余约36小时只是吞吐估算，不含后续codec/Pose及质量验收；继续观测。新进程启动前重新核全缓存SHA，读约66.8GB后才开始更新，不是重建缓存。后续真实Root进度必须读`full_training_v5/root/progress.json`，监督state里的root_progress保留交接前2000步记录，不能当实时训练计数。每10分钟跟进已更新v5身份、已停旧任务和有效加速证据。

2026-10-16 02:02 Asia/Singapore：用户再次要求实际加速后，已完成有效Root/codec优化及同optimizer接续证据，正在正式2000步检查点交接。**新监督full_training_v5 PID1717266**，不可变`code_full_workflow_v5/scripts/run_pfnn_bias_workflow.py`。旧v4监督PID3105618仅其自身已停止，`full_training_v4/bias_acceleration_handoff.json`保留交接；禁止恢复v4。原Root PID2942425继续原代码至下一个完整checkpoint，之后新Root/codec/Pose输出均在`full_training_v5`，总150000更新/38400000曝光不变。

实证原因：PyTorch2.5.1实际选择`_ConvBackend.SlowDilated2d`。官方NaiveDilatedConvolution.cu在每个样本内逐通道执行bias填充，再逐样本GEMM，64x512会产生大量细碎调用。显式cuDNN/channels_last虽快，但`owned_gpu_probe_v1`部分输出/梯度门失败，保留且不采用。新`motionbricks/data/dilated_bias_convolution.py`一次广播填好完整output的bias，仍按原每样本顺序addmm全部kernel taps，反向直接调用原生`aten.convolution_backward`，没有中心卷积近似或更改任何数值门。

`owned_gpu_probe_v2`四种实际形状合成FP32输入：output/input-gradient/weight-gradient/bias-gradient均逐值差0，三次重复也差0；完整前后向从约238..262ms降至20..23ms，0optimizer更新。`owned_full_bias_probe_v1`加载正式1000步Root检查点，真实64样本50/25/25混合batch：完整Root前后向热重复1.356s→0.187s，codec2.773s→0.299s；输出/loss/全部参数梯度/码本buffers/CPU-CUDA RNG全部逐值一致，原权重恢复，0optimizer更新。该证据是数值与性能，不是最终动作质量。

`owned_bias_resume_probe_v1`使用同一正式checkpoint、真实batch256/micro64和对应的下一批样本：原实现连续2步，对照第1步完整落盘后恢复并用加速实现执行第2步；模型/Adam/scheduler/RNG与每microbatch loss逐值一致，实际完整更新5.651s→0.808s。共3个明确诊断更新，不计入正式训练预算。三个小报告已下载本机`out/pfnn_mixed_20261015/owned_*probe_v*-report.json`。完整模型与optimizer证据hash分别e437a315...、62ed7ddd...，实现SHA `5759d40f9007b0b0f9fe329e0b33eb2d35e8a118fbf08e1a8edbbfb0277f8d15`绑定进迁移记录。

为避免继续等他人GPU6，本轮只短暂停住自己GPU0的Root进程、保留所有内存状态，在余量显存做限时探测，然后SIGCONT同一PID；四次分别约7.53/6.38/30.93/29.42秒。独立watchdog最多55秒恢复，探测有40/50秒timeout及5%/10%显存上限；远端CPU进程夹具验证探测异常、超时、控制进程被杀均恢复。其他用户任务未动。原CPU等待PID1123936/1148616在确认零CUDA/零optimizer后停止并写rescheduled.json，禁止重启旧队列。

当前runner没有安全停点API，已在本机和远端Linux用真实torch.save/Adam验证**checkpoint rename barrier**：仅将本任务旧latest.pt保存在`latest_before_bias_handoff.pt`，同路径创建空目录；原训练先完整写latest.tmp.pt，再在rename时抛IsADirectoryError，从而不会进入下一optimizer更新。1990步时已通过短时SIGSTOP原子布置，`full_training_v3/root/bias_handoff_barrier.json`记录预期2000步。这一预期failure必须保留并标注为主动交接，不是随机训练故障。新v5监督等待原Root正常异常退出，再核完整tmp checkpoint/Adam/scheduler/样本digest及最后progress相等，保存硬链接handoff，恢复latest.pt为有效文件，然后从同一更新继续。**严禁从旧1000步回退，也不可直接kill原Root。**

`--coalesce-dilated-bias`仅显式启用Root/codec；Pose沿用已验证token缓存。跨实现resume需要checkpoint SHA、旧新完整source hashes、原data/optimizer/预算契约逐项相等以及两份数值/恢复证明；不允许普通resume默默忽略差异。21项本地guard/layout/token-cache/迁移拒绝与rename barrier测试通过；一次初测因未设置kimodo PYTHONPATH导入失败，补齐既有vendor路径后通过，无模型失败。下一步核v5实际child、migration.json及root/progress，确认正式2000后更新与持续吞吐后再报告生产提速。最终全模型WASD验收仍未完成。

2026-10-16 00:27 Asia/Singapore：`dilated_layout_v2`已在新鲜空闲GPU6完成，原PID1068403正常结束。两组独立真实64样本的Root/codec各2次baseline/vertical执行，输出、loss、全部参数梯度、quantizer buffers、CPU/CUDA RNG均逐值一致；原权重恢复，0optimizer更新。**单例轴交换没有可靠提速，不部署**：Root热重复baseline1.839/1.759s，vertical1.933/1.757s；codec2.609/2.612s对2.661/2.608s。本机完整报告`out/pfnn_mixed_20261015/dilated_layout_v2-report.json`。

进一步核远端PyTorch2.5.1+cu124/cuDNN90100，并阅读同tag官方`aten/src/ATen/native/Convolution.cpp`的`ConvParams::use_cudnn`：连续内存格式在`deterministic && is_dilated()`时直接return false，而channels_last绕过该分支。这与膨胀卷积集中慢、benchmark/轴交换均无收益一致；仍需实际`_select_conv_backend`诊断确认运行时分派。来源https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/aten/src/ATen/native/Convolution.cpp 。未关闭deterministic、TF32仍false，未改任何生产模型。

现有两个顺序CPU队列（不是重复正式训练）：`cudnn_dispatch_v1` PID1123936、代码`code_cudnn_dispatch_v1/check_cudnn_dilated_backend.py`，在GPU6<=500MiB/<=5%后检查4个真实网络形状上的合成FP32张量，记录native后端，比较显式cuDNN(deterministic=true,allow_tf32=false)的输出/输入及参数梯度、3次重复确定性和耗时。SHA `93034a0746eccf2266a5a98e101fea48f8c7e485cfd78d762d1c95d843410d41`；00:27仍waiting_for_idle_gpu。它不是完整模型/真实batch准入。

随后`dilated_layout_v3` PID1148616、不可变`code_dilated_layout_v3`（依赖原v4），使用`--layout channels_last --after-probe cudnn_dispatch_v1`等待前项complete，再重新检查GPU6空闲。若前项异常failure则停止待人工诊断，不自动重试。候选将完整2D input/weight转channels_last以走官方支持的分支，再把输出恢复原NCT连续布局；同一真实两batch/原loss梯度门/码本及RNG逐值门保持。当前waiting_for_preceding_probe，0optimizer更新、production_approved=false。脚本SHA `2cfc29891f6c7a90bd231971e547f38c3a40e9c772f8c5e1b2f9948a4c5a9a36`。扩展后9项本机CPU float64前向/梯度/layout/guard/异常恢复测试全部通过，不能代替CUDA验证。所有新进程继承短TMPDIR，旧结果保留。

Root最新1230/50000实际更新、314880曝光，约7.372s/update；仍原PID2942425/v3训练实现和v4监督。下一轮查上述两个队列的state/progress/complete/failure及真实cmdline，健康不重复启动。若候选有真实加速且过数值门，再做正式checkpoint/连续恢复证据和无更新丢失的交接；源Root/Pose/码本/最终WASD验收要求不变。

2026-10-16 00:18 Asia/Singapore：原`layer_profile_v1`等待进程已趁GPU6空闲自动完成，未启动重复计时。真实64样本Root前向1421.35ms/反向572.59ms，6个512通道dilation3/9/27卷积（输入长度16/32）前向合计1348.32ms，约占94.86%；单层211..259ms，Transformer层约1ms。Codec前向2877.17ms/反向285.80ms，12个同类卷积（长度8/16）合计2798.46ms，约97.26%。报告证明瓶颈集中于短序列膨胀卷积执行路径；不是数据读取慢，也不证明任何候选已加速。0optimizer更新、原权重完整恢复；本机`out/pfnn_mixed_20261015/layer_profile_v1-report.json`，SHA `fe23f8c8119f59de257e65efb57aa46e07b8190439cc6d0996e028a14d0a54ba`。

据此新增诊断`scripts/check_dilated_conv_layout.py`：不删除side taps或裁短序列，将原Conv1d的完整权重/input升为Conv2d，交换单例空间轴，保持padding/dilation/FP32/确定性配置。仅诊断上下文临时替换forward，退出恢复原方法；生产代码无变化。5项CPU float64测试覆盖长度8/16/32与dilation3/9/27的前向及输入/权重/bias梯度、非连续input、适用范围和异常恢复，均通过；不是A100数值/速度证明。

GPU6随后被他人占用。未运行的`code_dilated_layout_v1`小源码保留；加上空闲等待后的不可变`code_dilated_layout_v2`已启动CPU等待PID1068403，receipt `dilated_layout_v2.process.json`，输出/状态`dilated_layout_v2`，依赖`code_full_workflow_v4`，短TMPDIR已显式传入。脚本SHA `53eaf88b10d1367e2f17eef9658ea5726cb259ccc381846668fe1689977c3210`与本机相同。仅GPU6新鲜<=500MiB且<=5%才开始，计划两组独立真实64样本50/25/25混合batch的Root/codec baseline对照：原loss/输出/全部梯度门、码本buffer及CPU/CUDA RNG逐值比较，不放宽门，0optimizer更新。00:18仍waiting_for_idle_gpu，无CUDA结论；健康队列不重复启动。旧center-only失败方案仍拒用。

Root当时1160/50000实际更新、296960曝光、约7.375s/update，仍原PID2942425/v3代码、v4监督，没有failure或重启。若新候选通过，仍需当前正式checkpoint上的验证与同optimizer/RNG/样本链恢复证据；现有Root runner没有运行时请求安全停点的signal/file接口，不能简单kill后回退最近1000步checkpoint。下一次先读`dilated_layout_v2`真实state/progress/complete/failure及进程，再决定后续，不得把CPU测试或诊断通过直接当作生产准入。

2026-10-16 00:06 Asia/Singapore：首个正式1000步验证与检查点已完成。实际Root PID2942425、v4监督PID3105618和诊断等待PID3187284的cmdline仍匹配；Root最新1060/50000实际更新、271360曝光，约7.383s/update，无failure，原进程/optimizer/样本链未重启。按当前均速，仅Root剩余48940步约100.4小时；这是未加速情况下的估算，不含后续codec/Pose及最终验收，不能据此承诺交付时间。

CPU只读重新打开`full_training_v3/root/latest.pt`（1090263600bytes，mmap、CUDA不可见、短TMPDIR）：schema正确、mechanics_only=false，actual_updates1000、actual_exposures256000；183组Adam参数状态的step全部1000，scheduler.last_epoch1000，Torch/CUDA/NumPy/Python RNG与样本digest均存在，读取前后文件inode/长度/mtime一致。该检查核对已保存计数和结构，未重新计算全部模型tensor hash，也未恢复或替换正在运行的模型。小型记录在本机`out/pfnn_mixed_20261015/root_1000_checkpoint_review_v1.json`；大检查点留在远端。

相同每domain128条validation样本的native loss：PFNN基线0.07547852→1000步0.04126608，PARC0.08779541→0.07116496，BONES0.01557294→0.01008881。三个域均改善，但这些目标值不等于闭环动作质量、楼梯通过或WASD可交付。

00:04..00:06实查GPU0仅有本Root训练进程，nvidia-smi报告其约68924MiB（包括allocator保留/工作区，不能据此断言活跃tensor或泄漏）；其他7张卡均有他人计算进程，2/6各约6980MiB且繁忙。因此没有启动抢占式诊断。`layer_profile_v1`仍是原健康CPU等待进程，尚无分层计时结果。代码审查确认Root更新shared scene encoder，codec随后冻结并使用其最终权重，现有阶段不能直接重叠而声称等价。后续先读分层结果再选Root/codec优化；已验证Pose缓存仍由v4监督按原阶段顺序部署。

2026-10-15 22:38 Asia/Singapore最新跟进：监督PID3105618和Root PID2942425真实cmdline匹配、无failure；Root340/50000实际更新、87040曝光，约7.45s/update。2、6号卡已有其他计算进程（约6.9GB且繁忙），没有在其上启动新GPU工作，也没有动原Root。

新的只读分层计时工具`scripts/profile_full_training_layers.py`已写好并py_compile通过：只保存Conv1d/Conv2d/Linear/MHA/Transformer层形状、CPU wall时间及CUDA events，不保留大输入输出，不启用此前开销过大的全算子trace；计划用真实64样本混合batch分别测Root/codec，0optimizer更新、结束恢复全部初始模型tensor。不可变脚本快照`code_layer_profile_v1`，依赖`code_full_workflow_v4`。CPU等待进程PID3187284，receipt `layer_profile_v1.process.json`，状态`layer_profile_v1/state.json`当前`waiting_for_idle_gpu`，只在2或6号卡新鲜满足<=500MiB且<=5%时自动启动一次；此刻没有GPU计时结果。健康等待不重复启动。下次查该进程/状态/日志及complete/failure，完成后检查最耗时的真实层，再决定优化方向。

本轮CPU读取实际bundle配置：Root max_tokens16、当前运行8tokens，Root与codec width512/depth4/dilation_growth_rate3、norm None。该配置本身不证明可直接裁剪padding或采用已有失败的center-only卷积；Transformer dropout布局、卷积mask/边界、全参数梯度和断点恢复仍需实测等价。没有修改任何生产模型计算。

2026-10-15 22:22 Asia/Singapore：用户明确要求加速。完整PFNN缓存已完成64780primitive，与PARC/BONES332528共397308增强primitive；`full_training_v3/corpus_audit.json`的14个完整数组SHA/三domain各split/家族检查通过。**Root正式训练已开始**，真实PID2942425，输出仍为`full_training_v3/root`；最新210/50000实际更新、53760曝光，loss0.03613、gradient norm0.22594，平均约7.54s/update。2次startup更新已在预算中，断点续到同一50000上限；没有重开或丢弃更新。

新增有效加速：`NativeTrainingStage(..., cache_pose_tokens=True)`仅允许Pose阶段，在codec保持eval且全部参数冻结时，按不可变cache primitive的全局sample_id缓存原生32帧对应8-token标签。所有增广已离线固化；命中仍重新准备相同history/Root/地形条件并执行原Pose目标。任一ID未缓存或batch尺寸无原生layout记录时仍整microbatch调用原encoder，不作局部batch近似。缓存不进checkpoint、不进入actor，重启后正常冷启动。`train_unified_full_corpus.py --cache-pose-tokens`显式optin并记录contract与hit/miss；Root/codec训练默认不启用，所有loss/batch/样本配额/更新预算未改。

`pose_token_speed_v3`真实四次重排/重组64样本batch验证通过（256不同真实cache样本，使用mechanics_world_v1最终模型）：token、loss、每个参数梯度及CPU/CUDA RNG**逐值一致，差0**；所有模型tensor未变，0optimizer更新。热缓存forward+backward从1.278..1.359s降到0.0821..0.0849s，15.25..16.35倍。**这是已命中样本的Pose microbatch速度，不是全训练倍数或动作质量。** 冷缓存会预热；后续正式codec训练后再建立标签，不能使用旧codec标签。

v1/v2失败完整保留：token数值差0但首次loss相差0.018426，因为原生token stride为(8,1,512)，普通stack产生另一内存布局，原生`randint_like`沿存储顺序分配随机扰动；最终RNG虽然相同，逻辑位置不同。v3通过`empty_strided`恢复原布局，未更改掩码算法/损失/随机数规则。3本地测试覆盖重排、不同microbatch尺寸、训练态/可训练codec拒用以及整数sample_id契约，均通过。

`pose_token_resume_v1`另做8个明确mechanics优化更新（连续4，对照2+2；不计入正式预算），模型、Adam、scheduler、RNG、样本digest、更新与曝光计数全部逐值相等，证明缓存未持久化时冷恢复仍等价；不是质量训练证据。新快照`code_full_workflow_v4`中训练模块/runner文件SHA与通过该测试的`code_pose_token_speed_v3`逐一相同。

当前监督为`full_training_v4` PID3105618，脚本`continue_pfnn_pose_cached_workflow.py`，已通过`--validate-only`并实际启动；状态`adopting_running_root`引用原PID2942425及原Root目录。旧v3监督PID1740827仅其自身已停止，`full_training_v3/acceleration_handoff.json`保留完整交接，**原Root子进程和DataLoader没有停止/重启**，190步后已观察到210步。不要恢复旧监督，不要从2步旧checkpoint重开Root。Root完成后新监督从其最终权重训练codec到`full_training_v4/codec`，再训练带缓存Pose到`full_training_v4/pose`，各50000实际更新，总预算150000/38400000曝光不变。未来每次新GPU任务仍实查空闲，继续短TMPDIR。

本轮其他性能证据：缓存慢的可见问题是逐窗口/四方向串行计算、重复加载同一动作，CPU约116%而GPU约22%，rchar约39.7GB但实际磁盘read约0.28GB（多数页缓存）；完成前未废弃重做已有89%结果。`runtime_profile_v2`微批分段计时显示输入准备仅约0.022..0.037s，主要耗时在网络forward；v1完整算子trace开销过大已单独停止并保留。`cudnn_review_v1`的benchmark选项无可靠大幅收益；禁用cuDNN的native路径在codec6个参数梯度门失败且更慢，未采用。此前center-only短卷积同样仍拒用。**当前Root和codec仍慢，未宣称它们已提速**；可继续对其执行做有证据的优化，不能靠减预算/放宽数值门/抢占他人GPU。

全部14份小JSON证据已取回`out/pfnn_mixed_20261015/acceleration_reports_v1/`，archive SHA `ee6331cd30b8a6219c57a270c71c417593804ed03142a395451749b2f16283a5`已核。模型/动画仍在远端，没有公网大传输。旧28942/28941/28936保持，未变actor推理或启动新服务。

下次先查新监督PID3105618、原Root PID2942425真实cmdline；Root进度读`full_training_v3/root`，后续codec/Pose读`full_training_v4`。健康训练不重复启动；若优化需要更换正在训练的实现，必须保留真实checkpoint/optimizer/样本位置和预算，不可丢弃未保存更新。数据准备/缓存/完整审计均完成不重跑。源history训练仍不证明恢复，最终仍需独立完整路线/flat起停转/训推与预取一致性/本地官方SOMA实际WASD30fps验收后才交付停跟进。

## Previous Work: Corrected PFNN Preparation Complete; Full Cache Building

2026-10-15 21:00 Asia/Singapore：`dataset_world_v2`已完成全部80源、233010真实30Hz帧，原PID1582020正常退出，**禁止重启已完成准备/retarget**。完成清单SHA256为`f8576d15230d3e493df9902d6a5fe23045263d90d5d338e23f0217cf0058d8bb`，7125地形配对、16195准入pair-windows；拒绝窗口及原始失败材料均保留，没有放宽门或修正动作。

新增只读最终清单复核`preparation_complete_review_v1.json`已完成并下载本机`out/pfnn_mixed_20261015/preparation_complete_review_v1-report.json`：完整manifest文件hash与complete匹配，v2schema/world_contract、80源/帧数/窗口总数、34录制家族split一致，所有已记录准入项满足原几何/stance门；全80源最大记录Root误差1.669e-6m、FK2.162e-6m。此复核读取原筛选结果，**不冒充独立重新计算skin或完整缓存数组SHA审计**。零fitting_failed，20773被拒窗口保留。

PFNN最终划分：train12058pair-windows/10097去重source窗口（28家族、62clips），validation3533/2872（4家族、12clips），test604/544（2家族、6clips）。镜像/重叠窗口仍相关，不算独立录制。构造环境类别flat9955、jumpy2182、rocky3859、beam199准入窗；这些是static patch/RBF配对，不是实测capture terrain。WalkingUpSteps01及mirror保持test，不为了扩大训练移动split；多条后续上下楼源已进入train。不能用准入源通过声称actor会过楼梯。

健康监督`full_training_v3` PID1740827已自动切到`cache_pfnn`，GPU0启动child PID2714031；真实cmdline为`cache_full_terrain_primitives --pfnn dataset_world_v2 --output cache_pfnn_world_v2`，GPU启动前18MiB/0%，子进程实际TMPDIR为`${PRIVATE_STORAGE}/mbtmp`。21:00缓存已有33692增强primitive/8423base窗口，预计目标64780（16195x4），仍在构建，**正式optimizer更新0**。勿因旧state中的waiting_for历史字段误判仍在数据准备；以stage/status/child实际进程为准。

下一步：先查`full_training_v3/state.json`、cache progress/complete及真实child，健康不重启。缓存完成后既有监督自动进行全数组hash/全部三domain划分与速度覆盖审计，再启动原预算Root→codec→Pose各50000实际更新/batch256/micro64/50-25-25。每阶段先2次计入预算的真实更新，再同optimizer继续；真正进度以stage progress/stopped/complete为准。短TMPDIR、所有旧失败与已认可28942/28941/28936保持。完整模型还需独立路线/flat/预取一致性/真实WASD30fps验收。

## Previous Work: Full-Model Committed Playback Parity Verified; Preparation Healthy

2026-10-15 17:06 Asia/Singapore：`dataset_world_v2` PID1582020与`full_training_v3` PID1740827真实cmdline仍匹配；监督实际TMPDIR仍为短目录`${PRIVATE_STORAGE}/mbtmp`。准备42/80段、4293地形配对、11835准入窗，持续产出；监督正常等待，无failure/complete/阶段progress，**正式optimizer更新仍0**。训练/数据任务和已认可服务均未重启。

本轮补齐全量schema的**串行离线→WASD预取后端一致性**诊断：新增`scripts/check_full_corpus_committed_playback.py`，使用已完成的`mechanics_world_v1/pose/weights.pt`（e55c73fd...）、严格full_corpus loader、完整Root/Pose/codec/adapter与最终诊断码本；未训练、未开server。GPU2实查空闲后运行`code_full_committed_probe_v3/full_committed_probe_v3`，PID1767020已完成退出、GPU2恢复空闲。v1因隔离快照缺full_corpus_inference、v2缺unified_committed_playback在导入时失败；原日志/快照保留，v3补齐两份已有模块并先CPU导入检查，再运行真实模型。未改训练快照或播放实现。

上楼/下楼/flat三个真实初态各10包，脚本包含持续前进、停止、90度转向和加速；总30包/240新生成帧，含15次新神经生成和15次既有buffer取样，未来history全部来自已发布模型输出。逐次匹配Torch CPU/CUDA RNG后，`UnifiedTerrainBackend`串行与`CommittedUnifiedBackend.prepare_next_plan/commit_prepared_plan`的8帧动作、4帧重编码history、剩余buffer差全部0，命令及生成时机相同。每次prepare前后live history/buffer/已发布history/command均未变；27次在真实边界history末root添加1mm的错误提交全部拒绝（扰动仅在负例诊断，未播放或训练），恢复真实history后正常提交。完整模型tensor hash前后相同，0optimizer更新。

本机报告`out/pfnn_mixed_20261015/full_committed_probe_v3-report.json`已下载，脚本哈希与远端执行值一致。**这是eager后端接口/状态机一致性，不是最终训练模型质量、完整上下楼、异步浏览器命令年龄或WASD30fps证据。** 未做IK/投影/Root替换/输出平滑；正式模型完成后须用同脚本不带mechanics optin复测，并完成原完整路线/flat/渲染/实际键盘与性能验收。旧28942/28941/28936保持。

下次继续先核数据/监督真实状态，健康不重复启动，不重跑本次已完成诊断。等待修正80段→cache_pfnn_world_v2→完整hash/split/覆盖审计→原Root/codec/Pose各50000实际更新。系统根盘仍满，新任务必须显式短TMPDIR；旧v1数据和旧v1/v2监督仍禁止恢复。不要把诊断完成误报为正式训练开始或交付。

## Previous Work: Verified Worker IPC Repair; Same-Budget Workflow v3 Waiting

2026-10-15 16:52 Asia/Singapore：进一步实测发现，上轮tempfile创建成功**不足以证明多进程DataLoader可用**。系统根分区满后，Python默认temp目录回退到长路径`code_full_workflow_v2`；真实缓存4worker/batch256测试在torch tensor resource_sharer的UNIX socket bind处报`OSError: AF_UNIX path too long`，主进程无法收到批次。`storage_worker_probe_v1/failure.json`保留，核4个worker共同PPid、真实cmdline/cwd后只终止本次诊断父子进程，未动主数据任务。

短目录`TMPDIR=${PRIVATE_STORAGE}/mbtmp`（当前用户所有、0700）解决了该实测问题。`storage_worker_probe_v2`在相同代码cwd读取既有真实PARC/BONES缓存，4worker、batch256、2批512样本，全字段/样本ID与serial逐值相同，worker临时文件及/data5原子输出替换成功。该检查只验证CPU缓存/共享内存/临时路径，未用GPU、未测CUDA pinned memory、0optimizer更新，不是训练或动作质量通过。缓存每primitive150262bytes，当前7个数组共约49.97GB；/data5约4.79TB可用、/dev/shm约93.86GB可用，系统根分区仍0可用。之前较长的`pfnn_mixed_20261015/runtime_tmp`仅完成了单进程审计，**后续DataLoader必须使用已验证短目录**。

原`full_training_v2` PID1613357虽尚在等待数据，但其子进程会继承有故障的默认temp环境。先确认仍0正式更新、无child且无Root/codec/Pose目录，再记录`full_training_v2/environment_repair_stop.json`并终止该等待监督。不要恢复它。新不可变`code_full_workflow_v3`的motionbricks/scripts/out/tests/docs中875份Python源码与v2哈希逐一相同；仅启动环境增加短TMPDIR。首次快照哈希比较误包含旧目录中torch导入生成的临时Python文件，失败记录`snapshot_build_failure.json`保留；按实际源目录比较通过后才停止旧监督。没有修改模型、loss、batch、数据、数值门或预算。

当前耐久监督为`full_training_v3` PID1740827，process receipt是`full_training_v3.process.json`，实际temp环境和前后代码证明在`full_training_v3.environment.json`与新快照`runtime_environment.json`。真实cmdline/environ已核，旧PID已退出；新state为preparation/waiting_for_existing_preparation，无failure，**正式optimizer更新仍0**。继承的原launcher字段仍写`supersedes_zero_update_workflow=full_training_v1`，直接前驱v2以environment/stop证据为准。新监督继续自动等dataset_world_v2，生成尚不存在的cache_pfnn_world_v2，完整audit后执行原Root→codec→Pose各50000更新/batch256/micro64/50-25-25，总150000更新/38400000曝光不变。

`dataset_world_v2` PID1582020从未重启，最新39/80段、4078地形配对、11398准入窗，持续健康。修正PFNN世界坐标/场景接触契约、已完成37段审计及独立42条评估源保持。小型失败/通过/环境报告已下载`out/pfnn_mixed_20261015/`；定时跟进已更新到v3，28942/28941/28936不动。

下次先查PID1582020、PID1740827的真实身份与新state/progress/complete/failure，不重复健康任务和已完成诊断。系统根分区仍满，尚未清理，所有新任务显式传短TMPDIR，不能只依赖cwd fallback。训练真正开始后检查optimizer进度与实际阶段日志。旧v1 PFNN数据和旧v1/v2监督均不恢复；真正验证完整WASD交付后再停跟进。

## Previous Work: Partial Corpus Repair Audit Passed; Corrected Preparation Continues

2026-10-15 16:43 Asia/Singapore：已重新核实真实cmdline。`dataset_world_v2` PID1582020健康，37/80段、3878地形配对、10966准入pair-window；最新contact sidecar在16:43:22继续写出。`full_training_v2` PID1613357仍健康等待修正数据，无failure/complete、无stage progress，**正式optimizer更新仍为0**。没有重启主准备或监督，预算和不可变训练快照未变；旧v1仍禁止消费。

新增只读CPU `scripts/audit_pfnn_preparation_snapshot.py`，远端不可变`code_preparation_snapshot_audit_v1`，已完成`preparation_snapshot_audit_v1`：固定开始时已完成的37份per-clip records，逐一核原retarget/motion/geometry/contact哈希、真实30Hz时间索引、全部source Root与重新解码的FK、每个scene的接触标签及家族split一致性。3878配对/1386076个接触标签与原retarget joints上的既有PARC规则逐值一致，Root最大误差9.61e-7m、FK最大1.222e-6m。源、场景、准入门、训练代码均未改。该审计**没有重新计算完整skin碰撞，也不是模型质量或全量数据完成证据**；小报告已下载`out/pfnn_mixed_20261015/preparation_snapshot_audit_v1-report.json`。

覆盖统计区分配对数与真实source窗口去重数（镜像/重叠仍相关）：当前train flat为6519pair-windows/5229unique-source-windows、10录制家族，末速度P5/median/P95=0.0575/0.9336/2.1825m/s，max3.5765m/s；按原审计定义有idle目标232、起步125、停止51窗。train jumpy为1313/1142窗、2家族；validation flat为3134/2510窗、3家族。这里只包含已完成的前37段，不能据此断言全部PFNN没有test或楼梯数据，更不能把hindsight末速度当录制WASD。后续完整cache/audit仍必须检查三domain各split和最终覆盖。

本轮发现A100系统根分区已100%、普通用户可用0bytes（inode未满），数据盘/data5仍约4.4TB可用。首次从SSH默认home工作目录启动新CPU审计，在torch导入创建temp目录时失败，尚未创建audit输出；证据保留`preparation_snapshot_import_failure_v1.json`。随后显式设`TMPDIR=.../pfnn_mixed_20261015/runtime_tmp`（仅本任务目录）成功完成相同审计。实测既有准备/监督代码cwd下Python默认tempfile会回退到/data5上的当前目录，均可实际创建并清理临时目录；因此没有因该新审计失败而中断健康主任务，也没有删除系统盘/他人文件。根分区空间问题本身仍未修复，后续新任务优先显式用本任务TMPDIR，其他系统盘写入仍须检查。

下一次跟进：先查原两健康进程及新产出，继续等待80段修正准备完成，再由既有监督自动缓存/全数组hash及split覆盖审计、启动同一预算。已完成的37段审计不重复跑；若后续出现新的数据边界错误，针对新增源审查。保持28942/28941/28936；最终仍需完整模型闭环、独立路线、flat起停转、官方SOMA真实WASD/30fps验收。

## Previous Work: Verified PFNN Repair; Corrected Full Workflow Running

2026-10-15 16:28 Asia/Singapore：PFNN坐标/接触缺陷已实现并通过真实数据审计。原监督full_training_v1 PID3958381在0正式更新停止，原配对PID3754207在48/80时停止并完整保留已拟合geometry及旧准入，记录分别为`full_training_v1/data_repair_stop.json`、`dataset_full_v1/coordinate_repair_stop.json`。**不要重启原v1任务或消费其canonical/world混用的PFNN清单。** 没有重做已完成的80段retarget。

新不可变 `code_world_repair_v1` 的 `dataset_world_v2` 已在GPU6实际运行，PID1582020，进程记录`dataset_world_v2.process.json`；最新34/80段重筛、3577配对、10382准入窗。它复用原有static geometry数组，修正native initial transform还原world，再重做全skin/retarget/接触准入；尚未拟合或损坏的场景按同种子/同原方法补建。每clip现在单独落盘world编码证明和records，旧日志/场景保留。

`world_repair_pilot_v1`用3真实源/6既有场景完成；`world_repair_audit_v1`已通过：旧Root world错位最大0.1703/4.7741/5.4345m，修复后Root最大<8.87e-7m，FK<1.19e-6m；6场景数组逐值未改，真实帧/姿态未改，230个旧接触标签改变，全部新标签与既有PARC terrain-relative detector在原retarget FK上的结果逐值一致。只有通过原几何门的3个pilot窗口进入cache attachment检查，失败保留；这不是动作质量改善声明。

数据新契约`pfnn-world-and-scene-contacts-v2`、manifest schema`motionbricks.pfnn_static_pairs.v2`。共享motion文件的接触通道是占位，**每个拟合scene必须加载其hash绑定的4通道contact sidecar**；真实source offset/frame indices/FPS及33帧窗口覆盖均检查，cache直接拒绝旧v1。接触规则沿用现有PARC世界速度<0.15m/s、地形相对关节高度<0.10m；只改监督标签，不改motion/geometry/loss。6个本地坐标、接触、旧清单拒用及pairing测试通过。

`preflight_world_v1`已通过，包含旧up/down/idle及3个修正PFNN真实窗口：原始观测/Pose教师/批量准备/未来Y隔离差0，Root输入max1.19e-7，三阶段梯度有限，0optimizer更新并恢复初始权重。`mechanics_world_v1`也完成8个测试更新，Root连续vs断点恢复权重/样本逐值一致，Root→codec→Pose完整链/冻结哈希通过；fixture明确非质量/正式训练证据。小型报告均在本机 `out/pfnn_mixed_20261015/`。

新不可变 `code_full_workflow_v2`，耐久监督 `full_training_v2` PID1613357已启动并在等待`dataset_world_v2`。新流程使用`preflight_world_v1`/`mechanics_world_v1`，随后生成`cache_pfnn_world_v2`，和不受该PFNN缺陷影响的`cache_mixed_v1`做完整hash/split/覆盖审计，再启动同一授权Root→codec→Pose各50000更新、batch256/micro64、50/25/25配额；原正式更新0，总预算没有增加。`pfnn-wasd`自动化提示已同步这些新路径和禁用旧v1的说明，继续每10分钟。

独立保留数据采集/筛选已全部完成，相关worker不应重启。`evaluation_with_generated_v1`共42合格完整片段；独立几何数validation up4/down5、test up4/down5（down各含1条旧保留源），每方向>=3目标达成。新32候选及失败全部保留，8新authored场景从未进入本次训练缓存。该数据只验证未来actor，不能称为MB模型质量通过；原回归10路线仍保留。

下一次跟进：先查PID1582020和PID1613357的真实cmdline、新state/progress/complete/failure及最新产出。健康时不重复已完成修复/诊断，不恢复PID3754207/3958381或旧数据入口。关注全量world编码/FK/sidecar检查是否出现新边界错误，以及最后split/类别/速度覆盖是否够；失败保留并针对修复。正式开训后报告actual_optimizer_updates；最终模型使用已验证的strict offline loader和完整闭环评估，再独立WASD/30fps。28942/28941/28936保持，短卷积候选未采用，几何和数值门不变。

## Previous Work: PFNN Coordinate And Contact Defect Found Before Training

本轮复核发现**必须修复的PFNN数据缺陷**：`build_pfnn_terrain_dataset.raw_features`调用native dual_rep后丢弃initial transform，将motion归一到起点XZ=0/heading=0；fit_terrain和scene却使用retarget真实world。实查LocomotionFlat01首root为[0.1620,1.0325,-0.0244]、NewCaptures03为[-0.0351,1.0175,0.9516]、NewCaptures07_mirror为[-0.0298,1.0104,1.6909]，旧编码XZ都变成0。另外未传foot_contacts触发native绝对地面高度<0.10m检测，不适用于已抬高地面。原先PFNN非平地准入结果和旧source-contact标签不能直接用于训练，问题来自新数据准备，不是SOMA运行时。

已在2026-10-15 15:48严格核cmdline/state/没有root训练目录后，仅停止尚在等待的full_training_v1监督PID3958381，记录 `full_training_v1/data_repair_stop.json`，正式optimizer更新0。**不要恢复原v1监督或消费旧PFNN v1清单。** 旧PFNN配对PID3754207暂保留，已有静态fitted geometry是基于真实retarget support拟合，可复用；其旧准入判断需重做，旧文件/日志/快照全部保留。原PARC/BONES缓存不涉及这段PFNN编码，仍保留。heldout采集/筛选独立继续。

本地已实现 `pfnn_world_contract.py`：使用native initial transform恢复真实world并核Root/FK；scene-specific接触沿用PARC现有terrain-relative SOMA关节速度<0.15m/s、高度<0.10m规则，按真实episode保存4通道sidecar。2个坐标/FK及joint+floor共同平移不变性测试通过。builder已改为v2schema、显式world_contract、每scene接触sidecar、可复用已拟合geometry且重做准入、每clip落盘records；cache拒绝旧v1，要求sidecar哈希/源帧钟/窗口覆盖一致并在增广前载入接触。**尚待A100真实pilot与新bulk启动，不能声称修复已全量验证。** 下一步先验证真实源/几何/接触和复用一致性，然后新不可变快照恢复同一训练预算；必要时在新流程可运行后停止旧无效准入进程并复用全部已落盘场景，不能丢弃已有拟合结果或盲重跑完整retarget。

## Previous Work: Readiness Recovery Complete; Automatic Held-Out Screening

2026-10-15 15:43 Asia/Singapore：PFNN主配对PID3754207仍健康，47/80源、4781配对、11827准入窗；full_training_v1监督PID3958381健康等待，正式optimizer更新0。

本轮发现heldout GPU5旧worker1425269在layout4生成结束后physics入口报 `Probe GPU is occupied`，生成4候选均已完整保存，旧recording_v1只有未完成入口目录。该不可变v2快照的physics脚本只有一次占用检查，日志未记录当次具体占用值，不能把遥测延迟当作已证明唯一原因。检查确认原worker/child已退出、GPU5当前19MiB/0%，以新不可变 `code_heldout_collection_v3`恢复同一plan/layout4..7，**不重生成候选**，新物理输出`recording_v2`保留旧目录/日志。恢复worker PID1450280，记录 `heldout_recovery_gpu5_v1.process.json`，状态仍`heldout_collection_v1/worker_5.json`。

新版physics入口保留原1024MiB/5%阈值，最多20次/3秒检查，逐次保存`gpu-readiness.json`；持续占用则退出，绝不占用其他任务GPU。2个本地测试验证util/memory必须同时过门及持续占用有界退出；Windows测试仅把fcntl导入移到Linux入口，锁行为未改。layout4恢复时首次19MiB/0%过门，已完成physics和SOMA，后续layout5继续运行。GPU4 worker1425265已完成layout1,2,3且collection_complete，未重启它。

新独立耐久筛选/汇总 `finish_parc_heldout_evaluation.py` 已启动，PID1472319，GPU2按锁和新鲜空闲检查顺序处理已完成layout，输出 `evaluation_with_generated_v1`、状态`evaluation_with_generated_v1/state.json`，不可变`code_heldout_finisher_v1`。已有layout0严格报告复用；layout1严格报告已完成，当前layout2在筛选。首次直接脚本启动缺PYTHONPATH在创建输出前失败，已用模块入口正确启动，未创建重复任务。运行期间`finished_layouts`在一轮扫描末才更新，若正screening应以各`heldout_layoutN_screen_v1/complete.json`和当前child为准。

该finisher验证所有worker覆盖8场景恰好一次、plan hash不变，沿用相同全skin/完整区间门；失败worker会记录并等待代理诊断，健康其他场景继续筛选，绝不自动重试生成。无完整路线的场景保存原候选并记skip。全部结束后与既有2条保留down合并评估case，核32候选库存，按**原始authored几何hash**统计独立validation/test方向覆盖，避免把同场景多候选或retarget缩放变体算作多个独立场景。结果只用于离线评估，不进入当前训练缓存，不授予quality/WASD批准。

下一步：检查主配对/训练监督、恢复worker5 PID1450280、finisher PID1472319；健康不重启。关注worker5后续physics的readiness记录与source筛选是否完成。finisher完成后查看独立场景数/失败原因/目标是否满足，取回小JSON；不要在当前模型未训练完时把验证源成功说成actor改善。所有生成/模型/动画仍留远端或私有传输，28942/28941/28936保持。

## Previous Work: First Reserved Layout Passed; Remaining Seven Running

2026-10-15 15:29 Asia/Singapore：PFNN主配对47/80源、4781配对、11827准入窗，PID3754207健康；full_training_v1监督PID3958381健康等待，正式optimizer更新0。GPU0训练预留、GPU7配对、GPU1他人均未触动。

新保留场景layout0的4候选已全部生成，3条在原生阶段覆盖全路线；46次生成中45次接续prefix检查、最大误差1.788e-7。官方physics recording已完成，3条SOMA转换通过既有screen。worker_3 PID1375262已collection_complete。**既有admission不是严格源门**：独立 `screen_parc_heldout_collection.py`已完成全部3条真实源的fullskin及上下楼区间核验，生成 `heldout_layout0_screen_v1/complete.json`，PID1420168已结束。

严格结果：3段上楼峰值0.5008/1.2094/0.8141cm、1段下楼峰1.8105cm，均P99=0、无unknown/uncertain，真实81帧区间满足原前后续帧定义。另两段下楼4.2256/2.6556cm失败保留；首条完整源峰7.5765cm也如实保留，不能把局部合格段说成整条源都干净。四个合格区间只代表**一个独立新几何家族**。源hash/官方physics-export/retarget报告、30Hz全帧、原生编码及world恢复均验证；恢复FK最大约8.03e-6m。无修正/补帧/训练；小报告已下载本机 `out/pfnn_mixed_20261015/heldout_layout0_screen_v1-report.json`。

首组已具备合格up/down证据，因此已受控派发预定义剩余7场景/28候选，总预算仍32。新不可变 `code_heldout_collection_v2`，GPU4 worker PID1425265负责layout1,2,3；GPU5 worker PID1425269负责4,5,6,7；启动时分别实查空闲，阶段也重新查空闲。进程记录 `heldout_remaining_v1.process.json`，状态 `heldout_collection_v1/worker_4.json`与`worker_5.json`，日志 `heldout_remaining_gpu4.log`/`heldout_remaining_gpu5.log`和各layout阶段日志。两个worker均已实际生成，非仅排队。先查cmdline再操作，健康不重启。

下一步：先查PFNN/训练监督及两个新PARC worker。每个新layout物理+SOMA阶段完成后，在空闲卡用不可变 `code_heldout_screen_v1` 的 `scripts.screen_parc_heldout_collection --layout N --collection .../heldout_collection_v1 --mixed .../long_route_mixed_10k_ab_v1/mixed_dataset --output 新目录`执行同一严格核验（launcher的heldout-screen分支目前只写死layout0，不能误复用）。保留全候选与失败，按geometry家族计数；所有新源只为validation/test，不加入当前训练缓存。直接screen通过旧转换helper核验完整来源并恢复真实world，**不要使用要求train来源的旧finisher混入anchor或伪造train行**。汇总后将合格独立场景接到已验证的离线评估链；这仍不是MB动作改善证据。28942/28941/28936与训练预算/快照保持，短卷积候选未采用。

## Previous Work: New Held-Out PARC Collection Preparation

2026-10-15 15:14 Asia/Singapore跟进：PFNN主配对已46/80源、4748配对、11774准入窗，PID3754207健康；full_training_v1监督PID3958381健康等待，正式optimizer更新0。GPU0保留既定训练、GPU7主配对、GPU1他人任务均不动。

复核旧 `native_preserve_cases_v1` 的4条review_only源：它们的录制家族在当前mixed全部属于train，不能补独立验证。又读取rugged32与pose_entry两套既有native数据集的validation/test，排除当前train家族/原始动作hash/转换hash重合、跳过201重复源记录，复用之前100源结果不重复几何评估，并使用各源数据集自己的统计量恢复真实world，避免跨统计量错误。`evaluation_expanded_sources_v1`已完成，累计107独立源，只新增validation下楼 `generated_7f68f4f989b6eebb`（105帧、峰1.96503cm），加之前test下楼共2条。仍没有满足现有完整路线/2cm门的独立上楼，所需每split每方向3条未达成。报告本机 `out/pfnn_mixed_20261015/evaluation_expanded_sources_v1-report.json`，新case张量只留远端。PID1346612已结束。

因此按已有PARC扩样授权准备新的**仅用于验证**场景，未改训练数据/预算。`build_parc_heldout_stair_course.py`定义8个新场景：riser0.115/0.135/0.155/0.175m、tread0.4/0.5m、5/6级，6.5m真实接近、5.5m平台、6.5m出口，两个水平轴。layout0..3固定validation，4..7固定test，所有候选/重试保留该split；不进入本次full_training。8个几何hash唯一且与已核pose-entry/rugged生成计划不重复，2项本地真实台阶/路段/独立性测试通过。新生成时间晚于初始模型，但完整上游训练史仍不夸称已穷尽审计。

不可变 `pfnn_mixed_20261015/code_heldout_collection_v1`由已验证 `model_terrain_v3_20261011/code_parc_pose_entry_20261014_v2`另建。`heldout_collection_v1/plan.json`最多32候选，**当前只派发layout0的4候选**，不得把其余28说成已启动。GPU3新鲜空闲后启动worker PID1375262、生成子进程PID1375279（均先核cmdline）；真实状态 `heldout_collection_v1/worker_3.json`、总日志 `heldout_collection_v1.log`、阶段日志 `layout_00/generation.log`。复用原生PARC checkpoint a03a9d4c...、既有factual-prefix flag preservation/30Hz接续、官方物理tracker及既有离线SOMA retarget/admission，不加停止任务或SOMA运行时修正。进程已实际产生逐段motion，尚未完成生成/物理/最终精确源几何核验。

下一步先读主配对/监督与heldout worker状态，健康不重复启动。首组结束后查全部4候选（包括失败）、prefix-contract、真实全路线覆盖、官方physical recording、retarget，再以18056顶点8substeps精确峰<=2cm/零unknown/uncertain及既有前后续帧定义复筛。旧admission通过不自动等于严格源门通过；先有首组可用证据再决定其余layout1..7的受控派发，不盲重试失败。不得使用旧finisher把该保留数据混进训练anchor；生成scene和模型/动画仅远端/私有Drive。新评估case可通过 `evaluate_full_corpus_controller.py --evaluation ...`使用，原回归仍保留。28942/28941/28936、原训练快照和短卷积拒用决定保持。

## Previous Work: Closed-Loop Evaluator Ready; Held-Out Route Shortfall

2026-10-15 14:58 Asia/Singapore跟进：主配对PID3754207与监督PID3958381持续健康，45/80源、4639配对、11619准入窗，NewCaptures07_000_mirror已产出至episode91，正式optimizer更新0。不要重复启动或中断健康主进程。

本轮实际核对旧10条楼梯回归路线与当前mixed accepted录制家族：8条不在新mix里，`official_dc03d6484b176db0-down`、`generated_0f7ea175eeec6664-down`属于新mix的train。旧模型已直接或间接接触过这些回归源，不能把它们当全生命周期独立测试。保留原有10条路线及失败作能力保留对照，不据此自动改训练数据/权重。

新 `prepare_full_corpus_evaluation.py` 完整检查100个validation/test PARC源；沿用既有 `stair_intervals` 定义（40真实接近帧、至少3级/30cm高差、24真实平台后续帧）及原18056顶点8substeps峰<=2cm门。99源未找到满足此完整定义的区间；1源产生两个方向候选，其中一条几何失败，只有test下楼 `official_51038699f247c9b9-down`通过（原始133起105真实帧、5级、峰1.4195cm、P99=0、无unknown/uncertain）。目标每split每方向3条未达成；这不是声称所有短片段都不可用，而是当前完整路线验证覆盖不足，未来需补独立真实上楼/下楼完整路线，不能以原回归成功或放宽定义冒充泛化通过。

`evaluation_sources_v2/complete.json`、`heldout-cases.pt`已完成，报告本机 `out/pfnn_mixed_20261015/evaluation_sources_v2-report.json`，动画留远端。v1只因快照缺既有 `prepare_unified_pose_cases.py`依赖在导入时失败，已在不可变 `code_evaluation_prepare_v2`补齐，旧日志保留。PID1315777已结束。

新 `evaluate_full_corpus_controller.py`已实现：严格完整模型离线加载，保留回归/当前split-heldout身份，复用原60包/16秒完整路线目标/峰值/横漂门和实际生成history，另记录16秒idle后walk/stop/run/90度turn/stop的135包命令与真实发布帧、全skin和跨包连续性。命令为离线脚本而非浏览器WASD；flat只报实测，不发明新及格阈值。`evaluation_smoke_v1`在不可变 `code_evaluation_smoke_v1`、GPU2实查空闲后运行完毕：mechanics模型的up/down各source/idle起步及flat各2个真实包共20帧，均无执行错误，模型张量hash不变、0optimizer更新；明确smoke_only=True、全部未完成完整路线，不作动作质量证据。PID1319940已结束；小报告已下载本机，recordings只留远端。

下一步先检查主配对/监督状态。若仍健康且无阶段变化，不重复本轮已完成诊断、不发送常规进度通知。完整模型结束后使用已验证离线加载/评估链；独立完整上下楼验证源仍是交付前缺口，可继续调查既有PARC review池或按已授权PARC生成方向准备新的保留数据，但不得把其加入训练后又称独立测试。训练快照/预算/数值门及28942/28941/28936均保持；短卷积候选仍未采用。

## Previous Work: Offline Full-Corpus Loader Verified

2026-10-15 14:43 Asia/Singapore跟进：主配对PID3754207、耐久监督PID3958381均cmdline匹配且健康，PFNN45/80源、4639配对、11619准入窗，正在NewCaptures07_000_mirror，文件持续产出，正式optimizer更新0。不要重新启动或中断主配对，末尾完整清单仍在内存中。每10分钟的 `pfnn-wasd` 继续有效。

本轮补齐离线全量模型加载契约 `motionbricks/data/full_corpus_inference.py`：严格检查最终Pose阶段、完成更新/曝光数、bundle/数据审计/runtime契约、全部Root/Pose/codec/adaptation张量及新码本词表形状；压缩文件删除optimizer但保留全部模型张量，重建controller后严格加载、刷新非持久codebook缓存，并核实原生inference各组件引用同一组已加载模块。只允许offline_evaluation，不授予interactive/quality批准；mechanics必须显式诊断optin，默认拒绝。10本地负例/压缩契约测试通过。

不可变远端 `code_full_handoff_v1` 的 `full_handoff_v1/complete.json`已通过，诊断PID1267455已结束。采用已完成 `mechanics_v4/pose/weights.pt`，不是未训练的全量候选；初始权重与压缩后独立新推理器在up/down/idle的world/local_prediction/Pose logits差全部0，active token变化0，新codec码本相对旧bundle最大变化0.0024804473且已正确加载。0optimizer更新、未启动服务器。诊断压缩模型留远端，不经公网SSH传输；小型JSON报告下载本机 `out/pfnn_mixed_20261015/full_handoff_v1-report.json`。这只证明导出/加载一致性，不证明新模型质量或WASD已交付。

下一步先检查配对/监督是否进入正式缓存或训练，健康时推进独立闭环评估准备。正式模型训练结束后复用离线加载器做完整路线/flat/命令切换评估，再做有质量证据的独立WASD loader/server。旧28942/28941/28936与训练快照未改；被拒短卷积候选未接入。

## Previous Work: Scheduled Monitoring And Production Capacity Check

2026-10-15 14:20 Asia/Singapore：用户要求立刻继续推进并开定时监控。已创建本聊天heartbeat `pfnn-wasd`，ACTIVE、每10分钟执行；数据库核实绑定本聊天 `01a12158-7eb0-7d83-bfe7-a823e60eb3f3`。远端耐久监督每30秒检查前置任务。无变化安静，重要进展、故障、需用户行动或可试用才通知；完整验证交付后停用。

最新实查：PFNN `dataset_full_v1` 完成45/80源、4639地形配对、11619准入窗；PID3754207健康。完整配对清单及准入详情仍只在原进程内存中，末尾才落盘；不要为了并行化直接中断丢失它。旧PARC/BONES `cache_mixed_v1` 全部2668源已完成：86115基础窗考察，83132准入、2983拒绝，四个旋转视角共332528primitive。`full_training_v1` 监督PID3958381健康等待PFNN，正式optimizer更新0。

为在等待时实际推进，另建不可变 `code_preparation_review_v1`，独立CPU审计 `ready_mixed_audit_v1` PID1209179检查已完成语料的家族split、各域速度/起步/停止/高度覆盖；GPU2实查空闲后启动 `production_capacity_v1` PID1209196，检验正式batch256/micro64三个阶段的前向反向/冻结梯度/显存/耗时，并对up/down/idle核对确定性pool与默认运行时的token/动作一致性。每阶段3次梯度评估、无optimizer更新、恢复初始权重；PFNN部分重复mechanics fixture，只是容量检查，非新质量/全量训练证据。先查complete/failure/progress/log，不重复启动。PID均只作线索，操作前查cmdline。详细结果应在完成后追加在本节。

上述两项已完成并下载小型JSON报告到本机 `out/pfnn_mixed_20261015/`。`ready_mixed_audit_v1` 家族split通过；训练PARC22086基础窗、259录制家族，BONES57995基础窗、155家族。BONES有16954idle目标、2153起步、2226停止窗；PARC仅8/38/28，不足以证明楼梯停走覆盖。PARC训练终端速度P5/中位/P95为0.518/1.794/3.651m/s，BONES为0.0031/0.531/2.147m/s。保留全部划分和失败，不据此盲改采样或标签。

`production_capacity_v1` 三阶段batch256/micro64前后向均通过，Root/codec/Pose热身后每批耗时7.30/10.63/5.13秒，峰值已分配显存34.13/9.38/2.38GiB；Adam动量额外预计0.27/0.18/1.05GiB。不是完整optimizer/IO/验证计时。up/down/idle默认pool vs确定性pool活跃token变化均0，最大world feature差1.60e-5，初始全部权重恢复、0optimizer更新。单卡原预算仅这部分算术耗时已约13.3天，故另启动只读算子性能剖析 `production_profile_v1` PID1215304、GPU2、不可变 `code_preparation_profile_v1`；每阶段1次热身+1次profile微批，0optimizer更新，不导出模型。下一步读取其complete/progress/failure，先按实测定位再决定无目标变化的实现加速，不擅自减样本/更新或放宽数值和质量门。

14:34更新：上述所有独立诊断已结束。`production_profile_v1`因record_shapes记录/聚合开销大、GPU保留约68GiB且数分钟无报告，经cmdline验证后仅终止该诊断，保存cancelled.json；原配对和监督不受影响。新 `production_profile_v2`轻量cProfile完成，主要耗时在短序列Conv1d；Root微批64的prepare仅0.038秒，不能归咎数据准备helper。函数计时含同步，非逐GPUkernel计时。

新 `short_sequence_convolution.py` 为显式context manager的离线候选：仅当3tap、stride1、zero padding=dilation且dilation>=实际序列长时，把无效两侧tap化为中心F.linear，参数/权重键不变。7个CPUfloat64前向/全梯度/异常恢复测试通过；**未接训练或运行时**。`short_conv_gradients_v1` Root梯度门通过(max2.32e-6、relativeL2 1.93e-5)，codec未通过原5e-6绝对梯度容差(max1.54e-5)，不能采用或放宽门。`short_conv_capacity_v1`虽三阶段容量通过、运行时up/down/idle活跃token差0，Root/codec/Pose耗时6.37/5.54/2.73秒，但不推翻梯度门失败。两个新探针均0optimizer更新、无模型交付。全部失败/候选快照保留。

**下一次跟进**：先核 `dataset_full_v1`/`full_training_v1` 与最近文件产出，配对现在正在NewCaptures07_000_mirror，仍45/80已完成。正式全量更新0；`code_full_workflow_v1`训练快照完全未改，预算和默认卷积保持不变。健康时继续独立代码/吞吐研究及可复核推理准备；不要重复跑已完成诊断、不要把被拒短卷积候选接入正式工作流。数据全备后监督自动启动既定全量训练，监控必须报告实际进度、故障与完整验证。定时任务已在本机创建并核实，不再沿用下文“无自动化”的历史状态。

## 当前授权与目标

用户最新指令：优先取得大量 PFNN 动作，retarget 到 SOMA；混入大量 PARC 和 BONES-SEED；完成数据准备后推演并 review 训练/推理代码，再启动全量训练。此指令明确授权新的全量训练，取代此前“仅诊断、不新增完整预算”的限制。禁止重启旧训练或覆盖旧快照，28942 的已认可模型保留作对照。

目标仍是 terrain-aware MotionBricks 的 WASD 控制，不是训练 PFNN 网络。以 MB 论文和实际代码为依据，保持 Root→Pose→codec 的可核对推理链。运行时不加 IK、锁脚、碰撞投影、Root 替换或输出平滑。骨架转换只在离线数据准备阶段进行。

## 已确认数据来源

- 本地现成包：`F:/dev/GASP_57/ai4animationpy/Demos/Dataset/PFNN`。
- 当前枚举 81 个 BVH，共 889249080 字节，全部与作者 ZIP 的 CRC/大小一致。实际为 80 段 120Hz 动作、1 个单帧参考姿态；931981 原始帧，34 个动作录制家族，原始/镜像各约 1.079 小时。
- 作者项目：<https://theorangeduck.com/page/phase-functioned-neural-networks-character-control>。
- 作者 Code & Data ZIP：<https://theorangeduck.com/media/uploads/other_stuff/pfnn.zip>，HEAD 返回 200、500215625 字节，可范围读取。
- 本地 `skeletondef.py` 开头是裸骨架文本，预处理脚本还存在自定义骨架/单位修改。必须核对作者原始代码与原始 BVH，不能直接运行这套修改过的预处理脚本。
- g3 仅查到 `p11/codec_cache/fc_pfnn` 和 `sqc/fc_pfnn` 衍生缓存；g2 查到 `deepacl/data/processed/clips/pfnn` 和 `s3/clips*/fc_pfnn` 的部分处理片段/软链；A100 所查目录没有完整原始包。g2 使用既有 `${SSH_CONFIG_DIR}/known_hosts` 和 HostKeyAlias `[remote-host.example]:12302`，未绕过主机验证。
- 私有源包：`out/pfnn_mixed_20261015/source_inventory_v3/pfnn_source.zip`，440618298 字节，SHA256 `e04c2ffe900e8904458cc3a6fbeebb54a51cd1af891940d5d37c037f31fd24d7`。heightmap 仅恢复原始 LF；本地有差异的标注从官方 ZIP 恢复到新包，原文件未改。失败的 v2 部分包保留，未上传。
- 通过用户私有 Drive 上传并下载到 A100，哈希核验、解包完成：`${PRIVATE_STORAGE}/motionbricks-20261010/pfnn_mixed_20261015/source/unpacked`。
- 新代码目录 `pfnn_mixed_20261015/code_pfnn_mixed_v1` 基于不可变 v8 另建。离线 GMR 依赖在 `pfnn_mixed_20261015/vendor/GMR`，严格固定 `bb1bbe40774794fceb2a7c579a3464a28e68c844`；没有改旧源码或共享环境。首次远端 smoke 因误指不存在的旧 GMR 路径失败，v2 用独立固定版本成功。

## 执行顺序

1. **已完成：来源审计。** 官方目录、BVH 身份、FPS/单位/31 关节、标注及镜像分组已核对，结果在 `source_inventory_v3/inventory.json`。物理单位为原作者 `5.6444` 厘米缩放，即每原始单位 0.056444 米。
2. **已完成：离线 retarget。** v2全部80动作、233010个真实30Hz帧完成，映射/FK/单位检查通过；原始失败与局部误差保留并按窗口筛选。
3. **进行中：地形配对。** 作者步态周期patch/RBF方法已实现，静态场景与真实SOMA蒙皮几何核验正在执行，不随帧贴地。
4. **进行中：统一训练集。** PARC/BONES全量几何复筛和缓存正在执行，PFNN配对完成后自动缓存；家族split、全数组哈希和速度/起步覆盖审计是正式训练启动门。
5. **已完成当前代码预检。** 实际模型输入/因果隔离/梯度通过；实际Root→codec→Pose更新链及冻结哈希通过，Root断点恢复逐值一致。详细推演和限制见 `pfnn_full_train_inference_review_20261015.md`。
6. **已排队自动执行：全量训练。** 独立耐久工作流已启动，等待完整数据及审计门通过。每阶段50000实际更新、batch256/micro64、PFNN/PARC/BONES50/25/25，最多150000更新/38400000训练样本曝光。当前正式优化更新仍为0，不能把排队、预检或fixture更新说成全量训练已开始。

## 当前工作状态

新策略中的 0.4 米重采样与高度信号诊断脚本仅写在本地，尚未上传或启动；它不是当前优先任务，也没有改动服务。用户已回复要求检查 g3/g2/A100，检查结果如上，不再等待路径回答。

PFNN 转换 v2 已完成80/80段、233010个30Hz真实帧。v1在36段后遇到四元数分量低通的奇异方向，全部原日志保留；v2使用时间戳SLERP与Root线性插值，不滤波、不补帧，80段重新转换，未混用v1。18段含局部任务误差，按33帧窗口复核，失败记录保留。

静态地形配对 `dataset_full_v1` 正在GPU7运行，PID3754207；最近核对34/80段、3577个地形配对、10316个准入窗口。作者patch/RBF方法作用于转换后SOMA的真实支撑点，场景在每段内静态；这不是实测capture terrain。所有13张高度图已生成653个patch。旧PARC/BONES混合语料的全量几何复筛与原始观测缓存 `cache_mixed_v1` 在GPU2运行，PID3839628，最近283源/9964增强primitive准入、1094基础窗被拒。每次操作前核cmdline。

`preflight_v6/complete.json`已通过：up/down/idle原始观测/Pose教师差0，Root输入最大3.73e-9，批量准备/未来高度隔离差0，三个阶段梯度有限，0更新并恢复初始权重。前期检查错误与训练helper时长形状、Root训练时长标签、原Pose构造器把requires_grad_赋bool等修复均留在不可变失败快照。

`mechanics_v4/complete.json`已通过，8次实际测试更新，Root连续训练vs恢复权重与样本逐值相等，完整Root→codec→Pose链及保护哈希通过。fixture包含重复真实源，只验证优化器机制，不能充当独立验证/质量/全量语料。v1微小浮点差、v2非确定性pool报错、v3codec原CUDAcumsum报错均保留。pool数学等价确定性实现单测通过；codec保留原生cumsum并显式warn-only，不承诺其bitwise复现。

不可变快照 `pfnn_mixed_20261015/code_full_workflow_v1`；工作流输出 `full_training_v1`，PID3958381，进程记录 `full_training_v1.process.json`，状态 `full_training_v1/state.json`。已核cmdline，当前 `waiting_for_existing_preparation`、正式训练未开始。监督会等待配对和旧缓存、缓存PFNN、审计完整split/覆盖/所有数组SHA，再在新鲜空闲GPU0执行各阶段；两步启动核验使用正式预算同一优化器，随后恢复剩余步数。失败自动停住、不重复启动。既有joint-sdf自动化ID在当前本机自动化库中未找到，未新建或改其他自动化；耐久远端工作流独立继续。

阶段推演为Root(含共享SDF编码器)→完整codec→Pose。Root接受连续真实history，独立于码本；共享编码器在后两阶段冻结，保证环境特征一致；codec更新编码器/EMA码本/decoder后，Pose显式重绑定新码本输入嵌入。保留MB原loss、源Root作为Pose训练教师，实际actor仍用预测Root。源history训练不能宣称解决闭环恢复或泛化。

大数据/模型/动画跨机器继续经用户私有 Drive、NAS 或内网；公网 SSH 仅用于小型源码和指标报告。保持严格 SSH 主机验证，不使用被拒绝的临时代理，不动其他人的 GPU 工作。
