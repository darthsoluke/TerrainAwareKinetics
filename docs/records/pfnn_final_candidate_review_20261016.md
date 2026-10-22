# PFNN Final Candidate Review

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
28950experiment2-frame selected/paused;28944original preserved/sharedPID25772.
4/2/1targets display30fps; actualupdates~7.5/15/27.4Hz, frozenmodel/0training.
GPUtokens/FK/skinparity passed,2878rendered-pluscommitted histories exact.
Idle drift0.031/0.050/0.131m/s preventsqualitypromotion; fullskinterraingate notrun.
3Dqueryfoundation34CPUcheckspass, notyetconnectedtonewtrained3Dbehaviors.

Latest priority: [3D-first framewise locomotion plan](kinematic_3d_reactive_plan_20261021.md).
Target PFNN-level control/contact quality before broader skills. Existing3D
sampling still derives from a heightfield; published PFNN demo uses framewise
regression and default foot IK/locking. Planning review only;28944/Xbox unchanged.

Latest23:52: Active page changed by user request to the trained pure kinematic
SOMA model:28944/PID11952,15maps,007/speed3, inline tuning. W800ms moves the
model at observed30fps/no visible error.28948physics27336/planner18108 stopped.
Same checkpoint0f5d8e2b;0training. Physical acceptance results remain below.

Latest23:44: [Retained registration repair](mb_scene_registration_repair_20261020.md).
373tests pass; chart/Foot-bank fixes and earlier real-reference renewal retained.
v50/v51 marker-fitting candidates rejected and removed. Current v52/28948/27336/
18108 restores15maps and normal controls, Ready. Urban33.859s physical fall at
30.016fps/0drops; no joint/motor limit or reference-coverage error. Stable terrain
control and delivery remain FAILED.0new training/filter/actual-state overwrite.

Latest23:07: [Registration/foot repair](mb_scene_registration_repair_20261020.md).
Fixed invalid local terrain coordinates and missing native foot bank.371tests
pass; physical result pending. v49/13988/28948 restores learned Root; v48 stopped.
Chart-invariance error0.7983m->1.634e-6m.0training; stable control remains open.

Latest21:26: [Session lifecycle repair](mb_session_reset_fix_20261020.md). Reset/
reload/Start verified; old commands no longer own new episodes. v44 later
coverage failure retained. v45 earlier renewal under unchanged limits tested
numerically; live continuous check pending.215tests;35132/11348/28948.

Latest19:47: [Intent/stopping repair](mb_intent_errata_20261020.md). Frozen Root
probes reveal incorrect braking before codec/tracker. Pre-Pose bounded planar
reference improves stopping/reversal sign. v42flat123205 and v43PFNN000136968
physical steps pass, both complete83s input,~30fps/0drops. PFNN penetration4.372mm.
381tests,0training/filter. Livev43/28948 Ready,29536/19844,15maps, DevGPU scope.
Side/back speed and latency remain insufficient; no all-terrain acceptance.

Latest: [Implemented repair](mb_online_repair_20261020.md). v39 completes the
83s continuous analog trace and187.46s total without falls/mechanical violations;
187360substeps audited,29.991fps/0drops under explicit DevGPU scope. Fixes:
native context/handoff, planar integration origin, material-axis retargeting.
Reverse strafe and stopping remain failed.376Python checks pass;0training.
Root-rate v40 rejected for input timeout. v41 restores15PFNN maps/28948 using
v39 runtime. No stable-Xbox/all-terrain approval.
v41 PFNN000 fails on unknown ground at28.279s,29.993fps/0drops. Page explicitly
reset after saving evidence; server33428/planner11348 retain15maps and normal input.

Latest: [Code and primary-paper review](mb_online_control_review_20261020.md).
Reference joins introduce large motion before history/model corruption. Frozen
tracker's missing direct Root target confirmed by2CPU calls.47saved plans and
8papers reviewed,0new training/physics/GPU,0runtime changes. No local Python
service or28947-28949listener remains; saved evidence is intact. Stable control
remains unresolved; repair frame/time handoff before new control/training layers.

Latest16:09: [Simulated Xbox validation](mb_offline_online_gap_20261020.md) FAILS:
PFNN35.326s/flat64 old4.885s/repaired-preview6.935s falls at~30fps/0drops. Future
target immutability is fixed; source Root frame joins/reference dynamics still
cause severe tracking mismatch. No training/filter. Failed GPU servers stopped;
only28949/PID22856 CPU comparison remains. No stable-gamepad/all-terrain approval.

Latest: [Continuous online repair](mb_offline_online_gap_20261020.md). Exact saved
reference reproduces offline physics bitwise on28949/30.016fps. Continuous v35
on28948/PID30992/GPU204 passes142,811PFNN000physical substeps/0falls or limits;
browser29.9933fps/0drops/errors. Explicit DevGPU scope, normal-power failure
retained. First visualP95=222.5ms/body onsetP95=919.6ms remain above target.
Episode2 Ready for Xbox,0training/filter. Original full-terrain quality remains failed.

Latest: [Online Contract Diagnosis](mb_offline_online_gap_20261020.md). The
fixed-reference video and Xbox path differ in observation/history semantics.
New fixed-state observer-mode comparison yields22.243degmedian maximum leg
history change;81/303plans mix prior identities. Prioritize a consistent causal
observer and timestamped plan continuation. Zero additional training or physics;
no live release. All previous numerical quality failures remain recorded.

Latest: [Zero-training Retarget Repair](mb_g1_retarget_repair_20261020.md).
Three exact-source/initial-state frozen-Proto comparisons complete16s each.
Reference extension singularity and ground penetration repaired on this flat
case. Support candidate actual penetration4.065mm/jitter energy-8.60%/planar
error3.625m, but lateral drift worsens, solveP99=289.979ms and6cm-dip stress
still fails. Hold deployment; no new30fps/Xbox approval, training or filter.

Latest: [Continuous Reference Isolation](mb_continuous_reference_isolation_20261020.md)
completed with frozen models. Native MB Root is continuous; stage2 retargeting
amplifies near-straight thigh orientation changes and permits2.328cmreference
ground penetration.16s Proto audit passes15900substeps, but motion deviation and
3.840mRoot drift remain. Prioritize reference mapping/contact feasibility.
Training/exposure totals and live baseline are unchanged; no quality approval.

Latest: severe gait oscillation remains unacceptable. Rejected v31 Root residual
removed; v32_reference_only109016/GPU150500/28948 is the native-controller baseline.
No motor post-processing or new training. [Robust Gait Repair](mb_g1_robust_gait_repair_20261020.md)
records new causal comparisons and the contact-aware reference/single-controller
redesign direction. No stable gait or delivery approval.

2026-10-20: zero-training reference and execution-controller repair progressed.
v30 full300s physical audit PASS; v31_root_task299.66s at29.994Hz, but one post-turn
side command moves against intent. Stable-WASD/delivery approval remains false.
Current experimental28948 parent183916/GPU45100;55588/28947 preserved. No training
or exposure added. [Current Repair](mb_tracker_comparison_20261019.md) has the paired
reference experiment, latency and rejected35%residual. Older entries are historical.

21:25 Xbox/XInput detected on28948; standard two-stick input revised with continuous
deadzone and independent release/rearm.26frontend+42CPU checks pass. Recorded
native/G1/action analysis and12fixed-sensor counterfactuals show large reference
switches feed into controller targets. [Jitter Review](mb_tracker_comparison_20261019.md)
contains literature and next zero-training continuity/control comparisons.
User episode remains active. No stability repair deployed, no new training.

21:05 user control contract fixed: left movement, right facing, with lateral and
backward motion retained. UI-only separation/focus fixes pass25frontend+37protocol
tests and desktop/mobile layout checks. Current v19_feedback/28948 parent136736/
GPU68072 remains experimental: poor strafe/braking, known-map exit at20.921s and
29.9933Hz. Receipt/failure preserved, manually reset to Ready; no new training or
delivery approval. STATUS：`../out/mb_sonic_e0_e3_20261018/task_artifacts/STATUS.md`.

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

Latest MB + SONIC + MuJoCo G1 integration: STATUS：`../out/mb_sonic_e0_e3_20261018/task_artifacts/STATUS.md`, Asia/Singapore. All28944 ports/PIDs and live/current descriptions below are historical2026-10-17 records; use that status page for the latest state.

12:31 authorized zero-training reference-IK validation finished: offline constraints and346CPU checks PASS; full66-frame adaptation about20ms. Online v14/v15/v16 FAIL unchanged200ms input admission, final display29.8Hz. All experimental28948services stopped;28947baselinepreserved.0training updates, no new delivery approval. Reports/failures in STATUS.

Final5688-state audit also found a fall/actual ankle overtravel max0.189897rad despite294legal references and0live QP corrections. Both physical tracking and timing remain failed. Consolidated evidence: `out/mb_sonic_e0_e3_20261018/constrained_reference_final_validation_v1.json`.

Earlier episode0003 stopped54.60s on infeasible future G1 waist reference:original max0.634952rad versus0.52, actual waist0.019736rad. Direct-Chest marker candidate fixed independent clavicle coupling but left8in-window violations (candidate max0.638628rad); NOT deployed. User subsequently approved reference-layer solving; original failure remains retained.

2026-10-19:28948/v13 reference-starvation repair deployed; continuous input no longer invalidates every fresh completed plan. Actual-consumption input deadline,100ms admission and genuine future coverage remain enforced. Latest flat WASD/QE/joystick sample29.818981Hz,0browser failures;313CPU tests pass/7CUDA opt-in skips. Neural weights/math and original terrain-quality findings unchanged;0optimizer updates.

## Historical User Work (2026-10-17)

Parent actualtab8dual-policy handoff complete, recorded in`dual_parent_delivery_v1.json/.png`: ONLINE,execute8/32,preview8,4Pose/4refresh,briefidle30fps/0underruns. User's originalpaused/spawn state restored, actual3m/s/Shift1.6/governorON/preview8preserved; staleURLspeed6not used as state. Final manifest includes these artifacts. No further testing or training follows this handoff.

2026-10-17 latest: gait/startup regression fixed by a declared dual-horizon policy, not by claiming full12/16Posewas trained. Live28944PID115796,007: preview8/12/16, nearRoot/Pose8,actual32frames/4refresh/4Pose. BothRoots use coarse8; real longRootstations enter actor memory alongside nearRootstations. Candidate2sstartup progress12/16=2.348/2.358m vs full-length.213/.149m; sustained1m/sstart.867s, coldmeans2.350/2.289m/s.8bitwise and native/token/FK/skin/history/ACK proofs pass; no output manipulation/training. UninstrumentedP95239/244ms. Explicit full-length legacyCLI retained, no automatic fallback.

UI now prevents horizon selection from overwriting current speed/Shift/governor/camera using stale drafts; same-viewer settings synchronize.6m/s→12/16preservation, truthful32execution plus24Root/40or56preview, and current-scene/stale-session checks pass. Parent owns final user-tab refresh; child review paused. Manifest`dual_horizon_delivery_v1.json`, full rationale/results`pfnn_gait_horizon_review_20261017.md`. Original terrain penetration/quality failures remain separate and unchanged.

Parent actualtab8final readback005/12tokens/40Rootframes/4refresh/CUDA, frame73130.0fps; laterframe1184screenshot shows`Confirming played frames`,28.6/28.7fps,0empty buffers. `horizon_parent_delivery_v1.json/.png`retained in the final manifest. This is near30fpswith residual long-packet waits, not fully stable30fps. Parent user page is running/focused; child review paused.

2026-10-17 latest: generic8/12/16horizon acceleration deployed on28944/PID166916,005/3m/s/12tokens/4refresh;28943/PID60524preserved. One parameterized split framework and shared frozen constants, dynamic current-scene queries, corresponding32/48/64decoder condition frames. Active tokens/FK/skin/history, private horizon+governor prefetched state and map rebind gates pass; maxrollingFK3.46e-6m. Whole-packetP958/12/16=220.96/209.55/212.33ms. BrowserOFF29.74/30.000/29.999fps with0empty buffers but6/0/0ACK wait ticks; ON and continuous-switch records retain their additional waits. Slow requests exceed267ms even atprefix0, so occasional shared-machine/scene-query latency remains. No lowering of4-frame refresh/4Pose passes, no sampling/math change,0optimizer. Parent has actual usertab8ONLINE and performs final UI review; child page paused. Full methods/evidence:`pfnn_horizon_acceleration_review_20261017.md`,source/service manifest:`horizon_delivery_v1.json`. Original geometric quality remains unapproved.

2026-10-17 latest: optional anticipatory target-speed control delivered on28944/PID86712, DEFAULT OFF;28943/PID60524 retained. Terrain panel offers one-click A/B, strength, minimum target speed and recovery acceleration with separate requested/target/played-speed readouts. Existing30Hz,4-frame default,8tokens/4Posepasses,8-frame packets and final weights unchanged. Governor-only config changes are computed in prefetch copies and committed with their candidates, preserving current packets and ACK/history identity. Five real packets/four toggles match serial payloads bitwise; browser4toggles sustain30.012fps over328frames with0newunderruns/0deferred. Final page007/3m/s/4-frame,paused,OFF. Source/service/evidence manifest:`anticipatory_speed_delivery_v2.json`.

The bounded comparison is mixed: forward sole peaks off/balanced/cautious25.77/26.58/29.36cm and times2.285/4.403/5.100s; diagonal29.55/28.96/13.29cm and2.485/4.617/5.895s. All six complete the declared common longitudinal endpoints; lateral progress is retained. This is why the experiment is default OFF. Original2cm quality remains failed. Correct sole supplement explicitly selects196official vertices and asserts denominator; earlier invalid provider-region report is retained as invalid. Full report:`pfnn_anticipatory_speed_review_20261017.md`.0optimizer; no new training or extended search.

2026-10-17 latest: latency engineering delivered on28944/PID186648, preserved28943/PID60524. SplitRootPathExecution performs one compiled Root, native path-aligned scene queries, compiled4-passPose/decoder and official geometry/history. Same007/4-frame complete-sessionP95294.73→144.54ms (20samples,3excluded warmups);8-frame154.00→75.57ms. Controlled browser24physical key-event runs at identical frame64 state/settings show prefix4P95~400ms versus0~495ms; production adapts0/2/4per packet using actual response times, rather than fixing4globally. Final liveWASD interval29.94fps/0underrun with all prefetches at chosen boundaries. The timing endpoint is the CPU return after WebGL submission.0optimizer; no new model-quality approval.

Rolling proof includes four equal active-token sets, maxFK2.83e-6m, actual-motion history<4.5e-7, each candidate's serial/prefetch bitwise equivalence and exactACKhistory. v3cross-candidate local_vel/body feature336difference2.145767e-5is retained. Camera-follow edit removes browser intent/timing noise, not model workload. Real pointer movement was verified through the drag handler with a temporary automation button mapping, then removed. Diagnostic-servingv5stall retained; clean productionv6releases initialization caches and serves normally. Current review page is paused; refresh the user's existing28944page to load the final JS. Complete engineering/evidence review:`pfnn_inference_latency_review_20261017.md`; service/source manifest:`root_path_latency_delivery_v6.json`; browser proof:`root_path_adaptive_live_v6.json/.png`. Original geometric quality and unapproved training budgets below remain unchanged.

2026-10-17: independent predicted-Root observation preview is live on28944/PID36260; baseline28943/PID60524 retained. User requested direct control rather than test work. Post-Root sampling now follows the model's curve for Pose/decoder, while Root retains the original coarse survey. Current weights/codebook unchanged,0optimizer. No test suites, route evaluations or full-map inference sweeps were run. Initial24-frame idle observed30fps/0underruns; user subsequently moved and selected4-frame refresh, with screenshot902frames/19.9fps/1underrun and371.7ms packet generation. User is now controlling the page; do not reset their map/settings. See`root_path_live_v2-browser.png` and runtime readiness/logs. Native neural execution plus compiled geometry is an interactive engineering preview; original training/quality review remains below.

2026-10-17: same-window compact tuning is live on28943/PID60524, with the original final model and all15maps. Actual browser applies/readback, desktop/narrow layouts, collapse/full-width restoration and playback were checked. Final review state:007/2m/s/original24-frame refresh. New source literature and local observations are combined into a12-route plan in`pfnn_response_and_tuning_20261017.md`. The new preference is to investigate effective, fast terrain response, not to start an unlimited training campaign. All listed training budgets remain proposals. Penetration is not solved: more frequent feedback reduces it on explored003scripts but also changes speed/path, and a fresh model prediction can still fail. Original release decision below remains unchanged.

## Historical Map Runtime (2026-10-17)

Latest UI observation after testing: the user selected002at4m/s, switch0.41s,1327frames/30.0fps/0underruns. The user's selected map remains untouched.

2026-10-17: All15original PFNN maps are available on28943/PID195804. The user's reported good control feel is retained separately from the numerical release decision below. Latest request to fix slow switching is completed: one reusable model graph, native current-map observations, startup scene validation/cache, in-place browser switching and browser mesh caching. First-visit switching measured0.33-1.35s and repeated switching0.13-0.21s, versus the previous32-34s. The page is restored to012at4m/s; sampled playback remains30fps/0underruns. No training, model replacement, runtime IK, smoothing or quality-gate change occurred.

All15maps passed native/compiled active-token/FK/full-skin checks (maxFK2.2492363314e-6m<3e-5), and a cached round reproduced every graph output bitwise while retaining the graph. Actual-service evidence: `out/pfnn_mixed_20261015/pfnn_fast_switch_readiness_v1.json` SHA1f73af87c38fab36286df83b4662e1b2d9ea2df33c587f1d12fcfb58bf4fa7b7; browser timings/screenshots under`pfnn_fast_switch_*`.74unique focused tests pass. Original full-route failures remain valid; the additional codec pilot is still unapproved. Historical PIDs and flat-only URLs below describe their review time, not the active map service.

## Training Review

2026-10-16 21:17 Asia/Singapore. **Training complete; motion-quality and final-delivery approval denied.** Local keyboard/rendering performance is usable as an explicitly experimental candidate, not an accepted replacement.

## Completed Budget

| Stage | Final directory | Actual Adam calls | Reference steps | Sample exposures |
| --- | --- | ---: | ---: | ---: |
| Root | full_training_v8/root | 50000 | 50000 | 12800000 |
| Codec | full_training_v12/codec | 20125 | 50000 | 12800000 |
| Pose | full_training_v16/pose | 4000 | 50000 | 12800000 |
| Total | | 74125 | 150000 | 38400000 |

Final Pose checkpoint SHA256: `b960fbe9f792cf46333c23762a8f047259132acd5f6b595bbc726dffa8b40c19`. All protected hashes passed. All formal jobs have exited. The exposure budget is exhausted; no additional formal updates have been made. Larger batches preserve exposure accounting, not the old optimizer trajectory.

The actual-loader thread-local frozen-label graph path passed native4/graph4/fresh2 optimizer comparisons, including pinned loading, preload, validation and saving. v14's earlier failed global capture and v15's lossless recovery remain recorded. No training execution graph enters the actor.

## Acceptance Results

| Check | Observed result | Decision |
| --- | --- | --- |
| Complete terrain routes | 52routes,104rollouts;0full-route passes,4geometry-only passes,101end-plane reaches,3runtime failures | Failed |
| Worst route penetration | 54.05cm against2cm limit | Failed |
| Worst lateral deviation | 1.277m against0.2m limit | Failed |
| Source-history windows | 0/52pass the original combined window gate | Failed |
| Fixed64m flat arena | 1084frames/135packets; no runtime/unknown/uncertain failures;3.05cm penetration | Failed geometry |
| Final-model HTTP/WebSocket | 24frames, serial payload/RNG equal, official18056-vertex rig equal, invalid profile/ack rejected | Passed protocol |
| Final compiled execution | Active tokens equal; max FK8.79e-7m against3e-5m | Passed execution |
| LocalRTX3080 browser | W/A/S/D each3seconds; approximately75seconds/2243frames; sampled29.8/30.0fps;0underruns | Passed observed interaction/performance only |

The original small flat scene failed after604frames outside observed terrain. The original rugged browser scene failed atframe82. Both remain preserved. The64m plane was fixed before generation; terrain was not moved with prediction. No motion repair, smoothing, skipped frames or relaxed thresholds were used.

Experimental local candidate: `http://127.0.0.1:28943/`, PID167808 at review time, paused in IAB tab12. Artifact `out/pfnn_mixed_20261015/final_acceptance_flat_v1`; compact weights SHA256 `0f5d8e2b05b8ec518a104a4979cca70f23efa64080abfc843f874634de5a9646`. The existing28942/28941/28936servers and user pages remain unchanged. Browser evidence is `local_flat_wasd_v1.jpg` and `local_flat_wasd_v1-trace.json` under the same output root. Sampled FPS and frame advancement do not prove all-frame timing or full-route physical quality.

## Failure Evidence

Five frozen-model stair/flat windows compare the actual actor with explicitly offline source-token/source-Root codec reconstruction. Stair RootY maximum errors are6.96,18.23and10.45cm. Source reconstruction improves leg accuracy relative to the actor, but down-stair source reconstruction still penetrates4.25cm. The source truth used in this diagnostic never enters the actual actor. Encoder-direct versus index-decoder differences are below5.49e-6.

A second diagnostic used three freshly idle A100s and64train plus64validation windows per domain,384total. It reused completed caches, chosequarter0, deduplicated source-SHA/start and excluded source windows from the periodic128-example validation. Weights remained unchanged;0optimizer updates. Windows may overlap or share a recording, and this is descriptive sampling rather than an independent recording survey.

| Domain | Codec leg MPJPE train/validation | Pose native loss train/validation |
| --- | ---: | ---: |
| PFNN | 2.42/2.75cm | 0.449/1.819 |
| PARC | 2.64/2.77cm | 0.985/1.457 |
| BONES | 2.23/2.12cm | 0.627/0.703 |

Codec reconstruction error is present on training examples as well as validation examples. PFNN/PARC Pose prediction also has a train-to-validation gap. This supports investigating both reconstruction and generalization; it does not prove an architectural capacity limit, a large-batch causal effect, or that additional updates would solve the gates. The second diagnostic does not compute skin intersections.

## Native Objective Review

2026-10-16 21:35. The final codec objective uses reconstruction1, commitment0.02, contact-skate0.01 and joint-velocity2.0. Root uses global-root2, local-root1 and token-count1. Pose uses masked-token cross entropy. Codec/native-stage source hashes match the formalv16snapshot; this review found no source drift in those files. The codec objective does not directly evaluate static-terrain full-skin penetration or enforce a2cm maximum.

`final_codec_objective_v1` reuses the same384frozen-model windows for a new output-gradient measurement, without resampling or rerunning route/skin tests. Previous codec losses replay exactly; leg metric replay differs by at most1.37e-9m. It differentiates only a detached reconstruction tensor at its unchanged value. Model parameters remain frozen with no gradients; all weights retain their hashes, and0optimizer updates are performed.

Across the six domain/split batches, Root features account for96.6%-97.6%of the squared gradient norm in the normalized reconstruction-output coordinates. The native gradient cosine with leg-MPJPE gradients is0.0110-0.0155; with relative-foot-Y error gradients it is0.0045-0.0092. This is a coordinate-dependent local output-space observation. It does not measure the decoder parameter Jacobian, Adam preconditioning, future training outcomes, or the cause of the large-batch quality failure. Commitment is constant with respect to this detached output leaf; actual encoder/codebook gradients were not measured. No output is moved along these directions and no actor path changes.

## Next Decision

An executable, bounded proposal is now prepared in `pfnn_codec_repair_pilot_proposal_20261016.md`: two decoder-only arms,250updates/256000exposures each, native versus native-plus-FK objective. Both zero-update GPU preflights and10local scope/budget tests passed. The512000additional exposures and the auxiliary loss require the user's specific approval; no pilot training has started. This is the next decision, not another request to repeat the completed diagnostic campaigns.

The completed candidate cannot be promoted. Preserve it as the fixed baseline, including all failures. Further zero-update source/conditioning/loss review remains within the existing scope; do not repeat completed datasets, audits, optimizer-equivalence proofs or final rollouts without a changed dependency.

A new training experiment needs a separate explicit budget and policy decision. Its proposal should first identify whether the codec reconstruction residual is reducible under the unchanged native objective, then address Root/Pose generalization and generated-history recovery. Validation data must remain separate from optimization, and the held-out route failures must not become training examples or checkpoint-selection targets. If encoder/codebook weights change, freeze and verify the new codec before rebuilding Pose labels and embeddings; old labels cannot be reused. A decoder-only change still requires an independently versioned candidate and full acceptance. No proposed experiment is running, and no quality improvement is claimed.

## Evidence

Original reports are extracted under `out/pfnn_mixed_20261015/`:

- `final_results_v1-small.tar.gz`: SHA256 `64a8734282f12dce546adf70347e3f69a9bd90aef12886518d7cece45c53dee5`; completed budgets, migration, actual-loader proofs, routes, flat arena, protocol and codec ceiling.
- `final_domain_diagnosis_v1-small.tar.gz`: SHA256 `29e9e736f08119cc0ce481c8da18f99233db63d4f8b610889600a8dbad937fde`; allocation records, all384window results and diagnostic source.
- Diagnostic source SHA256 `75d172bf16e9fc25a4a91c36595f680c0ee158ef6c3b7a7cb8b998890c3b165f`.
- `final_codec_objective_v1-small.tar.gz`: SHA256 `dcdfbd7377edbd6b2343b6bb8335b5e1d763fccb03e2d4c554414eac9d9003b6`; allocations, exact sample replay, loss components and output-gradient measurements. Diagnostic source SHA256 `20fd7cd38febf204980f94be17fb4182128ce1c8831b9b7e5c0df6999d06d69b`; summary `final_codec_objective_v1-summary.json`.

Keep the30-minute quiet follow-up active until final quality and local delivery actually pass. An idle card is an opportunity for necessary work, not a reason to repeat completed computation or allocate unused memory.
