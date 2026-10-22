# MB / G1 Tracker Comparison

## Scope

Zero-training integration of the published ProtoMotions G1 tracker and SONIC.
MotionBricks weights and training exposure totals are unchanged. All15PFNN maps
are selectable; stability is assessed separately from scene availability.

## Current Gait Review: 2026-10-20

Latest investigation: [Offline / Online Gap](mb_offline_online_gap_20261020.md).
Read-only v32 analysis verifies303plans:81mixed-prior histories,19of87adopted.
13matched moving histories reduce median maximum leg step4.026->0.745deg when
the prior identity is held consistent. A12case fixed-observation counterfactual
finds22.243degmedian maximum leg-history change from the stop/move observer
branch alone. The observer and causal history association now take priority.
0MB/actor calls,0physics steps,0training; live28948 unchanged.

Latest repair: [Fixed-Reference Retarget Comparison](mb_g1_retarget_repair_20261020.md).
Native material axes remove the34.558deg bending-plane singularity. Adding
original-foot collision/support tasks produces a ground-feasible reference.
On the identical16s experiment, actual high-frequency body energy improves8.60%,
peak penetration4.672->4.065mm and planar error3.840->3.625m; lateral excursion
increases. The289.979ms solveP99 and known abrupt-dip stress failure block live
deployment.70passed/1strict expected failure;0training/filter.28948 unchanged.

Latest user-requested experiment is complete:
[Continuous Reference Isolation](mb_continuous_reference_isolation_20261020.md).
One16.8s native MB source -> one841sample G1 reference ->16s frozen Proto tracking,
with0online reference switches/0new training. Source Root mean1.150m/s for1.2request.
Retargeting amplifies a1.280degSOMA RightLeg rotation into33.057degraw G1 hip yaw
near a straight leg; the constrained G1 reference still has34poses over2cmground
penetration (max2.328cm). Actual tracking passes15900substeps/795NPZ physical audit
but adds knee deviation and ends3.840m off the reference Root. Fix reference
mapping/contact feasibility first; no actuator post-filter or live fix installed.

User rejects pure post-processing. v31 Root residual is rejected for gait quality
and removed online. Current28948 is v32_reference_only parent109016/GPU150500;
frozen tracker, reference continuity and original physics limits remain. No motor
filter, new training or stable-gait approval. Full research decision and sources:
[Robust Gait Repair](mb_g1_robust_gait_repair_20261020.md).
The user's new native-only v32 run also fails the original200ms reference-input
deadline at24.46s/155inputs/five freshness discards. No deadline was relaxed;
v32_freshness_failure_review.json preserves the failure separately from jitter.

Measured first-forward setpoint step P95:.0779rad reference,.6374rad native actor,
.7525rad after residual; residual step P95=.30rad/20ms. Same-window above8Hz
setpoint power increases.10357->.20625rad2. Residual-only temporal regularization
failed its same-reference physical comparison and was not deployed. Reference
margin.04rad leaves21plans unchanged (joint/Root XYZ difference exactly0), so it
is not promoted as a jitter fix. Twelve fixed-sensor ONNX counterfactuals reproduce
exactly; old/new future swap changes setpoint median.04795rad.

Latest real Xbox run:75.62s/30.016Hz then200ms reference-input deadline failure,
preserved.3778NPZ/75560substeps physical audit passes; this does not pass gait or
timing quality.237scoped CPU tests pass; the new spectral/trajectory tests pass.
No more output-smoothing trials. Next design: feasible contact-aware reference,
support-transition planning, and a single consistent controller.

## Previous Repair: 2026-10-20 (Historical)

The user authorized reference Root continuity and zero-training execution-side
Root velocity/support correction. No model training or new exposure was added.
Current experimental service: v31_root_task, parent183916/GPU45100,28948;
SONIC55588/28947 is preserved. v31 is a research candidate, not stable-WASD approved.

### Reference And Clock Repair

Whole-horizon Cartesian reference optimization now binds Root XYZ/SO(3) and all
29joints to the original joint/velocity limits. An exact absolute-time committed
prefix prevents C0/C1 jumps at adoption. Six-control cubic splines and direct DAQP
keep the optimizer small. The measured-state observer uses the unconstrained
anatomical mapping as its prior, so reference optimization cannot hide actual
orientation error. Independent translation and yaw remain intact.

Background planning runs every60ms; steady-intent adoption every480ms, with new
intent eligible immediately. The adoption phase is anchored to generation time,
not the later adoption time. Physical timestamps use integer native integration
steps without writing MuJoCo data.time. Both repairs retain the original freshness
and real-reference coverage gates. v30 completed299.90s at29.99435Hz/zero dropped
display samples; all14991tick NPZ/299812controlled substeps passed the physical
audit. Weak translational control remained. All v20-v29 failures are retained.

### Execution Velocity Task

The frozen ProtoMotions actor has no Root translation/linear-velocity input.
An explicit opt-in controller now fits a small joint-target correction to measured
whole-robot COM velocity and pelvis yaw, preserving the actor's vertical and
roll/pitch dynamics. A private MuJoCo forward-dynamics copy supplies contact-aware
derivatives. A29-variable DAQP problem keeps the correction within15%of original
torque scale and0.15rad target displacement; the original physical motor limits
remain enforced. Actual qpos/qvel, Root actuation and external forces are unchanged.
The actor's next action-history input contains the target actually applied.
Latest execution intent and the older MB-reference intent have separate receipts;
the original200ms reference-input deadline still applies.

This follows the measured-momentum/contact-control direction of
[Herzog et al.](https://arxiv.org/abs/1410.7284), using
[MuJoCo's skipped forward dynamics](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html)
for the local actuator response. The implementation is a reactive residual task;
it does not yet optimize future foot placement or contact transitions.

Same recorded references and identical initial physical states,4seconds each:

| Command | Native actor m/s | With15%task m/s |
| --- | ---: | ---: |
| Forward | .194 | .291 |
| Backward | .060 | .151 |
| Left | .051 | .156 |
| Right | -.003 | .143 |

All four paired trials retained original joint bounds.800correction solves:
P99=1.858ms, maximum3.948ms. Increasing the residual to35%failed to improve three
of the four directions and was rejected; its source and result remain archived.

Live v31 completed299.66s:32seconds of keyboard translation,4seconds independent
yaw, remaining time idle/braking;29.994Hz/zero dropped display samples/1.00RTF,
no runtime failure. First forward average.399m/s, backward.159,left.117,right.338,
against1.2m/s requested. After turning, one subsequent left command averaged
-.227m/s along the requested direction. Directional control therefore FAILS.
Body-onset server P95=427.662ms; full MB-plus-control chain P99=95.719ms. These
remain above the research latency goals. The next control experiment must address
contact-phase/foot-placement preview, rather than increase this reactive gain.
After startup,17input changes reached the execution task at median8.907ms,
P99=23.398ms;15rendered input changes had first-visual P95=86.6ms. The all-input
UI first-visual P95=468.7ms includes cold startup. These measurements separate
fast intent consumption from the slower physical body response.

Evidence under out/mb_sonic_e0_e3_20261018:
root_velocity_paired_v{1,2_fraction15,3_fraction35}.json,
root_task_v31_client_full.json, trajectory_v30_client_full.json,
wasd_protomotions_v30_trajectory/episode_0001_000_audit_v1.json,
and immutable protomotions_live_launch_v31_root_task/source.
The combined hash-bound review is root_task_v31_review.json.
The completed v31 audit verifies14978tick NPZ and299560controlled substeps:
original joint and motor-force limits pass, all auxiliary forces zero, no fall
contact or solver warning, all ground queries known. Maximum scene contact
penetration is9.334mm on this flat scene. No terrain or stable-control approval.
333Python regression tests pass,3optional CUDA tests skip;24frontend protocol
tests pass with Node's --experimental-vm-modules flag. The legacy retarget
equivalence test now compares nested numeric metadata using NumPy equality.

## Jitter Review: 2026-10-19 21:25 (Historical)

Latest service is v19_feedback on28948, parent136736/GPUchild68072. Xbox standard
mapping is now observed in the browser: `Xbox 360 Controller (XInput STANDARD GAMEPAD)`.
Left stick controls translation; right horizontal axis controls independent yaw.
Standard connected pads are enumerated even when slot0 is empty. Radial translation
and scalar yaw deadzones are continuous from0at12%to full scale, with no temporal
filter. A newly attached controller must return neutral before driving. Input
state is included in measurement receipts.26frontend and42CPU analysis/protocol
checks pass. The user resumed a physical run; it was not interrupted for research.

### Recorded Evidence

Two closed v19 episodes were analyzed without new simulation or model updates:

| Measurement | Episode1 | Episode2, latest user run |
| --- | --- | --- |
| 50Hz physical/control samples | 1042 | 3043 |
| Hash-verified plan/tick NPZ files | 1337 | 3892 |
| Reference adoptions | 294 | 848 |
| Adoptions with unchanged consumed command | 284 | 817 |
| Median actual adoption interval | 80ms | 80ms |
| Median maximum future-joint change, same absolute time | 1.658rad | 1.051rad |
| P95 maximum future-joint change | 2.648rad | 2.501rad |
| Median foot-target change, same absolute time | 12.96cm | 11.73cm |

The future-joint statistic takes the maximum across29joints and the tracker's
four genuine future offsets20/40/80/160ms, then summarizes across adoptions.
It is not the change in the physical joint angle. In episode2, native SOMA local
rotation changes already have median maximum0.884rad at the same absolute time,
before G1 retargeting. This locates an upstream discontinuity as well as the
downstream response. Native point-coordinate changes at adoption are much larger
than the within-plan changes. The constrained retargeter modifies586of849used
plans, with a maximum single-coordinate adjustment4.684rad.

Current constraints enforce original position/velocity bounds but not acceleration,
jerk, contact-phase continuity or consistency of the entire future target window.
The37rad/s motor velocity allowance can permit0.74rad in20ms; that is too abrupt
for a natural reference even though it fits the motor's velocity bound. Forward-
difference velocity features expose those rapid changes to the neural tracker.

A counterfactual actor experiment selected12unchanged-command switches distributed
through episode2. It held all actual sensors and previous actions fixed and changed
only future joint positions/velocities/torso orientations from the new plan to the
previous still-covered plan. Original target replay is bitwise equal in all12cases.
Changing this reference alone changes the maximum PD joint target by median1.347rad
(77.2deg), P952.944rad. This is a direct reference-to-action sensitivity result;
full physical improvement still requires a separately executed candidate.

### Literature And Engineering Decisions

1. **Asynchronous chunk continuity:** [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/abs/2506.07339)
   fixes actions already committed during inference and regenerates the editable
   future. Adopt the time/commitment principle: one absolute clock, a short measured
   committed prefix, and soft consistency over the remaining overlap. RTC's
   diffusion/flow inpainting is not a drop-in sampler for MB's discrete masked tokens.
   For MB, investigate supported pose/keyframe conditioning and persistent latent
   phase; enforce continuity in decoded G1 reference space at the final interface.
2. **Trajectory retargeting, not isolated feasible frames:** [Retargeting Matters / GMR](https://arxiv.org/abs/2510.02252)
   demonstrates the effect of retargeting artifacts on physical tracking robustness.
   Our first candidate should retain anatomical tasks and original limits while
   optimizing the short trajectory jointly: measured adoption position/velocity,
   stable rotation branches, acceleration/jerk cost, overlap cost decaying into
   the future, and contact-conditioned foot tasks. Generated native input remains
   recorded separately. Only the reference is solved; the actual robot is actuated
   through its existing controller. This is the authorized reference-layer scope.
3. **Policy smoothness:** [CAPS](https://arxiv.org/abs/2012.06644) regularizes temporal
   and spatial action sensitivity; [L2C2](https://arxiv.org/abs/2202.07152) constrains
   local policy/value smoothness. Both require training. The pinned [ProtoMotions
   G1 experiment](https://github.com/NVlabs/ProtoMotions/blob/f69fdcaabff31846b4d2247207417b57dc788828/data/pretrained_models/motion_tracker/g1-bones-deploy/experiment_config.py)
   already has L2C2 plus action-rate penalty. Re-training is not the current step;
   first remove the measured out-of-distribution target jumps.
4. **Published deployment contract:** [ProtoMotions G1 deployment](https://github.com/NVlabs/ProtoMotions/blob/f69fdcaabff31846b4d2247207417b57dc788828/docs/source/tutorials/workflows/g1_deployment.rst)
   specifies joint order, torso orientation, pelvis-local angular velocity,
   processed-action history, reference horizons and PD gains. These are pinned in
   our adapter. This checkpoint uses EMAalpha1and no PD acceleration clamp; a
   missing optional deployment filter is not established as the fault.
5. **Support and Root control:** [Contact-aware Nonlinear Control of Dynamic Characters](https://grail.cs.washington.edu/projects/nqr/s2009/)
   plans through changing support to control an underactuated body. If continuous
   references still cannot follow translation/braking, add an explicit Root velocity/
   momentum/contact-force task through a physical WBC/MPC layer. The current G1
   deploy actor omits Root translation/velocity targets, so additional Root numbers
   cannot simply be appended to its existing ONNX inputs. This is a distinct
   controller change, not another scalar smoothing parameter.
6. **Physics numerics:** [MuJoCo computation documentation](https://mujoco.readthedocs.io/en/stable/computation/)
   motivates checking timestep, damping and stiff actuation separately. Current
   v19 already uses implicitfast/1ms with the pinned G1 gains. Preserve this baseline;
   compare smaller timesteps only if residual oscillation persists under a fixed,
   continuous reference. Increasing damping first would also slow intended response.

### Next Zero-Training Sequence

1. Isolate the four-frame feedback history and native generated overlap in recorded
   unchanged-command cases. Check pose phase, reference identity across history
   frames, rotation gauges and time alignment before changing neural architecture.
2. Build one explicit trajectory-level reference candidate with C0/C1 adoption
   continuity and soft acceleration/contact consistency; retain actual-state feedback.
   Keep the committed prefix no longer than measured unavoidable execution delay.
   Leave later motion free to follow new left/right-stick commands.
3. Replay the same saved native MB packets through the old/new retargeters with
   fixed adoption times, then use the same physical initial states and intent
   schedule for old/new trackers. Measure actual movement and braking as well as
   joint/torque high-frequency energy, jerk, support slip and falls.
4. Hold1000Hzphysics/50Hzcontrol/30Hzdisplay, original joint/force/penetration gates
   and existing100/200ms freshness/input limits. Report solverP99and input-to-body
   response separately. A quiet stationary robot is not the control objective.
5. Use physical Root-task WBC only if direction/braking remains wrong after reference
   continuity. Native-G1 generator/tracker fine-tuning is a later budget proposal,
   after the zero-training interface/control questions are resolved.

Evidence under `out/mb_sonic_e0_e3_20261018/`:
`actual_feedback_v19_jitter_episode{1,2}_v1.json`,
`actual_feedback_v19_switch_counterfactual_v1.json`,
`actual_feedback_v19_episode2_client_full.json` and retained physical JSONL/NPZ.
New optimizer updates0, physical resimulations0for this investigation. Stability
repair is not deployed; this iteration changes only gamepad input handling/UI.

## Same-Reference Replay

The original SONIC failure was replayed from saved MB references/adoption times,
with online generation and planner waits removed. Right-ankle overtravel repeats
at20.135s and pelvis/torso ground contact at20.960s. Reset/warmup is bitwise equal.
The recorded fall therefore persists without generation latency.

The original43DOF asset includes14 uncontrolled finger DOFs. The published
ProtoMotions deployment asset has29DOF and fixed rubber hands, plus different
inertia/armature/ankle tendons/collision details. Root/leg geometry and the14
reference orientations match the admitted calibration; small upper-body origin
differences are explicitly recorded in the asset binding. Both trackers were
also replayed with this same matching asset and the same saved MB numbers:

| Tracker | Duration | Actual joint crossing / fall | Joint RMSE | Control+physics P50 / P99 |
| --- | --- | --- | --- | --- |
| SONIC | 22s | 0 / 0 | 0.104979rad | 24.42 / 43.94ms |
| ProtoMotions | 22s | 0 / 0 | 0.129712rad | 20.79 / 39.64ms |

These timings exclude online MB and browser work. Each tracker retains its own
published observation/action/PD contract. ProtoMotions has lower compute and
requires160ms future instead of SONIC's900ms. SONIC tracks joints more accurately
on this trace. ProtoMotions was selected for the next interactive trial.

## Separate Failure Mechanisms

1. **Reference conversion:** the original SOMA-to-G1 waist could exceed0.52rad.
   Authorized reference-only constraints repair legal position/velocity/boundary
   conditions while preserving raw failed inputs. Physical state is never edited.
2. **Physical execution:** a legal reference can still overshoot actual soft joint
   limits or lose support. The matching robot and declared internal stop response
   address the observed overtravel; original joint ranges/motor bounds remain.
3. **Latency:** repeated2-block generation, unnecessary900ms readiness checks for
   Proto, stale planning lead, per-contact Python work, full JSON tick recording
   and in-process compression each added measurable work or queuing. They are
   handled independently, without extending100ms plan/state or200ms input bounds.

Current execution is one real AR8 block (0.8s/24future frames),0.6s consumer
horizon,60ms replans,1000Hzphysics/50Hzpolicy/30Hzdisplay. Original8tokens/4Pose
passes/4history and all native token/FK/query/skin checks remain. No held tail.
Lossless columnar recording retains every substep/contact, with one CPU-only
compression process and unchanged bounded queues/8GiB storage cap.

## Physical Findings

v9 flat native-key WASD/QE:210.819s,30Hz/1.00RTF. Audit verifies210819substeps,
10541NPZ hashes, zero actual joint crossings/illegal forces/warnings/fall contacts.
Maximum collision penetration4.361mm. v9 rough007105.180s also passed recorded
physical checks, but0.83RTF and stopped on recording storage. These records remain.

v12 rough007 sustained movement is a retained failure:30.0Hz/0.99RTF, then torso
contact at38.859s. All38859substeps and1943NPZ hashes audited. Actual joint
crossings0, solver warnings0, forces within original bounds.647adopted references
have0joint-position violations. Reference/actual planar error exceeds1m at27.361s,
2m at36.501s,3m at38.181s; actual tilt exceeds0.8rad at38.621s before impact.

The deployed G1 tracker does not directly consume world Root XY/height, and frozen
B1 MB conditions on reference history rather than executed G1 pose history.
Accumulated reference/execution drift and physical feasibility remain a rough-
terrain problem even when generation deadlines and frame rate hold.

v12 browser metrics:29.99365Hz, no dropped display samples, chainP9976.69ms,
first-visualP95244.4ms, body-onset-serverP95908.58ms. Original20ms/100ms research
response targets are not met. No all-terrain quality approval is issued.

## Evidence

Final v13 flat000 dense WASD/QE/reversal test also FAILED: torso contact37.200s,
30.01554Hz/0.99RTF, chainP9979.9342ms, first-visualP95289.3ms and measured body-
onset-serverP951025.93ms. All37200substeps/1860NPZ verified.625references are joint-
position legal; actual joint crossings0 and motor-force violations0. Actual tilt
exceeds0.8rad at36.741s; peak collision penetration23.174mm. This final result
prevents a stable-flat handoff despite the earlier successful210s trial.

Current v13/28948 is an experimental investigation page only, reset explicitly
to000. Original SONIC baseline28947 and all prior failures remain. The final UI
keeps JSON exports hidden and clears only the new episode's transient error state.
Stable-WASD approval=false; all-terrain approval=false; new training updates=0.

- `out/mb_sonic_e0_e3_20261018/tracking_isolation_v1/`: pinned assets, same-reference replays and native deployment checks.
- `wasd_protomotions_v9/episode_000{1_000,2_007}_audit_v1.json`: full physical audits.
- `wasd_protomotions_v12/episode_0002_007_audit_v1.json`: recorded rough failure.
- `wasd_protomotions_v12/episode_0002_007_loop_diagnosis_v1.json`: all-tick reference/execution comparison.
- `protomotions_live_v12_007_client_full.json`, `protomotions_live_v12_007_failure.png`: browser receipt and failure view.
- `tracking_isolation_v1/final_integration_cpu_v3.xml`:402CPU tests; final episode-reset change has218focused checks.
- `task_artifacts/STATUS.md`: current live candidate, exact configuration and handoff status.
