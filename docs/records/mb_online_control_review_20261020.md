# Online MotionBricks / G1 Control Review

Review date: 2026-10-20, Asia/Singapore. Scope: code, saved failing rollouts,
two frozen-actor CPU counterfactuals, and primary research papers. No new MB
inference, physics rollout, GPU allocation, optimizer update, or deployment.

## Decision

Repair reference semantics before tuning the tracker or optimizing its motors.
The earliest large speed in the reviewed failure is introduced at a source join.
That joined trajectory becomes the next generator history, creating a feedback
loop that magnifies the error. Protecting already-observed G1 targets fixes one
interface, but does not make the remaining source history coherent.

The intended decomposition is: latest command + measured robot state + immutable
accepted reference -> executable continuation -> frozen tracker -> physical state.
Each of these has its own identity and time. An observation must not silently
become a forecast, or a coordinate registration become an instantaneous motion.

## Findings

### P1: Re-registered history is spliced into an unregistered source

`motionbricks/mb_sonic/reference_stream.py:49` transforms the generator context
using the discrepancy between measured Root and the prior trajectory. Its output
is then joined directly to the prior source at line139. `source_join`, line22,
concatenates arrays without establishing a common anchor or a continuous boundary.
The transform changes positions and world rotations of the incoming segment;
retaining the old prefix does not transform its endpoint into that new frame.

The anchor also mixes translation from the raw SOMA source, line59, with yaw
from the optimized G1 reference, line68. These trajectories can differ after
retarget optimization, so their combination need not describe one rigid frame.

Saved v36 plans establish the order of failure:

| Adoption time | Command speed | Input-history peak | Native block peak | Source join peak | Optimized G1 peak |
|---|---:|---:|---:|---:|---:|
|1.82s|0.0781m/s|0.7143m/s|0.5818m/s|10.0328m/s|0.9454m/s|
|2.46s|0.2882m/s|11.7359m/s|7.3266m/s|57.2513m/s|11.9658m/s|
|2.52s|0.3076m/s|17.1560m/s|9.6716m/s|319.5605m/s|25.0360m/s|

At1.82s the join is a13.377cm displacement over13.333ms. The incoming native
block itself remains below0.582m/s. The subsequent history and generation then
become large. The original bad input cannot be repaired by changing the tracker.

Evidence: `out/mb_sonic_e0_e3_20261018/simulated_xbox_preview_v36/frame_review.json`.
All47adopted numeric records are checked against their recorded SHA256.

### P1: Generation is conditioned at the wrong side of its protected prefix

At `reference_stream.py:123`, a new command is generated from history ending at
the current physical time. The source before current+260ms is then retained,
and G1 targets through current+280ms are committed. Generation is not conditioned
on the state at this future handoff. Removing its first260ms does not turn the
remaining segment into a continuation of the retained reference.

This is especially harmful with continuous sticks: every new command sequence
takes the replacement branch at line109, even for small velocity increments.
The ramp repeatedly replaces the editable tail while preserving an incompatible
prefix. Existing tests use `FixedNativeForecast.generate` at
`scripts/fixed_mb_source_provider.py:26`, which uses history timestamps but ignores
its pose and heading values. Those tests verify packet continuity; they cannot
expose the autoregressive corruption observed here.

### P1: Root motion has penalties but no feasibility constraints

`trajectory_reference.py:173` gives the first six coordinates infinite bounds;
line177 selects only29joints for velocity constraints. Acceleration/jerk penalties
are soft costs. The Root-relative mode additionally removes Root/body coupling
and the deployed config has no foot-ground/support task.

The saved v36 reference reaches52.072m/s Root speed,143.928rad/s Root angular
speed and43.88degrees tilt, while all actual joint/motor limits remain satisfied.
The implementation solves its stated QP, but that QP is not an executability test.
The SO(3) principal-log chart also needs a branch-crossing stress test before
larger turns are admitted; a coordinate derivative is not automatically a physical
angular-velocity constraint.

### P1: The pinned tracker cannot directly regulate requested Root displacement

`protomotions_policy.py:109` supplies current torso orientation, joint state,
local angular velocity, previous action, and four future joint/torso targets.
There is no desired Root XYZ, planar velocity, terrain or contact-force input.

Two real CPU calls to the frozen ONNX held all sensors, action history and
joint/torso targets fixed. Translating the reference by[100,-30,2]metres and adding
[50,20,3]m/s to its linear velocities produced bitwise-identical inputs and motor
targets. Consequently, limiting reference XYZ alone will not directly change
this actor. It matters indirectly through future generation, body pose, contact
geometry and the torso targets that the actor actually consumes.

Evidence: `simulated_xbox_preview_v36/root_observability_review.json`.
This also corrects an overly simple explanation of the previous failure: large
Root linear speeds reveal a broken upstream contract, but are not themselves
passed to the actor as a linear-velocity demand.

### P2: Plan association is being reported as command effect

`runtime.py:2489` marks a new command consumed when the plan ID changes, even if
all actor-visible targets are still from the immutable old prefix. The use
receipt at line2575 labels this input-to-action. It measures association with a
new plan, not the first change attributable to that command.

Keep separate timestamps for receipt, planner selection, adoption, first affected
actor input, first motor-output difference, measured motion and presentation.
For causality measurements, replay the same physical/controller state with the
old and new command rather than attributing existing gait motion to a new stick.

## What The Papers Actually Support

The following primary pages were retrieved for this review. Paragraph caches are
in `tmp/reference_review_20261020/`; claims below are paraphrased.

|Work|Relevant mechanism|Consequence for this project|
|---|---|---|
|[MotionBricks, Sec7.4 / Appendix C](https://arxiv.org/html/2604.24833v1)|Its G1 application retargets data offline before training. Robotics replans at10Hz or command changes; the replanning ablation recommends3-9frame intervals and reports degradation from excessive replanning.|Our SOMA checkpoint plus online G1 conversion is a different deployment path. Continuous analog changes need bounded planning that keeps the newest command and respects phase;100Hz input does not require100full generations/s.|
|[RTC, Sec3](https://arxiv.org/html/2506.07339v2)|The new chunk is generated with the retained prefix as a condition. Frozen actions plus weighted overlap promote a compatible strategy during asynchronous execution.|Our protected-prefix QP is only part of this principle. For discrete MB, start generation from a coherent handoff context; adding output blending does not implement RTC.|
|[MGMT, SecIII-C / IV-B2](https://arxiv.org/html/2604.17335v2)|Generation uses past robot states, and the tracker is adapted with the frozen generator running in the loop. The tracker also observes terrain. Directly combining separately pretrained parts is a tested weaker baseline.|Once software contracts are correct, a remaining distribution mismatch should be quantified. Closed-loop tracker adaptation is a credible later intervention, with a separate training budget.|
|[BeyondMimic, tracking formulation](https://arxiv.org/html/2508.08241v4)|Relative body tracking, an anchor error, carefully matched actuation and minimal regularization are explicit. PD setpoints shape torque and need not resemble a smooth pose curve.|Use the exact observation/action semantics of the chosen checkpoint. Preserve one coherent controller; optimize references or train the policy rather than filtering its final torque-producing commands.|
|[OmniRetarget, SecIII](https://arxiv.org/html/2509.26633v1)|An interaction mesh preserves body-object-environment relations with joint, velocity, contact and collision constraints.|Reference feasibility must include supporting feet/hands and surfaces, not just Root-relative limb fits. Full offline retargeting is suitable for data; runtime uses a small active contact problem.|
|[PHP, motion matching](https://arxiv.org/html/2602.15827v1)|Transition queries include foot positions/velocities and Root velocity; significant command changes trigger matching. Contact-rich skill execution preserves an appropriate entry/exit phase.|Phase/contact compatibility can be a candidate-selection feature. This motivates an independent motion-matching baseline, not a hidden fallback or automatic skill switching.|
|[DartControl, Sec3 / Appendix D](https://arxiv.org/html/2410.05260v3)|Motion primitives have explicit overlapping context and local frames. Training progressively exposes the generator to its own history distribution.|A later native-G1 generator should train on consistent generated/executed histories. It does not justify feeding discontinuous synthetic histories into the current checkpoint.|
|[Ruckig](https://arxiv.org/html/2105.04830)|Online trajectories connect complete position/velocity/acceleration states with hard velocity, acceleration and jerk limits.|Useful for feasible intent/Root goal generation before Pose, or a simple comparison. It does not model balance or support forces, so it cannot be the complete humanoid controller.|

## Architecture Options

### 1. Repair The Existing Pipeline First

Keep one immutable native30Hz time origin. A replan chooses a handoff time on that
grid after committed motion; build its four context frames at that handoff.
Generate the continuation from those frames and discard only duplicated context.
The G1 reference is sampled on50Hz once from that coherent trajectory.

Represent coordinate mappings explicitly. Positions, rotations, vectors, commands
and terrain queries must use the same mapping. Keep accepted world targets fixed.
Measured tracking error becomes a future planning condition; do not translate
one new fragment back onto actual Root and concatenate it with an old world path.

First compare persistent native-frame continuation against the current rebasing,
on the exact recorded raw-axis trace. This isolates the software defect with no
new trained components. Actual feedback remains separately recorded and its
effect on scene queries/goals is explicit in each comparison.

### 2. Put Root Feasibility Before Pose Generation

The code already exposes `_predict_root_trajectories` then `_predict_pose_tokens`
in `motion_inference.py:177`. A bounded Root/heading goal can condition Pose before
decoding, allowing the body to adapt to braking and turns. Changing Root only
after Pose has committed its gait forces the reference solver to reconcile two
different motions.

Use initial reference velocity/acceleration, current measured tracking error,
terrain clearance and current support phase. Derive limits from robot dynamics
and successfully tracked references. Preserve raw requested intent and raw MB
prediction for comparison. This is reference planning, not display smoothing.
It is an architectural candidate requiring same-input numerical/physical trials;
it is not installed by this review.

### 3. Small Contact-Aware Reference Planner

Use a single whole-horizon optimization over Root, joints and active support
conditions. Preserve already-consumed targets; choose feasible release/placement
times for the next foot. Couple torso orientation/COM motion with supporting
feet, and retain original motor/joint/geometry limits. Reuse sparse structure
and warm starts. Benchmark this planner separately before it can enter the loop.

A reference governor may reduce achieved speed near difficult terrain, as the
user previously proposed, but must retain the original requested speed in logs.
It should solve a feasible motion goal instead of blindly scaling motion playback
or adding a second motor controller. Balance and direction progress remain goals.

### 4. Native G1 Generation And Tracker Adaptation

The long-term clean interface is G1 motion in / G1 reference out, with real G1
state history and shared scene geometry. Retain the terrain-aware MB backbone
and useful assets, but adapt representation/tokenizer/pose heads and train with
robot trajectories and measured execution feedback where needed.

First establish whether the frozen tracker can follow clean continuously varying
G1 references. If it cannot, test tracker-only adaptation as in MGMT before joint
generator/tracker training. The current ONNX input cannot acquire new Root/contact
conditions merely by appending fields. This route needs new approved training;
this review has performed none.

### 5. Independent Control Baselines

Motion matching plus the same tracker, a direct velocity-conditioned G1 policy,
and a model-based whole-body controller are meaningful separate comparisons.
They answer whether online MB provides useful motion diversity and terrain
reasoning after control costs are included. Do not combine independent motor
corrections from these approaches with the frozen actor and call it one policy.

Small candidate sets plus a frozen critic/short private rollout could reject
untrackable references, but serial multi-rollout search is expensive on this
machine. Treat it as offline data generation or a measured batched candidate,
not the first repair for a coordinate bug.

## Ordered Execution And Evidence

1. Reproduce the frame/time defect with saved same-world motions and known rigid
   transforms. Verify position/rotation/velocity invariance and one native grid,
   including actual/reference offsets and yaw differences. Cover the contaminated
   history feedback path, not just the fixed-video provider.
2. Repair the handoff/context contract, then run the same continuous raw-axis
   trace through MB generation alone. Compare seam derivatives, context errors,
   token changes and native reference direction progress.
3. Retarget that exact saved continuous reference to G1. Compare reference
   Root/joint dynamics, support-foot motion, body clearance and original bounds.
4. Replay the exact G1 reference with the frozen tracker and same initial state.
   This isolates downstream limitations from live planning and CPU/GPU timing.
5. Reintroduce actual feedback, then asynchronous planning and browser input.
   Keep phase-aware scheduling bounded while polling sticks at100Hz, tracking
   at50Hz, integrating at1kHz and rendering at30fps. Report causal input effect,
   direction progress, jitter, slip, failures and normal-power latency.

Stop expanding the runtime optimizer if clean-reference tracking already fails;
that result identifies the tracker/data interface as the next architectural task.
Do not spend another large trial budget masking impossible references. Faster
inference and TensorRT/export work follow a coherent, trackable motion stream.

## State At Review Close

- Additional work:47saved-plan frame audits,2frozen CPU actor calls,8primary papers.
- New training/physics/GPU model work:0.
- Runtime source changes in this review:0; diagnostic scripts and this review added.
- Local process/port scan: no Python services or listeners on28947/28948/28949.
  Earlier PID22856 is no longer live. Saved comparison artifacts remain intact.
- Stable continuous Xbox control and full-terrain quality remain unresolved.
