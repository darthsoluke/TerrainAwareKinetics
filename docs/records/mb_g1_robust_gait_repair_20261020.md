# Robust MB / G1 Gait Repair

## Decision

The user rejected pure output post-processing. No EMA, visual pose smoothing,
output acceleration clamp, Root projection or new training has been deployed.
The v31 instantaneous Root-velocity residual is rejected for gait quality and
removed from the active configuration. v32_reference_only is a research baseline
on28948, parent109016/GPU150500; SONIC55588/28947 remains unchanged.
It is not a repaired-gait delivery. The new user's native-baseline run reached
24.46seconds and then hit the unchanged200ms reference-input deadline.155inputs
were received and five references were discarded for freshness. The deadline
problem therefore also exists without the Root residual; it is a separate gate
for the redesign. The failure is preserved, not converted to a held reference.

The selected research direction is contact-phase-aware short-horizon reference
planning plus one coherent physical controller. MotionBricks supplies the motion
prior. Actual G1 state, support forces and geometry constrain what is executable.
A second independent velocity controller must not overwrite the tracker's action.

## New Evidence

Analysis is zero-training. Existing immutable NPZ records are hash-checked.
At50Hz, the first v31 forward window has maximum-per-joint target-step P95:

| Stage | Radians per20ms |
| --- | ---: |
| Current G1 reference | .07791 |
| Frozen actor PD setpoint | .63737 |
| Applied setpoint after Root residual | .75247 |
| Root residual alone | .30000 |

These are command signals, not actual joint-angle jumps. With4second Welch
windows and.25Hz resolution, summed above8Hz setpoint power rises from.10357rad2
before the residual to.20625rad2 after it in this same recorded window.
The residual has no temporal state or planned contact transition: its optimum
can change sharply when the measured state/contact-dependent dynamics change.

A new same-reference physical comparison penalized the residual's first and
second temporal differences inside its QP. This candidate still raised actual
joint jerk in the four tested directions; backward progress changed sign.
It was rejected and its source archived, not installed in the service.
Smoothing the added residual is therefore not the chosen repair.

Twelve fixed-sensor/action-history ONNX counterfactuals isolate reference changes
at steady-input adoptions. Native recorded actor outputs reproduce exactly.
Old/new futures change the largest setpoint by median.04795rad, maximum.17137rad.
The large every-tick oscillation therefore needs investigation beyond adoption
continuity, which was the dominant issue in the older v19 evidence.

A separate reference-constraint probe added an interior.04rad joint margin.
All21plans through8seconds remained unchanged relative to their baseline, with
zero committed-prefix error. The margin is inactive in this tested window and
has not been deployed as an anti-jitter fix. Its21solver timings haveP99=18.61ms;
these are offline reference optimization timings, not total online latency.

The real Xbox run failed at75.62seconds on the existing200ms reference-input
deadline. Its3778NPZ/75560controlled substeps pass the physical audit, with no
joint/force violation, auxiliary force or fall contact. Client display is30.016Hz
with zero dropped samples. Oscillation and the timing failure remain failures.

## Literature And Implications

### Retargeting And Executability

[GMR / Retargeting Matters](https://arxiv.org/abs/2510.02252) evaluates how
retargeting artifacts affect tracking robustness. Our feasibility test must cover
G1 body geometry, support-foot motion and dynamically achievable reference changes,
not only per-joint range checks. Correct source motion and coordinate conversion
remain necessary inputs to this test.

### Preserve The Learned Feedback Contract

[BeyondMimic](https://arxiv.org/html/2508.08241v4) explains that PD setpoints are
intermediate variables shaping torque, rather than kinematic poses to follow
exactly. Previous action and measured state also inform the realized dynamics.
Changing these commands changes the feedback loop. This supports testing a
single control formulation instead of filtering or adding another feedback output.
The pinned [ProtoMotions deployment](https://github.com/NVlabs/ProtoMotions/blob/f69fdcaabff31846b4d2247207417b57dc788828/deployment/test_tracker_mujoco.py)
has optional post-filters, but the installed checkpoint metadata has neither
enabled. We retain that native baseline.

### Contact Transitions Belong In Planning And Control

[TOWR+ / IHWBC](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.712239/full)
jointly considers motion/contact phases and uses continuous task/contact-force
transitions in whole-body control. We adopt this contact-management principle,
not its full trajectory optimizer as a drop-in online component: the paper
identifies receding-horizon computational speed as an outstanding issue.
Our implementation must benchmark a small warm-started problem on this machine.

### Training Smoothness Is A Separate Lever

[CAPS](https://arxiv.org/abs/2012.06644),
[L2C2](https://arxiv.org/abs/2202.07152) and the recent
[L2C2 redesign](https://arxiv.org/abs/2606.13169) address learned-policy smoothness.
They are relevant to a future tracker adaptation experiment, not an inference
flag that can retrofit the frozen checkpoint. No optimizer work is authorized
or executed in this repair.

## Implementation Sequence

1. Keep the published frozen tracker as the unmodified control baseline. Remove
   the rejected Root residual and preserve all failed trajectories and receipts.
2. Diagnose the interface using identical G1 references, initial states and
   commands. Separate current/future reference, actor setpoint, motor torque,
   measured joints and actual contact forces. Include original deployment and
   numerical joint-stop behavior in the physical-control comparison.
3. Build a support-state representation from grouped actual foot forces,
   relative slip, measured motion and planned gait phase. Distinguish touching,
   bearing load and releasing support. Preserve raw factual history separately;
   do not invent future contacts or erase contact chatter from the evidence.
4. Optimize the short reference horizon as a coupled Root/body/foot trajectory
   with a support-transition plan, original limits and static geometry. Use MB's
   motion as the warm start and prior, not as an independently overriding command.
   Check candidate executability through the same frozen tracker in private
   physical rollouts before changing the active reference contract.
5. If execution-side redesign is required, formulate body tracking, momentum,
   contact consistency and actuator limits in one whole-body decision problem.
   The neural output may be a nominal objective in that problem; do not implement
   another independent post-hoc delta on its final motor command. Compare this
   explicit controller against the native tracker as separate backends.

## Gates

- No added training, changed mechanical ranges, increased torque limits,
  hidden external forces, actual-state edits or future ground truth.
- Preserve independent translation/reverse/strafe and yaw. Quiet standing under
  a movement command is a control failure.
- Compare absolute high-frequency energy and measured joint/torque derivatives
  at matched commands and progress, with foot slip and contact quality. A smaller
  fraction caused only by a changed spectral denominator is insufficient.
- Measure input consumption, reference updates, motor action, physical onset and
  browser presentation separately. No additional playback queue. Maintain30fps,
  native50Hzcontrol/1kHzphysics and real-time simulation.
- Benchmark the proposed small execution solve against a4msP99 engineering
  budget before integration. Current full MB-chain latency is still too high;
  the future architecture is not admitted merely because its QP is fast.
- Validate continuous direction changes and support transitions before a new
  user-test handoff. Do not reuse the flat baseline as all-terrain approval.

## Evidence

Under out/mb_sonic_e0_e3_20261018:

- control_oscillation_v30_spectral_v2.json and control_oscillation_v31_spectral_v2.json
- trajectory_v31_switch_counterfactual_v1.json
- root_velocity_temporal_paired_v1.json and rejected_temporal_root_v2/source_manifest.json
- reference_safety_margin04_replay_v2.json (joint/Root XYZ changes exactly0)
- root_task_v31_user_jitter_client.json and root_task_v31_user_jitter_failure.png
- wasd_protomotions_v31_root_task/episode_0002_000_audit_v1.json
- protomotions_live_launch_v32_reference_only/{launch,source_manifest}.json
- v32_config_verified.json and v32_freshness_failure_review.json

Scope: measured diagnosis, rejected CPU candidate and baseline restoration.
The robust planning/control redesign is the next implementation, not a completed
or quality-approved feature.
