# Continuous Gamepad Repair

Date: 2026-10-20. Frozen trained MB, frozen ProtoMotions, original G1 mechanics.
No training, motor filtering, physical-state projection or auxiliary forces.

## Implemented

1. Native history stays in one world frame and on its original30Hz time grid.
   A changed command starts from the four accepted native frames at the future
   handoff, after the tracker's committed lookahead. Actual Root feedback places
   scene queries; it does not translate one motion fragment into another frame.
2. Future planar integration starts at the accepted context endpoint. The codec
   reconstructs past velocities with error; previously those errors were included
   in the next published displacement. A constant XZ integration-origin correction
   removes that past error while retaining every generated future displacement,
   absolute height, bone rotation and Root-relative body shape. Raw decoder blocks
   remain recorded separately. No temporal filtering is applied.
3. The interactive package uses native bone material axes for limbs. The package
   had been forcing the old marker-plane mapping despite the earlier extension
   singularity repair. On a v38 sample at46.22s, raw hip-yaw step decreases from
   15.542deg to6.174deg with exactly the same source motion.
4. An explicit optional Root angular-rate constraint was tested in the
   trajectory QP. Eight signed increments bound the L1 rotation-vector increment;
   the SO3 exponential cannot increase its path length. Independent Root/joint
   QPs retain the existing fast decomposition. Tests cover combined-axis changes,
   the179deg yaw boundary, unchanged committed samples and original joint bounds.
   The live2rad/s candidate v40 missed the existing input deadline at35.38s:
   generation29.30ms, retarget46.43ms, total provider91.30ms plus adoption delay.
   Chain P99 reached100.30ms. Its35,300physical substeps pass the physical audit,
   but the latency failure rejects it. The candidate source/config is retained
   under its trial directory; its implementation was removed from runtime.

## Controlled Results

All runs use the same83s simulated Xbox trace through the real browser input
mapping, same1.2m/s speed setting and same fixed64m flat arena. Native generation
remains responsive to continuous translation and independent yaw. Each trial has
its own source/config snapshot and immutable recordings under
`out/mb_sonic_e0_e3_20261018/`.

|Trial|Change|Outcome|Display|
|---|---|---|---|
|v36|Earlier preview-protection baseline|Fall6.935s|30fps|
|v37|Fixed world/time and future handoff|Fall58.104s|29.99365fps,0drops|
|v38|Also corrected planar integration origin|Fall48.104s|30.01412fps,0drops|
|v39|Also used native material axes|187.46s, explicit stop after full trace and idle|29.99118fps,0drops|

v37:58,004controlled substeps audited; the residual decoded-past error reaches
57.3cm at54.52s, yielding29.74m/s at a join. v38 has no source/history/G1 speed
above10m/s; maxima are5.755/5.465/5.368m/s. Its48,004controlled substeps retain
the fall evidence. Lower source discontinuity alone did not stabilize execution.

v39:9,368tick artifacts and187,360controlled substeps pass the full physical
audit. No fall, actual joint/motor-limit violation, hidden force or state write.
Root minimum height is0.66994m. All83s of analog inputs complete, followed by
more than100s of zero input. Development stable-GPU power scope is explicit.

Direction tracking still fails: after excluding the first400ms of each phase,
reverse strafe projects-0.194m/s against the requested direction, and rapid strafe
reversals project-0.087m/s. Stop-phase actual speed averages0.624m/s; post-trace
zero-input speed averages0.156m/s. First visual association P95 is198.3ms. These
measurements prevent a stable-Xbox acceptance claim despite the no-fall result.
The frozen actor still has no direct Root XYZ/linear-velocity target observation.

Evidence: each trial's `browser_receipt.json`, `frame_review.json`,
`physical_audit.json`; v39 additionally has `gamepad_review.json` and
`completed.png`. Phase analysis uses the first input's client/server clock offset
and all50Hz control records. Visual association is kept separate from causal
command-to-body response.

## Verification And Remaining Work

The selected runtime passes376Python tests, including a history-sensitive
generator with a57cm reconstruction-origin error. The timing-rejected candidate
had three additional passing QP tests; these were removed with that candidate.
Do not add actuator residuals or filters to
compensate for the remaining reference/control mismatch. Preserve the original
continuous trace, failed trials, speed requests and zero-training constraint.

The earlier review remains at `mb_online_control_review_20261020.md`. Its primary
references are MotionBricks (replanning), RTC (conditioned committed prefix),
OmniRetarget (reference constraints), and BeyondMimic/MGMT (tracker observations
and closed-loop data). No new trained controller has been introduced.

Service package `interactive_continuation_v41` restores the selected v39 method
and all15PFNN maps on28948. Root-rate experiments are disabled. Direction/stop
quality and normal-power latency remain open; this package is an experimental
control surface, not a stable-gamepad acceptance result.

PFNN000 v41 regression:28.279s/28,179controlled substeps,29.99294fps/0drops.
It stops on `unknown_canonical_ground_domain`, Root[-4.3681,9.8436,0.7519]m.
The full physical audit therefore fails the known-ground/terminal criteria.
No reset or boundary projection occurs within the recorded rollout. The browser
was explicitly reset afterward and restored to normal Xbox/WASD input, with
no simulated input running. All15PFNN map options remain available.

Final live process: server33428, planner11348, port28948, packagev41. Other trial
GPU processes were stopped and their stable-power scopes released. v41 keeps
the explicit development stable-power scope while the interactive service lives.
Every runtime source SHA matches the physically audited v39 package exactly.
