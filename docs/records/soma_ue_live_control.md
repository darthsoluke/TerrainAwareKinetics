# Live SOMA control in UE

## Launch

Run `F:\dev\GASP_57\Scripts\RunSomaLocomotion.ps1` to start/reuse the persistent
Conda backend and open `F:\dev\GASP_57\GASP_57.uproject` on
`/MotionBricksGroot/SomaControl/L_SomaLocomotion`. Press Play and click the viewport.

- WASD: camera-relative movement; face the movement direction by default.
- Shift: faster movement (2.5 m/s target); Ctrl: slower movement (0.6 m/s).
- Mouse: orbit camera, with inverted Y enabled in the input configuration.
- Hold right mouse: face the camera direction, allowing backward movement/strafe.
- Mouse wheel: zoom; R: reset camera; F3: show the source SOMA skeleton.
- Escape: stop PIE. The backend remains loaded for the next PIE session.

Jump, crouch, terrain traversal and multiplayer are not implemented by this route.
Other maps, character classes and the old G1 backend are unchanged.

## Model And Execution

The bundle is `out/soma_control_v3/baseline_backend.pt`, SHA256
`995e93ece7539db84bdab50d8fa5decdbbaac1d31b543a041686c4c06ff8fae7`.
It uses the trained MB Root/Pose models and frozen tokenizer on SOMA30. It is not
a G1 policy, animation replay, motion-matching lookup or physics controller.

Enhanced Input updates the latest planar velocity/facing intent. A worker thread
exchanges bounded JSON messages with `scripts.soma_ue_live_server` on loopback
port 28766. The Python process performs sequential GPU inference independently
of the UE game thread. Every plan contains 60 future frames for 60 Hz playback;
this does not mean 60 neural network calls per second.

Each default request uses the last four executed authoritative SOMA frames,
including the actual planar root after capsule collision sweeps. The native
retargeted Manny pose is not fed back as SOMA joint rotations.
Unplayed predicted futures
are never inserted into history. A late plan skips its already-played prefix;
past poses are not rewritten. There is only one outstanding request, with no
command queue. Each new PIE session has a fresh nonce and the bundle's initial
four-frame state; there are no ground-truth resets during a session.

Live token selection uses argmax, not a fresh stochastic sample on each replan.
CharacterMovement applies learned planar root displacement once. Crucially, it
uses differences between adjacent roots within the selected plan, rather than
subtracting the current actor position from that plan's stale absolute root.
A late plan cannot pull the actor back to an obsolete root estimate. The per-frame
velocity is no longer filtered; actual positions are returned in history.
No displacement is synthesized from keyboard speed.
The source pose
is expressed relative to that planar root, then passed through the existing native
SOMA-to-Manny retargeter. The animation proxy copies poses on the game thread and
evaluates that value snapshot on the animation worker. Fixed 60 Hz states are
copied directly without render interpolation. `ABP_SomaControl` now uses Pose
Snapshot -> Output, with no inertialization node or transition requests. The
spring arm follows the capsule and ticks after movement, with camera lag disabled.
There is no foot IK, foot locking, command-based root drift correction or per-frame
ground fitting.
The collision capsule remains at its initial flat-ground height; generated pelvis
height stays in the pose. Collision feedback is not a trained obstacle-avoidance
capability and can expose additional model errors.

## Verification And Limits

On 2026-10-05, the module-only UE 5.7 build and
`MotionBricks.SomaControl.PIE` passed on the RTX 3080/D3D12. The automated test
injected the movement action through Enhanced Input, checked real capsule movement,
retargeted bone lengths and input flushing, then stopped/restarted PIE.
It does not establish that every physical keyboard/gamepad mapping was manually
tested, nor that motion quality or sustained frame rate is approved.

The original, superseded movement snapshot recorded 10 received plans, 176 applied frames,
230.05 cm forward displacement, 14 buffer underruns and a 154.40 ms latest
round-trip time. Underruns hold the last pose rather than inventing motion.
The screenshot proves visible Manny rendering, not natural locomotion. Previous
offline quality failures (sliding, drift and pose discontinuity) remain relevant.
The live protocol/60-frame export is a new execution path; the earlier offline
timing benchmark is not a benchmark of this full UE integration.

Evidence is under `F:\dev\GASP_57\Saved\MotionBricks`:
`soma_control_pie.json`, `soma_control_pie.png`, `soma_live.jsonl` and server logs.
The UE automation report is `Saved\Automation\SomaControl\index.json`.
For rebuilding/rechecking without opening an interactive window, use the launcher
with `-Build -Author -Test -NoEditor`. Numbered module outputs are used so another
Editor/game process can retain its already-loaded DLLs without being stopped.

## Historical Playback Filters (Rejected)

This section records the superseded implementation, not the current route.
Velocity filtering, pose inertialization, render interpolation and camera lag
have all been removed at the user's request. Their numbers do not approve gait.

The first revised idle/forward PIE run recorded 29 plans, 400 applied frames,
295.19 cm forward displacement and zero buffer underruns. Its counterfactual
stale-absolute-root correction peaked at 13.55 cm; actual model-delta steps peaked
at 2.17 cm. These are different measurements within one revised run, not a matched
before/after quality experiment. Native inertialization was requested 28 times;
maximum plan-switch mean angular step was 0.602 degrees before the display blend
and 0.560 degrees after it. Those means do not bound individual joint motion.

The extended PIE test also exercises a right turn, camera-facing backward movement
and stopping. `CameraStability` applies a known 2 cm alternating displacement to a
transient spring-arm fixture, checking camera peak-to-peak below 0.5 cm. This is a
camera-component regression, not a claim about model drift or natural gait.
`PlaybackContinuity` reproduces a late-plan absolute-root reversal, verifies
positive model-delta execution, and checks the velocity filter's timestep
invariance and non-overshooting stop response.

Latest revised evidence uses `soma_control_pie_stable.json` and
`soma_control_pie_stable.png`, separate from the original raw test artifacts.
Python tests: 225 passed. No retraining, contact IK or foot correction was added.

The extended run passed all three UE tests. It exercised 87 plans/751 executed
frames, recorded zero buffer underruns and a 150.00 ms final round trip. Its
counterfactual stale absolute correction reached 32.22 cm while actual per-tick
steps stayed below 2.364 cm. Maximum switch mean angular step was 1.030 degrees
before presentation inertialization and 0.225 degrees afterwards (86 requests).
The separate camera fixture measured 0.1818 cm peak-to-peak for 2 cm input jitter.
These are playback/component results, not proof of stationary feet, natural gait,
low command latency or reliable 60 FPS under arbitrary scene load.

## Source-Based Revision, 2026-10-05

### Confirmed Implementation Defect

`SomaMotionInference` formerly overwrote the first four predicted roots with
observed roots, then differentiated that mixed trajectory. The released MB root
training loss differentiates the entire prediction. A regression with a constant
10 cm prediction bias proves the partial overwrite adds a fictitious 6 m/s step
at 60 Hz (1.2 becomes 7.2 m/s), despite identical future predictions.

The default root stage now preserves the released
`motion_inference._predict_root_trajectories` math, with an explicit batch axis
for its final-velocity gather/scatter (the released vector indexing fails for
batch size greater than one). Single-batch output equality is tested. No position offset,
velocity filter or second published-root correction was added. The legacy stage
is retained only for explicit counterfactual evaluation. `soma_live.jsonl` records
the root stage and separate model-generation/export timings.

The same-weight, same-command, single-seed 10 s raw comparison on takug3 changed
boundary joint-displacement P95 from 50.75 to 42.04 mm. This is a useful but small
improvement, not an animation-quality approval. Source-root Pose-only evaluation
does not depend on this root-stage change.

### Experiments Not Promoted

`out/soma_continuity_20261005` contains two paired 1,000-update trials over all
152 train clips, with 57 fixed validation windows; test clips were not used.
Decoder-only temporal training did not remove discontinuities. Pose training with
native hard-token decoded supervision reduced mean root-relative MPJPE from
87.40 to 73.07 mm, but retained a worst single-joint step of 64.55 degrees. These
weights are not used by UE. The straight-through derivative is an explicitly
biased training surrogate, not a method claimed by the MB or DART authors.

An opt-in v2 playback experiment commits immutable eight-frame blocks, conditions
the next block on their final four frames, and distinguishes committed context
from observed history in the protocol. No new block prefix is discarded. Its
GPU-export PIE run played 640 frames with zero skipped frames but 96 underruns;
backend P99 was about 322 ms against a 133 ms block budget. It therefore failed
the real-time requirement and is NOT the default. Use `-CommittedPlayback` only
for this experiment, not as a purported fix. Future collision feedback remains
delayed until the next observation and is not an obstacle-avoidance model.

CPU export uses exactly the same official FK/math and has a CPU/CUDA pose-equality
regression. It did not solve the deadline problem; its PIE experiment also failed
the end-of-sequence idle assertion. `-ExportDevice cuda` remains the default.
Do not enlarge the motion buffer to hide these failures.

### Latest Functional Test, Not Quality Approval

The unfiltered native-root/default-playback PIE run completed with 94 plans, 775
frames, zero underruns, 164.98 ms final RTT, zero blends and camera lag disabled.
Maximum individual retargeted joint step was still 47.00 degrees, and 762 stale
new-plan-prefix frames were skipped. Thus jitter and asynchronous pose mismatch
remain unresolved. All three UE functional tests passed; they do not certify
motion quality or consistent 60 FPS. Evidence: `soma_control_pie_native_root.json`
and `.png` under `F:\dev\GASP_57\Saved\MotionBricks`.

The final Python suite passed 246 tests, including single/batched native-root
inference, the artificial-velocity reproducer, source-only condition isolation,
frozen-module gradient checks, committed/observed clock separation, and official
CPU/CUDA export equality. Candidate CPU-export PIE failure is retained above;
the three successful UE functional tests refer to the default CUDA-export route.

### Method Basis And Next Development Gate

Sources: [MotionBricks Sections 6.1, Appendix C/G](https://arxiv.org/html/2604.24833v1),
[DART Section 3.1, Appendix C/D](https://arxiv.org/html/2410.05260v1), and the
released MB `root_model.py`, `motion_inference.py`, `full_agent.py` at repository
HEAD (not only our modified working copy).

MB conditions smart locomotion on body keyframes as well as root goals; its
decoder can refine root/body coordination. Our velocity-only body adapter and
provided-velocity decoder are substantive deviations, not an exact reproduction.
The next model revision must restore and test the intended context/target-pose
conditioning across training and inference, using a train-only reference-pose
library at deployment, never validation future body poses. DART motivates decoded
motion supervision and scheduled rollout history; our source-directed boundary
losses are an adaptation, not its exact self-consistency formula.

For every further repair: cite the source mechanism, identify our concrete
deviation, reproduce it, and run a bounded matched comparison. Admit only changes
that improve raw all-body continuity without worsening root tracking, contact
quality or command response. Separately require an actual UE deadline test on
RTX 3080 under rendering load. No IK, foot locking, output smoothing or forced
zero-velocity priors may be used to pass these gates.

## Sparse Body-Goal Trial, 2026-10-05

The next source-backed development step is implemented, but its weights are
**rejected for UE deployment**, not a fixed locomotion backend.

`soma_goal_conditioning.py` supplies four observed frames plus one endpoint pose,
using the released MB constraint/mask interfaces. Root, Pose and the native
conditional decoder all receive that goal. Training may use its training clip's
endpoint as an in-betweening condition. Inference never reads hidden future body:
`predict_local` now takes a separate optional goal, irrespective of training policy.

The runtime reference primitive selects a keyframe from TRAIN clips only, using
heading-local XZ command velocity. Pose canonicalization and placement use the
official representation's yaw transform. The frame is a model condition, not a
played animation or IK target. This minimal velocity lookup is our implementation;
it is not the private UE smart-locomotion blendspace. The selected pose's hip height
is an explicit authored/reference goal, not a validation future height.

On takug3 GPU 2, A (history only) and B (one body goal) each performed 1,000 Root
and 1,000 Pose optimizer updates over all 152 train clips. Both started from the
installed original bundle, with seed 2333, batch 4 and Adam 5e-5. Sampling digests
match (`f5207861...`); the entire tokenizer/EMA remained frozen. Networks retain
evaluation mode for this deterministic bounded adaptation. Native
`root_model.loss` and `pose_model.loss` are used unchanged; no new loss weights,
decoder update, output filters or contact-label revision were introduced.
Training times were 38.52/39.71 s, both with 1,196.23 MiB peak allocated memory.

Four evaluations used identical 57 validation windows, source-root diagnostics,
source-derived skinned contact masks and raw 10 s commands. The fourth arm disables
reference goals on B's same weights. Input hashes, scoring code and effective
sample counts are checked automatically by `summarize_soma_pose_goals.py`.

| Measurement | Original | Matched A | B + Reference Goal | B Without Goal |
|---|---:|---:|---:|---:|
| Teacher-history/root-relative MPJPE (mm) | 87.40 | 78.67 | 79.67 | 79.84 |
| Worst validation single-joint step (deg) | 68.39 | 93.10 | 73.21 | 81.36 |
| Raw control boundary joint-step P95 (mm) | 42.04 | 62.70 | 61.16 | 61.75 |
| Raw command velocity error mean (m/s) | 0.316 | 0.822 | 0.852 | 0.846 |
| Validation mean-window measured slip (m/s) | 0.995 | 0.892 | 0.920 | 0.891 |

These slip/MPJPE values concern stitched teacher-history prefixes, NOT the old
full-window tokenizer reconstruction test. Raw control has no matched ground-truth
body and is not scored with validation motion contacts. The delay-12 test emulates
prefix skipping and is not a UE wall-clock deadline test.

The target reduces some worst jumps versus A and B-without-goal, but does not beat
the installed original on raw continuity/command following. Additional training
itself changes behavior substantially; the sparse goal cannot be credited with all
changes versus Original. `summary.json` records `rejected_regression` and
`quality_approved: false`. Do not install either trial bundle.

Local evidence: `out/soma_pose_goals_20261005`, with the four full metrics files,
raw motion arrays, training contracts/logs and automatic comparison summary.
Weights and optimizer checkpoints stay remote under the same experiment path.
Early evaluation interface/device failures are retained in the remote empty
`review_A`/`review_A_v2` directories; the completed report is `review_A_v3`.
Official skin() must receive all source joints, and its plain tensor attributes
must be moved with CPU diagnostic poses; both fixes have regressions.

Final Python suite: **258 passed**, including single/batched goal propagation,
hidden-future isolation, unchanged tokenizer, CPU/CUDA skin evaluation and rejection
of mismatched/poor-quality results. No UE assets or default model were changed.

Restoring this interface alone is not sufficient evidence of natural gait. The
decoder still enforces the old `provided_velocity` contract, rather than MB's
learned full-body/root refinement; its root channels were not trained for that
capability. Removing the overwrite from old weights is NOT a valid next fix.
That change requires paired decoder training and generated-history tests while
preserving original Root as a control, before further UE deployment work.

## Joint Root/Body Decoder Trial, 2026-10-05

The next experiment restores the released decoder output contract rather than
turning off root overwrite on an untrained installed checkpoint. The old
`provided_velocity` route remains the default. A serialized `learned_refinement`
route returns all decoder channels unchanged. Portable loading requires recorded
decoder training; changing the old bundle's policy field alone is rejected.

Method basis: [MotionBricks Section 4.2](https://arxiv.org/html/2604.24833v1)
joint root/body reconstruction, and [DART Appendix D.2](https://arxiv.org/html/2410.05260v1)
teacher, scheduled and generated-history stages. This is our adaptation to MB's
discrete tokenizer, not DART's diffusion model or its full four-primitive training.

All three arms start from the unchanged installed original. Root, Pose, encoder
and complete quantizer/EMA are frozen and checked by state digest. Only the
existing conditional decoder is optimized, with identical source-derived region
geometry and native tokenizer loss coefficients from the original tokenizer run.
No new temporal weights, IK, foot locking, pose/root filters or camera changes.

- A: provided-velocity output, teacher history.
- B: learned root/body output, teacher history.
- C: learned output, with detached generated history for the second primitive.

Each arm uses all 152 TRAIN clips, Adam 2e-5, batch 4, two consecutive primitives
per update, 64-frame windows, seed 2333, and 10,000 optimizer updates. The first
2,500 updates use frozen encoder tokens and source-root conditions (codec warmup).
Thereafter decoder inputs are captured from the exact deployed frozen Root/Pose
argmax path, including max-token padding and its active mask. Source endpoint
root values are navigation goals; no source future body/height conditions are
passed. C ramps history replacement from zero to one over updates 2,500-5,000,
then always uses generated second-primitive context. A/B sample the same clips,
starts, yaw augmentation and random draws, without replacing teacher history.

Targets keep their original future world coordinates and clock. One origin
translation is applied to the complete context-plus-target window for native
local-root integration; future targets are NOT rotated or translated toward a
drifting generated pose. Ground Y remains unchanged. Labels use the same source
window index; source contact evidence for generated context and its splice is
excluded by a fixed mask before decoding, never selected by predicted geometry.

Validation separates privileged source-root diagnosis from native Root/Pose
continuation. Both use 57 fixed held-out windows, identical source surface labels
and scoring masks. Raw control additionally has three 60-second sequences with
different observed idle histories and no ground-truth resets, covering start/stop,
walk/run, 90/180 degree turns, backward and strafing. These are NOT three training
seeds. The existing 10-second delayed-prefix test is retained as a counterfactual,
not a UE deadline benchmark. Every scored bundle, source window, effective mask,
frozen model identity and paired sampling budget is checked by the summarizer.

Evidence directory: `out/soma_decoder_refinement_20261005/trial`. Weights and
optimizer states stay remote; metrics/contracts/videos are brought back locally.
The default UE bundle remains unchanged pending raw-motion and actual RTX3080
UE acceptance. Relative metric gains alone do not approve natural locomotion.

### Completed Result

All three arms completed 10,000 optimizer updates. Sampling digests match and
Root, Pose, encoder and quantizer/EMA digests remain unchanged. Training took
786.54 / 790.09 / 772.13 seconds for A / B / C, respectively, on three idle g3
GPUs checked immediately before launch. Peak allocated memory was about 569 MiB
per arm. Final Python regression suite: **270 passed**, 10 warnings.

The following measurements compare raw outputs, without IK or filtering:

| Measurement | Original | A | B | C |
|---|---:|---:|---:|---:|
| Native-root validation MPJPE (mm) | 42.26 | 43.88 | 46.26 | 46.67 |
| Validation measured slip (m/s) | 0.424 | 0.353 | 0.359 | 0.359 |
| Validation mean penetration (mm) | 0.900 | 0.691 | 0.737 | 0.590 |
| Validation contact height (mm) | 1.773 | 1.854 | 2.078 | 2.176 |
| Long-control boundary joint-step P95 (mm) | 52.30 | 51.17 | 48.65 | 54.04 |
| Long-control worst joint-angle step (deg) | 31.96 | 82.76 | 30.08 | 31.68 |
| Long-control command velocity error (m/s) | 0.393 | 0.417 | 0.378 | 0.323 |
| Idle actual speed/error (m/s) | 0.141 | 0.125 | 0.199 | 0.193 |
| 90-degree-turn velocity error (m/s) | 0.715 | 0.745 | 1.169 | 0.950 |

Validation geometry rows are means across 57 source-masked windows. Long-control
rows summarize three 60-second raw rollouts, not three independently trained
seeds. MPJPE is root-relative; root position error is reported separately in JSON.
Slip uses source material-point evidence and is not a free-control slip metric.

B improves long-control boundary displacement and source-masked slip compared
with Original, while C improves aggregate command following. Neither passes:
both worsen body reconstruction, idle movement and 90-degree-turn following;
C also worsens boundary displacement. Lower mean penetration is not enough to
approve gait. The automated decision is `rejected_regression`, with
`quality_approved: false`. No trial weights were installed in UE.

The local report is `out/soma_decoder_refinement_20261005/trial/REPORT.md`.
`render_native/rollouts_full.mp4` retains all 3,600 frames at 60 fps for one complete
60-second command sequence, with four panels (Original, A, B, C). Its camera
follows root XZ, but world Y and the ground stay fixed; motion arrays are unchanged.
The native-rate video is a skeleton comparison, not a native-rate skin render.

The next model-training scope, not yet executed, is generated-history adaptation
of Pose's token-selection network with decoded-motion objectives. This trial
adapted only the decoder; it did not teach the frozen Pose network to choose
continuations from its own generated history. The results do not prove that
updating Pose alone will resolve the remaining failures. Root and codebooks should
remain matched controls, and idle/turn/body regressions must stay explicit gates.
