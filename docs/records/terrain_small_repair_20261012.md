# Bounded Terrain Repair Experiment

## Results And Current Follow-Up

The first short/long arms both completed650 updates and complete source-case
and constant-WASD tests. Neither solved the problem. Full-model adaptation
damaged several previously working flat controls; no live model was replaced.

A source/geometry audit then found two separate issues:

- The five available ascending climbing candidates failed the surface probe.
  On the initially selected source, the deepest actual intrusion was about11cm
  and occurred on LeftHandMiddle4, not the feet. The source retargeter explicitly
  only calibrates wrists. PARC uses4cm hand spheres; SOMA uses a skinned hand.
- Voxel SDF interpolation assigned about6.7cm negative distance to some mixed
  source points whose exact union-of-air-box penetration was zero. These false
  collision gradients can conflict with factual pose reconstruction.

New data/terrain_exact_interior.py computes exact negative clearance into the
same heightfield solids used by PARC. Exterior values are vertical gaps, NOT a
claimed full exact SDF. Only the negative part is used by penetration training.
Three tests cover analytic step values, gradients, exact-reference equivalence,
batch equivalence and unknown-map masks. The optimized active-prism selection
matches the exhaustive-reference gradient; it avoids storing all candidate
prisms in autograd. The first running exact arm retained the original exhaustive
implementation in memory; subsequent probes use the equivalent optimized query.

The second A/B uses six surface-screened stair/mixed/flat windows, excluding
unrepaired climbing. Both arms use100 complete13-primitive episodes,100 Adam
updates with per-episode gradient accumulation,7800 primitive examples, same
initial weights and losses. Within this A/B only collision geometry differs.
It is not a single-factor comparison against the first per-primitive experiment.

| Training case, source waypoint rollout | Voxel sole vertical P99 | Exact sole vertical P99 | Exact root RMS |
| --- | ---: | ---: | ---: |
| stairs up | 22.7cm | 1.5cm | 6.7cm |
| stairs down | 12.6cm | 1.4cm | 8.3cm |
| mixed up | 12.1cm | 0.3cm | 6.7cm |
| mixed down | 22.6cm | 2.3cm | 7.9cm |

All four exact-arm training terrain cases pass the previously fixed source-case
screen. Held-out mixed terrain and some flat controls fail. Constant-WASD
platform passage also fails. Therefore this is a bounded learning improvement,
not a deployment-ready terrain controller. Evaluation also retains the previous
coarse full-body SDF metric for comparison; exact source probes and sole vertical
tests have separately defined geometry. Never interchange these metrics.

An offline hand-reference diagnostic optimizes only the two wrist rotations
with actual SOMA skin/terrain penetration, minimum-change and temporal-change
penalties. It preserves Root and every non-wrist rotation. It reduced the
128-probe exact maxima to7.89mm(up)/5.26mm(down). Grasp forces and physical
tracking are unverified. This is an explicitly modified training reference,
not unchanged ground truth and not a runtime correction.

The current bounded candidate far_adapters starts again from final V8, uses
the eight cases with the two corrected hand references, exact geometry and
100 complete episodes. Only root_far/pose_far/decoder_far are trainable.
Their existing flat-reference residual is zero on a uniform plane, so this
restricts terrain adaptation without overwriting the flat controller. Actual
flat output equivalence and independent WASD traversal still require testing.
Its checkpoint contract records the modified references and frozen main network.

Do not duplicate healthy runs. Inspect pgrep actual arguments and progress.json:
remote small_repair_20261012/{stairs_exact,stairs_voxel,far_adapters}. Physical
GPU2 runs far_adapters;0/4 are used only for bounded evaluation/export when free.
Source reference experiment: hand_reference/report.json and cases.pt.
Local reports: out/model_terrain_v3_20261011/small_repair_20261012/.

Authorized scope: repair the current terrain mismatch through small-sample
training and actual testing. Keep the existing WASD pages and model available.
No postprocessing, IK, collision projection, fabricated future height or pose.

Initial checkpoint: final V8 weights5709, SHA256
8f70da0f9db03e84af86dfd09e5fce63b7d7e761b23ed9a42aa7fd72a9bea9cc.
Remote directory: ${PRIVATE_STORAGE}/motionbricks-20261010/
model_terrain_v3_20261011/small_repair_20261012.

## First Matched Experiment

Eight real129-frame training windows: stairs up/down, climbing up/down,
mixed terrain up/down, BONES idle/walk. Four validation windows have disjoint
source groups: mixed terrain up/down and BONES idle/walk. There are no separate
stairs/climbing family validation windows in this selection, so no claim about
held-out stairs/climbing follows from the reference-case test alone.

Each arm starts from identical final V8 weights with fresh Adam5e-5, batch8,
clip0.5, the unchanged recorded geometric coefficients and native objectives.
Each performs50 episodes times13 optimizer updates,650 actual updates and5200
primitive examples. The encoder and discrete vocabulary remain frozen.
The only A/B difference is factual-history reset cadence:

- short: reset at source segments0/4/8/12; at most3 consecutive generated histories.
- long: reset only at0; all13 primitives remain continuous,12 using generated history.

This isolates feedback span within the selected event dataset. It does not
reproduce the old broad-corpus sampler or establish full-corpus generalization.
All source windows are real; no padding or moving references with predicted drift.
Training uses explicit source XZ waypoints, never source future height or pose.

Physical A1000/2 were checked free before each launch. A1004 runs the separate
read-only constant-WASD baseline. Other GPU users remain untouched.

## Test Contract

Reference and rolling-control104-frame tests never reset generated history.
Rolling-control source-case tests still derive speed/facing from the source;
they are not the free-WASD test. The separate12-second constant-WASD test uses
only the four-frame initial history, command and authored scene, including
flat idle/walk,16cm and30cm stairs, and a platform approach.

Report root error, root-relative joint error, progress, sole vertical penetration,
unknown geometry and all18056 official skinned vertices with8 swept samples.
Vertical heightfield gap is not renamed Euclidean wall penetration depth.
The full-body metric uses the existing target-side distance volume.

Source-case thresholds are fixed before training: complete104frames, no unknown
body geometry, root RMS<=10cm, relative MPJPE<=8cm, body penetration P99<=2cm,
sole vertical penetration P99<=4cm, non-idle progress ratio0.8..1.2. Sources'
own penetration is reported separately. These bounded thresholds do not certify
natural motion; source/generated visualization and free control are additional.

Online thresholds are complete360frames, sole vertical P99<=4cm, body P99<=2cm,
idle drift<=10cm or forward progress>=12m with cross-track<=1m. Online passages
also require visual inspection; a numerical pass is not a production approval.

## Execution

Script: scripts/terrain_small_repair.py. Remote production uses the copied
terrain_small_repair.py and unchanged code_control_v8_live dependencies.
Case provenance: cases/cases.json and cases/cases.pt. The initial13-update
smoke completed, recorded12 consecutive generated histories and finite gradients,
and reached about3.85GB allocated GPU memory. It is not a quality success.

Baseline and final evaluation save every emitted frame and source counterpart.
Failure remains in the report; no sequence is padded or silently discarded.
Large weights/animations must be returned using the user's private Google Drive
client, not public SSH transfers. Small reports and source scripts can use SSH.
