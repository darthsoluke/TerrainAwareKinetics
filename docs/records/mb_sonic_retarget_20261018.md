# SOMA30 / G1 Static Kinematic Adapter

## Interface

```python
from motionbricks.mb_sonic.retarget import CalibratedSomaG1Adapter
adapter = CalibratedSomaG1Adapter(robot_xml, paired_soma, paired_robot,
                                 representation=native_representation)
adapter.save(calibration_json)  # new file; original paired assets unchanged
loaded = CalibratedSomaG1Adapter.load(calibration_json,
    robot_model_path=robot_xml, representation=native_representation)
reference = loaded.retarget_packet(soma_reference, start_time_s=current_time_s)
full_raw_limit_audit = loaded.audit_joint_limits(soma_reference)
history = loaded.estimate_history(executed_history, end_time_s=actual_time_s)
```

`retarget` consumes B's actual `SomaReference` (native 30 Hz, global/local
rotation matrices, root positions, named SOMA30), producing B's `G1Reference`
on covered 50 Hz timestamps. Trust ends at the actual last selected timestamp.
It remains the full-trusted-source diagnostic interface. The new uniformly
bounded consumer interface is `retarget_packet`, specified below.
`estimate_history` consumes B's `ExecutedHistory`, with complete named G1 FK,
timestamps and measured contact forces. It returns B's `NativeSomaHistory`, four
366-feature native samples at 30 Hz, provenance kind `calibrated_executed_g1`.

Dependencies are NumPy, SciPy, MuJoCo, joblib and the existing native SOMA
representation. The joblib reader handles the original official paired assets;
no custom pickle decoder is included. No MB model or GPU is loaded.

## Calibration And Geometry

Calibration uses the non-mirrored official walk only. The `_M` pair is held out.
The source SOMA schema stores 26 named marker positions, Y-up root translation,
root quaternion and frame rate. It does not contain full SOMA30 local rotations.
Named anatomical marker planes establish segment frames; adjacent limb planes
set an explicit twist gauge. Collinear/missing markers produce a named error.
The two supported marker schemas use actual Thumb1/Middle1 (paired26) or
ThumbEnd/MiddleEnd (native30) positions, not synthesized finger markers.

The native skeleton's own neutral bone directions define anatomical rest frames.
Straight neutral legs use rig X as secondary twist axis; straight neutral arms
use rig Y. These are declared identity-rig axes, not collinearity fallbacks.
Static Markley SO(3) means align these frames with the original G1 XML FK.
This is a fixed frame calibration, not angle regression or a model optimizer.
Root XY is preserved in the common metric world; a calibrated constant Z offset
accounts for morphology. The shared proper rotation is `(X,Y,Z)->(X,-Z,Y)`.

Coordinate derivation, with `C` the shared world rotation and `F_rest` the native
Y-up anatomical basis:

```text
F_pair is already Z-up in the released soma_joints array.
R_human_world = F_pair * F_rest^T
offset = mean_SO3(R_human_world^T * R_robot_world)
R_native_for_paired_reconstruction = C^T * R_human_world
R_forward = C * R_native * offset
```

Thus paired reconstruction applies `C*C^T` exactly once. No additional axis
conversion is applied to `soma_joints`. `soma_transl` remains Y-up and is converted
once separately. A nonzero paired residual measures the static-frame model's
limitations; an algebraic G1/native/G1 roundtrip is a different check.

All G1 FK comes from the actual XML body transforms, joint axes and anchors,
including interleaved hand joints. FK is batched algebra and never allocates or
overwrites a physical `MjData`. The 29-DOF reference uses original named joint
ordering. Passive hand descendants use their explicitly declared model rest
configuration for FK, not invented 29-DOF reference channels.

Forward conversion uses analytic hinge/Euler decompositions with the original
hip and shoulder fixed frames. Ankle yaw and hinge off-axis rotations are
projection residuals, not extra robot DOFs. Joint limit violations report the
specific joint, time and value; there is no clipping or zero-angle padding.
Reference velocities are finite differences of neighboring real reference
samples; the terminal derivative uses the last available derivative.

## Observed And Inferred State

The robot provides 35 used configuration DOFs: pelvis 6 plus 29 body joints.
SOMA30 has 93 configuration DOFs: root translation 3 plus 30 ball rotations.
The remaining **58 rotational DOFs are inferred**, not human measurements.

The inverse matches observed G1 pelvis and homologous segment orientations
through the calibrated frame transforms, then derives native local rotations
and calls the official native SOMA FK. The trunk's observed endpoint rotation
is distributed over Spine1/Spine2/Chest by native segment length. Unobserved
neck, head, face, clavicles, finger endpoints and toe articulation retain the
official native neutral local rotation. Per-bone observability ranks and global
rotation constraint masks are returned in provenance. Hinge/ankle null spaces
are also counted among the 58 inferred dimensions.

Four 30 Hz samples are resampled strictly inside factual 200 Hz coverage using
position brackets and SO(3) interpolation. The result records bracket times and
weights. No extrapolation, copied reference history or fabricated physical
frames are used. Native features are encoded by the existing representation,
with initial heading and root world XYZ restored.

Contacts require actual `mj_contactForce` normal force greater than the declared
1 N threshold. Contact points are transformed into the actual foot-body frame;
heel/toe split X is fixed from the original foot collision-anchor extent. Only
foot/world contacts count. Labels use the latest actual contact sample at or
before the target time. Robot force/spatial labels are measured; correspondence
to human heel/toe is explicitly inferred. No height/distance/velocity contact
heuristic is used.

## Evidence And Gate

Calibration artifacts contain source/XML hashes, native bone geometry, matrices,
thresholds and a canonical content SHA256 identity. Loading requires the same
original XML and native skeleton. Calibration status is initially
`NOT_RUN`; kinematic representability alone is not a behavior-quality gate.
Paired and held-out mirror residuals, G1 roundtrip and CPU timing are pending
only quality review; the actual measurements are recorded below.

Formal plan v1 sections 4.1-4.3 apply: B1 uses an explicitly named legacy
reference-history mode. The estimated actual-state inverse is a separately
reported diagnostic, not evidence that the frozen SOMA MB was trained on G1
execution. The `_M` pair is held out from this static calibration but shares the
same source recording; it is not independent behavior/generalization evidence.

## Actual Evidence

Candidate calibration:
`out/mb_sonic_e0_e3_20261018/retarget_calibration_v1.json`.
File SHA256:
`6be2cc02ae225c81a1ec28b6af59239c1656660d0f88b592ee4d725235d0bc6c`.
Content identity:
`d0b8a6d2e56c7237bfee7712c9c3122a87657176a3aef8e689e7e22ea232e5cc`.
Detailed report: `out/mb_sonic_e0_e3_20261018/retarget_report_v1.json`.
Calibration/diagnostics status: **REHEARSAL_ONLY**; behavior gate: **NOT_RUN**.

Original A001 and held-out A001_M are each 1202 frames at 30 Hz. Verified fields:
robot `root_trans_offset[T,3]`, `root_rot[T,4]` xyzw, `dof[T,29]` in MuJoCo order;
human `soma_joints[T,26,3]` Z-up/root-relative, `soma_transl[T,3]` Y-up/world,
`soma_root_quat[T,4]`, 26 actual names, and `fps`.

| Metric | Calibration A001 | Held-Out A001_M |
|---|---:|---:|
| Joint RMSE (rad) | 0.0525124 | 0.0519032 |
| Max absolute joint residual (rad) | 0.463464 | 0.432299 |
| Root-relative body position RMS (m) | 0.00899822 | 0.00963766 |
| Root-relative body position max (m) | 0.0515868 | 0.0553236 |
| Body orientation RMS (rad) | 0.0750546 | 0.0757202 |
| Raw world body position RMS (m) | 1.074702 | 1.074678 |
| G1/native/G1 max joint residual (rad) | 1.08236e-7 | 1.01597e-7 |
| G1/native/G1 max root error (m) | 0 | 0 |
| Joint-limit violations | 0 | 0 |

The raw world-position discrepancy is not a constant origin offset. Actual paired
G1 root XY displacement is 0.774427 times the human displacement (least-squares
diagnostic fit RMS 0.0008284 m; mirror 0.0008524 m). That observed trajectory
scale is **not applied to runtime references or scene coordinates**. Raw error
stays in the report. Fixed morphology Z offset is -0.2103411853 m. The recorded
G1 inverse reconstructs shared native human landmarks with RMS 0.02133/0.02156 m,
including original-versus-canonical proportion differences.

### Genuine MB Proposal

Input: `out/mb_sonic_e0_e3_20261018/mb_proposal_v1/proposal.npz`, the parent's
genuine two-call, historical-weight proposal, 52 source samples, trusted to 1.6 s.
Full forward feasibility status: **FAIL**. First 50 Hz violation is
`left_ankle_roll_joint` at 1.240000 s: -0.273970 rad, original XML limits
[-0.2618, 0.2618]. Eleven native source frame/joint violations and their exact
values are retained in the detailed report. The entire 1.6 s reference is rejected;
there is no automatic prefix cropping or clipping. A short prefix in the unit
test is explicitly selected and is not a full-proposal or physical success claim.

The missing step for this proposal is a declared, evaluated joint-constrained
reference retargeting stage, or G1-native motion representation/output adaptation.
Changing actual qpos, loosening robot limits, or silently capping reference angles
does not validate the present conversion. The 1.075 m raw paired root difference
also remains a separate trajectory/morphology design issue.

The parent additionally measured original SONIC G1-mode translation sensitivity:
translating released reference root/body positions by (1,-0.5,1) m left encoder
inputs, tokens and actions bitwise unchanged. This adapter retains world XYZ for
geometry, ghosts and error logs; those coordinates are not a direct absolute-root
tracking condition of the original G1 encoder. Direct root-trajectory control
requires an explicitly trained condition/adapter, not physical root correction.

### Declared Fixed Consumer Contract

The parent subsequently authorized a **new uniform 1.0 s consumer contract**:
0.9 s required by the released SONIC offsets plus 0.1 s replan guard. It applies
unchanged to every command and map, not only this proposal. Public API:

```python
packet = adapter.retarget_packet(full_source, start_time_s=issued_from_sim_time)
audit = adapter.audit_joint_limits(full_source)
```

This remains the default v1 contract, without per-scene shortening. The source
must cover the complete requested interval `[start,start+1.0]`, and all 51 interpolated 50 Hz
reference samples must satisfy the original joint limits. An infeasible requested
window fails instead of shortening or clipping. Packet clock remains the
simulation clock; generation/adoption wall time never rebases it.

Packet `adaptation` records contract
`motionbricks.mb_sonic.fixed_1s_consumer.v1`, fixed horizon/guard, selected bounds,
and `raw_source_joint_limit_audit`. The public audit evaluates all raw source
samples, including unconsumed tails, even if a source view has a shorter trusted
bound. Packet provenance preserves original full-source trust. Raw proposal,
raw MB blocks and the earlier full-1.6 s **FAIL** report are not rewritten.

Actual frozen-proposal results:

| Requested Interval | Samples | Joint Legality | Minimum Limit Margin |
|---|---:|---|---:|
| [0.00,1.00] s | 51 | PASS | 0.04554055 rad |
| [0.18,1.18] s | 51 | PASS | 0.06385390 rad |
| [0.30,1.30] s | none issued | FAIL at 1.24 s | left ankle roll exceeds limit |

The first complete SONIC window starts at `start`; the last starts at
`start+0.1`. A request at `start+0.12` is outside packet coverage and is rejected.
A source trusted only to 0.9 s cannot issue the fixed 1.0 s packet. All 11 native
tail violations remain attached to the two legal packets. These checks establish
bounded packet legality/coverage, not full-1.6 s legality or physical behavior.

New artifact: `out/mb_sonic_e0_e3_20261018/retarget_packet_1s_v1.npz`, SHA256
`450e4a0c8579017528882c00f96e5fbecf7822ab708208d1effd4ab3eb0f6041`.
New evidence: `out/mb_sonic_e0_e3_20261018/retarget_packet_1s_v1_report.json`.
The NPZ contains the actual fixed-window G1 reference arrays and complete source
audit metadata. Its declaration is identical for future commands/maps.

Twenty CPU repetitions, two threads, including full-source audit, fixed-window
resampling/projection, full XML FK, velocities and returned snapshots:
P50 8.07345 ms, P95 8.53624 ms, P99 9.36821 ms, max 9.57620 ms. These remain
component timings. Default-environment suite after this addition: **PASS**,
15 tests in 8.79 s. No physical execution was run by this module owner.

### Preserved Source Seed Initialization

C's initialization query was checked against the actual preserved
`final_acceptance_flat_v1/scene.npz`, SHA256
`5c4b2049a2801f8be4268da42dec730f5094e9b992910acd661d0c21027f1464`.
Its original four `initial_history[0]` frames were decoded at
(-0.1,-0.066667,-0.033333,0) s and passed through the same calibration.
The six covered 50 Hz reference samples satisfy original joint limits with a
minimum margin 0.04400324 rad. Last root XYZ is
(-0.0003148006,0.00007742255,0.7907370627) m over known ground Z=0.

All eight original foot collision spheres were transformed by exact XML FK.
Their signed bottom-to-ground gaps range from -0.004197845 to +0.012753254 m:
the left front sphere initially penetrates 4.20 mm. No root adjustment was made.
This is a kinematic initialization diagnostic, not a force/physics result;
explicit initialization must record it and let the unchanged physics resolve it.
Evidence `out/mb_sonic_e0_e3_20261018/retarget_seed_init_v1.json` includes last
qpos/qvel and every sphere center, radius and gap. Physics/force status: **NOT_RUN**.

### Timing And Tests

Twenty repeated CPU-only checks, two Torch CPU threads:

| Operation | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| Actual 52-frame rotation projection + full XML FK | 4.1145 | 4.4738 | 4.5188 | 4.5300 |
| Four native history frames from recorded E1 execution | 4.6490 | 4.8162 | 4.9689 | 5.0071 |

The latter uses the actual `e1_official_walk_v2/recording.jsonl`, sampled to end
at 0.3 s, including measured forces and contact points; its file hash is in the
report. These are component timings, not total-chain P99. The full 1.6 s valid
reference timing is **NOT_RUN**, because its joint feasibility fails first.

Default environment (no joblib serial override), Python 3.10.20, NumPy 2.2.6,
MuJoCo 3.8.0, joblib 1.6.0 and cloudpickle 3.1.2:

```powershell
$env:PYTHONPATH = '.;out/vendor/kimodo'
& 'D:/.anaconda/envs/motionbricks/python.exe' -m pytest tests/test_mb_sonic_retarget.py -q
```

Result: **PASS**, 11 tests in 8.85 s. Checks include native XML versus batched FK,
neutral rig frames, actual source hashes, calibration identity and reload,
independent mirror algebraic roundtrip, actual E1 history/force labels without
state mutation, preserved raw trajectory discrepancy, genuine MB rejection,
explicit short-window timestamp resampling, and quaternion sign continuity.
The static test oracle alone writes a separately allocated `MjData` for FK;
production conversion never allocates or modifies a physical `MjData`.

The first diagnostic report was produced with joblib's official serial setting
`JOBLIB_MULTIPROCESSING=0` while its optional multiprocessing import was incomplete.
Default import and all eleven tests were subsequently verified after the parent
installed the required cloudpickle dependency. No loader fallback or downgrade
was added. No MB model, GPU, server, physical rollout or optimizer was run here.

A unique true-human inverse cannot be recovered from G1 FK alone: neck/face/toe
articulation and ball-joint hinge null spaces are not measured, and the paired26
files omit those rotations. Full original SOMA30/77 local rotations with bind
frames, synchronized force/heel-toe labels, plus independent calibrated motion
coverage would identify and test those missing dimensions. This module returns
a declared constrained estimate rather than substituting reference poses.

## Explicit Scheduler Reserve Contract V2 (2026-10-19)

Parent's actual E3 B1 v1 exhausted its active reference at simulation 0.30 s.
The uniform 1.0 s packet reserved 0.1 s beyond SONIC's 0.9 s lookahead, leaving
no processing-jitter reserve after a 0.1 s replanning period. The next provider
took approximately 81 ms. The newly declared run-wide duration is:

```text
1.12 s = 0.9 s SONIC future + 0.1 s replan period
       + 0.1 s maximum accepted plan wall age + 0.02 s controller period
```

The API now accepts an explicit caller-configured duration:

```python
packet = adapter.retarget_packet(full_source,
    start_time_s=issued_from_sim_time, consumer_horizon_s=1.12)
```

The default remains `consumer_horizon_s=1.0`, with unchanged v1 contract and
adaptation metadata. A non-default explicit duration uses
`motionbricks.mb_sonic.explicit_consumer_horizon.v2`. Durations must be finite,
at least 1.0 s and contain complete 50 Hz intervals. The duration comes from the
uniform run configuration, never from current pose feasibility or remaining
source length. Missing coverage and joint violations fail; no fallback to 1.0 s.

At 1.12 s, all 57 interpolated samples are checked. Metadata retains
`replan_guard_s=0.1` and declares the additional `scheduler_reserve_s=0.12`.
The first SONIC window begins at `start`, the final one at `start+0.22`; a window
at `start+0.24` is rejected. Accepted source-state age below 0.1 s requires at
most 1.22 s of the genuine 1.6 s forecast, without rebasing its timestamps.

CPU checks used the unchanged genuine source, SHA256
`a99128d25eed2a6b2d2b50c2d7c87afc9eb3627cbe3f0728600577f45d65636e`:

| Requested Interval | Samples | Joint Legality | Minimum Limit Margin |
|---|---:|---|---:|
| [0.00,1.12] s | 57 | PASS | 0.04554055 rad |
| [0.10,1.22] s | 57 | PASS | 0.06157969 rad |
| [0.12,1.24] s | none issued | FAIL at 1.24 s | left ankle roll exceeds limit |

The failing angle is -0.273970 rad against the unchanged -0.261800 rad lower
limit. Each legal packet still contains the full 52-frame raw audit with all 11
violations and FAIL status. The original 1.6 s rejection, 1.0 s reports and NPZ
were not rewritten. The preserved v1 NPZ hash remains
`450e4a0c8579017528882c00f96e5fbecf7822ab708208d1effd4ab3eb0f6041`.

This change only declares scheduling reserve and verifies kinematic coverage.
No model, GPU, server or physical rollout was run for this patch; parent owns
the separate v2 runtime configuration/output and physical experiment.

Final bounded CPU suite: **PASS**, 26 tests in 9.88 s. Besides retained v1 tests,
it checks default/explicit-1.0 identity, v2 starts at 0/0.005/0.095/0.1 s,
57-sample coverage, latest-window bounds, preserved failed raw audit, missing
coverage, the genuine 1.24 s violation and rejection of non-finite, too-short
or off-50-Hz-grid durations. Eight existing TorchScript deprecation warnings.
