# MB + SONIC Reference Contract

## E1: Released Walking Reference

CPU-only entry point, usable without loading MB or CUDA:

```python
from pathlib import Path
from motionbricks.mb_sonic.reference import load_official_reference

reference = load_official_reference(
    Path("../gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428"),
    scene_id=manifest.scene_id,
)
window = reference.sonic_window(0.0)
```

The walking release has 455 real frames at 50 Hz, timestamps 0 through 9.08 s.
Complete G1-mode windows begin on the 50 Hz grid from 0 through 8.18 s.
Each window gathers indexes `i + [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]`.
Outside coverage or between existing grid timestamps, `ReferenceCoverageError`
is raised. There is no tail repetition, extrapolation, reset or SMPL substitute.

The source CSV joint order is IsaacLab. `qpos[:, 7:]`, `qvel[:, 6:]`, `joint_pos`
and `joint_vel` use the named 29-joint MuJoCo order. The properties
`joint_pos_isaaclab` and `joint_vel_isaaclab` recover the original CSV ordering.
`qpos` is `(T,36)` and `qvel` is `(T,35)`. Root linear velocity is world-frame;
root angular velocity in `qvel` is root-local, as required by MuJoCo.
Body velocities remain world-frame. All quaternion arrays use `wxyz`.

The released 14-body order, corroborated by `MotionSequence::ComputeFK`,
`policy_parameters.hpp`, and `gear_sonic/config/manager_env/commands/terms/motion.yaml`:

```text
pelvis, left_hip_roll_link, left_knee_link, left_ankle_roll_link,
right_hip_roll_link, right_knee_link, right_ankle_roll_link, torso_link,
left_shoulder_roll_link, left_elbow_link, left_wrist_yaw_link,
right_shoulder_roll_link, right_elbow_link, right_wrist_yaw_link
```

Its IsaacLab full-body indexes are `[0,4,10,18,5,11,19,9,16,22,28,17,23,29]`.
The adapter checks this metadata and records SHA256 for all six CSVs and
`metadata.txt`. Reference arrays are immutable snapshots. No velocity filtering
or FK replacement of the supplied official body poses is performed.

```powershell
$env:PYTHONPATH='.;out/vendor/kimodo'
& D:/.anaconda/envs/motionbricks/python.exe scripts/inspect_mb_sonic_reference.py ../gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428 --scene-id YOUR_SCENE_ID
& D:/.anaconda/envs/motionbricks/python.exe -m pytest tests/test_mb_sonic_reference.py -q
```

Run with `motionbricks` as the working directory. No download, server, remote
connection, physics rollout, optimizer step or GPU operation is required.

## E3: Frozen Trained Proposals

`mb_reference.TrainedMBReferenceAdapter.load(artifact, bundle_path,
canonical_scene=scene, device=...)` loads the existing portable terrain bundle
through `portable_parc_online_soma.load_terrain_bundle`, then strictly replaces
all Root/Pose/codec/adaptation weights with the completed compact full-corpus
artifact through `load_acceptance_candidate`. No network download is performed.
Before model loading, it restores the original actor precision contract:
`set_float32_matmul_precision('highest')`, CUDA matmul TF32 off, cuDNN TF32 off.
Actual flags are recorded in `model_identity.precision_contract` and therefore
in every raw block, proposal and generation receipt.
The original host contract is also restored before loading with
`torch.set_num_threads(4)`. Actual intra-op and inter-op counts are recorded in
`model_identity.host_contract`. ORT session thread settings are not changed.

The preserved artifacts under `motionbricks/out` are:

| Artifact | SHA256 |
| --- | --- |
| `pfnn_mixed_20261015/final_acceptance_flat_v1/weights.pt` | `0f5d8e2b05b8ec518a104a4979cca70f23efa64080abfc843f874634de5a9646` |
| `a100_live_20261010/a100-terrain.pt` | `0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d` |
| Original full-corpus Pose parent | `b960fbe9f792cf46333c23762a8f047259132acd5f6b595bbc726dffa8b40c19` |
| Final model tensor digest | `a6366a7e8522371be01d632efa6a952839a690a1749e14d6fb65819ff584b50c` |
| Final codec tensor digest | `ee4a08d79a5c5268799352507c9da8d323e5e81a40c36828fc68fe35f042f109` |

All three completed training stages have 12,800,000 exposures. Their actual
update counts are Root 50,000, codec 20,125 and Pose 4,000. These are historical
training counts; this integration performs zero optimizer steps. The separate
`scene_control_v3` capacity probe is not loaded or represented as this model.

The trained legacy native model emits 32 frames at 30 Hz. Relative to the
latest of its four history frames, their timestamps are `t-0.1` through
`t+28/30`. Its fixed runtime plays 24 future frames and drops the last four
decoder-edge frames, so trusted future coverage ends at `t+0.8`.
The released G1 encoder requires `t+0.9`.

`generate(history, velocity_world_xz_mps=..., heading_world_rad=...,
command_id=..., blocks=2)` resolves coverage using two real frozen-model calls:

| Call | Context times | Kept future times | Next context indexes |
| --- | --- | --- | --- |
| First | `t-0.1 .. t` | `t+1/30 .. t+0.8` | `24:28` of first raw output |
| Second | `t+0.7 .. t+0.8` | `t+0.8+1/30 .. t+1.6` | `24:28` of second raw output |

Each call remains 8 tokens / 4 Pose passes. The stitched `MBProposal.source`
contains the first call's indexes `0:28` followed by subsequent calls' indexes
`4:28`. `raw_blocks` preserves every original 32-frame output, including the
unused decoder-edge frames. No frames are repeated, interpolated across joins,
smoothed or overwritten. Timing for every real call plus decoding is returned
in `MBProposal.timing_ns`; each raw block records reconstructed-history root
error and the next-frame root step.

The rolling context is the existing native
`reencode_played_history(world[:,24:28], representation)`, not the decoder's raw
redundant channels. Its original position and rotation reconstruction gates stay
at `2e-5`. The source raw frames are retained unchanged. This makes RIC/velocity
channels consistent with displayed FK; it does not postprocess motion or reanchor
Root XYZ. Old v1/v2/raw-context reports remain old candidates and do not certify
this corrected rolling contract or explain the old joint-limit failure by themselves.

Every new `generate` call starts from a `NativeSomaHistory` with provenance
`source_seed` or `calibrated_executed_g1`. Generated context exists only inside
that proposal; it is rejected as the factual starting history of a new replan.
Seed-only physical tracking is labelled open-loop reference. A two-block
proposal starting from estimated actual history is a predictive proposal,
not two blocks of measured future execution.

Formal B1 explicitly permits `legacy_reference_history`. It is accepted only by
`conditioning_mode='B1'`; generated internal AR context is still rejected as a
new replan's history. `reference_history_b1(source, end_time_s=...,
representation=cpu_native_rep)` samples four covered 30 Hz reference times,
uses SO(3) interpolation where needed, and then calls the same native played-FK
reencoder. It labels contacts as reference forecasts, never measured forces.

Both `prepare_compiled` and `generate` accept
`conditioning_mode='B1', query_origin_soma_xyz=...`. The caller converts measured
G1 XYZ using the canonical inverse rotation and subtracts the calibrated forward
Root-Z morphology offset from native Y. Raw robot Z is not silently treated as
human pelvis height. Query samples are expressed relative to unchanged model
history. For future block i, the query center is
`A0 + (Hi.last_root - H0.last_root)`: measured current anchor plus predicted
reference displacement, not future actual truth. If A0 equals H0's Root, this
reduces to the original native query centers. Receipts retain requested anchor,
predicted offset and actual near-patch/volume centers. The native comparison for
this branch uses exactly the same query tensors as its compiled comparison.

The generate command uses SOMA axes: native yaw zero faces SOMA +Z, which maps
to MuJoCo -Y. Use `world_xy_command_to_soma(velocity_xy_mps, heading_world_rad)`
for browser/world commands. It applies the canonical inverse transform to both
the velocity and forward direction: native XZ velocity is `(vx,-vy)` and native
heading is `wrap(world_yaw + pi/2)`, not `-world_yaw`.

```python
from motionbricks.mb_sonic.mb_reference import TrainedMBReferenceAdapter

# The parent allocates the device before loading the real models.
generator = TrainedMBReferenceAdapter.load(
    artifact, bundle_path, canonical_scene=scene, device=allocated_device,
)
history = retargeter.estimate_history(executed, end_time_s=current_time_s)
proposal = generator.generate(
    history, velocity_world_xz_mps=(0.0, 0.5), heading_world_rad=0.0,
    command_id=command_id, blocks=2,
)
g1_reference = retargeter.retarget(proposal.source, start_time_s=current_time_s)
window = g1_reference.sonic_window(current_time_s)
```

This first adapter reads the canonical manifest's hash-bound source heightfield
directly. It explicitly rejects manifests containing independent rigid objects:
those require a shared full-solid scene-query adapter, rather than silently
omitting ceilings or walls. No physical state is assigned by the generator.

The frozen baseline retains trained nearest-node local-height observations and
prism-SDF volume observations. These differ from the canonical MuJoCo triangular
heightfield on nonflat cells. Each actual block records `geometry_audit`: maximum
and RMS local-height quantization error at the real model query XZ locations,
known-mask disagreement counts, and legacy prism SDF minus canonical vertical
clearance at actual volume-query points. Vertical clearance is labelled as such,
not as the global Euclidean triangle SDF. `same_query_as_E2_claimed` is false.
This baseline measures the mismatch instead of claiming shared surface parity.

Explicit parent-allocated inference CLI, with a new output directory:

```powershell
& D:/.anaconda/envs/motionbricks/python.exe scripts/inspect_mb_sonic_reference.py --generate --device cuda:0 --artifact out/pfnn_mixed_20261015/final_acceptance_flat_v1 --bundle out/a100_live_20261010/a100-terrain.pt --seed YOUR_SCENE_SEED_NPZ --manifest YOUR_SCENE_MANIFEST_JSON --scene-id YOUR_SCENE_ID --output NEW_E3_OUTPUT_DIRECTORY --blocks 2 --velocity-xz 0 0.5 --heading 0
```

The CLI writes lossless `proposal.npz`, every `raw_block_XX.npz`, and
`generation.json` with actual wall times and geometry audits.
`load_soma_reference(path)` loads these without running MB again. The source-seed
experiment is explicitly open-loop. No inference latency is claimed before
running this baseline; a later speed implementation must reuse the existing
fixed-shape native neural trace/capture with parity against this result.

`--execution native|trace|compiled` selects execution explicitly; there is no
automatic mode switch. The neural-only path reuses
`ReusableTerrainExecution._scene_inputs`, `trace_fixed_call` and `StaticCudaCall`,
without loading skinning assets. Scene samples remain current explicit inputs.
All preparation phases use the existing `full_precision_evaluation` policy
(highest matmul, cuDNN TF32 off). Tracing matches the proven reusable execution:
`cuda_constants=True`, numerical optimization on, CUDA serialization/reload.

The preparation proof records seven probes, including a real generated second
block, at `prepared_eager`, `dense_eager`, `traced_cuda_constants` and
`captured_cuda_graph` stages. Metrics include active-token equality, normalized
Root-feature error, decoder local-feature error, world Root-XYZ error, global
rotation error and maximum full-body FK error. The unchanged FK gate is 30 um.
Model tensor digests and actual CPU/CUDA RNG states are compared before/after;
RNG states are never reset. A failed stage stops with its measured diagnostics.

Per-plan `timing_ns` contains full wall time, both block totals, history upload,
proposal assembly and final proposal decoding. Each `block_stages` record
separates model dispatch, finite-output checks, geometry audit, source
diagnostics, raw CPU transfer, native FK dispatch, pose CPU transfer, immutable
snapshots and next-context preparation. The compiled path additionally records
separate scene-query and neural CUDA-event elapsed times. Native CUDA elapsed
time is explicitly labelled as including scene queries, not neural-only.
CPU transfer/check/audit intervals include any CUDA waiting they perform.
CUDA elapsed intervals overlap wall intervals and must not be added to them.
This single-plan telemetry does not automatically run a hot-plan benchmark.

## Full-Plan Execution Candidate

`--execution full_plan` is a separate fixed-scene, two-AR8 execution contract.
It composes Zeno's admitted raw query/observe/FK ScriptModules and the raw neural
ScriptModule in one Python callable captured by `StaticCudaCall`. There is no
outer Torch tracer and no nested graph replay. A CPU regression retains the
known frozen-ScriptModule/outer-trace failure and verifies the selected composition.
Goal construction is pretraced too, so capture glue performs only tensor operations.

The graph runs queries, AR8, the native four-frame observer, queries at the future
rolling origin, AR8, and one native FK of the stitched 52 frames. It returns one
lossless byte payload preserving original float, integer-token and mask dtypes;
one CPU transfer supplies poses, raw blocks, actual conditioning histories and
query evidence. `RawMBBlock` deliberately has no eagerly expanded poses.
`save_raw_mb_block`/`load_raw_mb_block` retain GLOBAL366 and its four-frame input;
offline `decode_soma_reference` can expand a raw block when needed. Logging does
not trigger two more FK passes on the critical path.

Static scene/source/storage bindings are checked at the plan boundary. Preparing
the candidate admits the native query/FK/observer helpers and compares both active
eight-token blocks plus final FK52 against native same-query execution. The
original `3e-5` FK and `2e-5` observer gates remain unchanged. Model and RNG states
are compared. The parent has admitted the native-conditioning graph below;
B1 query-origin admission, hot timing and physical checks remain separate.

The independent source-seeded admission CLI uses the prior proof command with
`--execution full_plan --blocks 2` and a new output directory. B1 uses the explicit
`conditioning_mode`/`query_origin_soma_xyz` Python API or CLI
`--conditioning-mode B1 --query-origin-soma-xyz X Y Z` and needs admission at those
same actual-query inputs. These are absolute native SOMA coordinates supplied by
the caller, not raw G1 XYZ. C converts measured G1 XYZ by inverse canonical C and
subtracts the calibrated `root_z_offset_m` from native Y; history and model Root
are unchanged. The CLI does not perform a second morphology conversion.
Neither mode is an automatic performance fallback.
The new raw files are `raw_block_00_global366.npz` and
`raw_block_01_global366.npz`; `proposal.npz` still contains the complete timestamped
52-frame source and FK arrays. Native token IDs, actual four-frame input history
and geometry implementation SHA256 are retained in the evidence.

B1 admission retains six command cases and adds two different query anchors
offset from the supplied anchor by `[.05,.02,-.04]` and `[-.06,-.015,.03]` metres.
Both blocks compare native same-input query tensors at `1e-5` absolute tolerance,
exact valid masks, active eight-token IDs, FK at `3e-5`, and observer at `2e-5`.
The second query anchor is the initial anchor plus predicted reference
displacement. The proof checks the supplied origin and unchanged initial
history as well as final FK52, model tensors and actual CPU/CUDA RNG states.
The two offset anchors are admission probes, not measured future positions.

## Long-Horizon Decision

Full Pose12 is excluded from the same-baseline speed route. The formal training
uses 8 tokens / 32 frames. `pfnn_gait_horizon_review_20261017.md` already records
same-3-m/s-command mean actual speeds of 2.4147 / 0.2302 / 0.08523 m/s for full
8 / 12 / 16. The long Root puts acceleration near the far terminal rather than
the executed prefix. Native/compiled parity also established that tracing did
not introduce this slowdown. No repeat of those historical gait probes is needed.

The deployed repair was dual preview12/16 with execution8 and Pose8, not a repair
of full-length Pose12/16. A possible longer tensor horizon is therefore not an
execution-equivalent performance improvement for this trained actor.

The native online command and prediction functions already accept 6..16 tokens;
`root_path_split_execution.HORIZONS` explicitly contains `(8,12,16)`. One
12-token prediction produces 48 real frames. Keeping frames 0..43 retains four
history frames and 40 future frames, with trusted future `40/30 = 1.333333` s.
It covers the original G1 encoder's 0.9 s lookahead and the fixed 1.0 s consumer,
provided consumer adoption starts within the remaining 0.333333 s coverage.

Against two AR8 calls, one AR12 removes one Root call, four Pose inference passes,
one query and the cross-block observer. Pose-decoder output is 48 rather than
64 raw frames; final FK is 44 rather than 52 frames. Root attention's active
prefix changes from 17 to 21 tokens, so latency cannot be inferred by halving
the measured two-AR8 result. No one-12 inference or performance run was started.

This is a separate execution contract. The current adapter hardcodes token 8
in command-goal construction, query duration, prediction, token comparison,
raw shape, edge exclusion, stitching and metadata. The FK helper currently
admits only fixed 32/52 shapes; 44-frame FK needs its own shape admission.
The existing split RootPath executor adds path-dependent conditioning and
skinning, so swapping that executor into this baseline would change more than
the horizon. A separately approved full12 policy would need independent
12-active-token, FK44, model/RNG, latency and behavior qualification. The existing
behavior failure excludes it from the current performance implementation.

The prioritized exact route retains 8 tokens and four Pose passes.
`pfnn_horizon_acceleration_review_20261017.md` and
`root_path_split_execution.py::_build_graphs` provide the existing batch-one
Root packing: validate the original 17-token valid-prefix padding at capture,
run the original encoder on that unpadded prefix without a mask, then restore
the native zero tail. The old proof retained native NestedTensor equivalence
without changing observations, Root output, time constraints or precision gates.
It also records shared frozen model-storage constants and disabled JIT optimizing
executor during warming/capture. The current admitted B graph uses the full
padded dense Root; adopting prefix packing is a new execution strategy requiring
same-query token/FK/model/RNG admission under the original precision contract.
Shared constants concern storage/execution, not a change to actor math. A matched
official short-reference SONIC model/config remains a separate baseline option.

## Measured Parent Probes

`mb_proposal_v1` contains two actual 8-token calls with 1.6 s trusted future
coverage. Cold block wall time was 462.253 ms, second block 99.4159 ms, total
565.2774 ms: the native reference baseline fails real-time timing. Scene 000
reported zero measured height-quantization and validity-mask discrepancies.
Its precision flags were unspecified: retain it as a historical timing baseline,
not a numerical reference for the restored original precision contract.

`mb_compiled_smoke_v1` failed its proof after 23.6573 s allocation wall time:
active tokens were identical, but FK error was 167.431834 um, above 30 um.
Its logs are retained and no compiled reference was exported. CPU inspection of
the actual 52-frame native proposal found zero FK/rotation difference between
native and dual-global representation inverses. The original compiled attempt
used a different constant strategy and did not explicitly disable cuDNN TF32;
the staged proof above aligns the established policy and isolates remaining
Root/decoder/trace/capture error. It has not been declared successful.

The parent's `sonic_reference_translation_sensitivity.json` shows that globally
translating a real reference by `[1,-0.5,1]` metres leaves the original G1-mode
encoder input, latent tokens and motor actions bitwise unchanged. This encoder
does not consume reference world translation or height. World Root/body poses
remain recorded for comparison and rendering, not as evidence of direct Root
XYZ tracking. Physical-state replacement or extra forces are not used to fill
that missing condition.

The parent's restored-precision `mb_proposal_compiled_v2` passed 28 staged token,
FK, model-hash and CPU/CUDA RNG checks; maximum FK difference was 1.206557 um.
That execution still used raw AR context. Its original proof is retained with
that scope, not relabelled as corrected-history qualification.

The parent's 10-warmup/20-plan 10 Hz `mb_hot_profile_v1` failed: end-to-end
P50/P95/P99/max were 86.39285/103.17772/103.822504/103.9837 ms. Summing actual
per-block intervals within each plan gives P50 query CUDA-event interval 30.104 ms,
neural CUDA-event interval 28.232 ms and three native FK dispatch intervals 18.568 ms.
Final duplicate proposal decoding alone was P50 6.847 ms. Query event intervals
include host dispatch gaps. Finite-output wait intervals overlap neural execution;
they are not extra independent GPU work. The failed profile and frozen sources
are preserved; no repeated identical probe or result claim is made.

The parent's first `mb_full_plan_smoke_v1/execution.json` passed with all seven
bound source hashes unchanged, including `mb_reference.py` SHA256
`e16de886e1b9494a2baa57217f278bc2d83031bce802014ff502ca465ebb43f0`.
Six native-conditioning cases, each with two AR8 blocks and the official
observer, retained exact active tokens and maximum FK error 2.538626 um.
FK52, model tensors and CPU/CUDA RNG states passed; optimizer updates were zero.
Allocation wall time was 34.8349534 s. The first complete generation was
34.484 ms, CUDA elapsed 31.341 ms and CPU unpack/audit 1.608 ms. This removes
the prior query/FK host dispatch bottleneck; the two neural calls still dominate.
That receipt qualifies its frozen native source snapshot, not the subsequent
B1 varying-origin proof or a hot-plan/physical result.

## Estimated Execution History

The existing `helper/mujoco_helper.py` expects G1-specific `_skel` names and
hinge topology, not SOMA30. SOMA30 must not be passed to that converter.
`ExecutedHistory.append(model, data)` records actual qpos/qvel, body world
poses, body names, contact positions, body IDs and `mj_contactForce` wrenches
separately from references. Call it after the runtime's post-step `mj_forward`.
`native_soma_history(converter, end_time_s=...)` delegates to the explicitly
calibrated adapter's `estimate_history`; without a converter it raises an
embodiment error rather than treating a reference as physical history.

The retarget module owns the paired-reference calibration, measured/inferred
masks, native bone-length FK, force-channel interpretation and cycle residuals.
Unobserved local degrees of freedom need declared calibrated constraints.
Frozen physical tracking and measured input-distribution/retarget errors decide
whether the integration is usable; implementing this representation bridge
does not itself require new model training.

## Prepared CPU B1 History

E3 B1 v1/v2 retained valid numeric logs but exhausted consumer coverage. In v2,
the reference-history work alone was 28.6026 / 31.3519 ms on the next two requests;
those concurrent online costs remain recorded. Increasing the consumer to 1.12 s
did not eliminate the expiry. This change accelerates that same B1 representation
operation; it does not alter generation, the two AR8 calls or the consumer horizon.

Production has one explicit path:

```python
from functools import partial
from motionbricks.mb_sonic.mb_reference import (
    prepare_reference_history_b1, reference_history_b1,
)

# Startup, after the original adapter host/precision setup; use the pinned real seed.
observer = prepare_reference_history_b1(retargeter.rep, initial_history)
history_builder = partial(reference_history_b1, observer=observer)
# Existing snapshot call, with the same representation object:
history = history_builder(source, end_time_s=history_time, representation=retargeter.rep)
```

Preparation accepts actual FP32 CPU `[4,366]` or `[1,4,366]` data, owns a copy,
traces the existing `native_observed_history` via `trace_fixed_call` once with
numerical optimization disabled, and performs three warmup calls. It proves all
366 output channels and tensor checks equal to native on the startup input and
compares CPU RNG state. `observer.preparation` contains JSON-only receipt values.
There is no compile during snapshots, representation replacement or automatic
switch to the eager oracle. Wrong representation identity, dtype or shape fails.

The snapshot samples the identical four timestamps and scalar channels. SO(3)
uses the same SciPy Rotation/Slerp operations on only the original adjacent
keyframes required by those timestamps (at most five for these 30 Hz sources),
instead of rebuilding all 52 frames for every bone. Exact native-grid samples
retain their original bytes. Heading unwrap/interpolation and contact forecast
channels retain the old arithmetic. The prepared observer still re-encodes the
played FK state and enforces finite output and the original `2e-5` position and
rotation checks. Contact forecasts remain reference channels, not actual forces.

`reference_history_b1_oracle` preserves the old eager/full-sequence body for tests
and diagnostics only. An AST regression compares that body against the frozen
actual v1 source snapshot. Production callers must supply the prepared observer.

CPU qualification uses nine saved actual NPZ sources: the native, neural-compiled
and full-plan proposals plus two v1 and four v2 numeric raw sources. Nine time
offsets per source give 81 windows. Five existing-native coordinate heading
transforms over eight phases add 40 coordinate-frame tests, explicitly separate
from generated-motion/physical evidence. All 121 windows have bitwise-identical
366 features and contacts versus the unchanged oracle; coordinate-case native
Root, joint positions and local/global rotations also match bitwise. Sources and
pinned trained asset bytes are unchanged, CPU RNG is unchanged, and CUDA remains
uninitialized in the worker's CPU process.

Paired standalone CPU benchmark on v1 source plan `1:1121884641496400`, history
end `0.2200000000000001` s: original four intra-op threads, highest precision,
TF32 off, five discarded warmups, then 50 individual calls per path, no batching.
Startup preparation was 194.3409 ms; the first prepared call was 5.3408 ms.

| Single-Call Wall Time | P50 ms | P95 ms | P99 ms | Max ms |
|---|---:|---:|---:|---:|
| Unchanged eager oracle | 11.40735 | 12.12182 | 12.348629 | 12.4358 |
| Bracket SO(3) + prepared CPU observer | 4.83105 | 5.37449 | 5.640266 | 5.7758 |

All measured outputs matched all 366 channels exactly; CPU RNG was unchanged.
These are standalone CPU costs. The next actual v3 chain must retain its own
snapshot, generation, retarget, scheduling and consumer-age measurements.

## Verification

The reference worker ran 58 CPU tests using unmodified released walking and
mirrored CSVs, the actual four-frame trained-artifact SOMA seed, and actual
weights/bundle hashes. The shifted-clock regression selects 410 consecutive
windows with an origin of 0.18 seconds and an accumulated 0.02-second clock.
Window `trusted_until_s` uses the selected real frame timestamp, fixing the E1
v1 floating-point metadata error without changing coverage or copying frames.
CLI parsing checks preserve the explicit B1 native origin, reject omitted mode,
missing origin and nonfinite values, and leave CUDA uninitialized.

Actual model inference and physical tracking are parent-allocated integration
checks; the CPU reference tests do not claim GPU generation or E3 success.
