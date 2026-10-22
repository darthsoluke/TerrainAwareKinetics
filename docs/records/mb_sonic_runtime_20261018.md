# SONIC Physics Runtime, 2026-10-18

## CPU E1 Entry Point

Owned files: `mb_sonic/runtime.py`, `mb_sonic/sonic_policy.py`,
`scripts/run_mb_sonic_physics.py`, `tests/test_mb_sonic_runtime.py`, this document,
and the explicitly added `configs/mb_sonic_e1_v1.json`,
`configs/mb_sonic_b1_fullplan_v1.json`, and `configs/mb_sonic_b1_fullplan_v2.json`.
The explicit clock task additionally owns `mb_sonic/windows_clock.py`,
`configs/mb_sonic_e1_v4.json`, and `configs/mb_sonic_b1_fullplan_v3.json`.

`FrozenSonicPolicy(encoder_path, decoder_path, observation_config,
encoder_sha256=..., decoder_sha256=..., threads=1)` loads the original matched
frozen ONNX pair using CPUExecutionProvider only. No DDS, C++ deployment,
hardware connection, GPU, optimizer, or server is involved.

`G1Reference` from `mb_sonic.reference` is the only reference boundary.
`reference.sonic_window(runtime.time_s)` selects the ten actual samples at
native-50Hz indices 0,5,...,45. Missing trusted future coverage raises, never
holds/extends a final pose. Joint positions/velocities use MuJoCo body order,
SI radians/radians per second, world root quaternions wxyz. SONIC inference
uses the already-selected ten-sample window.

```python
from motionbricks.mb_sonic.runtime import load_physics_scene, SonicPhysicsRuntime
scene, model, robot_metadata = load_physics_scene(manifest_path, robot_xml_path)
runtime = SonicPhysicsRuntime(scene, model, frozen_policy, physics_dt=0.005)
runtime.reset(explicit_full_qpos, explicit_full_qvel)
warmup_frames = runtime.warmup_history()
tick = runtime.step(g1_reference, raw_reference=raw_reference_record,
                    issued_at_monotonic_ns=command_issue_ns, phase="walking")
tick.actual       # Actual PhysicsState
tick.physics_states  # Every actual 200Hz substep, not display frames
runtime.model, runtime.data  # Same model/data for rendering and measured history
runtime.scene    # Same canonical queries/manifest, including unknown domains
```

No independent scene description is constructed. The loader composes the
official `g1_29dof_with_hand.xml` robot with `scene.scene_include_path`, without
the infinite plane in `scene_43dof.xml`. All hand collision/dynamics remain.
The source explicitly warns of hand instability; passive hands are reported
as such, not silently repaired. MuJoCo `mjDSBL_AUTORESET` is set. Runtime
terminal failures preserve physical state. Task-specific failure evaluation
is injected explicitly; the generic runtime never declares a root-height fall.

Headless CLI (parent supplies allocation and launches):

```powershell
$env:PYTHONPATH='.;out/vendor/kimodo'
& 'D:/.anaconda/envs/motionbricks/python.exe' scripts/run_mb_sonic_physics.py `
  --scene-manifest <canonical-scene/scene_manifest.json> `
  --robot-model ../gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml `
  --policy-dir out/mb_sonic_e0_e3_20261018/sonic_6733128 `
  --reference-dir ../gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428 `
  --allocation-id <parent-cpu-allocation> --seconds 5 `
  --output <new-evidence-directory>
```

`--physics-dt 0.0025` and `0.00125` support separate convergence runs without
changing 50Hz policy rate. Output is actual `recording.jsonl` and `summary.json`;
it includes raw/adapted references, all physics states, measured contacts and
forces, PD controls, encoder/decoder inputs, native/simulation/monotonic clocks,
and stage timing. Display rate/latency stays null until measured by a viewer.
The CLI's failure rule is specific to walking: pelvis/torso scene contact over
1N, plus solver/numeric failures. It is not reused to reject crawling.

Future recordings include actual per-state `qfrc_applied`, `xfrc_applied`,
`root_qfrc_actuator`, and all MuJoCo warning counters/lastinfo in
`solver_warnings`. Existing v2/halfdt recordings are preserved unchanged and
predate these extra audit fields. Ground query coverage is explicit through
`root_ground_known` and nullable `root_ground_height_m`; outside the canonical
domain is not flattened or clamped.

## Evidence

At the E1 handoff, the targeted CPU suite had **18 passed**. It covered
official permutations/gains/scales, observation masking/layout/orientation,
no history padding, real model/hand collision preservation, native substep
clocks, clipped actual torque, MuJoCo contact wrenches, terminal preservation,
absence of a generic root-height fall threshold, actual pinned ORT inference,
recording exclusivity, and original-speed sampling of actual physical states.
The parent subsequently owned/completed the video artifact; the extra renderer
CLI/test was removed as requested. No redundant video render was launched.
Lab tests additionally check content-addressed actual mesh assets, read-only
command semantics, non-stepping snapshot polls and the explicit start gate.

Parent-executed E1 evidence, center of canonical flat scene `000`:

| Run | dt | Policy/physics steps | Terminal | Chain P99 | With recording P99 | Joint RMSE | Root RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `e1_official_walk_v2` | .005s | 250/1036 | none | 7.867ms | 11.931ms | .125592rad | .174442m |
| `e1_official_walk_halfdt_v1` | .0025s | 250/2072 | none | 11.989ms | 19.717ms | .125955rad | .177214m |

Both use five seconds of the independent official walking clip, original
matched frozen SONIC pair, CPU only, zero optimizer updates. Warmup contributes
the extra physics steps. The original `e1_official_walk_v1` pre-policy reference
timestamp failure remains retained; it was not overwritten or automatically
restarted. Half-dt trajectory/contact comparisons are parent-owned. No video
render was launched by this worker.

The matched pinned YAML describes a 1762-wide encoder input and 994-wide
decoder input; the repository's older 436-dimension comment is outdated.
Startup binds the matched YAML SHA256, not that older copy. G1 mode 0 uses absolute
reference joint positions/velocities (reordered to IsaacLab), 10 relative root
orientations, and the official all-zero mode-0 tag. Unused teleop/SMPL channels
are zeroed by the configured mode mask, not invented sensor data. Root height
is not active in release G1 mode.

The decoder concatenates token64, gyro30, offset joint positions290, joint
velocities290, previous actions290, projected gravity30. Histories are grouped
by quantity and oldest-first. Positions subtract official default angles;
velocities and angular velocity have no extra scaling. Relative orientations
flatten the first two matrix columns row-wise. Initial heading alignment is
explicit, not repeatedly reset during rollout.

`GatherEncoderMode` in official `g1_deploy_onnx_ref.cpp:1687` writes the numeric
mode into element 0 and zeroes the remaining three elements; it is not a
one-hot vector. Thus mode 0 is exactly `[0,0,0,0]`.

The physics runner owns 50Hz inference and four real MuJoCo 0.005s steps
per command. It will expose the canonical scene-owned model/data to rendering
and return actual qpos, qvel, contact wrenches, torques and phase clocks. Ten
real measured history frames are collected by explicit default-pose PD
warmup; no initial observation frame is copied or silently padded.

The fixed original pair is pinned to HF revision
`6733128a3d8a523b1418b06bca3cdf61c8b0987f`. CLI hashes require the matched encoder,
decoder, and `observation_config.yaml`. Do not mix low_latency/v1.1.

## Live E1 Adapter

`create_lab_runtime(config_path, allocation_id=...)` now returns a real
`SonicLabRuntime` with `config`, `asset`, `snapshot`, `start`, `stop`,
`submit_intent`, `select_scene`, and `begin_episode`. Construction loads CPU
assets and performs only an explicit initial reset; server `start()` arms a
worker blocked on an explicit start gate, without consuming reference time.
After geometry is ready, a parent/user action calls `begin_episode()` to start
the independently paced 200Hz physics worker and native-50Hz SONIC control.
Browser polls and ACKs never
advance either clock. Static geometry is packed once as little-endian float32
XYZ vertices followed by uint32 triangle indices; its SHA256 is the asset ID.
Per-frame snapshots contain only actual geom poses, reference points, contacts,
clocks and diagnostics, not the meshes.

The initial mode is `control_mode="official_reference_read_only"` and
`intent_enabled=false`. Every snapshot has `command=null` and null body-onset
association. Normal intents raise a clear read-only error; disconnect-stop is
explicitly reported unconsumed. This is live SONIC physics following the
independent reference, not recording playback and not joystick control.
Reference exhaustion/completion stops stepping without reset. Scene selection
is an explicit episode reset and starts a new recording, again behind the
explicit start gate. `config.start_required=true`, `start_action="begin_episode"`;
the viewer/server owner must expose a separate start action, never reuse WASD
or a velocity command to fake consumption. Calling `begin_episode` twice is
an error rather than a replay loop.

Runtime configuration JSON fields:

The ready parent startup configuration is `configs/mb_sonic_e1_v1.json`:
only scene `000` enabled, five seconds, CPU ORT, new recording directory
`out/mb_sonic_e0_e3_20261018/e1_live_v1`. This worker did not instantiate this
configuration or start its service/physics episode.

```json
{
  "schema": "motionbricks.mb_sonic.runtime_config.v1",
  "mode": "official_reference",
  "scene_root": "${PROJECT_ROOT}/out/mb_sonic_e0_e3_20261018/scenes",
  "initial_scene_id": "000",
  "enabled_scene_ids": ["000"],
  "robot_model": "${UPSTREAM_ROOT}/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml",
  "reference_dir": "${UPSTREAM_ROOT}/gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428",
  "policy_dir": "${PROJECT_ROOT}/out/mb_sonic_e0_e3_20261018/sonic_6733128",
  "encoder_sha256": "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3",
  "decoder_sha256": "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed",
  "observation_config_sha256": "466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b",
  "policy_threads": 1,
  "physics_dt_s": 0.005,
  "duration_s": 5.0,
  "recording_dir": "<parent-selected-new-recording-directory>"
}
```

The catalog is discovered from actual canonical manifest files, including
disabled scenes with their unavailable reason. Parent chooses enabled scenes;
there is no fabricated list, automatic placement rebase, or terrain-specific
friction change. The worker was not started by this agent.

## E3 B1 Integration

The authoritative desktop PLAN sections 1.1, 4, 10 and 11 govern E3. B1 uses
`conditioning_mode="B1"` and history kind `legacy_reference_history`.
Actual G1 snapshots are maintained
in a separate `ExecutedHistory`; optional calibrated SOMA estimation is labelled
diagnostic, not a formally trained G1 input condition. Inverse estimation does
not block the B1 forward-reference baseline.

The parent's actual B1 v1/v2 smokes are **FAIL** on reference exhaustion.
V1 stopped at .300s; v2 stopped at .420s and exposed the exact-age numerical
rejection documented below. Actual MB v3 subsequently stopped at 1.24s on a
future retarget joint-limit violation. The round ends **FAIL** on executable
reference and total-chain timing gates; no further retry or E4/T1 work is authorized.
B's `reference_history_b1(..., representation=...)`, exact
`legacy_reference_history` kind, and `generate/prepare_compiled` explicit
`conditioning_mode="B1", query_origin_soma_xyz=...` APIs have landed.
The runtime binds these real APIs before allocating an MB model; it does not
substitute a source. The approved pipeline re-encodes played history for
consistent FK/redundant channels; prior v1/v2 direct-context candidates and
their failures remain retained, not relabelled as the corrected baseline.
No MB/GPU/extra physical run was launched by this runtime worker.

Raw intent is `velocity_xy_mps[2]` plus `yaw_rate_radps`, in world Z-up axes.
Native 50Hz yaw integration starts from actual initial yaw and is independent
of translation: strafe/backward/pure turn are not auto-faced into travel.
Same-tick newer rates replace the tick's rate without integrating twice.
The original raw sequence/rate stays unchanged. Only an accepted plan actually
used by SONIC populates snapshot `command/effective_intent`; mailbox receipt,
heading integration and plan adoption alone never mark it consumed.

The producer has one latest-intent slot and one ready-plan slot. Requests use
10Hz start-to-start scheduling plus input events, capped at 50Hz, without FIFO
backlog. Late superseded results, aged snapshots, wall-age deadlines and real
future coverage are checked separately. Expired active references or obsolete
intent beyond its configured bound terminate the episode with its state intact.

The `ReferencePacket` binds robot XML SHA256, scene SHA256, named joints, SI
units, frame, original generation sim time/state version/raw command sequence,
plan ID, real future times, valid-until and conditioning mode. It separately
records actual G1 Root XYZ, query origin in `soma_world_y_up`, and calibrated
morphology offset: inverse-C(actual G1 XYZ), then native Y minus the G1 root-Z
offset. A robot root near .79m therefore queries near native human Y=1m.
No reference/history Root features or physical state are rebased. Actual policy-use
receipts preserve the exact ten consumed times, action-ready and motor-write
monotonic clocks. Initial receipt, adoption, first plan use and first command
use are separate events. Reused steady commands report their age rather than
being counted as fresh input-response samples.

After generation, the fixed consumer starts at the next actual 50Hz sim grid;
it selects existing global source coverage without shifting source poses or
timestamps. C explicitly passes
`retarget_packet(full_source, start_time_s=..., consumer_horizon_s=...)`.
The v2 uniform scheduling requirement is 0.9s SONIC future + 0.1s replan
period + 0.1s maximum plan wall age + 0.02s controller period = **1.12s**,
57 dense native-50Hz samples and 0.22s usable window. Startup validates the
explicit setting against that formula and the native duration grid. Metadata
retains each component and the actual requested interval. No per-command/map
shortening or copied tail is permitted. Gauss's default/explicit 1.0s remains
the historical `fixed_1s_consumer.v1`; the v1 config and reports are unchanged.
Full 1.6s/raw-block outputs and all tail joint-limit violations remain recorded.
Any requested consumer-window limit/coverage failure is fatal, not shortened.

MB setup additionally requires explicit `mb_artifact/mb_artifact_sha256`,
`mb_bundle/mb_bundle_sha256`, `mb_device`,
`mb_execution` (`native/trace/compiled/full_plan`), `mb_seed_path/mb_seed_sha256`,
`calibration_path/calibration_sha256`, `mb_conditioning_mode`,
`estimate_actual_diagnostics`, `mb_replan_interval_s`,
`max_generation_state_age_s`, `max_plan_wall_age_s`,
`max_pending_intent_age_s`, `mb_initial_reference_timeout_s`, `consumer_horizon_s`, and `body_onset`
thresholds (`linear_threshold_mps`, `heading_threshold_rad`, `sustained_samples`).
`robot_model_sha256` and the scene-ID-to-digest map `scene_manifest_sha256`
bind the compiled robot and selected canonical scene before allocating MB.
The explicit weight/bundle pins must match B's admitted pair; B's actual loader
then hashes the original bytes. Invalid execution names fail before model load.
The parent chooses bounds, allocates the device, and supplies a new recording
directory. Compilation/parity is explicit deployment initialization with real
seed data, no command sequence, zero physics steps and its own duration/report.
Required numeric logs also need an explicit `raw_log_pending_plans` capacity.

`configs/mb_sonic_b1_fullplan_v1.json` pins the actual preserved assets and
calibration. It enables only scene `000`, five seconds, single-threaded CPU ORT,
MB on `cuda:0` with explicit `full_plan`, B1 legacy reference history and
`estimate_actual_diagnostics=false`. Replan starts are 0.1s apart plus events;
generation-state age, plan wall age and pending-intent age are each bounded at
0.1s. Initial-reference timeout is 5s and the numeric queue capacity is two
plans plus the fatal-overflow reserve. The new recording directory is
`out/mb_sonic_e0_e3_20261018/e3_b1_fullplan_v1`. Parent ran this historical
configuration once; it predates the explicit consumer-horizon setting now
required by C. This worker did not instantiate it or start GPU/physics.

Static config checks rehashed seven robot/scene/SONIC/seed/calibration assets,
matched MB weights/bundle against the parent's byte-hash receipt and current
sizes, and confirmed before the parent's run that the recording directory did not exist.
Config SHA256 is
`7c1b0faa86b7f1fd755d5495587e03d62cd2ddb23a97d69de67141b1c78f1e2d`.

The new `configs/mb_sonic_b1_fullplan_v2.json` changes only the recording
directory to `e3_b1_fullplan_v2` and adds `consumer_horizon_s=1.12`.
All assets, age bounds, input/physics/controller settings and quality gates
are unchanged. SHA256:
`91913ed50c86e206af090f97d95826b79aba110e1571641c0b48444f52596b9a`.
Paths exist and its recording directory does not yet exist; it was not launched.
B source remains SHA256
`ce9eb5d9ea91632e4e3dd6a4e9b6f9da5c622b3987757eba610f3e104f1fd8b6`.

Large raw source/block/conditioning/adapted arrays are written as uncompressed
NPZ, with small JSON metadata containing array keys/shapes/dtypes. READY is
published after the newest-revision check and before numeric snapshot copies
or file I/O. Writer snapshots own read-only copies, not model/pending buffers.
One bounded queue plus one reserved fatal-overflow record prevents silent log
loss; backpressure terminates the episode and drains the retained records.
Late final-drain errors promote completion to `runtime_failure`. Receipts bind
artifact SHA256, bytes, snapshot-copy time and persistence time. No giant raw
array `tolist()`/JSON runs on the producer. E1 tick JSON remains separate and
its measured scope is retained.

The real seed is `final_acceptance_flat_v1/scene.npz` (SHA256
`5c4b2049a2801f8be4268da42dec730f5094e9b992910acd661d0c21027f1464`).
Forward retarget initializes actual G1 from its last source pose/velocity once.
Gauss measured 4.20mm initial left-front foot geometric penetration; no root
height correction is applied. The measured warmup resolves the declared
initial state through physics. Source seed clock assignment is logged without
any spatial rebase. Missing initial plans are an explicit initialization wait,
included in episode RTF; no physical/reference frames are fabricated.

Body onset is the first sustained command-directed actual root velocity or
heading change from the plan-consumption baseline. Its thresholds, baseline,
first sample and detection clocks are retained. A nonzero raw yaw rate determines
angular direction eligibility even if the first integrated target equals the
baseline or differs by less than the threshold. Actual measured heading change
still must exceed the fixed configured threshold in that direction.
Observation runs once per physical substep, with the actual `sample_hz` and
first-to-detection `sustained_span_s` logged. The new .005s configuration uses
12 samples at 200Hz: twelve sampling slots (three nominal control periods),
whose first-to-last span is .055s, not three controller observations.
Unobservable projected yaw is
not counted as measured angular response; the desired yaw integrator persists.

The MB primary ghost is `kind="adapted_g1"`: released `OFFICIAL_BODY_NAMES` in
14-body order, with nearest-included-ancestor parents derived from the actual
compiled G1 model. Its `body_pos_w` sample and Root deviation use the same
adapted reference index/time. Snapshot version, used command sequence/session
and plan ID remain associated with the plan actually consumed by SONIC.
The viewer label is `MB -> G1 reference`; the E1 official ghost is unchanged.
All raw SOMA source/blocks remain in NPZ for the source/adapted/actual audit.
No reference root or physical pose is relocated for display.

Future live timing includes necessary scene/reference handling, inference,
physics, history, geom-pose packing/publication, and scheduler work, excluding
only explicit waits and separately measured optional recording writes. Both
history snapshot/reference-builder cost and publish/scheduler work are included;
actual state-capture clocks precede history construction. Scheduler overshoot
beyond the requested wait is not subtracted as sleep. Both
controller work and each used plan's measured MB-provider work are reported;
reusing a plan is not counted as another generation. Existing E1 logs keep their
original narrower timing scope. No 90,000-tick/200-event E6 certification is
claimed from bounded E3 tests.

Future snapshots additionally split `lock_wait`, `state_snapshot`,
`reference_history`, and their `total` in `history_snapshot_stages`.
Their full elapsed snapshot work remains included in provider/necessary-chain
timing; the builder runs outside the actual-state lock.

Parent's actual v1 evidence is
`out/mb_sonic_e0_e3_20261018/e3_b1_physical_smoke_v1/report.json`, together with
the unchanged v1 episode JSONL and two raw numeric NPZ records. Initial plan
provider work was 46.126ms and initial wait 74.08ms. The next request measured
28.112ms snapshot + 43.3395ms generation + 9.7589ms retarget = 81.2116ms.
Its request/ready clocks were 1121884751070000 / 1121884832290100 ns; the
old plan was published at 1121884687705700 ns. The 100ms usable old horizon
expired before the next plan was ready. Only the initial zero command had
been issued; planned W/strafe/stop/yaw were not reached. Recorded root
actuator force, qfrc_applied and xfrc_applied were zero, solver warnings zero;
failure was temporal coverage, not a reported physical fall/joint-limit/NaN.

B's CPU-only replay of the saved conditioning point reproduced all 366
history channels exactly: 20 calls P50 12.02675ms, P95 13.626185ms, max
13.9508ms. Its profile attributed about 6.8ms/call to 30-bone full-keyframe
Rotation/Slerp construction and 5.6ms to the required native observer.
The online 28.112ms also includes concurrent scheduling/lock effects; it is
not replaced by this isolated timing. Bracket-only SO3 and prepared observer
optimizations await parent review and parity proof; they are not implemented
by C or folded into the horizon change.

Gauss's CPU legality check for the existing genuine source passed 57 samples
at [0,1.12] and [.10,1.22], with minimum joint margins .04554055/.06157969rad.
[.12,1.24] still fails left ankle roll at 1.24s, and the full raw 1.6s audit
still has 11 violations. These gates are preserved.

E3 scheduler/serialization/protocol/clock regression tests: **62 PASS**, using no motion-model
inference, MuJoCo step, browser, server or GPU. The remaining 17 tests include
the physical/model baseline and were not rerun. The merged physical suite is
**NOT_RUN** by this worker after E3 changes. Parent's prior-candidate actual MB
runtime timing is **FAIL**: its compiled parity proof passed 28 stages (max FK 1.2066e-6m), but the
first full two-block plan took 61.30ms; its hot-20 profile was P50 86.393ms,
P95 103.178ms, P99 103.823ms. These cannot be replaced by E1 controller timings.
Real-plan binary log throughput timing awaits parent allocation, separate from
the GPU profiling window.

Original release G1 mode ignores world root/body translation and root height.
Parent's actual CPU proof `sonic_reference_translation_sensitivity.json`
translated the full real reference by `[1,-.5,1]m` and measured bitwise-identical
encoder inputs, tokens and actions. Reference root deviation remains diagnostic;
it is not a directly conditioned SONIC XYZ tracking target. Actual body motion
is still produced only by joint torque and MuJoCo contacts.

## Frozen V2 Handoff

Final targeted CPU run: 41 passed, 17 physical/model cases deselected, 8.89s.
It includes exclusive recording/NPZ file creation: repeated filenames/directories
raise `FileExistsError` and leave existing bytes unchanged. Read-only numeric
snapshots retain their own arrays. All 16 existing v1 files, including the nine
source/config snapshots, were hashed before/after the final check and unchanged.

Parent may use the same `LabSession.intent` driver/program as v1 with
`configs/mb_sonic_b1_fullplan_v2.json`, a new driver/report/sources directory
`out/mb_sonic_e0_e3_20261018/e3_b1_physical_smoke_v2`, and the runtime-created
recording directory `out/mb_sonic_e0_e3_20261018/e3_b1_fullplan_v2`.
The requested full 1.12s window retains every coverage/joint-limit gate; an
invalid window fails without a shortened reference or retry. No worker GPU,
physical rollout or service operation was performed. Runtime/config/B/helper
sources below were frozen for the parent's single bounded v2 attempt. They
identify that run, not the later numerical-age correction below.

| File (relative to motionbricks) | Frozen SHA256 |
| --- | --- |
| `motionbricks/mb_sonic/runtime.py` | `c1a194d33d90eb74c5d117a92d0697664fdea9ee0aa7c87766426c2672c97630` |
| `motionbricks/mb_sonic/sonic_policy.py` | `58cbd157b9bcd48776784a422ea20c381a681c6fd79e3fbc5d7ba3dc70ee3ef0` |
| `motionbricks/mb_sonic/reference.py` | `e48667b2095d8dd7de06ec52b0cc17d681feaf254e33e838bb3e9d565456357f` |
| `motionbricks/mb_sonic/scene.py` | `08627880adec81da4e465bd2e369fc7cb4110a4aeb1bb03b747c3f343269eb3c` |
| `motionbricks/mb_sonic/retarget.py` | `9cdc2f17c9b015ddd7588b109d9fc620a9e3c951d839f1b023ed0e207d76d112` |
| `motionbricks/mb_sonic/mb_reference.py` | `ce9eb5d9ea91632e4e3dd6a4e9b6f9da5c622b3987757eba610f3e104f1fd8b6` |
| `motionbricks/mb_sonic/geometry_execution.py` | `07efbfcda276a032fb1c3351bf7682ce1a7fca10e4b21672544c9f4bdc649b3b` |
| `scripts/mb_sonic_lab_server.py` | `b9c4df3568cc137f7dee7b59ea84ac806d3161ee2d929997eca90c9d889656d5` |
| `configs/mb_sonic_b1_fullplan_v2.json` | `91913ed50c86e206af090f97d95826b79aba110e1571641c0b48444f52596b9a` |
| `tests/test_mb_sonic_runtime.py` | `54a3fa02287f3bb95f413e24b20f1f2ef9e461d2e10c34b7897f24aa13b23052` |

## V2 Consumption Review

The parent completed the actual v2 attempt: twelve policy ticks, 48 controlled
physics substeps, final sim .42000000000000026s, initial zero intent only.
The actual v2 JSONL/NPZ and frozen source snapshots were read without modification.

Plan `1:1123619343112900` was ready/published in 81.5104ms, leaving 18.4896ms
before its 100ms request-to-consumption deadline. Publication occurred during
the .30s controller tick. The next .32s tick discarded it for wall age; the
following policy start bounds its actual check age to (100,100.3689]ms.
This rejection is consistent with the unchanged wall-age limit. The prior tick
recorded 27.4272ms control wall and 15.1445ms scheduler residual. Ready polling
runs once at the native control boundary; synchronous `asdict`/JSON/tick writes
occur before the next poll. The residual includes wait overshoot and other
unassigned scheduling work; existing logs cannot identify a GIL-only cause.

Plan `1:1123619443840500` exposed a concrete numerical bug: captured sim time
.3200000000000002/state 64, consumed time .42000000000000026/state 84.
Twenty actual .005s steps are exactly 100000000ns, but float subtraction gives
`0.10000000000000009`, so strict `> .1` incorrectly rejected the legal boundary.
Its real adapted interval [.4,1.52] covers the requested SONIC [.42,1.32].
The subsequent terminal clock bounds request age at no more than 96.0242ms,
still within the unchanged wall-age gate.

The minimal correction compares individually rounded integer-nanosecond sim
timestamps against the same 100000000ns limit. CPU regressions accept exactly
100ms and reject 100ms+1ns; there is no increased age allowance, changed
deadline, horizon adjustment or changed joint-limit gate. Final targeted CPU
suite: 44 passed, 17 physical/model cases deselected, 9.29s. No rollout was rerun.

Numerical-fix-only runtime revision SHA256:
`395b77a8aa31dd99ab7df6c9d12b766de3731b3e91588506c782edb6f6476e45`.
Tests SHA256:
`25d16c10924f7137ba4d9d09b3ff6341803a855f0a58b96437eef850d38eed26`.
The frozen v2 runtime remains `c1a194d3...`; config v2 remains `91913ed5...`.
B's separately assigned exact CPU history acceleration subsequently passed its
CPU proof and was explicitly integrated as documented in the latest handoff below.

## Explicit Windows Clock

Parent's actual independent E1 live v3 completed 250 ticks with full-chain
P99 27.2499ms. One 27.73ms controller tick measured policy 4.305ms, physics
5.015ms, publication .896ms, history .060ms and scheduler 17.457ms, despite
only .361ms requested explicit waiting. Other high scheduler residuals were
15.87/17.26/16.75/16.85ms. These observations motivated the OS-clock change;
v3 logs/config remain unchanged.

Microsoft documents that wait timeouts depend on clock resolution and that
the application may request a finer period with
[timeBeginPeriod](https://learn.microsoft.com/en-us/windows/win32/api/timeapi/nf-timeapi-timebeginperiod).
Since Windows 10 2004 this request applies to calling processes rather than
the old global setting. Each successful request must be matched using the
same period with
[timeEndPeriod](https://learn.microsoft.com/en-us/windows/win32/api/timeapi/nf-timeapi-timeendperiod).
The [wait-functions documentation](https://learn.microsoft.com/en-us/windows/win32/sync/wait-functions)
recommends acquiring resolution once around the timed workload, not per wait.

New configuration is explicit:

```json
"clock": {"strategy": "windows_event_wait_1ms", "timer_resolution_ms": 1}
```

The helper requires Windows 10 build 19041 or newer, checks the native timer
capability and requests exactly 1ms. Unsupported platforms/periods or a failed
MMRESULT terminate explicitly. There is no default-resolution fallback,
priority/power-policy change, global timer request on old Windows, or busy spin.
Windows 11 can ignore requests for fully occluded window-owning processes;
the native return/actual wait measurements remain the evidence, not a claim
that every deadline will be met. QPC/perf_counter_ns and all quality gates
are unchanged.

The lease begins only after `begin_episode` opens the existing gate. It covers
the paced warmup/control and producer cleanup, and the context pairs release
on normal completion, stop and exceptions. Stop before the gate makes no timer
request. `timer_resolution_acquired` and `timer_resolution_lifecycle` records
contain the process strategy, OS build, capabilities, MMRESULT and actual
acquire/release clocks. Release failure is explicit runtime failure.

Waiting stays interruptible `Event.wait` against existing monotonic deadlines.
Each tick now reports `wait_requested`, `wait_elapsed`, and `wait_overshoot`
in nanoseconds. Only `min(elapsed, requested)` is subtracted as explicit waiting;
overshoot remains in scheduler and full necessary-chain timing. Past deadlines
do not sleep or enter a spin loop.

`configs/mb_sonic_e1_v4.json` differs from the actual v3 config only in `clock`
and the new `e1_live_v4` recording directory. `configs/mb_sonic_b1_fullplan_v3.json`
similarly differs from MB v2 only in `clock` and the new `e3_b1_fullplan_v3`
directory. Both output directories are absent. Earlier configs/logs are not
rewritten. The parent can review E1 v4 for its one bounded actual profile;
this worker performed no timer activation, physics, GPU or service startup.

Final CPU tests: **60 passed**, 17 physical/model cases deselected, 8.91s.
The WinMM boundary is controlled in tests; cases cover matching release on
normal/error exit, failed capability/acquisition/release, explicit 1ms-only
configuration, rejection of globally scoped old Windows, cancellation, past
deadlines and keeping 14ms synthetic wait overshoot in the scheduler.
These are lifecycle/arithmetic tests, not an actual OS timing profile.

| Clock-only revision file (relative to motionbricks) | SHA256 |
| --- | --- |
| `motionbricks/mb_sonic/windows_clock.py` | `b46c512e406469498c45f6bfebfe8a5c2282372841b884ad9dac3d799c9a89f2` |
| `motionbricks/mb_sonic/runtime.py` | `9f47441760a36dde2c694b667be883cc95315a05e5126b8539f00a60ef5187a1` |
| `tests/test_mb_sonic_runtime.py` | `3e77ec66e7efc7a44f623882580a66e27b04471033d05853f86b6523233f4182` |
| `configs/mb_sonic_e1_v4.json` | `01b044fe73b56ce4e2a89923ecafa40af53d38e8123c168e11bcf1f0cf3f4bca` |
| `configs/mb_sonic_b1_fullplan_v3.json` | `8db4baa76c46a9a08edce01d49cadab437d5e6a5f8638ac38f650610b3cb07ac` |

Include the new `windows_clock.py` in the parent's next source snapshot/hash
list. At the clock-only handoff, B's observer wiring awaited authorization;
it is now complete under the parent's explicit authorization below.

## Prepared History Handoff

B is verified frozen at
`8f02a51e74942acc648622868aa8f63af27a230b286f5284ddf1a727836231ff`.
Its 121-window CPU proof compared all 366 channels, contacts and FK bitwise
against the preserved oracle. C did not edit B, the math, the two-AR8 generator,
precision, horizon, age bounds or quality gates.

`FrozenMBProvider.prepare` now prepares the CPU observer once from the real
`retargeter.rep` and supplied seed `history.features`, before the native early
return. This follows actual model loading, which restores the native four-thread
host contract. It binds `partial(reference_history_b1, observer=...)`; the
snapshot continues to pass that same representation explicitly. Further scene
preparations reuse the admitted observer, rather than compiling per call.
No oracle or lazy compilation is reachable from production snapshots.

Both native/full-plan startup reports contain a deep-copied JSON-compatible
`cpu_history_observer` preparation receipt and `cpu_history_observer_prepared_now`.
The receipt is also in recording metadata. Every execution mode now requires
explicit successful preparation before `propose`; a failed full-plan preparation
does not leave a ready report.

The final C CPU run passed **62 tests**, with 17 physical/model cases deselected,
in 11.22s. Wiring tests cover native/full-plan preparation, one-time caching,
exact seed/representation forwarding, the bound production partial and receipt.
A real frozen-seed CPU observer/partial call matches the oracle bitwise, leaves
both seed arrays and CPU RNG unchanged, restores test thread settings, and
asserts CUDA remains uninitialized. No MuJoCo step or neural GPU call ran.

Parent's actual OS-only evidence is
`out/mb_sonic_e0_e3_20261018/windows_wait_probe_v1.json` (SHA256
`fb6cd42accc01b12db5f070b4197d5d61973f5415e42649e62168cef72b0a370`).
Thirty individual nominal-1ms interruptible waits in each phase measured:

| Phase | Elapsed P50 | Elapsed Max |
| --- | --- | --- |
| Before lease | 15.49245ms | 16.078ms |
| Active 1ms lease | 1.02115ms | 1.9953ms |
| After release | 15.48570ms | 15.9738ms |

The parent probe recorded Windows build 26200, begin/end MMRESULT 0 and final
status `released`, with no model/physics/GPU. It directly confirms the timer's
OS-wait effect. Actual E1 v4 / MB v3 chain measurements are parent-owned;
the completed MB v3 result is recorded below. No realtime PASS is inferred
from this isolated probe.

The existing E1 v3 process PID140812 was not operated. New configs remain
`configs/mb_sonic_e1_v4.json` and `configs/mb_sonic_b1_fullplan_v3.json`, with
new `e1_live_v4` and `e3_b1_fullplan_v3` recording directories. Use the existing
explicit start gate and `LabSession.intent` protocol; no driver or browser
substitution is needed. Runtime/B/configs below are frozen for parent review.

| Final handoff file (relative to motionbricks) | SHA256 |
| --- | --- |
| `motionbricks/mb_sonic/runtime.py` | `0b593f80895e73cb800b2901491b52c4eecb8aee0df9c81692ed1710e132520e` |
| `motionbricks/mb_sonic/sonic_policy.py` | `58cbd157b9bcd48776784a422ea20c381a681c6fd79e3fbc5d7ba3dc70ee3ef0` |
| `motionbricks/mb_sonic/windows_clock.py` | `b46c512e406469498c45f6bfebfe8a5c2282372841b884ad9dac3d799c9a89f2` |
| `motionbricks/mb_sonic/reference.py` | `e48667b2095d8dd7de06ec52b0cc17d681feaf254e33e838bb3e9d565456357f` |
| `motionbricks/mb_sonic/scene.py` | `08627880adec81da4e465bd2e369fc7cb4110a4aeb1bb03b747c3f343269eb3c` |
| `motionbricks/mb_sonic/retarget.py` | `9cdc2f17c9b015ddd7588b109d9fc620a9e3c951d839f1b023ed0e207d76d112` |
| `motionbricks/mb_sonic/mb_reference.py` | `8f02a51e74942acc648622868aa8f63af27a230b286f5284ddf1a727836231ff` |
| `motionbricks/mb_sonic/geometry_execution.py` | `07efbfcda276a032fb1c3351bf7682ce1a7fca10e4b21672544c9f4bdc649b3b` |
| `scripts/mb_sonic_lab_server.py` | `b9c4df3568cc137f7dee7b59ea84ac806d3161ee2d929997eca90c9d889656d5` |
| `configs/mb_sonic_e1_v4.json` | `01b044fe73b56ce4e2a89923ecafa40af53d38e8123c168e11bcf1f0cf3f4bca` |
| `configs/mb_sonic_b1_fullplan_v3.json` | `8db4baa76c46a9a08edce01d49cadab437d5e6a5f8638ac38f650610b3cb07ac` |
| `tests/test_mb_sonic_runtime.py` | `f137475cd7d2b0b7d1a69bba9c7854ebd4077dc54a25e42b3ae48dfe00150fcc` |

## Final MB V3 Outcome

**FINAL FAIL. This round stops on executable-reference and 20ms necessary-chain
gates. No further GPU retry, E4, T1, optimizer or formal-training work is authorized.**
This update reads existing evidence only; no production file or test was changed
and no CPU/GPU test, physical step, model execution or service operation was run.

Primary evidence under `out/mb_sonic_e0_e3_20261018/`:

- `e3_b1_physical_smoke_v3/report.json`
- `e3_b1_physical_smoke_v3/analysis.json`
- `e3_b1_physical_smoke_v3-allocation-v2/execution.json`
- `e3_b1_fullplan_v3/episode_0001_000.jsonl` and its twelve numeric NPZ records

Allocation `e3-b1-physical-cuda-v3` used 37.1947313s GPU allocation wall, exit
code 2, zero optimizer updates. The execution record binds eleven source/config/
driver files before/after and reports all unchanged, including runtime
`0b593f80...`, B `8f02a51e...`, clock `b46c512e...` and MB config `8db4baa7...`.
The actual controlled prefix has 53 policy ticks / 212 controlled substeps;
36 warmup substeps make state version 248 and final sim 1.2399999999999956s.
Ten plans were adopted. The sole ready discard was a superseded old intent,
not an age/coverage rejection. At sim .66s a plan with logged float age
.10000000000000009 was correctly admitted at its exact 100ms integer-clock
boundary. No starvation terminal occurred in this run.

### Executable Reference

The failed request is retained in numeric file
`f07d781c2545f8cfb0685d64067d941f175ff62235b9e55680cfa8289bb71a3d.npz`.
It records command 2, snapshot sim 1.174999999999997/state 235, genuine source
coverage [1.06,2.76]s, two raw AR8 blocks, and the unchanged explicit 1.12s
consumer interval **[1.22,2.34]s**. The dense G1 sample at **1.96s** maps
`right_ankle_roll_joint` to reported **-.261978rad**, below **-.261800rad**.
The reported excess is approximately .000178rad, well beyond the frozen
retarget check's 1e-6rad numerical tolerance. This sample lies .74s into the
request, inside the mandatory .9s SONIC lookahead, not just the scheduler reserve.
The complete requested window was hard-rejected; no clipping/shortening occurred.

The last valid adopted plan `2:1124815708515000` covers [1.12,2.24]s, with
latest SONIC start 1.34s. It still had about .10s usable coverage at termination.
The runtime stopped on the new provider's explicit failure rather than silently
continuing that plan. The infeasible future reference was never adopted/used.
Last actual right ankle roll was **+.00380552291rad**, root XYZ approximately
[-.05330,.07914,.77861]m. Last-state root actuator force, qfrc_applied and
xfrc_applied arrays were zero and all solver warning counters were zero.
The terminal evidence identifies future-reference infeasibility, not an
observed actual ankle-limit crossing, solver/NaN fault or root-height rule.

### Necessary-Chain Timing

| Measured Scope | N | P50 ms | P99 ms |
| --- | --- | --- | --- |
| Full necessary chain | 53 ticks | 67.8551 | **88.61218** |
| Controller-only work | 53 ticks | 11.6360 | 16.732336 |
| Provider total | 11 produced plans | 56.5193 | 75.99914 |
| History snapshot | 11 produced plans | 2.7509 | 17.9310 |
| Two-AR8 generation | 11 produced plans | **34.8682** | 37.48789 |
| G1 retarget | 11 produced plans | 18.9903 | 25.98923 |

The measured full-chain P99 fails the 20ms target; generation P50 alone exceeds
20ms. The controller subset and successful 1ms timer lease do not replace the
full chain. Stage quantiles are reported separately, not summed. Reused plans
remain reused compute receipts, not 53 independent generations. The failed
retarget request is retained separately from these eleven completed-plan timings.
The timer lease recorded begin/end MMRESULT 0 and final status `released`.

### Command Boundary

Only the first planned movement event was reached: command 2 at actual sim
.5000000000000003s, world XY velocity [0,-.5]m/s and zero yaw rate. Its first
SONIC use starts at .58s, with input-to-action **87.0815ms** and input-to-motor
87.178ms. Initial zero command latency was 265.1202ms, including warmup/startup.
The only recorded body-onset event belongs to initial command 1; there is no
recorded command-2 body-onset event. Later strafe/stop/yaw program events were
not reached. Browser measurement is false, so command-associated display latency
was not measured by this headless run. These receipts show the real API/plan/
SONIC/physics prefix, not completion of the five-second program or E6 certification.

| Evidence File | SHA256 |
| --- | --- |
| `e3_b1_physical_smoke_v3/report.json` | `377d35325eaa0d37fa23e38b79fabe09cf2a53f23636444403f4bec66f7c2c6a` |
| `e3_b1_physical_smoke_v3/analysis.json` | `8f7a88e100321a3c2c0892437e2cef1087722df74d8a1eaa7cc62c7efd0aab05` |
| `e3_b1_physical_smoke_v3-allocation-v2/execution.json` | `40bca2d462cc64417a676ca959b0427d021d108aa05afa6722d777e3f03b0558` |
| Failed-reference numeric NPZ above | `11c3cdc02dfba0de8b4412de1328c689985aeb736880ab3c2d5bf5161ce62146` |

## Authorized WASD Handoff

New explicit user authorization supersedes the previous round's stop for this
separate WASD integration. Baselines 55588/28947 were not operated. New service
28948 is parent/James-owned; C started no GPU, physics, browser or service.

Ready config: `configs/mb_sonic_wasd_v1.json`, SHA256
`1631cb2f215f974f988c938123c090594954965d64478cdb4c19bc0405617061`.
Runtime SHA256:
`3aedfe8b9ef933c3a9a581eb3ac1c01c6dd384ad490e6b37f519700cca060431`.
Runtime tests SHA256:
`4f7129b0bf5e62601a1e496bbeb9935a1edaa7e939c39158075970d6da654f48`.

- All actual 15 PFNN IDs: 000-007,009-014,urban. Each seed path/hash binds that
  manifest's own `source_scene.npz` and exact FP32 [1,4,366] history; no Root rebase.
- Explicit `warp_bvh_sphere_exact`, pinned 000 plane proof from v1 and fourteen
  rough-map v2 proofs; no legacy proof substitution or automatic backend choice.
- Manual Start/Reset only, at most 300s wall/control budget per run, 200Hz physics,
  50Hz SONIC, 30Hz display. Scope-bound episode/session/sequence/manifest checks.
- Two-slot CPU scene/asset and adapter/graph LRU caches, one shared controller;
  release only after quiescence. Packed assets capped at 128MiB, full geometry/no LOD.
  Scene Torch-storage accounting excludes unknown Warp/JIT/arena resident bytes.
- WASD full ticks use an independent four-pending NPZ queue, raw plans two-pending;
  all states/arrays retained, small JSON receipts, no array JSON in the control tick.
  Shared output-file byte budget 8GiB with 64MiB drain reserve, 4MiB numeric/1MiB JSON
  record bounds. Synchronous logging tail is included in the WASD chain receipt.
- Snapshot actual heading/quaternion/velocity are physical state. G1 reference
  preview stays distinct; no actor Root/pose correction or onset threshold change.

Final C CPU run: **93 passed, 17 physical/model cases deselected, 9.50s**.
New gauge retarget dependency `8fbcd960...`; MB adapter `f8c4898d...`;
geometry glue `37f19d41...`; Sphere helper `0b6230d2...` (include this new file
in parent source snapshots). UI/server hashes are James's frozen heading revision.
No known startup blocker remains in CPU contracts. Parent's first actual local
000 trial, movement-direction acceptance, browser FPS and all-map graph activation
are still pending actual measurement; this handoff is not a delivery/20ms PASS.
