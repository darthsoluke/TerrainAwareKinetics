# Kinematic Frame Budget Review

Current outcome: [gait repair](reactive_gait_repair_20261021.md). Query/geometry
acceleration is implemented, but faster replanning exposes gait instability.
v4repairs analog cadence bypass and displays30.0106fps; stopping still fails.
Original28944restored/sharedPID11492. The review below is historical.

2026-10-21. Bounded source review and CPU-only parsing of existing reports.
No model execution, tests, GPU queries, browser/service requests, process changes,
training or runtime edits. The existing 28944/PID11952 viewer is untouched.

## Decision

The first optimization target is the eager scene-query path around the two
already captured neural stages. Keep the current dependency:

`coarse geometry -> Root -> predicted-path geometry -> four Pose passes + codec`.

Capture those exact query calculations and compose them with the existing raw
traced neural functions. Add official SOMA rig and four-frame history work to
the same dispatch, with one publication transfer. Do not substitute the G1
FullPlan neural function: its geometry conditioning is different.

The existing records support a plausible 33.33ms target, not a measured pass.
They also establish that simply reducing two internal predictions to one is
insufficient: a historical single-prediction complete packet took P50 69.43ms.

## Measured Work And Measurement Scope

Source: `out/pfnn_mixed_20261015/root_path_latency_v3/failure.json`,
`report.settings`, retained through v4/v5. These are historical RTX3080/map007
measurements, not a new measurement under the current external Unreal load.
The post-timing rolling-history assertion failed; the raw timings are retained.
Later own-motion/serial/prefetch checks are documented in
`docs/pfnn_inference_latency_review_20261017.md`.

Complete packet timing includes `session.plan`, publication and JSON encoding;
socket/browser waiting is excluded. Each condition has 20 warm samples after
three excluded warmups. Stage samples are separate CUDA-event intervals, six
per condition, aggregated by label across the packet.

| Work | Two predictions, refresh4 P50 / P95 ms | One prediction, refresh8 P50 / P95 ms |
| --- | --- | --- |
| Complete eight-frame packet | 137.973 / 144.543 | 69.433 / 75.570 |
| Coarse scene queries | 48.386 / 50.551 | 22.910 / 24.281 |
| Root graph, including its terrain encoder | 7.126 / 7.488 | 3.300 / 3.911 |
| Post-Root path queries | 37.756 / 40.959 | 18.030 / 19.613 |
| Four-Pose-pass + decoder graph | 22.882 / 23.842 | 11.434 / 11.945 |
| Final publication geometry | 5.255 / 5.542 | 4.843 / 5.310 |
| JSON encoding, median of raw wall samples | 8.618 | 3.211 |

CUDA-event intervals include host submission gaps and guard-induced waits between
their recorded events. They are not isolated kernel-duration sums. Stage and
complete timing use different samples; their percentiles are not additive.
In particular, the final geometry label does not include the internal observe
calls after each short subpacket.

The later `dual_horizon_latency_v1/complete.json` reports materially slower
complete packets: preview8/12/16 P50 212.643/226.383/231.992ms and P95
274.088/239.463/243.843ms, nine samples each. Coarse/path event intervals for
preview8 were 79.269/71.556ms across the two internal predictions. Its first
preview8 sample was 140.552ms. This record does not identify the cause of the
variation. Consequently 145ms is a useful historical case, not a current bound.
`dual_horizon_candidate_v1/timing_scope.json` explicitly excludes its original
assertion-heavy timings from production-latency evidence.

Instrumentation definitions: `scripts/profile_root_path_runtime.py:25-50`,
`:95-129`; `scripts/benchmark_dual_horizon.py:39-105`.

## Source Findings And Smallest Equivalent Changes

### 1. Query Execution Is The Main Uncaptured Work

`data/root_path_split_execution.py:204-218` executes native `_scene_inputs`,
Root replay, native `_path_inputs`, then Pose/decoder replay in sequence.
`data/soma_reusable_terrain_execution.py:35-54` builds two height observations
(actor and survey) plus 1536 volumetric samples every prediction.

`data/joint_sdf_observation.py:58-95` scans all merged prisms in 512-query by
1024-prism chunks. It creates distance candidates, reduces nearest indices,
reconstructs normals and checks map/hole certainty. More height variation can
increase the prism count. This is actual geometric work as well as many eager
launches; the reports do not isolate their individual shares.

`data/local_terrain_observation.py:147-161` repeats CUDA scalar guards and a
whole-heightfield finite mask. `:163-174` rebuilds invariant lattices;
`:184-190` and `:283-288` perform more data-dependent host decisions.
`data/unified_scene_field.py:114-115,158-159` and
`data/joint_sdf_observation.py:59` introduce further scalar synchronization.

Smallest change: specialize/capture the existing query tensor calculations per
bound scene and fixed8 horizon, with the current history/goal/Root as dynamic
inputs. Cache scene-finite masks and invariant axes at scene binding. Preserve
nearest-cell bins, prism traversal/tie order, normals, unknown masks and physical
query coordinates. Return dynamic status with the publication transfer instead
of repeatedly materializing CUDA booleans on the CPU.

`mb_sonic/geometry_execution.py:178-215` already supplies raw traced query
functions and explicit scene/storage binding. Reuse this composition pattern.
Its coarse-only callback must be extended with the current post-Root callback.
Start with separate captured coarse/path queries; outer composition then removes
the extra dispatch/copy boundaries if still needed. Avoid a new SDF, fewer points,
or caching results across changed histories.

The current path already reuses body/actual-velocity samples within one prediction
(`data/root_path_split_execution.py:199-202`, `data/root_path_terrain.py:73-77`).
That optimization is present and must not be counted again.

### 2. Avoid Building Skin Vertices That The Browser Does Not Receive

`data/root_path_terrain.py:131-146` captures full official skin vertices along
with rig/joints/history. `:172-176` queries diagnostic bone distances and then
discards `vertices`. `data/soma_cuda_graph.py:12-16,55-58` clones every returned
tensor before that discard, including eight frames of 18056 XYZ vertices
(1,733,376 bytes). The actual rig packet is only 39,520 bytes.

At capture, intercept the official skin function at its LBS input and return that
rig directly. The project already uses this exact boundary in
`scene_control_v3/native.py:21-22,44-48,61-66`. Preserve the official 30-to77
expansion and FK. The official boundary is
`out/vendor/kimodo/kimodo/viz/soma_skin.py:94-130`.
Keep full-skin parity evaluation outside the live publication path. This changes
neither the model output nor browser skinning, and avoids materializing/cloning
vertices that are discarded. The saved 4.843ms publication category also contains
FK/history/query work, so it is not all recoverable LBS time.

### 3. Stop Turning Every Prediction Into A Large Debug Message

`data/soma_terrain_execution.py:221-255` copies predicted paths/patches and
then builds Python diagnostics. `data/unified_terrain.py:25-33` separately
transfers all 1536 SDF points, distances and validity. Dual preview adds more
transfers/lists in `data/dual_horizon_execution.py:87-95`.

`data/pfnn_tuning.py:181-189` retains both complete internal observations plus
a top-level snapshot. The measured JSON is 479972-482035 bytes for refresh4,
versus 175719-176352 for refresh8, before the 39520-byte rig payload.
`data/soma_lab.py:95-98,125-140` adds separate root/rig/joint transfers and bone
diagnostics to the critical response.

Publish rig, required generation/frame/input identifiers, and the exact committed
motion history first. Keep full observation overlays on a requested diagnostic
channel; avoid duplicate top-level/internal snapshots. Compose one lossless tensor
payload and one GPU-to-CPU transfer. Diagnostic removal does not alter numerical
inference, although it intentionally changes telemetry and must retain the timing
and history identifiers needed to assess feedback.

### 4. Reduce Synchronization Boundaries, Not Immutable Ownership

`data/native_mb_online_live.py:304-310,372-374,400-405` synchronizes around
prediction/publication; `data/root_path_split_execution.py:205` reads a horizon
already known to host runtime controls via CUDA `.item()`.
`data/soma_terrain_execution.py:158-162,175-188,223-225` adds repeated finite and
history-check materializations. Pass the fixed horizon as host metadata and carry
dynamic checks into a combined status payload. Keep one completion point before
transmitting the ready rig. Preserve owned copies for any published or committed
arrays: deleting all `StaticCudaCall` output clones would let a later replay
overwrite old history (`data/soma_cuda_graph.py:55-58`). Raw traced functions,
rather than nested `StaticCudaCall` replays, are the outer-graph composition unit.

## Why The G1 FullPlan Timing Is Not A Drop-In Result

The requested approximately24ms G1 graph figure is context supplied to this
review. The explicitly inspected older two-block record
`mb_proposal_full_plan_v1/generation.json` reports 31.341ms CUDA elapsed and
the accompanying reference document reports 34.484ms total. Neither establishes
the current pure-viewer full chain at33ms.

`mb_sonic/mb_reference.py:653-677` does one coarse query before each neural call;
`:459-475` supplies that observation through Root, Pose and decoder. This file
does not install `PredictedRootObservation`. In the viewer, post-Root query and
encoder replacement occurs at `data/root_path_terrain.py:41-52,54-93`, or at
the explicit split boundary. Dual12/16 additionally constructs stations from
short and long Root hypotheses (`data/dual_horizon_execution.py:34-67`).

FullPlan does show how to compose query/neural/observe/FK into one graph and use
one payload (`mb_reference.py:680-683,764-775,832-847`). It returns native SOMA
FK, not the viewer's official77-joint skin rig. Preserve these semantic differences
when borrowing its scheduling. Its optional second AR block and its internal
`world[:,24:28]` history are forecasting, not fresh played feedback.

## Scheduling And Published-But-Unplayed History

`data/pfnn_tuning.py:156-194` calls the model `8/refresh` times before returning
one packet. Each call uses the same input command and advances a private history
from its own generated prefix. Refresh4 therefore means two internal forecasts;
refresh1 would mean eight, not eight chances to sample the user's latest input.

`data/unified_committed_playback.py:12-17,36-47` prefetches using the full pending
packet's end history, even when only0/2/4 frames have played. This is a
deterministic continuation conditioned on committed model frames. It remains
valid only while every predecessor frame is played unchanged. Input sampled at
prefetch cannot influence those remaining predecessor frames. The existing ACK
commit enforces that order (`data/soma_lab.py:301-333`). Actual displayed history
and this future boundary must remain separately identified.

The current browser locks a prefix at packet start, samples the input once at
prefetch, and acknowledges the whole packet:
`web/full_corpus_lab/app.js:1768-1787,1807-1823,2107-2152,2156-2176`.
For prefix4, four already committed frames still follow the sampling boundary;
the first successor-frame slot is five30Hz ticks after the just-applied fourth
frame (166.67ms, plus any missed deadline). This matches the browser's
`MAX_BLOCK_FRAMES-prefix+1` timing formula for a nonzero prefix.
An event arriving just after that sample waits for the next packet's sample.
Shorter internal refresh alone cannot remove this delay.

For real4/2/1 feedback, the executed prefix is the commit unit. Maintain the last
four actually applied model frames; for1/2 steps concatenate prior history and
newly applied frames before selecting the last4. Recompute the canonical feature
channels, not just Root or raw redundant decoder channels
(`data/native_mb_observed_history.py:15-43`). Never advance history to an
unplayed forecast, then cancel/replace that forecast. Tie each result to scene,
history boundary, input version and first executable frame. With strict
played-history feedback, compute/transfer must finish before the next execution
slot; an extra precomputed committed prefix buys time by spending responsiveness.

## Budget Estimate And Recommended Order

Historical one-prediction Root/Pose/geometry stage medians total about19.58ms;
their P95 values total21.17ms. This arithmetic describes the available scale,
not a predicted end-to-end percentile. At that scale a33.33ms update leaves
about12ms for both query stages, host dispatch, completion and transport.
The current query-stage medians total40.94ms, so those stages must change
substantially; deleting JSON alone cannot meet the goal.

An engineering target is roughly15-16ms neural work,3-4ms rig/history,5-7ms
captured geometry and2-3ms host/transport, about25-30ms total. These are estimates
to test, not measurements. No report establishes P99 or33ms under Unreal load.
The framewise mode should first use preview8; preview12/16 adds a real second
Root pass and different sampling, so it needs its own budget and parity check.

Recommended order: finish genuine prefix-sized feedback in the main work;
capture existing coarse/path queries; use rig-only official publication; combine
one dispatch/status/transfer and compact live metadata. Only if kernel time still
dominates, consider the existing FP32 convolution/GEMM opportunities documented
in `docs/mb_sonic_neural_ops_profile_20261019.md`. Preserve8tokens, four Pose
passes, exact query masks and active-token/FK/history/skin equivalence throughout.

## Reviewed Source Identity

SHA256 at this review (concurrent main edits may subsequently change lines):

| Source under `motionbricks/data/` | SHA256 |
| --- | --- |
| root_path_split_execution.py | 57ea57b2a1c778f6077fce7b8a66577a17b677356328ef35a216353393b56f66 |
| dual_horizon_execution.py | e81492b3938640ff4fbb54f1664bafd1d38a2c2e4a00216355a9a0d007f239f9 |
| native_mb_online_live.py | 90cb0712bf44d50d749781ae34d13a452ee4d394db38a08223e95dbd46315457 |
| soma_terrain_execution.py | c50f6a99a528efefce14a7b03109e1199cb57034fa5d048809dc0cd0fd76abe0 |
