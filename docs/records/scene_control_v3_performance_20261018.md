# Scene Control V3 Performance Prototype

Current final status and training milestones are in
`docs/scene_control_v3_execution_20261018.md`; authoritative integrated evidence is
`out/scene_control_v3_20261018/performance/final_integration_v2/receipt.json`.
The final 256/128/128 committed-Root policy was remeasured only at H1024 across
four scenes: P99 12.218413/10.469507/10.301757/11.491062 ms, with 93 CPU tests
passing and 3 GPU tests skipped. The 300-frame browser segment has 18 associated
samples/P95 67.900 ms and last256 necessary P99 14.465700 ms. The original
383/900 interruption remains failed/incomplete. One exact CPU replay using the
parent's original CUA AX command transcription reproduced frame383: untrained
upward Root/body drift moved all512 queries above the known Y<=4 m domain and
the publication validity guard rejected the next frame. See the execution status
and `performance/root_cause_repro_383_v1.json` for complete evidence.
Earlier tables and v1 receipts below retain their historical policy/version.

## Scope

This independent prototype measures `architecture_capacity_only` with an
`untrained_capacity_probe`. Optimizer updates and new training are zero. The
existing formal 38.4M exposures, v2 services/pages and old quality diagnostics are
untouched. Raw global366 conditioning is a capacity fixture, not a trained policy.
Quality response and usable learned WASD are unmeasured.

The proposed RTX3080 10GB budgets are necessary-path P99 <=20 ms and
command-associated display P95 <=100 ms at 30 Hz. The CLI measures the necessary
server path through device-to-host binary packing. Network delivery and browser
display require command-ID-correlated acknowledgements; their result remains
null in this CLI. Full client skin quality and learned control retain the original
future acceptance gates, including full18056, eight substeps and 0.02 m peak
penetration. The new geometric representation does not establish those results.

## Native Boundary

`motionbricks/scene_control_v3/native.py` maps the immutable bundle archive and
reads only statistics, bone metadata and the genuine four-frame seed. It never
constructs the old Root, Pose or tokenizer networks. All base assets are reused
by explicit path and SHA256 in `configs/scene_control_v3_performance.json`.

`NativeSOMAAdapter.step(history, actor_output)` uses world XYZ root displacement
in metres per 30 Hz step and world-global native column6D rotations. The existing
native inverse/FK and feature APIs derive the authoritative body, hips heading,
RIC and velocity channels. Heading delta remains an auxiliary prediction; it
does not override body rotations. Native four foot channels remain unknown zero
forecast channels, with no invented 15-slot-to-four-foot contact mapping.

The returned state contains global366 `[B,4,366]`, local365 `[B,4,365]` and dual370
`[B,4,370]`. No source future root, IK, foot locking, collision projection,
smoothing or manufactured frames are applied. Only actual predicted frames
advance recurrent history.

The adapter instantiates `OfficialSurfaceSampler` for 512 actual official vertex
IDs. It applies every official influence through sparse official LBS without an
18056-vertex intermediate. Each predicted rig produces next-body surface points
inside the timed native stage; only its rendered ACK makes them the next query
base. The uniform index selection is a capacity selection, not a complete
anatomical contact layout.

The official Kimodo `SOMASkin.skin()` generates the 77-joint palette. An
instance-local LBS boundary returns those original transforms without doing full
skin on the server hot path. `fullskin()` calls official LBS on flattened B*T
frames and restores the batch shape. It retains all 18056 vertices and every
official influence. The official 77-skeleton neutral-joint dtype is preserved.

The wire remains `soma77-rig-v1`: little-endian float32 root XYZ followed by
77 row-major 4x4 matrices, exactly 4940 bytes per frame. Actor publication packs
the validity flag and geometry in one device-to-host transfer and rejects
unknown scene inputs. The binary matrix payload is exact.

## Fixed Contract

Inputs are history `[B,4,366]`, command `[B,3]` containing world vx/vz and heading,
contact state `[B,15,16]`, contact validity `[B,15]`, scene features `[B,512,16]`,
scene validity `[B,512]` and hidden `[B,H]`. Boolean masks are explicit. Scene
feature indices 0..14 retain the agreed XYZ/distance/normal/body/time/closest/
velocity layout; index15 is `distance_signed_flag`. No 15-feature compatibility
branch exists. Actor `input_valid[B]` and geometry `scene_known[B]` gate publication.

Profiles H=256/512/1024 have respectively 1,059,694 / 2,808,686 / 8,665,966
parameters. The default preview is 64 **30 Hz frames**, approximately 2.13 seconds,
not VQ tokens. Widths share the architecture and output contract; preview length
does not add parameters. Timing results determine latency/capacity eligibility,
not deployment quality. Budget comparisons are backend-independent. At least 30
warmup and 100 measured steps are required for capacity-statistics qualification;
smaller preparation/pilot runs retain `few_samples_not_qualified` classification.

## Shared Pipeline

`motionbricks/scene_control_v3/pipeline.py` contains the actual eager capacity
path used by both the benchmark and the v3 capacity server. The server never
imports the benchmark script. Actor, native adapter and geometry are shared
read-only resources; each connection gets its own `CapacityPipeline` and seed.
`propose((vx_world, vz_world, heading))` returns a validated 4940-byte candidate
and `next_state` without advancing history or hidden state. A second proposal is
rejected while one frame is pending. Only `commit(frame_id)` after the matching
rendered ACK installs that state. A duplicate ACK returns false without another
advance; an unrelated ACK is rejected. Commit performs host pointer assignment,
not GPU computation. Reconnection constructs a fresh per-session pipeline.

The pipeline exports the actual query mesh vertices/indices and object transforms
for rendering. Each candidate carries its corresponding object poses, which the
viewer applies together with that rig frame. Synthetic contact observations are
explicit capacity fixtures. Source replay remains a separate mode.

The benchmark keeps timing, GPU allocation and command fixtures in the CLI.
Five stage callbacks end contact/transfer, query/point construction, encode, actor
and native work; final wall time includes validity-gated transport and an explicit
simulated immediate display ACK. A real browser ACK is measured separately by
the server's command-associated display path.

## Geometry

All PFNN assets retain their complete node resolution and finite-cell triangles:

| Scene | Heightfield Z,X | Triangles |
| --- | --- | --- |
| PFNN011 | 954,1109 | 2,111,848 |
| PFNN007 | 1332,593 | 1,575,904 |
| Flat | 321,321 | 204,800 |

PFNN distance is nearest **unsigned open triangle-ground surface** distance. It is
not a pointwise acceleration of the old piecewise-constant prism SDF. True3D
clutter uses independent closed floor, overhead and side-wall meshes. Geometry
reduces minimum component distances, using winding-signed distance only for
validated closed solids. Negative interior distances take precedence over a
closer positive floor. This component reduction is not exact union-CSG distance.
The signed flag tells the actor which semantics generated a selected hit.

The final query policy is `committed_root_preview_256_128_128_v2`, described below.
Its 512 points combine current official surface samples, the previous committed
actor Root preview and immediate latest-intent probes. Query body-part IDs are
explicit opaque capacity groups, not anatomical contact labels. True3D object
poses move consistently with their nonzero supplied surface velocity. Missing
geometry is rejected, never replaced with flat ground.

## Measurement

Default execution is CPU preparation. CPU tests use two threads and hide CUDA
devices before Warp initialization. A small CPU timing mode exercises integration.
CUDA mode requires the parent's explicit allocation ID and stays on GPU0 RTX3080.
No SSH or service process is started by this runner.

The necessary wall timer includes CPU contact observation admission and feature
packing, prior proposal D2H, contact H2D, command/query-point construction, object
pose updates, Warp query, scene feature encoding, one actor forward, native rig
and history derivation, publication validity, binary packing and state commit.
Synthetic contact observations exercise active/release/rebinding transitions as
an explicitly labelled capacity fixture. Actor proposals never become factual
observations. Stage wall times and CUDA stream intervals are separate. CUDA
stream intervals can include host dispatch gaps; they are not pure kernel time.
CPU preparation and independent official fullskin checks are outside that timer.

Default sweep: four scenes, three widths, 100 warmup and 300 measured steps per
case, one bounded run of at most 180 seconds including cold construction.
Incomplete cases retain counts and receive no budget approval. Three real
fullskin samples per case are measured separately. Eager is primary; there is no
compiler or graph fallback and no automatic rerun.

Before a full sweep, the parent first allocates B's small CUDA correctness tests,
then a separate C pilot: true3D/256, 10 warmup and 30 measured steps, at most 30
seconds. The full sweep is not started automatically. A slow pilot is inspected
by stage before any fixed-shape execution change and subsequent measurement.
Experiment failures write `failure.json` plus a partial `receipt.json`, preserving
phase, completed counts and samples. This records a failed experiment; it does
not replace execution with another implementation.

Baseline actor/outputs use float32, matmul precision `highest`, TF32 disabled for
matmul and cuDNN, and cuDNN benchmarking disabled. Receipts record the actual
deterministic switches and dtypes, including the official skinner's preserved
float64 77-skeleton neutral joints. Low-precision work would be a separate profile.

Memory receipts include Torch allocation/reservation peaks, Warp pool current/
high-water values, explicit geometry/index/bounds storage, device free memory
before/after cold construction, and whole-device free samples every 30 steps.
They also record nvidia-smi GPU UUID/utilization/process data and Windows process
command lines before/after the run. The Torch 0.25 memory cap excludes Warp BVH.
Non-Torch free-memory deltas include Warp/BVH, context/driver and concurrent
desktop changes; they are not isolated BVH bytes. Whole-device sampled minima
are not a continuous peak. This is a controlled shared-desktop baseline, not an
exclusive-GPU claim.

## CPU Evidence

The preserved first preflight failure used a 2e-6 fullskin threshold and found
3.814697e-6 error on the far-world source frame. The adapter's extra float32
conversion of the official77 neutral skeleton was removed, and LBS batch packing
was aligned with the official frame dimension. The threshold was not widened.

`out/scene_control_v3_20261018/performance/prepare_20261017T201904146292Z/receipt.json`:
five native CPU tests passed; genuine source646 frames 0/315/645 have exact root
and fullskin parity (maximum absolute error 0), and source-joint maximum absolute
error 1.907349e-6 m. The original root/FK source gates remain 1e-5 / 3e-5 m.
That preparation predates final 64-frame/contact-admission integration; subsequent
receipts carry the complete updated pipeline.

`out/scene_control_v3_20261018/performance/cpu_20261017T202121123244Z/receipt.json`
completed all three widths with 2 warmup/5 measured steps, true3D 512-query and
64-frame preview. New v3 CPU preflight passed 56 tests with 3 CUDA tests skipped.
All three capacity cases have exact fullskin parity, 512 valid final queries and
zero optimizer updates. These small CPU counts establish integration only.

`out/scene_control_v3_20261018/performance/cpu_20261017T203012140714Z/receipt.json`
is the updated shared-pipeline, actual-surface run: 67 new v3 tests passed and 3
CUDA tests skipped. All three widths completed 2 warmup/5 measured steps with
64-frame previews and 512 actual official vertex samples. Root, full18056 and
selected512 source parity are all exact (maximum absolute error 0); all capacity
fullskin/surface checks also pass. The timer includes sparse surface LBS and
ACK-committed state through the same pipeline used by the new capacity server.

## Selected Backend

The parent accepted the measured capacity envelope and selected explicit
**CPU eager, H1024, preview64, true3D**, two CPU threads, float32/highest matmul
precision/TF32 off, for the capacity browser demo. The reference workstation CPU
is Intel Core i5-13600KF (14 cores, 20 logical processors); tests use two threads.
The immutable CPU evidence is
`out/scene_control_v3_20261018/performance/cpu_20261017T203333819266Z/receipt.json`
(SHA256 `acccd0545c446337c7e8ecba2d33e0bb70d875504a260723f48c1ccb125470ed`).

The following 12-case table is historical evidence for the prior command-only
query policy. The final 256/128/128 revision has its own H1024-only measurement;
it does not reuse these timings as current-policy approval.

All 12 scene/width cases completed 30 warmup and 100 measured steps; all P99
necessary-path times were 8.764..12.447 ms. The selected 8,665,966-parameter H1024
profile measured:

| Scene | P50 ms | P95 ms | P99 ms | Unpaced fps |
| --- | ---: | ---: | ---: | ---: |
| PFNN011 | 9.683 | 10.264 | 10.551 | 102.66 |
| PFNN007 | 9.814 | 11.293 | 12.027 | 99.30 |
| Flat | 8.322 | 9.213 | 9.517 | 118.59 |
| True3D | 9.213 | 9.855 | 10.441 | 108.15 |

Every case had 512 valid final surface queries. All 36 separate full18056 and
selected512 checks have maximum absolute error 0. Geometry compile options are
explicitly `enable_backward=False`, `fast_math=False`, `fuse_fp=False`. This
profile passes the necessary compute budget; command-associated display and
learned quality remain separate measurements. The backend is configured in
`selected_backend`, not changed on inference exceptions.

Final combined CPU integration evidence:
`out/scene_control_v3_20261018/performance/final_integration_20261018/receipt.json`.
Exactly one final combined pass completed: **90 passed, 3 GPU tests skipped** in
16.34 seconds. The receipt links the immutable CPU envelope and GPU pilot hashes,
records the selected backend and server/frontend source hashes, and confirms all
seven frozen native/model/geometry/pipeline/surface hot-path files match the
measured CPU code. Source root, full18056 and selected512 parity remain exact.
The geometry source SHA is
`d301104e8ef29d41ed023f9d3774177672ff87f19f187bf05a1776a7f6d630de`.

The parent will append separate browser command-associated display and reference
frame evidence. Those fields remain unmeasured in this integration receipt until
that evidence arrives; no training or learned-quality approval is recorded.

The subsequent client-only revision is recorded separately in
`out/scene_control_v3_20261018/performance/client_revision_1_integration_20261018.json`.
It updates initial capacity labels, viewport/aside layout and the default camera;
the client revision receipt reports nine passing frontend tests. SHA checks show
all seven measured hot-path files, the benchmark harness and server are unchanged.
The original 90-test integration receipt remains intact, with its original client
hashes. No combined suite or model run was repeated. Parent reload/visual and
command-display acceptance remain pending for the revised client.

## RTX3080 Pilot

Allocation `capacity_eager_pilot_v1` completed exactly once, true3D/H256/64-frame
preview, 10 warmup and 30 measured steps, with all necessary costs and actual512
surface LBS. Receipt:
`out/scene_control_v3_20261018/performance/cuda_20261017T203143935953Z/receipt.json`.

| Wall Stage | P95 ms | P99 ms |
| --- | ---: | ---: |
| Contact CPU admission/pack/transfers | 1.087 | 1.499 |
| Query and point construction | 0.702 | 0.781 |
| Encode | 0.923 | 1.096 |
| Actor | 1.307 | 1.707 |
| Native rig/history/512-surface LBS | 20.641 | 23.298 |
| Transport/validity/commit | 0.587 | 0.791 |
| Total | 24.520 | 26.774 |

Total P50 was 22.337 ms and unpaced measured throughput 44.14 fps. The necessary
P99 <=20 ms budget was not met. Full18056 and selected512 parity are exact. New
CPU preflight passed 78 tests with three GPU tests skipped. The process exited and
GPU resources were released. No full GPU sweep or CUDA compilation followed.

The fresh nvidia-smi baseline was 5309 MiB free and 25% utilization. Torch peak
allocation was 31,157,248 bytes and Warp pool high-water 16,875 bytes. CUDA driver
free-memory samples report a different WDDM view (about 9.4 GB free) from physical
nvidia-smi accounting (about 5.3 GiB free); these remain separate. Raw before/after
physical snapshots are preserved. Future receipts name the driver-view sampled
minimum explicitly rather than treating it as physical whole-device free memory.

Parent-reported PRO6000
environment readiness and a completed 64x64 hardware matmul smoke are metadata
only, not new-actor performance. No remote environment change is needed.

## Reproduction

Run from the motionbricks workspace with the supplied Python environment:

```powershell
$env:PYTHONPATH='.;out/vendor/kimodo'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
& 'D:/.anaconda/envs/motionbricks/python.exe' scripts/benchmark_scene_control_v3.py
& 'D:/.anaconda/envs/motionbricks/python.exe' scripts/benchmark_scene_control_v3.py --mode cpu --scenes true3d_clutter --hidden-dims 256 512 1024 --warmup 2 --steps 5
```

Only after the parent allocates the GPU, the same CLI accepts `--mode cuda
--allocation-id <reservation>`. Every run creates a new receipt directory under
`out/scene_control_v3_20261018/performance/`; previous failures and evidence remain
intact.
