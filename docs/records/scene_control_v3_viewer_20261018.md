# Scene Control v3 Performance Workbench

## Scope

Independent loopback workbench, port **28946**, with two explicit CLI modes:
`source` (default CPU factual reference) and `capacity` (random actor through the
shared actual geometry/encoder/native pipeline). Actor status is **UNTRAINED**,
optimizer updates are **0**, and motion-quality response is **unmeasured**.
There is no trained WASD controller in this page.

The capacity server imports `motionbricks.scene_control_v3.pipeline.CapacityPipeline`,
the same path used by the benchmark; it never imports the benchmark script. The
final scene contract is
`scene_features[B,512,16]`, index15 `distance_signed_flag`; unknown-scene
`input_valid=False` stops actor publication through the shared native validity
gate. Capacity accepts hidden profiles256/512/1024, with preview64 from the
performance config. Final policy fixes the query total at512:256 current
official surface points,128 previous committed predicted Root-plan points and
128 latest-intent probes. The world-space actor Root preview becomes the next
active plan only after the matching rendered ACK; initial plans use observed
history Root velocity. Per-frame receipts preserve origin, age, source counts
and time ranges in `query_metadata`.

## Launch After Parent Allocation

From `${PROJECT_ROOT}`, with CPU threads limited to two:

```powershell
$env:PYTHONPATH='.;out/vendor/kimodo'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:CUDA_VISIBLE_DEVICES='-1'
& D:/.anaconda/envs/motionbricks/python.exe scripts/soma_scene_control_v3_server.py `
  --bundle out/a100_live_20261010/a100-terrain.pt `
  --source-config configs/scene_control_v2_real_provider_cpu.json
```

Open `http://127.0.0.1:28946/`. Source-mode native conversion is CPU-only; browser GPU
rendering requires the parent's separate allocation. Source is the real
646-frame PARC retarget and its bound world scene; it ignores commands.

The selected workbench backend is CPU/eager/hidden1024 plus browser GPU skin:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& D:/.anaconda/envs/motionbricks/python.exe scripts/soma_scene_control_v3_server.py `
  --mode capacity --device cpu --scene true3d_clutter --hidden-dim 1024 `
  --max-frames 300
```

`--mode capacity --device cpu` selects the same actual pipeline on CPU. Source
does not accept a CUDA device; a CUDA capacity launch requires an allocation ID.
The workbench exposes only the explicit true3D-clutter capacity scene. Static
actor, native adapter and query meshes are constructed once and reused across
connections. Each connection gets a fresh pipeline with seeded history, zero
hidden state and its own one-pending-frame gate. The parent-authorized CPU1024
launch is recorded below. No browser was operated by this worker and no old
service was restarted.

## Source And Skin

- `load_bound_world` reads the accepted source, verifies its existing lineage,
  and retains world heading, metres, Y-up and original 30 Hz timestamps.
- Source ID: `generated_f733c63c38e13f58`; 646 frames; bound heightfield 76x147.
  No PFNN map is substituted. Unknown terrain cells have no triangles.
- `NativeSOMAAdapter.source_frame` supplies official77 world matrices. The
  immutable bundle provides native stats/metadata, not legacy model inference.
- Each binary frame is **4940 bytes**: float32 little-endian rootXYZ followed by
  77x4x4 **row-major** matrices. Three.js transposes each matrix on ingestion.
- All 18056 bind vertices, faces, inverse binds and native 4..8 influences are
  retained. Vertex and normal shader extensions match `full_corpus_lab`; weights
  are not truncated or renormalized. Root only moves camera/telemetry.
- `/api/scene-control-v3/reference` returns actual frame-0 official fullskin XYZ,
  separately from the timed stream. `window.sceneControlV3.referenceParity()`
  compares every vertex using the browser CPU mirror of the all-influence shader
  and float32 transport matrices. Actual GPU pixel proof remains parent-owned.
- Capacity renders the exact query triangles exported by the shared pipeline.
  The moving object's candidate transforms accompany the same rig frame and
  are applied together. No unrelated source/PFNN scene is substituted.

## Clock And Receipts

The **client presentation clock is the only 30 Hz clock**. After a rendered ACK,
the server immediately prepares/sends the next source frame, so it can buffer
before the next client deadline. There is one pending frame and one latest
command, not a command or frame backlog. Late frames never skip or catch up in
a burst. Pause/hidden tabs hold the pending frame; resume starts a fresh clock
segment. EOF stops after the original final source frame, without looping.

Metadata snapshots the latest input command ID, browser input timestamp, server
receive timestamp and preparation/publication timestamps. The client ACK carries
the exact session/packet/`frame_index`/command identity and ordered render-start/
render-completion timestamps. Only a successful actual `renderer.render()` call
can advance the rendered frontier; only its matching ACK advances the server
committed frontier. New commands never relabel an already published frame.

Capacity passes the actual received vx/vz/heading to `pipeline.propose`; these
enter point-path construction and actor input. Candidate history, hidden,
body-surface points, contact lifecycle and object poses remain uncommitted until
the matching rendered ACK calls `pipeline.commit`. Pause during preparation
retains the candidate in the paused client's one-slot buffer. A closed
connection abandons its pipeline without changing any other connection's seed.
Capacity contacts are explicitly synthetic capacity fixtures, not measured
contact truth or actor proposals. Capacity stops at its explicit frame limit.

The receipt button downloads and writes the **same single serialization** into
visible read-only `textarea#receipt-snapshot`; the textarea is never updated
per-frame. The lucide reference-check button pauses streaming and shows
`output#reference-parity`: all **18056** vertices and **max absolute per-axis
error in metres**, using the CPU mirror of the full shader transform. Source
checks frame0; capacity checks its native initial seed. This can be accepted by
normal CUA button clicks and DOM reads, without executing hidden application JS.

The receipt and `window.sceneControlV3.receipt()` retain bounded recent
samples (256 frames, commands and associations), counts, clock segment, source
lineage, native-rig timings, unique-rendered Hz, and first command-associated
render-submission P95. The display metric is successful renderer submission;
quality response is unmeasured. Pausing preserves the last continuous Hz segment
in the receipt. Only a real paused-to-running transition starts a new segment and
resets the presentation deadline; repeated visibility/pause events are idempotent.
Necessary inference is
**N/A** for source, not 0 ms or a passed inference budget. Capacity records the
actual necessary wall time around shared `propose`, including contact admission,
actual point query, encoding, actor, native history/rig, validity-gated payload
D2H and associated object-pose D2H. It reports necessary-chain P99; GPU native-rig
time is not separately inferred from host enqueue time. Proposed RTX3080 budgets remain
inference P99 <=20 ms and command-associated rendering P95 <=100 ms.

The diagnostics and reference HTTP endpoints are read-only. One same-origin
127.0.0.1 WebSocket session is admitted; there are no training routes or outbound
network calls. No IK, foot locking, collision projection, smoothing, fabricated
frames, source-root replacement, or old-page/service changes are introduced.

## Verification

```powershell
node --test web/scene_control_v3/protocol.test.mjs
& D:/.anaconda/envs/motionbricks/python.exe -m pytest tests/test_scene_control_v3_server.py -q
```

Frontend: eleven CPU tests pass. The separate real-WebSocket test runs
inside the Python suite, which allocates and closes an ephemeral loopback test
socket, runs the actual frontend ledger with a fake clock through Node, and
checks early buffering, factual ACK, pause/resume and close for **both modes**.
Python: fourteen focused tests pass, including capacity ACK-only history/hidden,
actual-command forwarding, unknown-scene rejection, static-resource reuse and
in-flight pause/close, bounded persistent closed-session history, and real async
pump failure with traceback logging and actual1011 close reason. These checks
preceded final UI measurement; no additional tests were run during final receipt
documentation.

CPU asset preparation confirmed 646 frames, 4940-byte packets, 77 matrices,
8 influences, 18056 reference vertices and the bound scene. C's separate native
receipt `out/scene_control_v3_20261018/performance/prepare_20261017T201904146292Z/receipt.json`
reports fullskin max-absolute error **0.0 m** at source indices 0/315/645 against
official skinning, outside timed loops. The final parent CUA short-duration
browser evidence is recorded below.

The actual CPU capacity-runtime smoke also passed: real16D scene features,
512 official surface samples, hidden256/preview64, 4940-byte payload, different
commands changed the actual actor hidden output, ACK-only commit and fresh
connection state with shared static resources. CUDA remained uninitialized.

## Authorized CPU1024 Run

On 2026-10-18 at 04:36:58 Asia/Singapore, the parent authorized and this worker
started only `http://127.0.0.1:28946/`, Windows process **40456**, with Hidden
window style. Port28946 had zero listeners immediately before launch; its actual
listener PID matches the new Python process.

```text
D:/.anaconda/envs/motionbricks/python.exe -u scripts/soma_scene_control_v3_server.py --mode capacity --device cpu --hidden-dim 1024 --scene true3d_clutter --max-frames 900 --port 28946
```

Environment: `PYTHONPATH=.;out/vendor/kimodo`, OMP/MKL/OPENBLAS2,
`CUDA_VISIBLE_DEVICES=-1`; startup sets Torch intra/inter-op2, FP32/highest,
TF32off. Shared Surface512 integration was explicitly confirmed with C.

The real HTTP config confirms CPU, hidden1024, preview64, actor8665966 parameters,
scene16D, all18056 vertices, 4940-byte one-frame rig transport, one-pending and
zero optimizer updates. The startup diagnostics have zero sessions; the stream
is paused until parent CUA plays it. Warp logs report no visible CUDA device
under the explicit CPU environment, and the HTTP service is healthy.

Launch evidence directory:
`out/scene_control_v3_20261018/viewer_capacity_cpu1024_20261018/`
contains `launch_receipt.json`, `config.http.json`, `diagnostics.http.json`,
`stdout.log` and `stderr.log`. No browser, display-latency or real rendered-Hz
proof is claimed by this startup receipt.

Latest prelaunch CPU checks: **22 passed** across server and new native tests
(including source/fullskin), plus frontend **8 passed**. C owns the separate
single final combined-v3 CPU preflight. Runtime and frontend code were frozen
before this authorized launch.

## Final Short-Duration Capacity Evidence

The final diagnostic-capable service is **PID179228**, still at
`http://127.0.0.1:28946/`: CPU/eager/hidden1024, preview64, mixed-query-policyv2,
one pending4940-byte rig frame, and a predeclared **300-frame / 10-second** limit.
The parent completed that limit normally with300 committed frames and no pending
packet or recorded failure. Service remains available for a reload to paused
native seed; no further worker restart was performed.

| Measurement | Observed |
| --- | --- |
| Unique rendered Hz, last256 actual frames | 29.994001 |
| Necessary pipeline P99, last256 frames | 14.465700 ms |
| Input-associated render P95,18 associations | 67.900000 ms |
| Render-submission P95, last256 frames | 0.300000 ms |
| Completion | 300 committed; final index299 |
| Query policy | 256 body +128 committed Root plan +128 latest intent |
| Seed full18056 CPU skin-mirror maxabs | 3.024308128e-7 m |

There are29 command messages including the initial zero command:28 subsequent
submissions, of which18 have actual rendered associations. Unassociated inputs,
including submissions after the limit, do not enter the association percentile.
This is a real short-duration **UNTRAINED architecture-capacity** measurement;
optimizer updates remain0, quality response remains unmeasured, and learned WASD
is not approved.

Final evidence directory:
`out/scene_control_v3_20261018/viewer_capacity_cpu1024_20261018_v3/`

- `browser_capacity_300_v2.json`:606983bytes, complete exported receipt.
  SHA256 `240ec5d9938f41ae923fac3e7f487c3b64911c32c1768a879171314e3983418a`.
- `desktop_capacity_300.png`:59626bytes, parent CUA desktop screenshot.
  SHA256 `ecb8da2b935f8d86679e9bc99dbb5268c641e90b69db4df122651fd7a760c66e`.
- `final_capacity_evidence_20261018.json`: compact final evidence and provenance.
- `closed_sessions.json`: bounded16-session persistent archive when sessions close;
  pump failures also log full tracebacks to `stderr.log` and show actual close
  code/reason in the client and its receipt.

The original383/900 interrupted-run artifact and all earlier service evidence
remain unchanged. Its CPU forensic reproduction and final project integration
are owned by C. No model, bounds, output Root, pose or sampling budget was changed
for the300-frame measurement.
