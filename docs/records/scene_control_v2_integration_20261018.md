# Scene Control V2 Integration, 2026-10-18

## Owned Files

- `motionbricks/scene_control_v2/__init__.py`
- `motionbricks/scene_control_v2/contracts.py`
- `configs/scene_control_v2_engineering_8x10.json`
- `configs/scene_control_v2_research_128.json`
- `scripts/preflight_scene_control_v2.py`
- `tests/test_scene_control_v2_integration.py`
- This report.

Other workers' implementation files and existing dirty changes were preserved.
No SSH, GPU kernels, optimizer construction/steps, old training, downloads,
or changes to services 28943/28944 were performed by this worker.

At the parent's explicit follow-up, two factual corrections were also made:
`docs/training_inference_config_snapshot_20261018.json` and
`docs/training_inference_architecture_review_20261018.md`. They had mistaken the
370-D dual normalization length for global feature width. CPU mmap seed shape
and native representation construction confirm global 366, dual/statistics 370.

## Configuration And Budget

`load_run_config(path)` resolves `motionbricks.scene_control_v2.run.v1` without
creating an output directory. Both profiles bind the actual bundle/checkpoint
SHA256, SOMA representation, frozen normalization, explicit 8-head codec,
query policy/quota, contact IDs, source lineage requirements and quality gates.

- Engineering: exact final-checkpoint 8x10 VQ weights, native encoder/decoder.
- Research: native encoder/decoder warmstart plus a NEW UNTRAINED 8x128 codebook.
- Explicit heads also support 256 entries. Tokens stay `[B,L,H]`; no combined
  vocabulary or packed index path is called by the new codec.
- CPU preflight, diagnostic and formal accounts are separately zero. This
  configuration authorizes preflight, not the complete training objective.
- The completed legacy 38.4M exposure count is immutable and separate.
- The parent's separately assigned training-worker diagnostic (4 Adam, batch 2,
  8 engineering-fixture examples, formal 0) is not executed or counted here.
- New formal training remains unallocated. No automatic resume or expansion.

## Shared Interfaces

`ContactKind`: support=0, brace=1, grasp=2, touch=3, slide=4.

`numpy_sample_to_torch_batch(sample, *, queries=None)` copies one paired NumPy
sample into CPU tensors with a leading batch axis. Float32, int64 and boolean
masks remain distinct. Source strings, provenance and ordered names remain
metadata. It does not turn paired future labels into actor guidance.

**Object IDs remain persistent entity IDs.** `contacts.object_id` and query IDs
are unchanged, including non-contiguous IDs. The additional
`contacts.object_table_index` and query `object_table_index` /
`target_object_table_index` are explicit derived indices, never substituted for
entity IDs. The loss worker's object-transform gather must resolve entity IDs
through `scene.object_ids`. A `[7,42]` swapped-table integration test covers it.

`TorchSceneQuery(scene_query, *, scene_metadata, mode, as_of_s)` is a causal and
batch-shape wrapper over the data worker's REAL `query_world_torch`; it contains
no second distance kernel or duplicate autograd implementation. Call with
float32 predicted points `[...,3]`, aligned long time indices on the same device, explicit
`timestamps_s`, and optionally `assigned_object_id` for contact-surface queries.
Body queries use minimum component SDF; contact queries address their assigned
object rather than whichever solid is nearest. Point gradients remain attached.
Assigned IDs and all result tensors follow the points device. Provenance time
indices are copied to CPU for admission, and the backend still synchronizes to
CPU BVH queries; this wiring does not claim query acceleration. The optional
CUDA VJP/device regressions require `SCENE_CONTROL_V2_CUDA_TESTS=1` and a separate
parent GPU allocation. They were not executed by this integration worker.

Actor admission uses every `scene.object_track_provenance[O]` record, not the
descriptive scene-wide transform role. Offline teacher tracks are rejected;
predictions/schedules must already be available; factual tracks cannot be queried
in their future; static tracks must actually be constant. SceneQuery and paired
scene transforms/validity/IDs are checked as one binding.

Full-scene queries retain `valid_semantics=all_declared_components_known` from
the geometry backend. An unknown declared component invalidates the scene point
and its VJP. Assigned-surface queries explicitly return
`valid_semantics=assigned_object_component_known` and do not claim full-scene
completeness. A cross-module regression tests both scopes at the same point.

The derivative comes from signed `(point-closest)/distance`, including edge and
corner cases, not triangle normals. It is first-order point differentiation,
with fixed meshes and transforms. Nearest-branch ties/zero-distance points use
the geometry module's declared branch/mask semantics. Overlap interiors retain
minimum-component semantics, not exact-union boundary distance.

## Coordinate Bridge

The actual native dimensions are global 366, dual-root 370, local 365, encoder
268. The bundle has 370 normalization statistics and a REAL 366-D four-frame
global seed. A provider cannot append invented columns to convert this seed or
the 13-region fixture into a native training window.

`NativeWindowFrame.from_global_history(history, vertical_origin_m=...)` binds
first-frame XZ, observed terrain vertical origin and initial heading. Native
rotations and XZ velocities retain heading: geometric restoration is translation
only; initial heading is restored separately to integrated Root heading channels.
An extra whole-body yaw would double-rotate the geometry.

`center_global_motion` translates actual native global/dual motion.
`paired_batch_in_window` translates paired body labels, scene object transforms,
bounds, query positions and valid closest points together. Object-local anchors,
normals and identities remain unchanged. The batch carries the explicit frame.
Training `world/history` are actual canonical global 366-D features; Root
conditions and scene observations must share that frame. No four-column padding
or reinterpretation of global features as the 370-D dual representation is used.
This function does not manufacture native motion, Root targets, recovery targets
or frozen Pose labels from region positions.

The source-seed check changes heading to 1.1 rad and translates by
`[2.4,0.65,-1.7]` m. Local-to-global restoration and selected official SOMA FK
are compared with the actual seed's global FK. Observed maximum error was
`4.76837158203125e-7` m, with 6 selected skin vertices; no fullskin gate is run.

## Actual Execution

From the project root:

```powershell
$env:PYTHONPATH='.;out/vendor/kimodo'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:/.anaconda/envs/motionbricks/python.exe' -m scripts.preflight_scene_control_v2 --config configs/scene_control_v2_engineering_8x10.json
& 'D:/.anaconda/envs/motionbricks/python.exe' -m scripts.preflight_scene_control_v2 --config configs/scene_control_v2_research_128.json
```

Both real CLI profiles were executed successfully: zero Adam/training exposures,
real native codec reconstruction `[1,4,365]`, explicit indices `[1,1,8]`, and
unchanged EMA/codebook buffers. The 8x10 decoder matches the legacy native codec
at `atol=1e-6, rtol=0`. Bundle and final checkpoint contents are hashed before
CPU mmap loading. No full Root/Pose allocation or live actor is required.

Stdout is one JSON receipt; third-party initialization messages go to stderr.
No checkpoint/config/report/output directory is written. Python bytecode writes
are disabled by the CLI. Warp can populate its ordinary compiled-kernel cache;
all its query arrays and launches are CPU. The CLI clears CUDA visibility for
its process; Warp initialization may still list driver devices, without GPU
query arrays or kernel launches.

Final cross-module run across integration/data/geometry/codec/losses/training/
runtime: **144 passed, 9 upstream warnings, 15.99 s**. This includes global366
canonical training input, per-object provenance, sparse/swapped IDs, unknown-scene
versus assigned-object scope, finite-difference point gradients and engineering
label-lineage isolation. Warnings are upstream TorchScript deprecation and nested
tensor API notices. A transitional training-test edit failure was reported to
its owner and fixed before this final run; no worker file was edited here.

The integration command used `-p no:cacheprovider` and
`PYTHONDONTWRITEBYTECODE=1`. The separately assigned four-Adam diagnostic helper
is not a pytest test and was not called during these runs.

Final follow-up after the same-device wrapper change: **19 passed, 2 skipped,
8 upstream warnings, 11.79 s**. Only the integration suite and the three
unknown-transform/unknown-component geometry regressions were rerun. The two
skipped cases are opt-in CUDA tests; no GPU allocation or kernel was used here.
Execution-wide accounting and runtime evidence are now collected in
`docs/scene_control_v2_execution_20261018.md` and
`out/scene_control_v2_20261018/execution_status.json`.

## Real Versus Fixture Evidence

- REAL: existing bundle/checkpoint content, original frozen normalization,
  four captured source frames and their recorded shard lineage, final codec
  encoder/decoder and 8x10 codebook tensors, real SOMA FK and official skinning.
- SYNTHETIC: 8-frame, 13-region, low-ceiling/overhang/moving-brace fixture;
  explicit offline teacher guidance; 512 queries (128/256/128); controlled point
  perturbation and differentiable contact/gap/slip/penetration checks. No new
  real paired entity or accepted training sample is loaded.
- UNTRAINED: new contact head and 128-entry candidate codebook.

The fixture data/query -> tensor batch -> assigned-surface/body SDF -> physical
codec-loss functions -> point-gradient path is actually exercised. Native codec
forward and non-origin real-seed frame restoration are exercised separately;
the synthetic region fixture is not relabeled as a native SOMA motion dataset.

## Parent Review Items

- The integration `[7,42]` test now passes against the fixed loss module: gap,
  anchor and nonzero slip remain unchanged after table reordering.
- Supply accepted real paired native windows, reliable contact events/intervals,
  actor-predicted scene tracks and matched recovery teacher data. No provider is
  inferred from public metadata or synthetic region points.
- Finalize codec/scene-encoder identities, normalization and label manifests
  before Pose labels/training. The run profiles intentionally have no final label
  identity; the training CLI's separate training-config schema needs real receipts
  and a separately approved allocation, not fabricated hashes.
- `make_training_label_lineage(..., data_kind=...)` is called explicitly in an
  integration fixture with genuine engineering tensor hashes. Missing
  `data_kind` is rejected; engineering cached labels cannot authorize a formal
  run, and admission is tested with Adam construction forbidden. No diagnostic
  Adam branch is re-executed here.
- The data worker's real PARC record remains validation/engineering only. Its
  old canonical native coordinates differ from retarget world by up to 1.464 m.
  A real window provider must begin with raw retarget-world roots/rotations/joints
  and the bound world scene, explicitly preparing the window frame in memory.
  That single-record availability audit is not rerun by this integration worker.
- The training worker's original four-Adam dual370 engineering artifacts remain
  historical. New canonical global366 conversion/forward checks use zero Adam;
  their receipt is owned by that worker, separately from this preflight.
- The runtime worker's SOMADualBackend is an engineering adapter around the
  existing native Root-path chain. These profiles do not publish or claim a
  trained v2 codec/contact-conditioned actor. Runtime/GPU proof remains parent-owned.
- Full 18056-vertex, 8-substep, peak-penetration 0.02 m acceptance, contact/task
  quality, complete routes and command latency remain required. Every preflight
  receipt remains `quality_approved=false`.
