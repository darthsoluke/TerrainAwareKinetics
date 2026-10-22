# PFNN MB Scene Binding

Date: 2026-10-19. Scope: the frozen trained MB adapter's geometry installation
and fixed-scene graph ownership. Runtime/cache/WASD/viewer integration belongs to
C and the parent. No GPU, SSH, server, browser or training was started here.

## Explicit Deployment API

Both `TrainedMBReferenceAdapter.load` and its constructor accept:

```python
scene_query_backend="native"       # unchanged default algorithm
scene_query_proof=None
```

For the admitted PFNN backend, explicitly provide:

```python
adapter = TrainedMBReferenceAdapter(
    shared_controller,
    canonical_scene=canonical_scene,
    identity=model_identity,
    scene_query_backend="warp_bvh_exact",
    scene_query_proof=proof_path,
)
```

`proof_path` is the actual existing per-map file under:
`${PROJECT_ROOT}/out/pfnn_mixed_20261015/pfnn_map_catalog_v1/<artifact>/geometry_query_review.json`.
Use each `catalog.json` entry's `artifact` value. The 15 existing map IDs are
`000,001,002,003,004,005,006,007,009,010,011,012,013,014,urban`; `008` is not a
catalog entry. Proof files are not in the canonical `mb_sonic` scene directories.

Installation checks the canonical source bytes, then reads the retained proof's
source SHA, helper SHA, 62-point count, zero distance/normal errors, exact valid
mask and zero optimizer updates. It does not execute those comparisons again.
The unchanged helper SHA is
`ccdb3c55616e76999e4a8821d4fe83c0e14888bd24c413ea848746483000ca00`.
`geometry_complete_v3.json` remains unchanged at SHA
`2af1790276366f9889832f30a7b5fced57813ddea125de6f541fc870437f180a`.

Fully known constant terrain keeps the exact existing bounded `AnalyticPlaneField`.
Rough terrain calls the existing `BVHHeightfieldPrismSDF` on FP32 CUDA with the
proven Warp 1.18.0. Missing/mismatched proof, unknown terrain, wrong precision or
unallocated CUDA on a rough map raises an error; it does not select native brute
force. Original full-resolution heights, source origin, `dx`, bounds and masks
are retained. There is no query-math replacement or map resampling.

## Cache And Graph Ownership

- `adapter.scene_field` is the strong owner of the exact installed field. The
  same object is registered in `controller._fields` for native and compiled calls.
- `adapter.activate_scene()` reinstalls that owner after a serialized cache hit,
  without rebuilding trees. Preparation and generation also activate it.
- `adapter.scene_graph_bindings()` holds the scene tensors, field owner, live
  field state, sampler/config, source identity and backend. BVH bindings also hold
  the field ID, both tree objects and their native IDs. `FullPlanExecution` uses
  this binding for capture and checks a fresh live binding before replay.
- `adapter.scene_query_receipt` is JSON-safe provenance, including proof/helper
  hashes, field kind, source hash, tree IDs and storage-accounting scope. The
  existing `prepare_compiled` report adds it as `scene_query`.
- `adapter.release_scene()` removes only its own controller cache slot and its
  compiled/native-prepared/full-plan references, then marks the adapter released.
  It works before preparation as well. The shared controller/model and other
  scenes remain live. The caller must stop/join pending work and ensure replay is
  quiescent before release/eviction, then drop the adapter. External retained graph
  bindings continue to own their field/trees; released adapters cannot reactivate.

C owns the bounded LRU, keyed by the canonical manifest/source identity. Use
activation on a hit; evict/release before constructing a replacement to bound the
peak. Cache admission remains scene-specific: reuse a prepared graph only with
its original scene, source, backend, field/tree identities and model contract.

`adapter.accounted_bytes` counts distinct CUDA Torch scene/field storages, with
views deduplicated. It excludes graph arenas and traced neural constants. Warp
1.18.0 exposes no public BVH byte-count API: `warp_bvh_bytes` is `None` for BVH,
not an invented zero. C must retain its slot bound and separately report measured
resident/arena limits. Existing neural trace serialization can own additional
model constants per graph; one shared controller does not eliminate those copies.

## Preserved Contracts

Native Root/Pose/codec, four Pose passes, eight-token horizon, original FP32/TF32
policy and frozen statistics are unchanged. SOMA inverse and native played-history
re-encoding are unchanged. Existing exact token, query `1e-5`, final FK `3e-5`,
history `2e-5`, model-digest and CPU/CUDA RNG gates remain in preparation. There
is no convolution optimization in this change.

## Verification

New CPU test: `tests/test_mb_sonic_scene_binding.py`.
Result: **PASS**, 21 passed in 2.89 s. It reads all 15 actual source/proof pairs,
checks exact canonical/catalog height/origin/spacing/seed arrays, verifies opt-in
and rejection paths, tests native plane equality, field reuse, live identity
checks, release before preparation, external-owner lifetime and storage aliases.
No Warp initialization or CUDA execution occurs in these tests.

Focused CPU regression:

```powershell
$env:PYTHONPATH='.;out/vendor/kimodo'
& 'D:/.anaconda/envs/motionbricks/python.exe' -m pytest `
  tests/test_mb_sonic_scene_binding.py tests/test_mb_sonic_geometry_execution.py `
  tests/test_mb_sonic_reference.py tests/test_mb_sonic_runtime.py -q
```

Result: **PASS**, 177 passed, 7 explicitly allocated CUDA tests skipped,
44 TorchScript deprecation warnings in 18.93 s. The worker did not rerun the old
62-point proofs or overwrite prior failures/reports. New adapter GPU/all-map
capture, model-output/RNG parity, cache switching and end-to-end 30fps remain
**NOT_RUN** by this worker; the parent owns those runs.

Frozen source SHA256:

- `mb_reference.py`: `d98e7a552e0272ee5d68b51169bf670dc0eb785e554171cf1b956b0db8bd40d2`.
- `geometry_execution.py`: `53aa9f001088f43ee96b18210b6a8e2b1948544a7a5bc4a2090579303d93d0cd`.
