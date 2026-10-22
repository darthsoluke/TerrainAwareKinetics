# PFNN Sphere BVH Admission

Date: 2026-10-19. Candidate: `mb_sonic/sphere_bvh_heightfield.py`, 104 lines.
It reuses the existing field/tree constructor and prism-squared function. The
Warp selector changes only the broadphase call from AABB cube to sphere at the
same conservative radius. The Torch query body is unchanged except for the
selector argument. CPU AST tests check both statements. The original helper and
all original geometry proofs remain untouched.

The pinned [Warp 1.18 sphere API](https://nvidia.github.io/warp/v1.18/language_reference/_generated/warp.bvh_query_sphere.html)
tests sphere/AABB overlap, includes tangency, and uses the standard query iterator.
The candidate keeps `fast_math=False`, `fuse_fp=False` and the original tie-break,
distance/normal calculations and native certainty/known-mask gates.

## Runtime API

The existing adapter constructor and `.load` now support the explicit backend:

```python
scene_query_backend="warp_bvh_sphere_exact"
scene_query_proof=Path(per_map_proof)
```

Default deployment remains `native`; the previous `warp_bvh_exact` backend is
unchanged. There is no automatic backend selection or numerical fallback.

Actual admitted proof paths, relative to
`${PROJECT_ROOT}/out/mb_sonic_e0_e3_20261018`:

- `000`: `wasd_sphere_query_v1/000/proof.json`, the retained unchanged plane proof.
- `001,002,003,004,005,006,007,009,010,011,012,013,014,urban`:
  `wasd_sphere_query_v2/<id>/proof.json`.

`000` still installs the original `AnalyticPlaneField`. It does not replace the
plane's surface-normal semantics with a sphere/prism calculation. Rough maps
install `SphereBVHHeightfieldPrismSDF`. Strong field/tree owners and current tree
IDs are included in graph binding checks for both BVH types. C uses the existing
`activate_scene`, `release_scene`, receipt and scoped-byte-accounting APIs.

## Verifier Contract

`verified_sphere_bvh_proof` reads the producer's actual schema, not translated
aliases. Required identity fields are `schema`, `passed`, `scene_id`,
`scene_sha256`, `baseline_implementation_sha256`, `implementation_sha256` and
`plane_unchanged`. The schema is
`motionbricks.mb_sonic.sphere_bvh_admission.v1`.

Distance/normal maximum errors must be zero. `valid_equal`,
`active_prism_ids_equal`, `traced_and_captured_equal` and `boundary_ties_equal`
must be true. Rough-map proofs additionally require:

```json
{
  "active_prism_comparison_scope": "native_valid_queries_only",
  "invalid_internal_id_differences": 28
}
```

The difference count is a nonnegative measured integer, not required to equal
the example. The receipt retains it. Old `000` plane proofs need no internal-ID
scope; their receipt labels it `not_applicable_native_plane` with count `None`.
Point counts must be at least 6,189 for rough maps and 6,144 for the plane.
`native_cube` and `sphere` each require at least 20 samples with finite positive
`p50_ms`, `p99_ms`, `max_ms`. `optimizer_updates` and `physical_steps` must be zero.
Both current source hashes must match the proof pins.

## Actual Parent Evidence

`wasd_sphere_query_v2/complete.json`: **PASS**, 14 rough maps, CPU/CUDA RNG equal,
source locks unchanged; parent allocated wall time 30.5803496 s. Together with the
retained `000` plane proof this covers all 15 catalog maps. Each rough map checks
four actual 1,536-point volumetric-sampler inputs (three lattices), plus 45
surface/boundary points: 6,189 points. Eager, traced and captured outputs match the
explicit original `BVHHeightfieldPrismSDF.query(field, points)` bitwise on the same
field owner. Valid-query prism IDs and boundary ties match exactly.

Hot captured query P50, milliseconds per 1,536 points, from the retained proofs:

| Map | Cube | Sphere |
|---|---:|---:|
| 001 | 30.183 | 4.693 |
| 002 | 18.499 | 1.671 |
| 003 | 36.052 | 11.586 |
| 004 | 31.959 | 2.323 |
| 005 | 37.104 | 8.112 |
| 006 | 32.815 | 1.664 |
| 007 | 15.611 | 1.522 |
| 009 | 17.458 | 3.914 |
| 010 | 29.890 | 5.808 |
| 011 | 2.539 | 4.825 |
| 012 | 13.474 | 5.559 |
| 013 | 1.411 | 0.186 |
| 014 | 3.866 | 0.188 |
| urban | 18.584 | 3.170 |

These are query-only timings, not model/physics/render rates. The measured 011
case is slower with sphere; no per-map automatic choice is introduced.

The first v1 allocation and 001 diagnosis remain preserved. CPU inspection of
`wasd_sphere_id_diagnosis_v1/failed-001/active_ids.npz` confirmed all nine ID
differences were native UNKNOWN observations. For example, point 447 has cube
distance 2.970305 m beyond certainty radius 2.960038 m; both public distance and
normal are masked zero. Native selection stops at its certainty limit, so those
internal IDs have no nearest-prism contract. The corrected proof compares IDs on
native-valid queries and logs unknown differences; all public outputs still have
the same bitwise equality gate. The sphere kernel was not changed for this issue.

## CPU Verification And Freeze

Worker CPU regression of sphere source contracts, proof verifier, scene binding,
geometry execution, reference and runtime: **PASS**, 260 passed, 7 CUDA tests
skipped, 44 reused TorchScript deprecation warnings in 20.63 s. This includes
offline acceptance of every actual parent proof and direct CPU verification of
the saved unknown-ID diagnostic. No GPU query was run by the worker.

Frozen SHA256:

- Sphere kernel: `0b6230d290782be0c9250daca98b0e10c5a60b1a097ae4ac56e43589c5fee886`.
- Original helper: `ccdb3c55616e76999e4a8821d4fe83c0e14888bd24c413ea848746483000ca00`.
- `mb_reference.py`: `f8c4898d000e31370fff2448060cbb79ab4896c9d9575c1cb621ca61ef80a145`.
- `geometry_execution.py`: `37f19d41386b3689119f8484075ffd475de3b53a95d4c3fac7ea46f70e76bc61`.

The parent/C own the following integrated all-map WASD/model/physics run. No
model weights, statistics, FP32 policy, eight-token horizon or four Pose passes
are changed by this geometry installation.
