# Scene Control V3 Geometry, 2026-10-18

## Scope

Independent performance prototype, ZERO optimizer updates and no training. The
actor is an **UNTRAINED capacity probe**, not usable learned WASD. Quality response
is unmeasured. Existing v2, services/pages, formal 38.4M exposures and prior
diagnostics are untouched. No IK, locking, projection, smoothing, root replacement
or invented frames are introduced by this module.

## API

`motionbricks.scene_control_v3.geometry` exports:

```python
WarpScene(meshes, *, signed, known_bounds_m, batch_size=1,
          query_count=512, device="cpu", max_distance_m=10.0,
          object_pose=None, ground_domain=None)
build_pfnn_scene(scene_npz_path, *, batch_size=1, query_count=512,
                 device="cpu", max_distance_m=10.0)
build_true3d_scene(*, batch_size=1, query_count=512,
                   device="cpu", max_distance_m=10.0)
scene.query(points_world_m, object_pose, object_known, object_enabled, query_valid)
encode_scene_features(points_world_m, query_result, root_world_m, heading,
                      body_part_id, query_time_offset_s, object_pose, object_velocity)
```

One owner holds 1..8 fixed local triangle meshes and their BVHs. Mesh data are
copied at construction; the hot path never rebuilds or refits them. `B` and `Q`
are fixed by the owner, `Q <= 512`; object ID is the immutable slot `0..O-1`.
Units are metres, seconds and 30 Hz, Y-up. Caller state starts from clones of
`initial_object_pose`, `initial_object_known`, `initial_object_enabled`.

Query inputs, all on the owner's device:

| Input | Shape / Type | Meaning |
| --- | --- | --- |
| points_world_m | float32[B,Q,3] | Current factual or causal model-predicted body points |
| object_pose | float32[B,O,7] | Object-to-world XYZ translation, XYZW unit quaternion |
| object_known | bool[B,O] | Transform/geometry availability of every declared component |
| object_enabled | bool[B,O] | Known presence; false is known absent, not unknown |
| query_valid | bool[B,Q] | Availability of individual body query points |

Output dict contains only tensors: `distance_m[B,Q]`, `normal_world[B,Q,3]`,
`closest_point_world_m[B,Q,3]`, `distance_gradient_world[B,Q,3]`,
`distance_gradient_valid[B,Q]`, `valid[B,Q]`, `object_id[B,Q]` int32,
`distance_signed[B,Q]` bool, and `scene_known[B]` bool. Float outputs are float32.
Selected-triangle provenance is also returned as `triangle_id[B,Q]` int32 and
`barycentric_uv[B,Q,2]` float32, with invalid IDs -1 and invalid UV zero.
Invalid output slots are ZERO, object ID -1. `scene_known` requires all declared
components known and at least one enabled; false is an explicit scene rejection
state for the caller. Invalid poses, nonfinite points, invalid query slots,
unknown terrain cells, out-of-bounds points and missed search radius invalidate
queries. No known/usable samples must be rejected upstream, never treated as flat.

## Distance And Derivatives

Triangle magnitudes are actual closest-triangle distances from Warp BVH traversal,
subject to float32 arithmetic. Open surfaces are **unsigned**, including below an
open ground mesh. Closed solid inputs must be outward, watertight volumes;
their sign comes from Warp fast winding-number classification (accuracy 2,
threshold 0.5), not an exact winding certificate and not a face-normal sign guess.

Reduction is **minimum component distance**, mixing explicitly flagged signed
volumes and unsigned open surfaces. Every object uses its own full search radius;
the preceding object's result never shrinks another object's traversal. Negative
solid distances override a nearby positive floor. Signed objects additionally
search at least their local bounding-box diagonal so deep interiors cannot be
hidden by a small positive query radius. Positive hits still require distance
strictly below `max_distance_m`. Ties keep the first object. Overlapping solids
use minimum component SDF, not an exact reconstructed CSG union boundary.

`normal_world` is the selected triangle's rigidly rotated oriented face normal.
At edges/corners it is a selected face, not a distance derivative.
`distance_gradient_world = sign * (point - closest) / norm(point - closest)`;
unsigned surfaces use sign +1. The point-distance VJP uses this vector, including
outside corners, and zero for invalid or <=1e-7m distances. Medial-axis/tie
branches provide the selected branch's derivative; a unique derivative need not
exist there. Only first-order point-distance VJP is supplied, not mesh/pose,
closest-point/normal, or second derivatives.

## Actor Features: 16 Channels

The original 15 fields retain their positions; parent P1 review explicitly added
channel 15. There is no 15-channel compatibility path.

| Indices | Field |
| --- | --- |
| 0..2 | Ego query XYZ |
| 3 | Mesh distance, signed only when channel 15 is 1 |
| 4..6 | Ego selected face normal |
| 7 | Opaque caller-bound body-part scalar ID 0..14 |
| 8 | Query time offset, seconds |
| 9..11 | Ego closest-surface XYZ |
| 12..14 | Ego rigid surface velocity |
| 15 | `distance_signed_flag`: 1 signed-volume semantics, 0 unsigned-open semantics |

The signed flag describes semantics, NOT whether the numeric value is negative.
All 16 invalid channels are zero and `scene_valid` is false. No contact-slot names
or native four-foot mapping are inferred from opaque body-part IDs.

Root origin is `root_world_m[B,3]`, heading is native `heading[B]`. World-to-ego is
`x=c*dx-s*dz, y=dy, z=s*dx+c*dz`; +pi/2 maps world +X to ego +Z and world +Z to
ego -X. `object_velocity[B,O,6]` is world linear XYZ m/s then angular XYZ rad/s.
Surface velocity is `linear + angular cross (closest - object origin)`.
Time offsets tag current/predicted body points against the supplied scene snapshot;
they do not magically advance objects. Callers must update rigid object poses
causally. No future actor ground truth is used.

## Assets And Memory

PFNN builder loads CPU `heightfield_y_m[z,x]`, `min_point_world_xyz_m`, `dx_m` from
the actual NPZ. Each source cell becomes two triangles at full resolution.
This is **continuous triangle ground**, unlike old piecewise-constant rectangular
prisms. There is no old SDF exact-parity claim. New training/quality work must use
the declared new representation. There are no prism side walls, bottom caps,
decimation or fallback flat geometry. Unknown cells touching missing vertices are
omitted and retain an on-device known-cell mask; outside map support is unknown.
The builder rejects entirely unknown/missing geometry.

Paths below are relative to the motionbricks workspace, all read-only:

| Scene | Asset under out/pfnn_mixed_20261015/ | Vertices | Triangles | Explicit Geometry Bytes |
| --- | --- | ---: | ---: | ---: |
| PFNN011 | pfnn_map_catalog_v1/011/scene.npz | 1,057,986 | 2,111,848 | 88,722,360 |
| PFNN007 | pfnn_map_catalog_v1/007/scene.npz | 789,876 | 1,575,904 | 66,211,056 |
| Flat | final_acceptance_flat_v1/scene.npz | 103,041 | 204,800 | 8,609,292 |

SHA256, respectively:

```text
e9a0c58793634c2c2500cafebc6d6df45ab367f9c91ee88e8a87adfcb9a36b13
ad9b3e8747554d6b6ff7849c8704d2cf707f140198848f83e2d7189d3425ef1d
5c4b2049a2801f8be4268da42dec730f5094e9b992910acd661d0c21027f1464
```

Independent true3D fixture has three closed boxes: floor 8x0.2x8 centred at
(0,-0.1,0), overhead slab 2.4x0.2x2 centred at (0,1.8,0), side wall 0.2x2.4x3
centred at (1.5,1.2,0). Both top/bottom overhead faces and wall sides are queried.

`storage_summary` lists counts, vertex/index/triangle-bounds bytes and domain-mask
bytes. Explicit geometry bytes EXCLUDE BVH nodes, winding structures, runtime
allocator overhead, tensors and workspaces. `bvh_and_winding_bytes` and
`total_device_bytes_measured` are None until externally measured. Torch's 0.25
allocator cap does not constrain Warp. Benchmark owners must separately record
Torch allocated/reserved and whole-device free-memory deltas across cold scene
construction; their residual is not an exact isolated BVH measurement.

## Execution And Verification

Dependencies: NumPy, trimesh, torch, Warp. Local CPU verification uses Warp 1.18.0,
Torch 2.11.0+cu128. The implementation uses public common mesh-query/torch-interop
APIs intended for the parent's Warp 1.17.0/Torch 2.8+cu128 environment; that
environment is not executed here. SAH is a CPU BVH constructor outside the hot
path. CUDA construction uploads static geometry and also performs Warp's cold
mesh setup work; it requires an allocated GPU session.
[Warp Mesh API](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.Mesh.html)
and [Warp runtime](https://nvidia.github.io/warp/v1.8/modules/runtime.html) describe
the public BVH/query and interop mechanisms used.

Eager GPU is the primary performance path. `query` wraps Torch storage without
copy and launches on Torch's current CUDA stream. There is no per-tick CPU tensor
read, NumPy conversion, synchronization or registry/field-identity lookup. Mesh
storage persists while actual body/egocentric samples refresh every tick.

Scene-local CUDA graph capture is designed to work after mesh construction and
kernel warmup, with fixed buffer addresses/shape, using the same Torch stream.
Owner/BVH lifetime must extend through every replay. Graphs do not switch maps;
create a new scene/graph on map change. Capture remains unverified; no
`torch.compile(fullgraph=True)` promise.

CPU tests use two Torch/OMP/MKL/OpenBLAS threads and `CUDA_VISIBLE_DEVICES=-1`.
Final CPU result: **20 passed, 3 skipped in 6.22s**; all skipped cases are the
explicitly gated CUDA comparisons below.
Coverage includes analytical boxes/plane, corner/edge finite differences and VJP,
rigid transforms, known absence vs unknown, 512-query budget, real PFNN source
counts/SHA, outside/hole semantics, new triangle-vs-old-prism slope semantics,
native heading, refreshed features, and inside-solid -0.5m vs open-floor +0.1m
in both object orders with a 0.2m positive radius.

Parent-only optional correctness tests are named `test_allocated_cuda_cpu_parity`
and skip unless `SCENE_CONTROL_V3_CUDA_ALLOCATED=1` is explicitly set after a GPU
reservation. They compare a box, true3D fixture and four actual PFNN011 points
against CPU; signs, validity, gradients/VJP and finite differences are checked.
Tied edge/corner face normals may differ, so normal equality is checked only on
selected smooth faces. GPU parity tests are not a timing benchmark. No GPU,
server, SSH, optimizer or old diagnostic work was started in the CPU phase.

## Allocated CUDA Diagnosis V2

Parent allocation `geometry_cuda_parity_v1` ran exactly three cases under Torch
memory fraction 0.05 and two CPU threads: box and true3D passed, PFNN011 failed.
That original failure receipt/log are preserved byte-for-byte. The next allocation
`geometry_cuda_diagnosis_v2` used two small GPU processes only: a same-four-point
fused/strict diagnosis and a separate recheck of the failed PFNN test. No actor,
training or full performance benchmark ran; box/true3D were not rerun.

Per-field evidence identifies the v1 failure as `closest_point_world_m`, not a
unit-gradient value: closest-point maximum error 8.620087e-5 m, scalar-distance
error 1.816079e-6 m, unit-gradient/VJP error 0.00905616. With fusion enabled, CPU
selected face 1058140 while GPU selected adjacent face 1058139 at the below-ground
query. Face IDs, u/v, vertex arrays, raw CPU/GPU outputs and float64 selected-face
references are saved in `out/scene_control_v3_20261018/geometry_cuda_diagnosis_v2.json`.

The root cause is FMA-permitted float32 arithmetic changing the adjacent-triangle
selection at a shallow crease. The module now explicitly sets
`enable_backward=False`, `fast_math=False`, and `fuse_fp=False`, matching the
existing BVH module's precision policy. `wp.get_module_options(geometry)` exposes
the actual compile options for performance receipts; the settings do not depend
on a scene identity or map registry.

For these four points, strict mode produces identical CPU/GPU distances, closest
points, gradients, validity, face IDs/u/v and VJP: all recorded maximum differences
are zero. The targeted new CPU trace/VJP test passed (1 passed, 23 deselected,
4.48s); the separate original PFNN CUDA test passed (1 passed, 2.653s call
duration). The original atol=2e-5, rtol=2e-4 gates were not widened. Both CUDA
processes exited and allocations were released. Parent owns the final combined
CPU regression; it was not repeated by this task after the targeted verification.

Float64 reference distinguishes the two near-crease branches: face 1058139 has
distance 0.0095185004444 m; face 1058140 has 0.0095188907966 m, a gap of
0.390352 micrometre. Float32 Y spacing at this 22.54m location is 1.90735
micrometres. Strict mode resolves CPU/GPU consistency by selecting face 1058140
on both backends; the float64 selected-triangle reference prefers face 1058139.
This near-crease ordering resolution is recorded for parent judgment separately
from the passed CPU/GPU parity gate.

Verified source SHA256 (runtime sources unchanged after the successful PFNN
recheck):

```text
motionbricks/scene_control_v3/geometry.py
d301104e8ef29d41ed023f9d3774177672ff87f19f187bf05a1776a7f6d630de
tests/test_scene_control_v3_geometry.py
d2dc7f0cecd9632e3b9af6b63db18d4b4b8d5b655dc521eefd28b8bd37d9e024
```
