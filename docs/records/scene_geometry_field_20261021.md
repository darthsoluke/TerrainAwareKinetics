# Bounded Static 3D Geometry

Independent implementation in `motionbricks/data/scene_geometry_field.py`.
No server, viewer, protocol, training or existing field implementation is changed.

## Scene Contract

`SceneManifest` describes a complete list of closed oriented solid boxes, with
known empty space between them inside an explicit world XYZ AABB. Units are
metres, Y is up, and each rotation maps local column vectors into world space.
Ground is an authored finite solid, with an underside and edges. No implicit
plane, downward fill, open mesh or guessed geometry is inserted.

Every `SceneBox` has a unique positive `box_id`, name, semantic role, centre,
positive half extents and proper rotation. Nested inputs are frozen into tuples
so later edits to JSON source lists cannot change the scene. `dataclasses.asdict(manifest)` is
JSON serializable. Reconstruct its `boxes` with `SceneBox(**box)` before
`SceneManifest(**data)`.

Three explicit example manifests are provided:

| Name | Authored geometry |
| --- | --- |
| `low_ceiling_uneven` | Finite ground, 0.12-radian tilted ground block, low ceiling with underside at Y=1.43m |
| `side_wall_step` | Ground, wall rotated 0.25 radians about Y, step with top at Y=0.4m |
| `bridge` | Ground, deck spanning Y=1.85..2.15m, two side supports; lower passage along Z and upper path along X |

Examples share known bounds `(-6,-3,-6)..(6,7,6)` and finite ground
`(-5,-1,-5)..(5,0,5)`. The bridge paths demonstrate distinct free levels;
access ramps and locomotion behaviour are outside this geometry fixture.

## Query Semantics

`SceneGeometryField.query(points)` accepts floating Torch `[...,3]` world points
and returns `(distance, normal, valid, surface_id)`, retaining batch axes,
device and floating dtype. IDs are int64; validity is boolean.

Each component is an exact OBB signed distance, positive outside and negative
inside. The composed distance is the minimum component distance. It is exact
outside the union; overlapping interiors generally do not have the exact distance
to the union boundary. Normals are outward unit gradients of the winning
component away from nondifferentiable ties, including diagonal edge/corner
normals outside boxes. An interior normal can point at a buried component face.

Face ID = `6 * box_id + face_index`. Face order is `-X,+X,-Y,+Y,-Z,+Z`.
For interior points, choose the nearest face; for exterior edges/corners,
choose the dominant incident face. Exact ties choose the first face in that
order. Equal component distances choose the lowest box ID, independent of
manifest order. Thus a bridge deck's upper and lower surfaces have different IDs.

Validity requires both a point inside the known bounds and its distance-radius
ball inside those bounds. This conservatively excludes queries whose closest
geometry could be in unknown space. Invalid samples return zero distance/normal
and ID `-1`, always accompanied by `valid=False`.

```python
from motionbricks.data.scene_geometry_field import (
    SceneGeometryField, example_scene_manifest,
)

scene = example_scene_manifest('bridge')
field = SceneGeometryField(scene)
distance, normal, valid, surface_id = field.query(points)
observation = sampler.observe_points(history_root, points, field.observation_field())
# Alternatively: sampler.observe_points(history_root, points, None,
#                                      samples=field.query(points)[:3])
meshes = scene.meshes()  # CPU by default; explicit dtype/device accepted.
```

Each mesh carries world vertices, outward-wound triangles and per-triangle
surface IDs from the same manifest. These are component meshes, retaining
internal faces where solids overlap, rather than a boolean-union boundary mesh.

## CPU Verification

`python -m pytest tests/test_scene_geometry_field.py -q`

Result: 34 passed on CPU, with CUDA devices hidden for the test process.

Checks cover both bridge levels, solid deck, sloped terrain and ceiling, wall
rotation and step, all six rotated/transformed faces, overlap and tie semantics,
normal gradients, batched/scalar/empty inputs, unknown bounds, absence of implicit
ground, mesh/query identity, JSON round trip and existing sampler compatibility.
