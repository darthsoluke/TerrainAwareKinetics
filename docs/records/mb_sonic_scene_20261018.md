# MB + SONIC E2 Scene Contract

## Runtime Interface

Only `motionbricks.mb_sonic.scene` is needed for scene loading. Runtime dependency:
NumPy. MuJoCo 3.8 is used by the separate verification/probe CLI.

```python
from motionbricks.mb_sonic.scene import load_scene
scene = load_scene(manifest_path)
scene.manifest                 # SceneManifest, motionbricks.mb_sonic.scene.v1
scene.scene_include_path       # asset + worldbody MJCF include
scene.query_height(xy_m)       # ground top height, face normal, known, normal_unique, object_id
scene.render_meshes()          # SceneMesh tuple: stable object ID, geom name, vertices, faces
scene.box_signed_distance(xyz_m, object_id)
```

All query/render coordinates are MuJoCo Z-up metres. Shared transform:
`soma_to_mujoco(X,Y,Z) = (X,-Z,Y)`, a proper +90-degree X rotation;
`mujoco_to_soma` is its inverse. No scale, spawn rebase or actor-dependent offsets.
The manifest stores the matrix, quaternion and both frame names explicitly.

Ground ID is `0`, geom `mb_ground`. Optional rigid boxes have positive stable
IDs and names `mb_low_ceiling`, `mb_sidewall`, `mb_platform`. They are separate
closed objects and never collapsed into ground height. `query_height` is explicitly
ground-only, not a whole-scene signed-distance API. Unknown returns `known=False`,
`object_id=-1`, NaN height/normal. Nodes and triangulation edges have non-unique
normals, indicated by `normal_unique=False`.

## Export Contract

`export_catalog(catalog_path, output_dir, friction=(slide,torsion,roll),
friction_source="...")` exports all 15 existing map entries and a separate
`rigid_boxes` scene based on map `000`. Original scenes remain obstacle-free.
Every output contains the original `scene.npz` bytes, SHA256 and source metadata,
a full-resolution float32 binary heightfield, an MJCF include and the manifest.
Output directories must be new. Old source/corpus/cache/audits are not modified.

Height arrays use source `height[z,x]`; source Y heights are absolute, not added
to origin Y. Native XY extent is `(ncol-1)*dx, (nrow-1)*dx`, not catalog `n*dx`.
Binary rows are reversed into increasing MuJoCo Y. Binary input holds original
metric float32 values, with no image conversion or downsampling. MuJoCo normalizes
to [0,1]; geom Z is original minimum height and elevation scale is max-min.
A flat input uses positive scale 1 with all elevations zero. Base depth is 0.1 m,
explicit in the manifest and shared closed render geometry. Outside the closed
last-node rectangle is unknown. Missing nodes/masks are rejected, never filled.

The native cell triangles are `(00,10,11),(00,11,01)`. Render geometry uses the
same triangles, boundary walls and base. Physics uses native hfield collision,
not the convex hull of a concave mesh. Boxes use native box primitives.

Friction is caller-supplied, fixed scene-wide and includes a required provenance
string. Sliding is dimensionless; torsional/rolling coefficients are metres.
These are declared simulation parameters, not measured PFNN materials. All scene
geoms use priority 1, condim 3; no skill-specific changes. Includes do not alter
solver, timestep, gravity, robot defaults or actuators. Heightfield paths are
absolute to avoid an inherited G1 `meshdir`; re-export after relocating a bundle.

Compose with the official robot model include, not its flat-floor scene wrapper:
otherwise the wrapper's infinite plane would add geometry outside this manifest.
The integration owner chooses the official robot asset and 0.005 s baseline.

## Bounded CPU Checks

`scripts/export_mb_sonic_scenes.py` supplies explicit `export`, `verify`, and
`probe` subcommands. No default subcommand, GPU import, server or native viewer.
Probes use only an initial XML body pose; no qpos/qvel writes, reset, external
forces, optimizer or control policy. Verification and execution evidence follows
below. The parent owns the one actual all-map export and all-16 verification run;
this implementation did not duplicate that export.

CLI (working directory `motionbricks`, `PYTHONPATH=.`):

```powershell
python scripts/export_mb_sonic_scenes.py export --output out/mb_sonic_scenes_20261018 --friction 1 0.005 0.0001 --friction-source "Declared E2 fixed simulation coefficients; not measured"
python scripts/export_mb_sonic_scenes.py verify out/mb_sonic_scenes_20261018/urban/scene_manifest.json
python scripts/export_mb_sonic_scenes.py probe out/mb_sonic_scenes_20261018/rigid_boxes/scene_manifest.json --kind foot --position -2 0 0.425 --steps 20
```

Probe shapes are explicit diagnostic bodies: sphere radius 0.04 m, foot box half
sizes (0.11,0.05,0.025) m, palm ellipsoid radii (0.045,0.025,0.065) m; mass 1 kg.
They are not substituted G1 collision shapes. The report contains actual native
contact object IDs, positions, geom1-to-geom2 normals, separation/penetration,
contact-frame wrench and timestamps. It records the last observed contacts,
maximum simultaneous contacts, and all execution flags. Default 0 steps calls
`mj_forward`; positive steps are a free-body physics rollout, bounded at 2000.

### Preserved Native Boundary Failure

Initial parent all-map verification stopped on map `001`, not flat map `000`:
MuJoCo 3.8 `mj_rayHfield` vertical rays exactly on the finite hfield boundary
returned the base at 9/128 sampled points. The worst raw error was
5.427687487294597 m at XY (7.96545582673242, 5.944881916046143), canonical
height 1.8860934588216967 m, native distance 8.315850496292114 m from Z
4.774256467819214 m. This is a grazing-ray numerical issue, not a terrain offset.

The verifier keeps exact vertical grazing mismatches in every report. It also
compares *every* compiled height node against source data, including all boundary
nodes, with the original float32 normalization error bound. Boundary rays use
explicit inward incidence (at most 0.01 grid spacing horizontally before hitting)
and compare height/normal against the canonical query at the actual ray hit XY.
No height tolerance is widened and no source geometry is changed. Independent
physical sphere contacts at the failed edge are checked separately; vertical
grazing-ray parity must not be claimed from the passing non-tangent checks.

Independent real-map edge probe at the worst point, sphere center
(7.96545582673242, 5.944881916046143, 1.9210934588216967) m, generated 3 actual
ground contacts at Z 1.88566-1.88686 m. Native outward normal Z components were
0.92701-0.96193; minimum native separation was -0.00633272 m. Contact normal
forces were 4.83823-16.49436 N. The surface collision is present at that boundary;
it does not share the vertical grazing-ray/base-hit failure.

## Execution Evidence

Local Python 3.10.20 / NumPy 2.2.6 / MuJoCo 3.8.0:
`python -m pytest tests/test_mb_sonic_scene.py -q` completed with **24 passed in
1.06 s**. Coverage includes original bytes/hash/float32 row order, asymmetric
non-square grids, positive and negative flat elevations, absolute source Y
semantics, both native triangle interiors and normals, exact known boundaries,
unknown queries, no missing-node fill, immutable asset hash checking, closed
render mesh topology, independent box SDFs, six native contact configurations,
free fall without state writes, four physical edge contacts, all 15 actual source
hashes and metric grids, and official 43-DOF G1 compilation without an extra plane.

Actual map `001`, 32 samples/group, strict verifier result:

- All 152,208 native nodes: max reconstruction error 6.9051515084e-7 m.
- Sampled node/interior/inward-boundary rays: max height error 6.1687444708e-7 m.
- Max unique face-normal vector error: 2.4297377032e-5.
- Original height tolerance: 2.6163520204e-6 m; unchanged.
- 9 exact vertical grazing-edge mismatches retained in the JSON result.
- Four outside-domain rays: no native hit, canonical unknown.

The parent's retained `out/mb_sonic_e0_e3_20261018/scene_verification_failure_v1.json`
initially labels the 5.4276874873 m failure as map `000`; minimal reproduction
identifies map `001`. Map `000` passes all 228 sampled rays with zero height error.
The original failure receipt was not overwritten by this implementation.

No GPU, training/optimizer updates, SSH, model/dataset transfer, service, native viewer,
old corpus/cache/audit rebuild, or production fallback was used. Remaining
integration work belongs to the parent: persist the all-16 check/probe reports,
bind the runtime and renderer to these manifests, and evaluate E1/E3 control and
latency. Native exact-boundary vertical grazing rays remain an explicit numerical
limitation, not an accepted parity claim.

MuJoCo references:
[heightfield format and normalization](https://mujoco.readthedocs.io/en/3.8.0/XMLreference.html#asset-hfield),
[ray geometry and normals](https://mujoco.readthedocs.io/en/3.8.0/APIreference/APIfunctions.html#mj-rayhfield).
