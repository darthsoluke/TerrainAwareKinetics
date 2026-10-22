# scene_control_v2 data handoff (2026-10-18)

## Longest Existing Train Engineering Candidate

Additional check: selected the maximum declared `frames_30hz` among the same
`native_pilot_v1/accepted.json`'s nine train rows; inspected only this candidate's
existence, NPZ member headers and scalar FPS/spacing. No full-frame scan or
cache rehash. The longest candidate exists: **646 frames, 30 Hz**, original
split **train**, ID `generated_f733c63c38e13f58`, origin `parc_generated`.
This is an existing simulated PARC recording retargeted to SOMA, not a newly
fabricated fixture or raw human capture.

- Raw retarget-world motion (3,976,151 bytes):
  `${PROJECT_ROOT}/out/parc_curated_20261009/reference_soma_v3/three_stair_flights/retarget.npz`
- Bound scene (1,111 bytes):
  `${PROJECT_ROOT}/out/parc_training_20261009/native_pilot_v1/terrain_geometry/generated_f733c63c38e13f58-30c4d74c84_paired_heightfield.npz`
- Existing source-contact sidecar (2,518 bytes):
  `${PROJECT_ROOT}/out/parc_training_20261009/native_pilot_v1/source_contacts/generated_f733c63c38e13f58-30c4d74c84_parc15.npz`
- Upstream recording exists (3,728,198 bytes; not unpickled):
  `${PROJECT_ROOT}/out/parc_curated_20261009/reference_origins_v2/export/recordings/0002_three_stair_flights_dm.pkl`

Motion header fields: `local_rot_mats[646,30,3,3]`, `root_positions[646,3]`,
`target_positions_yup[646,30,3]`, `target_global_rot_mats[646,30,3,3]`,
`target_names[30]`, `target_parents[30]`, `source_contacts[646,15]`, selected
sole vertices/IDs, `source_fps=30`, `uniform_scale=1.0293180673509565`.
World frame is scaled Y-up metres per the bound accepted row, not the legacy
native shard's canonical frame. Scene headers: `heightfield_y_m[76,147]`,
`min_point_world_xyz_m[3]`, `dx_m=0.2058636099100113`, `scene_scale`.
Retarget additionally retains source-oriented `hf[147,76]`, `min_point[2]`,
`dx`. This is an explicit legacy heightfield input; no independent 3-D mesh,
volume or full-body material anchors are supplied.

Parent chain: source/recording family
`generated:local/complete_20261008_original_recovery_r2/results/stairs_s20262014`
-> PARC physics recording `three_stair_flights` -> official conversion
`0002_three_stair_flights_dm.pkl` -> `reference_soma_v3` raw retarget-world
-> `native_pilot_v1` bound scene/contact snapshots. Parent evidence:
`${PROJECT_ROOT}/out/parc_training_20261009/native_pilot_v1/upstream_manifests/generated_f733c63c38e13f58-30c4d74c84_upstream.json`.
It records `whole_motion=true`, `start_fraction=0`, `recorded_state=simulated`,
contacts from simulated force threshold, original candidate SHA
`6578bf5caafeda9d0c170066cc491ea7caf85c087996a982c011f5a4fd8f121b`, and
`intent_status=unavailable_no_training_admission`; its `training_approved=false`
is unchanged.

License evidence: neither accepted row nor its export snapshot supplies a
per-record dataset/derivative license. Existing local
`out/vendor/PARC/LICENSE` contains BSD-3-Clause base plus MIT modifications;
`out/vendor/kimodo/LICENSE` is Apache-2.0. These are code-license evidence;
the record's motion/character/model/asset license chain remains unrecorded.
The candidate removes the local **train motion+scene entity availability**
blocker for a later separately allocated bounded codec engineering run, while
preserving its original train family. No forward/backward run, feature rebuild,
split change or new formal dataset was performed here. Header count is adequate
for 33/49/65-frame extraction; this pass did not recheck frame values/continuity.

Provider handoff sent directly to Raman `01a14b08-6b49-7d30-864f-047526889cd4`
at parent's explicit request: use this train raw-world record and bound HF,
authoritative `target_names`/`target_parents` mapping, 30 Hz derived frame time,
in-memory global366 and existing 8x10 codec labels. Unknown v2 contact state/
actual/anchor validity remains false; no fabricated anchors, cached canonical
370-D/world mixing or second scene scaling. Scope is zero-Adam engineering;
formal quality approval/exposure remains zero. No additional data reads, cache
rebuild or provider execution was performed by the data worker for this handoff.

## PASSAGE Public Entry Check

Checked 2026-10-18 using only original paper, author project and author-hosted
repository metadata. [Paper v1](https://arxiv.org/html/2609.18732v1), submitted
2026-09-16: authors report VR+inertial capture of about 100 h across 1,500
procedural cluttered scenes, with 19,310 screened sequences. Recordings are
retargeted to 29-DoF G1; the stated planner representation is 50 Hz, 65-D, with
four history frames, 25 future frames and a three-layer 3x31x61 elevation map.
The planner/tracker and RTC are relevant to scene-aligned traversal, but this
paper's robot representation is not a supplied SOMA motion sample.

[Author project](https://galaxygeneralrobotics.github.io/PASSAGE/) currently says
`Code (Coming Soon)` and exposes no motion/scene dataset, schema manifest or
paired small-sample download. The public
[GalaxyGeneralRobotics/PASSAGE repository](https://github.com/GalaxyGeneralRobotics/PASSAGE)
is the static project-page source, not the planner/tracker implementation.
Read-only GitHub metadata: tree
`d2f2a1d75e6e8a91993dfd548907e8ece7667754`, pushed
`2026-10-13T11:10:31Z`; recursive tree complete, zero releases. It contains
website source, images/videos/fonts and the paper PDF; no NPY/NPZ/PKL/BVH/
mesh/URDF/checkpoint/schema JSON/YAML payloads were listed.

Licensing: GitHub reports `license=null` and no license file in the inspected
tree. No dataset license/terms or author-published paired entity sample was
found at these entrypoints. The [paper's arXiv license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
is arXiv's perpetual non-exclusive distribution license; no motion-data grant
was provided. Only public metadata was read; no PDF/media/data archive was
downloaded, form submitted or license accepted.

Current acquisition status: **promising source, not presently actionable data**.
Author 100 h collection size is not downloadable availability. Retain it in the
data-source report; do not add it to provider inputs, current code or training.

## Read-Only Real Sample Availability Addendum

Checked only the previously inspected PARC15 record, not the old corpus audit.
The following entities are readable and their SHA256 matches the same native
manifest / accepted row:

- Manifest: `${PROJECT_ROOT}/out/parc_training_20261009/native_pilot_v1/native_manifest.json`
- Record table: `${PROJECT_ROOT}/out/parc_training_20261009/native_pilot_v1/accepted.json`
- Motion: `${PROJECT_ROOT}/out/parc_training_20261009/native_pilot_v1/shards/official_f900cf9e6fd06b59-cccb5abdae_native30.npz`
- Contacts: `${PROJECT_ROOT}/out/parc_training_20261009/native_pilot_v1/source_contacts/official_f900cf9e6fd06b59-cccb5abdae_parc15.npz`
- Scene: `${PROJECT_ROOT}/out/parc_training_20261009/native_pilot_v1/terrain_geometry/official_f900cf9e6fd06b59-cccb5abdae_paired_heightfield.npz`
- Upstream world motion: `${PROJECT_ROOT}/out/parc_training_20261009/local_catalog_v1/retarget/official_f900cf9e6fd06b59/retarget.npz`
  (also readable; matches `accepted.source_shard_sha256`).

Source `parc_official`, ID `official_f900cf9e6fd06b59`; recording/source family
`official:mdm_castle_stairs`, terrain family `dec_release/initial_aug/misc_mdm`,
split **validation** (v2 spelling `val`, not train). Upstream identity:
`mxucg/PARC@e8f119532d2f03955031ce629d08243403443e58:dec_release/initial_aug/misc_mdm/mdm_castle_stairs_dm.pkl`.

Actual arrays: native `dual_features[798,370]`, `global_features[798,366]`,
`local_rot_mats[798,30,3,3]`, `root_positions[798,3]`, FPS=30. Retarget includes
`target_positions_yup[798,30,3]`, world `root_positions`, local/global rotations,
SOMA30 names/parents and selected soles. Source contact indices are exactly
`0..797`, consecutive; source labels equal the retarget's source contacts.

Continuous, finite SOMA windows were actually read in memory, excluding frame0:

| Frames | Python slice | Source frame IDs | Time from recording start |
| --- | --- | --- | --- |
| 33 | `[1:34]` | 1..33 | 1/30..33/30 s |
| 49 | `[1:50]` | 1..49 | 1/30..49/30 s |
| 65 | `[1:66]` | 1..65 | 1/30..65/30 s |

Coordinates require explicit provider work. The **retarget** world is scaled
Y-up metres. The scene is the corresponding legacy heightfield `[38,32]`,
`dx=0.4117272198200226 m`, origin `[-11.913408279418945,0,-3.9575843811035156]`,
scale `1.0293180673509565`; its values match retarget `hf.T[::-1]`.
The **native shard** has first-heading/position canonicalization: frame0 XZ is
zero and initial frame metadata is absent. Native versus retarget world roots
differ by up to **1.4636917 m**; a single diagnostic rigid frame conversion
explains them to `2.8611e-6 m`, while non-root local rotations differ at most
`7.1280e-7`. This conversion was measured only, not installed or used to rewrite
anything. Do not pair old native 370-D coordinates directly with the world
heightfield. For the new provider's real engineering input, start from retarget
world root/local rotations/target joints plus the bound world heightfield, then
explicitly prepare the new window frame in memory. No captured timestamp array
exists here; times above are reconstructed from source frame IDs / 30 Hz.

PARC15 contains only `source_contacts`, names, indices and FPS. Missing v2 labels:
target-SOMA surface-region mapping, intended versus actual state distinction,
canonical contact kind, persistent scene/object identity, object-local material
anchors, local surface normals, per-anchor validity, calibrated confidence and
their annotation provenance. No hand/forearm/knee anchor or load-bearing/sliding
classification is available. The scene has no independent mesh/3-D volume,
object instance table or dynamic object track. Provider contact anchor and
actual/intended supervision masks must remain false until those fields exist.

Use scope: **existing real validation engineering fixture only**. Old manifest
quality/physics/skill approval is false; this check changes no admission or
quality status and does not promote `world_v1` or preidentity data. No cache
rebuild, new feature artifact, training, old audit rerun or download occurred.

## Stable Dictionary Interface

Validity is **all declared components known**, not "at least one object hit".
`SceneObject.valid[t]` / `scene.object_valid[t,o]=False` means unknown transform
or geometry, never known absence. Such a component invalidates every full-scene
query at that time. A component's local `valid=False` (unknown volume nodes or
an unreturned mesh query) also invalidates that full-scene point, even when the
nearby floor is valid. Query output records
`valid_semantics="all_declared_components_known"`; unknown distances/normals/
gradients are zero placeholders with `valid=False`, object ID -1. Torch returns
the identical mask and zero point VJP for those slots. The current schema has
no presence/absence field; a future presence contract must be explicit, not an
interpretation of this validity mask. Assigned-object-only contact queries have
an explicitly narrower query scope and must not claim full-scene completeness.

### Torch Bridge For Plato/Wegener

Integration action: actor-side Torch wrappers must validate
`scene.object_track_provenance[O]`, not the descriptive scene-wide
`transform_role`. A mixed scene can have static objects and an offline dynamic
track; the dynamic teacher track must not become actor input.

`SceneQuery.query_world_torch(world_points_m: torch.Tensor[Q,3],
time_index: torch.Tensor[Q]) -> dict` supplies differentiable `distance_m[Q]`.
The CPU geometry query is detached; its exact local point VJP is supplied through
`distance_gradient_world[Q,3]`, NOT through `normal_world`. This is a first-order
point-gradient bridge, not mesh/transform gradients or a Hessian implementation.
Outputs also expose `closest_point_world_m[Q,3]`, `closest_point_valid[Q]` and
`distance_gradient_valid[Q]`. Mesh gradients use signed `(p-closest)/norm`;
zero-distance nondifferentiable slots are masked. At nearest-surface ties the
selected local branch supplies the VJP. `normal_world` remains a triangle
surface normal. Volume gradients are analytic derivatives of the trilinear SDF
interpolant, not the supplied normal grid; volumes have no exact closest point
and set `closest_point_valid=False`. Legacy heightfields reject Torch physical
objectives. No mesh/transform gradients, Hessians or fullskin gate substitution.

All arrays are NumPy on CPU. Coordinates are right-handed Y-up, metres; time is
seconds. Invalid numeric slots are zero, IDs are -1, and must be masked before
losses. Missing labels never become negative contact supervision. Synthetic
fixtures are explicitly `source="synthetic"` and are not entity samples.

`make_synthetic_fixture()` returns `{schema, source, family, split, fps,
skeleton, recording_id, scene_id, timestamps_s, body_positions_world_m,
body_valid, body_part_names, contacts, scene, provenance}`.

- `schema`: `motionbricks.scene_control_v2.paired.v1`.
- `body_positions_world_m`: float32 `[T,B,3]`; `body_valid`: bool `[T,B]`.
- `contacts`: `state`, `actual_state`, `valid`, `actual_valid`, `anchor_valid`,
  `confidence`: `[T,C]`; `anchor_local_m`, `normal_local`: `[T,C,3]`;
  `object_id`, `kind`: int32 `[T,C]`; `body_part_id`: int32 `[C]`;
  `provenance`: string `[C]`. Kind IDs: support=0, brace=1, grasp=2, touch=3,
  slide=4. `valid` masks intended state; `actual_valid` masks measured state.
- `scene`: `{geometry_kind, object_ids, object_names, object_to_world,
  object_valid, valid_bounds_world_m, object_track_provenance, provenance}`; transforms float32
  `[T,O,4,4]`, validity bool `[T,O]`, bounds float32 `[2,3]`. Geometry assets
  are supplied separately to `SceneQuery`, not serialized as fake paths.
  Each `object_track_provenance[O]` entry is `{role, available_at_s, source}`;
  actor roles are `static`, `factual_history`, `model_prediction`,
  `known_schedule`. Offline tracks use `offline_teacher`. Static tracks must
  actually be constant. An unavailable object transform invalidates that frame's
  scene query instead of silently removing the object. The fixture's global
  `transform_role`/`available_at_s` fields are descriptive only; actor checks
  use the per-object provenance.
- `provenance`: source URI/version, license evidence, parent recording chain,
  actor and original frame interval; split checking groups recording and scene
  families as well as derivative parents.

`sample_scene_queries(scene_query, sample, *, guidance: QueryGuidance,
mode: Literal["actor", "offline"], as_of_s: float, budget=QueryBudget(), seed=...)`
returns `{schema, policy, world_points_m, distance_m, normal_world, valid,
object_id, body_part_id, time_index, time_s, region_id, contact_index,
target_object_id, sample_valid, distance_semantics, distance_gradient_world,
distance_gradient_valid, closest_point_world_m, closest_point_valid}`. Arrays use `[Q,3]` or
`[Q]`; `Q=budget.total`, default 512 = global128 + body256 + contact128.
Region IDs: coarse_global=0, body_time=1, contact=2. Body/contact IDs index their
ordered tables; object IDs are persistent scene IDs, not table indices.
`object_id` is the selected solid (minimum component signed distance);
`target_object_id` is contact intent.
`sample_valid` masks absent guidance separately from geometry `valid`.
Distance is positive outside, negative inside. Minimum component SDF is not
the exact union boundary distance inside overlapping solids.

`QueryGuidance(role, available_at_s, body_positions_world_m, body_valid,
contacts)` requires role `factual_history`, `model_prediction`, or
`offline_teacher`. Actor calls reject `offline_teacher`; factual history cannot
have valid body/contact slots after `available_at_s`. Model prediction is an
explicit caller-owned prediction, not read from paired future labels. Geometry
object transforms obey the same causal rule through per-object provenance:
factual tracks cannot guide future queries, predictions/schedules must already
be available at `as_of_s`, and static tracks must be constant. Offline synthetic
oracle calls use `offline_teacher` explicitly. The guidance availability time
must also be no later than actor `as_of_s`.

Integration worker Plato: files below are worker-owned, no package/config edits.
These names/keys are the integration target. No internal multi-agent messaging
tool is exposed to this worker; this report is the coordination surface.

Public metadata entrypoint: `inspect_public_metadata(dataset, root)` returns
availability/missing fields and shape checks, never downloads entity archives or
accepts license terms. PARC15 is a source-label sidecar inspection only.

## Implementation And Checks

Owned files:

- `motionbricks/scene_control_v2/data.py`
- `motionbricks/scene_control_v2/geometry.py`
- `motionbricks/scene_control_v2/sampling.py`
- `scripts/inspect_scene_control_v2_data.py`
- `tests/test_scene_control_v2_data.py`
- `tests/test_scene_control_v2_geometry.py`
- This handoff.

Implemented mesh CPU queries using existing Warp 1.18.0 BVH and winding-number
sign classification, with trimesh input checks; no new dependency installation.
Magnitude is closest-triangle distance; Warp fast winding sign is not a
certified exact solid test. Closed outward meshes are required for signed mode;
open meshes require explicit `signed=False`. `VolumeField` uses scipy trilinear
interpolation of caller-supplied XYZ distance/normal/valid grids.
`LegacyHeightfieldField` must be selected explicitly and reports vertical
clearance semantics, never a 3-D SDF.

Quotas: global/body/contact=128/256/128. Body and contact channels are
round-robin balanced (counts differ by at most one), each cycling its available
time slots. Missing guidance keeps reserved invalid slots, with no quota
reallocation or borrowed ground-truth coordinates.

Actual checks with `D:/.anaconda/envs/motionbricks/python.exe`, project cwd:

- `-m pytest tests/test_scene_control_v2_data.py tests/test_scene_control_v2_geometry.py -q`:
  38 passed (4.18 s). Covers low ceiling, overhang/through-hole air versus solid,
  invalid bounds/regions/object tracks, rigid translation/rotation, volume
  unknown corners, signed mesh corner and edge gradient finite differences,
  transformed Torch VJP, trilinear derivative, actor future-label rejection,
  unknown dynamic object despite a known floor, unknown component geometry in
  either object iteration order, conservative full-scene validity and zero VJP,
  fixed budgets and balanced channels, family/recording/parent/scene lineage
  leakage and public metadata absence.
- `-m scripts.inspect_scene_control_v2_data synthetic`: 512 queries, 512 valid,
  regions `[128,256,128]`, explicit offline oracle, zero real entities/optimizer
  updates/GPU launches. Warp initialization lists the local CUDA device but all
  geometry buffers/kernels are explicitly `device="cpu"`.
- Existing `out/parc_training_20261009/native_pilot_v1/source_contacts/official_f900cf9e6fd06b59-cccb5abdae_parc15.npz`:
  798 frames, 30 Hz, 15 named channels, finite range `[0,1]`;
  `certified_contact=False`, `anchor_valid=False`. Read-only, no cache rebuild.

## Public Data Status

`inspect_public_metadata(dataset, root)` reads NPY headers and binding presence;
never unpickles `scene_name.pkl`. `adapt_public_window(...)` accepts a small
already converted window with explicit FPS/skeleton/scene/provenance/contacts,
checks published required motion-array shapes and retains `native_fields`.
No retarget or scene pairing is inferred.

Author field references checked 2026-10-18:

- [LINGO](https://github.com/mileret/lingo-release): body pose/orient/transl,
  Y-up selected joints, segment bounds, scene binding, language and separate
  hand-event arrays. Hand events stay metadata, not full-body anchors.
- [TRUMANS](https://github.com/hygenie1228/TRUMANS): pose/orient/transl,
  segment/frame/scene flags, object flags and 5 object matrices, bad-frame list;
  author documentation's dynamic-object fields are retained without inventing
  a source-to-instance flag remap.
- [CIRCLE](https://github.com/Stanford-TML/circle_dataset): bounded file-presence
  checks for BVH/SMPL-X NPZ/GLB/URDF; native Y-up/-Z-forward frame. No assumed
  unpublished SMPL-X archive key layout.
- [Warp query contract](https://nvidia.github.io/warp/latest/language_reference/_generated/warp.mesh_query_point_sign_winding_number.html):
  closest triangle plus winding sign; not heightfield geometry.

No author entity samples were provided for these three datasets. Workspace
`out` filename inspection found no published motion-array stems or matching
dataset names; `out/vendor` contains GMR/kimodo/PARC/PARC-v0.1. Public missing-data
paths are covered by tests, not populated with synthetic entities. No archives
were downloaded and no license agreement/form was accepted.

## Remaining Integration

- Plato owns canonical `ContactKind` and Torch batch conversion. Serialized IDs
  here match support0/brace1/grasp2/touch3/slide4; no duplicate enum is introduced.
- Loss worker can call `SceneQuery.query_world_torch` directly on flattened
  predicted points, and must combine `valid` with its body/contact validity.
  CPU geometry bridge is a correctness path, with synchronous host transfer for
  CUDA callers; no runtime speed claim is made.
- Authorized LINGO/TRUMANS/CIRCLE entity samples, full target-skeleton conversion
  and reliable contact anchors remain data acquisition/acceptance work.
- No native 370-D SOMA motion/features are fabricated from the synthetic
  13-region fixture. It demonstrates schema/geometry only.
- Old caches, retargeting, formal training, services 28943/28944 and quality gates
  were untouched. No SSH, optimizer run or GPU kernel launch was performed.
