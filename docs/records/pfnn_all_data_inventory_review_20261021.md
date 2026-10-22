# PFNN All-Data Inventory and Environment-Perception Review

Date: 2026-10-21, Asia/Singapore. Scope: local, read-only corpus/code audit,
followed by this document only. No SSH, GPU work, training, browser actions,
service changes, data conversion, or data rebuilding was performed.

## Decisions for the training plan

1. Use the complete existing PFNN **training split**: 12,058 admitted
   motion/scene windows, or 48,232 existing quarter-rotation examples.
   Keep the 3,533 validation and 604 test windows in their existing splits.
2. Preserve our environment perception: causal body/command/actual-motion
   volumetric observations before Root, then resample along **predicted Root**
   before Pose/decoder. Query the original paired scene at each generated state.
3. Replace the pilot's bounded subset sampler for this experiment. Its 48 PFNN
   training windows are not an epoch over the complete training corpus.
4. Treat complete source conversion, admitted-window coverage, and continuous
   time coverage as three different quantities. All 80 motion files were
   processed; all 16,195 admitted pair-windows were cached. The available local
   reports do not establish that every source file has an admitted window, nor
   the fraction of each clip's timeline covered by admitted windows.
5. Reuse existing source motions, static scenes, contact sidecars and caches.
   For longer supervision, index continuous admitted spans within one original
   motion/scene/contact record. Do not join adjacent episode scenes merely
   because they reference the same BVH.

## Evidence and identity

All relative paths below resolve against
`${PROJECT_ROOT}/`.

| Local evidence read | What it establishes |
| --- | --- |
| `out/pfnn_mixed_20261015/source_inventory_v3/inventory.json` | Original ZIP identity audit, BVH files, recording groups, mirror flags, source rates and annotations |
| `out/pfnn_mixed_20261015/source_inventory_v3/source.receipt.json` | Existing packaged source receipt; not recomputed in this audit |
| `out/pfnn_mixed_20261015/preparation_complete_review_v1-report.json` | Completed world-v2 preparation, split/kind counts and stored gate review |
| `out/pfnn_mixed_20261015/acceleration_reports_v1/pfnn_cache_complete.json` | Full PFNN primitive cache size, shapes and source-manifest binding |
| `out/pfnn_mixed_20261015/acceleration_reports_v1/corpus_audit_v3-report.json` | Prior complete cache array hashes, split/family/command coverage |
| `out/pfnn_mixed_20261015/preparation_snapshot_audit_v1-report.json` | Earlier partial clip/contact audit; explicitly not final coverage |
| `out/pfnn_mixed_20261015/reactive_prefix_training_20261021_v1/reactive_prefix_pilot_20261021_v2/source_history/selection.json` | Actual pilot sample identities, not assumed from quotas |
| Same directory, `source_receipts.json` and `contract.json` | Source/cache recovery checks and the executed pilot's conditioning contract |
| `out/pfnn_mixed_20261015/pfnn_map_catalog_v1/catalog.json` | Fifteen packaged interactive maps; separate from motion/scene training pairs |

Small-file SHA256 values recomputed locally during this audit:

| File | SHA256 |
| --- | --- |
| Source inventory | `52d62dc277de8f5dc563f68f6f871ce071ec81d321faab53d82ea79fbcc7db1f` |
| Preparation complete review | `59a3d22912753455b0cd1f1d02d7cecebf19b137b58e8e8ec9f2f641e63d4f1e` |
| PFNN cache complete receipt | `43e5313467be1fa386a43878f13a13764c441c4e67a886d198d45aee518a4053` |
| Complete corpus audit report | `e068529b219c9a8925bfe58a3a84cab504c55e2ebd2f71492aff5423ae156d31` |

The preparation and cache receipts bind `dataset_world_v2/manifest.json` to
`f8576d15230d3e493df9902d6a5fe23045263d90d5d338e23f0217cf0058d8bb`.
The corpus audit binds PFNN `data/records.jsonl` to
`10e7f7c1b1bba275e93b94c09ddae6efb6122c0e76a498e65993edb4929b66e8`.

Recorded remote artifact locations, copied from the evidence rather than
invented or accessed:

- Dataset: `${PRIVATE_STORAGE}/motionbricks-20261010/pfnn_mixed_20261015/dataset_world_v2`.
- Cache: `${PRIVATE_STORAGE}/motionbricks-20261010/pfnn_mixed_20261015/cache_pfnn_world_v2`.
- Correct world/contact schema: `motionbricks.pfnn_static_pairs.v2`,
  `pfnn-world-and-scene-contacts-v2`.
- The retired `dataset_full_v1` is excluded.

These are stored artifact identities, not a fresh remote availability check.
The complete dataset manifest, cache record index and full converted corpus
were not found in the local PFNN report tree inspected here. No remote files
were read. The prior array audit is evidence of its completed check, not a new
rehash of those arrays in this turn.

## Sources, mirrors and real frames

The inventory's source directory exists locally:
`F:/dev/GASP_57/ai4animationpy/Demos/Dataset/PFNN`.
Its `data/animations` currently contains 81 BVHs. The private packaged source
also exists at `out/pfnn_mixed_20261015/source_inventory_v3/pfnn_source.zip`,
440,618,298 bytes. Large source files were not rehashed in this audit.

The inventory was originally checked against
`https://theorangeduck.com/media/uploads/other_stuff/pfnn.zip`:
81 official BVHs verified, no missing official or unverified extra BVHs.

| Quantity | Verified interpretation |
| --- | --- |
| 81 BVHs | 80 motion files plus the one-frame `rest.bvh` calibration pose |
| 40 original motion files | 34 recording families; file segments within a family remain related |
| 40 mirrored motion files | Derived from the original motion, not independent capture |
| 931,981 inventory frames | Includes both mirror sets and the one rest frame |
| 931,980 motion frames | 465,990 original plus 465,990 mirrored source frames |
| 233,010 real 30Hz converted frames | 116,505 original plus 116,505 mirrored timestamps |
| Approximately 1.079 hours | Original motion only, before counting the mirrored copy |

Source frame rates are read per file, not assumed to be exactly 120Hz.
`pfnn_soma_retarget.py::sample_source_motion` uses original timestamps,
joint-rotation SLERP and Root interpolation. Applying its frame-count formula
`floor((source_frames-1)/source_fps*30 + 1e-8) + 1` to the inventory reproduces
233,010. This is resampling the recorded time span; it does not create new
capture durations. The historical `original_hours` inventory field also
includes `rest.bvh`; motion-only duration is 3,883.09467 seconds per mirror set.

## Fixed splits and cache coverage

`scripts/build_pfnn_terrain_dataset.py::split_group` assigns an entire recording
family using SHA256 of `pfnn:2333:<group>`, first eight hex digits modulo ten:
0=test, 1=validation, otherwise=train. Recomputing this over the local inventory
matches the complete preparation report's family/file counts.

| Split | Families | Original + mirrored files | Converted frames, derived from inventory | All scene-pair records | Admitted scene-pair records | Admitted pair-windows | Unique source-time windows | Augmented cache rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 28 | 31 + 31 | 180,876 | 5,428 | 3,543 | 12,058 | 10,097 | 48,232 |
| Validation | 4 | 6 + 6 | 38,312 | 1,272 | 1,003 | 3,533 | 2,872 | 14,132 |
| Test | 2 | 3 + 3 | 13,822 | 425 | 212 | 604 | 544 | 2,416 |
| Total | 34 | 40 + 40 | 233,010 | 7,125 | 4,758 | 16,195 | 13,513 | 64,780 |

Converted frame counts in this table describe full source files in each split,
not admitted temporal coverage. Unique source-time window counts remove repeated
scene pairing of a clip/start; mirrors and overlapping windows remain related.
The admitted scene-pair counts equal the full cache audit's `source_ids`, whose
IDs include `:terrain:<episode>`. They are not BVH counts.

Validation families: `LocomotionFlat02`, `LocomotionFlat05`, `LocomotionFlat08`,
`WalkingUpSteps05`.

Test families: `WalkingUpSteps01`, `WalkingUpSteps04`.

All other inventory motion families belong to train. No split is moved to
increase training coverage. Both preparation and full cache reports record
absence of cross-split recording-family leakage.

| Constructed terrain kind | Source files considered, including mirrors | Admitted pair-windows | Rejected windows |
| --- | ---: | ---: | ---: |
| Flat | 38 | 9,955 | 8,442 |
| Jumpy | 10 | 2,182 | 3,092 |
| Rocky | 30 | 3,859 | 8,573 |
| Beam | 2 | 199 | 666 |
| Total | 80 | 16,195 | 20,773 |

There are 2,367 scene-pair records without an admitted window. The cache builder
skips those records. It then caches every declared admitted start, rejecting
zero additional PFNN windows; `16,195 * 4 = 64,780` exactly. Four rotations move
both world motion and scene together; non-plane observations are re-queried.

The local summary cannot prove per-BVH positive admission or compute unique
frame/interval coverage. To establish those later, read the existing full
manifest and records index, compare `(source_id,start,quarter)` sets and aggregate
by BVH before `:terrain:`. This is an index audit, not a new dataset build.

## Why cached windows do not mean all times

Read implementation: `scripts/build_pfnn_terrain_dataset.py` and
`motionbricks/data/pfnn_terrain_pairing.py`.

- Footstep annotations define episodes. Starred events, incomplete temporal
  margins and invalid event intervals are excluded by `footstep_cycles`.
- Each episode gets its own static patch/RBF-fitted scene, using retargeted SOMA
  support geometry. These are author-style constructed environments, not
  measured capture-scene geometry.
- Candidate starts use stride eight plus the terminal valid start, around the
  episode core. They are not every possible source start.
- Each candidate spans 33 actual frames and must pass retarget quality and
  scene admission. Rejected windows stay recorded.
- Scene-specific contacts are stored with `source_frame_offset`,
  `source_frame_indices`, `fps=30`, and a finite interval length. Original phase
  is retained for audit, not supplied as an actor control.

The scene-contact sidecars use the existing terrain-relative SOMA foot-joint
detector: world speed below 0.15m/s and terrain-relative joint height below
0.10m. They are derived kinematic labels, not measured support forces. Original
PFNN velocity-derived stance is also retained and is used during fitting and
admission; those two label sources serve different roles.

The completed preparation review checks the manifest and recorded gates; it
explicitly records `independent_geometry_recomputed=false`. Its all-80 count
does not replace per-time coverage. The earlier 37-clip snapshot is partial and
must not be reported as final data coverage.

## Continuous supervision already represented by the sources

The builder writes one complete `motions/<stem>.npz` per converted source,
containing `global_features`, source-kinematic stance and 30Hz timing. Each
scene record references that motion plus one geometry and one contact sidecar.
The records retain original accepted/rejected starts and episode intervals.

`ReactiveSourceBank._pfnn` reads the full motion, verifies those three artifacts,
checks the contact clock, and applies the scene contacts to the matching source
interval. However, `ReactiveSourceBank.load` currently returns only
`world[:, start:start+33]`, and the prefix loss requires `[1,33,366]`.
Longer supervision requires a loader/loss change, not retargeting the sources.

For a proposed longer segment `[s,s+L)`, require all of the following:

1. The entire segment exists in one genuine motion shard and the same split.
2. It remains inside that record's contact interval `[offset,offset+count)`.
3. The same original static geometry applies throughout.
4. Its frames and swept intervals are covered by the union of that record's
   already-admitted 33-frame windows, with no gap or rejected-only extension.
5. Report maximal continuous spans and available segments by requested length,
   source family and behavior before assigning a long-rollout budget.

Overlapping accepted windows within one scene can provide a longer covered
span without rescreening existing geometry. Joining different scene records
does not establish a valid continuous scene. No local complete interval index
was available to count 2s/4s/8s/16s spans in this audit. In particular, long idle
targets and sustained braking targets cannot be assumed from the full-clip
duration or from a low terminal speed in a one-second window.

## Behavior counts relevant to stopping and independent facing

The complete cache audit defines idle by terminal speed `<0.05m/s`, start by
history speed `<0.1` and terminal speed `>0.3`, stop by history speed `>0.3`
and terminal speed `<0.1`. Side/back uses terminal velocity versus facing.
Counts below divide its four-view counts by four; scene duplicates remain.

| Split | Idle endpoint | Start | Stop | Side/back terminal |
| --- | ---: | ---: | ---: | ---: |
| Train | 436 | 202 | 130 | 1,564 |
| Validation | 6 | 16 | 8 | 768 |
| Test | 3 | 11 | 11 | 44 |

Train terminal speed min/P5/median/P95/max is approximately
`0.00138 / 0.08285 / 0.84168 / 2.04716 / 3.57654 m/s`.
Training at 4-6m/s therefore needs a separate speed-generalization decision;
the present PFNN train cache does not contain matching terminal speeds.

These are source-motion classifications, not recorded joystick intentions.
The pilot uses another classifier based on averaged history/next-four/last-four
speeds and a 0.12m/s threshold. Its categories are not interchangeable with the
counts above. Its actual selected PFNN train set contains 48 q0 windows from
28 families and 35 distinct BVH stems (mirrors included); validation has
12 windows from four families and seven stems, with zero selected idle windows.
The train categories are 7 idle, 6 start, 8 stop, 9 strafe, 9 reverse, 9 forward.

## Environment perception that the next experiment must retain

Read source modules:

- `motionbricks/data/unified_scene_field.py`: `field_from_heightfield`,
  `VolumetricEgoSampler` and the 15-feature observation contract.
- `motionbricks/data/joint_sdf_observation.py`: `HeightfieldPrismSDF`.
- `motionbricks/data/local_terrain_observation.py`: two 9x9 metric patches,
  explicit validity, trained query controls, predicted-path sampling support.
- `motionbricks/data/unified_terrain.py`: shared scene encoder and scene
  attention into Root, Pose and codec decoder.
- `motionbricks/data/root_path_terrain.py`: `PredictedRootObservation.sample`.
- `motionbricks/data/reactive_prefix_training.py`: `TrainingRootPathObservation`.
- `motionbricks/data/reactive_path_execution.py`: deployed coarse/Root/path/Pose
  execution and observation outputs.

The current algorithm has two observation stages:

1. From four factual or generated-and-executed history frames, obtain current
   body position/facing and actual velocity. Sample three 8x8x8 lattices:
   body, command corridor and actual-motion corridor. Each point carries
   relative world XYZ, local XYZ, signed distance, normal, heading and region
   identity: 15 features with explicit validity. The shared encoder forms
   24 scene tokens, which condition Root's stages.
2. Use the newly predicted Root trajectory to bend the command-corridor region
   along the playable path. Query that geometry and the height patches again,
   re-encode the scene, then condition Pose and decoder. Body and actual-motion
   regions remain associated with the same current history.

This preserves geometric anticipation along the model's planned motion rather
than observing only a straight joystick ray. Training's dedicated path observer
already calls the same `sample` implementation as inference. For new generated
history rollouts, static source caches cannot substitute for current queries:
their coordinates depend on the original source history and command.

The existing cache stores raw coarse observations, not learned embeddings or
a ready-made predicted-path observation for every future rollout state. Keep
the cache for source identity, supervised targets and unchanged source-input
checks; run our scene sampler on the current history and predicted path for
closed-loop training. The earlier full-training teacher path and the dedicated
reactive pilot path should not be treated as an identical observation procedure.

Scene distances come from the paired heightfield's exact prism geometry,
including step sides, or an analytic plane for an exactly constant field.
Unknown cells and boundaries carry invalid masks. Current PFNN scenes remain
single-valued heightfields, even though the observation points occupy 3D space.
`scene_geometry_field.py` supplies independent finite-solid 3D fixtures, but
the inspected production chain still queries `scene['heightfield_y_m']`.
Ceilings, bridge undersides and multilevel training scenes are not present in
the existing PFNN scene-pair receipts.

## Complete-coverage sampling proposal

This is the PFNN-only phase. PARC/BONES are not sampled in its main training
stream. Retain them as separate later-domain checks if the parent plan needs
to quantify forgetting.

The current runners require an implementation change before this policy can
run. `BalancedPrimitiveBatches` requires all three domains and fixes 50/25/25
quotas. `train_reactive_prefix_pilot.py` fixes two caches and a 4/2/2 domain
minibatch. `select_records` scans a bounded candidate subset and selects q0;
`ReactiveSourceBank.load` also rejects non-q0 records. Do not launch those
interfaces unchanged and describe the result as PFNN-only full coverage.

**Primary epoch:** seeded permutation without replacement of all 48,232 PFNN
training cache rows. This guarantees each of the 12,058 admitted motion/scene
windows and all four stored rotations are visited before the next epoch.
Do not drop the last partial batch. Multi-GPU sharding must preserve a single
global permutation without duplicate-padding training samples.

**Additional stratified pass:** after primary coverage, oversample rare start,
stop, sustained idle, side/back, turning and terrain-transition strata from
train only. Define strata from genuine source spans first; missing long spans
stay explicitly missing. Record extra exposures separately from primary epoch
coverage so a large oversampling budget cannot conceal unseen base examples.

A lower-cost alternative is a 12,058-window base epoch with a deterministic
rotating quarter; all four quarters are then covered over four base epochs.
Name its epoch unit explicitly. It is not a 48,232-example augmented epoch.

Publish after every epoch: expected/visited/missing cache IDs, distinct pair
windows, source-time windows, BVH stems, families, quarters and strata, plus
optimizer updates and executed/supervised frames. Persist the permutation seed
and cursor. Family balancing may change order or add exposures; it must not
replace the primary coverage pass with a bounded subset.

Validation is fixed, train-free, and evaluated across the original validation
split. Test remains sealed for model/hyperparameter selection. The existing
test families are not moved or inspected for training-sample choices.

## Interactive maps are a different asset set

The local catalog contains 15 original map IDs:
`000,001,002,003,004,005,006,007,009,010,011,012,013,014,urban`.
The source inventory lists 13 `data/heightmaps` fields; the map packager also
reads `demo/heightmaps` and deduplicates names. Its `Demo`/`Terrain` category
records source-folder membership, not a motion-capture pairing or data split.

The 7,125 training scene-pair records are separately constructed static
episode scenes. The catalog supplies no original-movement-to-demo-map temporal
alignment or contact labels. Thus loading every interactive map is useful for
controller evaluation, but supplies no additional supervised PFNN trajectories.
Do not count those maps as 15 independently captured motion/scene datasets.

## Remaining index facts needed before execution

- Per-BVH admitted counts, including any zero-admission clip despite all-80
  processing; reconcile both report and actual cache identities.
- Per-source unique covered timestamps/intervals, including tails, starred
  annotation exclusions, rejected intervals and repeated scene pairings.
- Maximal continuous same-scene, same-contact-clock admitted spans and their
  length/behavior distributions, particularly true sustained idle and braking.
- Train/validation/test geometry-kind cross tables and long-span independent
  facing coverage; global kind counts alone do not establish these cells.

All four can be computed later from the existing immutable indices and shards.
They are not reasons to regenerate or relax the completed dataset. Literature
comparison, architecture selection and training-budget planning are owned by
the parent plan; this document provides the verified corpus/perception inputs.
