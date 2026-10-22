# Native MB Local Terrain V1

## Implemented Scope

This is an explicit extension of native MB, not an unmodified-paper claim and not
an approved terrain controller. The default flat model has no new parameters.
Its accepted UE bundle and live services are not replaced by this work.

The existing tokenizer, Root duration/trajectory objectives, Pose masked-token
objective and decoder remain the baseline. Enabled Root and Pose backbones add a
small masked terrain CNN with zero-gated residual cross-attention. Pose also adds
a zero-gated per-token embedding of four relative root-height samples because
the original continuous root condition excludes hip height. At initialization,
loading all original weights and leaving the new gates at zero preserves the
baseline output. This is transfer initialization, not a trained terrain skill.

Reference implementations:

- MB: <https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks>
- MB paper: <https://arxiv.org/html/2604.24833v1>
- PARC: <https://github.com/mshoe/PARC>
- PARC paper: <https://arxiv.org/html/2505.04002v1>

The local geometry conditioning follows PARC's separation of offline environment
storage from bounded runtime observations. The dual patches and residual
adapters are our explicit integration choices, not architecture claimed from MB.

## Online Contract

At each planning call, provide one observation from the last available history
frame, not a sequence of observations along the future ground-truth motion.

- `points`: `[B,2,9,9,3]`, meters, Y-up, current root origin and body-yaw axes.
- `valid`: `[B,2,9,9]`, boolean; unknown is not flat ground.
- Patch 0: body-centered near field, +/-0.4 m at 0.1 m spacing.
- Patch 1: control-facing field, 0.2..1.4 m forward, +/-0.6 m laterally.
- Maximum horizontal radius is 1.6 m; larger maps cannot enter the encoder.
- The command is root-local XZ; zero command keeps a body-forward observation.
- Direction changes do not remove the near-body patch.
- An offline heightfield is only the source from which these queries are sampled.
  In UE, the corresponding local scene queries must produce the same contract.
  UE raycast integration and its frame-time benchmark are not yet delivered.

Both backbones receive the same observation, reused across Pose iterations.
The additional Pose height path is ground-truth Root during staged training and
predicted Root during inference, measured relative to the last history frame.
No future terrain, future contacts or full map is supplied to the controller.

The heightfield schema is suited to stairs and block/platform tops. It does not
represent overhangs, stacked floors or arbitrary handholds. The near-body patch
does not guarantee coverage of every extended hand. Full climbing needs explicit
local 3D/probe coverage and acceptance tests; retained hand labels alone do not
establish that capability.

## Data And Sampling

Terrain-enabled native shards add `terrain_points [T,2,9,9,3]` and
`terrain_valid [T,2,9,9]` beside the existing `global_features`. Each observation
must be constructed from that frame's state and explicit command. Data-loader
crops use the exact motion crop; batching pads validity with false. The native
training sampler's `chosen_ids` and `frame_indices` then select only the last
context frame. No prefix shortcut or independent terrain crop is permitted.
Root-local geometry is invariant to the common global heading/XZ augmentation.

PARC source conversion retains all 15 body-contact channels, including both
hands. Its source skeleton is not SOMA30. The proper coordinate rotation is
`(x,y,z) -> (x,z,-y)`; the heightfield's axis reversal and cell-center origin must
follow the same rotation. Source local joint frames remain source-local and
cannot be relabeled as SOMA rotations.

`hf_mask_inds` marks terrain cells protected during augmentation, not sensor
visibility. It must not become model validity. `hf_maxmin` is also an
augmentation constraint, not a measured contact or force label.

Raw official PARC generator outputs are candidates, not physics-corrected truth.
The released sample audit and pilot generation preserve their provenance. No
SOMA training is authorized by file existence or successful deserialization.

## Training Entry

`scripts/train_native_mb_control.py --terrain-conditioning --skip-review` opts
into the extension. The flat-only evaluator is deliberately not used to approve
terrain behavior. Existing run directories are never overwritten. A verified
flat 50k warmstart uses all original backbone keys exactly, allows only named new
terrain parameters, and retains zero gates. A flat optimizer state is not
silently resumed into a different architecture.

The native dataset manifest must bind a retarget/alignment acceptance report and
declare the terrain schema. Records must identify source groups kept within one
split. Augmentations of a base motion must not be split independently. Production
splitting must also hold out terrain families/seeds, not merely file names.
The existing tokenizer binding and normalization checks still apply.

Before behavior training, validate PARC-to-SOMA contact geometry and joint frames
on ascent, descent and hand-contact examples. Evaluate tokenizer reconstruction
on these motions first; a flat tokenizer must not be assumed to encode climbing.
Check normalized feature ranges, especially world root height on elevated
platforms, before reusing flat statistics. Do not clip motion or project feet to
hide a mismatch. No new behavioral training has been run by this implementation.

## Automated Checks

Tests cover bounded/unknown observations; coordinates and batch broadcasting;
current-frame row/window alignment; unchanged flat keys and zero-gate outputs;
terrain and height sensitivity; gradients through the real MB training methods;
frozen labels/tokenizer; and the complete native inference call forwarding
predicted Root height to Pose. These are engineering checks, not motion quality
or realtime performance evidence.

Behavior acceptance must additionally exercise stop/reverse/turn, ascending and
descending, a changed-terrain counterfactual, and held-out geometry with fixed
source masks. Score hand and foot contacts separately, collision/clearance,
slip, joint continuity and goal progress. Unknown geometry remains unknown.
There is no runtime IK, foot lock, root-height projection or corrective smoothing
in this extension.

## Verification Snapshot

On 2026-10-08 the combined focused suite passed 211 tests, with two deliberate
Root/Pose fixture exclusions (Pose-only height tests on Root), not failed data
checks hidden as skips. This includes native flat model/data/runtime regressions.

The real source bridge was exercised on all six modern paired samples in
`out/parc_release_audit_20261008/stairs/inspection.json`: frame counts were
130, 120, 295, 100, 258 and 798. The final sample retained 52 frames with a hand
contact; the first ascent and descent had root-height deltas +1.030 m and
-1.034 m. All six remain `training_ready=False`, as intended: this confirms
source loading, coordinate conversion and bounded observations, not SOMA motion
quality or newly generated terrain skill.

The official small PARC generator also ran all three requested pilot cases in
`out/parc_generation_pilot_20261008/run_20261007T200425Z/`. It produced 119-frame
ascent and descent candidates and a 15-frame hand-contact continuation (10 frames
with generated hand labels). The staircase endpoint distances were 0.890 m and
0.795 m, exceeding the pilot's 0.5 m completion threshold. The historical report
is `partial`. A subsequent frame-by-frame audit found that both clips had already
entered the goal region and then overshot; terminal distance alone was not
evidence of a generation timeout. Hand-contact continuation means generation
completed, not a successful climb. Outputs are uncorrected kinematic candidates, not collision-certified or
physics-tracked motions. They are not admitted to MB behavior training.

The executed generator script hash and official repository/checkpoint identities
are in `pilot_report.json`. An additional seven unit/real-artifact checks passed
against this exact run. No generator process was left running and no GPU was
used for this bounded CPU pilot.

## Official PARC Correction Pipeline

`scripts/run_parc_official_correction.py` supersedes the batch-one raw pilot for
quality comparisons. It invokes the released generator, filter and optimization
functions, using `data/configs/parc_2_kin_gen_default.yaml` without changing its
weights: 128 candidates, top two, DDIM stride 5, CFG 0.65, and 3,000 Adam
optimization steps at 0.001. These are released-code settings, not a claim that
every experiment in the paper used the same values. The test terrains and paths
remain the fixed pilot cases; they are not the full procedural terrain suite.

Two minimal vendor fixes are retained in `patches/parc_runtime_compat.patch`:

- Clone batch-expanded condition flags before upstream CFG mutates them.
- Record path completion before breaking, retain the last executed history frame
  rather than an extra predicted chunk, and retain all frames on timeout.

The patch does not alter architecture, weights, diffusion equations or losses.
Tests exercise the actual small checkpoint with batch two and fixed-noise
batch-one equivalence, plus immediate, staggered and timed-out path completion.
Apply this patch once to upstream commit
`846ed98bd526c1dd93ddc1701b601036d553cd2f` before using a fresh vendor checkout:

```powershell
git -C out/vendor/PARC apply ../../../patches/parc_runtime_compat.patch
python scripts/run_parc_official_correction.py --source-run out/parc_generation_pilot_20261008/run_20261007T200425Z --device cuda:0
```

Every run keeps all candidate scores and separate `raw`, `optimized` and `final`
motions. `optimized` precedes official hesitation removal and is frame-aligned
with `raw`; `final` may have fewer frames. The comparison renderer uses the same
camera and terrain for both sides, the saved official FK and unaltered body
poses. GIF playback is half speed, while source motions are 30 FPS. Body-origin
contact speed is explicitly not a measurement of material-point foot slip.

The small released checkpoint is not established as the paper's final-stage
generator. The official physics tracking/recording stage has not run: its matching
tracker checkpoint and supported Isaac Gym runtime are not available locally.
Kinematic optimization only makes candidates easier to track; it does not prove
dynamic balance, plausible forces or successful climbing. The hand case is a
15-frame continuation from two source history frames, not a full climb.

All these outputs remain `physics_corrected=false` and `training_approved=false`.
This is an offline teacher-data step, not an IK, foot-lock or smoothing layer in
the MB/UE runtime. No new MB training or UE changes are part of this correction.

### Executed Comparison

The staircase artifacts are under
`out/parc_official_correction_20261008/run_20261008T043915Z/`. Both variants of
ascent and descent completed all 3,000 iterations. The process subsequently hit
the explicit 35% GPU memory limit while scoring the 128 hand candidates; its
top-level `failed` status and error are intentionally preserved. This did not
invalidate or overwrite the completed stair artifacts.

Hand-only recovery uses `run_20261008T045907Z/` in the same output root. All 128
motions are still generated together, but their independent official SDF scores
are evaluated eight at a time. A real-model regression verifies equality with
unsplit scoring. This changes workspace memory, not candidates, scores, thresholds
or selection mathematics. No staircase optimization was rerun to recover it.
The hand-only recovery exited successfully. Across the two runs, all six
selected candidates completed 3,000 optimization steps each. Independent CPU
FK recomputation and SHA256 verification passed for all 18 saved stage motions;
the largest position difference from the saved GPU FK was 9.54e-7 m.

Primary candidate comparisons (raw selected candidate versus frame-aligned
optimized candidate, not the old batch-one videos):

| Case | Sampled max penetration, raw -> optimized (mm) | Body jerk P95, raw -> optimized (m/s^3) |
| --- | --- | --- |
| Ascent | 61.870 -> 1.298 | 3967.2 -> 892.2 |
| Descent | 73.817 -> 0.698 | 3430.6 -> 904.8 |
| Hand continuation | 40.542 -> 0.375 | 629.7 -> 824.8 |

Ascent and descent retain goal completion with final average-foot distances
0.210 m and 0.258 m, inside the official 0.5 m region. This is not proof of valid
force or slip. The hand clip's jerk P95 worsened; improvement is not uniform.
Likewise, the descent body-origin contact speed increased from 0.848 to
1.088 m/s on the fixed raw contact mask. That proxy is not material-point slip,
but it certainly does not support a claim that sliding was eliminated.

Each run has `renders/` with side-by-side GIFs and first/middle/last PNGs. The
three primary renders preserve every input pose, use identical stage terrains,
and have zero saved-FK body-origin placement error. Focused generation, data and
render tests passed 53 checks, including real checkpoint batching and opt-in
historical artifact checks. None of these checks approves physical training data.

## PARC Usage Review, 2026-10-08

The current generator-plus-kinematic-optimization outputs are not a reproduction
of the official long-horizon demo. The [project page](https://michaelx.io/parc/)
explicitly attributes those results to the generator and motion tracking
controller together. The [paper](https://arxiv.org/html/2505.04002v1), sections
5.3 and 7.1, describes blended denoising's smoothness/compliance tradeoff and the
subsequent simulation tracker. Section 7.2 separately evaluates raw generators;
it does not imply that kinematic inference can never produce good motion.

Verified findings:

- The historical raw pilot instantiated DDIM stride 10 and capped rollout at
  5 seconds, versus stride 5 and 10 seconds in the released generation YAML.
  The newer correction runner already uses the released settings. These old
  differences cannot by themselves explain the new videos, and the old
  endpoint miss must not be called a timeout without its completion history.
- `model_40000.ckpt` has d_model 256, four layers, 21 diffusion steps, 15-frame
  sequences and two history frames. Its saved `use_hf_collision_loss` is false.
  The checkpoint's source motion-library path refers to `jan022025`. Neither
  that path nor its public release establishes final-iteration provenance.
  This is a configuration mismatch to the paper, not proof of the sole cause
  of quality loss. Do not enable a training loss at inference and call it fixed.
- Inference loads raw `model` weights, matching the official loader's default
  `use_ema_model=False`; the presence of EMA weights is not an invocation bug.
- Two-frame history, 15-body contact order, local direction rotation and
  generated chunk consumption were checked against the actual source API.
  No additional invocation defect was established in these areas.
- A source-model/render-model comparison matched all 18 body geoms on owning
  body, type, local position, orientation and dimensions. FK rotations and
  terrain top footprints also agree. Decorative terrain column bottoms are
  not verified physical terrain. The primitive humanoid is not a skinned mesh.
- The original hand candidate and source context used different playback
  speeds (0.25x versus 0.5x). Both were labeled, but were not a useful cadence
  comparison. The hand generation is only 0.5 seconds, not a completed climb.
- Lower sampled penetration does not establish contact quality or balance.
  Some corrected metrics worsened, as recorded above. All candidates remain
  unapproved for MB training.

`scripts/render_parc_release_reference.py` now replays a released pair through
official CPU FK and the same primitive renderer, reading source FPS and keeping
every pose unchanged. Restricted loading, source hashes, per-frame FK/orientation
assertions and GIF timing checks are retained. No generator or optimizer runs.
Two actual replays completed under `out/parc_usage_review_20261008/`:

- `source_stairs_up`: initial-augmentation source, 130 frames at 30 FPS, 1x.
  First/middle/last frames were inspected. This is the reference presented to
  the user, not a new generated success or a matched-terrain quality comparison.
- `source_april_iter5`: a released iteration-5 sample, 128 frames at 30 FPS, 1x.
  The selected camera is occluded by tall terrain in the middle frame. This
  render is not presentation-approved and is not used as visual quality proof.

Both replays have zero FK-origin placement error. Correction-runner/render
regressions passed 15 tests in this review; actual replay is the new script's
end-to-end check. There was no training, new optimization, UE change, or change
to a released motion. An `_opt` filename does not establish physics provenance.

Development direction: use released paired motions as the next data candidates,
with stage provenance and SOMA retarget/contact acceptance, rather than continuing
to tune this small checkpoint and calling it the demo. Fresh synthetic data
requires an established generator/tracker pipeline; the official README suggests
MimicKit for future tracker work, but that port and its training are not done.
An offline physics teacher does not require physics, IK or foot locking in the
realtime MB/UE backend.

## Official Paired Source to SOMA Preflight

Executed on 2026-10-08, after the user agreed to prioritize released paired
motions over the small-generator candidates. Source selection is recorded in
`out/parc_source_selection_20261008/manifest.json`: 130 ascent frames, 120 descent
frames and 798 mixed stair/hand-contact frames, all at 30 Hz. Archive member,
release revision, inspected SHA256, byte size and body-label counts are bound.
These members belong to `initial_aug`; exact physical-recording provenance is
not established. They are grouped conservatively and have no train/test split.

Implementation:

- `motionbricks/data/parc_soma_retarget.py` adapts official PARC body FK to the
  official SOMA30 rig using the same pinned, unmodified GMR/Mink core as the
  LAFAN retargeting route. This is a custom skeleton adapter, not an official
  PARC-to-SOMA converter, and not a new IK solver.
- The 15 observed source bodies drive 30 distinct GMR task aliases, avoiding
  dictionary collisions for shared source bodies. Unobserved toe, finger and
  additional spine articulations get inherited orientation targets, not invented
  motion. Feet use reference sole axes rather than the sloping ankle-to-toe bone.
- Leg-length calibration gives scale 1.02931806735. Source positions, heightfield
  values, cell spacing and cell-center origin all receive that same world scale.
  A fixed morphology-derived root offset (approximately 0.11 m) is part of
  retarget task calibration, not a per-frame floor fit. Root-task errors below
  are relative to these calibrated targets, not the unmodified source pelvis.
- Foot targets align the official PARC box bottom-center and a reference anchor
  on the actual SOMA skinned sole region. Full SOMA LBS is used for parity before
  evaluating the identical selected-vertex LBS on all 196 sole vertices.
- The generated `local_rot_mats`, root trajectories, source contacts and paired
  terrain are preserved in NPZ. Neither playback nor any UE runtime applies IK,
  foot locking, pose smoothing or terrain shifting.

Results: `out/parc_soma_retarget_20261008/preflight-v2/report.json`.

| Case | Frames | Foot task position P95 (mm) | Labelled nearest sole-surface distance P95 (mm) | Sole penetration P99 / max (mm) |
| --- | ---: | ---: | ---: | ---: |
| Ascent | 130 | 17.33 | 16.67 | 2.80 / 18.90 |
| Descent | 120 | 18.23 | 28.45 | 0.29 / 12.80 |
| Mixed stair/hand contact | 798 | 18.43 | 16.56 | 2.03 / 9.70 |

All three pass the declared initial screening bounds; all remain
`training_approved=false`. A nearest patch point being close is not a certificate
that the intended material point is supporting the body. Surface measurements
use fixed source body labels, not labels selected from the prediction. No claim
of zero slip, correct palm contact, whole-body collision avoidance or dynamic
balance is made. MB tokenizer reconstruction has not yet been evaluated.

An actual scoring issue was corrected during preflight: pure vertical height
differences approach 0.9 m even on some labelled source-contact frames in the
mixed clip. Vertical top clearance cannot represent side-face contact. The
revised surface score uses PARC's official normal-box exterior and inverted-box
interior distance, including exposed stair faces; unknown map cells stay unknown.
The vertical values remain in the report as diagnostics. This correction did
not change the motions. An analytic step test checks both +5 cm exterior distance
and -5 cm interior distance where vertical differences would be +20/-80 cm.

Independent code review also caught an angular score limitation: equal step
magnitudes could hide opposite rotation directions. Per-frame orientation error
and geodesic relative-step error are now included and tested with +10/-10 degrees.
The largest relative-step errors were 2.72, 2.49 and 3.05 degrees respectively.

`scripts/render_parc_soma_retarget.py` produces matched-camera 1x comparisons:
official source primitives on the left, schematic SOMA bones and exact sole skin
points on the right. Both staircase clips are complete. The hand-contact render
is explicitly the contiguous 600:720 excerpt of its fully converted 798 frames.
Files are under each case's `render/` directory. All input hashes were unchanged;
all rendered joint/sole placement checks reported zero error. Representative
PNGs were inspected. The skeleton is not presented as a full skinned character.

Verification: 47 focused tests passed, including existing LAFAN and PARC-source
regressions. Runtime dependencies are recorded in the report; Mink 0.0.13 and
MuJoCo 3.8.0 were used. No model was trained, no UE code was changed, and none of
the old datasets or model outputs was overwritten. Remaining admission work is
actual contact/material-point correspondence, hand-surface/whole-body checks and
native MB encoding fidelity, not another search for a new generator.

Subagent policy from the user's latest instruction: gpt-6-luna at Max effort.
The active renderer task was restarted with that configuration, and the retarget
correctness review also used it. Earlier completed work is not relabeled as Max.

## Large PARC Generator and Platform Expansion

On 2026-10-08 the official README's legacy SharePoint release was successfully
inspected. Contrary to the earlier small-checkpoint-only inventory, it includes
`parc_models_demo_august_12_2025.zip` (1,225,656,443 bytes), with an iteration-3
generator and its matching tracker. This is the largest generator found in the
accessible official releases, not a verified final paper-iteration checkpoint.

The original generator has 43,073,051 parameters, width 1024, six layers, and
1001 diffusion timesteps. `convert_parc_demo_checkpoint.py` hash-pins the original
object, derives configuration from that object, verifies the reference YAML and
character asset, preserves feature statistics, and exports both raw and EMA
weights. Raw and EMA denoiser parity errors at three timesteps are all exactly
zero. This verifies those forwards and configuration contracts, not an end-to-end
legacy rollout equivalence claim. No fitting is performed.

The verified export is `out/parc_legacy_models_20261008/demo_iter3_verified.ckpt`,
SHA256 `a03a9d4cc7b3a43b217d67116429d241c92e31e7de93e02849832b0371860e2e`.
The original archive hash is
`330e19a24334d903a9311f489b83ce161022534a917b42f13322096482d66391`.

Review corrected two orchestration mistakes before the expanded batch: the
legacy runner selects EMA when enabled, and released generation uses DDIM
stride 5, not the training evaluation stride 50. The new generation runner reads
all `mdm_gen` and `mdm_path` settings from the official generation YAML, with
explicit batch-size and maximum-duration overrides. Missing EMA fails instead
of silently using raw weights. Frame-chunked scoring is regression-tested
against the upstream additive score, without changing its objective.

Eight preliminary raw-weight, stride-50 stair candidates exist in
`out/parc_long_stairs_20261008/local_batch8`. They are historical diagnostics,
not the corrected EMA batch or training-approved data. A complete 513-frame,
30 FPS render is in `candidate_000/rendered_raw_kinematic_20261008`; its source
hash is unchanged, FK-origin rendering error is zero, and orientation matrix
error is below 1e-6. A ten-second GIF is explicitly only an excerpt. No source
motion was concatenated, no pose was post-processed, and no physics was stepped.
The old report's `course_complete` is only endpoint plus summit visitation and
must not be interpreted as complete stair traversal. New reports instead record
dense ordered route milestones for every ascent, descent and platform segment;
even complete geometric coverage is not a contact or physicality certificate.
`local_batch8/geometry_reaudit.json` applies the new 0.6 m XY / 0.35 m vertical
milestone bounds to all eight historical candidates: none covers every route
segment within those bounds. This records route deviation, not proof that every
frame or stair interaction is physically invalid. No historical clip is promoted
to accepted training data by its endpoint flag or by the video.

Platform geometries are stored in `out/parc_platform_batch_20261008/courses`:

| Variant | Platform height | Width | Platform route | Whole route |
| --- | ---: | ---: | ---: | ---: |
| Low | 0.48 m | 4.8 m | 21.0 m | 30.2 m |
| Medium | 0.72 m | 6.4 m | 22.6 m | 33.4 m |
| High | 0.96 m | 8.0 m | 24.2 m | 35.0 m |

Each has separate ascent/descent stairs, two wide platform U-turns, and explicit
ordered world-space waypoints. The middle variant reverses lateral turn sense.
These are static 2.5D geometries, not overhangs or multi-level floors. PARC motion
sampling determines the realized gait; authoring a route does not add a native
stop/strafe/speed control to the released generator. The existing 41 m course
with three separated stair flights remains unchanged.

The user restricted the next server work to g2/g3 at `remote-host.example`, ports
12302/12303. Existing host keys remain verified. Do not use g1 pending trusted
key verification, do not displace other users' GPU work, and do not transfer
files larger than 10 GB directly through this address. All new generated clips
remain `training_approved=false` until the downstream admission stages. Matching
tracker weights are now available; the supported Isaac Gym runtime and actual
physics recording are still not established. No new MB training or UE change
is included in this expansion.

### Remote Batch Dispatch

`scripts/deploy_parc_remote_batch.py` deployed an immutable snapshot to both g2
and g3. Batch ID: `ema_stride5_20261008_v1`. Both hosts successfully loaded the
actual EMA checkpoint on CPU, with strict parameter loading and finite 15x91
normalization tensors. g2 uses PyTorch 2.5.1+cu124 and g3 2.11.0+cu126 in isolated
private-user-owned overlays; parent environments were not modified.

At dispatch, both queues are **waiting for idle hardware, not generating**:
g3 queue PID 1366980, started 13:17:56 UTC; g2 PID 2901287, started 13:21:08 UTC.
Each server holds six platform batches (three geometries, two seeds each) and
one long-stair batch, eight candidates per batch. The maximum is 96 platform
and 16 stair candidates, not a count of completed or accepted samples. A clip
has a 45-second generation limit, official EMA/stride-5 sampling, and no physics
tracking, retargeting or training in this queue.

The finite queues expire after 24 hours. Every job needs two idle GPU samples,
no existing compute process, at least 20 GiB free, a per-GPU private-user lock, and a
CUDA smoke check. If a foreign compute job appears, only our own child process
is stopped. Missing telemetry fails closed. Existing outputs are never rerun or
overwritten. Runtime states and individual job logs are persisted under each
server's `${PRIVATE_STORAGE}/parc-remote-batch-20261008/runs/ema_stride5_20261008_v1`.

Local dispatch evidence:
`out/parc_remote_deployment_20261008/ema_stride5_20261008_v1.deployment.json`.
Source snapshot SHA256:
`79975c2002632ddcb3da61dc766e8228bca6e8f8b7d4be2e2e0743bbb8b2a857`.
The post-snapshot queue bootstrap is separately hash-recorded in each launch
record. Transfer ledger total is 710,118,420 bytes including bootstrap revisions;
the largest object is 354,918,400 bytes, below the user's direct-transfer limit.

Verification for this change: 46 new/affected geometry, converter, generation
and renderer tests passed; 55 existing PARC regressions passed with two skipped;
11 deployment tests passed. This is focused verification, not the entire repo
test suite. Automatic generation evidence and later physics/retarget admission
remain distinct from tests of the software contracts.

### Raw Batch Completion and Physics Boundary (2026-10-09)

The finite batch is now generated and rendered, not physics-approved:
112 original candidates plus 48 extensions, 160 distinct source-file hashes,
and 160 full-length 30 FPS videos. Source motion spans sum to 2659.2667 seconds
(44.32 minutes); encoded video duration is 2664.6 seconds because each video
includes its final frame. The frame count of every decoded video was checked
against both its render manifest and its generation report. No poses were
modified, no IK or foot locking was applied, and no physics was stepped.

All artifacts are under `out/parc_complete_20261008`:
`gallery/index.html`, `gallery/catalog.json`, `summary.json`, and seven full
representative videos in `showcase/`. All 160 candidates, including route
failures, remain in the gallery. Desktop (1440x1000) and mobile (390x844)
browser checks passed video playback, filtering, and overflow checks.

| Geometry | Candidates | Complete geometric route coverage |
| --- | ---: | ---: |
| Ascending terraces | 16 | 15 |
| Elevated platforms with bridges | 16 | 5 |
| Broad S-shaped platform route | 16 | 11 |
| Low platform | 32 | 11 |
| Medium platform | 32 | 8 |
| High platform | 32 | 16 |
| Three separated stair flights | 16 | 1 |
| Total | 160 | 67 |

The geometric gate is unchanged (ordered route milestones within 0.6 m XY and
0.35 m vertically). Passing it is not evidence of physical feasibility or
correct contact. The 93 remaining candidates are not silently discarded or
promoted through endpoint-only success. No candidate has been admitted to MB
training or retargeted to SOMA in this batch.

g3 produced 80 candidates. Other users occupied the server GPUs afterward;
only our idle queues were cancelled and unstarted jobs were reassigned. The
local RTX 3080 produced the remaining 32 original candidates and all 48
extensions. The immutable source snapshots and job commands are retained.
No other user's process was stopped, and no file above the user's 10 GB limit
was directly transferred. The newly authorized 4090D is not yet connected:
its SSH identity and NAS mount still need to be identified.

Completion-path review added exact candidate-identity matching and direct
source-frame-count checks to the summary verifier. Its regression tests and
the queue/collection tests pass (25 tests). An additional 46 existing PARC
regressions pass after fixing test imports that collided with the vendor's
own `scripts` package. These are focused suites, not a full repository run.

The user correctly identified that physics recording is still required.
The official guide specifies `parc_3_tracker.py` for controller training and
`parc_4_phys_record.py` for physics recording. The released iteration-3
`p3_tracker/model.pt` (42,573,728 bytes), agent configuration, and environment
configuration are already present. They have not yet been executed and
validated against the current runtime. The current g2 and g3 generation
environments use Python 3.10 and 3.11 respectively and cannot find `isaacgym`;
this is an environment probe, not a claim that no installation exists elsewhere.

The next admission stage must use the official tracker on each paired terrain
and reference motion, recording actual simulated body states and force-derived
contacts. Do not substitute kinematic playback or gravity-only replay. Record
full-start success, partial salvage, and failures separately: the official
recorder retries at 10%-50% start offsets, so its aggregate printed success is
not a full-trajectory success metric. Preserve source provenance and terrain
localization transforms. Visualize raw and physically tracked motion side by
side before any SOMA retargeting or terrain-aware MB training. A pretrained
tracker is not guaranteed to repair every new candidate; failed tracking must
remain failed, with controller training following the official third stage
only if needed.

### 4090D Official Physics Recording (2026-10-09)

**R1 qualification withdrawn after visual review:** the table and durations
below describe an actual run, NOT validated terrain-paired data. Input terrain
spacing was 0.2 m while the archived environment packed it at 0.4 m. The public
square builder copies cells without resampling, but creates collision meshes
at source spacing. This desynchronized height observations, termination
heights and recorded terrain from collision geometry. A rendered terrace
recording exposed an entirely flat saved heightfield. R1 remains preserved;
none of it was training-approved. R2 aligns the packed grid to source spacing
and adds all-cell and saved-terrain world-coordinate checks. Controller ray
spacing, model weights, physics parameters and motion inputs are unchanged.

The user-provided endpoint `private-user@remote-host.example:32297` is now verified using
the existing RSA private key and the already trusted SSH host key. Host-key
verification was not disabled. The host is HKU-4090D with two 24 GB RTX 4090 D
GPUs. Only idle GPU 1 was used; no other user's process was stopped.
`/data1` is the NAS mount and `${PRIVATE_STORAGE} passed a write/read probe. Free-space
snapshots were about 7.7 TB on NAS and 270 GB on the local root filesystem.

An isolated Python 3.8 / Torch 2.2.0+cu118 / NumPy 1.23.5 environment was
created at `${PRIVATE_STORAGE}/projects/motionbricks-parc-physics-20261009/env`.
The official Isaac Gym Preview 4 archive SHA256 is
`c51d77ac1144b91b356f439c7313d8d59468147e37187ebffe7cdbc3154fdb8f`.
A GPU PhysX gravity/contact smoke test passed: the 0.1 m cube settled at
0.052 m center height, with nonzero measured contact force. Existing Conda
environments and system proxy settings were not modified.

The immutable 97,638,400-byte input bundle contains all 160 raw candidates,
the required official code/character asset, and the matching archived
iteration-3 tracker configuration and checkpoint. Bundle SHA256:
`8aaa06059b83a79e2efb8ebddd068abe8e0c78861ab1e6a53791d2ca8abcbeb7`.
All 259 bundled files were hash-verified on NAS. The checkpoint SHA256 is
`b96c3ac9e2e111527e7232955e82548cacf1a892179d15359705049cb379492d`.
It loaded strictly into the actual 1312-observation/28-action environment.
NumPy 2 pickle module-name aliases allow reading raw arrays in NumPy 1.23;
49 representative array byte hashes matched across environments.

`run_parc_physics_record.py` calls the official `record_dm_motions` unchanged,
with the archived physics/controller settings. The public recorder still
uses `_motion_lib` while the public environment renamed it `_mlib`; the
wrapper installs an identity-preserving alias only. The first failed pilot
and its four saved recordings remain preserved. The second pilot completed
with 4 full-start successes and 3 failures. No reference-state recording,
failure bypass, IK, foot locking, or smoothing was enabled.

The full run `physics_all_r1` then processed all 160 candidates in 224.825 s.
Its actual saved files, not the recorder's cumulative printed counters,
partition the batch as follows:

| Terrain family | Candidates | Full start and all frames | Partial only | Failed |
| --- | ---: | ---: | ---: | ---: |
| Ascending terraces | 16 | 12 | 2 | 2 |
| Platforms with bridges | 16 | 13 | 2 | 1 |
| Broad S-shaped route | 16 | 10 | 5 | 1 |
| Low platform | 32 | 24 | 5 | 3 |
| Medium platform | 32 | 23 | 7 | 2 |
| High platform | 32 | 12 | 18 | 2 |
| Three separated stair flights | 16 | 1 | 3 | 12 |
| Total | 160 | 95 | 42 | 23 |

The 95 complete recordings span 1518.433 seconds (25.31 min); the 42 partial
recordings span 545.2 seconds (9.09 min). Partial retries begin at the official
10%-50% offsets and are never called complete clips. Contact labels come
from the recorder's actual simulated contact-force threshold, not source
contact labels. Original motion/terrain hashes remained unchanged.

Important boundary: the only full-start three-flight reference does NOT
cover its complete requested route. There is still no evidence of a full
physical traversal of that three-flight course. 41 complete physical
recordings have references that covered their raw route, but route coverage
has not yet been recomputed on the simulated trajectories. No recording is
automatically training-approved. Retargeting to SOMA and terrain-aware MB
training have not been performed for this batch.

NAS run directory:
`${PRIVATE_STORAGE}/motionbricks-parc-physics-20261009/runs/physics_all_r1`.
Local reports: `out/parc_physics_20261009/engine_smoke_r2.json`,
`tracker_preflight_r1.json`, `physics_pilot_r2.json`, `physics_all_r1.json`,
and `summary.json`. Original recorded files retain observation arrays on NAS.
All individual transfers were below 202 MB, far below the user's 10 GB limit.

### Corrected Grid-Consistent R2 Results (2026-10-09)

Use `physics_all_r2` and `summary_r2.json`, not the withdrawn R1 summary above.
The only additional environment configuration change sets the packed terrain
spacing to the input's measured 0.20000000298 m. No resampling, reference-pose
changes, controller-weight changes, IK, or smoothing is involved. In particular,
the pretrained policy's 0.05 m ray layout remains unchanged. The official
square builder at PARC commit `846ed98bd526c1dd93ddc1701b601036d553cd2f`
requires identical source and packed cell sizes because it copies cell arrays
without resampling. The two modified vendor files belong to earlier generator
work; tracker and conversion code used here are unmodified official files.

R2 checks all 2,432,640 padded tile cells in world coordinates before recording:
maximum source/packed height difference is 0.0 m. Every saved motion also
checks its localized terrain against simulated world heights; all 153 saved
files have 0.0 m maximum error. Each records its localization translation.
The summarizer, converter and renderer now reject missing terrain evidence.

| Terrain family | Candidates | Full start and all frames | Partial only | Failed |
| --- | ---: | ---: | ---: | ---: |
| Ascending terraces | 16 | 16 | 0 | 0 |
| Platforms with bridges | 16 | 13 | 3 | 0 |
| Broad S-shaped route | 16 | 15 | 1 | 0 |
| Low platform | 32 | 30 | 1 | 1 |
| Medium platform | 32 | 31 | 1 | 0 |
| High platform | 32 | 27 | 4 | 1 |
| Three separated stair flights | 16 | 3 | 8 | 5 |
| Total | 160 | 135 | 18 | 7 |

The run completed in 219.417 s. Complete motion spans total 2205.533 s
(36.76 min), partial spans 212.433 s (3.54 min). 63 full-start recordings
correspond to raw references that covered their requested route, including
one three-flight reference. This is not yet recomputed simulated-route coverage.
All raw input hashes remained unchanged and no recording is training-approved.

All 153 saved files were converted with the official legacy converter into
new nested files, preserving pose, contacts, terrain and observation arrays.
Converter SHA256:
`ef4b0c432df7c0c7f16ea12c0a6f493e8b7835fc81993319180fec4e605e6d1f`.
It is separately recorded as supplemental code, not silently added to the
immutable bundle manifest. Output round-trip and per-file hashes were checked.
NAS originals remain at `runs/physics_all_r2`, converted data at
`exports/physics_all_r2`, under `${PRIVATE_STORAGE}/motionbricks-parc-physics-20261009`.

The recorder preserves `hf_maxmin` verbatim; that auxiliary field is not rebuilt
by the official square builder. It must not be assumed valid for later terrain
supervision. The grid checks above validate the actual `hf` surface used for
policy observations and rendering, not every auxiliary terrain representation.
Further admission still needs simulated-route/contact quality checks and SOMA
retarget validation. No MB training or UE code was changed in this task.

Seven complete raw/simulated side-by-side videos are delivered at
`out/parc_physics_20261009/comparison_r2/index.html`. All 3,451 video frames
were decoded and counted at source FPS. Every rendered body position matches
official FK exactly; maximum orientation-matrix error is 7.78e-16. Nonblank
checks pass for both panels, and terrace, high-platform and stair midpoints
were visually inspected. All poses are replayed unchanged. The render-only
MuJoCo scene does not simulate physics; the right panel's states come from
the already completed Isaac Gym run. Terrain cell columns visualize the
recorded surface, and cameras follow each paired world independently.

Focused verification: 65 tests pass across new recording/export/comparison/
summary contracts and existing source/long-course renderer regressions.
Both physics runs have ended, with no task-owned GPU process left running.
