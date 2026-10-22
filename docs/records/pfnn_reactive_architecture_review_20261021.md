# PFNN Reactive Locomotion: Independent Architecture Review

Final parent decision: `pfnn_all_data_3d_plan_20261021.md` makes the compact
framewise controller the main performance route and keeps MB four-frame
execution as a comparison. The final plan requires the full3D geometry backend.
Inventory correction:80motion files consist of40originals and40mirrors,
not80independent captures. Initial recommendations below are retained as review history.

2026-10-21. Planning and source review only. This document is the sole output:
no runtime edits, model calls, service changes, SSH, GPU work or training.
The literature discussion reuses the primary-source reviews already in this
repository; links below were not freshly fetched for this review. The parent
review's fresh literature work and separate PFNN data audit should be merged
into the execution plan before implementation.

## Decision

Revision: the primary proposal is now a **trained one-frame fast policy with
our shared environment perception**. MB four-frame adaptation is a bounded
comparison, rather than a prerequisite for starting that policy. Use all
admitted PFNN training data with the existing recording-family splits.

The current deployed observation is **3D sampling over 2.5D heightfield-derived
solids**. It samples XYZ points, but its source geometry cannot describe an
overhang, ceiling or two independently traversable levels at the same XZ.
The proposed backend must describe complete static 3D solids and known free
space. That geometry upgrade is part of this plan, not a deferred extension.
Keep our body/path/actual-velocity feature construction and shared encoder:

```text
ONE scene manifest -> matching rendered geometry / 3D queries / quality checks

played history + recurrent state + latest movement/facing intent
  -> body / intent / actual-velocity 3D distance-normal-valid observations
  -> shared environment encoder -> small Root/preview head
  -> resample the 3D corridor along that predicted Root
  -> same environment encoder -> small pose/contact head -> next actual frame
  -> commit played history and recurrent state once
```

The policy is trained for one 30 Hz step. Its preview anticipates terrain;
preview frames are not committed body motion. Retain MB as an explicit
four-frame baseline and an optional offline motion prior, with current-intent
conditioning and contact objectives tested separately below.

Keep **geometry capability** and **learned behavior coverage** separate. PFNN
first supplies its admitted locomotion/terrain pairs. Full 3D queries can see
a low ceiling or platform underside, while crawling, climbing or support
transitions still require corresponding motion/scene examples. Section 4
defines geometry acceptance independently of those later behaviors.

## 1. Findings That Affect The Architecture

**P1: Prefix supervision retained the old command interpretation.**
`native_mb_online_conditions.py::sparse_velocity_conditions` writes velocity
and heading into sparse slot 7. Inference expands the terminal condition to
frame 31 in `motion_inference.py::_predict_root_trajectories`. The substituted
local Root then conditions Pose and the decoder. The pilot changed command
labels and losses, but this interpretation remained active. Its generated-
history endpoint improved forward speed from 0.8463 to 1.1946 m/s, while idle
16-second displacement rose from 0.2696 to 1.1793 m and stopping from 0.933 to
2.200 s. Those results favor a changed condition and objective over a larger
repeat of that pilot. See `reactive_prefix_training_20261021.md`.

**P1: Removing the terminal condition alone is insufficient.**
`terrain_model_v3.py::JointTerrainController` already uses
`sparse_soft_intent_conditions`, zero-output-initialized command embeddings and
no future hard conditions. The historical deployment also changed weights,
scene branches, supervision and rollout history; it produced reduced leg
excursion and trembling. `joint_sdf_results_20261012.md` records these results.
Reuse its mask-construction experience, but start the new comparison from the
immutable final parent, not that failed candidate. Isolate condition changes
from decoder and perception changes.

**P1: Fresh-window generation has no retained gait state beyond four poses.**
`motion_inference.py::_predict_pose_tokens` starts with masked tokens on every
call. The codec has stride four and a wider symmetric temporal receptive field.
One- or two-frame execution changes the alignment of the next latent block.
Four-frame execution aligns blocks but still permits different continuations.
The existing exact-history audit establishes correct bookkeeping; source review
and replay show that this alone does not recover a stable gait. See
`reactive_pose_context_audit_20261021.md` and `reactive_gait_repair_20261021.md`.

**P1: Contact and geometric objectives were absent from the prefix pilot.**
`reactive_prefix_training.py::prefix_loss` has Root XYZ, Root-relative FK,
rotation and boundary-inclusive joint velocity. Better reconstruction or
progress can therefore coexist with worse support and penetration. The new
objective must evaluate geometry on the decoded, globally integrated body,
and retain source-derived support intervals independently of predicted contact.

**P2: The pilot reader is not an all-PFNN, long-sequence reader.**
`reactive_training_data.py::select_records` is quota-limited and q0-only;
`ReactiveSourceBank.load` returns exactly 33 frames. Its source, scene, contact
clock and hash checks are useful. Its selection and temporal extent must change
for this task. The latest pilot's 96 mixed-domain training windows are not the
next training pool.

## 2. PFNN Scope And Command Meaning

Use the complete admitted PFNN **train** manifest, including its admitted
terrain pairings and documented augmentations. Keep validation for selection
and test for the final fixed protocol. Historical corpus counts are 80 capture
sources, 233,010 real 30 Hz frames and 48,232 augmented training primitives;
the separate data audit should bind the exact eligible counts and hashes.
Report recording families, real frames, paired windows and augmented examples
separately. Existing mixed-pretrained initialization remains mixed-pretrained
even when all new optimizer draws are PFNN-only.

Keep PFNN world_v2, original scene/contact bindings and existing splits.
Load full admitted source intervals for sequence objectives; reuse the existing
retarget and geometry assets. A longer interval is eligible only while its
motion, scene transform and contact clock remain jointly valid. Repeating
frames, joining neighboring cache rows or extending a finite scene cannot
supply missing long examples.

The actor receives current desired world VX/VZ and desired body heading
independently, matching left-stick movement and right-stick facing. Translate
them to the observed body's yaw frame for the new embedding. For the MB
comparison's temporal position i, propose:

```text
c_i = [vx_body/3, vz_body/3, cos(desired_yaw - observed_yaw),
       sin(desired_yaw - observed_yaw), 4/30, 28/30, tau_i]
tau_i = (4*i + 1.5 - 3)/30
```

Three m/s is an input scale, not a speed clamp. Reuse the repository's yaw
conversion and verify it against the native representation. A body-relative
command does not by itself make the whole model rotation invariant; rotate
history, source labels and scene together in augmentation.

The one-frame policy takes the same first four current-intent fields. Its
integration step is 1/30 s, and preview sample times belong to the Root preview
head. The MB-specific four-frame execution interval and token-relative time
are not its action interface.

Source-derived near-prefix velocity and heading provide hindsight intent
labels for imitation. PFNN does not contain measured joystick inputs. In
particular, an average upcoming velocity label describes an achieved motion,
whereas a sudden zero joystick request asks the model to brake. Include genuine
braking and idle intervals, label the intended stopping event explicitly on the
data side, and supervise their real transition. Do not label a moving source
pose sequence as stationary or force its next-frame velocity to zero.

Count left/right strafe, reverse and independent turning relative to body
facing. Rotating forward motion and the body together does not create strafe.
If a required control class is absent, retain that coverage gap in the plan;
the data audit determines what PFNN alone can supervise.

## 3. Primary One-Frame Policy And MB Comparison

### One-frame policy: reuse the causal skeleton, replace its perception interface

`scene_control_v3/model.py::CausalSceneActor` already contains a small GRU,
Root/preview heads, a pose head and caller-owned memory. Reuse that structural
work and `scene_control_v3/native.py`'s official SOMA/FK boundary. The existing
model is an untrained capacity probe, with a 512-point, 16-dimensional feature
encoder and one joint Root/body forward. It does not implement the perception
and post-Root ordering proposed here.

Replace its scene encoder with our `SharedTerrainEncoder` and 15-dimensional
body/intent/actual-velocity features with explicit validity. First encode the
coarse three-region observation, fuse history/current intent, and advance the
GRU **once**. Predict next Root XYZ increment and an anticipatory XYZ Root path.
Re-query that path with the full 3D backend, encode the refined scene with the
same shared weights, then predict next-frame rotations and foot contact from
the new hidden state plus refined scene features. Do not advance the GRU a
second time when refining the pose.

Root and pose are learned outputs. The one-frame output is integrated through
the existing native representation and official FK, with no independent Root
replacement or pose filtering. Feed back the actual emitted frame; an accepted
display ACK commits its history and hidden state together exactly once. A
candidate must not advance recurrent memory before that commit. Latest intent
is read for each step, independently of the old four-frame cadence.

Train new history/intent fusion, GRU and Root/pose/contact heads. Initialize the
shared encoder from the final parent and first freeze it, so the first policy
comparison retains the existing feature representation. A separately declared
encoder-adaptation stage can then train it on admitted 3D scene pairs if terrain
response is inadequate. Keep the original MB Root, Pose, codec and codebook
frozen outside the fast policy. It requires no MB token CE or online MB call.

Use only admitted foot labels for the initial contact head. The v3 probe's
synthetic contact observations and 15-slot capacity layout are not PFNN source
labels. Any observed contact inputs must come from causal played-body geometry
and explicit valid mappings, rather than future source contact values.

### MB comparison: separate current intent from hard motion conditions

For the four-frame comparison, use a new candidate protocol and loader.
Preserve only the four observed Root
and pose frames, plus the three measured velocity intervals. Mask all future
Root, heading, pose and velocity fields as unknown. Desired intent enters two
small MLPs, `adaptation.intent_root` and `adaptation.intent_pose`, each
Linear(7,128), GELU, Linear(128,width), with the final layer initialized to zero.

For the new protocol, derive Pose/codec local Root from the learned Root
trajectory. Do not inject commanded velocity at the endpoint. Clear the old
terminal heading mask as well: deleting only `apply_terminal_velocity` leaves
another terminal request in sparse slot 7.

The reusable `sparse_soft_intent_conditions` clears the relevant future masks,
but the surrounding production path still dispatches by velocity-v2 versus
planar-v1. Register the new protocol explicitly through construction, inference,
serialization and loading. In particular,
`UnifiedTerrainController._predict` currently reads requested velocity from
`args[2][:,7,1:3]`; after clearing that slot it would see zero. Pass latest intent
explicitly to the sampler instead. Update `online_command_goal`, condition
dispatch and accelerated query inputs together.

### MB hooks are feasible, with exact tensor ownership

| Location | Proposed use | Source constraint |
| --- | --- | --- |
| `root_backbone.py::_root_token_transformer_model` input | Add Root intent to the temporal-token suffix, after the existing unified scene pre-hook. | The input also contains a duration token and eight history-frame embeddings; do not treat those as future temporal positions. |
| `pose_backbone.py::_transformer_model` input | Add Pose intent to valid motion-token positions on each of four passes, after the existing unified scene pre-hook. | An optional text token precedes motion tokens; use the actual suffix and token mask. |
| `unified_terrain.py::_decoder_hook` | Retain trained scene and predicted-height conditioning. | No new direct decoder-intent branch in the first comparison. |

Register the intent hooks after the existing scene hooks, record their order,
and capture that same order for deployment. Fixed eight-token operation needs
no new duration model or extra Root pass. A proposed intent injection into Root
stage 1 is a later ablation, not part of the minimum.

Zero initialization gives zero **added residual** for otherwise identical
inputs. Removing the parent's terminal conditions changes those inputs, so
whole-actor parity with the old terminal-conditioned parent is not an admission
criterion. Check zero-residual parity against the same new, history-only
condition path, and retain the original parent as a separate behavior baseline.

### MB comparison's trainable scope

First warm only the two intent MLPs and existing
`adaptation.root_stage1`, `adaptation.root_stage2`, `adaptation.pose` residuals.
Then, within a declared second stage, include:

- Root stage-2 transformer's last two layers and `root._conv_output`.
- Pose transformer's last two layers, `pose._proj_pose_output_logit`,
  `pose._proj_local_root_values` and `pose._proj_local_pose`.

The recorded architecture snapshot has three Root stage-2 layers and sixteen
Pose layers: expected indices are Root 1/2 and Pose 14/15. Expand the actual
loaded names, shapes and count into the training contract before an update.
Keep dropout and EMA behavior in eval mode while enabling selected gradients.

Initially freeze the shared `adaptation.encoder`, native terrain encoders and
cross-attention, codec encoder, quantizer, EMA, the entire codec decoder,
`adaptation.decoder` and `adaptation.height_encoder`. Frozen weights still
propagate gradients to their inputs. At zero initialization, lower intent-MLP
layers can receive zero gradient on the first update; require the final
projection to receive a valid gradient and inspect the lower layers after it
has moved.

Measure source-token reconstruction under the frozen decoder on every terrain
class before deciding it is sufficient. Existing codec-ceiling failures make
this material. If reference reconstruction already breaks the contact gate,
follow with a separately paired decoder-only adaptation of `codec.decoder`,
`adaptation.decoder` and `adaptation.height_encoder`, retaining source
reconstruction and token-usage checks. Changing only the decoder preserves
encoder token labels; changing the encoder, normalization or codebook requires
freezing that new codec and regenerating Pose labels.

## 4. Full 3D Geometry With Our Perception

Keep `VolumetricEgoSampler`'s three 8x8x8 regions: 1,536 points with location,
signed distance, normal and validity. `SharedTerrainEncoder` forms 24 spatial
tokens by pooling eight cells per region. Preserve this common feature layout
for the fast policy and MB comparison. The MB comparison additionally retains
its native local terrain branch and decoder's predicted-height features;
include those additional inputs and costs in comparison reports.

### Replace the heightfield-only backend

Use one versioned scene manifest containing the admitted PFNN terrain component,
closed 3D solids/meshes, world transforms and explicit known-volume bounds.
Rendering, actor queries and independent quality assessment must resolve the
same manifest and transforms. Keep the original PFNN surface and contact
bindings when importing it. Ground can remain a terrain component inside the
general scene; ceilings, walls, decks and free space above/below them must be
independently representable. A single maximum/minimum height per XZ cannot be
the common geometry backend.

`scene_geometry_field.py::SceneManifest`, `SceneGeometryField.query` and
`SceneManifest.meshes` supply an existing bounded-OBB foundation, including
distance, normal, validity and persistent surface IDs. Its `observation_field`
adapter already exposes the three channels used by our sampler. It currently
supports boxes, not a production PFNN-terrain/mesh composition. Extend that
scene representation rather than silently routing production back through
`UnifiedTerrainController.scene_field -> field_from_heightfield`.

The current OBB field takes a minimum of component SDFs. That gives correct
exterior distance, but overlapping interiors can have a different distance to
the union's exposed boundary. Establish union-boundary semantics and certify
interior depth before using those values for the 2 cm penetration gate. Keep
the geometry oracle independent of the accelerated actor query implementation;
both consume the same scene. Open meshes need an explicit surface-only or
closed-solid interpretation, rather than guessed signed distance.

`scene_control_v3/geometry.py` supplies another useful BVH/mesh-query prototype.
Its 512-query/8-object bounds, disabled Warp backward and separate 16-dimensional
feature schema need explicit adaptation. Reuse acceleration and native-world
registration where applicable, not its old observation contract or synthetic
contact state. A production backend must return the same signed-distance,
normal and validity semantics for PFNN terrain and added 3D solids.

The auxiliary local-height observation in MB needs a designated support-surface
query on a multilevel scene. Choose it from current support and the predicted
path; do not select the highest surface at that XZ, which could be a ceiling.
Keep this auxiliary input separate from the full 3D distance field. The fast
policy's common perception does not require flattening the scene for that
legacy branch.

### Follow the predicted path in XYZ

Every actor call uses this ordering:

1. Query body, latest-intent corridor and actual-velocity corridor from the
   played history. Encode the coarse observation for Root.
2. Predict Root once. Follow its anticipatory XYZ path for the fast policy, or
   playable Root path for MB, retaining body and actual-velocity coverage.
3. Encode the refreshed observation with shared weights. The fast policy's
   pose head uses it in the same step; MB's Pose/decoder use it after Root.

`PredictedRootObservation.sample` currently moves corridor centers in XZ and
retains coarse-volume Y coordinates. Extend corridor centers to predicted XYZ
while preserving a defined vertical span and the region identity. This allows
queries below a deck, near a ceiling and along an ascent. Record the vertical
origin, path times and sampling extents. Sparse 8-cube sampling can still miss
thin features; stress thin beams, edges and high-speed approaches before
changing the point budget or feature pooling.

For MB training, reuse `TrainingRootPathObservation._after_root`: the base
observer explicitly skips training. Implement the same query ordering in the
fast policy's training forward. Re-query after generated-history updates;
cached source-coordinate samples are not the new played-state observation.

### Geometry acceptance and behavior coverage are separate

| Case | Geometry requirement now | Motion/data requirement |
| --- | --- | --- |
| Original PFNN paired scenes | Match admitted terrain surface, normals, known area and contact frame. | Train and evaluate only the locomotion and transitions present in the admitted source. |
| Ceiling over uneven ground; wall beside step; bridge above free lower space | Query/render/oracle agree on solids, both free levels, normals and surface identities. | These existing fixtures are geometry tests; the bridge fixture has no access-ramp or climbing motion labels. |
| Additional solids outside a source motion's swept body/support region | Retain exact scene identity and demonstrate unchanged required clearance/support. | May preserve an existing motion label after full swept-body and contact validation; it adds no new skill label. |
| Low ceiling intersecting a walking source; climbable platform; hand-supported traversal | Represent and assess the constraint correctly. | Require an appropriate paired motion or separately validated offline synthesis before behavior training. |

PFNN-first behavior reporting should cover its available walking/running,
starting/stopping, turning and terrain transitions after the source audit.
Full 3D perception can be validated now with authored geometry. Crawling,
climbing and new support transitions remain explicit later data/curriculum
items. Do not attach an unchanged walking target to an intersecting ceiling,
invent contact labels or count a geometry fixture as a successful skill trial.

### Geometry gradients

There are two different geometry gradients to implement deliberately:

| Operation | Gradient policy |
| --- | --- |
| Actor's scene-query coordinates and field observations | Detached, as in the current training helper. The actor learns from refreshed causal inputs without differentiating through query placement. |
| Collision/support loss evaluated at decoded body surfaces | Differentiable with respect to predicted XYZ, FK, rotations and Root; geometry and source masks remain fixed. |

`VolumetricEgoSampler.observe_points` is no-grad and cannot be used to obtain
the second row's gradient. The new full 3D geometry-loss path must expose
gradients to body query points, even if the actor's fast query runs without
backward. Detached nearest-face selection followed by a differentiable local
distance reconstruction is one implementation route, with separate handling
of signed interior/union semantics.

For the PFNN component, `HeightfieldPrismSDF.query` already selects a nearest
prism without gradients and reconstructs distance from live point tensors.
The accelerated sphere selector follows the same discrete-selection pattern;
establish gradient equivalence separately from its forward-value proof. The
OBB prototype's differentiable component distance does not fix its overlapping-
union interior limitation.

Check gradients away from ties, at terrain tops, wall faces, underside/ceiling
faces, edges and corners, and through actual FK/skinning. Verify gradients into
the fast policy's Root and pose heads. For MB, also check intended Root/Pose
groups with hard token forward values. `straight_through_codes` is a biased
soft-gradient estimator for final-pass logits; it does not differentiate
earlier argmax choices across all four Pose passes.

Do not reuse `JointGeometryObjective` wholesale: its existing methods replace
uncertified volume values with height gaps. Use the exact certified distance
path for this experiment. Invalid or uncertain body queries remain recorded
failures with their source identity.

## 5. Train Control, Continuation And Contact Together

Retain the pilot's causal boundary: first new velocity is measured from the
last **played** frame. The primary policy trains its integrated next-frame
Root, body rotations and preview. In MB comparisons, use Root loss on both
the intermediate forecast and final decoded world Root, and report their
disagreement; the decoder's reconstructed history endpoint is not the played
boundary.

Use these objective groups, with normalized physical units and weights fixed
before the run:

| Group | Supervision and role |
| --- | --- |
| Motion preservation | Genuine source FK, rotations, Root and boundary-inclusive joint velocities. Token CE applies only to MB comparisons on complete source windows compatible with their source-derived commands. |
| Immediate control | Signed progress, lateral error, independent facing, acceleration from idle and braking toward a settled stop. Evaluate the actual executed prefix. |
| Stationary stability | Long zero-intent rollout displacement, Root/pose velocity and valid stance slip. Preserve natural pose variation using real idle examples. |
| Support and swing | Source contact/stance masks, sole gap and tangential stance velocity; source-derived swing clearance and release timing. |
| Body geometry | Full 3D signed clearance at predicted soles/body surfaces, including the history-to-first-frame interval; assess floors, walls, ceilings and undersides. Full-skin tail checks run at scheduled training evaluations. |
| Continuation | Source-matched boundary dynamics and adjacent future agreement under unchanged intent, while retaining gait amplitude and acceleration. |

Select support from the admitted contact sidecar and certified source geometry,
not from predicted contact probability. Check which source labels map to heel,
toe and sole regions, keeping support-surface identities on multilevel scenes.
Use contact BCE as an auxiliary; do not let predicting
"no contact" disable stance losses. Constant low velocity is suitable for
stationary examples, not for all moving feet. Prefer source-relative dynamics
and support-specific velocity terms over a global jerk penalty that suppresses
the gait.

Training can use differentiable foot/body subsets plus scheduled exact
full-skin losses; final evaluation keeps all 18,056 vertices and eight swept
substeps, peak penetration <= 2 cm and zero unknown/uncertain body samples.
`terrain_skin_tail_loss.py::skin_tail_penetration` provides a training-only
distance/tail-reduction pattern, but its current heightfield-gap sign test must
be replaced by the certified full 3D inside/distance query. Its default four
sweep samples are distinct from final eight-substep evaluation. Source foot
support is assessed against its designated surface; full-body collision is
assessed against all solids.

Progress from source histories to generated histories and from short to longer
rollouts. For the fast policy, specify duration in real frames/seconds and
advance one source frame per call. Increase the supervised/truncated interval
from 8/16 to 32/64 frames where genuine labels exist. For MB, the corresponding
short comparison can use two/four calls, then eight/sixteen. Detach recurrent
memory and feedback at declared truncation boundaries without resetting their
values. Retain a source-imitation stream so task losses cannot turn the motion
into stationary poses or small trembling steps.

For K one-frame calls, pose targets need `4 + K` real source frames, or one more
for the final forward velocity label. Thus 16 seconds needs 484 pose frames or
485 with that derivative. The fast policy has no codec-label extension.

For K four-frame MB calls, pose targets need `4 + 4*K` real source frames. A final
forward velocity label needs one more. Complete shifted 33-frame codec labels
at every call need `33 + 4*(K-1)` frames. Thus a 16-second supervised rollout
needs 484 pose frames, 485 with that final derivative, or 509 with full shifted
codec targets. The present 33-frame reader cannot provide it.

If PFNN lacks that long an uninterrupted idle interval, train shorter real
idle examples and task-level long free rollouts with the retained motion prior;
report the longest source-supervised duration. Do not repeat a short sequence
to manufacture a 16-second source target. Arbitrary-command free rollouts have
task/contact/geometry objectives, not invented source pose targets.

## 6. Comparisons That Decide What To Keep

Use the complete same PFNN training pool and paired draws where applicable.
Publish unique-source coverage, exposures, executed frames, actor calls,
optimizer updates and aggregate GPU time separately. Architecture comparisons
should show both sample-matched and compute-accounted results.

| Arm | Purpose |
| --- | --- |
| F: Trained compact one-frame controller with our full 3D perception | Primary deployment proposal; test fresh per-frame control, long gait continuity and geometry use. |
| A: Immutable original, cadence-correct four-frame execution | Retained control and motion-quality baseline. |
| B: Terminal-conditioned MB architecture, new objectives | Separates the objective change from removal of endpoint semantics. |
| C: Explicit current intent in MB, same new objectives and scope | Bounded four-frame architecture comparison against B. |
| D: C with generated-history curriculum | Tests feedback training at matched data/budget. |
| E: D with bounded decoder adaptation, if reconstruction warrants it | Separates decoder capability from condition/control improvements. |

Arm letters retain the previous review's identifiers; F is now the primary
path. F begins on genuine PFNN source and generated-history sequences after
3D scene admission. It does not wait for B-E to succeed. A retains its original
heightfield-only perception as the historical deployed system; B-F share the
new full 3D scene/query contract. Report legacy local-height branches explicitly
so input and cost differences remain visible. On original PFNN scenes, add a
paired F comparison through equivalent terrain-only versus general-3D backends
to verify surface/normal/validity preservation. Additional 3D scenes require
their own geometry and source-compatibility admission.

These are a staged comparison plan, not simultaneous launch instructions.
For terrain use, compare aligned versus deliberately displaced observations
on a fixed validation diagnosis, and measure relevant clearance/Root changes.
Score correctness on valid paired scenes and complete routes. Keep all training
arms terrain-conditioned; the displaced-input probe is not a deployable arm.

Include long idle, idle-to-run, run-to-stop, reverse, both strafes, independent
facing, analog jitter and sharp direction changes. Cover full flat approaches,
rough-ground entry and complete ascent/descent, with matched seeds and raw
commands. Preserve failures and source-specific outcomes. A fixed endpoint is
selected with validation; test is opened after the protocol is frozen.

## 7. Real-Time Full 3D Performance And Selection

Measure on the deployment RTX 3080, batch one, including both full 3D query
stages, two shared-encoder evaluations, Root/pose computation, official
FK/history encoding and the publication copy. MB comparisons additionally
include all four Pose passes and codec.
A100 throughput governs training time; it does not determine the local
interaction budget.

Historical reactive v3 complete predictions had P50 around 25-28 ms and P95
around 42-46 ms. Only 64.34% of one-frame requests met 33.33 ms. These are
recorded results for the old weights/execution, not timing of the proposed
candidate. New MLPs do not address that established tail-latency problem.

The separate untrained v3 prototype records H1024 P99 around 10-12.3 ms across
four scenes for its old 512-query path. That is useful implementation evidence
for a small recurrent model, but it has a different encoder, query budget and
sampling schedule. Its old performance and zero training do not establish the
proposed policy's cost or locomotion quality. Source:
`scene_control_v3_performance_20261018.md`.

Keep the initial query budget explicit: 1,536 coarse points plus 512 refreshed
path points if unchanged body/actual-velocity results are reused within that
same step. Both encode full 3D values. If the implementation re-queries all
three regions after Root, charge 3,072 points instead. Test increasing solid/
triangle counts, overlapping objects, thin ceilings, multilevel passages,
scene bounds and high-speed path changes. Fixed point count alone does not
bound traversal cost or memory traffic.

Build static geometry BVHs and GPU buffers on scene load, with bounded scene
residency and manifest binding. Keep dynamic query points/transforms on the
GPU and avoid per-frame host nearest-neighbor work or map voxel rebuilding.
Capture the two-query/two-encoder chain with fixed tensor contracts. Measure
scene loading/switching separately from steady-state control. The actor does
not evaluate all 18,056 skin vertices in its hot path; independent quality
evaluation still does.

For the candidate, require the product's **P99 <= 20 ms** necessary computation
goal and at least 30 genuine motion/display frames per second; report the full
33.33 ms frame deadline's miss rate as well. Time from receipt of an input to
the first ready motion using that input, and then to its displayed frame.
Report input-visible P95 against the <= 100 ms goal. Use 10,000 timed calls per
fixed stress case, with disclosed warmup, all PFNN map families, locomotion
states and actual analog replay. Record per-stage CUDA events and complete
wall time from the same trials; stage percentiles do not add to an end-to-end
percentile. Include sharing/foreground conditions in the results.

Separate latest-input arrival, fixed adoption cadence, computation lead and
display ACKs. In four-frame mode, boundary waiting alone has a 126.7 ms P95
under uniformly phased commands, before additional lead/display delay. Faster
inference cannot remove that adoption bound. The four-frame experiment can
establish better gait/contact and a useful MB reference, but satisfying the
strict response target requires a shorter trained control step.

**Decision rule:** develop and evaluate F as the primary per-frame policy.
Keep MB four-frame results as quality/control references even if they improve.
If F misses a quality or timing gate, preserve the original viewer and diagnose
the specific geometry/model/feedback cost; do not silently revert its perception
to a heightfield or reduce its decision rate. A proposed one-frame MB challenger
needs its own rollout training, shifted real targets and treatment of latent
alignment, followed by the same full 3D latency and motion evaluation.

For F, use a small continuous autoregressive controller:

```text
four played poses / velocities + recurrent state + current movement/facing
  -> full 3D scene backend -> SAME coarse 3x8x8x8 features and shared encoder
  -> small Root head: next Root increment + anticipatory Root path
  -> SAME predicted-path resampling and shared terrain encoder
  -> small pose/contact head -> next-frame rotations / pose -> official FK
  -> played state and recurrent state advance by exactly one frame
```

Root and pose remain learned outputs. The model generates the next actual
frame, with a learned preview for anticipation; no external Root substitution,
foot locking or pose filter is added. Start with genuine PFNN supervision and
generated-history training. Use only validated MB/offline source continuations
as optional distillation targets, since the failed prefix candidate is a poor
teacher for braking and idle. Keep recurrent state across commands and detach
it only for training truncation; clear it at a real episode reset.

F retains the environment work and upgrades its geometry, so benchmark the
complete chain before choosing width. It removes the four Pose passes and
symmetric stride-four decoding from the online decision path. Jointly assess
full 3D perception, PFNN-supported behavior and real-time delivery as three
distinct outcomes; all must pass their applicable gates for deployment.

## 8. Literature Applied To These Decisions

The following reading is inherited from `terrain_tokenization_literature_20261010.md`,
`whole_body_scene_methods_datasets_review_20261018.md`,
`joint_sdf_results_20261012.md` and `kinematic_3d_reactive_plan_20261021.md`.
Fresh verification belongs to the parent's literature pass.

| Primary source | Design lesson used here | Consequence for this experiment |
| --- | --- | --- |
| [PFNN, SIGGRAPH 2017](https://www.research.ed.ac.uk/files/35467734/phasefunction.pdf), [author project](https://theorangeduck.com/page/phase-functioned-neural-networks-character-control) | Phase-conditioned per-frame regression uses trajectory/terrain context. The reviewed demo also enables foot IK and contact locking. | Compare control, learned contact and final contact separately; record both PFNN IK settings. Retained temporal state motivates F. |
| [MotionBricks](https://arxiv.org/html/2604.24833v1) | Short primitives, learned Root/timing and masked token prediction; prior source review notes repeated early-phase replanning and Smart Locomotion's task construction. | Four-frame MB is a sensible retained baseline. Terminal velocity alone is a reduced task interface; use explicit intent and measure prefix gait. |
| [PARC](https://arxiv.org/html/2505.04002v1), sections 5-6 | Scene features and motion are fused separately; history dominance and penetration receive explicit treatment. | Preserve our scene pathway, add decoded-body geometry losses and test terrain sensitivity. Copying its random-terrain branch would require a separately defined objective without contradictory source pose labels. |
| [DartControl](https://arxiv.org/html/2410.05260v3), section 3 and appendices | Generated-history exposure addresses rollout distribution shift; scene optimization and online generation have different costs. | Use a feedback curriculum and real closed-loop endpoints. Do not put iterative scene optimization on the frame path. |
| [Neural State Machine](https://www.ipab.inf.ed.ac.uk/cgvu/nsm.pdf) | Goal-driven character interaction combines temporal control with scene context. | Investigate retained control state for F rather than restarting a fresh latent window each frame. |
| [LINGO](https://arxiv.org/html/2410.03187v1), section 3.3 | Current-body and lookahead scene volumes have separate roles. | Preserve our body and actual-motion observations when replacing the intent corridor with the predicted Root corridor. |
| [MotionStreamer](https://openaccess.thecvf.com/content/ICCV2025/papers/Xiao_MotionStreamer_Streaming_Motion_Generation_via_Diffusion-based_Autoregressive_Model_in_Causal_ICCV_2025_paper.pdf) | Causal compression and incremental generation are relevant to streaming continuity. | Treat a causal-codec change as a separate representation retraining study; it is larger than F or the intent repair. |

Prioritize the fresh reading of PFNN's training/control construction, MB's
replanning and Smart Locomotion details, and causal streaming representations.
For each, extract the executed step, persistent state, command meaning,
training-history distribution and complete deployment cost. Those mechanisms
bear directly on our failure; adding a new model family without resolving
them would make the next comparison difficult to interpret.

## 9. Implementation Boundary For The Parent

Paths below are relative to this repository root and currently exist.

| File / symbol | Planned work or reuse |
| --- | --- |
| `motionbricks/scene_control_v3/model.py::CausalSceneActor` | Reuse causal GRU/head structure for primary F; replace its scene encoder with our shared encoder and split coarse-Root/refined-pose computation while advancing memory once. |
| `motionbricks/scene_control_v3/native.py` / `pipeline.py::CapacityPipeline` | Reuse official next-frame SOMA conversion and ACK ownership; replace probe sampling, synthetic contact inputs and prior-Root-only query schedule. |
| `motionbricks/data/scene_geometry_field.py` | Extend manifest/field/mesh identity to admitted PFNN terrain plus full 3D solids, known free space and certified union semantics. |
| `motionbricks/scene_control_v3/geometry.py` | Review BVH/mesh execution for the full 3D backend; adapt query count/features and distinguish actor no-backward queries from training loss gradients. |
| `motionbricks/data/unified_scene_field.py` | Preserve body/intent/actual-velocity features and masks; accept the general 3D scene field instead of constructing a heightfield-only field. |
| `motionbricks/data/unified_terrain.py::SharedTerrainEncoder` | Reuse the encoder/weights in F's two perception stages and the MB comparisons. |
| `motionbricks/data/root_path_terrain.py` / `reactive_prefix_training.py::TrainingRootPathObservation` | Preserve train/infer query ordering; extend corridor centers from XZ to predicted XYZ and resolve the MB auxiliary support surface explicitly. |
| `motionbricks/data/native_mb_online_conditions.py` | MB comparison only: reuse factual-history masks and define the explicit-intent protocol. |
| `motionbricks/data/native_mb_online_live.py::predict_planar_terrain` | MB comparison only: new condition dispatch, explicit coarse-query intent and full 3D scene binding. |
| `motionbricks/data/unified_terrain.py::UnifiedTerrainController` | MB comparison only: narrow intent residuals, no slot-7 intent extraction or forced heightfield factory for the new protocol. |
| `motionbricks/motion_backbone/inference/motion_inference.py::_predict_root_trajectories` | MB comparison only: learned Root conversion without commanded terminal overwrite. |
| `motionbricks/motion_backbone/neural_modules/root_backbone.py` | MB comparison only: verify stage-2 suffix and exact trainable layers. |
| `motionbricks/motion_backbone/neural_modules/pose_backbone.py` | MB comparison only: verify motion-token suffix and per-pass injection. |
| `motionbricks/data/reactive_training_data.py::ReactiveSourceBank` | Reuse provenance/contact checks; expose all PFNN train rows and genuine long intervals through a dedicated reader. |
| `motionbricks/data/reactive_prefix_training.py::prefix_loss` | Reuse actual-history boundary semantics; add separately reported control/contact/geometry objectives. |
| `motionbricks/data/joint_sdf_observation.py` / `terrain_skin_tail_loss.py` | Reuse training-only surface-loss patterns; replace heightfield-only inside/support tests with certified full 3D geometry and designated support surfaces. |
| `motionbricks/data/reactive_path_execution.py` / `root_path_split_execution.py` | MB comparison: rebuild captured inputs for the new intent and 3D backend. Primary F requires its own one-frame two-query capture. |
| `scripts/train_reactive_prefix_pilot.py` | Reuse reporting patterns; primary F needs one-frame recurrent rollout training on all admitted PFNN train data, with geometry-aware objectives. |
| `scripts/evaluate_reactive_prefix_control.py` | Reuse raw rollout/motion metrics, replace the four-frame/64-update/two-arm assumptions and heightfield-only scene admission for F; retain full-skin gates against the common 3D scene. |

Before training, the parent should freeze the all-PFNN source receipt, full 3D
scene contract, exact scope/losses, stage endpoints, wall-time/GPU budget and
validation protocol. Admit the backend with matching rendering, independent
geometry checks and local two-query cost. Train F on genuine PFNN sequences,
then expand generated-history duration at fixed endpoints. Run bounded MB
comparisons without making them a prerequisite for F. Keep every failed
endpoint and the original user-facing model.

Report three separate results: (1) full 3D geometry/query correctness,
(2) learned motion/control on available PFNN behavior, and (3) measured local
one-frame response/frame cost. Crawling/climbing requires a later paired-data
milestone. This planning revision launches no training or runtime change.
