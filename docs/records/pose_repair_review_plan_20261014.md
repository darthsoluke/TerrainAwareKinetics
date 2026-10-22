# Pose Repair: Review And Plan

Date: 2026-10-14. Status: REVIEW AND PLAN ONLY. No new training, weight edits,
runtime changes or deployment are performed by this plan. The user's current
request is to review and plan a focused Pose repair. Keep28941and28936; do not
restart28940. Existing unsuccessful experiments and snapshots remain immutable.

## Decision

Pose needs an explicit learning experiment, rather than another Root-only fit.
Retain the MB Root -> masked-token Pose -> conditional codec chain. Separate
three questions: can the fixed codec express the required motion; can Pose
select it under causal conditions; can this selection survive closed-loop
approach and traversal? A success at one level is not a success at all three.

Do not promise that Pose alone can solve the task. Current Root anticipation
is wrong, and source Root plus source tokens still leave reconstruction error.
The first controlled learning stage freezes Root to isolate Pose; complete
delivery must later pass the whole chain with normally predicted Root.

## Review Findings

### P1: Recent candidates did not train Pose

`scripts/pilot_unified_goal_alignment.py:29` freezes the controller and opens
only`adaptation.root_stage1`and`adaptation.root_stage2`. Its Root objective at
line79returns only`native_root`. Shared encoder, native Pose, Pose adapter,
codec and decoder adapters remain frozen. Thus the current blue/orange pair
tests Root fitting and source coverage, not repair of pose token selection.
The initial200-update unified pilot trained new adapters, but the native Pose
backbone remained frozen there too (`data/unified_terrain.py:102`).

Evidence: orange live stair frames100..195span0.979mof Root height, but mean
knee flexion27.49deg versus28.38deg on its preceding flat segment.71.35%of
ankle-center/time samples are below the queried tread. Existing full-skin
peak41.57cm independently fails the traversal gate. These are not labels
classifying every pose as walking; they corroborate failed stair articulation.

### P1: Token-selection errors are especially large off the training routes

The read-only`unified_stair_gait_20261014_v3` factorial audit gives these
Root-relative leg-joint errors for the orange candidate:

| Window split | Windows | Normal actor | Source tokens only | Source Root + tokens |
| --- | ---: | ---: | ---: | ---: |
| Two training routes |4|4.507cm|3.803cm|3.672cm|
| Four review routes |8|9.891cm|3.820cm|3.360cm|

Across all12windows, replacing Root HEIGHT only gives8.096->7.918cm; replacing
the complete Root gives5.129cm. Pose-token errors matter, particularly on the
review routes, but these are nonlinear interventions, not additive causal
percentages. They do not justify naming particular heads as left/right legs.
Source-token identity accuracy is66.4%on train versus32.4%on review; equivalent
decoded motions can have different tokens, so accuracy is secondary evidence.

### P1: Source-history imitation is not recovery supervision

`data/unified_terrain.py:242` deliberately rejects off-source history.
`data/terrain_decoder_pilot.py:47` preserves real source frames and their clock.
The current capacity fit uses one up route, one down route and two BONES clips;
73overlapping windows per extended stair route are not73independent examples.
Good predictions from already stair-like source history do not establish entry
from self-generated flat walking on the live map. Source/live differences in
history, terrain and commanded speed confound attribution to any one factor.

Do not remove the guard to mix generated histories with unchanged fixed-time
source futures. Earlier archived feedback audits show infeasible55.6cmnext-frame
targets and targets69.8cmbehind the generated Root. Nearest-frame matching,
shifting the source to follow the Root and stitched pose prefixes were not
validated fixes. Codec tokens have nonlocal temporal support; the first token
cannot be discarded as pure history.

### P1: SDF conditioning is not a nonpenetration objective

`data/unified_terrain.py:242` computes native reconstruction, masked-token CE
and native Root losses. It does not add the exact terrain-SDF penetration or
source-support objectives. The `JointGeometryObjective` in the current pilot
is used for source screening and review. Existing older Joint-SDF campaigns
did use exact geometry losses; this finding is specific to the new unified
pilot and must not be generalized backward to those runs.

Keep exact geometry as a hard evaluation requirement. Adding training geometry
terms is a separate controlled change requiring units, source-contact validity
and parameter-gradient checks. Do not hide new weight tuning in the Pose fit.

### P2: Teacher masking and runtime masking need separate measurements

Actual bundle configuration:16Pose transformer layers, width1024,16attention
heads;8pose tokens with8codebook heads in the current32-frame primitive.
`masked_token_ratio=0.8`; focused-token count follows the native cosine schedule.
Consequently the random CE teacher never supplies the same fully masked64-entry
input as runtime's first pass. It can also receive correct/noised source token
context. `inference/motion_inference.py:367` starts generation fully masked,
then reselects ALL token IDs from each pass's logits. There is no evidence here
of stale earlier tokens differentiated through unrelated final logits.

This difference is part of the official method, not automatically a code bug.
Measure teacher CE, first-pass all-mask CE, all4actual inference passes and
decoded motion separately. Only if the gap is material should an explicit
full-mask training arm be added, retaining the original cosine curriculum.

### P2: Height reaches Pose, but gradient and sharing contracts matter

The main Pose Root projection uses global`root_without_hip_height`(4channels
per frame). Height arrives separately through`root_height_condition`and the
learned height projection at`pose_backbone.py:169`. Actual base checkpoint
height gate=-0.5083273 and terrain-attention gate=-0.0965556: neither is zero.
A negative residual gate is not by itself an error.

Height conditions are detached at`pose_backbone.py:178`. Pose projection and
transformer parameters can still learn from them; this only blocks a gradient
back into Root through that height path. Keep the detach for a staged Pose fit.
Do not claim joint Root-height credit without explicitly auditing that path.

The new SDF encoder is shared by Root, Pose and decoder. Training it changes
Root outputs even with Root weights frozen. A first Pose-only comparison must
freeze shared encoding, Root modules and decoder; later shared-encoder changes
require their own complete-chain comparison.

## MB Reference And Explicit Extensions

Reviewed the actual paper HTML and repository baseline (`git HEAD`, upstream
sync commit`b307976`), not just our modified training wrapper.

- [MB Section5.2](https://arxiv.org/html/2604.24833v1#S5.SS2): Pose predicts
  multi-head token distributions, conditioned on Root and optional pose
  keyframes. Root values come from the dataset during training and from Root
  predictions at inference. The paper retains cosine masking as a curriculum
  even though a single inference pass often works.
- [MB Section4](https://arxiv.org/html/2604.24833v1#S4): the conditional codec
  combines pose tokens, Root conditions and available keyframes. Root refinement
  is part of generation; copying an external Root over the output is not our fix.
- Repository baseline`models/pose_model.py`uses native mask/noise generation
  and focused-token CE. The unified adapter fit additionally uses decoded-motion
  reconstruction through a straight-through estimator. That extra objective
  is not the paper's standalone Pose objective and is not an exact gradient
  through discrete argmax.
- Our4history/terminal-VXZ protocol, predicted-Root-only training conditions,
  unified SDF memory and fixed4-pass deployment are explicit online extensions.
  They must remain consistent between our training and deployment. Future source
  heights/poses/tokens may be targets or OFFLINE oracle inputs, never normal
  actor conditions. No train-time oracle branch will be counted as actor quality.

## Proposed Sequence

### 0. Freeze The Experiment Contract

Use the unchanged28941candidate as the common initial model:
`ade10ff143d855073b0760ca3f6a12cc8b1f59d7627eb883f9e981579edc5979`, with bundle
`0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d`.
Do not initialize one arm from another failed trained arm.

Lock8tokens,4inference passes,4played-history frames,24published frames per
neural generation in8-frame chunks,30Hz. Actor conditions remain actual history,
command, measured geometry and model-predicted Root only. Re-encode feedback
with the same official FK helper. Retain the old official85-frame failure,
current live stairs and all existing review-only source families as regressions.

Add a schema that contains explicitly changed native Pose weights if an arm
trains them. The current adaptation-only loader must reject that schema until
updated and parity-tested; never silently drop native weights on local load.

### 1. Read-Only Preconditions Before Any Optimizer

1. Codec ceiling: source encode -> quantize -> decode using the native source
   boundary and source Root, with the existing decoder SDF adapter enabled and
   disabled ONLY offline. Measure Root-relative legs, rotations, all-skin SDF,
   source support and boundary continuity. Identify whether its3.46cmresidual
   comes from codec limits, the learned decoder adapter, or conditioning. If
   source replay itself violates the final gate, record that independent blocker;
   do not demand Pose optimization magically solve an inadequate codec.
2. Pose trace: capture all4passes, raw conditions, component masks, height values,
   actual decoded IDs, per-head CE and logit margins. Compare native teacher
   corruption against fully masked inference, split by approach/entry/mid-stair/
   exit and by up/down. Analyze head effects empirically, not anatomy labels.
3. Gradient audit: finite, nonzero gradients reach precisely the selected Pose
   modules on stair and BONES samples. Native Root/shared encoder/codec hashes
   and Root predictions on identical inputs stay unchanged. Future target
   mutations leave normal actor output unchanged. Source-token encoding stays
   deterministic in float32 and uses the real complete33-frame source window.
4. Condition feasibility: stratify Pose failures by predicted Root XZ/Y/timing
   error against real sources. No target retiming, source translation or hidden
   source-Root injection to make a teacher score look better.

These measurements determine whether the first Pose fit is meaningful. Do not
launch a chain of repeated fits when a precondition fails.

### 2. Prepare Diverse Real Entry Data

Target a small pilot set of12independent existing training routes (6up/6down)
and6independent BONES locomotion clips covering idle, walk, run, start and turn.
These are requested counts for a dataset audit, not claims that eligible data
have already been collected. Use only records permitted by their manifest split;
source families, terrain variants and overlapping windows must not leak into
validation. If insufficient eligible routes exist, report the shortfall and
prepare new real PARC sequences through the already authorized private pipeline,
without promoting any existing review route into training.

Every stair route must contain genuine flat approach, first step, sustained
alternating ascent/descent and real exit frames. Reuse exact prism/full18056
skin-vertex/8substep screening; preserve failed records and exclusion reasons.
Sample all gait phases from complete33-frame windows, retaining original
timestamps. Balance actual primitive exposure50%PARC/50%BONES in each arm;
keep flat samples on their authored plane in the same unified environment path.

Source interval31->32provides terminal VX/VZ and frame31heading. Those are
hindsight source goals, not logged keyboard inputs. Do not label every source
window with one constant approach speed. Arbitrary constant-speed WASD remains
a separate closed-loop test and a known distribution-coverage question.

### 3. First Matched Pose Capacity Pair

This is the recommended first TRAINING stage after preconditions. Proposed
ceiling per arm:300actual updates,batch4,1200primitive exposures,Adam5e-5,
same initialization, source schedule, command labels and precomputed corruption
draws. It is a capacity diagnosis, not a convergence promise or full campaign.

| Arm | Trainable | Frozen |
| --- | --- | --- |
| P-A | New`adaptation.pose`SDF residual only | Native Root/Pose, shared SDF encoder, codec and all other adapters |
| P-B | Same residual, last2of16native Pose transformer layers, Pose output head, existing Pose Root/height input projections | Native Root, shared SDF encoder, tokenizer encoder/codebook/decoder, earlier14Pose layers and remaining parameters |

P-Bspecifically includes`_proj_local_root_values`, `_terrain_root_height_projection`
and its gate. Its old Pose terrain encoder/cross-attention stays frozen initially;
this isolates added Pose capacity without changing every environmental path.
Exact parameter names and counts are recorded before launch.

Use the native focused-token CE and original cosine mask/noise curriculum as
the first objective. Root comes from the same frozen causal actor in both arms.
Native decoded-motion/SDF metrics are measured, not silently added to this CE
comparison. Frozen-codec decoding lets the review test whether better selected
tokens actually produce better legs. Do not equate a drop in CE with traversal.

Check at8updates for execution/gradient contracts, then complete the fixed300
unless a technical/numerical failure occurs. Evaluate both completed arms on
the locked suite. More trainable parameters mean different compute; equal
updates/exposures do not imply equal FLOPs. No wall-clock claims before timing.

### 4. Conditional Follow-Ups, Not Automatic Extra Runs

- Teacher good / all-mask poor: matched follow-up of the selected scope with
  the native curriculum versus25%fully masked examples, replacing that fraction
  of the same source exposures rather than adding samples. Keep75%native
  curriculum,4-pass deployment and all else fixed. The25%is a proposed experiment
  factor, not a validated optimum. Balance the mask variant across terrain/flat
  phases. Do not add this factor to P-A/P-Bat the same time.
- Source-history good / closed-loop bad: inspect the earliest divergent played
  prefix and conditions. Use generated states as diagnostics. Before supervised
  feedback learning, obtain a verified feasible SOMA continuation for that state;
  existing native PARC-to-SOMA recovery bridge is not certified. Do not pair it
  with a fixed-time future or choose a nearest reference that skips the route.
- Pose tokens better / support still bad: compare fixed-codec ceiling with
  actual actor output. Only then separately test decoded-motion supervision and
  exact terrain/support objectives through the frozen codec. Check dimensional
  scales, parameter gradients, and contact labels before setting coefficients.
  If codec adaptation is required, retain its native self-reconstruction branch
  and protect the vocabulary; make it a separate bounded comparison.
- Root remains the limiting factor: test the selected Pose checkpoint against
  frozen original and improved learned Root candidates on identical factual
  inputs, then consider a explicitly scoped Root/Pose co-adaptation. Audit the
  detached height path before claiming joint gradients; no runtime substitution.
- Both P-A/P-Bfail: use the traces to choose data or representation work; do not
  rerun the same budget, open every layer, or multiply SDF loss weights blindly.

No follow-up is launched merely because it is listed here. It needs the named
evidence and a concrete matched contract; no full training restart is implied.

### 5. Acceptance And Local WASD

Report3levels separately: source-history imitation, complete generated-history
traversal, and live control. Root rise, mean knee angle and token accuracy are
diagnostics; none alone is an acceptance gate.

- Full route: naturally alternating advancement across successive treads,
  correct support/clearance, no repeated stationary stepping, retreat, bypass,
  floating, fabricated tail or skipped failed clip. Slower traversal is allowed;
  compare distance/events and actual completion, not same-time speed matching.
- Geometry: keep existing full-skin peak<=2cm/P99/known-sample requirements for
  this unified experiment, with all18056vertices and8substeps. Historical user
  approval of another orange3.68cmcase does not silently change this suite.
- Coverage: both up/down, full real prefixes, several starting distances and
  headings,0.8/1.6m/srequests, held-out stair families, original official failure
  and the current authored live map. Keep validation selection separate from
  the repeatedly viewed regression suite; it is not a fresh blind test.
- Flat: preserve the17current start/run/turn/release cases, extend reporting to
  stopping drift, foot alternation, stance slip and block-boundary jumps. Current
  `review_unified_flat.stats`only reports speed/range/displacement and finiteness;
  that is insufficient to certify natural gait or stopping. Compare to unchanged
  28941and the retained user-approved flat checkpoint, without restarting28940.
- Stability: check every8-frame publication and24-frame regeneration boundary,
  plus command changes, with actual frames. Do not fix seams by smoothing.
- Local: only after offline quality, a separate port with explicit full checkpoint
  loading, eager parity, then verified-CUDA parity, official SOMA skin and actual
  WASD30fps/input latency. Confirm model/scene/profile hashes. Keep28941untouched
  while reviewing. Never advertise source/oracle playback as live generation.

## Deliverables And Completion Boundary

Before any training: immutable contract, manifest/splits, source geometry report,
codec/Pose traces, module allowlist and preflight evidence. After each pair:
both checkpoints, failures, actual frame counts, per-case metrics, and official
source/generated/before/after SOMA comparisons. Large artifacts use only the
existing private Drive/NAS path and verified receipts, never public SSH.

Training success means a reproducible complete controller passes all3levels;
it does not mean a fixed update count completed. Current status remains
unqualified and the task is not solved. No IK, contact locking, output Root
replacement, collision projection, output smoothing or future-truth actor inputs.

## Evidence

- `docs/unified_terrain_20261014.md`: completed failures and runtime contracts.
- `out/parc_rugged32_20261013_v2/unified_stair_gait_20261014_v3-report.json`:
  paired live/source Root/height/token interventions and replay checks.
- `out/parc_rugged32_20261013_v2/unified_entry_learning_20261014_v3-report.json`:
  causal SDF sensitivity and actual-history replay.
- `docs/joint_sdf_results_20261012.md:2751`and`:2894`: archived codec receptive
  field and infeasible feedback-target findings; different older controllers,
  not direct quality measurements of the current unified checkpoint.
- [MotionBricks paper](https://arxiv.org/html/2604.24833v1), especially4,5.2,6.1
  and the runtime integration appendix. Checked2026-10-14.
