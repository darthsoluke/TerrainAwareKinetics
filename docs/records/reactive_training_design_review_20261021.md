# Reactive Kinematic Training Design Review

2026-10-21. Design and source review for the parent agent. The user authorizes
training changes and A100 use to repair the pure kinematic controller. This
review writes only this document; implementation, remote inspection and training
remain with the parent. No GPU, SSH, browser, model forward or training was used.

## Decision

Train the existing eight-token generator to continue **four actually executed
frames at a time**, conditioned on an immediate movement/facing request. Start
with the existing motion vocabulary, decoder, terrain encoders, normalization
and source splits frozen. Update a small, explicit Root/Pose scope, then compare
source-history training with generated-history training at the same budget.

The first experiment tests three linked claims: the near prefix accelerates
instead of repeatedly deferring acceleration; generated four-frame histories
retain the gait; stop, sideways motion and facing are separate learned controls.
It keeps the current predicted-Root-path terrain observation. It adds no runtime
Root substitution, pose filtering, foot locking or synthetic playback frames.

Use a new experiment schema, proposed `motionbricks.reactive_training.v1`, and
new output directories. Keep the original final model as an immutable control.
Do not restart the October 16 codec repair pilot or any completed formal stage.

## 1. Audit Findings That Change The Training Design

| Finding | Source evidence | Required change |
| --- | --- | --- |
| The command describes the end of the window. | `unified_training_goals.py:11-26` takes interval31->32 and heading31; `motion_inference.py:299-305` applies terminal velocity. | Give current intent its own learned input; supervise the executed prefix. |
| Training history and Pose's Root teacher are source motion. | `unified_full_training.py:28-47,177-192`. | Run the real predicted Root/Pose/decoder path on generated histories. |
| Existing full-stage losses assume complete source windows. | `unified_full_training.py:83` removes masks; `root_model.py:238-275` reduces over all supplied frames. | Implement explicit prefix/availability reductions; never feed padded labels to these losses. |
| Existing unified losses reject generated history. | `unified_terrain.py:251-259` requires equality with the source prefix. | Add a separate training objective; retain the existing source-only contract. |
| The live path observer skips training. | `root_path_terrain.py:41-50` returns when `context['training']` is true. | Explicitly query predicted-path geometry during training before Pose/decoder. |
| Cached observations were built for a source history and a terminal command. | `cache_full_terrain_primitives.py:48-62,159-188`. | Re-query at the generated history with the new command and predicted Root path. |
| Four history poses are supplied, but predicted continuation is unconstrained across calls. | `reactive_pose_context_audit_20261021.md`, findings2-4. | Train repeated executed prefixes, including boundary position/velocity/rotation. |
| Boundary-velocity supervision already exists. | `motion_vqvae.py:327-345` uses `executed_history_positions`. | Extend its causal boundary semantics; do not describe the old loss as absent. |
| Analog inputs previously bypassed cadence. | `reactive_gait_repair_20261021.md`:517/763 adopted plans lasted one frame. | Keep the cadence correction in every training/evaluation protocol. |

The later gait-repair report supersedes the short audit's suggestion to deploy
anchored8: the long analog replay diverged in height for anchored and longer
execution candidates. Those failed candidates are not initialization targets.
Recorded v4 still drifts at idle despite the scheduling repair.

The architecture is unchanged by this diagnosis: there are eight temporal tokens,
each spanning four frames, and four Pose passes. Temporal token count and the
number of independent quantizer heads are distinct dimensions.

## 2. Data: Reuse Source Caches, Reconnect Their Geometry

The local `acceleration_reports_v1/corpus_audit_v3-report.json` binds these
existing remote caches under the recorded `pfnn_mixed_20261015` run:

| Cache | Complete receipt SHA256 | Record index SHA256 |
| --- | --- | --- |
| `cache_mixed_v1` | `1744823c35c971a9c1dafdfae048ba7129aa6ca198f062689b6e9f1f8bc9bf5f` | `13c17f82821fb13beece2eebee40acf4e7d8593740c43f5ebf47c7d2dbd58f3b` |
| `cache_pfnn_world_v2` | `43e5313467be1fa386a43878f13a13764c441c4e67a886d198d45aee518a4053` | `10e7f7c1b1bba275e93b94c09ddae6efb6122c0e76a498e65993edb4929b66e8` |

The audit reports397,308 augmented primitives across all splits. Train counts:

| Domain | Train primitives | Recording families | Terminal idle / start / stop | Sideways-or-backward terminal |
| --- | ---: | ---: | ---: | ---: |
| PFNN | 48,232 | 28 | 1,744 /808 /520 | 6,256 |
| PARC | 88,344 | 259 | 32 /152 /112 | 1,796 |
| BONES | 231,980 | 155 | 67,816 /8,612 /8,904 | 71,116 |

These existing categories use terminal speed and prefix speed. Recompute a
small **training-only control index** for the near-prefix definitions below;
the terminal sideways/backward column does not establish left/right/reverse
coverage separately. Quarter rotations rotate both body and travel, preserving
their relative direction; they cannot turn forward walking into strafing.

`PrimitiveDataset` returns33-frame `world`, cached observations, domain and
concatenated `sample_id`. Its `records.jsonl` also holds `source_id`,
`source_group`, split, source hash, start and quarter, but `__getitem__` does not
return them. The new reader needs those immutable records and a verified join
back to the original scene manifest. It must obtain the exact geometry, contact
sidecar and scene transform used for that primitive. PFNN rows identify separate
terrain pairings even when the capture is the same. Join by the full pairing,
not by recording family or motion filename alone.

The old caches do not contain the complete scene field. Reusing their sampled
points after the actor moves would preserve the old observation, not terrain
awareness. Load the existing source scene once, apply the recorded quarter to
motion **and scene**, and re-query it. Check this reconstruction against the
cached source window/observation before allowing generated feedback. Missing or
ambiguous joins stop that admission; do not substitute a plane.

`TerrainLearningDataset.full_reference_item` and
`cache_full_terrain_primitives.sources` already expose complete unchanged source
motion plus scene. They are useful for optional longer windows. Preserve the
PFNN world_v2 contact interval and every original group split. Do not rebuild
retargeting, terrain fitting or the completed cache, and never consume the old
`dataset_full_v1`.

Existing real-time recordings and `reactive_*` forecasts are diagnostic traces.
They are not correct motion labels. New training-time rollouts may supply
**input histories** only, starting from train-split sources. Labels remain
unchanged source motion, source contacts and source scene geometry. Exclude
weak wall/stop pairings, free-standing synthetic pose sequences, test failures
and generated successful motions as imitation targets from this first pilot.

Preserve the original50/25/25 PFNN/PARC/BONES domain quota. Within each domain,
draw half uniformly by the existing deterministic permutation and half from
the available near-prefix bins: idle, start, stop, left strafe, right strafe,
reverse and other locomotion. Balance nonempty bins and recording families;
record the resulting probabilities, repeats and effective source counts.
Empty bins remain visible as missing coverage. Do not meet them by changing
the facing label, stitching clips or importing validation examples.

## 3. Correct Multiple-Rollout Supervision From33 Frames

Use zero-based source frames `X[0:33]`, output `Y_j[0:32]`, and shift
`s_j=4*j`. History at the initial call is the official FK re-encoding of
`X[0:4]`. Source-history control calls use that same re-encoding on their
source prefix. The actor predicts32frames, plays only `Y_j[4:8]`, and the
next input is exactly:

```text
H_(j+1) = stop_gradient(reencode_played_history(Y_j[4:8]))
s_(j+1) = s_j + 4
```

The official FK re-encoding makes Root XYZ and displayed rotations authoritative.
Do not reintroduce redundant decoded joint positions as factual observations.
Source time advances by4 independently of the actor's speed or position.
This re-encoding does not alter the source motion used as supervision.

| Call | Source-time history | Executed output | Source labels for that execution | Available source positions in the whole output |
| --- | --- | --- | --- | --- |
|0|0..3|4..7|4..7|local0..31, plus endpoint32|
|1|4..7|4..7|8..11|local0..28|
|2|8..11|4..7|12..15|local0..24|

The first bounded comparison uses `K=3` calls per sequence. All three have
complete executed-prefix labels in the existing33-frame cache. The actor's
unlabelled suffix is still generated normally. It is neither fed source future
poses nor assigned fake targets.

For a source position loss, validity is `0 <= n < 32` and `s_j+n <= 32`.
For a **forward** local derivative at n, validity additionally requires
`s_j+n+1 <= 32`. Future motion losses start at n=4. Intersect these masks with
the objective's chosen prefix. An empty mask produces a recorded absent term,
not a zero-filled ground truth average.

For the four executed velocity intervals, use predecessor `H_j[3]` followed
by `Y_j[4:8]`:3->4,4->5,5->6,6->7. Their source labels use
`X[s_j+3:s_j+8]`. The3->4 interval must not use reconstructed `Y_j[3]`.
If also supervising output forward-local fields through index7, the source
endpoint `X[s_j+8]` is required and available for all initial pilot calls.

With executed-four supervision and its derivative endpoint,33frames permit
up to **seven calls**, s=0,4,...,24. Call7 at s=28 lacks a complete next four
targets. Any deeper curriculum must explicitly obtain longer real source data.

### Token labels are a separate availability problem

The symmetric codec encoder has a wider receptive field than one four-frame
block. Do not shift old token IDs, encode a four-frame fragment, or pad an
incomplete shifted source window and treat its tokens as source labels.

For the cache-only pilot, token CE is applied **only at call0**, using the full
unchanged source32-frame encoding prepared from its33 real frames. Later calls
receive physical Root/FK/rotation/contact losses on their labelled four-frame
prefix, through the frozen decoder and the hard-forward gradient estimator.
They receive no new CE label from the incomplete suffix. Both comparison arms
follow this identical CE rule.

If full-window token CE is wanted at every call, obtain
`33 + 4*(K-1)` consecutive real frames in the same source/scene/contact interval:
41frames for K=3;57for K=7. Verify each shifted33-frame window against original
data admission. A neighbouring cached row is not proof of scene continuity.
Encode every complete shifted window under the fixed encoder/normalization;
cache keys include source/pairing hash, absolute start, quarter, encoder hash
and normalization hash. Do not reuse a primitive's one-token-shifted labels.

No later source truth is transported with prediction error. Coordinate
recentering is allowed only as one common invertible chart for history,
source labels and geometry; restoring world coordinates must give unchanged
source positions. No source-height or source-pose trajectory enters actor input.

## 4. New Intent Contract And The Minimum Trainable Scope

### Immediate intent is a learned condition, not an endpoint overwrite

Proposed protocol `reactive-immediate-v1` provides world VX/VZ and desired
body heading independently. It replaces the terminal interpretation for this
new checkpoint; it does not reinterpret the original model's input schema.

Add two small zero-output-initialized residual MLPs, proposed names
`adaptation.intent_root` and `adaptation.intent_pose`. For temporal token i,
their seven inputs are:

```text
[vx_world/3, vz_world/3, cos(desired_yaw), sin(desired_yaw),
 4/30, 28/30, (4*i + 1.5 - 3)/30]
```

The constant3m/s is an input scale, not a speed clamp. All directions and
headings live in the same scene chart as Root. The history remains the existing
four poses/Root states; movement direction never overwrites body facing.
The MLP is Linear(7,128), GELU, Linear(128,model_width), with the last weight
and bias initialized to zero. Inject Root intent into the temporal-token portion
of the stage2 transformer input, after the scene residual; inject Pose intent
into its token input before the Pose transformer. Do not add transformer layers,
extra Pose passes or a second Root prediction.

For this protocol, all future hard Root/pose condition masks are false. Only
the four factual Root/pose frames and three known velocity intervals are true.
The desired velocity/heading lives in the new MLP input, not sparse slot7.
There is no `apply_terminal_velocity` substitution, desired-Root projection or
debug-ground-truth override in this execution. Teach braking from data rather
than imposing zero velocity on the next sample.

### Explicit pilot scope

| Trainable group | Exact proposed whitelist under `controller` | Reason |
| --- | --- | --- |
| New command inputs | `adaptation.intent_root.*`, `adaptation.intent_pose.*` | Learn immediate intent without changing source statistics or vocabulary. |
| Root continuation | Last2 layers of `root._root_token_transformer_model`, `root._conv_output.*`, `adaptation.root_stage1.*`, `adaptation.root_stage2.*` | Refit near-term Root/height/braking from real or generated history. |
| Pose continuation | `pose._transformer_model.layers.14.*`, `.15.*`; `pose._proj_pose_output_logit.*`; `pose._proj_local_root_values.*`; `pose._proj_local_pose.*`; `pose._terrain_root_height_projection.*`; `pose._terrain_root_height_gate`; `adaptation.pose.*` | Use changed Root/history and retain leg motion across short executions. |

The actual loaded Root stage2 layer count must be recorded and the two concrete
layer indices written into the run contract. Require at least two; the parent
must revise the plan if the loaded architecture differs. The Pose scope assumes
the verified16-layer parent. Expand the whitelist to exact parameter names,
shapes and total count before any update; broad `requires_grad_(True)` on the
whole Root, Pose or codec is outside this pilot.

Keep actor forward modules in eval mode while enabling gradients for the
whitelist. This disables dropout and aligns the hard forward with deployment;
trainability is controlled by gradients, not by activating training-mode noise.

Freeze `adaptation.encoder`, both native terrain encoders/cross-attention
modules, all other Root/Pose tensors, the **entire codec** including decoder,
`adaptation.decoder`, `adaptation.height_encoder`, encoder, quantizer and EMA
buffers. Frozen modules stay in eval mode; frozen operations still permit
gradients to their inputs. The vocabulary and call0 source-token labels remain
stable. Record hashes before/after; do not confuse the nonpersistent
`controller.codebook` clone with protection of the actual quantizer state.

If this small scope restores Root control but cannot reconstruct acceptable
legs/contact, examine the frozen codec's source-reconstruction floor. A later,
separate decoder-continuation study could train exactly `codec.decoder.*`,
`adaptation.decoder.*`, `adaptation.height_encoder.*`, with the same prefix
objectives. It is not an automatic expansion of this pilot. Changing the
encoder/codebook would require freezing the new codec and rebuilding Pose
labels before training Pose. Keep that larger intervention separate.

### Train with the deployed terrain chain

At every call, obtain coarse body/request/actual-velocity geometry from H and
the latest intent, predict Root once, detach its query coordinates, sample the
same scene along **that predicted Root**, then condition Pose/decoder on the
resulting geometry and predicted Root. Keep body/actual-velocity observations
available. Re-query after every generated-history update.

The current source-Root teacher and the live observer's training guard cannot
be reused as-is. Root supplies its prediction for both actor decoding and CE
conditioning. Scene queries may run without coordinate gradients; this still
trains the actor on what it will observe at deployment. Never query along the
future source Root or substitute its heights for predicted heights. The
learned terrain encoders are frozen; the **observations** are freshly sampled.

This first repair uses the current source-paired heightfield field. The new
independent full3D field is not yet wired into the actor. Do not silently change
its feature semantics or claim ceiling/crawling capability from this training.

## 5. Commands And Objectives

### Source-derived command annotations

Build labels only from train motion and record them as inferred intent:

1. For ordinary continuation, use mean planar source velocity over the next
   four executed intervals and source heading at the end of that prefix.
   Near-constant sections have speed range<=0.2m/s and heading range<=15deg.
2. Identify real starts from a history speed<0.1m/s followed by a real increase
   above0.3m/s. If the available source has a following near-constant cruise
   section, use its mean speed/direction as the start request, beginning at the
   observed acceleration onset. Otherwise retain the ordinary continuation
   label and mark cruise intent unavailable.
3. Identify real stops from an approach>0.3m/s, a deceleration interval, and at
   least eight real frames below0.1m/s. Assign zero request from the first
   sustained deceleration interval through rest, retaining the real braking
   motion as the pose/Root target. Record the chosen onset and thresholds.
4. Label a genuine idle only when all available prefix intervals are<0.05m/s;
   retain its real poses rather than repeating one frame. Classify strafe and
   reverse from signed velocity relative to the demonstrated body heading.
   Left/right need substantial lateral velocity; reverse needs a negative
   forward component. Do not relabel heading to make the request easier.

Use one deterministic annotation pass over each complete real source interval,
then slice it at the call's source timestamp. If only33frames are accessible,
record unavailable start/stop evidence rather than inventing a transition.
Future source motion may determine these offline velocity/heading **labels**;
actor input contains only the current annotated request, never a future
command schedule, future Root height, stance schedule or body trajectory.
Unseen abrupt joystick changes are evaluated separately from this source
reconstruction task.

### Prefix objective, applied at each call

Let P be the actual world output from the hard-token decoder, R the Root module
proposal, and Q the unchanged source. Compute FK from official rotations in
float32. Define Huber as SmoothL1(beta=1), reducing over valid scalar components
and then examples. For vector differences below, divide by the stated physical
scale before reduction. The following are fixed starting coefficients:

| Term | Definition on executed local frames4..7 | Weight |
| --- | --- | ---: |
| Root | Mean of proposal-R and decoded-P XYZ Huber at0.05m scale; plus `1-cos(yaw_error)` for each. Target is Q at the same absolute time. | 1.0 |
| Body FK | Root-relative joint positions vs Q at0.02m scale, equal mean per joint, with legs/feet reported separately. | 0.5 |
| Rotation | Local SO(3) geodesic angle vs Q, Huber at0.2rad scale. | 0.25 |
| Velocity | World Root/joint velocities vs source, using actual H[3] at the seam; Huber at0.5m/s scale, equal Root/body group means. | 0.25 |
| Boundary acceleration | First played velocity minus H[2]->H[3] velocity, compared with the source's corresponding acceleration; Huber at5m/s2. | 0.10 |
| Source support | Source-stance sole gap to its fixed scene support at0.01m scale, plus source-stance interval velocity error at0.15m/s scale. | 0.5 |
| Collision | Mean squared `max(0,-scene_sdf-0.005)/0.01` on fixed sole/body samples and swept intervals, including H[3]->P[4]. | 0.5 |
| Steady intent | Mean decoded planar velocity vs the request at0.5m/s scale and body yaw vs requested yaw; enabled only on source-confirmed near-constant/idle blocks. | 0.25 |

For acceleration, velocity differences are multiplied by30 before the5m/s2
normalization. Include equivalent angular seam-rate diagnostics. Source targets
contain the actual acceleration, swing and landing; no blanket zero-velocity
or zero-acceleration regularizer is used. Do not enforce cruise velocity on a
source braking/start interval. Those transitions use source trajectory losses.
For decoded heading, measure facing from the authoritative FK Root rotation
with the skeleton's declared forward axis. Report the redundant Root heading
feature separately; changing that feature alone need not turn the rendered body.

Use source contact/stance and a consistent source surface identity for support
losses; the prediction cannot avoid loss by declaring itself noncontact. Apply
support velocity only when both interval endpoints have source stance on the
same surface. Keep source slip and foot banking. Collision loss queries the
actual scene at predicted locations. Unknown geometry is counted and ends the
affected training sequence; it is not converted to empty space or hidden by a
source-history reset. Preserve completed losses and count the failure.

At call0 also add0.1 times source-token CE, and0.1 times the corresponding
labelled forecast losses over frames8..27 to preserve the longer preview.
Use half native corruption masks and half all-mask CE batches; condition them
on predicted Root and the real current observation. Add0.1 times a context
reconstruction term against H on decoded frames0..3, so decoder-past errors
are visible to the trainable upstream modules. Do not copy H into output.

The pilot's per-sequence loss is the mean over its K actual calls, plus the
call0 auxiliaries once. A and B use identical weights/reductions. Split Root
proposal versus decoded Root errors in logs; this distinguishes planning from
codec integration error. Report each component's physical units and trainable
gradient norm instead of relying on total native loss.

### Gradient contract

Each actor call uses the production four-pass hard argmax forward. Reuse
`straight_through_codes`: hard codeword forward values, soft-expectation
gradients from final-pass logits into trainable Pose parameters. It is a biased
gradient estimator. Do not replace the forward with soft averaged poses.
Earlier hard selection decisions and past calls are detached. Backpropagate
each call before releasing its graph, accumulating the mean sequence loss
before one optimizer update. H for the next call is the detached actual FK
encoding, with no source reset within a generated episode.

The source-only audit's `prepare_native_batch` is decorated no-grad and uses
a source Root teacher; it is not the new differentiable rollout implementation.
The existing codec runner also cannot train Root/Pose. Admission must demonstrate
nonzero gradients in every intended group and zero gradients/unchanged buffers
in every protected group, including an actual optimizer update on a disposable
candidate and hard-forward parity with the deploy path.

## 6. Bounded Pilot And Accounting

Propose this initial cap, including source-only control, before a long run:

| Phase | Updates | Global/micro batch | Actor calls per sequence | Purpose |
| --- | ---: | --- | ---: | --- |
| W: immediate source warmup | 256 | 128/8 | 1 | Fit new intent semantics and the near prefix. |
| A: source-history control from W | 512 | 128/8 | 3 | At every timestamp use its actual source history. |
| B: generated-history treatment from the same W | 512 | 128/8 | 3 | Feed executed4frame predictions at calls1/2. |

A/B use the same sampled source IDs, timestamps, commands, scope, endpoint and
CE eligibility. In B, choose a whole sequence as generated with probability
ramping linearly0->1 over its first128updates, then keep1. Its first call always
starts from source; no per-joint history mixing. Log the actual number of
generated and teacher calls. Both arms execute three calls even while B ramps.

Total: **1,280 optimizer updates,163,840 sequence draws,425,984 actor-call
examples, and1,703,936 supervised executed frame examples** at complete budget.
Overlapping frames are counted as exposures, not new recordings. Call0 CE
passes, encoder-label computations, diagnostics and validation forwards have
separate counters; they are not concealed inside the actor-call total. Count
source-family IDs and unique raw source frames separately. At early failure
report actual calls/frames rather than these planned maxima.

Use fresh Adam for W and fresh Adam independently for A/B, seed2333,
gradient-clip0.5, no weight decay. New MLP LR5e-5; existing trainable parameters
LR1e-5. Warm linearly for16updates within each phase, then cosine to1% of each
group's peak by that phase's fixed endpoint. Batch128 preserves64/32/32 domain
counts. Accumulate16microbatches of8; average over K calls and all128 examples
before clipping and stepping. Do not replace this accounting with old
batch256 reference steps or append it to the completed38.4M formal exposures.

Cap the experiment at **six aggregate A100 GPU-hours**, including residency,
admission and evaluation: at most1h W/admission,2h per arm,1h evaluation.
Use one card for W; A/B may each use one confirmed-idle A100 concurrently.
Keep per-process memory<=90%. The parent must inspect current GPU ownership
and measure full K=3 throughput before dispatch; this review makes no resource
availability or elapsed-time claim. If a cap is reached, stop and record an
incomplete result. No automatic extension, alternate LR search or deeper run.

Intermediate checkpoints at128-update intervals are for failure diagnosis and
recovery evidence. **Compare the fixed512 endpoint of A against B.** Do not
select a lucky intermediate model by validation minimum. An implementation
change invalidates the old preflight binding and needs a new source-bound
admission. A process failure does not authorize an uncounted restart.

Borrow the useful parts of `run_codec_repair_pilot.py`: plan/source binding,
parent/cache receipts, protected tensor/buffer hashes, fresh-optimizer policy,
small immutable reports and an export-ineligible experiment schema. Its
decoder-only scope, terminal-source conditions, exact quotas1024 and old budget
are not the reactive plan. Do not run or loosen its old plan validator.

## 7. Deployment Contract And Qualification

The candidate must ship its own schema, checkpoint/architecture hashes, intent
normalization, source/scope manifests, query policy and training ledger.
Preserve original final controls: compact SHA256
`0f5d8e2b05b8ec518a104a4979cca70f23efa64080abfc843f874634de5a9646`
and training parent SHA256
`b960fbe9f792cf46333c23762a8f047259132acd5f6b595bbc726dffa8b40c19`.
The original weights and codebook remain separately loadable controls.

Deployment uses8tokens,4Pose passes,4history,32decoded/24playable frames,30Hz
motion,0refinement and4frame adoption. Left stick moves in the declared world/
camera frame; right stick controls heading independently. Keep the corrected
input coalescing/cadence. Record the command timestamp/sequence actually used
for every plan rather than reporting the latest UI value as actor input.

The published history is four actually displayed frames. Where asynchronous
lead requires an accepted future prefix, distinguish it explicitly from played
history and condition at its exact immutable boundary. A candidate starts at
its exact first forecast frame; no skipping into a late forecast, resampling,
tail hold, manual gait-phase reset or mutation of an accepted displayed frame. Train the
four-frame generated loop first, then replay the measured lead/commit contract
against that same candidate before opening it for user control.

### Fixed development qualification

Register the validation sources, feasible command routes, traces, thresholds
and equal seeds **before** W/A/B. Use untouched source-group validation for
selection; use the already-inspected Urban/gamepad failure traces as development
regressions only. Do not relabel them as held-out tests. Hold the original test
split sealed until the fixed candidate is chosen. Source-normalized results
and arbitrary-command control results are separate scorecards.
The previous final-evaluation test failures are already observed evidence;
retain them as such. They cannot become a fresh unseen benchmark, determine
new training membership or select this pilot's objective. Register any fresh
scene/input combinations before running candidates and use the test split for
fixed-candidate reporting rather than an iterative repair loop.

Compare original model/original consumption, original model/cadence-corrected4,
W, A and B. For paired trials use the same scene, initial four frames, actual
world command trace, heading, seed and asynchronous policy. Include:

- Sustained forward at0.6/1.2/2m/s; true left/right strafe and reverse with held
  body heading; combined movement plus independent facing changes.
- Real start, braking, direction reversal, analog magnitude noise and rapid
  right-stick changes; commands arrive at100Hz while adoption stays4frames.
- At least16seconds of genuine idle and stop after motion, followed by a new
  start. Measure motion progress and gait activity as well as drift.
- Flat, slopes, stairs and rough-ground entrances on feasible fixed routes,
  including transitions approached quickly. Keep the commanded speed trace
  fixed; improving quality by silently reducing it is a failed control result.
- Long raw-output rollouts: development sequences>=120seconds, then three
  30-minute continuous sessions for final delivery, with no corrective resets.

Initial **pilot screening thresholds**, to be registered rather than tuned
after seeing B: on flat, settled signed command-axis progress80%-120% of
requested speed in every feasible movement direction; orthogonal speed<=0.15m/s;
heading error P95<=15deg when heading is held;1.2m/s starts reach half target
speed within0.4s; stop falls below0.1m/s within0.5s and stays there. Sixteen-second
idle displacement<=0.05m. Root-relative adjacent-joint displacement P95 must
not exceed1.25 times matched source P95 plus0.01m, and boundary errors must be
reported separately from within-plan changes. Report gait cycles and stance
sliding so stationary legs cannot pass using Root progress alone.

Evaluate original full-skin criteria independently: all18,056vertices,
eight swept substeps, peak penetration<=0.02m, no unknown/uncertain queries,
original source stance, complete routes and cross-plan continuity. A sparse
training collision loss is not the full-skin acceptance measurement. Report
every failure and each family/direction separately; do not hide reverse or
stairs behind the aggregate mean. A source-reconstruction gain alone does not
qualify a new Xbox controller.

Require actual display29.5-30.5fps, zero missing/duplicated motion frames,
measured deadline misses and complete input->consume->forecast->adopt->draw
timestamps on the RTX3080 deployment machine. Measure A100 training throughput
separately. For this **four-frame pilot** use input-to-visible P95<=200ms as an
intermediate screening target; retain the project's final P95<=100ms target.
At fixed four-frame adoption, a randomly phased command can wait up to133.3ms
for the next boundary; its ideal95th-percentile boundary wait is126.7ms before
new computation. A four-frame-only result cannot guarantee the final100ms gate.

If B passes motion/terrain screening but misses response, the next separately
bounded experiment can train2/1frame feedback with the same availability
rules, or a compact next-frame controller with a longer MB preview. The latent
grouping changes under1/2frame shifts, so previous-token shifting is invalid;
train the actual shifted history/forward path. Do not expose an untrained faster
mode solely because it renders30fps. No further experiment starts automatically.

## 8. Parent Implementation Checklist

1. Lock the new plan/source/split/parameter whitelist and endpoint budget.
   Reconstruct source-scene joins and the33-frame target/command index on CPU.
2. Add a separately versioned immediate-intent actor and prefix-loss wrapper.
   Keep the original inference path and source-only loss guards intact.
3. Verify exact4frame advance, unchanged physical source labels, valid derivative
   endpoints, call0-only CE, predicted-path queries, no future truth conditions,
   and hard-forward decode parity. Test a truncated source tail explicitly.
4. Check one full optimizer update's scope, frozen EMA/decoder/encoder hashes,
   per-term gradients and reproducibility; measure complete three-call cost.
5. The parent then runs bounded W/A/B using the authorized A100 resources and
   its single persistent SSH connection. Keep accounting and artifacts separate
   from all previous full stages and the old codec proposal.
6. Review fixed endpoints on registered validation/control cases, then run the
   deployment timing/continuity replay. Open a separate Xbox candidate page only
   after these pass. Export/delivery flags remain false until the final gates.

## Review Binding

These are current local source hashes read during this review, not a remote
deployment receipt. All source paths are relative to the `motionbricks` root.

| File | SHA256 |
| --- | --- |
| `docs/reactive_pose_context_audit_20261021.md` | `a60ec09c21c22f7bfe5daa799d6ae19af69e59b05bcd3c64c0fefda0bf02e070` |
| `motionbricks/data/unified_full_training.py` | `37c672b82473bb82ef3bfe50b3554f56795202f663207b9c488daa60a1fd7a9a` |
| `motionbricks/data/unified_terrain.py` | `dcf2f3116a5d826929d5af99e87a697a9bc1e0d987800516ccc415dd215ce10f` |
| `motionbricks/data/full_terrain_cache.py` | `b194b5d5ac6dbba42deea00e054f1c8359c299d86e9e413cd6754cfba6962e53` |
| `motionbricks/data/root_path_terrain.py` | `684c1bd3f7f537cbfe7af2bbc17877f6b0bdf81fde5ecb0806d2bf14f8e56fa9` |
| `scripts/run_codec_repair_pilot.py` | `353d0bb71f7f9c708407cb82fb73a67551e58b94d8b88f4182f335ba852d2578` |

Additional inspected sources: `unified_pose_training.py`,
`unified_training_goals.py`, `terrain_decoder_pilot.py`,
`terrain_root_supervision.py`, `native_mb_online_conditions.py`,
`native_mb_observed_history.py`, `terrain_learning_data.py`,
`terrain_learning_episode.py`, `terrain_joint_objective.py`,
`terrain_model_v3.py`, `reactive_path_execution.py`, Root/Pose backbones and
losses, codec loss, `scripts/cache_full_terrain_primitives.py`,
`scripts/train_joint_terrain_v3.py`, and `scripts/train_unified_full_corpus.py`.
The old episode class advances by8 and consumes8:12; it is a useful source-
clock/provenance reference, not a four-frame continuation implementation.

## 9. Implementation Review: The Smaller64-Update Pilot

The parent's subsequent decision supersedes sections4/6 for the **first run**:
`configs/reactive_prefix_pilot_20261021.json` and
`scripts/train_reactive_prefix_pilot.py` now define64updates per arm,
global8/micro1, K=7, existing input slots, and no new intent MLP. Root, Pose,
codec decoder and their listed adapters train; the shared scene encoder,
codec encoder, quantizer and EMA remain frozen. The original model remains
the control. The larger W/A/B design above is a later proposal, not a second
run to launch alongside this one.

This is a static implementation review. I read the runner and
`reactive_prefix_training.py`, `reactive_training_data.py`, their inference
dependencies and the prefix tests. I did not execute tests, preflight or Adam.
The source bank was being edited concurrently; hashes below identify the
reviewed versions. Recheck the final frozen snapshot before launch.

### Launch Findings

**[P1] The runner alone does not enforce its3,600-second wall cap.**
`train_reactive_prefix_pilot.py:177-184,190-191,208-210` checks elapsed time
only before starting a whole update. Each update executes56actor calls.
Baseline validation happens before that loop; final validation and checkpoint
serialization happen after the last check. A long admission/data load,
evaluation, update or save can exceed the declared cap. Before launch, provide
an external process-group timeout covering the entire arm, or enforce a common
deadline throughout those operations. Record partial completed updates and
in-progress actor calls on termination; no automatic restart. This is a
resource-bound blocker, independent of whether the numerical preflight passes.

**[P1] The preflight source binding omits code that determines the actor.**
`train_reactive_prefix_pilot.py:36-43` hashes the runner, two new helpers and
`root_path_terrain.py`. A change to `unified_terrain.py`,
`native_mb_online_live.py`, `native_mb_online_conditions.py`,
`native_mb_observed_history.py`, `motion_inference.py` or
`terrain_model_v3.py::straight_through_codes` currently leaves that proof
valid. These files determine terminal injection, forward gradient behavior,
history and emitted output. Bind the complete actor/loss/preparation dependency
snapshot, parent/bundle identity, exact trainable names and the selection/data
receipt in the preflight-to-execute check. The parent/bundle/cache identities
are checked during construction; the missing part is binding the numerical
proof to that whole implementation. Freeze the snapshot after these fixes.

**[P1, interpretation/contract] Near-prefix relabeling leaves the terminal
mechanism active.** `prefix_command` is correct as a label function, but the
runner passes it to `UnifiedTerrainController.forward` unchanged.
`unified_terrain.py:236-249` still declares `VELOCITY_PROTOCOL`; the native
condition builder sets slot7 VX/VZ and desired heading. Inference then calls
`apply_terminal_velocity` at `lengths-1`, index31 for this horizon. This
overwrites the **local Root condition supplied to Pose/decoder**, not every
executed frame's velocity. The actual decoded Root is still learned output.

This small experiment can proceed as **near-prefix relabeling under the
existing terminal-conditioned actor**, once the resource/proof blockers are
resolved. It tests whether its short-prefix loss learns a better continuation
despite that architecture. It does not isolate removal of the moving terminal
deadline. Before launch make the run contract explicit:

```text
conditioning_protocol = native-mb-velocity-online-v2
command_label_source = next-four-source-intervals-average-and-last-heading
pose_decoder_terminal_velocity_override_index = 31
new_immediate_intent_encoder = false
```

The current JSON's `command_semantics` describes label provenance but omits
this actor-side interpretation. Preserve that distinction in the report and
candidate loader; do not describe the result as the proposed immediate-intent
architecture from section4.

### Correctness Checks That Pass Source Review

- Seven calls use offsets0,4,...,24. Every emitted4:8 corresponds to source
  offset+4:offset+8. The last call labels28..31 and its boundary predecessor27.
  The current loss needs no label beyond31; source frame32 is unused. There
  is no fabricated full32-frame shifted target and no encoder label from a
  padded suffix.
- `prefix_command` takes displacement from offset+3 to offset+7 divided by4/30,
  exactly the mean of four successive source intervals, plus heading at+7.
  It reads XZ/heading only. The full source window is held by the loss/label
  code; only that command, four factual history frames and scene queries enter
  the actor. Later source body/height/contact values are not actor conditions.
- In the generated arm, output4:8 is detached and officially FK re-encoded
  before the next call. In the source arm the corresponding real four frames
  are FK re-encoded. Source labels keep their original world coordinates.
  Boundary velocity starts at actual history, not the decoder's reconstructed
  past. No Root correction/filter or within-sequence source reset is added.
- `TrainingRootPathObservation` explicitly enables the same post-Root query
  in training and evaluation. Geometry coordinates are detached; the query
  follows predicted Root. Pose/decoder receive the refreshed path observation.
- The outer `predict_planar_terrain` no-grad decorator does **not** disable
  all training gradients: `UnifiedTerrainController._predict` re-enables them
  around the neural chain when `context['training']` is true. The existing
  hard-forward/soft-gradient dequantizer then exposes final-pass Pose gradients.
  Actual Root/Pose/decoder nonzero finite gradients still need the preflight.
- `configure_scope` sets the entire controller to eval and enables the declared
  prefixes. Thus dropout and EMA updates stay off. Broad `root.*`/`pose.*`
  includes their native terrain encoders; the shared `adaptation.encoder` and
  codec encoder/quantizer are frozen. This is broader than the earlier minimal
  scope, consistent with the parent's revised pilot decision.
- Source bank selection reads train/validation motion only, q0 only, preserves
  source-group splits, recovers matching manifest-bound scene/contact assets,
  and compares the real reconstructed33frames to the existing cache. Test
  record metadata is inspected for split integrity, not used for label sampling.
  Selection receipts now expose missing categories. The new code does not
  restart the old decoder pilot.

### Budget And Pairing

The present constants give8samples/update with4/2/2domains; each sample has
seven backward passes weighted `1/(8*7)` before one Adam step. This is the
correct mean sequence loss at micro1. Each arm draws512source-window exposures
and3,584supervised prefixes, totaling14,336executed frame exposures. Across both
arms:128actual Adam steps,1,024window exposures,7,168prefixes and28,672frame
exposures. None belongs to the completed38.4M exposure ledger.

There are at most96distinct selected q0 train windows, sampled with replacement;
512does not mean512unique sources. Baseline/final each evaluate24validation
windows times7calls=168actor examples, or336per executed arm. Record those
separately from training, along with diagnostic preflight calls. The numerical
preflight tests both arms but performs no update. The same NumPy sampling seed
gives A/B matched window draws; save a sample-sequence digest as evidence.

The current plan arithmetic check covers `64*8=512` only. Before freezing it,
also assert micro1, stride4, K7, `512*7=3584`, runtime8/4/4/30/4, domain quotas,
scope and export flags. The runner's batch construction and prefix slicing are
hardcoded to these values. A modified JSON must not silently describe a
different batch or rollout contract from what is executed.

### Is Omitting CE A Blocker Here?

**No: it is acceptable for this64-update mechanism pilot.** The rollout loss
already has a defined gradient into Pose through hard-forward codewords, while
encoder/codebook labels remain fixed. Later shifted windows do not have valid
full32-frame token targets, so omitting their CE is correct. Call0 CE is possible
but is an additional preservation choice, not required for causal supervision.

The concrete risk is drift of the unexecuted suffix and token usage while the
decoder and all Root/Pose weights adapt to a small repeatedly sampled bank.
Frozen vocabulary does not freeze what the decoder does with it. Both arms
omit CE, so their comparison still isolates history source under that choice.
Keep the experiment export-ineligible and retain the original checkpoint.
Before and after training, report token usage, source reconstruction/leg motion
and unexecuted preview quality on fixed validation windows. If those regress,
a later bounded run can add call0 CE or a preserved-source auxiliary; do not
retroactively change this run's loss or pick an intermediate checkpoint.

The current four-term loss has no explicit scene collision, source support,
steady-intent, or context-reconstruction term. Its weights are rootXYZ1,
Root-relative joints1, global-rotation matrix MSE0.1 and joint velocity0.25.
That is a coherent source-following numerical pilot, not the full contact/
arbitrary-command curriculum in section5. Its validation reports only0.933s
of reconstruction. Root RMS and endpoint distance cannot establish stopping,
signed strafe/reverse control, idle drift, full-skin clearance or Xbox30fps.
Those fixed-command/long-rollout gates remain required before deployment.

### Admission Still Needed From The Parent

After the snapshot/timeout/contract corrections: use the new preflight to verify
all seven calls, finite hard-forward parity and intended model-group gradients.
Include at least one admitted window from each domain so the scene joins and
predicted-path queries are exercised beyond the first PFNN record. Bind loaded
source-bank receipts instead of discarding `bank.load(...)["receipt"]` from the
run artifacts. Check an actual Adam update's protected buffers and finite model
tensors in a disposable admission candidate; charge that work explicitly and
start the real arms from the untouched parent. This review supplies no GPU
admission evidence and launches no process.

Reviewed hashes:

| File | SHA256 |
| --- | --- |
| `configs/reactive_prefix_pilot_20261021.json` | `f1cf1a7d971be81105b82606d94c602d8be544a69669b35ebd2c8522bef46ea4` |
| `scripts/train_reactive_prefix_pilot.py` | `f68ea50ddc8a16a9bffe2ef1c762c1c396ce3a6e74cda57dc76015e0477f8cab` |
| `motionbricks/data/reactive_prefix_training.py` | `55c3a3f8adbb9ab56743740494f6fe345f63134ac0c7fb458f942f40dc89d0d2` |
| `motionbricks/data/reactive_training_data.py` | `1bccee80d942a0805260542c8bc76308a372a6fec4ba4d786aa0274832758826` |
