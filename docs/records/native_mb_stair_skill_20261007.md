# Native MB Stair Skill, 2026-10-07

## Scope and authority

Start from the exact 50,000-update LAFAN SOMA30 Tokenizer, Root and Pose
checkpoints currently used by the native UE trial. Preserve those checkpoints
and the live UE bundle. Use the released MB models and losses; no runtime IK,
foot locking, smoothing, terrain projection or decoded-root overwrite.

References:

- [MB Sections 4-6](https://arxiv.org/html/2604.24833v1): separate pose tokens,
  predicted root conditions, sparse keyframes and decoder root refinement.
- [MB Section 7.4](https://arxiv.org/html/2604.24833v1#S7.SS4): paper optimizer,
  warmup, batch size and schedule; executable release architecture is retained.
- [MB Appendix E/G](https://arxiv.org/html/2604.24833v1#A5): no continuous terrain
  data; authored task actors provide sparse target poses to the neural backbone.

This phase tests **keyframe-conditioned ascending and descending skills**, not
autonomous traversal of arbitrary terrain. Original Pose root conditioning is
global XZ and heading (`root_without_hip_height`). The local decoder root
condition is XZ velocity. Predicted Root height is not a hard height trajectory
input to either; height enters through sparse target pose/root-height slots and
the decoded motion. Raising a root curve alone therefore is not a stair proof.

In the released inference wrapper, the height in the input sparse global-root
slots is prepended to the internal pose-keyframe feature **before** Root
prediction (`motion_inference.py`, `predict`). It is not taken from the later
predicted dense Root-height curve. This distinction matters for stair goals:
authoring only a vertical root path does not tell Pose which foot lands where.
Smart Object/task actors provide sparse body/landing intent keyframes; the
unchanged decoder then generates the full motion and refines its final root.

## Immutable input lineage

Baseline runs under the remote project:

- `out/native_mb_lafan_paper_20261006/stage50k-v1/tokenizer-50000`
- `out/native_mb_lafan_paper_20261006/stage50k-v1/root-50000`
- `out/native_mb_lafan_paper_20261006/pose-efficiency-v1/training/pose-50000`

Each checkpoint hash, configuration hash, 50k update count and manifest-bound
normalization are validated before transfer. New runs use a fresh paper Adam
optimizer and its unchanged 10,000-update warmup / 2M schedule. This is explicitly
weights-only fine-tuning, **not** uninterrupted optimizer-state continuation.
Inherited and newly executed updates are recorded separately. Pose transfers
only its backbone; its newly bound frozen tokenizer is not overwritten by the
old supporting-network checkpoint state.

## Stair dataset

Remote dataset: `${PRIVATE_STORAGE}/motionbricks-bones/native-stairs-50k-v3`.
Manifest SHA256: `36e82bba18447ebfe83576343aa368160dfd32a18a2b53c2da059ab70834fc9c`.

Of 34 eligible BONES `stairs_climbing` BVHs, 21 pass conservative weak-geometry
checks. A300 provides 5 up and 5 down training clips (50.8 seconds total); A301
provides 6 up and 5 down validation clips. Thirteen clips without adequate
distinct-level/bilateral stationary evidence remain excluded. All 9 original
accepted LAFAN native shards are byte-copied with their original splits. Combined
split counts: train 15, validation 13, test 2. No test stair clips are invented.

Conversion uses official Kimodo BVH interpretation, SOMA77 to SOMA30 conversion,
official FK and SOMASkin LBS, and existing 30 Hz resampling. Source rotations/root
remain archived. Only one declared whole-recording vertical translation anchors
the lowest observed weak level; no per-frame changes. A single official heading
and XZ transform puts features, soles and static polygons in the same frame.
Selected-vertex LBS/full-mesh parity and feature/FK roundtrip checks run during
conversion. Mean/std bytes are copied from the 50k baseline, not recomputed.

The four contact feature channels encode positive stationary material-point
evidence as 1, other frames as 0. Separate confidence/unknown masks remain
metadata: the unchanged native reconstruction loss does **not** mask those zero
feature targets. These are explicitly heuristic features, not airborne/load
truth. Do not describe the masks as consumed training supervision.

Static support hulls are weak source-derived observations. Full tread extents,
riser faces and unvisited space remain unknown; hovering ambiguity remains.
They cannot certify collision-free movement through an independently measured
staircase. Full source motion-derived patches are never future network inputs.

## Training and automatic evaluation

New run root: `out/native_mb_stairs_20261007/ft5000-v1`.
Tokenizer and Root execute 5,000 additional updates on separately checked idle
cards. Pose then executes 5,000 updates with the completed frozen child tokenizer.
The original models, losses, paper batch256/Adam5e-5/warmup/schedule stay intact.
Finite losses/gradients, actual update counts and tokenizer immutability are
checked on the actual training path.

`review_native_mb_stairs.py` chooses held-out windows from source vertical travel
and bilateral evidence only; no prediction-dependent selection. Input: observed
body/global root frames 0-3, observed local root intervals 0-2, plus target frame
63 and its authored endpoint derivative. No source interior root trajectory or
encoded source tokens are supplied. The unknown frame-3 velocity is masked,
not taken from future source frame4. Generated interior frames4-62 and intervals
4-61 are scored; conditioned endpoints are reported separately. Reconstruction
with complete source tokens/root is diagnostic only, never a skill proof.

Source evidence and numbered-material-point masks are fixed across arms. Report
vertical/horizontal root progress, expected-support height gap, known-patch
penetration, source-timed material slip and unknown spatial coverage. Unknown
query heights produce null statistics rather than zeros. Engineering probe gates
are declared before child evaluation: height-change error <=8cm, XZ travel error
<=15cm, gap/penetration P95 <=3cm and slip P95 <= source P95+5cm/s. These are not
paper metrics or physical constants. No gate relaxations after seeing predictions.
Metrics are written before optional rendering. The same validation windows and
flat-locomotion regression checks must accompany the trained-child result.

## Baseline result

`out/native_mb_stairs_20261007/baseline-context4-v1/report.json`:
three up and three down windows, **0/6 pass**. Source up travel 45-59cm is decoded
as only 13-16cm. Source down travel 19-37cm is decoded as only 7-13cm. Sole support
errors and source-timed slip also fail. The current pure-flat 50k checkpoints do
not establish stair skill. No trained-child capability is claimed until its
independent generation review completes.

Earlier `native-stairs-50k-v1/v2` conversion outputs and pre-context baseline
reviews are failed/historical artifacts, not training inputs or valid A/B arms.

## Fine-tune validation progress

Root completed 5,000 new optimizer updates with zero skipped batches. Tokenizer
also completed 5,000 updates; its first post-training check failed because it
compared the original FP32 straight-through output `x + (code - x).detach()` with
direct code retrieval using an absolute 1e-5 decoder-output threshold.

Measured on all 78 conditioning/window pairs: no token-identity differences,
all outputs finite, worst raw feature error 4.0054e-5, latent differences at most
2.3842e-7. The shared evaluation check now requires exact token identity, an
exact replay of the shipped straight-through arithmetic within its floating-point
roundoff bound, and unchanged decoder replays for both paths at the original
1e-5 threshold. Raw discrepancies remain in the report. No original quantizer,
model weights, objective or generated output was changed. Six regressions using
the shipped quantizer cover roundoff, damaged outputs, changed token IDs,
nonfinite outputs and training-mode rejection. The focused suite is 42 passed.

The completed Tokenizer review is
`ft5000-v1/tokenizer/validation-roundoff-replay-v1/metrics.json`, with provenance
in `validation_recovery.json`. On the same 39 held-out source windows and context4:

| Behavior | Windows | Baseline MPJPE (m) | Fine-tuned MPJPE (m) |
| --- | ---: | ---: | ---: |
| Stair up | 18 | 0.6965 | 0.0772 |
| Stair down | 15 | 0.4908 | 0.0891 |
| Flat walk | 3 | 0.0635 | 0.0735 |
| Flat run | 3 | 0.0637 | 0.0728 |

These are **source-encoded reconstruction**, not generated stair skill. Flat
reconstruction regresses about 1cm, so the old live UE bundle remains unchanged.
Pose completed 5,000 new updates with zero skipped batches and the child
tokenizer tensor state unchanged. Runtime: 727.6 seconds, peak allocation
9,871MiB; final checkpoint SHA256
`151467e172a194cebc265468528dcbefb4ec69b34f10f59eadbad809fe0cb187`.

## Generated-child result: not approved

`review-ft5000-v1/report.json` and six source/generated full-body GIFs completed.
The same held-out windows and gates remain unchanged: **up 0/3, down 0/3 pass**.
All six generated height changes have the correct sign, but that is insufficient:

- Source up changes 0.453/0.499/0.591m; generated 0.432/0.450/0.507m.
- Source down changes -0.186/-0.255/-0.374m; generated -0.208/-0.436/-0.561m.
- Up root height MAE 0.023-0.067m; down MAE 0.196-0.379m.
- Expected source-support gaps P95 0.123-0.554m and source-timed material speeds
  P95 0.488-3.096m/s fail the declared source-phase support test. These speeds
  are not measured load-bearing slip: a predicted swing during a source stance
  can also produce them. The weak patches do not certify alternative footsteps.

`flat-regression-ft5000-v2/report.json` compares the same six LAFAN windows with
the same four observed context frames and single endpoint. Free-frame generated
MPJPE changes from 0.1586 to 0.2432m for walk and 0.5677 to 0.5711m for run. Walk
root error increases from 0.0973 to 0.1638m. This is a paired in-betweening proxy,
not a full UE/WASD regression or a unique-motion/naturalness metric. Do not deploy
the child. The v1 flat review wrote its full report but failed its final console
print; v2 reran with the corrected driver and exited zero. No v1 numbers are the
authoritative delivery.

`oracle-stage-diagnostic-v1/report.json` uses the **original MB debug flags** for
ground-truth root trajectories and pose tokens, separately and together. All
such results are future-informed diagnostics, never stair-skill evidence. With
true root but generated pose tokens, down root-height MAE remains
0.174/0.375/0.376m. With true tokens it falls to 0.008-0.014m; both true root and
tokens give free-frame MPJPE 0.041-0.063m. Thus improving Root alone cannot fix
the current downstream height/motion selection. This does not identify a unique
training cause or justify modifying the original losses.

Next actionable boundary: improve actual sparse-condition Pose generation and
test the paper's authored stair-landing keyframe sequence with generated history,
not an upward Root curve or source-motion replay. Sequence targets, their origin,
timing, observed context and future information must be declared separately;
stage diagnostics must remain separate from capability acceptance. The old 50k
live UE bundle is unchanged. No terrain perception, arbitrary rough-terrain
policy, physical feasibility or successful stair skill is claimed.

Focused local suite: **94 passed**, including original checkpoint-review checks,
warmstart, real-quantizer parity, official-geometry conversion, stair probe and
flat regression. This is not a full-repository test result.

## Sparse-intent implementation and 15k continuation

This section supersedes causal interpretations of the earlier four-context
generation probes, not their historical numbers. Stair skill is still **not
approved**. The live UE flat-50k bundle was not replaced.

### Original MB contracts, not a new terrain model

The paper's [Smart Object workflow](https://arxiv.org/html/2604.24833v1#S6.SS2)
uses authored sparse intent keyframes. The new
`scripts/review_native_mb_stair_sequence.py` follows that contract: provide a
short initial observation and sparse landing goals, then feed back only the
generated motion. No source interior poses, source motion tokens or dense source
root trajectory are supplied. Outputs are never overwritten with the goals.
There is no IK, foot locking, filtering, root correction or terrain projection.

Calling the original `_construct_keyframe_prob` at the saved training steps
shows that Root and Pose currently train on **one start and one end keyframe**.
Four available history slots do not imply that all four should be constrained.
On the same six short windows, unchanged 5k weights and common free-frame masks:

| Input condition | Up root height MAE | Down root height MAE |
| --- | ---: | ---: |
| Four context poses, one goal | 3.77 cm | 31.10 cm |
| One context pose, one goal | 3.36 cm | 3.09 cm |

These are arithmetic means over three windows per direction. This establishes
an input-distribution problem in the stair probe; it does not establish that the
existing flat UE controller had the same problem. See `intent-ablation-v2`.
The earlier oracle results used the four-context probe and must not be treated
as a unique diagnosis of Pose training failure.

The sequence evaluator also directly invokes the original navigation agent's
`_predict_inbetween_motion_features` input packer. This rebuilds constraints from
actual FK joint transforms, like the official runtime, rather than directly
reusing redundant decoded feature channels. Raw-feature and official-FK arms are
both retained. Their effects are mixed: this is contract alignment, not evidence
that repacking alone cures discontinuities. Pose's shipped
`root_without_hip_height` mode does not consume dense Root Y, and the decoder's
external root input is horizontal velocity. An ascending root curve alone is
therefore not a sufficient stair command; sparse pose intents also matter.

### Actual training completed

Both original trainers resumed their 5k stair checkpoints to 15k, preserving
optimizer and scheduler state: **10,000 additional updates each**. Adam, learning
rate 5e-5, batch 256, 10k warmup and the 2M schedule remain unchanged. No new loss,
model component or dataset alteration was introduced. The 5k stair tokenizer
remained frozen, with identical tensor-state hash in Pose before and after.
RNGs were reseeded on resume, so this is not bitwise uninterrupted sampling.

| Module | Output under `ft15000-v1` | New updates | Time | Peak allocation |
| --- | --- | ---: | ---: | ---: |
| Root | `root-resume` | 10,000 | 1,086 s | 4,271 MiB |
| Pose | `pose` | 10,000 | 1,468 s | 9,875 MiB |

The first Root launch was rejected by the runner-source guard before training.
After reviewing the already-recorded tokenizer evaluation-only STE roundoff
repair, the retry explicitly recorded that reason. The generic run-contract
`change_scope` mentions a numerical-policy transition, but recorded before/after
numerics are identical; the explicit reason states it was **not** a matmul change.
`ft15000-v1/root` is not a completed run; use `root-resume`.

### Whole-recording results

Six held-out recordings, 108-144 frames at 30 Hz, use 6-10 sparse authored goals
each. Goals come from the held-out source recordings, not from perception or an
independent terrain planner. After the initial four frames all history is
generated. All variants use identical source records, goals and free-frame
masks, excluding goal poses and derivative endpoints. Source-derived support
patches are weak observed footprints, not complete stair treads or risers.

Official runtime packing, one Pose iteration; arithmetic means over three
recordings per direction:

| Direction | Root/Pose stair updates | Root height MAE | World MPJPE | Joint acceleration P95* |
| --- | --- | ---: | ---: | ---: |
| Up | 5k / 5k | 3.76 cm | 14.50 cm | 49.92 m/s2 |
| Up | 15k / 15k | 4.53 cm | 15.33 cm | 49.03 m/s2 |
| Down | 5k / 5k | 3.02 cm | 11.78 cm | 32.96 m/s2 |
| Down | 15k / 15k | 1.73 cm | 9.21 cm | 32.84 m/s2 |

*Acceleration covers all frames, including joins; the table averages per-clip
P95 values, not a pooled percentile. Source P95 is 13.96-17.03 up and 11.51-13.85
down. MPJPE is agreement with one source realization, not a complete naturalness
metric. Downward height/path agreement improves, but smoothness is not solved.

Known-patch sole penetration P95 at 15k spans 0-5.93 cm up and 0.18-5.09 cm down
across clip/foot pairs. Known vertex coverage is only 8.2-44.2% across all pairs;
unknown spatial queries are not zero-error observations. Near-surface material
speeds still show errors, but those candidates depend on predicted geometry and
their coverage changes between runs. Do not interpret a change in their mean
alone as a paired slip improvement. Source-phase agreement remains a separate
fixed-mask diagnostic, not measured load-bearing slip.

Root/Pose swaps (5k/15k and 15k/5k) were also evaluated without training new
models. Neither is uniformly best for ascent, descent and smoothness. Four Pose
iterations likewise do not consistently remove the defects. Selecting a module
combination after these validation results is not an independent skill test.

### Flat regression and tokenizer boundary

Matched one-context flat probes, three windows each, show world MPJPE:

| Behavior | Original flat 50k | Stair 5k | Stair 15k |
| --- | ---: | ---: | ---: |
| Walk | 9.72 cm | 12.47 cm | 13.27 cm |
| Run | 12.73 cm | 18.69 cm | 18.04 cm |

This is an offline paired generation check, not a new UE/WASD test. It is another
reason not to deploy the stair child over the accepted flat model.

The token audit compares identical motions without remapping code IDs: only
22.59% of IDs agree between the flat and stair tokenizers (12 held-out windows,
8 heads of 10 codes). This is substantial representation change, not proof of a
single cause or a license to remap learned embeddings. Pose is correctly bound
to the frozen child tokenizer. Existing source-encoded tokenizer reconstruction
also has centimetre-scale support errors; those measurements used four-context
conditions and are not a precise ceiling for every current one-context rollout.

The next training boundary is therefore **original-MB tokenizer fidelity on the
mixed flat/stair data, followed by a frozen vocabulary and Pose adaptation**,
not more Root-only training or output repairs. Reuse the automatic source-encoded,
sparse-generation and flat checks during development. First compare tokenizer
reconstruction under the same one-context conditions, and keep any tokenizer
continuation as a separate version; never silently replace the vocabulary under
the existing Pose model. The current stair training portion is only 50.8 seconds,
so broader terrain or unseen-stair generalization is not supported by this probe.
No further training job is left running by this experiment.

### Artifacts and verification

All artifacts are under `out/native_mb_stairs_20261007`:

- `intent-ablation-v2` and `intent-ablation-ft15000-v1`: short matched-condition probes.
- `whole-official-ft5000-v1` and `whole-official-ft15000-v1`: full-recording rollouts.
- `whole-official-r5-p15-v1` and `whole-official-r15-p5-v1`: module-swap controls.
- `flat-regression-sparse-ft5000-v1` and `flat-regression-sparse-ft15000-v1`: flat checks.
- `token-semantic-audit-v2`: authoritative token and frozen-model audit.
- `comparison-ft15000-v1`: source / 5k / 15k videos for the first up and first
  down recording, selected by fixed report order, not by their scores.
- `ft15000-v1/root-resume` and `ft15000-v1/pose`: local training contracts/results;
  full checkpoints remain on g3.

The sequence tests cover sparse masks, no unprovided future leakage, generated
history, official FK packing, token-grid timing, goal exclusion and unknown
support. Focused local suite: **136 passed**. This is not the full repository
suite or a quality approval. UE bundle SHA256 remains
`a17027e3cf8fc4958437329fa91a0889324949f37458a87af36c8d9c824e963c`.

## Decoder and contact semantics follow up

The accepted UE bundle remains unchanged. This iteration keeps the released
models, objectives and paper optimizer settings. It does not introduce runtime
IK, foot locking, blending, root replacement or terrain projection.

### Original tokenizer continuation

`tokenizer15000-v1` resumes the v3 5k tokenizer to 15k, with the original
optimizer and scheduler states: 10,000 additional updates, no skipped batches,
868.9 seconds, 2,828 MiB peak allocation. Training reconstruction loss falls
from 0.01492 to 0.00846, but that is not a deployment gate.

The one-context source-encoded review uses pose frame 3, true encoded source
tokens and dense source root inputs. Across 39 held-out windows, MPJPE falls
from 7.94 to 7.16 cm; root-relative MPJPE changes from 4.78 to 4.82 cm. On six
fixed stair geometry windows, ascent support gap improves, but descent gap and
source-timed foot material speed get worse. This checkpoint was not silently
substituted underneath the existing Pose model.

### Versioned native plant labels

The v3 training feature used stationary shoe material points as positives.
The original tokenizer skate objective instead measures FK foot JOINT speed.
Rolling contact therefore created conflicting targets. Among original positive
labels in training, approximately 15.4% of ascent and 11.1% of descent foot-joint
frames exceeded the shipped 0.15 m/s stationarity threshold.

`scripts/relabel_native_mb_stair_plant.py` creates the separate dataset
`${PRIVATE_STORAGE}/motionbricks-bones/native-stairs-plant-v4`, manifest SHA256
`c2897484c57be7b07a442a1a2290004c0ff300e800a837218cb98b7eeb1a4824`.
It calls the shipped FK, forward velocity function (repeated final interval)
and foot detector. Height uses the foot's known weak support patch, rather than
world Y=0. Thresholds remain 0.15 m/s and 0.10 m. Unknown support cannot create
a positive. The resulting feature is the intersection with the old positives;
it never creates additional contact evidence.

Training positives change from 2,772 to 2,405; validation positives from 2,474
to 2,032. All source weak evidence, masks and geometry remain archived unchanged.
Flat shards, splits and statistics are byte-identical. Only four columns of
the stair global training features change. The native objective still consumes
zero-coded negative/unknown targets without confidence masking: this is a
documented limitation, not measured force truth.

`plant-v4-tokenizer5000` starts from the same flat-50k tokenizer as the historical
v3 5k control. Original configuration, seed, batch and 5,000-update budget match;
the historical control was not rerun in this iteration. Training takes 435.9 s,
peak allocation 2,828 MiB. Checkpoint SHA256:
`5205ff80a8f5ec89d43770c76a762e80f1af4326cadaaf7003792689c4fbfb60`.

All models are reevaluated on the SAME v4 source features and fixed original
geometry masks in `plant-v4-tokenizer-geometry-c1`:

| Direction | Model | MPJPE cm | Support gap P95 cm | Known penetration P95 cm | Source timed speed P95 m/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Up | v3 5k | 4.35 | 3.71 | 0.95 | 0.219 |
| Up | v3 15k | 3.69 | 2.86 | 1.25 | 0.189 |
| Up | v4 5k | 3.71 | 4.15 | 0.17 | 0.208 |
| Down | v3 5k | 5.53 | 4.49 | 3.61 | 0.190 |
| Down | v3 15k | 4.73 | 5.31 | 2.10 | 0.297 |
| Down | v4 5k | 3.97 | 4.50 | 1.59 | 0.205 |

MPJPE is averaged over three windows per direction; other columns average the
six per-window/per-foot P95 values, not a pooled percentile. Known-patch
penetration has prediction-dependent spatial coverage. Source-timed speed is
reconstruction agreement on fixed material-point masks, not actual inferred
load-bearing slip. The new labels improve reconstruction and penetration but
do not solve floating or sliding. The separate 39-window v4 context1 review
reports 5.78 cm world and 4.72 cm root-relative MPJPE.

### Playback alternatives that did not fix the problem

The original runtime adapter in `native_mb_smart_object.py` calls
`full_navigation_agent._predict_inbetween_motion_features` directly. Native
Root duration prediction and the fixed-source-clock baseline use the same
sparse authored poses and backward terminal derivative, with generated history
only after the initial seed. Root-predicted duration did not consistently
improve quality and increased ascent jitter.

The shipped preview's default `PRED_OFFSETS=4` reads four UNPLAYED buffer
frames as context and restarts emission at regenerated frame zero. The earlier
stair diagnostic instead used four played frames and discarded regenerated
frames 0-3. A separate lookahead experiment tests the former cursor semantics
with late soft advancement before landing goals. It preserves a tested
absolute-frame ledger and does not blend or replay frames. This schedule is
NOT the preview's periodic replanning policy. Intermediate authored goals may
be overwritten and are not hard reach guarantees.

On the existing Root15k/Pose15k model, all six clips have worse mean
join acceleration with this lookahead schedule. For three ascent clips, total
acceleration P95 increases from 37.2/46.3/63.6 to 48.3/60.8/76.7 m/s2.
These are explicit all-frame smoothness diagnostics, including conditioned
frames and joins, not free-frame reconstruction scores. No playback alternative
is approved as a fix. Artifacts: `timing-r15-p15-v1` and
`lookahead-r15-p15-v1`, including source / played-context / lookahead examples.

### Frozen tokenizer and Pose comparison

The v4 tokenizer is fixed for two equal 5,000-update Pose runs: inherited
flat-50k backbone versus original initialization using the current codebook.
The latter checks the original two-stage training route, not a new architecture
or loss. A changed token vocabulary makes old Pose transfer an empirical
initialization choice; it is not guaranteed beneficial by the MB paper.

Root15k reuse is permitted only after `verify_root_contact_only_revision`
proves byte-identical splits/statistics and identical non-contact training
features, and verifies from the original feature extractor that Root consumes
none of the four changed channels. The two dataset identities remain explicit
in the bundle. This exception does not permit swapping tokenizer dependencies
under a trained Pose checkpoint. Both complete source manifests remain checked.

Both Pose runs completed 5,000 real optimizer updates with zero skipped batches
and an unchanged frozen tokenizer. Each took approximately 734 seconds and
9,871 MiB peak allocation:

- Warm transfer: `plant-v4-pose5000`, checkpoint
  `bc1ebda31117766b6d813cc454d17d6d5549b87c7433034ebee29a37a8852ed6`.
- Original initialization: `plant-v4-pose5000-fresh`, checkpoint
  `57ca47d7436662317bec57ddaa112e3b3da112604d451cd5b83c23479ceda803`.

The post-training evaluator initially assumed the old pilot's 57 windows and
five behaviors, so both launchers exited with saved checkpoints but incomplete
reviews. The evaluator now derives windows and behavior coverage from verified
manifest records. Empty validation sets, missing shards, split leakage and
nonfinite inputs still fail. Both checkpoints were independently reevaluated
in `validation-recovered-v2`; contracts retain the recovery reason and original
launcher exit code. No optimization was repeated or checkpoint rewritten.

### Complete generated sequences

The same Root15k, sparse authored landing goals, source recordings, free-frame
masks and official input packer are used for all rows. Only the first four
frames are source seed; later context and intermediate motion are generated.
The old control uses v3 tokenizer5k/Pose5k; the warm and fresh candidates both
use v4 tokenizer5k. Values are means over three recordings per direction.

| Direction | Candidate | World MPJPE cm | Root height MAE cm | All frame acceleration P95 m/s2 |
| --- | --- | ---: | ---: | ---: |
| Up | Old v3 | 15.29 | 3.11 | 49.22 |
| Up | Repaired labels and warm Pose | 11.65 | 5.89 | 62.72 |
| Up | Repaired labels and fresh Pose | 16.71 | 7.17 | 90.43 |
| Down | Old v3 | 10.20 | 3.07 | 32.83 |
| Down | Repaired labels and warm Pose | 10.69 | 5.01 | 34.37 |
| Down | Repaired labels and fresh Pose | 20.50 | 15.38 | 77.41 |

The warm candidate improves ascent world position agreement but worsens height
agreement and jitter. Descent is not improved. Fresh Pose is worse within this
short budget; this does not establish its eventual converged performance.
All are single-seed engineering comparisons, not statistical quality claims.

Flat regression against the accepted 50k bundle also fails: walk world MPJPE
is 9.72 -> 13.25 cm, run 12.73 -> 17.20 cm on the six fixed sparse-input windows.
Therefore neither candidate is approved for UE deployment, and the existing
flat bundle remains intact. These results are in `plant-v4-whole-warm`,
`plant-v4-whole-fresh` and `plant-v4-flat-warm`. `plant-v4-comparison` renders
the first up/down recordings in fixed report order with Source / Old v3 /
Repaired labels / Fresh Pose panels. The videos are generated-keyframe probes,
not independent terrain navigation or successful stair demonstrations.

### Current decision

Keep the semantic data correction, but reject the new checkpoint as a runtime
upgrade. Do not respond to the failed generation gate by overwriting decoded
roots, locking feet or adding blending. Tokenizer reconstruction improvement
alone does not establish a good generative model. Runtime timing alternatives
also failed, and restarting Pose did not help at the tested budget.

The released Root/Pose curriculum still permits only one keyframe per end at
these saved steps (`max_num_keyframes * global_step / 200000`, floored and
clamped to at least one). Weights-only transfer restarts this training clock;
it is not a completed multi-keyframe curriculum or the paper's full training
run. Four-context stress results must not be mistaken for in-distribution
quality. Completing the original curriculum with a fixed vocabulary is a
remaining training requirement to investigate, not an already-proven cure.
The stair training set still has only 50.8 seconds from one performer, with
weak observed support patches rather than complete stair geometry.

Focused regression suites: 172 stair/native tests plus 12 control-review tests
passed locally. No model/loss changes or UE asset changes were made by this
iteration; no training job is left running. Checkpoints remain on g3, with
contracts, metrics and selected comparison videos mirrored locally.

### UE inspection scene (not live stair navigation)

An isolated review map now exists at
`/MotionBricksGroot/StairReview/L_SomaStairReview` in
`F:/dev/GASP_57/GASP_57.uproject`. Launch it with
`F:/dev/GASP_57/Scripts/RunSomaStairReview.ps1`.
The accepted flat bundle and the original flat map remain unchanged.

- The possessed character still uses the accepted flat 50k live backend at
  loopback port 28769. WASD is live inference only in this flat area.
- Four independent Manny actors show source versus v4 warm-generated ascent
  and descent. They replay the existing held-out sparse-goal evaluation, not
  newly generated actions or autonomous terrain navigation. The generated
  fixtures retain Root15k/Pose5k/tokenizer5k identities and failed quality status.
- Review poses use native 30 Hz frame holds, without interpolation, IK foot
  fixes, root correction, or time warping. One shared seconds clock drives all
  four actors; shorter clips hold their last frame.
- Cyan outlines are weak source-derived observed support polygons. Wider
  colored slabs are illustrative, non-colliding context, NOT measured treads.
  Scene placement translates each source/model pair together, including its
  support geometry; it does not alter the source's support-relative motion.
- Enhanced Input: 1 returns to flat control, 2 selects ascent, 3 descent;
  Space pauses/resumes; Home restarts; Left/Right step native frames; F4 toggles
  the SOMA source skeleton overlay. Fixed review views suppress player intent.

`scripts/export_native_mb_stair_review.py` exports the four v2 fixtures and a
hashed manifest to `F:/dev/GASP_57/Saved/MotionBricks/StairReview/`. The UE loader
continues to support legacy v1 60 Hz source fixtures. A native-frame boundary
roundoff guard fixes exact-frame FK verification, without relaxing the existing
0.02 cm position / 1e-4 rad rotation thresholds. All four UE FK checks pass
(maximum position error below 0.000064 cm).

Verification: 3 exporter/FK tests passed locally, and
`MotionBricks.StairReview.PIE` passed with no warnings in
`F:/dev/GASP_57/Saved/MotionBricks/StairReview/session-20261007-200802/automation/`.
The PIE check covers fixture loading, rendered poses, Enhanced Input camera
selection, pause, one-frame stepping, suppression of movement in review mode,
and return to the live pawn. Screenshots: `StairReview/ascent.png` and
`StairReview/descent.png` under the same Saved/MotionBricks directory.
This verifies the inspection tool, not successful learned stair traversal.

### Live authored stair sidecar

The new maps `/MotionBricksGroot/StairLive/L_SomaStairUpLive` and
`/MotionBricksGroot/StairLive/L_SomaStairDownLive` use the actual stair model,
not the accepted flat backend or generated replay fixtures. Launch through
`F:/dev/GASP_57/Scripts/RunSomaStairLive.ps1`; port 28771 is independent of
the preserved flat service on 28769. `-Course down` selects the descent map.
The portable bundle is `out/native_mb_stairs_20261007/ue-live-v1/native-stair.pt`,
SHA256 `39b12499ea924d272e7a796b95c9aafa511175fe6c259baa1e14157eecba600c`.
Its stage identities are the Root15k / warm Pose5k / tokenizer5k above.

This implements a bounded **authored Smart Object interaction**, following
the original packer and paper section 6.2, not autonomous terrain perception.
`export_native_mb_stair_courses.py` ships four seed frames and sparse landing
goal blocks from the identified source examples, plus their weak support
polygons. No complete source animation or generated replay is loaded by the
live server. Coordinates undergo only one common planar origin translation;
source heights, rotations and support-relative geometry are preserved.

- W requests the next authored goal; S requests backtracking to earlier goals.
  A/D offset input goals within a bounded 25 cm lateral range. These inputs do
  not imply a trained free-steering controller or reliable backwards stair gait.
- Release requests a stationary authored landing target. Following the Smart
  Object `tau=0` phase policy, an active landing finishes before changed intent
  takes effect. The model continues to generate hold motion; outputs are not
  frozen, blended or overwritten. A phase is complete when its neural frames
  are consumed, not proof that the approximate network reached its exact pose.
- The Root model chooses duration. Four regenerated context frames are omitted;
  remaining generated frames are consumed without dropping the four-frame tail
  when durations do not divide into the eight-frame protocol blocks.
- UE executes full decoded XYZ once, without StepUp, floor projection, gravity
  or collision correction in this isolated kinematic scene. Mesh root factoring
  cancels duplicated vertical actor travel without changing the world pose.
  The world-space check runs after the engine's deferred movement update.
- Enhanced Input remains the gameplay input path. Home resets the course;
  1/2 select ascent/descent, mouse controls the camera, and F4 shows source bones.
  Camera lag, output pose smoothing and foot IK remain disabled.
- Illustrative treads are built around observed support levels. They are
  query-only visuals and do not demonstrate measured or complete terrain.

Verification including the final camera framing adjustment: 64 Python regression
tests passed. Both UE automation tests passed in
`F:/dev/GASP_57/Saved/MotionBricks/StairLive/session-20261007-210636/automation/`.
The PIE test covers real backend inference, both courses, Enhanced Input,
release/hold, exact root displacement and rendered world-space root factoring.
One existing source hierarchy skeletal-mesh LOD warning was recorded, with zero
test errors. Twelve-second forward trials consumed 351/360 native motion frames;
ascent had three buffer underruns and descent zero. These are playback cadence
measurements, not a guarantee of uninterrupted rendering at 30 FPS.
Actor height changed by +139.17 cm / -98.27 cm, with zero measured decoded-root
delta error and zero rendered world-space root factoring error. The default
stair camera uses a 420 cm arm and -35 degree pitch; both mid-course screenshots
were inspected. The launcher then opened the ascent map in the visible Editor.

The learned motion remains **quality_approved=false**. Contact accuracy,
interrupt/backtracking quality and generalization to other stairs have not
passed a quality gate. This sidecar closes the live control integration gap;
it does not supersede the failed motion-quality findings above. The accepted
flat bundle hash remains unchanged.

### Live floating-foot investigation

The user reported visibly floating feet in the authored stair sidecar. The
recorded input session `5C5726E74567402A7C6195B2E8701D8D` is reproduced by
`scripts/review_native_mb_stair_live_contact.py`: 291 requests, 2,328 unchanged
generated frames, and exactly 0.0 m maximum discrepancy against the logged
last root of every request. Results are under `live-contact-repro-v1`.

The native SOMA motion already has clearance and penetration errors before
retargeting. Weak footprint coverage includes both feet in only 393 frames;
17 of those have both lowest covered sole regions more than 5 cm above the
patches. This is a spatial observation, NOT a load-bearing contact label.
The remaining frames cannot be silently counted as good contact.

`SomaStairContactAudit` separately runs the actual UE retarget processor and
CPU LBS on fixed Manny sole vertices. It uses the live planar-root factoring
convention and loads actual saved map block tops without modifying the map.
The 595-frame fixture contains 13 source seed/goal samples and 582 generated
samples (every fourth frame). On those actual visual blocks, native SOMA has
120 generated samples with both sole regions above 5 cm; Manny has 122.
Neither has such a sample in the 13 authored seed/goals. Large clearances also
include departures from the finite stair footprint onto the lower entry-ground
projection; these counts do not isolate vertical pose error or support phase.
See `ue_surface_summary.json` and the full UE audit in
`F:/dev/GASP_57/Saved/MotionBricks/StairLive/contact_audit.json`.

The retarget translation probe gives a pelvis gain of 0.949083 for a +1 m
source vertical translation. This is target body-size scaling, not an upward
20 cm offset. The root helper is snapped to zero, so a passing root-helper
factorization test alone cannot establish sole placement. No retarget settings
or output pose translations were changed to hide this discrepancy.

A distinct UE startup defect was measured: the ascent walking capsule moved
the captured motion origin upward by 0.399997 cm before BeginPlay. The isolated
stair pawn now starts with custom kinematic movement, no capsule collision and
zero gravity in its constructor. This removes the initialization displacement;
it is not a per-frame correction. Flat gameplay is unchanged. Both courses now
pass the zero-origin check in `session-20261007-214645/automation/`. This small
fix does NOT resolve the model's floating feet.

The live bundle has not been replaced. A bounded original-MB Pose continuation
completed 10,000 additional optimizer updates, from 5k to 15k, restoring the
optimizer and scheduler. There were no skipped batches. The frozen v4 tokenizer
and Root15k remain unchanged. Training took 1,454.59 seconds, with 9,874.60 MiB
peak allocation and final epoch loss 0.27057 (previously 0.3548). Artifacts are
in `live-contact-pose15000-v1`; portable SHA256 is
`6ef039840c117cf35ac96dca1403aa10991f8df50e813581cbdc6a6d106c4b64`.

The six held-out recordings use FIXED source-authored endpoint timing, not
Root-predicted duration. With the same official runtime-input arm:

| Direction | World MPJPE cm, 5k -> 15k | Root height MAE cm | Acceleration P95 m/s2 |
| --- | --- | --- | --- |
| Up | 11.65 -> 10.87 | 5.89 -> 3.98 | 62.72 -> 48.87 |
| Down | 10.69 -> 8.76 | 5.01 -> 2.67 | 34.37 -> 31.27 |

These improvements do not establish contact quality. The mean per-side
known-patch penetration P95 worsens from 1.22 to 3.20 cm for ascent, and
2.18 to 2.65 cm for descent. No new comparison video was rendered.

The independent replay of the user's actual inputs DOES use the live
Root-predicted duration and controller. Its native weak-footprint both-foot
coverage is 372 rather than 393 frames, with 18 rather than 17 bilateral gaps
above 5 cm. Those different coverage sets prohibit treating the raw counts as
a like-for-like improvement score; they do establish that the candidate still
has gaps. Actual scene-top sampling gives 126/582 native SOMA and 128/582 Manny
samples with bilateral clearance above 5 cm, versus 120 and 122 previously.
Departures from the finite stair footprint are again included. The authored
source fixture is unchanged. Candidate artifacts are in
`live-contact-repro-pose15k` and UE `contact_audit_pose15k.json`.

Decision: reject this candidate as a floating-foot fix. Training loss alone,
upward actor displacement, and integration tests are not release gates.
Continuous traversal, stopping, backtracking and lateral input remain distinct
unapproved behaviors. Repeated landing INPUT poses request zero terminal
velocity, but this does not prove those dynamic source poses are appropriate
stationary targets or that the model has learned stopping on stairs. Further
work must distinguish that input-authoring issue from network error, without
adding output IK, locking, root projection or blending.

Verification: 72 focused Python tests passed across the tested suites (70
initial tests plus two added report-alignment/unknown-coverage cases). Both UE
origin PIE tests passed; both retarget audits completed. The candidate audit
recorded unrelated Mandalorian camera-asset warnings and the existing SOMA LOD
warning, with zero commandlet errors. No training or evaluation job remains
running, no candidate was deployed, and the existing visible Editor was not
restarted. The startup-origin fix is compiled and tested but is not claimed
to have hot-reloaded into that existing Editor.

### Reviewed decoder-input experiment and hold-stage isolation

The follow-up retains the original network/loss code and the accepted flat
service. Paper section 7.3 describes keeping predicted Pose tokens fixed while
interpolating the decoder's root inputs. The public preview does not implement
the interpolation utility (`smooth_root_traj` is unused). The diagnostic in
`native_mb_root_conditioning.py` is therefore an independent implementation of
that paper-described boundary, NOT a recovered official implementation.
It interpolates XZ anchor residuals before the original decoder; no decoded
pose, root, rotation, height, or foot position is rewritten. It is offline-only.

Review found that a constant residual extrapolated to frame zero is lost when
the official representation differences positions then integrates from zero.
Revision v2 retains the canonical zero origin. A real official representation
roundtrip regression covers nonzero frame-3 residuals. The v1 experiment is
preserved separately and is not the approved policy.

Same recorded user inputs, same Pose15k/Root15k/tokenizer5k bundle, 2,328 raw
frames queried against the same actual UE block tops:

| Diagnostic | Original | Decoder input v2 |
| --- | ---: | ---: |
| Both sole regions above the visible surface by 5 cm | 502 | 272 |
| Any sole region below its projected block top by 3 cm | 892 | 637 |

All frames have spatial coverage in this scene query, including the large
entry-ground slab. These are geometric counts, NOT support/force labels or
paired contact-slip scores. Generated phase timing and target counts can change.
Four context frames were separately rejected: their corresponding counts were
532 and 1,136, versus 502 and 892 for the one-context baseline.

A second fixed test holds W for 45 eight-frame requests, then releases for 30
requests. Each arm produces 600 raw frames, with independent up/down courses.
The counts below use actual map geometry, not per-prediction fitted ground:

| Course | Bilateral clearance >5 cm, original -> v2 | Below top >3 cm |
| --- | ---: | ---: |
| Up | 470 -> 477 | 102 -> 60 |
| Down | 10 -> 9 | 64 -> 56 |

Upward holding fails in BOTH arms: all 240 released-command frames have
bilateral clearance above 5 cm. The baseline drifts beyond the last platform;
v2 reduces that drift but its final right sole is still 8.4 cm above the top.
The raw SOMA final root is Y=2.39336 m versus authored Y=2.30881 m. Therefore
v2 is **rejected for deployment**, despite the partial recorded-session gain.
The exact generated phase ledger handles four-frame boundaries inside requests
and discarding unplayed hold tails; it never relabels played frames as the new
phase. Frozen replay artifacts are not overwritten by later scene analysis.

The down-course UE retarget/LBS audit completed with zero errors and one existing
SOMA LOD warning. It also reveals disagreement between some authored source
samples and illustrative block tops; the blocks are not measured terrain truth.
The earlier suggestion that descent incorrectly starts below its first target
was refuted: initial Y=1.81879 m, first goal Y=1.54342 m, with course hash
`44bd058e51ecd93cc75fa8ce5d1b6dae8c3f07123875b94d4fdb0feafb72d05b`.
`selection_window_start` is provenance, not an instruction to crop this course.

`review_native_mb_stair_hold.py` isolates a synthetic constant authored pose.
This is NOT observed standing data or proof of balance. Its oracle encodes the
entire constant sequence and has dense future information; it is never a
runtime replacement. For the actual 12-frame runtime hold duration, the old
tokenizer's source-encoded ascent-exit root-height MAE is 3.53 cm and mean foot
joint error 8.20 cm, versus 0.25 cm / 2.94 cm at the ground-level ascent entry.
A separate 64-frame stress probe gives 7.44 cm / 12.55 cm at the ascent exit.
This establishes a reconstruction limitation for these hold inputs in addition
to Pose-generation and root-path errors; it does not identify one universal
cause for every floating frame.

Artifacts: `fixed-{up,down}-{baseline,root-input}-v2`,
`live-contact-repro-pose15k-root-input-v2`, `hold-stage-v1` (64 frames), and
`hold-stage-12frames-v1`. Each scene-analysis companion hashes its original
replay/report and actual UE scene audit. All retain `quality_approved=false`.

The next bounded training experiment resumes the corrected-v4 tokenizer from
5k to 15k with the original paper-profile batch 256, model, losses, Adam and
2M-step schedule. Optimizer/scheduler state is restored; the Lightning loader
warns that its mid-epoch sample cursor is not resumable, so this is not claimed
to reproduce the uninterrupted batch sequence bit-for-bit. The old tokenizer,
Pose vocabulary and live bundles remain unchanged. Source-encoded geometry,
fixed hold probes and flat reconstruction must be reviewed before adapting
Pose to any new vocabulary. This is not the paper's completed training budget.

#### Corrected-v4 tokenizer continuation result

`plant-v4-tokenizer15000` completed 10,000 additional updates (5k -> 15k), zero
skipped batches, 887.55 seconds and 2,827.93 MiB peak allocation. Final checkpoint
SHA256: `80d101a65e381e405fb8b0c9a2752579e9421b3e9bc5c492ba96aee25ba17634`.
The configuration hash remains `d63a2719e00a29d7dbee07daa86507719c832fb54c3f8d5134fa59c96ec38c17`.
The source guard passed; recorded before/after numerical policies are identical.

Strictly paired context-1 reconstruction on 39 validation windows:

| Behavior | Windows | World MPJPE cm, 5k -> 15k |
| --- | ---: | ---: |
| Walk | 3 | 6.73 -> 7.73 |
| Run | 3 | 7.13 -> 8.45 |
| Stair up | 18 | 5.64 -> 6.68 |
| Stair down | 15 | 5.49 -> 5.65 |
| All | 39 | 5.78 -> 6.50 |

Root error rises from 3.12 to 3.88 cm. Root-relative FK error slightly improves
from 4.72 to 4.63 cm; joint-velocity error changes from 0.2805 to 0.2778 m/s.
Neither improvement cancels the world-placement regression. These are
source-encoded reconstructions, not generative or UE control scores.

The 12-frame ascent-exit stationary probe improves root-height MAE from
3.53 to 1.18 cm but worsens mean foot-joint position error from 8.20 to 9.22 cm.
The descent-entry probe worsens height MAE from 0.87 to 4.40 cm. Neither pose is
claimed to be a validated balanced standing target. The 64-frame stress results
are archived separately; do not substitute them for the 12-frame runtime case.

Six fixed stair windows also use official skin geometry and identical source
material-point masks. The following are arithmetic means over the three
windows and two feet per direction, not pooled distribution percentiles:

| Direction | Expected-support absolute gap mean cm | Expected-support below-plane mean cm | Source-mask material speed mean m/s |
| --- | ---: | ---: | ---: |
| Up, 5k -> 15k | 2.03 -> 1.88 | 0.23 -> 1.50 | 0.115 -> 0.084 |
| Down, 5k -> 15k | 1.67 -> 1.22 | 0.33 -> 0.62 | 0.095 -> 0.104 |

Reduced absolute gap is accompanied by greater below-plane error. The source
material-speed means are 0.020 / 0.011 m/s. This candidate does not solve contact
or preserve flat reconstruction and is rejected as a replacement vocabulary.
No new Pose adaptation, bundle export or UE deployment follows this failed gate.
The accepted flat50k and experimental live stair bundles remain unchanged.

Reports are mirrored in `plant-v4-tokenizer15000-context1`,
`plant-v4-tokenizer15000-geometry`, and `hold-tokenizer15000-{12,64}frames`.
Training metadata is mirrored in `plant-v4-tokenizer15000`; the full checkpoint
remains on g3. The geometry review rechecked GPU occupancy and moved to idle
GPU5 after GPU0 became occupied by unrelated work; no foreign process was stopped.

The next repair boundary is the model's reconstruction/training coverage for
real stair landing and stopping poses. Repeating a moving source keyframe has
not established a hold skill. Before another Pose run, require a vocabulary that
reconstructs those intended inputs and held-out stair motion accurately without
regressing the preserved flat examples. Do not interpret this failed bounded
continuation as evidence against the full-paper training regime, and do not
promise that more updates alone will cure it.

Review/verification: 52 focused tests pass. An independent read-only review
found no additional concrete coordinate, phase-ledger or metric defects in the
new hold/scene diagnostics. These checks establish the measurement path, not
motion quality. No job launched by this iteration remains running. No output
IK, locking, blending, root shift or height projection was added.

### 2026-10-08: scene, exit intent and UE world-height correctness

This iteration fixes input/scene/retarget integration defects; it does NOT
approve unrestricted stair locomotion. `quality_approved=false` remains set.
The user explicitly deferred performance testing while other graphics programs
share the RTX 3080. No foreign program was closed.

#### Source-compatible test geometry

`author_native_mb_stair_scene.py` now authors explicit rectangular tread tops
from source motion only. A vertex below a tread height minus 1 cm constrains
that tread's riser to lie beyond the vertex along the ascending course axis.
The selected riser is 1 mm beyond the tightest constraint. Heights remain the
weak support-height estimates; width is 1.1 m, minimum tread depth 8 cm and the
terminal platform is at least 1.2 m deep. No predicted motion places the scene.

This is weak, source-compatible geometry, NOT a recovered measured staircase
or proof of physical support. Some source sole regions overhang the edges.
The old midpoint-riser scene put source feet below visible tops by over 3 cm
in 31/144 ascent and 50/116 descent frames. The new actual-UE-geometry query
gives 0/144 and 0/116 respectively. Bilateral source clearance above 5 cm is
0/144 and 2/116. Original maps and data are retained.

New maps:

- `/MotionBricksGroot/StairLive/L_SomaStairUpLive_SourceFit`
- `/MotionBricksGroot/StairLive/L_SomaStairDownLive_SourceFit`

Course bank: `out/native_mb_stairs_20261008/source-authored-scene-v2/courses.json`.
SHA256: `ab10cc9967a69db320bbcc1ac54e32665585830da8cf60fec68323ae4d1e9a35`.
Both authoring and contact-audit commandlets accept `-MapSuffix=_SourceFit`.

#### Native exit skill and observed idle target

The final moving stair keyframe is no longer repeatedly requested as standing.
After its committed landing finishes, `NativeAnchoredIdle` supplies an observed
idle goal to the original MB smart-object inference using the accepted flat50k
bundle. History is expressed in a fixed, authored platform-height frame and
the exact same translation is inverted on return. The height is never inferred
from predicted feet. No decoded root, pose or foot is corrected. Reverse input
returns to the stair primitive; A/D shifts the INPUT idle goal within the
existing course offset bounds rather than being discarded at the platform.
Pending frame queues remain immutable.

The old flat bundle's idle reference was the LAFAN calibration T-pose
(`walk1_subject1_30hz.npz`, start 28), selected by low speed alone.
`author_native_mb_idle_reference.py` now selects observed quiet, arms-down
training windows, retaining source provenance and rejecting invalid inputs.
No LAFAN clip passed the unchanged quiet-selection criteria. A BONES training
idle did: `idle_loop_002__A063`, start 138, shard
`71a596a7ed82cc0a2bea_30hz.npz`. These are semantic target-selection criteria,
not new contact truth or training loss. Both bundles remain immutable.

Reference: `out/native_mb_stairs_20261008/relaxed-idle-bones-v1.json`.
SHA256: `e62ad0ccfa8beea4eadf6146ba72058c0d4c087cdefd6196435334bf5ce33a15`.
The reference loader validates schema, SOMA identity, train provenance, FPS,
four full feature frames and finiteness.

Stair bundle SHA256:
`39b12499ea924d272e7a796b95c9aafa511175fe6c259baa1e14157eecba600c`.
Exit flat bundle SHA256:
`a17027e3cf8fc4958437329fa91a0889324949f37458a87af36c8d9c824e963c`.
Ready logs record both model IDs, reference and course hashes. This is our
bounded MB task-actor adaptation, not NVIDIA's unreleased UE implementation.
The inference/keyframe boundary follows the original implementation and
[MotionBricks](https://arxiv.org/html/2604.24833v1), section 6.2 / appendix G.

Frozen W45/release30 inputs produce 600 raw frames per course. Counts use the
actual new scene, no output post-processing and the exact generated phase ledger:

| Course / phase | Frames | Both soles above 5 cm | Any sole below top by 3 cm |
| --- | ---: | ---: | ---: |
| Up traversal | 128 | 32 | 69 |
| Up exit idle | 472 | 0 | 0 |
| Down traversal | 124 | 41 | 15 |
| Down exit idle | 476 | 0 | 0 |

All these frames have scene coverage. These are geometric flags, not measured
contact or force labels. The narrower weak footprint coverage is reported
separately; it must not replace the scene-query denominator. In particular,
successful exit idle does not erase the traversal failures.
Artifacts: `live5k-{up,down}-relaxed-idle-v1` under the 20261008 output directory.
The idle input comparison image/GIF shows only the final two seconds and is
not a full-traversal or contact-quality demonstration.

#### Preserve world travel through UE retargeting

Epic's `PelvisMotionOp::DecodePose` scales source pelvis height by the reference
body-height ratio, approximately 0.949083 for these assets. Passing world stair
altitude in that quantity made a 1 m source elevation become only 0.949 m on
Manny. The error accumulated with floor altitude.

`UMotionBricksIKRetargetComponent::RetargetWorldRootPose` now factors source
vertical travel out before the standard retarget call, then restores that
same world translation to all target component-space bones and target local
roots. Child local transforms remain unchanged. There is no terrain query or
foot target. Only `bFullRoot3D` uses this path; accepted flat control is unchanged.

Tests on actual UE assets cover -40, +100 and +180 cm translations, unchanged
rotations and reference-pose parity. A +1 m source translation now gives +1 m
for every target bone, maximum error 0. In the SAME frozen 150-frame generated
ascent sample (old T-pose idle target), Manny below-top flags reduce 135 -> 13;
bilateral clearance flags increase 8 -> 10. This isolates the coordinate fix,
not an improvement from swapping generated motion. Authored-source ascent
samples reduce 6/13 below-top flags to 0/13.

Separate audits of the relaxed-idle candidate sample 150 generated frames per
course: up SOMA clearance/below counts 8/17, Manny 10/13; down SOMA 10/3,
Manny 3/11. Both commandlets finish without errors. These residual differences
and the one below-top authored descent Manny sample mean retargeted contact
quality is not fully approved either.

UE audit files are in `F:/dev/GASP_57/Saved/MotionBricks/StairLive/`:
`contact_audit_sourcefit_{up,down}_relaxed.json`. They use `-WorldRootFrame`,
actual map geometry and CPU LBS, independently of rendering performance.

#### Full-state training experiment: rejected, not deployed

`--continue-dataset-run` restores the original flat50k tokenizer model,
optimizer, scheduler and loop state when changing to the stair dataset.
The validator allows only explicit data/statistics/output/stop changes and
rejects changed model, loss, optimizer, schedule or numerical policy. First
batch evidence verifies step 50000, scheduler 50000 and 90 optimizer states.
This does not promise identical uninterrupted sample ordering across datasets.

`tokenizer-state-continuation55k-v2` completed 5,000 updates, 447.51 seconds,
2,826.97 MiB peak allocation and zero skipped updates. On the same 39 held-out
context-1 windows, world MPJPE rises 5.78 -> 10.44 cm, root error 3.12 -> 7.92 cm
and root-relative FK error 4.72 -> 5.48 cm versus the preserved stair5k baseline.
The candidate is rejected: no Pose adaptation and no deployment followed it.
Training metadata is mirrored locally; paired metrics are in
`tokenizer55k-context1/metrics.json`. No training job from this iteration remains.

Other diagnostics (free-locomotion exit, per-step authored input-frame
recentring and four Pose iterations) also failed their bounded comparisons
and are not enabled in the launcher. Do not use them to mask remaining errors.

#### Verification and active candidate

97 focused Python tests pass (eight upstream Torch deprecation warnings).
The real port-28771 socket probe also completes six deterministic input trials:
each course has forward, early mixed commands, and forward-to-platform followed
by A/D/S/release. Tick continuity, committed history, finite full-3D neural
frames, platform skill use and reverse skill handoff are asserted. Artifact:
`source-authored-scene-v2/live_protocol_relaxed_v2.json`. These protocol checks
do not approve the generated reverse or strafe contact quality; for example,
the ascent reversal still has a conspicuous root-height excursion.
UE normal Development Editor build succeeds. The true-asset
`ContractAndRootFactoring` and `WorldRootRetargetEquivariance` automation tests
pass. An independent review found no additional concrete defect in the scoped
UE coordinate changes and tightened continuation configuration validation.

Full rendered `UpDownInputsAndScreenshots` automation FAILED its frame-delivery
deadline under shared GPU load (153/162 frames instead of at least 330 in the
12-second phases, buffer underruns). It is not recorded as passing or as a
proven 30 FPS result. Performance is deferred at the user's request. Existing
PIE screenshots predate the relaxed idle reference and must not be presented
as screenshots of the final candidate.

`F:/dev/GASP_57/Scripts/RunSomaStairLive.ps1` defaults to the new maps, course
bank and independently hashed idle reference. Its `-NoEditor` smoke passes.
The stair service on port 28771 is the experimental candidate; accepted flat
port 28769 and its bundle are untouched. Normal user controls remain available,
but mid-stair stopping, backtracking, strafing and traversal contact have not
passed motion-quality acceptance. The next unresolved boundary is raw stair
generation and target-body contact, not a reason to add IK or root projection.
