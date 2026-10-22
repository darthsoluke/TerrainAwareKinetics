# Causal terrain playback and height failure review

## Scope

SDF is deferred. The deployed PARC heightfields, SOMA representation, native MB
models, weights and losses are retained. Isolated decoder experiments below do
not replace the deployed model. No IK, foot locking, smoothing,
Root replacement, ground projection or invented future height/pose is added.
The deployed portable bundle still has 1,000-update online Root/Pose weights.
It is experimental and does not pass terrain traversal acceptance.

## Confirmed baseline failure

`out/parc_training_20261009/online_rollout_review_v1/baseline.json` records an
independent websocket session: six idle and eighteen straight blocks, eight
native 30 Hz frames each. Every block was played before its history ACK.

The initial pelvis Y is 0.9627 m; the last decoded Y is 9.7533 m. The maximum
between-block Root step is 2.6092 m. Ground-relative pelvis gap is measured only
at known in-bounds source cells; it is not foot clearance or contact evidence.
This is a failed rolling motion sequence, not an approval from lower loss or FPS.

At fresh generation 16, Root predicts Y=1.1369 m at the eighth published frame,
while final decoded Y=2.2844 m. At generation 19, those values are 3.4733 and
5.7159 m. This makes final decoding a concrete source of height disagreement;
after its result is actually played, the changed height also enters subsequent
Root contexts. It does not establish that token selection, codec reconstruction
or Root prediction is the sole cause.

The shipped codec's external condition excludes hip height. Our terrain branch
passes Root height to Pose token prediction, but not as a dense external Y
trajectory to the frozen codec. This dependency needs controlled reconstruction
and paired-checkpoint rollout assessment before changing representation or
training. Do not replace decoded Y with Root Y as a shortcut.

## Runtime changes

The inference loop follows the buffer principle in
[MB Algorithm 1 and Appendix G](https://arxiv.org/html/2604.24833v1#A7).
The websocket implementation below is an engineering adaptation, not the
unreleased UE runtime and not proof of original-demo quality.

After two frames of a published eight-frame block have actually played, the
client can request `prefetch`. Four context frames are reconstructed from the
previous two actual observations and the two newly played frames. The existing
four-played-frame prefix remains supported and tested.

If the command is unchanged and compatible native future exists, prepare the
next eight frames from that same generated buffer. Otherwise, return a
`prefetch_deferred` response without a binary frame packet. After the full
playback ACK, the next normal plan uses the actual final four played frames.

The earlier prefix forecasting policy is superseded: it generated a new
candidate from the midpoint, discarded six/four predicted overlap frames, and
joined the remaining candidate onto a different immutable motion. That aligns
time but not the state at the join. The self-review below measures the defect.
No committed frame is skipped, altered or used as an unplayed observation.

The prepared continuation is queued, not committed history. Only full ACK of the
current block installs the cached continuation's buffer. The browser cannot
promote the queued block before that ACK. Resets discard the pending queue,
including a late deferral reply, and pauses do not advance history.

The playback timer retains its 30 Hz phase instead of resetting to `now + dt`
after every late callback. Severe overruns rebase wall time without skipping
motion or producing catch-up bursts. Raw frame values remain unchanged.

## Encoder reuse

Within one no-gradient/evaluation prediction, Root and Pose have separate
ephemeral terrain-token caches. The four Pose iterations share the same current
observation encoding. A new prediction creates a new cache; it never reuses an
old command/scene snapshot. Training bypasses caching and retains gradients.

CPU complete native inference tests require bitwise output parity and four vs
one actual Pose encoder calls. The real RTX3080 baseline and cache-only
24-block rollouts have exactly equal published Root float values. Timings are
single short runs with cold-start effects; they do not establish a general
speedup or 30 FPS approval.

## Acceptance

`scripts/check_soma_prefetch.py` deliberately delays a full ACK and checks that
already prepared frames do not play early. It separately records actual raw
playback intervals, timer overruns and idle/moving FPS. Desktop/mobile TPS and
diagnostic overlay checks remain separate. Thirty FPS presentation does not
approve motion, contacts or parkour.

The historical, now superseded prefix-two trial recorded 30.006 idle and 30.003 moving
FPS, 37.905/37.700 ms P95, and no intervals above 50 ms or below 20 ms. The prior
prefix-four trial recorded 27.829/26.432 FPS and did not pass the timing gate.
The first prefix-two test overlapped another browser check and retained one
98.7 ms delay; both failed and successful timing evidence are retained.
These are short trials, not a sustained performance certificate.

The isolated timing trial's final pelvis Y was still 4.600 m and the SOMA mesh
center was outside the camera viewport. Character framing and motion failure
are recorded separately from passed playback timing. Do not move the camera
vertically or move the mesh down to hide this failure.

`scripts/probe_soma_online_rollout.py` retains missing traces and unknown terrain
as unavailable instead of fabricating values. Its default paced and explicitly
offline-accelerated modes are distinguished. It never substitutes future truth.

The remote 10k continuation is now complete for the matched Root/Pose pair.
Root resumed from update 1,000 to total 10,000; Pose performed 10,000 additional
updates with that frozen Root dependency. Neither it nor the isolated dense-Y
decoder candidate is approved or deployed. Current height failure remains open.

## Mechanism Review Update

### Played state is not a redundant decoder vector

Rendering uses decoded Root XYZ and SO(3) rotations followed by official FK.
The old feedback also copied independently decoded RIC positions and velocity
channels, which can disagree with that played skeleton. `reencode_played_history`
now reconstructs those redundant channels with the official MotionRep forward
conversion, while preserving rendered positions and rotations within 2e-5.
Contacts remain unused forecast channels, not observations of support forces.

The 10k source-scene rollout's mean RIC/FK context disagreement falls from
0.1414 m to about 1.8e-7 m. Floating remains: correcting input consistency is
not proof of motion improvement. The bundled test scene has an approximately
12 m drop; it is not a staircase. Root height alone cannot establish climbing,
and out-of-bounds terrain remains unknown.

### Dense Y is an isolated hypothesis, not a fix

The experiment uses the already implemented `root_without_heading` feature
mode and existing decoder skip connections to add predicted Root Y alongside
VX/VZ. The encoder/codebooks are frozen and verified unchanged. Training uses
rigid world-Y translations, including RIC Y because that coordinate is absolute
in the official representation. No future truth enters live inference.

After 2,000 updates, eight matched windows show decoder/Root Y disagreement
0.1650 -> 0.1431 m and FK MPJPE 0.4849 -> 0.4565 m. However, current-frame Root
error worsens 0.0967 -> 0.1107 m, and rolling motion still floats. Do not deploy.
Rigid translation shifts both dense Y and known keyframes; this augmentation
alone does not prove that future-Y dependence has been learned or disentangled.

`height_conditioning_v1/decoder_height_response.json` is INVALID: an early probe
read channel 0 of the full 365-feature decoder output as if it had the target
condition layout. That channel is not Root Y. The corrected v2 report derives
Root Y at channel 3 and has a regression test with distinct yaw/Y values.
The corrected fixed-token +/-1 m counterfactual changes endpoint decoded Y by
about +/-0.10 m. This is sensitivity, not physical feasibility, and gain 1 is
not an acceptance requirement for an arbitrarily altered trajectory.

### Offline oracle factorization

`evaluate_parc_online_control.py --decoder-review` first executes unchanged live
inputs, captures the actual decoder call, then replays four source/model token
and Root-condition combinations offline. Observed keyframes and masks stay
fixed. The model/model replay must reproduce the online output within 2e-5.
Source futures are explicitly labeled oracle-only and never returned to the
controller. Crossed conditions are not guaranteed physically valid.

Eight fixed held-out windows, source scoring frames [4,63), mean errors in m:

| Tokens / Root condition | Original codec Y / relative FK | Dense-Y 2k Y / relative FK |
| --- | --- | --- |
| model / model | 0.2674 / 0.2218 | 0.2504 / 0.2211 |
| model / source | 0.2611 / 0.2131 | 0.2127 / 0.2122 |
| source / model | 0.0988 / 0.0704 | 0.0872 / 0.0655 |
| source / source | 0.0938 / 0.0591 | 0.0789 / 0.0538 |

Thus fixing only the dense Root input cannot repair the body-token prediction.
Even source/source reconstruction retains current-frame Y errors of 0.0828 m
(original) and 0.0772 m (dense-Y). This decomposition does not uniquely assign
blame to architecture versus optimization, coverage or representation.

### Training/playback boundary contract

An entire predicted segment can have a constant vertical offset while its
internal finite-difference velocities remain unchanged. In the real player,
frame 4 follows the actual observed frame 3, not its decoder reconstruction.
The existing joint-velocity loss must measure that same boundary to see the
jump. A regression test uses a coherent constant world-Y shift, rather than
an arbitrary loss coefficient or an output correction.

The explicit `played_context4` decoder adaptation uses four true observed
context keyframes and supplies their detached source FK positions to the
existing `executed_history_positions` velocity path. Existing reconstruction,
commitment, skate and joint-velocity coefficients are unchanged. This is a
runtime-matched training-distribution adaptation, not an unchanged-paper claim.
Old artifacts retain `native_random` and their old single-current-KF policy.
Composer and portable loader bind the trained context policy to inference;
turning on four-frame decoder context for old weights is not allowed as a fix.

The bounded comparison has completed with the same source pair, codebook,
PARC split, seed, optimizer, schedule and 2k update budget; only the
context/boundary contract changes. Frozen encoder commitment/perplexity values
match exactly at all 21 logged updates, but this is not a full sampling hash.
Local related regression suite: 58 passed; remote training/binding tests: 14
passed. The physics-consistent constant-offset regression proves that the old
internal difference misses the jump and the played-context loss sees it.

Eight held-out windows, dense-Y `native_random` vs `played_context4`:

| Metric (m) | native_random | played_context4 |
| --- | --- | --- |
| current-frame Root Y error, generated tokens | 0.1075 | 0.0485 |
| first-future Y step error vs source, generated tokens | 0.1102 | 0.0457 |
| future Root Y MAE, generated tokens | 0.2504 | 0.2872 |
| root-relative FK MPJPE, generated tokens | 0.2211 | 0.2217 |
| full FK MPJPE, generated tokens | 0.4565 | 0.4810 |
| current-frame Root Y error, source tokens/source Root | 0.0772 | 0.0293 |
| first-future Y step error, source tokens/source Root | 0.0788 | 0.0293 |

This repairs a real boundary mechanism but does not repair generated motion
overall. In the matched 24-block played-FK source-scene rollout, final Y is
3.5286 m (previous adapter 3.0559 m); maximum boundary step is 0.2464 m. On the
diagnostic flat plane, final Y improves 1.2825 -> 1.0513 m, but maximum Y still
reaches 1.4548 m. Pelvis heights are not sole contacts, and the synthetic test
plane with a source-scene seed is not a clean held-out flat-motion benchmark.

The candidate remains NOT APPROVED and was not deployed. The completed artifact
is on the remote NAS under `codec_played_context_2k_v1`, bundle SHA256
`786051d7abecf492950ff2e698946d12a178d41d46d249770fd402dfb5641b30`.
Reports/receipts are local in `height_conditioning_v1`:
`factorization_native_random_v2.json`, `factorization_played_context4_v2.json`,
`feedback_source_played_context2k.json`, `feedback_flat_played_context2k.json`,
`played_context_contract.json`, and `played_context_bundle_receipt.json`.

Do not increase this trial's budget just because the boundary improves. The
remaining investigation concerns generated body-token coherence, absolute-Y
information in the encoder, and feasible task intent. The oracle experiment
does not uniquely prove a defective Pose architecture: multiple motions can
satisfy the same partial goal. It does show that a decoder-only repair is not
an overall quality solution for this fixed pair and this trial.

The distinction follows the
[MB decoder and playback formulation](https://arxiv.org/html/2604.24833v1#S4.SS2).
PARC's official `diffusion/gen_util.py` and sampler also use a consistent
vertical reference for motion and terrain; this motivates testing reference
invariance, not changing our coordinate system without retraining and binding.
The official MB interaction demo supplies smart-object target keyframes, and
[PARC path generation](https://arxiv.org/html/2505.04002v1#A2) selects feasible
terrain paths. Neither establishes that an arbitrary forward endpoint over a
large drop is a valid traversal request. Our online training currently uses
expert hindsight planar endpoints. Autonomous WASD traversal needs a causal,
learned feasible-intent policy and matching data; it must not get fabricated
future heights or hand-authored future body poses under a model-prediction label.

## Personal Self-Review: Boundary Scheduling

This review used no subagents and started no training. The same verified
played-context 2k bundle was used before and after the scheduling change.
Evidence is in `out/parc_training_20261009/self_review_v1/`.

### Confirmed: the displayed join was not the trained boundary

At prefix-two prefetch, the model sees history ending at current frame 1 and
predicts frames 2 onward. The old implementation discarded predicted frames
2..7 and displayed new frame 8 after old frame 7. These frame-7 states differ.
The displayed step equals:

`new[8] - old[7] = (new[8] - new[7]) + (new[7] - old[7])`.

The second term is an extra splice displacement. The previous boundary-loss
repair supervised the observed-context boundary at new frame 2; it could not
supervise this later splice against a different prediction's frame 7.

The first fresh join measured a Root predecessor discrepancy of 0.2201 m,
mean FK discrepancy of 0.2920 m, and maximum joint discrepancy of 0.5828 m.
After deferring fresh inference to the actual full-playback boundary, those
values are 0.0217 m, 0.0471 m and 0.1063 m. The remaining errors are the neural
decoder's context reconstruction, not proof that motion quality is solved.
The first two cached blocks are identical in this comparison; subsequent
rollouts diverge, so later joins are not paired-frame quality comparisons.

`played-buffer-v2` now prefetches only continuation frames from the existing
generation. A command change or exhausted buffer returns `prefetch_deferred`
without binary geometry. The client finishes the committed block, receives its
ACK, then requests a fresh plan using its actual final four frames. A late
deferral during reset releases the outstanding request without deadlocking.
No overlap is discarded, no pose is blended, and no output is projected.

### Checked: training/runtime tensor meanings agree in sampled cases

Two fixed held-out windows at both 8 and 16 tokens were compared using the
actual frozen training Root helper and online inference. Root masks, pose masks
and observed poses agree. Maximum normalized root/velocity input discrepancy is
4.3e-6; terrain-point discrepancy is at most 2.4e-7; Root output discrepancy is
at most 9.6e-7; first-pass Pose logit discrepancy is at most 1.8e-5. Comparisons
use eval mode and highest matmul precision, including the padded 8-token path.
This rules out a large input-layout mismatch in these cases, not every possible
training or dataset defect. The expert endpoint remains a hindsight XZ/heading
goal; source future Y/pose are not supplied to the online call.

### Confirmed: whole-world height translations change model behavior

Translate the four observed frames and the entire paired heightfield by the
same Y offset. Intent is unchanged. Terrain tokens' input points differ by at
most 4.8e-7 m, with identical validity, yet predictions do not translate rigidly:

| World Y shift | Root-stage Y deviation | Decoded Root Y deviation | FK deviation |
| --- | --- | --- | --- |
| -1 m | 0.2179 m | 0.0831 m | 0.1243 m |
| +1 m | 0.0554 m | 0.0190 m | 0.0937 m |
| +3 m | 0.0777 m | 0.1275 m | 0.1663 m |

This is a representation/generalization gap, not evidence of an upstream MB
implementation bug or a sole explanation of floating. Terrain is root-relative,
while motion history and tokenizer inputs retain absolute world Y. The current
relative Root-height hint alone does not remove those absolute-Y channels.

The proposed correction is a versioned common vertical frame anchored to the
last observed pelvis: transform Root Y and every RIC Y consistently in training
windows and online conditions, retain root-relative terrain observations, and
restore the same known origin after decoding. The origin must come only from
observed history. Normalization and tokenizer/Root/Pose training must use that
same representation. Do not enable it on old checkpoints or present coordinate
equivariance as a contact/parkour guarantee. PARC's actual generator implements
such relative-height framing in `PARC/motion_generator/gen_util.py`; the frame
choice is separate from collision geometry and does not require SDF.

### Browser Verification and Remaining Work

The fixed server is running on port 28880 with the existing 1k preview weights.
The improved-decoder experiment remains undeployed. Forty-four focused tests
passed. Real headless Edge verification passed full-ACK ordering, cached-only
prefetch, WASD operation, reset with a delayed deferral reply, nonblank SOMA
rendering and no page errors. The screenshot still shows floating.

Measured playback is 24.01 FPS idle and 24.32 FPS moving. Fresh inference averages
96.2 ms and skinning 15.8 ms in this short test; maximum frame intervals reach
234/239 ms. Correct scheduling exposes the inference wait previously hidden by
an incompatible forecast. Thirty FPS remains unapproved. Performance work must
preserve boundary inputs and neural outputs rather than restore the unsafe
splice, extend blind playback, or hide gaps with pose corrections.

The actual terrain protocol uses forward queries 0.2..3.4 m, overriding the
observation class's generic 1.4 m default. No claim that this deployed model
only sees 1.4 m is supported. Step risers are implied by the box heightfield;
the observation lacks explicit vertical-face samples, overhangs and ceilings.

Reports: `contract_before.json`, `contract_after.json`, `browser/checks.json`,
and `browser/prefetch-desktop.png`. The first two reports bind the same bundle
SHA256 and separate model equivariance from scheduling measurements.
