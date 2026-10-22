# Reactive Pose Context Audit

2026-10-21. Read-only source audit of frozen final `0f5d8e2b` and native
8-token / 4-pass / 4-history / 24-playable-future execution. Only this report
was added. No model forward, GPU use, browser operation, training, filtering,
or runtime edit was performed. Decoded-context integration origin is excluded
because it is being investigated separately.

## Matched 600-Frame Results and Candidate Choice

The completed reports supplied during the audit are:

- `out/pfnn_mixed_20261015/reactive_integration_probe_20261021_v1/report.json`
  (Urban, eight cases).
- `out/pfnn_mixed_20261015/reactive_integration_flat_20261021_v2/report.json`
  (flat64-probe, six cases).

Both bind the same final checkpoint and zero optimizer updates. Each case uses
600 frames: forward180, stop120, strafe120, reverse120, final-stop60. Moving
commands have magnitude1.2m/s and heading0. Reported command velocities/speeds
exclude the first30 frames of each phase. These are synchronous prefix-consumption
rollouts (`scripts/probe_reactive_integration.py:54-81`), so they isolate model
execution intervals from the asynchronous cadence bug described below.

### Flat: Original4/8 Preserve More Gait Than Original1

Values below are settled signed command-axis velocities, stop speed, global
Root acceleration P95, and mean leg speed relative to Root. The leg measurement
was computed read-only from the saved NPZ joints during this audit.

| Case | Forward VZ m/s | Strafe VX m/s | Reverse VZ m/s | Stop / final-stop speed m/s | Root accel P95 m/s2 | Forward leg speed m/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original1 | 0.674 | 0.161 | -0.938 | 0.173 / 0.132 | 8.469 | 0.647 |
| original4 | 0.888 | 0.166 | -0.799 | 0.037 / 0.039 | 5.432 | 0.938 |
| original8 | 1.039 | 0.017 | -0.958 | 0.032 / 0.032 | 4.242 | 1.069 |
| anchored1 | 0.683 | 0.796 | -1.249 | 0.049 / 0.071 | 4.623 | 0.715 |
| anchored4 | 0.887 | 0.178 | -0.800 | 0.031 / 0.027 | 3.922 | 0.938 |
| anchored8 | 1.040 | 0.022 | -0.963 | 0.029 / 0.022 | 3.483 | 1.069 |

The one-frame prefix has weaker forward progress and leg motion even on flat
ground. Anchoring improves original1's stop and lateral/reverse behavior, but
its forward velocity stays near0.68m/s. Its forward leg speed is33% below
anchored8. A smaller Root jump therefore does not establish recovered forward
gait.

For four/eight-frame execution, anchoring preserves the forward gait measures:
leg speeds remain0.938/1.069m/s and forward Root speeds remain0.887/1.040m/s.
Root acceleration P95 falls from5.432 to3.922 for four frames and4.242 to3.483
for eight; planar acceleration P95 falls4.495 -> 2.622 and3.786 -> 2.536m/s2.
The main effect here is improved translation continuity, not stronger walking.
Eight-frame consumption advances farther through a single prediction and gives
the strongest forward progress and lowest acceleration of these flat cases.

Flat strafe is a separate failure: anchored8 moves only0.022m/s along requested
X, and anchored4 only0.178m/s, against1.2m/s. Anchored1 reaches0.796m/s in that
phase. The eight-frame candidate wins forward gait/continuity, not every
directional command.

### Urban: Anchoring Helps Jumps; First1 Remains Weak Forward

| Case | Forward VZ m/s | Strafe VX m/s | Reverse VZ m/s | Stop / final-stop speed m/s | Root accel P95 m/s2 | Max body-relative joint step m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original1 | 0.723 | 0.845 | +0.045 | 1.875 / 1.632 | 57.947 | 0.417 |
| original4 | 0.708 | 0.392 | -0.057 | 0.128 / 0.032 | 49.914 | 0.254 |
| original8 | 0.734 | 1.124 | -0.304 | 0.110 / 0.156 | 31.758 | 0.356 |
| anchored1 | 0.260 | 0.875 | -1.293 | 0.272 / 0.081 | 6.735 | 0.122 |
| anchored4 | 0.701 | 0.625 | +0.023 | 0.061 / 0.041 | 26.202 | 0.146 |
| anchored8 | 0.730 | 1.230 | -0.393 | 0.079 / 0.114 | 20.774 | 0.145 |

Anchored1 suppresses large jumps, while forward velocity falls to0.260m/s and
forward leg speed is0.374m/s versus anchored4/8's0.809/0.840m/s. The latter two
retain similar forward gait; eight frames gives a much better Urban strafe
response and lower Root acceleration than four. Urban reverse remains weak for
eight frames and is in the wrong signed direction for four.

**Candidate judgment:** anchored8 is the strongest next deployment candidate
for preserving forward gait and reducing jumps, with the cadence/origin fix
retained. Anchored4 trades more frequent adoption for weaker forward progress
on flat and poorer Urban strafe/reverse. First1 should not be selected merely
for its smoother anchored Root. The current `STEP_CHOICES=(4,2,1)` does not yet
expose8 (`reactive_kinematic.py:14`); selecting this candidate requires the main
implementation to support an eight-frame execution interval. This audit makes
no runtime edit. The flat lateral failure and Urban reverse/stop residuals
remain part of the candidate's measured behavior.

The leg measure is the mean, over forward indices30..179, of the per-frame RMS
speed across Left/Right Leg, Shin, Foot and ToeBase, after subtracting Root XYZ.
Joint IDs22..29 come from the supplied SOMASkeleton30 declaration at
`out/vendor/kimodo/kimodo/skeleton/definitions.py:206-237`. This removes Root
translation and measures actual saved leg activity. It does not count completed
gait cycles. Endpoint anchoring's implementation/origin diagnosis remains with
the main investigation; this section consumes its matched output evidence.

### Input-Sequence Cadence Bug: Fixed by the Main Investigation

The earlier `due()` allowed any changed input sequence to bypass
`step_frames`. Exact duplicate suppression in the client did not cover small
analog changes. A selected four-frame interval could consequently behave like
first1 consumption under continuously changing input.

Current `motionbricks/data/reactive_kinematic.py:85-99` removes that bypass and
requires the next planning origin to be at least the latest accepted origin
plus `step_frames`. This also protects cadence when measured lead shrinks.
`tests/test_reactive_kinematic.py:186-204` sends ten changed intents per played
frame, checks dispatch stays at frame4, and checks the next origin advances by
at least4. The main investigation reports its tests passing; this audit read
the fix and test without rerunning them. This is a concrete scheduling
correction, independent of codec target conditioning or integration origin.

## Findings

**The concrete mismatch is between a fresh, terminal-conditioned motion window
and repeated execution of its earliest prefix.** A call predicts a new window;
it does not advance a retained gait state. Executing 1 or 2 frames also changes
the four-frame latent grouping before the previous future token finishes.
The source establishes these mechanisms. The matched results above show that
first1 forward gait remains weaker after anchoring, while4/8 retain substantially
more leg activity; their relative contribution is not separated by those trials.

**A four-pose codec target-mask mismatch was ruled out for this artifact.**
History reaches both Root and Pose. Its presence is conditioning, however,
rather than a hard cross-call constraint on the next generated pose or token.

### 1. Terminal velocity is repeatedly deferred beyond the executed prefix

Evidence:

- `motionbricks/data/unified_training_goals.py:11-26`: the training command is
  source interval **31 -> 32** VX/VZ and source frame31 heading, for a 32-frame
  output with four history frames.
- `motionbricks/data/unified_full_training.py:28-36,42-47,57-64`: every training
  example starts with source frames0..3 and uses exactly eight tokens.
- `motionbricks/data/native_mb_online_conditions.py:68-83`: only sparse slot7
  receives desired terminal VX/VZ. History velocities cover intervals0..2;
  interval3 -> 4 is correctly unknown.
- `motionbricks/motion_backbone/inference/motion_inference.py:299-305` and
  `motionbricks/data/native_mb_online_conditions.py:112-117`: the explicit
  velocity substitution is at `lengths - 1`, index31 for eight tokens. It
  conditions Pose/codec; it is not a command imposed on every future frame.
- `motionbricks/data/reactive_path_execution.py:99-101` rebuilds an eight-token
  goal for every call. `motionbricks/data/reactive_kinematic.py:279-293` publishes
  only indices4..27, excluding the terminal index31 entirely.

From last history index3 to terminal index31 there are 28 frame intervals,
about 0.933 seconds. A 1-frame execution takes index4; a 2-frame execution takes
indices4..5. Replanning then places the terminal target another 28 intervals
ahead of the new history endpoint. Even the full 24-frame playable segment ends
four frames before that terminal state.

This permits a low-speed prefix followed by later acceleration to recur on each
new window. The same moving deadline permits continued prefix movement under
a zero terminal-speed command. The model receives an endpoint-speed request,
while the short-step controller expects an immediate speed policy. Changing
`step_frames` never changes this command meaning.

**Minimal correction within frozen weights:** preserve and advance the adopted
native forecast while the intent is unchanged, instead of restarting its
earliest prefix at each requested update. Existing buffering in
`motionbricks/data/native_mb_online_live.py:407-415` already expresses this
operation. This reduces repeated deferral while keeping genuine model samples.
Moving the target to index4 or27, scaling velocity by the execution interval,
or changing the token horizon would change the trained conditioning contract.
There is no source-supported one-line conversion from this terminal model to an
immediate framewise speed policy. A zero terminal command also supplies no exact
stationary-pose constraint.

### 2. Every call restarts token inference; 1/2-frame execution reblocks phase

Evidence:

- `motionbricks/motion_backbone/inference/motion_inference.py:341-352`: each call
  allocates all pose tokens as `POSE_MASK_ID` and supplies the first four poses.
- The same file, `367-399`: four refinement passes sample a new set of token
  IDs. No previous forecast IDs, absolute phase, or token offset are arguments.
- `motionbricks/motion_backbone/neural_modules/pose_backbone.py:135-167`: Root
  features are packed in groups of four frames; pose conditions replace the
  corresponding pieces of token input embeddings. Positions restart at zero.
- `motionbricks/vqvae/neural_modules/encdec.py:19-32`: two stride2 convolutions
  give a four-frame latent stride, with symmetric padding and temporal Resnets.
- `motionbricks/vqvae/neural_modules/encdec_double_cond.py:37-49,159-171`: the
  decoder uses temporal convolutions and two nearest-neighbor upsamplings.
  `resnet.py:42-43,78-83` supplies symmetric dilated temporal taps.

Let the last history frame be absolute frame T. In the adopted forecast, latent
position1 nominally covers T+1..T+4 and position2 covers T+5..T+8. These are
temporal positions, with wider receptive fields than their four-frame grouping.

| Frames executed before a fresh call | New position1 covers | Relation to old positions |
| --- | --- | --- |
| 1 | T+2..T+5 | Last3 of position1 plus first1 of position2 |
| 2 | T+3..T+6 | Last2 of position1 plus first2 of position2 |
| 4 | T+5..T+8 | Position2 grouping, newly inferred in a new window |

For 1/2 frames, a simple shift of old token IDs cannot represent the new groups.
For four frames, grouping aligns but source and decoder temporal context have
changed; equality of the overlapping forecasts is still unconstrained. This is
why reverting to four frames is not a complete correction for the reported
four-frame failure.

**Minimal correction:** retain the remaining decoded samples of an adopted
token block and advance a cursor. If replacement is required, make adoption
respect that block's absolute frame boundary. Generation frequency and adoption
frequency can be separate. For longer continuity, retain more of the adopted
24-frame forecast. This uses existing native samples without interpolation.
It fixes mid-block replacement, while the terminal-goal issue above remains.

Encoding only four played poses and pinning the resulting first token would
introduce another mismatch: training labels come from the full32-frame codec
encoder (`unified_full_training.py:187-197`), whose symmetric receptive field
includes future source frames. A four-frame encoding has a different boundary.

### 3. Four-frame history is real input, but neither a latent lock nor hard output

Evidence:

- `motionbricks/data/native_mb_online_conditions.py:159-175` retains all four
  poses and Root states with true masks. It retains the three known velocity
  intervals rather than fabricating the fourth.
- `motionbricks/motion_backbone/neural_modules/root_backbone.py:188-215` embeds
  all four start poses, local Root values and global Root values with positions.
  Thus the Root history encoder does not collapse history to the last pose.
- `pose_backbone.py:140-156` projects all four pose conditions. With all four
  known, they replace the complete first token's pose input embedding. That
  supplies pose context; it does not set the output token ID to an observed ID.
- `motion_inference.py:394-399,472-477` still predicts every token and decodes it.
- `encdec_double_cond.py:132-171` substitutes target **hidden embeddings**, then
  applies external conditioning and temporal convolution, including final
  output convolutions. There is no final copy of target poses into decoder
  output and no hard match of the next pose's velocity to played velocity.

Consequently, re-encoding played FK repairs the input representation, but each
new window can still choose a different continuation. Small motion errors can
feed back into increasingly stationary short histories or changing leg phase.
The code contains no recurrent gait state beyond those four poses.

**Minimal correction:** preserve adopted continuation state at the execution
boundary, as in finding2. Expanding the input history, changing its mask, or
adding a history-to-token constraint changes model inputs and is not an
established frozen-artifact correction.

### 4. Training uses source histories and source Root; execution feeds predictions

Evidence:

- `unified_full_training.py:28,42-47` uses factual source history and a source
  Root teacher when preparing Pose inputs.
- `unified_full_training.py:177-185` reconstructs source codec tokens with
  source external Root conditioning and first-four source pose targets.
- `unified_full_training.py:187-192` and `unified_pose_training.py:95-126` train
  Pose by corrupted source-token cross entropy, with source-history conditions.
- `motionbricks/motion_backbone/models/pose_model.py:263-281` confirms that
  objective is token cross entropy. The unified Pose branch does not decode
  repeated 1/2-frame closed-loop prefixes as its training objective.
- `reactive_path_execution.py:115-124` instead supplies predicted Root to Pose
  and codec. `reactive_kinematic.py:172,274-278` rolls played/generated states
  into the next request.

The checkpoint contract explicitly records `source_history_only=True`.
Replanning more frequently increases the number of generated-history feedback
transitions per second. Together with fresh token inference, this supplies a
specific route for phase or amplitude errors to persist without a recovery
objective on those states.

The codec does have a boundary velocity loss: `terrain_decoder_pilot.py:60-66`
stores played-history FK positions, and `vqvae/models/motion_vqvae.py:327-345`
uses them for the interval from actual history index3 to predicted index4.
`unified_full_training.py:83` removes frame/interval masks, retaining the
`executed_history_positions` field. Therefore missing boundary-velocity
supervision is not an accurate diagnosis. The mismatch is source-conditioned
training versus repeated predicted-condition feedback.

**Correction boundary:** no small inference-only change makes generated
histories into source histories or creates recovery behavior in fixed weights.
Retaining forecast samples reduces repeated feedback transitions. This audit
does not propose a new training run.

## Codec Policy and Short Backend Checks

The actual architecture bundle's pickle metadata declares
`codec_context_policy=played_context4` for tokenizer and Pose, with matched
adaptation `matched-mixed-velocity-control-v2`. Its encoder/decoder parameters
are `down_t=2`, `stride_t=2`, depth4, dilation growth3, and
`cond_fusion_last_layer=False`. These were inspected with standard-library ZIP
and pickle opcode readers, without deserializing tensor storage or importing
the model. The loader checks this policy binding at
`scripts/portable_parc_online_soma.py:62-73`.

`motion_inference.py:455-470` therefore keeps all four context target masks,
matching `unified_full_training.py:182-185`. The legacy `native_random` branch
that retains only frame3 is inactive for this bundle.

Both baseline and reactive execution re-encode played FK:
`soma_terrain_execution.py:89-95` and `reactive_path_execution.py:115-117,162-168`.
The reactive engine is not introducing the absence of this operation.

Short execution really regenerates prefixes:
`motionbricks/data/pfnn_tuning.py:163-178` creates a worker with an empty future
buffer and publishes short subpackets; `native_mb_online_live.py:407-413`
truncates the generated future to the refresh count. The offline counterfactual
at `scripts/diagnose_pfnn_frame_feedback.py:37-45` explicitly executes
`motion[:,4:4+stride]` and reobserves it.

Reactive selection is absolute-time based:
`reactive_kinematic.py:90-109,116-122,141-148`. Its lead/committed-prefix logic
means a requested interval is not necessarily the exact number of samples
ultimately played from every plan. It still has no constraint tying new tokens
to the old forecast at adoption. Numeric intent duplicates are suppressed by
`web/full_corpus_lab/app.js:118-125`; unchanged input is not repeatedly being
misclassified as a new command by that client.

## Recommended Minimal Change Boundary

The independent execution correction is **stateful consumption of an adopted
native forecast**, with explicit absolute-frame/token-block adoption boundaries.
The existing `plans`, `origin_frame`, `committed_until`, and accepted-prefix
representation provide most of the needed state. Keep receiving input promptly;
avoid treating each compute opportunity as a requirement to adopt another
fresh earliest prefix under unchanged input.

This addresses temporal reblocking and moving-prefix execution. The matched
600-frame data above now favor anchored8 as the next candidate among the tested
fixed intervals. First1 still weakens forward leg activity after anchoring;
merely selecting four frames does not resolve the reported four-frame failure.
The source findings explain why the controller must preserve actual execution
cadence and continuation state. The matched trials establish candidate behavior
without separating all causal contributions. No runtime change or model
experiment was made by this audit.

## Evidence Binding

`out/pfnn_mixed_20261015/final_acceptance_flat_v1/acceptance.json:11-23` and
`reactive_control_20261021_v3/run/service.json:6` bind the frozen final artifact
to `0f5d8e2b05b8ec518a104a4979cca70f23efa64080abfc843f874634de5a9646`.
The acceptance receipt binds the architecture bundle to
`0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d`.

The final checkpoint contains training-time source hashes. All nine of the
following current files matched those stored hashes exactly: unified full
training, unified Pose training, unified training goals, terrain decoder target
preparation, motion inference, Pose backbone, codec loss, double-condition
decoder, and codec encoder. Those comparisons and the static metadata inspection
were performed during this audit. Only leg-activity summaries of already saved
trajectories were additionally computed; no model rollouts or timing runs were
performed.
