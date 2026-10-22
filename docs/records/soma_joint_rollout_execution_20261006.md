# Coupled Root/Pose Continuation Trial

## Basis And Scope

- [MotionBricks](https://arxiv.org/html/2604.24833v1), Sections 4-5: keep the released
  Root -> root features -> Pose argmax -> conditioned decoder path. Native root
  differentiation, final-root conditioning and the installed provided-velocity
  decoder policy are unchanged.
- [DART](https://arxiv.org/html/2410.05260v1), Appendix D: generated-history training
  and decoded-motion supervision motivate this adaptation. Joint Root/Pose
  adaptation and the biased softmax straight-through token gradient are OUR
  experiment, not an official MotionBricks or DART reproduction.
- No architecture, tokenizer, contact thresholds, dataset, runtime scheduler,
  output correction, IK, foot locking, blending or camera-lag changes.

## Matched Trial

A optimizes Pose only; B optimizes Root and Pose. Both use native token CE plus
the SAME existing decoded-motion objective and all its existing coefficients.
Both use teacher histories for 2,500 updates, ramp to generated histories over
2,500, then fully generated histories for 5,000. Only Root optimizer membership
differs. A new full-native differentiable path is used for BOTH arms.

Each arm runs 10,000 Adam updates, lr 2e-5, seed 2333, batch 4, four overlapping
64-frame primitives per update. Eight consecutive frames are published at 60 Hz;
their last four become detached history. Each sample needs 89 genuine source
frames. Dataset/labels, clip/start/yaw/curriculum draws and source hashes are
retained; validation and test are not training data. Tokenizer/encoder/decoder/
codebooks/EMA stay frozen. Timing is supplied, as in the deployed fixed horizon.

Source endpoint ROOT constraints are a diagnostic/training navigation goal,
not future-body input or a WASD control sequence. Source future body and height
are never available to inference. Targets remain on the source world axes and
clock; only the common XZ origin is translated. Source labels do not describe
generated history or its splice; existing fixed evidence exclusions remain.

## Automatic Checks

Training invokes the same native inference implementation with autograd enabled.
A scoped decoder input hook supplies a straight-through latent with the ACTUAL
no-grad deployed argmax embeddings as forward values. Root retains its native
continuous derivative through Pose conditioning and decoder external conditions.
Hook cleanup is guaranteed even on failure. Active logits, root predictions and
decoded outputs must match the paired native reference at float32 atol/rtol 1e-4;
numerical deltas and near-tie argmax differences are logged, not hidden.

Real CUDA preflight exposed TF32 convolution and fused/no-grad attention
differences. BOTH arms use TF32 off, deterministic convolution and a scoped
unfused MHA/math SDPA reference for training parity. Attention settings are
restored even on failure. No threshold was relaxed. Held-out and free-control
evaluation retain DEFAULT deployed kernels, independently of this training
reference. The initial failed smoke/full attempts are retained, not resumed.

Tests cover final motion and official sole-slip gradients into Root/Pose, no
target gradients, empty evidence, future-body/height isolation, exact total-loss
composition, frozen tokenizer, failed-hook cleanup and adapted-root serialization.
Every optimized network must receive finite nonzero gradients; A's Root receives
none. Full tokenizer state hashes must remain unchanged; only B's Root changes.

## Evaluation And Decision

Use fresh Original/A/B evaluation with identical 57 held-out windows, source
contact evidence and scoring range. Record actual non-regression per window,
not only aggregate means. Run three fixed 60-second WASD-like sequences with
no future goals, body truth or source resets; these are different initial histories,
NOT multiple training seeds. Report sliding, penetration, MPJPE, boundary and
worst angular jumps, command tracking, idle, stops, turns and reversals separately.

Persist metrics before rendering. Render complete raw skeleton comparisons and
official SOMA skin excerpts, without processing. A candidate must improve both
sliding and continuity without sacrificing reconstruction, control or stops to
qualify for UE review. No speculative absolute quality threshold or automatic
default-weight replacement. If the quality gate fails, retain negative results
and do not deploy the candidate as fixed locomotion.

Actual UE acceptance remains 30 fps active rendering with the unchanged native
60-Hz clock. Only qualifying candidates proceed to isolated, SHA-pinned PIE tests.
Freshly query g3 GPU occupancy immediately before smoke and formal launch. Busy
cards are never used. Independent paired arms may run on separate idle GPUs.

## Completed Result

Both formal arms completed 10,000 updates. A took 2,652.2 seconds (44.2 minutes),
B 2,953.2 seconds (49.2 minutes), running concurrently on separately verified
idle cards. Sampling hashes and generated-history counts match. Full tokenizer
state is unchanged; A's Root is unchanged and B's Root changed as intended.

The complete fixed-window and three-history evaluations reject BOTH candidates.
B lowers mean command error from 0.362 to 0.280 m/s, but measured sole slip only
changes from 0.272 to 0.265 m/s. MPJPE rises from 45.3 to 48.3 mm and long-rollout
boundary position steps worsen. At source frame 1512 (25.2 seconds), B's Hips
rotation changes by 166.3/166.4/173.0 degrees in the three initial histories.
This is a repeatable observed failure, not proof of its underlying cause.
No UE/default model promotion, post-processing or additional training follows.

Four sequential primitives per update, native parity references, decoder/sole
gradients and full-precision reference kernels explain the workload. An independent
two-update profiler also records many small launches and host/device synchronizations.
That profile includes startup/saving and cannot establish steady-state percentages.
Future execution optimization should batch logging/check transfers, not remove
checks or change training mathematics inside this matched experiment.

## Subsequent Supervision Repair

The 10k experiment above used a defective continuation target. It substituted
generated history for the first four source frames before differentiating the
target. A 0.10 m history offset therefore introduced a fictitious 6 m/s target
velocity at the splice. Its recorded motion failures remain real, but they do
not establish that correctly supervised joint adaptation is ineffective.

The corrected default preserves continuous source targets and uses generated
history only as conditioning. Prediction velocity starts at the last actually
executed pose, not the decoder's reconstructed context. Rotation increments use
continuous source increments. Global source geometry retains its displacement
relative to the observed input origin; local integration must not erase that
error. Context-only reconstruction/contact terms are excluded. Legacy splicing
is retained only behind an explicitly recorded comparison policy.

This follows the history/future separation in the official
[DART trainer](https://github.com/zkf1997/DART/blob/main/mld/train_mld.py).
The MB straight-through adaptation and rotation-increment coefficient remain
experimental, not official DART training.

## Three Bounded Follow-up Comparisons

Each valid comparison ran 1,000 optimizer updates per arm on freshly checked
idle g3 GPUs, with matched sampling, initial weights and source labels. Each
uses 57 held-out windows and three 60-second deterministic control rollouts.
These are one training seed and three initial histories, not three seeds.

| Candidate | MPJPE mm | Source-masked slip mm/s | Long boundary step P95 mm | Long worst angular step deg |
|---|---:|---:|---:|---:|
| Original installed weights | 45.294 | 272.104 | 60.286 | 31.383 |
| Legacy splice, 1k | 46.666 | 267.169 | 76.770 | 126.299 |
| Corrected source supervision, 1k | 46.390 | 270.087 | 73.187 | 31.586 |
| Corrected supervision plus decoder training, 1k | 53.494 | 273.033 | 89.728 | 33.900 |
| Corrected supervision plus observed FK inputs, 1k | 47.150 | 284.064 | 70.716 | 49.173 |

The corrected-supervision frozen-decoder arm repeats exactly across the three
paired experiments. The decoder comparison changes only decoder optimizer
membership. The observation comparison changes input geometry in BOTH training
and inference, keeping the complete tokenizer frozen. All use the same
provided-velocity root-output policy. No candidate passes joint continuity,
contact, reconstruction and command non-regression. No UE model was replaced.

Artifacts relative to the MotionBricks directory:

- `out/soma_supervision_20261006/pilot/REPORT.md` and `summary.json`.
- `out/soma_supervision_20261006/decoder_pilot/REPORT.md` and `summary.json`.
- `out/soma_supervision_20261006/observation_pilot_v2/REPORT.md` and `summary.json`.
- `out/soma_supervision_20261006/pilot/render_worst/rollouts_full.mp4` is a native
  60 Hz two-second matched excerpt around the legacy arm's worst event, not a
  complete quality demonstration. No motion correction is applied.

## Observed State Contract

Official MB `full_navigation_agent._process_input_to_joint_transforms` observes
actual joint transforms before constructing conditioning features. Our old path
instead fed redundant predicted XYZ channels back alongside FK rotations.
An opt-in observed-FK adapter now reconstructs inputs with the official
representation without modifying published poses. The input policy is saved in
the bundle; loading a switched policy without matching training is rejected.

The first adapter attempt missed initial-heading restoration in official feature
extraction. `observed_history` and the intentionally terminated
`observation_pilot` are INVALIDATED, not evidence against this method. Their
artifacts are retained and marked. Nonzero-yaw and translated-world invariance
regressions cover the correction. Only the `_v2` results above are valid.

With original weights and no adaptation, the corrected observation-only change
also gives mixed results across the three histories. The default remains
`predicted_features`; an observed-FK policy is not promoted as a quality fix.

## Current Decision

353 local tests pass, including actual native-forward gradient parity, continuous
source derivatives, executed-boundary gradients, source-label detachment,
coordinate invariance and matched-report rejection. Tests establish contracts,
not natural movement. There is still no evidence of solved sliding or twitching.

Do not extend these failed pilots to 10k by default, change UE/camera behavior,
or add IK/output filtering. The next learning experiment must separately verify
the decoder's source-token/source-trajectory reconstruction and the committed
prefix continuation objective before another long joint run. The current training
unroll publishes only 4 x 8 frames (0.533 seconds) per sample and supervises a
longer 64-frame prediction. Whether executed-prefix supervision and longer
generated-history recovery improve quality is still an untested hypothesis.
