# Pose Continuation Execution, 2026-10-05

## Scope And Basis

Train the existing MB Pose token-selection network, not a replacement locomotion
engine. Root and the complete tokenizer (encoder, decoder, codebooks and EMA)
remain frozen. Initial weights are the installed original, not either rejected
joint-root decoder candidate from the preceding experiment.

- [MotionBricks Section 5.2](https://arxiv.org/html/2604.24833v1): retain the
  released masked-token classifier and its native cross-entropy objective.
- [DART Appendix D.1-D.2](https://arxiv.org/html/2410.05260v1): train the motion
  predictor on generated histories and supervise decoded motion. This is an
  adaptation to discrete MB tokens, not a reproduction of DART diffusion.
- [VQ-VAE](https://arxiv.org/abs/1711.00937) and the
  [PyTorch hard/soft straight-through pattern](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html)
  motivate a training-only gradient surrogate for discrete choices. Actual
  deployed argmax code embeddings are the forward values. Softmax/codebook
  gradients are biased; this particular estimator is our adaptation, not an
  official MB objective. No random Gumbel noise or soft-code decoding is used
  during inference.

## Paired Protocol

Three fresh arms, 10,000 optimizer updates each, seed 2333, Adam 2e-5, batch 4.
All 152 TRAIN clips and the existing full source-derived region labels are used.
Four overlapping 64-frame prediction windows per update; each consumes eight
frames and passes its last four published frames as the next generated context.
Starts advance by 8 (0, 8, 16, 24) and require 89 genuine source frames, no padding.

- A: observed source history and native Pose cross entropy only.
- B: scheduled generated history and the same cross entropy only.
- C: scheduled generated history, cross entropy plus the existing tokenizer
  decoded-motion objective at coefficient 1. Native reconstruction, joint velocity,
  FK/sole position and sole-region coefficients are unchanged.

B/C use observed history for 2,500 updates, ramp replacement probability over the
next 2,500, and use generated history for the final 5,000. Histories are detached.
Clip, start, yaw and curriculum random draws match across arms and are hashed.
Sparse body conditions are observed history only; source endpoint root values
serve as navigation goals, never future body or ground-truth endpoint height.
Source future token/motion targets retain their dataset clock and world axes.
One XZ origin translation is shared across context and targets; floor Y is fixed.
Source contact evidence for generated context and its splice is excluded before
prediction, not selected by network outputs.

## Automatic Integration Checks

Training captures the actual runtime Root/Pose arguments rather than rebuilding
an approximate request. The hard decoder forward uses exactly the deployed code
IDs and the released padded decoder mask. All active logits from the autograd
path must be float32-close to captured inference logits (atol/rtol 1e-4). Differences
and near-tie argmax flips are recorded; tokens from the autograd path never
replace the actual runtime IDs. Padding-only token differences cannot affect the
masked decoder.

Root and complete tokenizer state digests must remain unchanged; Pose must change.
Nonfinite objective/gradients stop the run. Original/native decoding equivalence,
decoded-objective gradients into Pose, frozen states, hidden-future isolation,
hook cleanup and eight-frame handoff history have regression coverage.
`UE_PUBLISH_FRAMES` is shared by the Python live server and training contract;
this trial rejects a value other than 8. Legacy four-frame offline experiments
remain available but are not passed off as the current UE handoff protocol.

## Evaluation And Deployment Gate

Re-evaluate Original/A/B/C with identical code and eight-frame continuation:
57 held-out windows, source-root diagnostic separated from native Root/Pose,
identical source material-point masks and scoring range. Raw control comprises
three 60-second sequences with different observed idle histories, without future
body truth or source resets. Idle, turns, start/stop, run/walk, backward and
strafing errors stay separate. The histories are not independent training seeds.
The old delay-12 four-frame counterfactual remains a labeled diagnostic, not a
measurement of the actual UE protocol or deadlines.

Persist metrics before rendering. Produce a full 60 fps skeleton comparison and
official-SOMA surface evidence; report reconstruction, root error, contact geometry,
material-point slip, boundary and worst-joint jumps. Verify trained checkpoint
hashes, source identities, effective masks, frozen networks and sampling before
crediting an improvement. No candidate is automatically deployed on mean gains.
Benchmark the unmodified inference/FK/CPU-packet path on local RTX3080 with the
same eight-frame argmax contract; distinguish throughput from actual PIE frame time.

No IK, foot locking, output blending, pose/root filters or camera lag. No old data,
default model or UE asset is overwritten to make a candidate appear accepted.

## Execution Paths

- g3: `${PRIVATE_STORAGE}/motionbricks-preparation/GR00T-WholeBodyControl/motionbricks`
- Data: `${PRIVATE_STORAGE}/motionbricks-bones/pilot-v7`
- Complete labels: `out/sole_region_repair_v3/labels_train`
- Completed eight-frame smoke: `out/soma_pose_rollout_20261005/smoke_C_v3`
- Current paired trial: `out/soma_pose_rollout_20261005/trial_v3`
- `trial/` retains the stopped active-token equivalence failure; `trial_v2/`
  retains the intentionally interrupted four-frame training. Neither is a final
  training result. Formal conclusions belong to the completed trial report.

GPUs 3/4/5 were checked idle by the launcher before use. User authorization allows
freshly verified idle g3 GPUs without further approval; occupied cards are not used.
