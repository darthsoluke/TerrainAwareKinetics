# P2 closed-loop preparation

2026-10-22. Continue the existing approved full-PFNN direction; PRO6000 GPU3 only.
No P2 optimizer updates, GPU preflight or smoke have been launched.

## New foot evidence

`review_pfnn_frame_gait.py` reuses the saved P1 trajectories. It runs CPU FK and
official sole-region skinning only, with zero model inference and zero training.
It checks the original192 validation and92 train-terrain windows. Source stance
is `certified-source-velocity-intervals-v2`: known surface, both interval endpoints
within2cm, source material-point speed at most0.1m/s. Predictions cannot alter masks.

| Validation metric | Initial teacher | Trained teacher | Trained closed |
| --- | ---: | ---: | ---: |
| Foot-span/source-span median |0.978 |1.003 |0.913 |
| Foot-velocity/source-velocity RMS median |0.974 |1.018 |0.847 |
| Mean sole speed on source stance, cm/s |3.66 |8.75 |42.36 |
| Mean stance gap error, cm |0.288 |0.231 |3.772 |

Source stance speed averages2.48cm/s. The same480849 certified stance material-point
intervals are used for all validation variants, with zero unknown predicted stance
points. These are three-dimensional sole speeds, not just horizontal sliding.
Trained teacher foot span is preserved, but its interframe sole motion worsens.
Closed-loop span ratios are also broad (P10=0.531,P90=2.844); a preserved median
does not establish reliable gait on every window. Train-terrain diagnostics show
the same teacher stance regression (3.79->7.85cm/s) with span median1.004.

This supports proceeding with P2 preparation while keeping stance velocity and
height as separate quality gates. P1 remains unapproved for control/deployment.

## Implemented local objectives

`motionbricks/data/pfnn_frame_closed_loop.py` is separate from the P1 objective.
It does not modify the deployed actor or the immutable remote P1 source.

- Root target: factual source-next minus source-previous displacement. A generated
  Root XZ offset does not create a hidden absolute-position pursuit target.
- Boundary velocity: generated-next FK minus actual generated-previous FK, compared
  against factual source-next minus source-previous FK. The actual boundary no
  longer cancels from both sides of the loss.
- Path: compare predicted offsets from the current generated Root with factual
  source displacements at the existing0.1--0.8s stations. Tail stations without
  real labels remain excluded. Labels do not determine observation coordinates.
- Mixed history: choose teacher or generated history per sample for the whole
  rollout. Recurrent memory persists; gradients pass through generated history.
  Both sensing passes remain fresh and the fine query follows the predicted path.
- Keep existing Root/body/rotation/contact/path weights and scales. This module
  adds no inference-time IK, correction, filtering, or new geometry objective.

Eight CPU tests passed: translation-invariant source displacement, sensitivity
to the actual boundary, detached source labels, relative path/tail handling,
window-bound violations, mixed-history selection, persistent state and gradient
flow through earlier generated frames. Full16-frame GPU/backward memory and
runtime proofs are still required.

## Draft budget and launch gates

`configs/pfnn_frame_actor_p2_pro6000_gpu3_draft.json` proposes global256/micro128,
10 full coverage epochs at most,602900 windows, and a12h single-GPU wall cap
(12aggregate GPUh, within the original24GPUh P2 phase ceiling). Wall/exposure
caps take the first reached. The proposed schedule uses4/8/16-frame rollout and
0/25/50/75% generated rows while retaining teacher rows. Maximum supervised
next-frame targets sum to7234800 if all epochs finish. This is a draft, not a
launch receipt or additional resource authorization.

Initialize from the saved P1 endpoint (1652updates/422030windows), with fresh
P2 AdamW. Do not recover the unsaved P1 tail or conceal its consumed exposure.
Record P2 and its smoke separately from P1/P0/the old38.4M campaign.

Remaining work before execute:

1. Integrate the new objective and explicit curriculum into a source-bound P2
   runner; deterministic row assignment, partial batches and exact transition
   accounting must be checked. Preserve atomic checkpoint-before-progress order.
2. Reserve shutdown/checkpoint time inside the external wall cap and propagate
   the actual launch deadline, instead of starting an independent late timer.
3. Use fixed validation identities and compare corrected-source and closed-loop
   metrics at a fixed horizon. Keep original P1 reports as separate evidence.
4. Freeze the new plan/source/runtime identities; scan all GPUs and select only
   emptyGPU3. Zero-update proof must cover the16-frame generated-history case,
   finite gradients, unchanged weights and actual memory/time.
5. Run the separately accounted two-update smoke only after that proof passes.
   Then assess the documented P1 entry evidence and P2 execution gates. No
   automatic retry/resume/expansion or promotion.

## Evidence

Remote CPU report: `runs/pfnn_frame_gpu3_v1/p1_gait_review_v1` under the PRO project.
Archive184221bytes, SHA256
`748ec1071603088fc2dfb64a21b6cf017fbfa9045a427afb1e5f9c72a3cc4ab8`.
It contains all per-window and grouped metrics plus the exact analysis source.
Private-Drive upload/download completed and the local archive hash matches.
Local root: `out/pfnn_mixed_20261015/pfnn_frame_actor_20261022_v1/p1_gait_review_v1/`.
The single SSH session45034 was closed after the CPU review and upload completed.
