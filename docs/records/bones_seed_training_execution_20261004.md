# BONES-SEED Training Execution

Date: 2026-10-04. The real-data pipeline and bounded pilots have run. This is not a trained locomotion release or a 60 Hz acceptance result.

Follow-up: final acceptance is UE 5.7 PIE with current Enhanced Input and the trained SOMA bundle. See `soma_ue_end_to_end_route.md`. V5 is now a failed diagnostic baseline: a later full-window review found acrobatic/interaction contamination not caught by its structural audit. A passed structural audit did not certify planar semantics.

## Implemented

- Private conda environment on taku-g3: `${PRIVATE_STORAGE}/motionbricks-preparation/envs/mb-bones`. PyTorch 2.11.0 CUDA 12.6, Lightning 2.6.1, vector-quantize-pytorch 1.28.2 and AdamAtan2 0.3.3. No shared environment or GPU driver was changed.
- Official GR00T source revision `b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`; official Kimodo revision `58e781898b3d7e328a676a75d3e338c45dce3ad9`.
- Official SOMA77 BVH parsing/reference-pose conversion, reduced to Kimodo's official SOMA30 skeleton. No G1 weights, invented 27-joint rig or UE reference pose is used in these human training runs.
- Native MotionBricks feature math, multi-head VQ-VAE and Root Transformer. The tokenizer retains official variable body keyframes; root-only evaluation supplies no unavailable future body targets. Root-only height comes from observed history. Human loss adapters use automatically labeled, confidence-masked source contacts and calibrated sole proxies, not manually certified ground truth.
- Physical-timestamp 60 Hz sampling with anti-alias filtering and rotation SLERP; one fixed support plane per clip. Official skin geometry supplies rigid heel/toe proxies. These are not exact deformed-mesh contacts.
- Recording/take/actor-variant/mirror and propagated-annotation families are grouped before splitting. A separate actor-held-out evaluation is still required.
- Boundary calibration T-poses are cropped. Train-only statistics use recentered 129-frame windows and four yaw quadrature angles, matching short-window heading augmentation. Old statistics and earlier exploratory pilots are retained separately.
- CSV logs, held-out validation, best/last/final checkpoints, optimizer/scheduler state and normalization provenance. Explicit idle-GPU selection and Lightning DDP are implemented; multi-GPU execution has not been tested because only GPU 5 was idle.

## Dataset Snapshot

Source BVHs remain unchanged at `${PRIVATE_STORAGE}/bones-seed/uniform_orig/soma_uniform/bvh`.

The public [NVIDIA timeline annotations](https://huggingface.co/datasets/nvidia/SEED-Timeline-Annotations) were used for candidate filtering, not as frame-level ground-truth foot contacts. Revision: `b2cf916d8ef7a1e49fc4f0ce9e00c1981d3b9d8f`.

Derived pilot: `${PRIVATE_STORAGE}/motionbricks-bones/pilot-v5`.

| Item | Result |
| --- | --- |
| Original corpus inventoried | 71,132 BVHs |
| Filename/annotation-filtered candidates | 15,116; not all converted or quality-approved |
| Pilot converted | 190 accepted, 2 rejected |
| Pilot splits | 151 train, 19 validation, 20 test |
| Behaviors | Idle, walk, run, turning/curves, backward, lateral |
| Header hierarchy/rotation-order audit | Passed for all 190 |
| Recording-group/content-hash split audit | Passed; 75 groups |
| Finite feature/shard audit | Passed |
| Maximum SOMA30/77 body FK discrepancy | 0.000000586 m |
| Maximum feature roundtrip discrepancy | 0.000007636 m |
| Boundary-calibration cropped clips | 2 |
| Contact review/actor-held-out gates | Not passed |

Shards, statistics, manifests, skeleton/sole contracts and rejection reports are retained. Local report copies and source-motion review images are under `motionbricks/out/bones_pilot_reports`.

## Completed Pilots

Both used only physical GPU 5, an RTX 4090. Other users' busy GPUs and processes were untouched.

| Stage | Steps | Elapsed | Held-out result |
| --- | --- | --- | --- |
| Tokenizer | 1,000 | 52.5 seconds | Reconstruction loss 0.2452; absolute joint-position error 0.2809 m |
| Root Transformer | 1,000 | 45.1 seconds | Validation loss 0.2948; mean root XZ error 0.2071 m |

Tokenizer agreement with automatically derived contact labels is 0.7323. This is not precision/recall against manually annotated ground truth. Its small measured sole penetration does not certify animation quality or target-character ground clearance.

Checkpoints: `${PRIVATE_STORAGE}/motionbricks-preparation/runs/{tokenizer-pilot-v5,root-pilot-v5}/checkpoints/`.

These runs prove data loading, gradients, validation and checkpointing work, not that reconstruction or interactive control is acceptable. Pose Transformer training has deliberately not started because tokenizer quality is insufficient. No long full-corpus run is active.

Verification: ten focused tests passed, including a real BONES BVH roundtrip, time/rotation resampling, contact/support rejection, T-pose cropping, independent CPU worker statistics and future-conditioning masks. A root checkpoint restore retained step 1,000 and reproduced the same validation metrics; no additional optimization steps were performed in that restore check.

## RTX 3080 Screening

The tested shape is one character, 128 frames, compact Pose Transformer width 512 / 8 layers, the official root/tokenizer configurations, decoding and FK. Networks are untrained for this screening. Transport, command-loop buffering, retargeting, IK and UE rendering are excluded.

| Mode | Samples | Wall p50 | Wall p99 | Deadline misses over 16.67 ms |
| --- | --- | --- | --- | --- |
| Eager FP16 | 300 | 21.79 ms | 28.21 ms | 300 |
| Fixed-shape traced FP32 | 10,000 | 11.24 ms | 14.92 ms | 20 |

Maximum traced latency was 47.61 ms. FP32 trace/reference equality passed the fixed-input and changed-root-input checks. The FP16 trace failed a 2 mm numerical-equivalence screen. CUDA graph capture failed on upstream unpinned CPU-to-GPU constants. Neither route is accepted for deployment yet.

This leaves insufficient validated margin for the complete backend and UE scene. It is not a claim of sustained 60 Hz control.

## Next Gates

1. Diagnose tokenizer errors separately for root drift, root-relative body pose, rotations, contacts and codebook usage. Check reconstruction videos, especially starts/stops and turns, before setting a longer step budget.
2. Add manually reviewed contact examples, behavior coverage checks and actor-held-out evaluation. Treat semantic selection and rigid sole proxies as provisional.
3. Improve/export the fixed-shape runtime, validate trained outputs and run a complete current-input-to-result latency test on the RTX 3080. Keep UE rendering/retargeting acceptance separate.
4. Freeze an accepted tokenizer, then implement/train the compact Pose Transformer with history and root control only. Scale the data and use additional GPUs only after rechecking that they are idle.

## Follow-up Reconstruction Diagnosis

`scripts/evaluate_bones_tokenizer.py` evaluates three fixed 128-frame windows per held-out clip, confidence-masked automatic contacts, rotations, root/body errors and codebook usage. The V5 checkpoint was evaluated on 19 validation clips / 57 windows without IK or foot locking.

| Diagnostic | V5 result |
| --- | --- |
| Absolute joint error | 0.2862 m |
| Root-relative body joint error | 0.2041 m |
| Root XZ error | 0.0516 m |
| Hip height error | 0.1459 m |
| Mean local rotation error | 34.48 degrees |
| Trusted stance-foot speed | 0.3306 m/s; source 0.0152 m/s |
| Automatic confidence-masked contact F1 | 0.8471; not manually validated |
| Oracle-root reconstruction error | 0.1973 m; diagnostic only |

The worst turn window was `turn_handstand_270_R_002__A305`, incorrectly classified as ordinary turning. The worst walk review was `walk_forward_grab_relax_002__A006`, an interaction rather than ordinary locomotion. Both are excluded in V6, which also checks upright body geometry and hip height against the inferred support plane. These rejection checks are not replacements for semantic/contact review.

Local review: `out/bones_pilot_reports/reconstruction-review/`, containing the full per-window report, worst-per-behavior comparison image and walk/turn GIFs. Raw source is gray; raw model reconstruction is blue. These visualize a failed pilot, not usable motion.

Fourteen focused tests now pass, including half-turn rotation error, separation of root drift/body error, contamination filtering and upright/support-plane rejection.

## Corrected Data Snapshot

V7 derived data: `${PRIVATE_STORAGE}/motionbricks-bones/pilot-v7`. V6's wider semantic blacklist still admitted non-basic styles, so V7 also restricts filename action tokens. The new pilot converted 191 clips and rejected one short clip, with 152 training / 19 validation / 20 test clips. It covers backward (35), idle (30), lateral (26), run (33), turn (34) and walk (33) clips overall. The 12,912 filtered corpus candidates are not all converted or quality-approved.

New normalization: `stats-window-yaw-v1`, 3,804 augmented train windows / 490,716 frames, root XZ standard deviation about 0.80758 m. V5 checkpoints must not be resumed against V7 statistics. At this corrected-data snapshot no V7 optimization had run; the subsequent bounded M1 diagnostic is recorded below. Full quality-training approval still requires the independent semantic/contact and acceptance checks.

The frozen V7 audit passed all 191 hierarchy/rotation-channel, finite-feature, automatic semantic-policy and source group/hash-isolation checks across 64 recording groups. Maximum feature roundtrip error remains 0.000007636 m.

Manual contact/semantic validation and actor-held-out evaluation remain open. Lateral motion is absent from both validation and test; idle is absent from test. Thus full held-out behavior coverage failed, and the small pilot is not the full acceptance split.

## M0 Source Replay Implementation

Four real V7 fixtures (idle, start, jog-stop, turn; 1,358 frames) are now available in `/MotionBricksGroot/SomaReplay/L_SomaSourceReplay`. They use a fresh Kimodo SOMA30 standard contract with explicit replay-only identity and no checkpoints. UE FK matches Python within 0.000073 cm; native retargeting preserves Manny bone lengths and passes reference/basis/chain checks. The PIE route uses SkeletalMesh/AnimInstance proxy snapshots, not G1 qpos or a fabricated procedural walk. See `soma_source_replay.md`.

No training gate was promoted on this evidence: source replay is a geometry integration check, not manual approval of the whole corpus or a neural quality result. The next learning stage remains accepted-data tokenizer reconstruction, followed by Root/Pose only after their gates pass.

## M1 Preflight Review

See `m1_paper_review.md` and the frozen `m1_tokenizer_contact_protocol.json`. The next experiment restores the official variable-keyframe curriculum, uses first-four-frame observed context for validation, preserves saved contact confidence through official window sampling and binds checkpoints to dataset contents. Following the user's request, trusted-skate and penetration losses are enabled together after engineering preflight; the original disabled-loss proposal has not been trained. The paper-informed compact profile remains an explicitly adapted diagnostic, not a reproduction of the paper's full training scale or a quality-approved human model.

Geometry preflight audited all 191 source clips. Saved contacts had zero violations of their confidence, 35 mm support-height band and 0.2 m/s central-speed conditions. Global-to-local sole height error was zero. Source mean stance speed was 0.01512 m/s; mean penetration was 0.000212 m, p99 0.00465 m and peak 0.02955 m. Rare source penetration remains real and is not described as perfect ground truth. Thirty local tests passed, including five additional geometry-unit/masking/invariance/gradient tests. Manual semantic/contact review and full training/UE acceptance remain open.

## M1 Contact-Aware Diagnostic Completed

Run: `${PRIVATE_STORAGE}/motionbricks-preparation/runs/tokenizer-m1-v7-contact-20261004`. Fresh SOMA V7 initialization, physical GPU 5 only, 1,000 optimizer updates in 51.81 seconds. Both requested geometry losses were active from the start: source-confident sole skating 0.01 and penetration 1.0 with 5 mm tolerance. Upstream predicted-contact skating was replaced to avoid double weighting; contact reconstruction, commitment loss 0.02 and joint velocity loss 2.0 remained active.

All 264 recorded CSV rows were finite. The total loss matched reconstruction plus the configured weighted terms across 100 logged training samples, with maximum absolute difference 0.0000000525. This confirms inclusion in the actual objective, not merely logging unused metrics.

The best checkpoint was step 1,000. Deterministic reconstruction evaluation used all 19 validation clips / 57 windows, four observed context frames, 124 scored frames per window, oracle root velocities for decoding, no IK or foot locking. This is VQ-VAE reconstruction of recorded motions, not generated interactive locomotion.

| Diagnostic | V7 M1 result |
| --- | --- |
| Mean absolute joint error | 0.15326 m |
| Mean per-frame maximum joint error | 0.39253 m |
| Peak joint error | 0.86686 m |
| Root-relative body error | 0.13828 m |
| Root XZ / height error | 0.05925 / 0.00939 m |
| Local rotation error | 35.50 degrees |
| Confidence-masked stance-foot speed | 0.36548 m/s; source 0.01532 m/s |
| Stance-foot movement per 60 Hz frame | 0.006091 m |
| Maximum sole penetration | 0.01929 m |
| Fraction with penetration greater than 2 cm | 0 |
| Worst-window support absolute clearance p95 | 0.03309 m |
| Automatic contact F1 | 0.85951; not manually certified |
| Used entries per 128-entry codebook head | 84, 76, 85, 79 |

The frozen provisional numerical screen failed; `quality_approved` remains false. Raw review images show too little leg swing and substantial mismatch during walk/run/turn examples. Lower penetration does not certify correct gait. V5 and V7 differ in data, architecture profile, conditioning and evaluation, so their numbers are not a controlled loss ablation.

The first report accidentally used equal-window averaging for meters-per-frame skating while meters-per-second skating was stance-sample-weighted. `review-context-v2/evaluation.json` fixes only this aggregation, adds evaluator source hashes and verifies the saved geometry preflight. The weights/checkpoint and all other numerical metrics are unchanged; no extra training occurred. Original `review-context` reports and images are preserved. Since evaluation policy source changed after this run, strict resume must not silently treat it as the original training-policy snapshot.

Local artifacts: `out/bones_pilot_reports/m1-v7-contact/`, including CSV, exact config/protocol, dataset/run contracts, source geometry audit and raw review images/GIFs/NPZs. Current focused tests: 31 passed locally and on the server. No new Root/Pose training or UE code changes were made; no longer run is active.

## M1 Bounded Convergence Completed

After the user's request to continue, one distinct convergence experiment ran under `m1_tokenizer_convergence_protocol.json`. It retained the V7 data/statistics, official network architecture, variable-keyframe curriculum and every loss weight. The step-1,000 checkpoint supplied model/EMA-codebook weights; Adam and the 500-step-warmup cosine schedule were fresh, not resumed. The run performed 10,000 additional updates, approximately 11,000 weight updates including its source pilot, in 386.36 seconds using physical GPU 5 only.

Run: `${PRIVATE_STORAGE}/motionbricks-preparation/runs/tokenizer-m1-v7-convergence-20261004-r1`. The initial unsuffixed launch failed before any optimization because an already saved trainer option was supplied twice; its evidence directory was retained. The corrected launch used a fresh directory. Best checkpoint: `checkpoints/best-step=10000.ckpt`, SHA256 `224013c20a7e4e17343295bd1f5d17c12ab030483076498f5b6ee189de0be1b2`.

Checkpoint selection now uses all 57 deterministic beginning/middle/end validation windows, matching offline review. Training remains 152 clips; training reconstruction was separately reviewed on 456 fixed windows. Both evaluations supply four observed poses plus recorded root-velocity conditions, score 124 future frames per window and use the recorded sequence in the encoder. They are reconstruction, not future-free interactive generation. The test split was not used.

| Held-out Diagnostic | Prior 1,000-step Pilot | Convergence Candidate |
| --- | --- | --- |
| Mean joint error | 0.15326 m | 0.06273 m |
| Mean per-frame maximum joint error | 0.39253 m | 0.16003 m |
| Peak joint error | 0.86686 m | 0.56377 m |
| Root-relative body error | 0.13828 m | 0.04887 m |
| Root XZ error | 0.05925 m | 0.03505 m |
| Root endpoint / height error | 0.09012 / 0.00939 m | 0.05131 / 0.00382 m |
| Local rotation error | 35.50 degrees | 8.06 degrees |
| Source-confidence stance-foot speed | 0.36548 m/s | 0.20407 m/s |
| Automatic contact F1 | 0.85951 | 0.96843 |
| Maximum sole penetration | 0.01929 m | 0.02296 m |
| Penetration over 2 cm fraction | 0 | 0.000531 |
| Worst-window support clearance p95 | 0.03309 m | 0.06052 m |

Training reconstruction still has 0.04749 m mean joint error and 0.16074 m/s stance-foot speed. Thus residual error is not solely a held-out generalization gap. Contact classification improved, but it does not guarantee the foot remains stationary at the surface. Rare penetration and floating-support cases worsened relative to the short pilot; they are retained as failures, not hidden by their small aggregate means. Raw review now follows walking/running/turning motions more closely, but still visibly misses limb and root trajectories.

Official packed/per-head token-path tests pass. On the trained candidate, encode-index-decode agrees with the training forward path within 0.000002146 normalized feature units and 0.000000966 m FK on validation, and within 0.000003815 / 0.000000986 on training. There is no evidence of a token-decode interface mismatch.

An unquantized-encoder diagnostic through the unchanged decoder produced validation MPJPE 0.06940 m, compared with 0.06273 m through the learned codebooks; training MPJPE was 0.05001 m versus 0.04749 m. This is a diagnostic, not a deployable replacement, and does not support attributing the remaining error solely to quantization. Four 128-entry heads used 94/110/109/104 entries on validation and 122/126/123/125 on training. EMA snapshots had 37/39/42/39 entries at reset cluster size 1; batch eight provides 48-to-256 assignments per head depending on window length. This snapshot does not measure reset frequency or prove it caused the errors. Larger-batch and explicit distributed EMA synchronization checks are candidates before future multi-GPU use, not claimed validated improvements.

All 2,538 CSV rows were finite. Across 1,000 logged training samples, total loss matched the unchanged weighted objective within 0.0000000479. Thirty-five focused local/server tests passed before optimization. Actual saved network dimensions are encoder/target 267, decoder 365 and root condition 3; skeleton/fps/trainer fields now describe SOMA30/60 Hz/one GPU. Unused inherited logger/callback/dataloader template sections are not the executed loader; the entry script and protocol remain authoritative for reproduction.

Learning permission was separated from deployment approval to avoid requiring trained UE timing before training the tokenizer it depends on. The bounded scope requires the frozen protocol, content-bound source audit, initial checkpoint/configuration identity and token-path parity. Dataset full-training approval remains false. The unchanged provisional numerical screen still fails. No new Root/Pose training, UE edits or 3080 acceptance tests occurred; the run stopped at its budget and GPU 5 returned to its prior idle state.

Local artifacts: `out/bones_pilot_reports/m1-v7-convergence/`, with source initialization review, exact protocol/config/run contract, CSV, train/validation evaluations, raw skeleton comparison PNGs, walk/turn GIFs and review NPZs. Next priorities: improve residual root/stance geometry, review representative failure/contact examples, complete held-out behavior coverage and add the distinct tokenizer-reconstruction PIE route. Do not promote this candidate to accepted generation or simply increase the step budget without reviewing the remaining failure modes.

## M1 Contact/Root Consistency Repair Completed

After the user requested fixing skating, a new bounded changed-objective experiment ran under `m1_tokenizer_support_repair_protocol.json`. The official encoder, quantizer and decoder layers/parameter shapes are unchanged. A `ConditionedBonesVQVAE` adapter uses the three already supplied normalized angular/XZ rate channels as the final root-rate authority in both training forward and token decoding. Height, body rotations and contacts are not copied from source. This is an explicit integration policy; official MB also supports decoder root refinement, so the existence of its decoded root channels is not itself a bug.

The support loss measures contact/root consistency: after FK, a planted foot must compensate moving root so its world position remains stationary. Detached source contacts/confidence, not predicted contacts, select support intervals. Added terms supervise FK joint/sole positions, distance from the predicted contiguous stance's centroid and support height outside a 5 mm band. These are differentiable training losses, not inference foot locking or oracle foot placement. Airborne motion is not penalized by stance terms.

Weights: trusted skate 1.0, penetration 2.0, joint velocity 2.0, stance anchor 2.0, support height 2.0, FK position 5.0, sole position 2.0, commitment 0.02; the predicted-contact skating term remains zero. The changed objective is explicit, not silently resumed against an older contract. Batch size increased from eight to sixteen, so this is not a single-term causal ablation.

Run: `${PRIVATE_STORAGE}/motionbricks-preparation/runs/tokenizer-m1-v7-support-repair-20261004-r1`. Source model/EMA weights were the preceding convergence checkpoint; optimizer and 500-step warmup/cosine schedule were fresh. Physical GPU 5 only, 10,000 additional updates in 470.77 seconds. The repeated 191-clip source geometry audit passed and retained SHA256 `6886fa1c758393c280d68dbff82296936d1e6fac5fca5f0de4ab4c851e9ce6be`. Best checkpoint is step 10,000, SHA256 `6a214b0f7c3a3b382fb64cd9664950229cdb6dc0031519dcc0987cf8f864761b`.

| Held-out Diagnostic | Previous Convergence | Support Repair |
| --- | --- | --- |
| Mean joint error | 0.06273 m | 0.03626 m |
| Root-relative body error | 0.04887 m | 0.03716 m |
| Mean per-frame maximum joint error | 0.16003 m | 0.10891 m |
| Peak joint error | 0.56377 m | 0.46926 m |
| Source-confidence stance speed | 0.20407 m/s | 0.08426 m/s |
| Source stance speed | 0.01532 m/s | 0.01532 m/s |
| Automatic contact F1 | 0.96843 | 0.97425 |
| Peak penetration | 0.02296 m | 0.02498 m |
| Penetration over 2 cm fraction | 0.000531 | 0.000283 |
| Worst-window support clearance P95 | 0.06052 m | 0.01795 m |

New tail diagnostics expose residual failure: worst-window stance-speed P95 is 0.89016 m/s, peak stance speed 1.37137 m/s, and maximum deviation from a contiguous stance's centroid is 0.11749 m; source centroid deviation peaks at 0.02521 m. These maximum-over-window diagnostics are not global pooled P95s. Root rate/XZ equality is enforced by the supplied command, not proof that the model learned accurate root prediction. The strict screen fails and `quality_approved` remains false. Running is the worst reviewed behavior, with 0.20501 m/s stance speed; walking remains 0.09450 m/s. Do not claim skating is solved.

The raw internal decoder output is reported separately: 0.35215 m MPJPE and 0.20491 m/s skating. Its overwritten root-rate channels no longer receive the authoritative output's supervision, so this is an audit of the unused raw branch, not the deployed adapter's result. Runtime must retain the exact provided-rate contract rather than accidentally consuming that raw branch. The repair does not hard-copy future height or body poses.

No-context reconstruction is a separate diagnostic (128 scored frames, versus 124 after four context frames): skating 0.08375 m/s and MPJPE 0.03636 m. It does not support attributing the remaining error solely to first-four-frame decoder constraints. The differing scored frame sets prevent treating this as a strict paired context ablation.

Training reconstruction on 152 clips / 456 fixed windows still fails: MPJPE 0.02513 m, root-relative body error 0.02566 m, stance speed 0.05876 m/s (source 0.01672 m/s), peak stance-centroid deviation 0.07876 m and peak penetration 0.03568 m. Thus failure is not solely held-out generalization. Training token-path parity also passes, with maximum normalized feature difference 0.000005722 and FK difference 0.000000971 m.

Official encode-index-decode parity passes on all 57 validation windows, with maximum normalized feature error 0.000002384 and FK difference 0.000000608 m. Forty-two focused tests pass locally and on the server, including root authority, no future height/body/contact copying, packed/per-head parity, stance/channel isolation, drift/floating gradients, compensated-root contact consistency and airborne masking. All 3,011 CSV rows are finite; the 1,000 logged training objectives match the weighted terms within 0.0000000878.

Local artifacts: `out/bones_pilot_reports/m1-v7-support-repair/`. The encoder still observes complete source windows, and root conditions are recorded source velocities. These are offline tokenizer reconstructions, not live WASD control or human Root/Pose generation. Source/contact manual validation and behavior coverage remain incomplete, the test set stays reserved, and no UE/3080 acceptance is promoted. Soft training consistency cannot guarantee strictly planted runtime feet; any explicit runtime contact solver must be reported separately from model-only quality, not used to conceal failed reconstruction metrics.
