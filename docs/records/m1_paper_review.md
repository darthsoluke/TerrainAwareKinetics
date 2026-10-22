# M1 Method Review

Date: 2026-10-04. Review before the next V7 optimization step.

## Findings

1. **P1: missing keyframe curriculum.** `BonesTokenizer` forced every training sample to have zero body keyframes. That removed a core conditional-training path and did not train context anchoring. Removed the override; the official 0-to-10 curriculum is restored. Validation supplies only the first four observed poses, zeros all unavailable target values, and scores the following frames. Unconditioned reconstruction remains a separate diagnostic.
2. **P2: published config is not the paper experiment.** The released template has eight ten-entry heads, 512 channels, AdamAtan2 and only horizontal root-velocity conditioning. The new `paper-informed` profile uses pose position/rotation inputs, angular plus XZ root velocity as decoder conditions, four 128-entry heads and Adam. Compact convolution widths are retained, so this is not an exact paper configuration.
3. **P2: confidence drift.** The old wrapper recomputed contact confidence with forward differences while preprocessing used central differences. Four M0 clips had 38 disagreements across 5,432 channel-frames. Dataset loading now carries the saved confidence; the official sampler exposes its selected frame indices so the mask follows exactly the same crop and augmentation. Network layers are unchanged.
4. **P2: incomplete checkpoint identity.** A path and mean/std hash did not protect against changed shards or a changed split manifest. Runs now bind the accepted manifest, skeleton/sole contract, train statistics, quality report, every shard, training policy and experiment protocol. Changed or incomplete identity rejects resume/evaluation.
5. **P2: insufficient quality evidence.** Prefix-only validation and MPJPE could hide a bad joint or floating feet. Checkpoint selection now uses mean per-frame maximum joint error after context. Deterministic review also reports root/body errors, rotations, heading, acceleration, source-contact skating, support clearance and peak penetration. These are reconstruction diagnostics, not generated-control scores.

## Paper Basis

[MotionBricks](https://arxiv.org/html/2604.24833v1): Section 4 separates pose tokens from root conditions, uses multi-head quantization and variable keyframes; Sections 7.2-7.4 discuss maximum-joint precision, per-head capacity and training/deployment. Appendix G describes conditional inference plus motion-buffer playback in UE. The paper's 30 Hz motions, 27-joint animation data and RTX 5090 timing are not our SOMA30/60 Hz/RTX 3080 acceptance conditions. The paper's unseen-content and actor-content-combination splits also differ from our additional actor-held-out goal.

## Locked Diagnostic

Current protocol: `m1_tokenizer_contact_protocol.json`. Before any V7 optimization, the user requested enabling the geometry terms together after checking their preconditions. The original `m1_tokenizer_protocol.json` remains an unrun upstream-reference proposal, not a trained comparison. All 191 V7 clips passed automatic support-label consistency and ground-height conversion checks; loss-unit, masking, invariance and deployment-FK gradient tests also passed. These checks do not certify manual contact ground truth.

Fresh initialization on V7, batch eight, a single idle GPU, at most 1,000 updates, no test-set tuning. Joint velocity loss remains 2.0. Source-confident FK-sole skating is enabled at 0.01, penetration at 1.0 with 5 mm tolerance. It replaces, rather than duplicates, the upstream predicted-contact skating term. Contact reconstruction remains confidence-masked. The trainer requires the content-bound geometry preflight and records the exact report, weights, configuration and source with checkpoints.

This run checks the corrected pipeline, not convergence. It changes several settings versus V5 and cannot isolate the cause of any numerical improvement. Do not chain smoke runs or promote full training approval. Missing held-out behavior coverage, manual contact/semantic checks, actor-held-out evaluation, PIE reconstruction and complete 3080 timing remain independent gates.

## Bounded Convergence Follow-up

After the completed 1,000-step diagnostic and the user's request to continue, `m1_tokenizer_convergence_protocol.json` authorizes one distinct, bounded tokenizer-convergence experiment, not a series of smoke retries. Preserve the exact architecture, V7 data/statistics and geometry weights. Initialize model and EMA-codebook weights from the content-bound step-1,000 checkpoint; start a fresh Adam/cosine schedule with 500 warmup steps and a fixed maximum of 10,000 additional updates. This is explicitly not optimizer resume.

The source checkpoint passed official encode-index-decode equivalence on all 57 validation windows: maximum normalized feature difference 0.000001431 and FK difference 0.000001019 m. Thirty-five focused local/server tests pass, including packed/per-head token equivalence, evaluation window coverage and bounded-scope guards. Training checkpoint selection now uses the same three beginning/middle/end windows as offline review, rather than clip prefixes alone. Saved configs report actual SOMA/60 Hz/device/feature dimensions, not inherited G1 template fields.

Separate permission to learn from permission to release: source geometry/split/statistics/gradient/parity checks permit this small convergence experiment, while manual semantics/contacts, full behavior/actor generalization, PIE and complete 3080 timing remain open. Do not require the trained end-to-end deployment to pass before training the tokenizer it depends on. The dataset's full-training flag is unchanged and false; this does not authorize full-corpus or Root/Pose training. Stop at the frozen budget and report held-out errors even if they fail.

Completed result: 10,000 additional updates in 386.36 seconds. Held-out MPJPE decreased from 0.15326 to 0.06273 m and local rotation error from 35.50 to 8.06 degrees; stance-foot speed remains 0.20407 m/s and rare penetration/support-clearance cases still fail. Training MPJPE is also above the screen at 0.04749 m. Continuous latent decoding did not improve mean joint reconstruction, so no quantizer replacement is justified by this diagnostic. See `bones_seed_training_execution_20261004.md` for complete results and preserved failure evidence. Neither Root/Pose training nor UE/3080 acceptance was promoted.

## Contact/Root Consistency Repair

The user requested repairing skating before treating the reconstruction as usable motion, and proposed contact-driven movement supervision. A planted sole obeys `delta(world_root) + delta(world_axis_relative_sole) = 0`: FK leg motion must compensate root translation. This is a kinematic consistency constraint, not a ground-reaction-force model. Airborne root motion and stationary ground contact must remain possible; multiplying root velocity by a predicted contact flag is incorrect and lets a model evade supervision.

The existing zero-world-sole-speed term already measures this residual. The repair strengthens it, makes the planar command authoritative during both training and token decoding, and adds direct FK/sole position, contiguous-stance centroid and support-height supervision. Source labels select loss intervals but are never decoder inputs. Neither IK nor inference foot locking is applied. Stance centroids are differentiable training references derived from the predicted stance, not oracle runtime foot anchors.

[MB Section 4.2](https://arxiv.org/html/2604.24833v1#S4.SS2) permits decoder root refinement, and the official `MotionInference` has `from_pose_module` and `from_root_module` policies. Thus predicting/refining root is not itself an upstream bug. Our explicit provided-rate adapter is a SOMA/CMC-oriented integration policy, not a claim that the paper requires hard root copying. It overwrites only the three root-rate channels already supplied as conditions; height, body rotations and contacts remain network outputs. The internal raw root head is no longer trained as the authoritative output and must not be silently used at deployment.

The preliminary source-video diagnostic replaced only XZ root translations: walk skating increased from 0.29363 to 0.32710 m/s and turn from 0.27851 to 0.30367 m/s. Merely substituting root after training does not solve foot compensation. A separate `m1_tokenizer_support_repair_protocol.json` therefore freezes changed-objective weights-only initialization, batch 16, 10,000 updates and a contact-sensitive checkpoint score. This is not unchanged-objective convergence or a controlled single-term ablation.

The completed repair still fails acceptance: validation skating is 0.08426 m/s, peak contiguous-stance centroid deviation 0.11749 m, worst-window stance-speed P95 0.89016 m/s and sole penetration peak 0.02498 m. MPJPE is 0.03626 m and root-relative body error 0.03716 m. Exact planar-root equality follows by construction, not learned trajectory accuracy. Stronger soft losses improve averages but do not establish zero skating. Keep model-only reconstruction, explicit runtime contact correction and future generated locomotion as distinct evaluations.

The encoder continues to observe the full source window, as intended for tokenizer reconstruction. The human Root/Pose generation modules have not been trained. No WASD locomotion, UE deployment or full RTX 3080 acceptance is inferred from these results.

## Other Open Methods

MB remains the chosen backbone: first SOMA flat locomotion, then terrain awareness. Preserve its editor-authored pose constraints and real-time in-betweening path. Other methods supply candidate modules or comparisons; they do not replace this first milestone. Keep shared SOMA/units/contact/history contracts and isolate changes before combining modules.

- [PFNN](https://theorangeduck.com/page/phase-functioned-neural-networks-character-control): inspect its terrain fitting, trajectory/history features and fast controller as a separate baseline. Do not fabricate terrain supervision from BONES root paths alone.
- [CCM / AI4Animation](https://github.com/sebastianstarke/AI4Animation/blob/master/AI4Animation/SIGGRAPH_2024/ReadMe.md): candidate for direct responsive locomotion. Its existing rollout/responsiveness tradeoff and processing path should be evaluated, not silently substituted into MB tokens.
- [AI4AnimationPy](https://github.com/facebookresearch/ai4animationpy): official Python tooling candidate for feature extraction, contact/IK review and an independent locomotion baseline; not yet integrated or benchmarked here.
- [PARC](https://arxiv.org/abs/2505.04002): later candidate for physically augmented terrain data. Physics during data preparation does not require a physics policy at runtime, but its data and retargeting must still pass our SOMA contract.

Do not merge incompatible feature conventions or weights. First establish this small reference experiment, then change one module at a time and keep the original comparison.
