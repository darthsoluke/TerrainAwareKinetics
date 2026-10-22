# Real Training-Source Memorization Probe

## Question

Can the current controller and loss fit existing training data? Separate
one-primitive prediction with real history from repeated generated-FK history.
This is a train-source diagnostic, not validation, a new runtime or evidence
of generalization.

## Controlled Setup

Use the immutable control_50k_v4 weights-002000.pt and its source bundle, dataset
identity and physical-loss coefficients. Freeze the same encoder/codebook. Keep
Root, Pose, decoder, geometry losses, four denoising iterations, hard forward
codewords, straight-through training estimator, 5e-5 Adam and 0.5 gradient clip.
Use a fresh optimizer for the explicit diagnostic. No new future Y/pose inputs,
no IK, root replacement, clipping or postprocessing.

Fix four real 129-frame clips from the train split: idle, walk, stairs, climbing.
Choose the quietest idle window and source-defined active windows for the others.
Record IDs, groups, offsets, hashes and exact source arrays. No padding or sample
replacement. Prediction windows have 33 source frames, 32 local output frames;
score only emitted frames 4:12. The complete diagnostic generates 104 frames
(13 primitives, 3.47 seconds) per case.

Stage 1: 600 updates on fixed-corpus, randomly selected real-history windows.
Stage 2: continue the same optimizer for 600 updates using two primitives with
detached generated-FK history in the second primitive, as in the main trainer.
No augmentation or new data is introduced. Both stages repeatedly see all four
clips. Evaluate the same 13 windows with teacher history and generated history.

Report native losses, Root XYZ error, full FK joint error, source and prediction
full-skin penetration, completed frames and failures. Memorization screens are
5 cm Root/FK for teacher history and 15 cm for generated history; these are
declared diagnostic thresholds, not physical or deployment acceptance. Store
uncorrected motion and source for direct visual comparison. A diagnostic
checkpoint uses a separate schema and must not be loaded as the main model.

## Execution

Script: scripts/probe_terrain_overfit.py.
Remote task: model_terrain_v3_20261011/overfit_v1; code: code_overfit_v1.
PID at launch: 738533. Output log: overfit_v1.log.
Uses physical A100 3, shared with our existing two-card main run; no extra card
and no other user's job is changed. Main training stays active and unchanged.

## Initial Measurement

The starting checkpoint is at 2000 optimizer updates, not an untrained model.

| Case | Teacher Root RMSE | Generated-History Root RMSE | Teacher FK | Generated FK |
| --- | ---: | ---: | ---: | ---: |
| idle | 0.0215 m | 0.2062 m | 0.0252 m | 0.1780 m |
| walk | 0.0419 m | 0.8511 m | 0.0385 m | 0.7114 m |
| stairs | 0.0535 m | 0.3207 m | 0.0643 m | 0.3295 m |
| climbing | 0.1377 m | 0.5237 m | 0.1379 m | 0.5268 m |

## Completed Result

Both 600-update stages completed. This is a sequential capacity test, not a
matched-budget ablation: improvement after stage 2 cannot be attributed solely
to history exposure rather than its additional optimizer updates.

All four teacher-history cases pass the predeclared 5 cm Root/FK screening
threshold. Only idle and walk pass the 15 cm continuous-rollout screening
threshold after stage 2. This is not a complete terrain memorization success.

| Case | Initial Rollout Root | After Teacher Stage | After Feedback Stage | Final Teacher Root |
| --- | ---: | ---: | ---: | ---: |
| idle | 20.62 cm | 4.59 cm | 2.65 cm | 0.52 cm |
| walk | 85.11 cm | 4.73 cm | 0.99 cm | 0.78 cm |
| stairs | 32.07 cm | 34.91 cm | 16.43 cm | 2.90 cm |
| climbing | 52.37 cm | 10.70 cm | 19.54 cm | 4.87 cm |

Visual inspection confirms that the selected terrain windows show descending
stairs and descending a climbing structure. Do not present these as ascent
evidence. These are four short, seen training references and one initialization;
they establish neither full-corpus fit nor held-out/live WASD quality.

## Error Decomposition

After removing each motion's own Root translation, final generated-history FK
errors are 0.86/1.03/1.70/2.37 cm (idle/walk/stairs/climbing). Horizontal Root RMSE
is 2.64/0.86/16.19/17.82 cm. The remaining terrain error is dominated by whole-body
placement rather than relative limb configuration on these examples.

Read-only probe `scripts/probe_terrain_overfit_root.py` also records the Root plan
and the actual decoded Root in the same factual world frame. Their total errors
must not be attributed directly to the Root network: both inherit previous
generated-history error. The probe checks the exact per-frame vector identity:

    actual_error = inherited_history_error + new_plan_step_error + decode_minus_plan

The identity passed at 1e-6 m absolute tolerance. RMS values are not additive.
For stairs, new horizontal planning and decode-minus-plan errors are both about
3.0 cm per primitive; their repeated contributions leave a final X error of
-27.47 cm (-14.85 cm planning sum, -12.62 cm decoding sum). For climbing the
corresponding RMS values are 3.44 and 3.78 cm. This is descriptive decomposition,
not a causal intervention proving a single faulty module.

The training feedback stage spans only two primitives, with generated history
at the second primitive (8 emitted frames of exposure). Evaluation chains 13
primitives. Longer feedback exposure and Root/decoder trajectory consistency are
the next controlled hypotheses; neither is declared solved by this experiment.

The climbing source itself has 7.54 cm full-skin penetration P99 under the current
terrain audit. Do not assume the composite geometry plus reconstruction objective
can reach zero or silently reinterpret this as zero source error.

## Artifacts

Remote: model_terrain_v3_20261011/overfit_v1/complete.json,
root_attribution_v2.json and initial/teacher/feedback recorded motions.
Local copies: out/model_terrain_v3_20261011/overfit_v1.
Official-skin visual comparison includes source, initial, teacher and feedback
arms with identical clocks. Animation files move through private Google Drive;
only small code/metric JSON files use SSH transfer. The main training snapshot,
loss configuration and existing live WASD page were not changed by this probe.
