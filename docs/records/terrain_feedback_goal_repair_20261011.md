# Feedback Target Identifiability

## Reproduced Defect

The previous overfit experiment localized most remaining error to accumulated
Root placement, not root-relative limb pose. A read-only translation test on
its real BONES walk sample found a separate, concrete contract defect:

- Translate the observed four-frame history by 0.1 m or 1 m on unchanged flat
  terrain, preserving velocity, facing and body pose.
- Rolling goal = generated current XZ + requested velocity times horizon.
- After neural recentering, control embedding differs by only 2.98e-8; main
  Root arguments differ by at most 1.86e-6. Far query coordinates have <=1 mm
  numerical differences after route refinement. Flat relief remains zero.
- Model output translates with history, within 0.1 mm.
- Yet the fixed-world source target shifts by exactly -0.1 m or -1 m in that
  neural coordinate frame. The goal input discarded the reference position
  that the feedback loss asks the network to recover.

This demonstrates missing conditioning information for that feedback target.
It is not proof that every observed drift has this single cause.

Audit: scripts/audit_terrain_feedback_contract.py; remote artifact
model_terrain_v3_20261011/feedback_contract_audit.json.

## Method Interpretation

The native MB Root path explicitly accepts spatial target conditions. Keep a
requested planar goal independent of the observed/predicted body position.
The goal remains a soft neural input. No future Root Y, joint pose, hard future
condition mask or output position replacement is introduced.

For reference-path imitation, the source endpoint XZ supplies the intended
planar waypoint. Its inferred velocity and facing remain separate. The source
world, terrain and clocks are unchanged when generated history drifts. An actual
online caller can supply the same explicit planar request through the common
constructor. Free WASD still derives a rolling goal solely from actual history
and joystick input. Reference-path replay and free joystick evaluation must be
reported separately: the former has an intended path and is not evidence that
the latter has already improved.

New control version: explicit-planar-request-independent-facing-v2. Old exported
models cannot silently claim this new contract. The model architecture, frozen
vocabulary, losses and their weights are unchanged. Weak synthetic wall labels
do not provide this request and are rejected by the main supervised trainer.

DART code reviewed at 06b8588429d4c85ea6418b1cceddee03cb4575b4:
https://github.com/zkf1997/DART/blob/06b8588429d4c85ea6418b1cceddee03cb4575b4/mld/train_mvae.py
and data_loaders/humanml/data/dataset.py. It canonicalizes primitives and rolls
generated history through a body-model feature reconstruction. Its text-driven
motion setting does not justify silently moving motion relative to a fixed
terrain scene in this task. Use its progressive feedback principle with the
actual conditioning/coordinate contract made explicit.

## Matched Experiment

Two independent arms use the same overfit_v1/feedback-diagnostic.pt initialization,
four unchanged 129-frame train cases, fresh Adam 5e-5, gradient clip .5, source
physical coefficients, and 100 optimizer updates. Each update traverses all
13 primitives (104 emitted frames, 3.47 seconds) using detached generated-FK
history after the initial prefix. Both arms therefore have equal optimizer
updates and predicted primitive counts. Only the requested-goal semantics differ.

- rolling: preserve the old target construction as a diagnostic control.
- requested: retain the explicitly requested source waypoint during feedback.

Evaluate both true-history and generated-history predictions on the same source
clock, then independently test 10 seconds of constant causal joystick input
for idle/walk. No source future goal or pose is used in those joystick probes.
Report results and raw official-skin visualizations before selecting a change.

Scripts: probe_terrain_feedback_repair.py and launch_feedback_goal_ab.py.
Remote snapshot: code_feedback_goal_v1. Outputs: feedback_goal_ab_v1/rolling and
feedback_goal_ab_v1/requested. Use only physical A100s 2/3 after verifying them
free. No extra card is claimed. Both controllers remain diagnostic artifacts.

## Main Training

control_50k_v4 completed its bounded 5000 updates and stopped with
pilot_needs_revision. All budget checks except train_complete_cases_retained
passed. The failed completeness criterion is retained, and no automatic
continuation or deployment occurred. Checkpoints and old visualizations remain.
After this correction is assessed, define a new versioned training contract;
do not resume the old optimizer while silently changing its inputs or evaluator.

## First Matched Result

Both arms completed 100 updates from identical SHA256
099fc591057570268d630b874a3a7868c19f396fe26a8bf1a1190b3712b78812.
Both pass the existing four-case generated-history memorization screen.
Root RMSE (idle/walk/descending stairs/descending climbing): rolling
1.86/3.46/3.88/1.93 cm; requested 1.19/5.82/4.06/2.40 cm.

The explicit requested-goal arm did not outperform the rolling arm. In the
separate 10-second causal joystick test, maximum idle drift was 8.17 cm for
rolling and 21.03 cm for requested. At 1.6 m/s, forward progress was 13.25/12.80 m
with 1.32/1.36 m endpoint cross-track error. Neither is approved for live delivery.
Do not promote requested goals as a demonstrated free-joystick improvement.

The corrected translation audit does confirm identifiability: a 10 cm/1 m
history shift changes the control embedding by .0166/.1409 rather than 2.98e-8.
That fixes the reference-path conditioning defect, not every control problem.

Both arms also used much longer feedback than the preceding experiment. A third
arm, requested_short, keeps the same initialization, 100 optimizer updates, all
13 source windows per update, batch, losses and explicit requests, but resets
history to the reference every two primitives. This isolates feedback span
without changing the number of source frames or optimizer steps. Its evaluation
still uses all 13 generated primitives. The original two-arm artifacts remain
immutable. Script copy: probe_terrain_feedback_repair_v2.py; output under
feedback_goal_ab_v1/requested_short. No production model is deployed yet.

## Matched Feedback-Length Result

All three arms completed. With the same requested goals, data, initial weights,
optimizer-update count and number of predicted primitives, generated-history
Root RMSE was 6.66/7.91 cm for downstairs/downclimb with two-primitive resets,
versus 4.06/2.40 cm with 13 consecutive generated primitives. The shorter-feedback
arm had lower training loss, illustrating why loss alone does not select a
controller. Its 10-second idle drift was 14.22 cm and joystick walk cross-track
error 1.20 m. None passes live control acceptance.

Visuals: feedback_goal_ab_v1/visual_comparison contains source plus rolling,
requested, requested_short arms with official SOMA skinning and no corrections.
These are four memorized train clips; both terrain examples descend. They are
not evidence of novel-scene traversal, ascent or a working free-WASD controller.

The 34-file visual bundle was transferred through private Google Drive and is
available locally at out/model_terrain_v3_20261011/feedback_goal_ab_v1/visual_comparison.
All manifest-referenced rigs are present. The existing viewer displays its eight
comparison clips with source and three diagnostic arms. Browser inspection of
stairs-generated_history confirmed the source and requested official skins at
frame 65; no browser error was reported. This visual check is supplementary,
not a full motion-quality approval. The earlier overfit page remains unchanged.

Local viewer:
http://127.0.0.1:28932/web/decoder_pilot/?data=/out/model_terrain_v3_20261011/feedback_goal_ab_v1/visual_comparison/&clip=stairs-generated_history

## Native Yaw and Update Semantics

The native MB Root/Pose/Tokenizer training paths call change_first_heading with
uniform random yaw. This joint trainer omitted that operation despite using a
representation with removing_heading=False. A direct rotational-covariance probe
found up to 4.61 cm error over eight output frames for the main 5000-step model,
and 25.46 cm for the overfit checkpoint; the latter can reverse forward progress
at 90 degrees. Artifact: yaw_covariance_audit.json. This is a reproduced omitted
training operation, not proof that yaw augmentation alone fixes all drift.

terrain_scene_frame.py uses the existing native representation transform.
Motion, requested direction and endpoint rotate together. Queries and physical
losses use the inverse transform into the unchanged terrain scene. FK equivalence,
inverse transforms, terrain observations and gradient flow have regression tests.

DART's actual mld/train_mvae.py performs an optimizer update for each primitive
while carrying detached generated history onward. V6 follows this unit of update,
instead of averaging 13 primitive losses into one update. Its 257-frame episode
is sampled once, native yaw is held fixed for the episode, and source time advances
with the generated history. Reports distinguish episode draws from effective
per-domain primitive exposure. Do not equate v6 update cost or update counts with
the diagnostic's 13-primitives-per-update implementation.

## V6 Execution

V6 warmstarts the v4 5000-step model with fresh Adam and unchanged physical loss
coefficients. It learns 45000 further updates toward 50000 cumulative, initially
bounded at 5000 new updates by the retained traversal budget gate. Free joystick
is tested independently at four yaw angles, without source future waypoints.
The final directional gate requires progress ratio .85-1.15, cross-track <=.25 m,
known terrain, and penetration <=3 cm. Unknown predicted paths fail and are never
padded, projected or omitted.

The real two-rank CPU trainer's uninterrupted five updates and three-plus-two
resume have identical entire checkpoints. A two-A100 batch-32 smoke also passed
three real optimizer updates. V6 is now launched on the freshly free NVLink 2/3
pair; follow control-50k-v6-state.json and the primary 50k document. V4 remains
stopped. Changes are model-side training/conditioning/coordinates, not output
postprocessing. The 50k job must still demonstrate actual motion quality.
