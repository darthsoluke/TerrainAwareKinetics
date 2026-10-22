# Model-Side Terrain Controller V3

## Latest Review Decision

The recovered v6 run reached its 1,000-update evaluation and was deliberately
stopped during the subsequent training-contract review. Current weak goal and
heading labels do not match the online task. Paired wall/open-ground rollouts
show partial terrain response but inadequate command following. The saved
checkpoint is not approved. See `terrain_braking_training_review_20261011.md`
for measured findings and the required correction before further long training.

## Current Run Recovery (2026-10-11)

The v5 run failed before its first 1,000-update validation. Its last progress
record is update 551. The first error is `OSError: [Errno 24] Too many open files`
in the DataLoader multiprocessing tensor-storage reduction, followed by a
30-minute NCCL collective timeout. Only initial weights were saved; no trained
v5 optimizer checkpoint exists, so recovery must restart from the original
source model. This run supplies no post-training quality result.

The worker's long-lived scene cache was passed directly into the multiprocessing
queue. PyTorch shared-memory transport retained file descriptors in that cache.
`collate_learning_items` now sends batch-owned scene clones. The original cache
stays private, with identical data values and sampling. A Linux regression loads
1,200 cached scenes through two spawned workers under a 512-file-descriptor
limit; cached storage must never become shared and worker descriptor count must
stay below 256. Both Linux transport tests passed. The local selected model,
quality and transport suite passed 19 tests with the Linux-only case skipped.

Recovery uses the independent `code_guarded_v6` / `long_guarded_v6` run and free
physical GPUs 2/3, verified as an NV12-connected pair. Reserved cards 0/7 remain
excluded. Model, geometry losses, data distribution and maximum 200k budget are
unchanged. Recovery checkpoints now save every 250 updates independently of
1,000-update quality reviews; bounded prefetch and a 120-second worker timeout
prevent another silent DataLoader stall from waiting for NCCL's long timeout.
Current supervisor files are `guarded-v6-state.json` and
`guarded-v6-training.log`; the older run descriptions below are retained history.

## Objective and Budget

Learn obstacle anticipation, braking, terrain-dependent pose preparation, stable
support and continuous online motion in the MB model. The existing demo stays
available. No collision interception, velocity clamp, output root replacement,
IK, foot lock, floor projection or motion smoothing counts as a model improvement.
The authorized long run is up to 200,000 optimizer updates, after bounded pilots
demonstrate that the new training path works. Use two currently free A100s and
preserve the reserved physical cards 0 and 7 and other users' jobs.

## Reasoning Rounds

1. A terminal-velocity condition does not specify the executed early prefix.
   Continuous replanning can keep promising to stop later. Future terminal values
   also represent desired input as an exact condition. Replace them with learned
   soft intent, including a planar goal/time and requested velocity. Future Root
   XYZ, velocity and body pose are all model outputs. A persistent planar stop goal
   is an intention condition, never an overwritten root trajectory.
2. Raw far XYZ cannot reuse the old near-distance encoder: flat ground produced
   eleven meters of zero-input drift in eight seconds. Retain a near encoder and
   add a distinct far encoder with bounded coordinates, metric scale and reach
   time. Model far conditioning is a learned residual relative to a flat surface.
   This preserves the flat-terrain symmetry across survey ranges inside the model.
3. Native Pose cross-entropy cannot pass collision gradients through argmax to
   Pose. Use the actual hard codewords in the forward pass and an explicit
   straight-through soft-code gradient for geometric learning. This is a declared
   extension of MB, inspired by its VQ straight-through estimator, not an exact
   unchanged-paper claim. Keep encoder/codebook frozen and matched.
4. A frozen Pose and tiny new height columns did not fix the decoder. Jointly
   adapt Root, Pose, dense Root-Y conditioning and scene-aware decoding, retaining
   native regression/token/reconstruction objectives. Add official SOMA FK and
   surface collision/contact/continuity objectives on the final generated body.
5. Only generated FK actually committed as output becomes the next history.
   Following DART's principle, progressively introduce detached generated history
   over multiple primitives. Keep source clock/world targets fixed and supply
   their soft planar task goals, so recovery supervision is observable. Do not
   transport ground truth to conceal drift.

## Missing Supervision

The source-intent sidecars are causal velocity proxies, not captured keyboard
intent. Existing labels do not explicitly say "continue forward but brake at a
wall". Build weak obstacle/stop pairs from genuine BONES stopping motion. Walls
are placed beyond the entire source skinned swept trajectory; references remain
unchanged and feasible. Desired continuation is a soft task label. Record that
these are authored weak pairings, preserve source-family splits, and reject any
pair whose source body intersects the wall. Combine them with original BONES and
official/generated PARC motions. Never relabel an intersecting reference as valid.

## Training and Evidence

- New adapters first, followed by joint Root/Pose/decoder adaptation. Original
  encoder/codebook tensors are hashed and remain frozen throughout.
- Native losses plus separately unit-labelled final-FK, swept surface collision,
  source-evidenced contact gap/slip, and actual-history boundary velocity losses.
  Physical objective coefficients are calibrated on training gradients, recorded
  and frozen; no success-threshold tuning on held-out examples.
- Use a proven distance-transform implementation and differentiable trilinear
  queries for training collision geometry. Scene fields are target-side labels;
  the actor receives only current near/far route observations.
- Pilot comparisons share data, seeds, budgets, inference and geometry. Reject
  candidates that win by freezing, floating, sacrificing motion reconstruction,
  or ignoring terrain. One-step terrain sensitivity alone is insufficient.
- Save resumable checkpoints, optimizer/RNG states and periodic numerical and
  full-body visual rollouts before considering a long run successful.

## Acceptance Cases

Held-out rest/start-stop, straight approaches to walls at multiple speeds,
turning around obstacles, low-obstacle traversal, ascending/descending stairs and
mixed PARC routes. Assess official skinned geometry including intermediate times,
contact coverage, penetration, slowdown onset, progress and discontinuity. Include
counterfactual flat surveys at 0.8-20 m. Use actual source groups, not overlapping
random crops, to separate validation. No checkpoint is approved by training loss.

PFNN comparison requires its actual complete reference implementation and the
same tasks/inputs. The current project has no verified local PFNN benchmark.
Passing absolute safety/progress cases is a required intermediate goal, not proof
of outperforming PFNN. References: MB 4.1-4.3/6.1, PARC 5.3/Appendix C, DART 3 and
Appendix D.2, TokenHSI 3.3-3.4; existing literature/code audits remain applicable.

## Executed Evidence and Revised Training

The first matched pilots finished 400 updates each, global batch 16. Both used
the same initial model, source stream and weak pair corpus. The full extension
reduced some failures relative to native-only adaptation, but failed the
predeclared pilot gates. It is not a candidate for the running web demo.

| Held-out measurement | Initial soft interface | Native pilot | Joint pilot |
| --- | ---: | ---: | ---: |
| Mean short FK MPJPE (m) | 0.2213 | 0.1787 | 0.2428 |
| Mean rest/start-stop drift (m) | 0.1781 | 3.1461 | 1.8270 |
| Minimum start progress ratio | 0.0408 | 0.3320 | 0.0399 |
| Mean wall maximum body penetration (m) | 0.0443 | 0.1646 | 0.1669 |
| Mean final wall speed (m/s) | 0.2228 | 0.7217 | 0.3269 |

These are failed experiments. Lower training loss did not establish online
control. Full official 18,056-vertex audits and eight swept samples were used
for flat/wall tasks. The numerical comparison is stored locally at
`out/model_terrain_v3_20261011/comparison-000400.json`; official-skin recordings
are under `visual_pilot_v1` and explicitly show these unsuccessful pilots.

Five implementation/training issues were repaired before the revised run:

1. A 9x9 point survey over 20 m could miss a thin wall. The far encoder now
   receives max/min relief pooled from a 201x121 route survey. Regression cases
   preserve both a 20 cm wall and pit at 3.4/12/20 m ranges. Near coordinates
   retain their original metric encoding; flat far relief still yields exactly
   zero residual inside the network at every tested range.
2. Adapter warmup also froze the newly added dense Root-Y columns and local
   component-mask embedding. New height columns now train during warmup while
   copied decoder columns have exactly zero gradient. Root/Pose/decoder base
   parameters remain protected until the soft task interface learns to start
   without sacrificing rest stability. This is a training phase decision.
3. Weak pairs were always consumed from their prefixes. With short batch
   windows, much of their late braking/stopping evidence was never used. Each
   source now samples a complete aligned window from its entire available
   motion; one third of weak pairs select the final complete window. Source
   coordinates, real frames, fixed terrain and authored continuation intention
   are unchanged. No padding or repeated final frames is introduced. Real DDP
   checks recorded nonprefix and late weak windows; generated primitives keep
   advancing within the selected source window's original clock.
4. The contact-height objective queried ground beneath the predicted misplaced
   foot. Inside a high wall this incorrectly taught the foot to rise onto the
   wall top. It now retains the real source support-surface correspondence;
   collision supervision supplies the independent three-dimensional obstacle
   gradient. Source skinned penetration remains reported, but no longer grants
   equivalent penetration to predictions. The fixed training penetration tolerance
   is 5 mm, with collision validity determined by the actual known scene.
   Regression cases verify both the contact gradient direction next to a wall
   and a nonzero penalty when prediction reproduces an invalid source position.
5. Refinement formerly replaced the intention-direction survey with the
   proposed-route survey. A proposal could remove a hazard from the final
   observation before any motion was committed. Both surveys now remain
   available to Root, Pose and decoder. Shared CNNs encode independent patches
   before attention fusion; the two grids have no invented spatial adjacency.
   A regression verifies that a clear proposed route cannot erase a known wall.

The revised task is `model_terrain_v3_20261011/long_guarded_v5` on A100 physical
cards 1 and 2. Cards 0 and 7 and all other users' jobs are preserved. The available
cards do not form an NVLink pair. The maximum budget is 200,000 updates, global
batch 32, microbatch 16/rank, fresh Adam 5e-5 with 2,000-update warmup and cosine
decay. Encoder and vocabulary remain frozen and hashed. New interfaces train
first for at least 2,000 updates; base unfreezing depends on fixed validation
criteria. Generated FK history ramps in after update 2,000. An interface that
still fails at update 8,000 stops the run for revision; four successive reviews
without an eligible quality improvement also stop it while retaining candidates.
The initial phase remains a protected pilot, not evidence that the full budget
is already justified. The earlier guarded runs were deliberately interrupted
as source-window, contact-target and observation-retention defects were found;
their files remain as historical evidence.

Every 1,000 updates saves optimizer/RNG/model state and fixed numerical rollouts.
Every 5,000 updates exports official-skin comparison recordings independently
of metric collection. The supervisor publishes reports/visualizations through
private Google Drive; checkpoints remain on the server until model selection.
It never switches the existing local demo. Launcher: `launch_joint_terrain_training.py`;
run state: `guarded-v5-state.json`; training log: `guarded-v5-training.log`.
The local Drive proxy tunnel is restored with `keep_motion_drive_tunnel.ps1`,
bound only to remote loopback and retaining SSH host verification.

Local relevant regression suite: 32 passed. Actual two-rank training, phase
transition, hard-code forward gradients, source-window sampling and full-surface
evaluation completed. No learned improvement, deployment approval or PFNN
superiority is claimed for the revised weights before their reviews exist.

Failed-pilot comparison viewer (separate from the still-running controller):
`http://127.0.0.1:28932/web/decoder_pilot/?data=/out/model_terrain_v3_20261011/visual_pilot_v1/&clip=flat-0-rest`.
Desktop/mobile Chrome smoke checks loaded both official skinned bodies without
page errors. Local screenshots: `out/model_terrain_v3_20261011/pilot-desktop.png`
and `pilot-mobile.png`. This is recorded playback of the initial/400-update
pilots, not a control trial of the currently training guarded model.

PFNN source audit: the original C++ demo mirrored at
`https://github.com/sreyafrancis/PFNN/blob/master/demo/pfnn.cpp` projects both its
future trajectory (lines 1519-1532) and current position (1959-1972) away from
walls, and has optional IK (1728). This defines the complete reference demo for
a later fair comparison. Those runtime corrections are not used by this model.
