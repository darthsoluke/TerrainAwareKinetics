# Terrain Braking Training Review

Current priority correction: this is a historical audit of the weak braking
task. The main 50k experiment now learns real PARC terrain traversal plus BONES
locomotion; artificial walls no longer drive the training mixture or acceptance.
See `terrain_control_50k_20261011.md` for the current executable contract.

## Decision

The model-side braking task is not corrected end to end. The observation,
geometric-gradient and data-transport fixes are real, but the current weak
control labels and evaluation do not represent the actual continuous WASD
controller. Do not spend the remaining 200k budget on this contract or deploy
the checkpoint. The v6 training process was deliberately stopped during this
review after its 1,000-update checkpoint and evaluation were saved. The latest
full recovery checkpoint was subsequently verified at update 1,250 with Adam
state and both ranks' RNG state. Training
model/loss code was not changed by this audit. The supervisor's resulting
nonzero process-exit report describes this deliberate interruption.

## Findings

### P1: Different control tasks in training and online use

`prepare_obstacle_stop_pairs.py:86` places a constant planar goal two meters
beyond an authored wall. `terrain_learning_data.py:24` returns that entire
six-field override unchanged for every crop and horizon. Training supplies
source-derived endpoint goals for other motions. Online
`SoftCommandState.intent` instead uses `current_xz + velocity * horizon` while
movement is requested. Wall evaluation reuses the weak fixed goal at
`evaluate_joint_terrain_v3.py:149`, concealing the mismatch.

Measured over the first and final 33-frame windows of all 256 training weak
pairs, the fixed-goal distance divided by the online goal distance has median
4.235 and P95 15.136. This is a condition-distribution mismatch, not an inference
bug that can be fixed by increasing a scan-distance slider.

### P1: Travel direction is incorrectly authored as body heading

The pair builder derives desired heading from initial planar velocity at lines
87-88 while keeping the original full-body motion unchanged. Backward, sideways
and turning stops therefore receive arbitrarily conflicting desired-heading
labels. In 167/256 training pairs the final demonstrated body-heading vector
differs from the requested vector by more than 90 degrees; median difference
is 126.85 degrees. All six validation pairs exceed 90 degrees (median 166.82).
Movement direction and body-facing direction are different control variables.
These are authored weak labels, not evidence that the source actor intended
to face the direction supplied by the pair builder.

### P1: The wall gate accepts stopping without proving terrain causality

The current wall pass criterion checks penetration and final speed. It records
net displacement but does not require progress, smooth deceleration, braking
before contact, or continued movement when the same wall is removed. Three of
the first four selected validation pairs come from the same backward-stop
family. All weak wall shapes use the same straight 8 m wide, roughly 0.4 m thick
barrier construction and heights 3/3.5/4/4.5 m. The existing gate does not test
unseen shapes or reject a controller that simply stops on open ground.

### P2: Collision supervision alone is not a braking objective

`terrain_joint_objective.py:115` penalizes predicted penetration over the
predicted segment. It provides zero collision gradient while all sampled
future points remain outside the terrain. It can support anticipatory avoidance
when a predicted segment collides, but does not by itself specify when or how
smoothly to brake. The imitation data must consistently teach that behavior.
Only 49/256 weak training clips contain at least one second of contiguous final
motion below 0.12 m/s; their median contiguous still tail is 0.367 seconds.
Long stationary behavior against sustained input is poorly covered.

## Actual 1,000-Update Evidence

The standard validation's mean wall final speed is 0.0305 m/s and all four wall
cases pass its present gate. However, the minimum flat start progress ratio is
only 0.0299, and mean rest/start-stop drift is 0.3046 m. The global quality gate
correctly remains false. These are adapter-warmup results, not a claim that a
fully trained version of this architecture cannot work.

A new read-only counterfactual test used the fixed 1,000-update checkpoint and
one held-out source from each of three validation source families. Each case
ran six seconds with identical four-frame history, requested velocity and
heading, comparing the original fixed goal with actual online goal updates.
Within each protocol the only intervention was removal of the authored wall.
No output corrections were applied. Full official skin and eight swept samples
per interval were assessed. The test deliberately preserves the original
heading labels to isolate the goal/terrain intervention; it does not repair or
validate those heading labels.

| Source case | Fixed goal: open / wall final speed (m/s) | Online WASD: open / wall final speed (m/s) | Online open-ground progress ratio |
| --- | --- | --- | --- |
| backward stop | 0.524 / 0.046 | 0.055 / 0.054 | 0.314 |
| fast forward stop | 0.115 / 0.021 | 0.345 / 0.055 | 0.215 |
| curved stop | 0.043 / 0.044 | 0.235 / 0.031 | 0.299 |

There is partial genuine terrain dependence: removing the wall changes several
rollouts. The earlier suspicion of completely terrain-independent freezing is
too broad. The supported conclusion is that obstacle response is inconsistent
and open-ground command following fails, especially under the online protocol.
No unseen-terrain generalization claim follows from these three cases.

Machine-readable local evidence:
`out/model_terrain_v3_20261011/braking_training_review/counterfactual_step1000.json`
and `validation_step1000.json`. Reproduction script:
`scripts/audit_terrain_braking_counterfactual.py`.

## PFNN Mechanism

The public original-demo source mirror
<https://github.com/sreyafrancis/PFNN/blob/master/demo/pfnn.cpp>
explicitly projects the desired future trajectory away from authored wall
segments (1519-1531), computes a proximity-related gait input (1574-1581), and
projects the current character trajectory position out of walls (1959-1970).
Thus the complete demo can handle newly placed collision walls without learning
each wall shape from motion data. Its demonstrated nonpenetration cannot be
attributed solely to the neural network recognizing unseen terrain. This is a
mechanism distinction, not a reason to accept worse control in our project.

The relevant lesson is to represent desired intent separately from feasible
motion, and train terrain-dependent decisions. The user's chosen model-only
runtime requires learning that decision inside Root/Pose/decoder. Copying the
PFNN runtime projections would not meet that requirement.

## Required Correction Before Another Long Run

1. Define one shared online intent builder for training, tests and runtime.
   Generate goals from the same requested speed, horizon and actual observed
   state. Keep facing direction independent of travel direction. Preserve
   human demonstrations; reject weak pairings whose authored controls cannot
   support their demonstrated behavior.
2. Add paired scene/control evidence and counterfactual tests: same history
   and sustained command should continue on open ground and prepare/brake
   against an impassable obstacle. Include varying approach speed/distance,
   wall angle, width, low traversable obstacles and passage openings. Generate
   missing references offline using PARC and its physics tracker, with source
   and scene-family split separation. Removing a wall alone must not magically
   relabel an unchanged stopping reference as continued locomotion.
3. Evaluate rollout-level progress, precontact deceleration, acceleration/jerk,
   full-body penetration, support/foot motion and sustained stop together. Use
   held-out geometry families. Compare against the open-ground counterfactual,
   not final speed alone. Keep native MB reconstruction/token objectives and
   the working generated-history geometry path.
4. Re-run a bounded matched pilot under the corrected contract, including
   control starts from idle. Scale only if the online protocol passes movement
   as well as stopping tasks. Do not add runtime collision projection, foot IK,
   speed caps or output root replacement to hide failed learning.

This turn audits and records the missing correction. It does not claim the new
shared-control data or a successfully trained corrected controller already exists.
