# Final V8 Terrain Response Audit

The latest user priority is motion/terrain mismatch, including stairs and
climbing through walls. The feet-together gait investigation is deferred.
No production weights, training run, live settings or existing server were
changed. Read-only diagnostics used final weights5709, SHA256
8f70da0f9db03e84af86dfd09e5fce63b7d7e761b23ed9a42aa7fd72a9bea9cc.

## Reproduced Live Failure

Independent sessions on28934 started from the existing world spawn histories,
with constant world +Z1.6m/s input, facing0,30Hz,45 publications of8 frames.
Only recorded model rig output was skinned with the official SOMASkin.lbs;
no collision correction, filtering or IK was applied.

| Scene / inference | Sole vertical penetration P99 | Maximum | Frames over2cm |
| --- | ---: | ---: | ---: |
| Flat /8 tokens | 0.42cm | 1.42cm | 0/360 |
| Stairs16cm /8 tokens | 25.72cm | 32.51cm | 129/360 |
| Stairs16cm /16 tokens | 27.24cm | 34.29cm | 126/360 |

These are vertical distances below the solid heightfield column, measured over
196 official sole vertices, not shortest Euclidean signed distances to risers.
They establish terrain intersection; do not rename them wall-normal depth.
The stair far observation contained1.6m of raised terrain from the first plan.
The first >2cm penetration occurred at2.57s (default) /2.67s (long horizon).
Root height rises with the stairs, but root-relative joint motion changes by
only4.62cm on average against the flat run. This does not prove complete terrain
ignorance; it demonstrates insufficient limb adaptation despite visible terrain.

Thus neither missing distant observations nor simply increasing the prediction
horizon explains away this concrete failure. A1.87-second emitted prediction
range (16 tokens, dropping context/end) did not solve it.

Evidence: out/model_terrain_v3_20261011/gait_terrain_audit_20261012/live/
summary.json,skin_report.json,terrain_response.png, and the raw rig/plan records.
Reproduction: scripts/audit_terrain_live_gait.py and analyze_terrain_live_skin.py.

A later separate authored-platform live recording disconnected before producing
a complete case. No platform live metric is reported from that attempt. The
existing28934 process was still listening; local GPU usage was97% of10GB and
100% utilization at inspection. No user service was stopped to continue this
diagnostic. The completed real training climbing case remains separate evidence.

## Training-Set Interventions

An idle A1000 was checked and used for a finite read-only probe. Same final
checkpoint, stored training source scenes and actual source histories. The
probe made no optimizer updates. It compared source-history predictions,
104-frame generated-history rollouts with source planar waypoints, and rolling
goals using the same source-derived velocity/facing. The latter is a protocol
intervention, not a claim of free user joystick coverage.

| Real training case | Root RMSE: teacher / reference feedback / rolling feedback | Sole vertical penetration P99, same order |
| --- | --- | --- |
| Climbing official_b02813d3480433ed | 4.11 /11.54 /16.47cm | 6.30 /21.42 /23.73cm |
| Boxes official_034289b9c55f0b4e | 2.46 /4.08 /4.28cm | 0.05 /0.52 /1.02cm |
| Mixed generated_054c2050e9d10bb0 | 3.21 /6.53 /9.37cm | 3.97 /4.86 /6.99cm |
| Stairs official_dc03d6484b176db0 | 2.25 /8.38 /9.69cm | 2.51 /3.69 /4.28cm |

All104 requested frames completed. The selected official stairs window is not
an ascent demonstration: its initial28-frame net rise is only1.1cm. The case
selection searches only the first five eligible source IDs per family, so this
probe is not a corpus coverage estimate or a held-out benchmark. No suitable
platform case was selected under this rule; do not fabricate its result.

Keeping history/control fixed and replacing the scene by a plane also changed
the output: Root-Y RMS changed23.6cm on climbing,82.9cm on boxes,40.1cm on mixed;
root-relative joint positions changed4.6/14.8/7.7cm. Near validity was95-100% in
these cases. The terrain branches are active. Learned gate magnitudes alone
are not evidence for weak/strong influence because internal feature scales vary.

Files: training_cases.json and stairs_training_case.json in the audit directory.
Reproduction: scripts/audit_terrain_response_v8.py. A corrected second invocation
admits129-frame sources; the original137-frame candidate filter excluded stairs.

## Confirmed Training Gaps

1. train_joint_terrain_v3.py takes the minimum source length over every rank and
   batch row, then caps every episode by that minimum (lines401-409). The saved
   sampler audit recovered mean4.27 primitives, maximum6, none13. The live and
   training-set feedback failures now corroborate that long generated-history
   stability is inadequate. A controlled retrain is still needed to measure how
   much fixing that specific bug improves it.
2. Training intents supply a fixed source endpoint (line95); online controls
   construct a goal from current generated XZ plus requested velocity/horizon.
   The source reference is appropriate for path imitation, but it is not the
   same input distribution as persistent joystick control after drift. In the
   measured cases, rolling feedback usually makes error worse. Future source
   Y and pose remain target-only; no cheating condition was found here.
3. Half PARC /half BONES does not balance terrain events. The saved contract has
   41.83% total probability in long PARC mixed clips,0.0405% in the explicit
   stairs family and0.2287% in climbing. Mixed clips can contain both skills,
   so those percentages are NOT total ascent/climb exposure. Current sampling
   has no event-level accounting for approach, takeoff, hand support or landing.
4. Every supervised rollout keeps the source future and source-clock foot
   support correspondence. Collision loss can oppose penetration, but the
   reconstruction objectives keep asking for the same source pose/time when
   generated history has approached an obstacle differently. There is no
   counterfactual scene branch that removes an incompatible reconstruction
   target, nor physics-repaired recovery demonstrations. This objective conflict
   is a design risk; gradient dominance was not measured in this audit.
5. Contact supervision explicitly selects sole vertices. Whole-body collision
   and FK losses exist, but there is no explicit hand-support/grasp continuity
   supervision for climbing. A body that rises is not therefore a valid climb.

The near input is two9x9 metric patches pooled to3x3 feature grids. The far
branch pools dense201x121 queries to9x9 maxima/minima. It preserves obstacle
presence but coarsens edge placement. Representation resolution is a plausible
contributor, not a demonstrated primary cause; adding SDF alone is unjustified.

## What PFNN and PARC Actually Supply

PFNN's uneven-terrain locomotion must not be dismissed as an IK trick. Its model
uses trajectory-aligned center/left/right terrain heights, previous joint
positions/velocities, gait inputs and phase-dependent weights. These give it a
specific supervised mapping from upcoming ground to gait-consistent steps.
The inspected demo source also has external future/current wall projections;
that explains wall nonpenetration assistance, NOT all of its stair adaptation.
This audit does not establish that PFNN autonomously recognizes arbitrary walls
and climbs them. Our requested climbing controller has to learn that behavior.

Source mirror: https://github.com/sreyafrancis/PFNN/blob/master/demo/pfnn.cpp
Relevant sections:1519-1531 future wall projection;1617-1660 gait, joint and
trajectory-height inputs;1959-1970 current wall projection. Snapshot saved under
the audit references directory.

PARC section5.3 explicitly describes the same failure mode: excessive dependence
on previous frames can produce wall-running despite terrain conditioning.
Its remedies include penetration loss,10% random-terrain generations with
reconstruction disabled, history-condition dropout/blended denoising, and a
physics tracker that corrects motion in simulation. Our pipeline copied neither
that complete learning objective nor the tracker into the live controller.
Using PARC-generated data is not equivalent to inheriting PARC's full method.

Paper: https://arxiv.org/html/2505.04002v1#S5.SS3
Official local implementation: out/vendor/PARC/PARC/motion_generator/mdm.py,
construct_conds: previous-state attention dropout during training. The current
MB extension does not implement that conditional-dropout learning path.

## Next Experiments, in Dependency Order

1. Repair episode advancement by compatible lengths or independent rows;
   preserve short sources and original domain mass. Assert realized13-primitive
   exposure and exact source/history clocks, not just a configured maximum.
2. Audit event exposure inside mixed clips using true source motion/contact and
   scene geometry. Stratify real approach/ascent/descent/climb/landing windows;
   retain flat-control examples. Do not balance only coarse file labels.
3. Train matched small pilots on actual source terrain approaches. Compare the
   repaired feedback baseline against a PARC-inspired counterfactual/recovery
   branch. Only factual paired data use reconstruction; changed-terrain cases
   need compatible targets or geometry/control objectives without false pose
   labels. Require forward traversal as well as low penetration to reject freezing.
4. Check hand and foot surface/contact timing for climbing, and compare errors
   before contact, at first contact and during support. Add reference-derived
   hand support only where evidence exists; do not invent grasp labels.
5. If edge placement still fails with correct source-history and matched goals,
   ablate finer trajectory/effector-aligned terrain tokens against the same
   baseline. Do not change terrain encoding, sampler and losses in one opaque run.

Keep model outputs uncorrected. Existing pages stay available. No new long run
or claim of repaired terrain behavior follows from this investigation alone.
