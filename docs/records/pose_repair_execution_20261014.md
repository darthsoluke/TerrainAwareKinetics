# Pose Repair Execution

Latest strategy and completed native route curriculum are documented in
[native_stage_repair_20261014.md](native_stage_repair_20261014.md). The user
rejected the repeated adapter trial strategy. Three native four-window capability
gates now pass, including normal predicted-Root actor up/down; full feedback from
flat still fails. The whole-route normal actor now passes88/88source-history
windows after native codec/Root/Pose fitting. Constant-command feedback and
17caseflat preservation still FAIL; current28941is not replaced. All new runs
completed; source-command timing controls and115privately verified visual files
are documented at the top of the native-stage document. Preserve all
completed experiments below; do not restart them or treat them as current work.

## Current Work: Joint Root/Pose Native Motion Supervision

The sparse-history test below is now COMPLETE and not adopted. Same source
schedule and corrupted-token hashes as the full-history control;278/297/308/317
examples use4/1/2/3historical poses, respectively. All frozen/reload checks pass.
Both live goals are reached, but peaks31.09/56.26cm at0.8/1.6m/sremain failures;
low-speed lateral drift38cmalso fails. Candidate
SHA`ca3ead55f2824b3e563a81ca6ab6a9b4b61299ebb4a15d5b7f77c5226042f626`.
`unified_pose_sparse_history_export_20261014_v1`contains48recordings/102verified
files, downloaded privately; archive SHA
`22df0bd539a08dea431d6a8fc90928a716acd32e9f4858e5d6e5d2a3fe24d91e`.

Oracle export`unified_entry_oracles_export_20261014_v2`has203verified files,
100recordings across20overlapping28-frame windows from10sources. Clearly labeled source,
normal model from source history and three OFFLINE truth interventions. Actual
four-frame source history is displayed, followed by24UNCHANGED generated frames;
no future correction or padding. The v1export lacked per-arm frame metadata and
the viewer rejected it; v2fixes metadata only, v1preserved. Archive SHA
`c6a2000f0ea434bad555b1a6f9fa262509a11f86542efa296dc02086ba5de344`.
Actually viewed source versus normal actor at frame18of`22928f55fcd9aa9b-up-start24`;
saved`source-vs-actor-frame18.png`. Both articulate bent legs, but normal actor
Root is1.092mversus source1.206mat this frame. This source-history diagnostic is
not a current WASD repair or a complete route.

Conditional joint test: previous separate Root/Pose fits and oracle factors show
both trajectory and token-selection errors. Test joint Root-stage1/stage2/Pose
ADAPTER updates with the existing native CE + native decoded-motion + native
Root objective coefficients. This explicitly extends the paper's standalone
Pose CE; no new coefficient tuning, geometry penalties, source Root actor inputs
or codec adaptation. Shared scene encoder and native Root/Pose/codec remain
frozen.722,816trainable parameters. Fixed4-history/4-pass runtime unchanged.

`unified_root_pose_joint_gate_20261014_v1`COMPLETE,0updates: up/down/idle/walk
actor-ST forward difference0, future-target actor difference0, native loss
equivalence and all freeze checks pass. Motion and CE give finite gradients to
Root-stage2/Pose. Root-stage1only has~1e-9native Root gradients at this fixed
horizon; do not claim strong learning there. Pose-height detach remains; main
Root features and decoded-motion/Root losses supply other paths. ST is a
surrogate derivative through hard tokens, not exact argmax differentiation.
25targeted local tests pass. Preflight case/initial hashes match the control.

`unified_root_pose_joint_20261014_v1`is COMPLETE. New immutable code
`code_unified_root_pose_joint_20261014_v1`, freshGPU2launch PID1516063(identity
hint only). Same original zero-Pose-residual initialization and corrected
schedule/data/300updates/batch4/1200exposures as the coverage control. This is
a joint adaptation candidate, not a single-loss causal ablation: both trainable
scope and the two existing native objectives change. Final review is complete;
do not duplicate or overwrite snapshots.
Its v1progress field`mean_last10_ce`contains the TOTAL joint loss; final named
`component_losses`are authoritative. The local future runner now calls this
`mean_last10_loss`; the completed immutable v1snapshot is not overwritten.

Final source/corruption hashes exactly match the coverage control; all protected
hashes and full checkpoint reload pass. Candidate
SHA`3364bd8c55e3efb29db1fb4e3df464daa7c0abbd7e387e73bd0f55726c8d9dc4`.
Both live goals finish, but peak31.95/44.99cm at0.8/1.6m/sFAIL. On-stair mean
knee24.23/28.00deg still lacks the required change from flat gait; ankle-below-
tread fractions40.79/64.10%are diagnostic evidence, not collision certification.
The comparison control is23.24/28.94deg,24.95/52.52cmpeak. No consistent repair.
Last10updates average CE0.90/motion0.07/Root0.03does not demonstrate natural gait.
The17flat sequences complete with finite output; release travel~0.73m includes
braking. No new local WASD model is deployed and28941weights remain unchanged.

`unified_root_pose_joint_export_20261014_v1`contains48recordings/102files,
downloaded through owned private Drive and hash verified. Archive SHA
`5449e3e7517b12967c26e57895a2325ff98f8299d3efb95c96d6f22eadf8af83`.
URL: <http://127.0.0.1:28936/web/terrain_route_review/?data=/out/parc_rugged32_20261013_v2/unified_root_pose_joint_export_20261014_v1/visuals/&clip=live-stairs-speed1.6&left=control&right=joint>.
Actually viewed frame144and saved`stairs-frame144.png`: both still show flat-like
leg movement and substantial immersion. The comparison does not visually pass.

No training or diagnosis remains active. Do not repeat these completed budgets.
The primary user-visible issue is STILL autonomous flat-to-stair gait entry,
not a few residual penetration centimeters. Source-history knee amplitude,
offline oracle quality, lower CE and successful process execution do not solve
that issue. Before another fit, audit concrete source-codec reconstruction and
terrain/support supervision on actual decoded outputs; the unified objectives
so far still have no explicit collision/support term. Existing oracle errors
do not prove the entire vocabulary incapable, nor justify indiscriminately
opening the whole model or inventing recovery labels. Preserve source clocks,
all failed cases, fixed-time controls and the already approved live version.

## Earlier Work: Native Sparse Pose History Test

The user correctly rejects the coverage candidate as still showing flat gait
on stairs. It is not a successful stair-motion repair. All older completed
candidates and live28941remain preserved; no IK/runtime correction is added.

`unified_entry_oracles_20261014_v1`completed20genuine entry-window factorials
on the corrected-coverage Pose model. Normal actor exactly reproduces the
archived metrics, weights unchanged,0updates. Mean root-relative leg errors:
normal10.27cm, source Root only6.24cm, source tokens only4.41cm, both3.76cm.
Mean knee-angle errors19.66/12.21/7.63/6.94deg. Source Root/tokens are explicitly
OFFLINE truth interventions; neither is fed into the normal actor. Even both
oracles have mean per-window skin peak4.26cm and worst14.83cm, so a separate
codec/conditioning geometry limitation remains. Those means are not passage
success rates. The source-history actor's mean up-knee69.82deg versus source
69.38deg shows available bending amplitude, not correct timing/foot placement.
Actual live-history stair means remain near29deg; the contexts are different.

`unified_pose_history_20261014_v1`completed8actual live contexts at0.8/1.6m/s.
Only Pose's history hints were hidden OFFLINE; Root and codec actual full
conditions are exactly unchanged and normal output replays archived frames.
42..55of64tokens change. At1.6m/sframe96mean knee23.69->45.00deg and new-frame
skin peak41.13->31.06cm. Other cases worsen or change little. This is evidence
of strong history dependence, not approval to remove runtime history. Root and
codec input parity is measured; no optimization, smoothing or future truth.

The native MB source was rechecked: Pose training samples start keyframes;
the loaded configuration has max4/no-start-probability0/warmup200000. The online
extension instead always supplies all4. A bounded single-factor test now reuses
the ORIGINAL`sample_keyframes`helper's mature distribution over1..4start frames,
restricted to the factual first4. Root and codec keep all4; runtime Pose also
keeps all4. This is a training regularization hypothesis, not an inference fix.
No full-budget restart and no unfreezing of additional modules.

Run`unified_pose_sparse_history_20261014_v1`completed under immutable
`code_unified_pose_sparse_history_20261014_v1`. Same original zero-Pose-residual
initialization, corrected window schedule, data, native CE,300updates/batch4/
1200exposures as`unified_pose_schedule_20261014_v1`; initial report/logits must
match before optimization. Only teacher Pose history masking changes. Independent
RNG preserves source order and corruption draws; final hashes must confirm it.
Seven local tests pass, including unchanged input/Root tensors, no future hints,
repeatability and RNG preservation. FreshGPU2receipt PID1467468is an identity
hint only; check cmdline/state before actions. No claim of quality until final
full generated-history, flat, continuity and official-skin review.

## Previous Work: Source Window Coverage Repair

The placement pair is COMPLETE; neither is adopted. Each completed300updates/
1200exposures. Baseline reports/logits, schedule and corruption hashes match;
protected weights and reload checks pass. Input-stage peaks at0.8/1.6m/s are
33.14/54.68cm; output-stage29.10/57.44cm. Both reach the goal, neither passes
geometry. Review-route final-token CE2.204/2.091 does not establish gait quality.
Input SHA`9ebc1f2c2e86bafb52ab6f0323e5bd8eb8065adae639415fc9f6207b68469e79`;
output SHA`f3d0fe7d16042085849e8b59234b49b468301d71431f85ed6ed1e893f2b8f5c1`.

`unified_pose_placement_export_20261014_v1`contains48official-SOMA recordings,
including real sources, original28941, matched zero residual, input and output.
All102files were downloaded through owned Drive/proxy7897 and verified;
archive SHA`45cb69de9788fec5ad8d4219734f6bb79f2f884cd67f126e99e93239f8f15261`.
Original lengths and failures are retained. This is research playback, not a
new WASD deployment. Current28941and28936still return200and are preserved.

A concrete sampling defect was discovered and reproduced against the ACTUAL
archived schedule in`unified_pose_schedule_audit_20261014_v1`: source selection
and early/middle/late window selection both used global update modulo periods.
With6up sources, each source receives only one third:50/0/0,0/50/0or0/0/50.
Each of6BONES likewise receives only one third of its clip. The4down sources
do cover all thirds. One up source(`529af1b219d0f459`)had0first-riser exposures.
These are real-time thirds, not anatomical gait-phase labels. This defect
does not mean every old stair entry was absent or explain the entire failure.

The fixed sampler rotates thirds by each source's OWN occurrence count. The
same300updates give17/17/16per up source and34/33/33per BONES source. Source
identity order and mask seeds are exactly unchanged; only real source windows
change. No source/time transformation. Six focused tests pass, including
2/2,6/4and6/6stair counts, coverage and complete checkpoint loading.

Bounded single-factor fit`unified_pose_schedule_20261014_v1`is COMPLETE. It uses
new immutable`code_unified_pose_schedule_20261014_v1`, input-stage adapter-only,
same zero-residual initialization/data/loss/300updates/batch4/1200exposures as
archived`unified_pose_input_20261014_v1`. Baseline report/logits must match before
the optimizer. Root/codec/shared encoder remain frozen. FreshGPU7launch receipt
PID808976is only an identity hint; check cmdline/state before acting. Never
overwrite older snapshots or restart the full budget. This is a validated
sampler repair. Initial report/logit equality, eight-step Root/future isolation,
final protected hashes and full reload pass. Candidate SHA256:
`d271117c8a363dc8609a0300c39414adc240d72e6b0994fbfb56a2a900c53a13`.
At0.8/1.6m/speak is24.95/52.52cm versus control33.14/54.68cm. Both finish the
route, but both FAIL. Review final-token CE2.204->2.020 is not gait success.
Stair knee mean at1.6m/s28.09->28.94deg; ankle-below-tread fraction70.39->72.97%.
Flat17cases have complete finite output, but this alone is not a flat quality
certificate. Release48-frame travel0.749->0.781m includes braking, with last8
speed about0.03m/s. No additional training is active and this model is NOT deployed.

Complete coverage comparison`unified_pose_coverage_export_20261014_v1`contains
48official-SOMA recordings across10clips, with sources/original/zero/control/
corrected coverage.102files downloaded through private Drive and verified;
archive SHA`9039118b841326a7067568c193dcdf91ab0f2b4913546d96dea5b76da4844ee3`.
URL: <http://127.0.0.1:28936/web/terrain_route_review/?data=/out/parc_rugged32_20261013_v2/unified_pose_coverage_export_20261014_v1/visuals/&clip=live-stairs-speed0.8&left=control&right=coverage>.
Actually viewed frame304at0.8m/sand saved`stairs-frame304.png`; both show
insufficient stair articulation and visible submerged legs. Different real
lengths460/468are preserved. The original live checkpoint was rehashed locally
and still equals`ade10ff143d855073b0760ca3f6a12cc8b1f59d7627eb883f9e981579edc5979`.

`unified_pose_candidates_20261014_v1`is COMPLETE,0updates,102saved candidates:
original and output-stage model;3factual histories;argmax plus16native Gumbel
samples per history. Predicted Root is exactly identical across all interventions.
At frame48(before stair entry), best new-frame peaks1.63/1.81cm do not certify
full traversal/support. At frame72 best14.25/14.14cm; frame96best36.06/36.55cm.
Their inherited histories already peak2.73/14.44cm respectively, reported
separately from new frames and boundary sweep. Finite sampling cannot prove
no feasible motion exists, but does not support changing runtime to stochastic
sampling or treating sampled motions as clean recovery expert labels. No source
future or optimizer, unchanged weights, deterministic replay verified. Diagnostic
script lives separately from the immutable placement training snapshot.

`unified_root_pose_pair_20261014_v1`also COMPLETE,0updates: a factorial of
original, previously learned approach Root, corrected-coverage Pose and both.
Only previously learned adapter WEIGHTS are combined; no trajectory replacement,
future source actor condition or runtime change. Other adaptation tensors are
checked exactly equal to the original.3actual live histories plus20real source
entry windows give92saved primitives. Same-history Root is exactly unchanged
by Pose choice. At live frames48/72/96, both peaks2.29/13.78/34.91cm versus
original2.73/14.53/37.30cm. Thus this is not a usable repair either.
Across20source entry windows, mean Root-Y RMS8.74->7.56cm; mean root-relative
leg error11.61->10.12cm; mean per-window skin peak14.19->6.72cm. Those are
averages over source histories, not worst-case collision or closed-loop passes.

Next evidence-based decision: the validated sampler must remain fixed, but
more repetitions of Pose-only CE are not justified. Need a feasible Root/Pose
conditioning and decoded-support objective check on genuine source entries
before any further small co-adaptation. The MB paper's source-Root Pose teacher
versus our predicted-Root-only extension is a remaining hypothesis, not a proved
bug; future source Root may only enter a clearly isolated training/diagnostic
teacher, never the normal actor. Preserve the codec ceiling failure, actual
feedback mismatch, all failed clips and the unchanged live28941weights.

## Earlier Execution Record

Latest user steering: requested/observed speed must not become a restriction on
natural ascent/descent. Allow learned slowing and acceleration while maintaining
real forward traversal.0.8/1.6m/sare regression requests, not acceptable-output
speed bounds. Keep source motion at its genuine speed/clock; do not time-warp
it into those values. Live28941already advertises a12m/srequest upper limit,
verified via`/api/config`; generated motion is not clamped to0.8..1.6m/s.
The internal terminal-velocity conditioning is being audited independently;
do not call an enlarged UI slider a learned speed-adaptation repair.

Latest source milestone: the bounded collection has completed all32candidates.
29passed existing physical-recording/SOMA admission;19then passed complete-route,
lateral and2cmfull-skin checks across8new terrain groups. The exact report is
`parc_pose_entry_20261014_v1/exact_source_audit_all_v1/complete.json`. The merged
native dataset is`parc_pose_entry_20261014_v1/dataset_v3/parc_native`, manifest
SHA256`9f20df7fa9e67176f759c1428681b2112aea4787a16ea6cd723d455fddab1e2a`.
No speeds or frame clocks were altered. Strictly rejected new records are
excluded before window selection. `unified_pose_data_20261014_v3`is preparing
the expanded split-preserving pilot list; its count is not assumed in advance.

Expanded curation is COMPLETE:6up/4down/6BONES in
`unified_pose_data_20261014_v3`, retaining the original screening and excluding
all known review families and newly rejected full sources. The original12-stair
target remains short by2down routes; no duplication or speed filtering fills it.
The data-only capacity comparison`unified_pose_diverse_20261014_v1`was launched:
same original live initialization, same capacity scope as B, native CE/masking,
300actual updates/batch4/1200exposures,50/50PARC/BONES. Only the training source
set changes from4to10stairs. It uses immutable
`code_unified_pose_capacity_20261014_v1`, and does not continue failed Bweights.
Inspect cmdline/state/complete before acting; no other training is active.

That expanded-data comparison is now COMPLETE and not adopted. All protected
weights/reload checks pass; source count expansion alone does not repair gait.
Live peaks are31.39/51.33cmat0.8/1.6m/s; review-route final CE2.632 versus the
smaller-data capacity arm2.585. This is a fixed-budget comparison, not proof
that diverse data cannot help with sufficient appropriate training.
Checkpoint SHA256:`b95446a4687173bbcb73e94fa6b4eb5887cf1d9f35108cd17d43d1f296b46577`.

The next representation test targets the measured weak SDF-to-Pose influence:
same328,320-parameter Pose SDF residual at transformer INPUT versus immediately
before the final token classification projection OUTPUT. No other native
modules/encoder/codec train. Both begin by zeroing the same Pose residual output
matrix after loading the preserved initial weights; their baselines must match
each other, but are not claimed bitwise identical to live28941's nonzero Pose
residual. Root/decoder adapters remain original. Same expanded source set,
300updates/1200exposures/native CE/corruption schedule.31local tests pass,
including default-stage compatibility and rejecting checkpoint placement mismatch.
The stage is explicit in the checkpoint contract; live28941uses the unchanged
default input stage. Inference-only relocation of trained weights is not adopted.
Code:`code_unified_pose_placement_20261014_v1`; initial real-model zero-residual
gate:`unified_pose_placement_gate_20261014_v1`. Launch only after it completes.

The gate is COMPLETE:16windows have exact0world/logit difference between the
two placements before training. The matched pair was launched on freshly
checked GPUs2/7: `unified_pose_input_20261014_v1` and
`unified_pose_output_20261014_v1`. Each is an adapter-only300-update fit. Check
actual process cmdline and state; do not duplicate or modify the code snapshot.

The user approved the review/plan and requested execution. Preserve the original
28941 candidate and all review pages. No IK, Root replacement, output smoothing,
future-truth actor conditions or full-budget restart.

## Preconditions

Completed `unified_pose_preflight_20261014_v1` using the unchanged live candidate
and immutable `code_unified_pose_preflight_20261014_v1`. Eight source windows
(six stairs, idle, walk) have exactly four normal Pose passes. The explicit
all-mask teacher reproduces first-pass logits within the declared float tolerance.
Future-target actor changes are zero; ordinary replay is exact; no optimizer
was constructed and no weights changed.

Both proposed scopes receive finite, nonzero Pose gradients on up/down/idle/walk.
P-A trains 328,320 parameters; P-B trains 17,219,537. P-B opens the existing
last two native Pose layers, output head and Root/height projections plus the
same Pose SDF residual. Root, shared encoder and codec stay frozen.

The source-codec diagnostic exposes a separate quality limitation. With source
tokens/Root and the decoder adapter active, source up/down/train peaks are
0/2.96 cm; four review cases peak 9.87/3.53/2.13/1.96 cm. Disabling the adapter
offline gives 0.78/4.53 cm and 10.84/3.79/2.62/4.00 cm respectively. Therefore
simply disabling the decoder adapter is not a repair, and even correct tokens
cannot be advertised as guaranteeing the 2 cm full-skin screen. Pose capacity
testing remains meaningful for action selection, not sufficient for deployment.

First-pass all-mask CE is not uniformly worse than the native corruption
teacher; do not prematurely change the official mask curriculum. Final-pass
CE worsens on some review cases and remains a separate diagnostic.

## Data

`unified_pose_data_20261014_v1` scans the current PARC database and finds two up,
one down and six BONES clips. `unified_pose_data_20261014_v2` also checks the
existing mixed database and finds two up, two down and six BONES clips. The
requested six up/six down target is NOT met. No gate or split is relaxed.

The current bounded pair uses those four independently grouped real stairs as
a smaller capacity diagnosis; it is not the planned broad generalization gate.
Additional eligible PARC generation remains necessary before expanding training.
Every selected source keeps real full approach/exit frames and passes exact
18056-vertex, 8-substep geometry screening. Existing review-only families and
the three archived context-review families are excluded as complete groups,
even where the original manifest called them train. Six independent BONES
clips cover idle, walk, run, turn, start and another walk on fixed planes.

## Implementation

`data/unified_pose_training.py` defines explicit parameter scopes, causal
predicted-Root examples, native CE/mask draws and a strict new checkpoint schema
`motionbricks.unified_terrain.pose_capacity.v1`. Checkpoints contain every changed
native Pose tensor as well as adaptation state; the live adaptation-only loader
must not silently load them. Three local tests cover exact freeze boundaries,
complete native-tensor loading and repeatable 50/50 exposure schedules.

`scripts/pilot_unified_pose_capacity.py` uses the same original live initialization
for both arms, 300 actual updates per arm, batch 4, 1,200 exposures, Adam 5e-5,
and identical source windows and corruption seeds. Each batch is one up, one
down and two BONES. Native focused-token CE only; no silently added geometry
loss. Eval-mode dropout is retained for deterministic conditioning and is an
explicit adaptation choice. Root predictions are checked unchanged after eight
updates; all protected hashes and checkpoint reload are verified. Original
eight source reviews, two live stair speeds and 17 flat cases run before/after.

## Current Status

Both arms completed300actual updates/1200exposures after fresh idle-GPU checks: `unified_pose_adapter_20261014_v1`
on GPU0 (PID191158) and `unified_pose_capacity_20261014_v1` on GPU2 (PID191566).
These are identity hints only; check cmdline, state and complete before acting.
Immutable code: `code_unified_pose_capacity_20261014_v1`. Baseline reports,
exposure schedule and corruption hashes match exactly. Frozen hashes, eight-step
Root replay and full native-Pose checkpoint reload all pass. Neither arm is adopted.
Initial SSH connection closures
returned no receipt; a read-only check confirmed no arm existed before retry.

| Live ascent | Before | Pose adapter | Added native Pose capacity |
| --- | ---: | ---: | ---: |
|0.8m/s full-skin peak |30.95cm|34.44cm|27.77cm|
|1.6m/s full-skin peak |46.80cm|45.02cm|50.36cm|

All reach the post-stair goal; all fail geometry. Stair-section knee means stay
near28degrees at1.6m/s, with71..72%ankle-center samples below treads. Thus improved
token fitting has not produced the required autonomous change of stair gait.

Across12windows from the four existing review-only routes, native corruption /
all-mask / final-generation CE are1.413/1.715/2.283before,1.276/1.577/2.171for
the adapter arm,1.349/1.658/2.585for the capacity arm. These full-token CE values
are compared on the same labels (not focused CE versus all-token CE). The larger
capacity arm does not resolve the generation gap. Frozen Root remains a known
independent limitation; no causal claim is made from CE alone.

`unified_pose_downhill_20261014_v1`completed six additional closed-loop requests
from the measured top platform at[-20,-100],headingpi,to Z<=-112. Only the initial
flat seed is rigidly placed at the measured spawn; no subsequent correction.
Before/A/B all reach the goal but lateral drift is0.90..1.38m, failing the25cm
screen. Peaks are1.83/2.39cm before,4.21/2.86cm A,1.49/2.06cm B at0.8/1.6m/s.
Low penetration here is not a pass or permission to bypass the route.

`unified_pose_outcomes_20261014_v1`extends recorded flat reporting. During the
48-frame release interval, before/A/B move0.799/0.759/0.791m and end at
0.0303/0.0305/0.0351m/s. These include braking travel, not pure stationary drift.
Foot ranges and acceleration remain per-case diagnostics; flat preservation
is not approved from these averages alone.

The authorized conditional mask experiment is complete and not adopted:
`unified_pose_mask_20261014_v2`,code`code_unified_pose_mask_20261014_v2`,GPU2,
PID254173(now historical only). Same original initialization/adapter-only scope,
300updates/1200exposures and source schedule as A; replace25%of corruption draws
with all-mask while keeping75%native. Exactly150stairs+150flat full-mask examples
are stratified by source and early/middle/late phase. Tests verify assignment.
The first mask launch(v1)was cancelled during baseline review, before any
optimizer update, after noticing slot rotation correlated with source parity.
Its log/state remain; no training snapshot was edited. Do not duplicate v2.
The completed full-mask follow-up has identical source schedule and baseline
to A,300full-mask draws, protected hashes and checkpoint reload error0. Peaks
are30.53/44.38cmat0.8/1.6m/s; both reach the goal but neither passes. Review-route
final CE2.146 versus A2.171 is a small change, not resolved generation or gait.
Checkpoint:`e793306aa48ca017adc1f6ef69b235ff6ac8b9ccfa38c22fcfbbe17c604ef45b`.

## New Source Collection

`parc_pose_entry_20261014_v1` prepares eight new all-train geometries, each with
6mflat approach,5mtop platform,6mflat exit,5..6steps,12..18cmrisers and30/40cm
treads, including both horizontal axes. Eight unique heightfields and route
surface checks pass. Existing review geometries remain reserved. Four native
PARC candidates are initially dispatched on layout0only; the other seven are
not automatically started. Command logs remain unit directions, not m/s.

Layout0completed native generation, official physical tracking and existing
SOMA retarget screening:4/4geometric complete routes,4admitted records,34.67s,
maximum native prefix error1.79e-7. This is one terrain group, not four independent
geometries, and not yet approval under the stricter full-skin/entry-window screen.
Supervisor completed PID321117; code`code_parc_pose_entry_20261014_v2`immutable.
The first supervisor launch used an older snapshot CLI and exited before any
generation; its log remains. The new snapshot includes the existing verified
prefix-preservation and physical-tracking wrappers, without vendor changes.

Native conversion initially rejected a generated-only catalog because the
established dataset contract requires both official/generated origins. The
correct incremental conversion retains the existing official base records and
normalization in`layout_00/native_v2`; no validation bypass or split relabeling.
Exact source audit will assess only the four new records, keeping original
frames and requiring complete up/down passage and real approach/exit.

The exact audit is complete:`layout_00/exact_source_audit_v1`. All four real
records cover up and down plus return to flat; three pass full-route peak2cm
and lateral35cm source screening. Their peaks are1.846/1.425/0.808cm; the fourth
is retained with2.525cmpeak and is excluded from strict source admission. Real
flat approaches are42..82frames; source clocks/speeds are unchanged. The three
passing records are still ONE independent terrain group.
GPU0and7became occupied by other users after collection; no other task was
modified. The exact audit ran on freshly checked idleGPU2 instead.

After this bounded successful source check, the remaining7prepared geometries
were dispatched on freshly idleGPU2using the same32candidate total plan:
`parc_pose_entry_remaining_20261014_v1`,worker`worker_2.json`,layouts1..7.
This only generates, physically tracks and retargets sources. It starts no
model training; exact full-skin/route/split checks are still required on every
new record before assembling a next Pose training set.

Worker2completed layout1generation(4/4routes) but its physical stage's single
GPU-readiness check rejected occupancy. This is an execution/availability
failure, not a rejected motion. Its processes exited and all artifacts remain.
The physics helper now waits up to57susing the SAME memory/utilization limits;
it never kills or takes another process. The changed helper is in new immutable
`code_parc_pose_entry_20261014_v3`.
After fresh occupancy checks, remaining layouts are reassigned without overlap:
worker4handles1,2,3; worker5handles4,5,6,7; completed worker0retains0. Layout1's
existing generated candidates are reused; physical recording writes a new
`recording_v2`directory, preserving the failed`recording_v1`attempt. Worker2is
historical and must not be restarted over the new assignments. Dataset finisher
`finish_parc_pose_entry_collection_v2.py`checks exact0..7coverage from workers
0/4/5and builds`dataset_v2`; it starts no model training. `dataset_v1`retains
its failure report from the prior worker2attempt.

Latest resource recovery supersedes workers4/5: worker4exited before touching
layout1because the existing directory required explicit`--resume`; worker5
finished layout4generation and physics/export, then timed out waiting for a
GPU now occupied by another user. Their failed states/logs remain. Neither is
active. A retirement helper found worker5had already exited and sent no signal.
Freshly idleGPU7now resumes ALL remaining layouts1..7using the preserved
generation/physical outputs and explicit`--resume`. Active supervisor:
`parc_pose_entry_worker7_20261014_v1`,PID499798(identity hint),same codev3.
Only worker0(completed layout0)and worker7are authoritative for current progress.
Detached CPUfinisherPID501386checks workers0/7and writes`dataset_v3`; its exact
command/log are in`dataset_v3.process.json`. Earlier finisher attempts remain
under`dataset_v1/v2`and are not the current data artifact. No source or actor
quality failure was hidden by these execution recoveries.

## Speed Coverage

`unified_pose_speed_coverage_20261014_v1`is read-only and complete. Four training
stairs have median terminal speeds3.33/2.69/1.88/2.40m/s. Fractions in1.4..1.8m/s
are4.1/8.2/26.0/5.2%; in0.6..1.0m/s2.0/0/6.8/2.1%. The low-speed condition is
underrepresented, but not absent. Same-history commands0.8/1.6/2.4/3.2m/sstill
do not produce a reliable stair gait; at live frame72Root peaks stay below1m.
At frame88a3.2m/scommand raises Root but knees remain near33.6degrees. Changing
command also changes lookahead, so this does not isolate geometric causality.
Do not silently increase the user's requested speed as a fix.

`unified_terminal_velocity_20261014_v1`checks the user-approved speed-freedom
question on six actual histories. Removing ONLY the final decoder velocity
assignment leaves Root predictions and all64pose token entries unchanged;
Root-relative leg differences are0.061..0.531mm, knee differences0.006..0.064deg.
The main requested-speed condition still enters Root, so this is not a full
soft-intent protocol test. It does rule out treating that last assignment as
the main explanation for the missing stair gait. Nothing changed in28941.

`unified_speed_freedom_20261014_v1`completed a wider closed-loop review for the
original and full-mask candidates: requests0.4/0.8/1.6/3.2/5m/s, up to60seconds
per recording, no generated-speed bound. All ten reach the goal, but none pass
geometry/lateral checks. The full-mask candidate peaks19.60/30.53/44.38/64.56/
102.38cm respectively;0.4m/salso drifts1.40msideways. Thus neither speeding up
nor allowing a longer slow traversal cures the gait. Completion-time limits
are reported explicitly; slow speed itself is not a quality failure.

## Local Visual Evidence

Read-only`approved_stair_recovery_20261014_v1`also tested the preserved,
user-approved stair checkpoint`ae558ee8...`on the current map and flat history.
Its complete rollout reaches the Zgoal but drifts5.58msideways and peaks18.62cm;
it is not accepted as a recovery-label teacher for these states. Three one-block
probes are saved; their skin metric INCLUDES the inherited four input frames,
so it must not be read as purely newly introduced penetration. This new-map
diagnostic does not revoke the user's prior approval of that model's original
visual examples. No old server or checkpoint was changed.

The first new PARCsource group is locally available as
`pose_entry_source_export_20261014_v1`:11verified files, archive SHA256
`7f273927dda8d58f7f12fbf58555b1843b6ed1b1e55eea45bc4827ade7509122`.
Source-up frame53was visually inspected. The viewer status now says
`Source motion recordings`for explicitly source-only manifests; mixed model/
source pages say`Recorded motion`. This changes labels only, not animation.

`unified_pose_export_20261014_v2`contains56official SOMA recordings: real sources,
before and all three Pose candidates, two live up/down speeds, actual lengths
and all failures.120files verified through the private Drive archive SHA256
`38a874265347af8bdc0639cd6b48cbbcbb1d6c9ce942486870310e54f816f314`.
The earlier A/B export has94verified files and archive SHA256
`9ac0fc30ef89c68ef3727b2b559dc9a1523d50d1457869b56686884e5685ba56`.
A/Bstairs frame144was visually inspected and saved; the larger Pose arm still
penetrates. Source playback and trained output remain distinct labels.
