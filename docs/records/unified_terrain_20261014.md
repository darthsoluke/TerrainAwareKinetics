# Unified Terrain Conditioning

## Latest Request: Approved Pose Repair In Progress

The user approved the [Pose repair plan](pose_repair_review_plan_20261014.md)
and requested execution. Read [Pose Repair Execution](pose_repair_execution_20261014.md)
for current status. The matched adapter/native-capacity pair and stratified25%
full-mask follow-up have each completed300updates/1200exposures and still fail
live stair quality. None is deployed. Native Pose checkpoint persistence,
freeze/gradient/Root parity and full-mask source allocation are now explicit.
Longer real PARC approach sources are being prepared; inspect the execution
document and remote state before any new work. Keep28941unchanged.

## Current Finding: Root Rise Is Not Stair Gait

The user correctly flags that both current comparison arms still look like
flat walking on the stairs. Do not describe those candidates as having learned
natural stair traversal, or treat Root rise as recovery of foot placement.
No model/runtime was changed and no training was launched in this check.

`unified_stair_gait_20261014_v3`completed a read-only chain audit of the exact
Root-stage and real-approach candidates shown on28936. Ten archived neural
plans replay using the official FK-reencoded histories (5per candidate), and
24real-source windows (12per candidate:6up/down routes x2starts) receive
separate offline Root/height/token interventions. All ordinary replays are
exact; archived plans match3e-5/1e-5tolerances; weights remain unchanged.

For the orange real-approach candidate, recorded flat approach frames28..75
have mean knee flexion28.38degrees; stair frames100..195have27.49degrees.
In the stair section Root Y spans0.979m, but per-foot Root-relative vertical
ranges are0.216/0.217m, close to the flat section's0.239/0.210m.71.35%of
ankle-center/time samples lie below their queried tread, with33.73cmmaximum
depth. These ankle statistics are not full-skin certificates; the independent
full-skin failure remains41.57cm. Knee statistics alone are not a classifier,
but the unchanged articulation and failed support agree with the visual issue.

Keeping the SAME generated history, command and Root trajectory, replacing
terrain by the current support plane changes knee flexion by only0.38..1.01
degrees and foot-relative positions by0.49..1.15cm over these5windows. Only
2..3of64decoded multihead token entries change. The same-history all-flat
counterfactual is not a valid flat demonstration or a published rollout.
It establishes weak direct environment-driven articulation on these states.

On12matched REAL source-history windows, the orange candidate's mean
Root-relative leg-joint errors are:

| Offline diagnostic | Leg-joint error |
| --- | ---: |
| Ordinary actor |8.096cm|
| Exact source Root HEIGHT only |7.918cm|
| Exact complete source Root |5.129cm|
| Exact source pose tokens only |3.814cm|
| Exact source Root and pose tokens |3.464cm|

Source Root/tokens never enter the normal actor or played history. Actual
decoded tokens are captured and checked against final Pose logits; oracle
tokens are checked against the real source encoder. Interventions are scoped
and restored, including exception cleanup (3focused local tests pass).
The improvement from tokens identifies a significant action-selection issue
under these source-history conditions. It does not prove that Pose alone is
responsible; correct full Root helps too, and even both oracles leave decoder
error. Token accuracy alone does not measure motion quality.

With real stair source histories, the same candidate already produces knee
flexion around43..70degrees across those12windows. Therefore the model is not
universally incapable of stair-like articulation. Reliable initiation from
flat locomotion on the live authored stairs remains absent. Differences in
history, scene and command between source/live tests prevent assigning all
of the gap to feedback history alone.

The two viewed trials train ONLY Root environment adapters with the native
Root objective. Pose, decoder, shared encoder and their adaptation modules
are frozen. This is a valid Root isolation test, not a complete stair-gait
repair. Do not continue this scope and assume correct Ywill teach the legs.
Next learning design must jointly address feasible entry supervision and
Pose token selection, with original Root/codec behavior and flat locomotion
explicitly checked. No blind full unfreeze, copied future targets into actor,
or runtime Root/IK correction is justified.

Local report:`out/parc_rugged32_20261013_v2/unified_stair_gait_20261014_v3-report.json`.
Plot:`unified_stair_gait_20261014_v3.png`and`.pdf`in the same directory.
Snapshot:`code_unified_gait_audit_20261014_v3`, completed PID12994only a
historical identity hint. V1failed before model inference because its new
snapshot omitted the existing Root-oracle helper; V2completed the initial
diagnostic and V3adds height-only/tokens factorial controls. All logs retained.

## Priority Repair Status

All label/coverage/Root-objective/height-readout trials below are COMPLETE and
failed live stair entry. None is deployed. The additional completed trial is
`unified_approach_fit_20261014_v1`: audited real24-frame earlier prefixes,
same original initialization,200updates/batch4/800exposures,50/50PARC/BONES,
native Root objective, frozen native Root/Pose/codec and shared encoder/body
adapters. Only Root-stage environment adapters train. It compares with
`unified_root_stage_20261014_v1`; held-out and live reviews remain unchanged.
Immutable snapshot:`code_unified_approach_fit_20261014_v1`. Launch PID4097175
is now only a historical identity hint; no training or diagnostic remains
active. No full budget was started. Its baseline report matches the
Root-objective control exactly; all frozen hashes and4future-isolation/
training-runtime checks pass with error0. Ten targeted local tests pass,
including preserving all original/reference frames during prefix extension.
Both live speeds reach the post-stair goal, but fail full-skin penetration;
this rules out adopting the candidate, even though slower motion is allowed.
Relative to the matched Root-stage control, low-speed peak improves33.09->
27.64cm, while1.6m/s worsens40.09->41.57cm. Prefix coverage alone is not a fix.
Candidate SHA256:`b9b7c813294d9b998eed5bf37259e4ff8a5c9e3ed55d6e45e8347affbc0b6a89`.

| Completed arm | Full-skin peak at0.8m/s | Full-skin peak at1.6m/s |
| --- | ---: | ---: |
| Original28941 |30.95cm|46.80cm|
| Legacy-label control |33.84cm|43.58cm|
| Matched terminal labels |31.74cm|46.87cm|
| Dense source windows |34.25cm|49.25cm|
| Native Root objective |33.09cm|40.09cm|
| Root height readout |34.03cm|49.02cm|
| Real earlier approach |27.64cm|41.57cm|

The height-readout arm opens only1537effective parameters in native Root's final Y
readout, in addition to Root adapters. Other output rows are bitwise protected.
Baseline and exposure schedule match the Root-objective control exactly;
all frozen hashes and actor future-isolation checks pass. Its separate
`motionbricks.unified_terrain.root_height_readout.v1` checkpoint schema must
never be loaded as adaptation-only. These checks are not quality approval.

### Entry Learning And Causal SDF Audit

`unified_entry_learning_20261014_v3` is the authoritative read-only audit:
78contexts/interventions across original, Root-stage and height-readout models.
The same actual history, command and query locations are kept; replacing only
new SDF distances/normals by the current plane keeps native heightfield and
vertical origin factual. A separate all-terrain plane intervention is reported.
All interventions are discarded; weights unchanged,0updates, exact replay.

For original28941at logical frames64/72/80/88, SDF-only Root-Y effects are
0.143/0.100/0.291/0.343mm; all-terrain effects are18.26/25.12/35.28/66.71mm.
The all-flat origin change is0at these entry states. Actual SDF and encoded
memory already differ from plane. New SDF influence is weak on these states,
while the existing heightfield path is much stronger. These nonlinear
interventions are not an additive attribution or proof of a single cause.
Fresh replans remain late, so browser latency/cache alone is insufficient.

Live history is re-encoded from archived played FK with the exact publication
helper; frame72's normal Root prediction matches the archived neural plan
with error0. V1/V2used raw redundant recorded features directly and are
superseded for runtime conclusions. Their reports remain as diagnostic history.

Entry coverage is anchored to the INITIAL flat floor, not each level stair
tread. Each route has24overlapping windows whose published future changes
ground height while history remains on the entry floor: old-relative starts
-7..16. The17nonnegative starts received72exposures per route in the dense
200-update control; the earlier7received0. Thus entry is not wholly absent,
and additional coverage is a bounded hypothesis, not an established fix.
Both real24-frame prefixes pass a fresh full18056-vertex/8-substep screen:
peaks1.046/1.629cm, no unknown/uncertain samples, original overlap error0.
Training-only windows become73per route instead of49; review cases stay81
frames. They are overlapping windows, not new independent demonstrations.

`unified_root_placement_20261014_v2` tests the same adapter weights moved
from before the Root transformer to before its convolutional decoder, on
20source/live contexts. It does not fix entry or increase entry-state SDF influence.
The existing native heightfield also conditions after the transformer, but
code ordering alone does not prove signal suppression. No placement change
is adopted. V1used raw recorded features; V2uses actual FK re-encoding.

Local reports are under`out/parc_rugged32_20261013_v2/*-report.json`.
`unified_priority_review_v2`contains12official SOMA recordings/all6completed
arms at two speeds, original lengths and every failure retained. Private Drive
archive SHA256`13aa2507a74a50214140ae6349a63c756cd7c26ee02558e70b2b392484647af5`;
32files verified locally. Frame111was inspected and saved as
`unified_priority_review_v2/entry-frame111.png`; both current and height-readout
candidates visibly penetrate. No motion correction or resampling was applied.

The complete latest export is`unified_priority_review_v3`, adding the real
earlier-approach candidate:14recordings,7arms,two speeds,37verified files.
Archive SHA256:`35706bed7c523d220f8d3fc2531e079d7d1b26036cdb8c8727268f29e1a936d9`.
Matched Root-stage vs approach frame111was visually checked and saved as
`unified_priority_review_v3/entry-frame111.png`; both still penetrate.
Background review URL:
<http://127.0.0.1:28936/web/terrain_route_review/?data=/out/parc_rugged32_20261013_v2/unified_priority_review_v3/visuals/&clip=live-stairs-speed1.6&left=root_stage&right=approach>.
Source/SDF attribution chart:`unified_entry_learning_20261014_v3.png`and`.pdf`;
these are diagnostic plots, not improved-motion demonstrations.

Both local services were found absent during this follow-up; restored the
unchanged28941with verified CUDA (PID157044) and static review28936(PID64108).
28940remains closed. New live logs:`server-cuda-v4.*.log`. HTTP200and original
candidate hash confirmed; CUDA replay revalidation keeps identical tokens and
maximum FK error1.53e-5m. Process IDs remain identity hints only.

### Investigation Boundary

The training/runtime forward contract now agrees on controls, units, frames,
FK history and future isolation. This does not make source-history imitation
equivalent to closed-loop control: all current small fits still use just two
stair demonstrations plus two real BONES clips. The new scene features are
observed but have little influence on the entry Root forecast. Current tests
do not establish whether diverse real entry states, the learnable geometry
encoder, or a different causal supervision curriculum is the principal
remaining limitation. Do not declare the native Root representation broken,
remove the old terrain path, multiply SDF weights, relocate adapters, or
repeat the same200updates as if any of those had been validated.

Keep the MB Root->Pose->codec chain and original live endpoint. Any next
learning comparison must address the measured weak environment dependence
with feasible real source histories and explicit flat preservation; genuine
entry/traversal remains the gate. No output Root/IK/collision corrections.

### Earlier Priority-One Comparison

Priority-one A/B is COMPLETE; neither candidate is deployed. Identical baseline
reports (all numerical differences0) and source-window schedule hash
`d2b72a38eb308836be8bda309f67226a1a0651df09f08db36b0dc0d69b190c37` verify the
comparison. Both8-step preflights have training/runtime forward difference0,
future-height/body-pose actor difference0and protected tensor hashes unchanged.
All final protected hashes also pass.

Correcting labels improves the17case flat review's1.6m/s subset: average
absolute speed error0.26447->0.08907m/s (legacy control0.26295m/s). This is not
a claim of unchanged gait; foot range also changes and has not been visually
approved. Live world skin peaks do not improve:0.8m/s30.95->31.74cm and
1.6m/s46.80->46.87cm, compared with legacy33.84/43.58cm. Corrected low-speed
run has not reached the goal within the bounded16sreview; do not reject a
candidate solely for being slower or confuse fixed-time source progress with
complete-route feasibility. The penetration failure independently prevents
deployment. Label consistency is repaired, but entry anticipation is not.

Priority-two controlled coverage run is complete and not adopted:
`unified_goal_coverage_20261014_v1`, immutable code
`code_unified_goal_coverage_20261014_v1`, physicalGPU0, launch PID3479096only
an obsolete identity hint. Same original live initialization, corrected labels,
200updates/batch4/800exposures, native losses, freeze scope and review. Only
source-window stride changes24->1: two stair sources now supply98overlapping
windows instead of6, not98independent motion sequences. No new source frames
or yaw augmentation are included. Compare against completed terminal arm,
not against its trained checkpoint as an initialization. Inspect state and
complete before acting; do not duplicate or overwrite the snapshot.

A separate CPUread-only audit confirmed that24earlier real source frames can
extend each approach (`audit_unified_approach_prefix.py`); it requires exact
overlap with existing data, a known flat approach and the same full-skin2cm
screen. That audit did not alter cases or train; the explicitly separate
approach trial above now uses only these audited real frames. No synthetic recovery.

## Priority One Goal Alignment Repair

The user explicitly authorized repairs in priority order. Preserve28941and
its live weights. The first repair aligns source-history training conditions
with the existing terminal-velocity protocol, before changing data coverage,
architecture, loss coefficients, or playback cadence.

Confirmed mismatch: the old pilot reused one approach proxy for each source.
Up-source command speed0.754m/s contrasted with actual window-terminal speeds
2.197/1.866/1.630m/s; down command1.123m/s contrasted with2.283/3.889/2.137m/s.
The protocol defines those inputs as terminal VX/VZ, not a separate soft intent.
This mismatch is established; its causal share of the entry failure is not.

`data/unified_training_goals.py` now derives raw world-XZ velocity from real
source interval31->32and heading from frame31. It uses no future height/body
pose, does not mutate the source, and labels these hindsight goals rather than
recorded WASD. `UnifiedTerrainController.forward` records the actual controls;
`native_losses` rejects mismatched source goals by default. Only the explicit
legacy negative-control arm bypasses that check. Existing live forward math
is unchanged. Pilot/check runners now use the matched per-window goals.
Source-history reconstruction review also uses matched goals; fixed-command
closed-loop reviews retain their old inputs and are labelled separately.

51distinct focused local tests passed (50initial tests plus the new freeze
scope test). A100matched pair launched from the same live200-update candidate:

- `unified_goal_legacy_20261014_v1` on physicalGPU0, old proxy negative control.
- `unified_goal_terminal_20261014_v1` on physicalGPU2, corrected terminal labels.
- Immutable code:`code_unified_goal_alignment_20261014_v1`; entry
  `scripts.pilot_unified_goal_alignment`. Both start from candidate hash
  `ade10ff143d855073b0760ca3f6a12cc8b1f59d7627eb883f9e981579edc5979`.
- Each arm200updates/batch4/800exposures, same seed/windows/source mix/native
  losses/Adam5e-5and24frame playback. Only root_stage1/root_stage2 environment
  modules train. Native Root/Pose/codec and shared encoder/Pose/decoder/height
  adapters remain frozen and are hash checked. No complete-budget restart.
- Each arm reviews8source cases, the live16cmstairs at0.8/1.6m/s, and17flat
  start/run/turn/release cases before and after training. Source-history
  teacher metrics are not directly comparable to older fixed-proxy teacher
  metrics. Live reviews permit slower traversal but require actual progress.
- Both runs are complete. Launch PIDs3384896/3386793are historical identity
  hints only. Do not relaunch or overwrite the code snapshot. No result is
  adopted; the live endpoint must not load a candidate merely because loss fell.

## Stair Entry Anticipation

The latest user reports that28941responds too late on approach, while the
stair gait looks correct once Root catches up. Prior average errors across
different source windows do not refute that phase-specific observation.
Prioritize anticipation at stair entry without replacing the approved gait.

Read-only A100 reproduction: `stair_anticipation_20261014_v3`, using the exact
live bundle/candidate/world hashes, default zone03and1.6m/s forward command.
The240logical playback frames use unchanged24frame reuse. Every8frame
boundary also gets a fresh shadow forecast, which is discarded and never
enters played history. This excludes browser/network waits from the motion
timing comparison. Native generation and matching shadow predictions agree
within3e-5absolute/1e-5relative tolerance. Weights are unchanged, no optimizer
is used, and28941is untouched.

- First riser is at worldZ=-108.000001m, height0.16m.
- Frame48: Root is2.538m before the riser and actual SDF already differs from
  the flat scene. With the SAME history and command, replacing the scene by a
  flat counterfactual changes predicted Root Y by at most1.49e-5m. This is a
  response diagnostic, not proof that the encoder received no information.
- Frame80: Root is0.737m before the riser; a fresh forecast still peaks at
  Y=0.9965m, near the initial0.9946m pelvis height. Cached forecast peaks at
  0.9886m. At frame88,0.268m before the riser, fresh maximum reaches1.0399m
  versus cached0.9886m. Cache reuse adds delay but does not explain all of it.
- Frame96: Root has passed the first riser by0.190m. The observed-terrain
  coordinate origin is now0.16m; predicted maximum worldY=1.1652m decomposes
  into this origin plus1.0052m relative Root Y. This is an algebraic trace of
  the existing coordinate gauge, not evidence of a runtime Root replacement
  or sufficient causal proof that changing the origin would fix anticipation.
- Played Root first exceeds its initial height by10cm at frame107,
  Z=-107.1389m. Root at frame191matches the earlier local trace within0.12mm.
  The10cm marker is diagnostic, not a prescribed gait height or contact gate.
- Surface checks use196sole vertices and566total sampled vertices at each
  played frame, without substeps. The first2cm sole penetration includes flat
  floor as well as stairs; do not mislabel it as the first stair collision.
  This is not a full-skin collision certificate, and later traversal remains
  unapproved despite the user's positive assessment of its motion style.

Local report: `out/parc_rugged32_20261013_v2/stair_anticipation_20261014_v3-report.json`.
Entry: `scripts/audit_unified_stair_anticipation.py`; two boundary tests pass.
V1stopped before inference because the remote world lacked the spawn receipt;
V2stopped before inference because the immutable training snapshot predates
the newer `scene_field` helper. V3uses the matching uploaded small receipt
and existing geometry factory, without modifying the snapshot. All logs remain.
Full motion stays remote; only the small numerical report returned over SSH.

## Native Root Validation

The latest user requested numerical validation of whether the native Root can
be retained. A read-only A100 diagnostic completed in
`parc_rugged32_20261013_v2/native_root_validation_20261014_v1` using the same
baseline bundle and current live candidate. No optimizer, checkpoint update,
deployment change, or runtime correction was introduced. The diagnostic
process has exited. Code entry: `scripts/audit_native_root_validation.py`;
remote dependencies: immutable `code_unified_terrain_pilot_20261014_v1`.
Local report: `out/parc_rugged32_20261013_v2/native_root_validation_20261014_v1-report.json`.

- Six real up/down routes, 510 source frames, four coordinate headings each
  (24 round trips). Source height ranges are 0.70 to 1.35 m. Native normalized
  global/local conversion preserves height exactly; maximum Root position
  error is 9.536743e-7 m and maximum FK joint error is 1.911070e-6 m.
- Root remains XYZ plus cos/sin heading; body features retain full 6D rotations
  for all 30 joints, including the pelvis. This verifies representability of
  these recorded stairs, not learned planning capacity on arbitrary terrain.
- Nineteen source-history windows, 24 evaluated future frames per window.
  Current mean window Root-height RMS is 7.525 cm; world MPJPE is 16.879 cm.
  All six per-route MPJPE values exactly match the archived pilot report.
- Offline substitution of source Root only, with unchanged history, command,
  scene, horizon, Pose weights and decoder weights, reduces world MPJPE to
  7.559 cm and root-relative MPJPE from 7.641 to 4.848 cm. Pose tokens may
  change in response to the Root intervention; this is a chain-level effect,
  not an isolated decoder attribution. The substitute uses future truth and
  is strictly diagnostic, never a usable actor.
- Despite exact oracle Root input, decoded-height RMS remains 5.292 cm
  (mean of window RMS values). Root planning is therefore not the only source
  of error. Same-time errors are not penetration metrics or traversal gates.
- 76 normal-input checks found no future height/pose constraints. Normal
  output replay after every oracle intervention is bitwise identical. Native
  and adaptation tensor hashes remain unchanged. Two local hook-isolation
  tests pass, including exception cleanup and preserving original storage.

Decision: retain the native Root representation; there is no representational
requirement for a new 6-DoF Root just to climb these stairs. This audit does NOT
establish that the present weights, source-history-only training strategy, or
deterministic architecture will generalize to all desired control situations.
Root/Pose/decoder learning and closed-loop behavior still require validation.

## Current Live Fix And User Override

Latest user explicitly requested closing28940and retaining28941. The verified
old recovered-flat server PID50572was stopped; its files/weights are retained.
Do not restart28940without a new request.28941now runs verified CUDA execution,
same200-update candidate, no new training or motion correction. Active server
PID78676is only a hint; check cmdline before operating. Command:
`python -m scripts.soma_unified_terrain_server --port 28941 --execution verified-cuda`.

Current evidence:`out/unified_terrain_20261014/live_v1/live-receipt-v2.json`.
Logs:`server-cuda-v3.stdout.log`/`server-cuda-v3.stderr.log`.

### Execution And Stutter

The eager endpoint measured256..381msneural time and310..454msplan response
time, with26..118mscached responses. Fresh generation intentionally waits for
actual playback acknowledgement; these costs caused repeated buffer stalls.
Added optional controller-aware static CUDA execution to the existing verified
runner, preserving commands, token selection, prediction horizon and geometry.
Five eager/captured probes have identical active tokens; max FK error is
1.52587890625e-5m. Actual SDF points/distances/validity, observed history and
official skin geometry are checked before enabling capture. Unsupported runtime
controls fall back to eager rather than replaying an incompatible graph.

Hot responses improved to roughly30..45ms, but capture alone still had an
intermittent browser run at18.0fps/max371.8ms. It was NOT accepted as fully fixed.
Enabled the existing serialized, active-viewer-only keep-ready replay. All
warmup results are discarded; no backend history or played motion advances.
Warmup stops when plan/prefetch/played activity ceases. Added shader precompile
on viewer initialization and read-only per-request/ack timing diagnostics.

Browser tests with active warmup:10second and20second W tests had trailing120
interval rates29.716/29.295fps,max45/52ms. Latest5.5second test played162frames,
trailing120interval rate29.785fps,P9538.4ms,max41.6ms,3short underrun ticks.
This materially reduces large periodic stalls; zero-underrun sustained30fps is
NOT certified. Report windowed rates honestly, not as whole-session averages.

### Response Investigation

Fixed W/initial-scene24vs8frame reuse comparison is in
`live_v1/latency-response-v2.json`. SDF contains nonplanar geometry at frame48,
Root Z=-110.538m, before the first riser nearZ=-108m, but Root/decoded height
response remains late. The first10cmroot rise occurs atZ=-107.139m for24reuse
andZ=-107.612m for8reuse. Shorter reuse also slows traversal and worsened browser
underruns(9in8seconds versus1in the preceding24frame8second test). It was not
adopted. Defaults remain8tokens/4passes/24frame reuse/0refinements.
The comparison is not a gait/collision acceptance result. It establishes that
late response is not explained solely by missing geometry or render delay.
Do not patch Root motion or change trained SDF inputs to conceal the delay.

### Correct SDF Visualization

The old viewer displayed legacy near/intent height patches, NOT the new SDF
observation. It now displays actual generation-time1536input points in body,
requested-direction and actual-velocity volumes, with signed distance colors
and validity. Cached plans retain the original world snapshot and frame age.
Legacy height patches remain optional and default hidden for this model.

All30displayed joint centers are additionally queried against the same scene
and rendered as `Skeleton SDF check`. This is explicitly diagnostic-only, NOT
an added model condition and NOT a skin-surface collision certificate. The
current trained model uses volume sampling, not full-skeleton SDF conditioning;
changing that input design requires matched training, not a live substitution.

Independent rendered-mesh vertical intersections at8points on16cmstairs match
heightfield levels exactly; SDF queries5cmabove those treads have max error
4.842877388e-8m. Scope is those authored heightfield treads, not arbitrary3Dmesh
or overhang support. Screenshot:`live_v1/sdf-and-skeleton-v2.png`.
49focused tests passed before final presentation-only refinements; JS syntax
checked. No new training or changes to checkpoint hashes. Model-quality issues
remain despite the runtime improvement.

## Explicit Experimental WASD Request

After reading the failed-quality results, the user explicitly requested WASD
connection anyway. An EXPERIMENTAL preview now runs at
`http://127.0.0.1:28941/` with the200-update candidate, not a quality-approved
replacement. The approved old28940service is unchanged and HTTP-checked.

Entry:`python -m scripts.soma_unified_terrain_server --port 28941 --execution eager`.
PID153296is an identity hint only; verify cmdline before operating. Candidate was uploaded/downloaded only through private
Drive with verified user-owned OAuth, and SHA256matches
`ade10ff143d855073b0760ca3f6a12cc8b1f59d7627eb883f9e981579edc5979`.
Local files:`out/unified_terrain_20261014/live_v1/`.

The existing256x256m authored terrain-world scene, area menu, WASD, reset,
inference sliders and official SOMA skin are reused without changing shared
viewer assets. Defaults remain8tokens/4passes/24frame reuse/0refinements.
Official rig transport versus full skin vertex error0on the initial history.
20focused server/controller tests pass, including checkpoint/profile rejection.

Browser testing exercised actual W/A/S/D keys and1146played frames. Observed
22.2..24.9fps,4underruns; stable30fps NOT verified. Browser console errors0.
Current local GPU was already busy; no other app/process was stopped. Desktop
capture`wasd-tested.png`shows generated motion at the staircase, and`ready.png`
shows the reset entry. Canvas-region sample has976distinct colors. During the
optional narrow-view check the tab became unavailable; the viewport override
was reset and mobile behavior is NOT marked verified. Both HTTP endpoints still
respond200. `live-receipt.json`contains exact process/hash/test evidence.

No IK, smoothing, output correction or quality-gate changes were added. This
user-requested experimental preview does not revoke the failed-quality verdict
or authorize a full training expansion. The latest user can open28941directly;
the page starts paused and its play button enables keyboard motion.

## Authorization And Scope

The user approved a single environment-conditioned generation chain on
2026-10-14. Flat ground is an ordinary scene, not a runtime motion branch.
Preserve the approved old WASD at28940 and its bundle unchanged. No IK, contact
locking, collision projection, output smoothing, root replacement, synthetic
recovery labels, future source pose/height conditions or new large budget.

This supersedes the prior native-preserving flat gate / actual-minus-plane
architecture. Historical reports are not evidence for this new implementation.
The native velocity protocol and8token/24frame playback profile remain unchanged;
this is not a claim to reproduce every official Smart Locomotion component.

## Implementation

- `data/unified_scene_field.py`: static analytic plane, exact single-box geometry
  test field, bounded heightfield factory and causal three-volume sampling.
- `data/unified_terrain.py`: one original Root/Pose/decoder call per replan,
  always-on scene memory, no proposal generation, plane subtraction, flat gate or
  delta-V goal editing. Decoder query time is separate from its identity path.
-1536points(3x8x8x8),24spatial memory tokens. This is an initial test budget,
  not an established optimum. Body/request/current-velocity corridors cover
  vertical space. Reversed directions are not averaged. Unknown samples stay
  masked; wholly unknown observations fail explicitly.
- The live-compatible adapter currently accepts authored planes and existing
  heightfield prisms. Single-box tests check walls/ceilings; integration of
  arbitrary triangle-mesh SDF world assets is not yet implemented.
- Original heightfield observations/vertical frame still enter the frozen
  checkpoint for compatibility. Removing plane contrast does not remove this
  old interface. Queries anchor on actual history3; predicted heights are used
  only as relative decoder conditions, never mistaken for world query points.
- No fused K/V caching or compiled new runtime yet. No local latency claim.
- `native_losses` rejects off-source generated histories rather than inventing
  recovery targets. The current fitting stage is source-history capacity only.

## Verified Gates

145local tests passed, including15new tests and native locomotion, online
condition and buffer regressions.

A100full gate COMPLETE at
`parc_rugged32_20261013_v2/unified_gate_20261014_v1`.
Immutable code:`code_unified_terrain_20261014_v1`.
Local report:`out/parc_rugged32_20261013_v2/unified_gate_20261014_v1-report.json`.

- Bundle SHA256:
  `0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d`.
-17flat closed-loop cases/7968frames, zero-initialized error exactly0.
-337native calls/337scene encodings including gradient probes.
- Native hashes, codec identity and condition configuration unchanged.
- Two diagnostic Adam steps discarded. First encoder gradient0as expected;
  second encoder norm0.000884, height encoder2.866e-6.
- Training straight-through/runtime forward difference0in both tested stair
  windows. Factual-history gradient norm2.39706. No production deployment.
- Trained flat output equality is NOT claimed; quality must be tested separately.

## Bounded Capacity Pilot

Code:`code_unified_terrain_pilot_20261014_v1`, a separate immutable snapshot.
Run:`parc_rugged32_20261013_v2/unified_pilot_20261014_v1`.
Receipt:`unified-terrain-pilot-20261014-v1-process.json`.
Entry:`scripts.pilot_unified_terrain`; state is inside the run directory.
PID1668362is an identity hint only; inspect actual cmdline before acting.
GPU2was checked idle by the guarded launcher; other users' tasks untouched.

Maximum200actual updates, effective batch4,800primitive exposures. Each update
contains two PARC stairs and two BONES flat samples, actual50/50exposure.
Original native losses andAdam5e-5; no new loss weights, source warping,
reference matching, yaw augmentation or teacher recovery. This is not a final
closed-loop curriculum or a full training budget.

Two PARC train sources are the screened one-up/one-down81frame sources; four
other stairs remain review-only. Two BONES129frame records are unchanged from
the previous prepared context dataset and freshly audited:

| Source | Max skin penetration | Closest sole gap P95 |
| --- | --- | --- |
| bones:aa6c730420ffb24f3b70_30hz idle |0.519cm|0.025cm|
| bones:f5ab6e82246166f838b7_30hz walk |0.715cm|1.116cm|

Both fixed floor0,skinP990,no unknown/uncertified interior points. Nearest-sole
gap is plane eligibility, not contact-force certification. Controls are retained
PARC approach proxies and BONES mean-walk/zero-idle proxies, not recorded keys.

The pilot evaluates baseline/final source-history and actual closed-loop motion
on all8cases, retaining actual frames and failures. `candidate.pt` contains
adapter/optimizer state, not an approved controller. Reports explicitly mark
quality_approved:false/deployed:false. Trained flat regression, full route/gait
review, source/generated visualization and actual WASD are still required.

Pilot COMPLETE:200updates/800exposures, no restart or budget extension approved.
Local complete report:`out/parc_rugged32_20261013_v2/unified_pilot_20261014_v1-report.json`.
Candidate SHA256:
`ade10ff143d855073b0760ca3f6a12cc8b1f59d7627eb883f9e981579edc5979`.
First/last20mean native losses approximately0.92/0.67. Frozen hash passed.

| Case | Source-history MPJPE before/after | Closed-loop max skin before/after |
| --- | --- | --- |
| train up |23.11/7.15cm|23.64/14.36cm|
| train down |14.75/9.59cm|5.58/6.67cm|
| held b8b up |19.93/24.33cm|3.15/13.00cm|
| held afe up |36.23/39.80cm|58.45/68.64cm|
| held1a9down |8.58/12.17cm|2.63/3.58cm|
| held eeb down |10.73/13.44cm|5.56/13.14cm|
| BONES idle |3.27/3.46cm|0.49/0.39cm|
| BONES walk |6.87/6.20cm|1.98/2.50cm|

The training-up closed loop reaches only55%source-relative progress and38.18cm
lateral offset; training-down73.8%and45.91cm. All6stairs exceed the2cmpeak gate.
Thus source-history fitting improved two train cases but did not solve actual
traversal or held-source generalization. This is not deployable. Do not equate
lower native loss/teacher-history MPJPE with improved closed-loop motion.

Read-only trained flat regression COMPLETE:
`parc_rugged32_20261013_v2/unified_flat_review_20261014_v1`, script
`review_unified_flat_20261014_v1.py` at remote task root, same immutable pilot
code dependencies. No optimizer updates. Guard rejected GPU2because it was
occupied; guarded launch then used freshly idleGPU7. PID1708320identity hint
only. All17cases completed without exceptions. Local report:
`out/parc_rugged32_20261013_v2/unified_flat_review_20261014_v1-report.json`.
For the eight1.6m/s cases, mean last-second speed changes1.4977->1.8645m/s;
mean absolute command error0.1023->0.2645m/s. In yaw0idle16s/start-walk,
left/right body-relative foot fore-aftP90range changes0.780/0.797m to
0.541/0.583m. Fast-running cases retain substantial alternating foot range;
this does not repeat the earlier severe run-collapse diagnostic, but does not
establish naturalness or no-slip. These ranges are not literal step lengths or
contact-force measures. Trained flat behavior is therefore not accepted as
preserved. No further updates or deployment were launched.

All new gate, capacity-fit and trained-flat-review jobs are complete. The current
next problem is learned flat preservation and legitimate closed-loop credit,
not adding another scene switch or expanding the full training budget. Any
later reference/recovery objective needs its own feasibility and gradient test;
do not adopt the rejected damped reference or perfect-flat-teacher assumptions.

Inspect state/complete.json before resuming; do not repeat completed work.
Large weights/animations stay remote until private Drive/NAS/internal
transfer. Only small source code and JSON travel over SSH. Local RTX3080 was
8668/10240MiB occupied and57%busy, so no competing model process or local30fps
test was started, and no other application was stopped.
