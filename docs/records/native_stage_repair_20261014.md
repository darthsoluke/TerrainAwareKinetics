# Native Stage Repair

## Current Work

New read-only gap audit is COMPLETE; see
[native_terrain_gap_audit_20261015.md](native_terrain_gap_audit_20261015.md).
Exact88-window commands: up1.872..3.726m/s, down2.227..2.862m/s. The selected
walk_backward_start crop is nearly static, so category names overstate locomotion
coverage.22matched condition cases replay archived metrics exactly. Stairs/plane
interventions show real Root geometry dependence; Pose-only legacy ablation changes
109/1024tokens, volume-Pose ablation1/1024. This does not measure the volume Root/
decoder paths. Eight sources outside the latest bank pass0/8source-history gates;
20constant-command source/idle routes pass0/20full route gates. Earlier base-model
exposure is not exhaustively audited. All raw failures retained remotely; local
JSON and plots verified. No training or live-service change; model tensor hashes
unchanged. Correct source selection/coverage before expanding the same recipe.

Latest user instruction visually accepts the orange result and authorizes a
WASD trial with correct inference. An independent28942endpoint is available;
see[native_bank_wasd_20261015.md](native_bank_wasd_20261015.md). This supersedes
the earlier no-trial recommendation, but does not change the recorded quality
failures or replace28941. Full model/state hashes and eager/CUDA/packet parity
are verified. Final real W test:30.030fps, zero underruns/commit waits; actual
A/S/D directions and key release reached the model. The user subsequently
requested the full map:28942now uses the verified256x256m authored world with
37spawn entries and the same complete native model. Full-world CUDA preflight
passed all five commands; the browser rendered the arena and accepted live
user inputs at approximately30fps. No automated reset was made once the user
began controlling it. See the linked live-trial document for exact evidence
and current process identity. General motion-quality limitations below remain.
No new training was started.

All native-stage training and diagnostics below are COMPLETE. No active training
or review remains as of this handoff. Latest native Pose model is
`unified_native_pose_fullbank_20261015_v2/weights.pt`, SHA
`03580db6f19c407b605fb43a6dd23514f8d5f17c5ad7b869c7ec95059ca68ae1`.
It completed50actual updates/4400exposures and passed88/88source-history actor
windows, up from25/88with the same codec/Root before Pose training. Normal actor
predicts its OWN Root and tokens: max published skin1.56557cm, leg-relative
error1.06609cm, final Root XYZ RMS0.86150cm. Frozen module hashes pass. This is
source-window capability, NOT a solved WASD controller.

Continuous feedback FAILS. `unified_native_bank_feedback_20261015_v1`retains all
ten real rollouts. Up from learned entry finishes with1.28969cmpeak; full-source
start14.97673cm; idle starts30.80976/45.53513/36.42118cm at source-window anchor
command/0.8/1.6m/s. Down learned-entry peak4.04585cm. Reaching the goal does not
make these routes pass. Current28941and28936are untouched;28940stays closed.

Two read-only controls narrow the remaining failure:

- `unified_native_source_commands_20261015_v1`: same model/source-start history/
  terrain/publication, but each24frameplan gets that actual source window's
  terminal XZ velocity AND heading. Both real76frameup/downrollouts reach the
  post-stair goal; peaks2.12758/1.09118cm, P99both0, lateral7.70/5.09cm. The up
  peak still exceeds the unchanged2cmgate. These are OFFLINE source-derived
  horizontal command labels, not logged human keys or autonomous WASD. No
  future height/pose/Root trajectory/tokens enter the actor. Both speed/direction
  and heading timing change together, so do not isolate speed alone as the cause.
- `unified_native_refresh8_20261015_v1`: same fixed commands/model, replan every8
  rather than24frames. It does NOT solve the problem; source-start up/down peaks
  12.775/12.556cm, some idle lateral drift exceeds70cm. Do not adopt the refresh
  change or disguise bad routes using lowerP99.

`unified_native_bank_flat_20261015_v1`completed17matched schedules for current
28941and the native candidate, with all18056vertices/8substeps. Flat preservation
FAILS: yawpi/walk foot foreaftP90~0.755/0.791m ->0.222/0.222m, terminal speed
2.05->0.95m/s; yawpi/2walk peak2.05->11.00cm. Some other geometry improves, but
that does not cancel the gait/control regression. Native full-network fitting
on6BONESclips is not sufficient rehearsal of general flat locomotion.

Interpretation: corrected native codec/Root/Pose training proves the model CAN
produce these stair motions. It does not yet learn robust constant-command,
generated-history control or preserve broad flat behavior. Source-command replay
success is evidence of condition/generalization dependence, not proof that the
velocity protocol itself is mathematically invalid or the unique root cause.
Do not return to blind adapter/loss-weight trials, deploy this checkpoint, or
run another full campaign. Next work must audit command provenance and coverage,
build adequate real BONES/multiple-heading rehearsal, then test a bounded matched
control curriculum. Preserve actual source frames, prior failures and all gates.
Training label terminal31->32 velocity is a hindsight goal; it is not a recorded
held-key intent. Do not silently relabel it or feed offline schedules to live WASD.

`native_route_study_export_20261015_v1`contains54recordings/14clips, including
original28941, native continuous feedback, refresh8and offline source commands,
plus source-history Pose before/after anchors. All115archive files privately
downloaded and verified; SHA
`89d342c674fdb3b80a5889206ecd524904e89f9caafe507a82fcab8e1b605edd`.
Local28936page uses`/web/terrain_route_review/?data=/out/parc_rugged32_20261013_v2/
native_route_study_export_20261015_v1/visuals/&clip=up-source_start&left=before&right=native`.
Actually inspected frames42/50with zoom, and same-model constant/source-command
frame50. Three screenshots are in the local export directory. Browser tab8was
opened in the background and marked deliverable. Source replay remains distinct
from learned output; independent actual lengths are not padded.

## Completed Native Bank Stages

The full-bank codec fit and separate actual-publication revalidation are now
COMPLETE. All88windows pass the unchanged gates: maximum published skin
penetration1.59424cm, leg-relative error0.63929cm, Root XYZ RMS0.38555cm,
with all geometry queries known/certified. Checkpoint
`unified_native_codec_fullbank_20261014_v1/weights.pt`, SHA
`dd365f2f3ecfd1b9e2a422a5da8b2c767d1ef20db203427587827dbcb4608f5a`.
Receipt:`unified_native_codec_publication_gate_20261014_v1/complete.json`.

The old raw84/88report remains unchanged. Its four remaining peaks were all in
reconstructed frame0, which runtime does not publish. The separate receipt
checks observed history4 + unchanged generated24 +8swept substeps, including the
actual history/future boundary. It rechecks ALL88windows, not just the four
failures. No threshold, runtime, future frame or integrated future Root was
changed. Two publication tests pass, including retained gradients from past
decoder velocities into future positions. This is a source reconstruction gate,
not autonomous terrain response or WASD approval.

Full-bank native Root fitting from this verified codec completed300updates/
26400exposures in`unified_native_root_fullbank_20261014_v3`, immutable
`code_unified_native_route_20261014_v5`. Gate57/88, protected hashes unchanged,
checkpoint SHA`a6df6870b48058360a8a3c6c72196eef1dd87658d92b5899a1711b0dcecb269d`.
Loss0.0494->3.745e-5; worst Root XYZ RMS1.223cm/max Y3.839cm. Most remaining
failures are BONES vertical trajectories, plus four down windows. All eight up
Root trajectory windows pass; codec composition has NOT run because not all
trajectory windows passed at that time. Root is not piecewise-constant
at token stride: the official convolution outputs per-frame values.

COMPLETED:`unified_native_root_continue_20261015_v1`, immutablev6. Same data/native
loss/frozen modules, retained Adam moments,1e-5cosine to5e-7, at most300additional
full-bank updates/26400exposures. This explicit schedule extension tests residual
convergence; it is not attributed to a changed architecture or geometry loss.
Final64/88Root trajectory gates, up8/8/down7/8, most remaining failures in BONES.
Worst Y3.52767cm. Source-balanced native loss2.2929e-5. No further Adam continuation
is authorized by this note; inspect the next bounded optimizer check instead.

`unified_native_root_residual_20261015_v1`completed a separate100update/100exposure
single-worst-window capacity diagnosis with unchanged native loss. The exact
same causal window's max Y3.52767->0.48559cm, RMS1.08351->0.13474cm. Other bank
gates64->1/88, so this is NOT a candidate and no weights were saved. It demonstrates
the native Root can express this failed waveform, not a general repair.

COMPLETED:`unified_native_root_lbfgs_20261015_v1`, script
`polish_native_root_fullbank_v2.py`, imports immutablev6. Starts from the complete
88-window Adam candidate, NOT the isolated overfit. Changes only optimizer to
deterministic L-BFGS(history10, strong Wolfe); same native source-balanced loss,
source bank, runtime and frozen modules. At most100actual changed steps and
400loss/gradient evaluations(35200window exposures). Counts line-search loss
evaluations separately from optimizer steps. At budget exhaustion a trial step
is rolled back to the preceding accepted model. This is a bounded convergence
check motivated by the successful isolated native fit, not a geometry-loss or
architecture change. Final100changed updates/209loss-gradient evaluations/
18392exposures, raw Root67/88, worst Y3.39845cm. Checkpoint SHA
`a551b784d8109a9ae84f78716362807b9ac51d0db9490dbbf55c5aa01b8018cb`.

The subsequent READ-ONLY`unified_native_root_chain_audit_20261015_v1`corrects
the overly restrictive stage ordering: native MB explicitly refines Root in the
codec, so raw planner RMS<=1cm/maxY<=2cm is a diagnostic, not a necessary
condition for a correct final actor. Predicted Root + OFFLINE source tokens
passes ALL88actual codec composition gates, unchanged leg<=1.5cm/Root RMS<=2cm/
skin<=2cm with18056vertices/8substeps/all queries known. Max skin1.58926cm,
leg0.64016cm, final Root RMS0.68417cm. Original raw Root67/88failure remains
unchanged; this is a new distinct downstream receipt, not a threshold rewrite.
History global Root/local Root/pose all match targets exactly; all four observed
poses are provided. Nearest normalized input RMS is0.001625, not an exact
duplicate-condition contradiction. This does not prove broad conditional uniqueness.

`unified_native_pose_fullbank_20261015_v1`failed before updates at a logit parity
tolerance(6.998e-5versus5e-5). Its preservedv7snapshot remains unchanged.
COMPLETED:`unified_native_pose_fullbank_20261015_v2`, immutable
`code_unified_native_route_20261015_v8`. The explicit`--root-chain-gate`requires
the matching checkpoint hash, all88ordered source windows and unchanged final
motion/geometry limits. Root67/88is explicitly retained in the Pose contract.
At most300updates/26400exposures, source-balanced50/50PARC/BONES. Original native
cosine masking/noise and per-window CE, source-Root teacher as MB5.2. Native Pose
plus existing Pose adapter train; Root/codec/shared encoder frozen. Real A100
batched/per-window logits, CE and parameter gradients passed before updates:
logit max6.998e-5(atol1e-4/rtol1e-5), identical discrete token predictions,
probability max7.51e-6, CE difference4.77e-7, gradient max3.13e-7. Increasing the
numerical logit tolerance was coupled to exact categorical and probability gates;
no motion/geometry quality threshold changed. Stopped after50updates at88/88.
Final actor always uses its own predicted Root/tokens; full feedback and unseen
commands/terrain remain separate mandatory checks. Do not deploy from this bank.

New
`fit_native_root_fullbank.py` first checks batched/per-window forward, weighted
native loss and native parameter gradients, and rejects future height/pose Root
conditions. At most300updates/26400window exposures, same source-balanced50/50
PARC/BONES objective, Adam5e-5cosine to5e-7. Root and its existing two adapters
train; Pose/codec/shared encoder remain frozen. The later explicit composition
receipt corrects raw-Root-only stage blocking as explained above. No live model is replaced.
Preflight max Root output2.38e-6, finite duration logits3.81e-6, objective1.86e-8,
parameter-gradient2.30e-6. The v1launcher lacked a helper import; v2reporting
subtracted matching masked -inf logits and could not serialize NaN. Both failed
before any update. Preserved logs; v3compares mask identity then finite values.
Six targeted publication/schedule/batching/weighted-native-CE tests pass. The
Pose full-bank runtime numerical checks and final actor gate are complete.

## Completed Codec Optimization

The first full-route codec fit completed1000updates/4000exposures on8clips/
88overlappingwindows. Only42/88passed; worst skin6.27cm. Root/Pose bank stages
were NOT started. Native-loss continuation and a Root/FK-fidelity addition then
ran from that same checkpoint with the same schedule/budget ceilings. Both
completed1000more updates. Native control37/88passed, mean Root RMS1.06cm,
leg0.85cm, worst skin5.51cm. Fidelity17/88passed, Root1.62cm, leg1.60cm, worst
skin6.87cm. Fidelity is NOT adopted. No coefficients were swept; the tested
extension used native Root coefficients2(global)/1(local) and a declared FK
group weight1normalized by existing RIC scales. It has zero loss on real-source
identity, finite decoder gradients and no actor/output mutation, but that alone
did not make it a better objective. Local loss/schedule tests passed.

Native control checkpoint SHA:
`5d07cca89ce94b8dc729e8a80cc831f03deb3a234642b8fc86d17946caf72fde`.
Rejected fidelity checkpoint SHA:
`665d2b07c087395683515d481ad8cdae2302d9d375b150aa27d49af29911118d`.
Both use immutable`code_unified_native_route_20261014_v2`; both processes ended.

COMPLETED: `unified_native_codec_fullbank_20261014_v1`, PID2035964on freshly checked
GPU7(identity hint only). Script task-root`fit_native_codec_fullbank_v1.py`,
imports immutablev2. It starts from the native control above, keeps native loss/
model/data unchanged, and computes the exact source-balanced entire88-window
gradient. Native loss group weights remain50%PARC/50%BONES, six BONES clips
balanced equally. Batched versus per-record local forward max3.81e-6; native
objective difference2.79e-9. Fresh Adam5e-5cosines to5e-7, up to300FULL-BANK
updates/26400exposures; saves optimizer/scheduler/RNG. This is an explicit
additional optimization budget, not another4000-exposure arm or full campaign.
At50updates85/88raw training windows passed; final raw gate84/88. The separate
publication receipt above is authoritative for actual playback. Batching, learning-rate schedule and extra exposures all
changed; improvement is not attributed solely to batch size. Read complete,
then advance Root/Pose only if the whole training bank gate passes.

The current v2route trainer pre-creates`checks/`and adds a Root-stage composition
gate: after Root trajectory thresholds pass, predicted Root + OFFLINE source
tokens must also pass through the actual decoder. Normal actor never receives
source tokens. This prevents launching Pose over another inadequate upstream
condition. The v2feedback loader explicitly supports full route-state schema.

`native_stage_study_export_20261014_v1`has40recordings/14clips, including raw
source-codec before/after (OFFLINE), source-history actor before/after native Pose,
and ALL10feedback outcomes including failures. Original frame counts preserved;
training windows are not called generic WASD. Published actor histories display
the four observed frames; future output is unchanged. Archive private-Drive SHA:
`2e3f1e57f1b0a72e2a26bc48cfa6c24866470f3be22d3035cc357ab74ea8fd6b`.

## Strategy And Earlier Evidence

The user rejected the repeated adapter-only small trials and asked for the
strategy itself to be corrected. Preserve live28941and all prior failed runs.
No IK, output smoothing, runtime Root replacement or future-truth actor input.
The new order is codec reconstruction -> native Root learning -> source-Root
Pose learning -> causal actor -> generated-history routes. Do not skip gates.

## Passed Four-Window Capability Gates

All three stages use the same four genuine33-frame windows: first selected
up/down at start36, BONES idle/walk at start0. They are deliberate memorization
tests, not generalization or a WASD delivery. Each stage300updates/batch4/
1200primitive exposures. Existing native losses/coefficients, Adam5e-5, clip0.5.
Dropout stays eval for this deterministic capacity test. Codec encoder/codebook
and all non-selected modules are hash protected. Runtime remains8tokens,
4Pose passes,4real history frames,24published frames in8-frame chunks,30Hz.

1. `unified_native_codec_capacity_20261014_v1`: train native codec.decoder,
   including all32real reconstruction frames, with source Root as AUTOENCODER
   condition only. Up leg error5.13->0.32cm, Root XYZ RMS4.56->0.28cm, full-skin
   peak14.83->1.84cm. Down/idle/walk all pass leg<=1.5cm, Root RMS<=2cm,
   all18056vertices/8substeps peak<=2cm and known/certified-query gates.
   Source encoder versus indexed decoder and32-frame versus padded runtime
   forward agree within9.1e-6world units; no padding bug was found here.
   Decoder checkpoint SHA`0b99aaa4f91b1cf93ffea713b435dce259d1b283a825bf41a7bbf5f6e3a7dcaa`.
2. `unified_native_root_capacity_20261014_v1`: train native Root plus its two
   existing SDF adapters,35,714,997parameters. ONLY causal history/command/scene
   input; future height/pose masks checked absent. Up Root RMS15.52->0.04cm,
   down5.36->0.06cm; flat0.02cm. All pass Root RMS<=1cm/max Y<=2cm.
   SHA`c616f3404bb07aeb2062b5bcaccd840ac1890b704d2819822e29546896c65a65`.
3. `unified_native_pose_capacity_20261014_v1`: native Pose + its existing SDF
   adapter,141,411,858parameters; SOURCE Root teacher as MB Section5.2. Native
   cosine masking, known starting4poses only, no future pose keyframes. Root,
   codec and shared SDF encoder frozen.8-token training versus padded runtime
   logits checked. After fitting, normal actor uses OWN predicted Root/tokens:
   all64tokens correct in all4windows; up/down leg errors0.32/0.42cm, peaks
   1.84/0.14cm. Idle/walk pass too. No oracle Root/tokens enter that actor test.
   SHA`52d665cccc1a8217fa9fd9a43f19fb4853e6e9e4403622086a9ed9e89d964ede`.

Schema`motionbricks.unified.native_stage_capacity.v1`is diagnostic-only and
incompatible with the existing live adaptation-only loader. Large checkpoints
remain on A100. Correct source-window output is not a complete controller.

## Continuous Feedback Result

`unified_native_feedback_20261014_v1`completed10normal feedback rollouts on
the two training courses. All published frames are actual model output; no
padding, Root repair or output constraints. Starts include source route start,
the learned entry and a measured flat BONES idle placed ONCE at the route spawn.
Commands are constant training-window velocity,0.8or1.6m/s; the former is a
source-derived proxy, not logged WASD. A16secondbudget is not a speed limit.

Up from learned entry reaches the post-stair goal in44frames, peak1.84cm,
lateral8cm, longest both-feet-above10cm run3frames. Up from full source start
peaks2.13cm but floats38frames and drifts38cm. Flat-idle up starts peak21.74/
45.99/33.69cm. Down learned-entry rollout peaks12.01cm after the initially
passed primitive; idle down also fails. Thus capability gates pass, full control
does NOT. Do not deploy this four-window model.

Counterfactual same entry history on flattened terrain retains~36.7cmup rise
versus36.4cmon actual stairs, max Root Y difference5.25mm, origin difference0.
This overfit candidate still strongly reproduces the learned entry pattern.
The counterfactual is not a clean flat reference label, and does not establish
the unique cause of all earlier failures. New full-route and field-response
checks remain necessary after broader native fitting.

## Active Whole-Route Curriculum

New script`train_unified_native_routes.py`continues the passed native model.
It uses complete selected up/down clips plus ALL6prepared BONES clips, not new
synthetic transitions. Full33-frame source windows at stride8include the old
capacity anchors and each real final window. Exact counts are in contract.json.
Each batch is up/down/two rotating BONES, with PER-SOURCE window cursors to
avoid the previously fixed modulo alias. The focused coverage test passes.

Each stage has a1000-update/4000exposure CEILING and evaluates its complete
training bank every100updates. It stops early when ALL module gates pass.
Only then may the next stage start. Maximum codec+Root+Pose total12000exposures;
this is not a restart of the1600128-exposure campaign and not equal compute
to adapter-only fitting. No new geometric loss or coefficient changes.
Native Pose still trains on source Root; normal actor still uses predicted Root.

COMPLETED: `unified_native_route_codec_20261014_v1`, freshGPU7, PID1790864only an
identity hint. Check cmdline/progress/complete before actions. Immutable code
`code_unified_native_route_20261014_v1`. An SCP returned a session before launch;
it completed successfully and all3used script hashes were checked equal to local
files while preparation ran. Never overwrite this snapshot. The local trainer
now pre-creates`checks/`for Root-stage JSON output; use a NEWv2snapshot for later
Root/Pose stages. Codec v1creates this directory through its visual checks.

New stage schema`motionbricks.unified.native_route_curriculum.v1`contains the
complete controller state, stage provenance, source windows and actual budget.
The current feedback loader must explicitly support it before use. Current
28941still uses the original preserved candidate; no deployment has occurred.

## Files And Remote Paths

Local root:`${PROJECT_ROOT}`.
Remote base:`${PRIVATE_STORAGE}/motionbricks-20261010`.
Task:`model_terrain_v3_20261011`; runs under`parc_rugged32_20261013_v2`.
Bundle:`long_route_mixed_10k_ab_v2/terrain_reference/pipeline.pt`.
Initial:`unified_pilot_20261014_v1/candidate.pt`(ade10ff...preserved28941).
Cases:`unified_pose_data_20261014_v3/cases.pt`(4d8e671d...unchanged).
Capacity code scripts live separately at task root:
`probe_unified_native_codec_v1.py`, `probe_unified_native_stages_v1.py` and
`review_unified_native_feedback_v1.py`, importing immutable joint snapshot.
All small complete reports are downloaded locally as`*-report.json`.

Only owned private Drive/NAS/internal paths for models/data/animations. Existing
7897proxy works; never retry the rejected27891helper. No subagents; fresh idle
GPU check before every new launch; never act on another user's process.
