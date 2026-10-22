# Joint SDF Results And Decoder Diagnosis

## Latest Live State: 28940 Closed, 28941 Accelerated

User explicitly closed28940; recovered server was identity-checked and stopped.
Do not restart it.28941retains the same candidate with verified static CUDA
execution and active-viewer keep-ready replay; no history advance/output change
from warmup. Five probe token IDs identical,max FK difference1.526e-5m.
Current server PID78676identity hint only. Read`unified_terrain_20261014.md`and
`out/unified_terrain_20261014/live_v1/live-receipt-v2.json`first.
Current viewer shows actual1536SDF volume inputs plus separate30joint SDF
diagnostics. Joint SDF is not falsely claimed as trained model input. Corrected
cached-observation age metadata.8rendered stair-tread heights match field exactly.
Runtime stutters materially reduced: warm playback trailing windows29.3..29.8fps,
max41.6..52ms,still short underruns. Capture without warmup had a bad18fpsrun,
retained in evidence.8frame reuse was tested but not adopted; keep24default.
Geometry is observed before the step but model height response is still late;
no runtime root/pose patch or new training was used to hide it.

## Current Preview: User Requested Experimental WASD

The user explicitly requested connecting the failed-quality200-step unified
candidate for manual WASD. Experimental service is now28941on the authored
256mworld;28940approved baseline is unchanged. See`unified_terrain_20261014.md`
and`out/unified_terrain_20261014/live_v1/live-receipt.json`. Candidate hash verified
through private Drive/user-owned OAuth. Actual W/A/S/D tested;22.2..24.9fps,
4underruns,official skin rig parity0. Not stable30fps or quality approval.
Server entry`scripts.soma_unified_terrain_server`,PID153296identity hint only.
Do not mistake this explicit experimental-preview request for a deployment
quality pass or permission to restart/expand completed training.

## Current Work: Unified Single Chain Terrain Conditioning

Latest user authorizes the unified environment design, without flat motion
routing or actual-minus-plane features. See`unified_terrain_20261014.md` first.
New local145tests pass. Full A100zero-init17case/7968frame parity passed,
weights/protocol unchanged,337native calls for337scene encodings. Two diagnostic
updates discarded. Preserve approved28940and all prior pages.
Bounded source-history pilot:`parc_rugged32_20261013_v2/unified_pilot_20261014_v1`,
code`code_unified_terrain_pilot_20261014_v1`,max200updates/800exposures,50%PARC/
50%BONES. Unchanged BONES idle/walk passed fresh fixed-plane source audit.
No invented recovery targets,IK,output correction or new full budget. Inspect
run state/complete.json and actual cmdline before action; never duplicate launch.
Pilot COMPLETE200updates/800exposures. Train-source-history MPJPE improves
up23.11->7.15cm/down14.75->9.59cm, but all6closed-loop stairs fail2cmpeak gate;
train up reaches55%source progress with14.36cm skin peak. Do not expand training
or deploy. Independent trained-flat17case review COMPLETE onGPU7at
`unified_flat_review_20261014_v1`: no exceptions, but mean speed for1.6m/s
requests1.4977->1.8645m/s, some idle-to-walk foot ranges shrink. Not accepted
as flat-preserving. All three new runs complete; no new training active.
Baseline/final8case evaluations retained. Not a deployment candidate.
The older hard-gate architecture below is historical.

## Current Execution: Native Preservation Gate And Official Locomotion Authority

User authorizes implementation,flat preservation first,then tiny stair training,
complete before/after and actual WASD;noIK/runtime corrections. User reiterates
that the MBpaper is the method authority and official flat locomotion the primary
implementation reference. Keep28940approved flat experience and all prior pages.
Do not call our native velocity extension a full official Smart Locomotion clone.

Added independent data/native_preserving_terrain.py. It does NOT inherit
JointTerrainController and does not rebuild codec/change masks or protocol.
It wraps the original velocity pipeline and buffers using an explicit context;
only new terrain modules train. Geometry-relative residuals affect internal
Root/Pose/decoder features;additional decoder memory comes from model-predicted
Root height. The goal module is a learned delta-V research extension,not an
official primitive. No output skeleton/Root edits. It must not bypass official
goal/style/timing compatibility review or be advertised as established MBmethod.

COMPLETE code_native_preserve_v1/native_preserve_flat_gate_cpu_v1:
17continuous flat cases (8immediate,8after16srest,1W/A/D/S/release sequence),
original8token/24frame reuse/0refinement. Zero initialized modules,random nonzero
modules,and after2diagnostic optimizer updates each reproduce all old motion
features exactly0error;7968frames per comparison. Original codec identity,
Root/Poseconfiguration and all native tensor hashes unchanged. Two source-window
gradient diagnostics show nonzero goal/Root/Pose/decoder gradients and native
parameters none. ForwardSTerror6.44e-6/1.53e-5. Diagnostic updates discarded and
initial state restored;NOsmall-sample fitting weights/no new deployment.
Report local native_preserve_flat_gate_cpu_v1-report.json. SSH session later
closed with exit1,but remote complete/state and full report verified successful;
do not infer training failure/restart from SSHexit alone. Initial missing helper
import failed before model loading;terrain_root_supervision.py was staged.

Local v2 then repairs a training-only issue: once a terrain-generated history
enters a flat region,the zero-residual fast path must still propagate factual
history gradients to earlier terrain actions. Evaluation still returns the
original forecast. Dedicated real CPUcheck native_preserve_flat_gradient_v2.json:
forwarddiff6.497e-6,historygradientnorm2.41653,basehashunchanged,0updates.
New immutable code_native_preserve_v2 separately staged;v1not overwritten.
Implementation native-preserving-v2-flat-history-grad. Need A100v2parity before
any GPUfitting. Six residual/gating unit tests pass;no motion-quality claim.

COMPLETE native_preserve_cases_v1,caseSHA
27e3719bd082f31d60dd4e5b73c7fef0f9e7069acb40b0b5862212089ad6143f.
Uses real same-source records with prefix equality verified,aligned to24frame
reuse. Train only official_6d8935e4152e794f-up and official_dc03d6484b176db0-down:
81real frames each,3native predictions,publish throughframe75,33real flat-exit
frames after last riser. Four other source IDs review-only,24..44exit frames,
57..105real frames. All6source skin peaks.75..1.63cm,existing2cmgate passed,
no unknown/uncertain points,no padded/spliced poses. Constant weak approach
speed/facing retained,not recorded keyboard labels. Ground is exact heightfield
prisms,not proof against original triangle-mesh continuous collision.

Official-source re-review: project page links to NVlabs/GR00T-WholeBodyControl
preview,not a complete release of the UE5demo. Re-read upstream full_agent.py
and localHEAD b3079767abb6bb31a3d28a8bb9c4efcc856da6d4. Local diff only removes
unused qpos fetch and extracts a no-MuJoCo inference entry;spring/proxy placement
logic remains directly bound in NativeSmartLocomotion.109existing native
locomotion/primitive tests pass,including units,masks,duration and source input
handling. Official flow includes spring target proposals,style proxy poses,
masked duration/Root refinement,Pose and decoder;skip_ending_target_cond does
not erase Pose's style conditions. Current velocity checkpoint instead fixes
horizon and rejects future pose/RootYconditions. Do not silently disable its
validator or inject scene-reference futures to mimic official conditioning.
Independent style exemplars are intentional goal inputs,not the actual next
ground-truth motion;their provenance and checkpoint compatibility must be clear.
G1optional source_root_realignment/filter_qpos/padded initial frames are not
permission to alter factual histories or smooth/correct played output. Target
proposal construction is distinct from output correction.

No active fitting process. A100launch guard rejected GPU2(60GBoccupied) before
spawning;later fresh snapshot0/1/2/3/4/5/6/7all had other allocations,2..7busy.
No other's process touched. CPUgate/data prep completed. No new automation.
Next:official-method compatibility/deviation review,then A100v2gate when actually
free,then bounded one-up/one-down fitting with exact same runtime publication
policy and full route/flat return checks. Do not skip these prerequisites or
expand training merely because CPUunit/parity tests pass.

## Current User Approval: Preserve The Recovered Flat Capability

User explicitly approves the flat locomotion at28940 and asks for careful
reasoning about terrain repair. This approves the observed flat capability,
not stop accuracy or terrain traversal. Preserve the original0672abe1... bundle,
native velocity-control interface,codec,played-history observer,and original
8token/24frame buffer/0refinement profile. No training or actor changes in this
reasoning turn;both this baseline and approved stair visuals remain intact.

Code re-review confirms JointTerrainController.__init__ does more than adding
SDF: it rebuilds the height-conditioned codec,changes online_conditioning to
soft-intent,and installs control/terrain hooks. JointSDFController inherits it.
Its flat-subtracted residual therefore cancels back to THAT changed controller,
not automatically to the user's approved original native controller. Do not
reuse that constructor while claiming a zero-change extension of the baseline.
Correction to the earlier note below: current8 contract.original_conditioning
is actually native-mb-soft-intent-online-v3 after the constructor mutation;
old8 is native-mb-velocity-online-v2. The field name alone was misleading.

Proposed direction,NOT implemented or quality-proven: an isolated terrain
extension around the preserved native pipeline,with initial/output-equivalence
tests before fitting. Keep the native goal interface;learn feasible local
terrain-conditioned goals from factual history,user intent and observed geometry
instead of removing that interface. Preserve the base codec;any added terrain/
predicted-Root-height conditioning is an independently controlled residual,
not a replacement or post-decoding correction. Flat inputs should reduce to
the original path by construction and be tested over full closed-loop motion;
frozen weights alone are not proof. Native buffer/command timing stays fixed
during the first comparison. Transition back to flat needs its own rollout
test: equal outputs for equal flat histories do not prove recovery from a
terrain-adapted history.

First scope is one clean real ascent and descent with complete approach/exit,
continuous forward input and generated history. No source-history resets,
future source poses/heights in actor,IK,root edits,frame padding or stop-in-place
substitute. Source targets must remain feasible;do not revive failed nearest-
frame skipping or fixed-time large recovery targets. Additional control/codec
components need staged comparisons,not simultaneous unrestricted fine-tuning.
Only after flat preservation AND actual stair traversal/gait/support pass should
PARC diversity or training budget expand. Gate failure is evidence to diagnose,
not a reason to increase weights or automatically add more samples.

Re-read primary MBpaper2604.24833v1,sections3/6.1 andAppendixE. It separates
high-level goals from Root/Pose/decoder generation and explicitly notes absence
of continuous terrain information in its dataset. This motivates the separation,
but does NOT prove the proposed learned terrain planner works or make this
existing velocity-conditioned SOMA extension a literal Smart Locomotion replica.

## Current Work: Earlier Flat Controller Recovered

User clarifies the reference is the EARLIER FLAT RUNNING version,not the later
approved stair model. Stop extending the failed restart candidates. No optimizer
updates this turn. Found Oct11 original A100-B28930/28931 launch receipts and
preserved bundleSHA0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d.
It already includes BONES;do not blame the presence of BONES data alone.

Read-only review_flat_controller_recovery compares old native versus final
restart_phase_v1,8and12token horizons,each two headings x1.6/5mps x immediate/
16s-rest start.32cases COMPLETE,A1002/7freshly checked,all lengths retained.
Old/current within each horizon both replan8frames,use4argmax passes,same four
world-seed frames and authored zero plane. Model/control/observation paths still
differ;this is a whole-version comparison,NOT a single-module causal ablation.
Reports flat-recovery-old8/old12/current8/current12-report.json are local;
raw recordings remain remote. No animations/models transferred throughSSH.

Immediate yaw0,5mps request,8tokens:old forward31.85m/current13.26m over8s;
last-second actual speed4.51/1.53mps. Last4s body-relative foreaft foot range
(P95-P05)old L/R.870/.892m,current.343/.324m. Above5Hz normalized foot-motion
energy old.124/.090percent,current13.27/3.13percent. These are kinematic
diagnostics,not step length,force contact or an independent pass threshold.
Even old1.6request achieves1.61mps with.792/.820m foot range;current5request
at1.53mps has much smaller leg swing. Thus reduced speed alone does not explain
the observed small trembling motion. Other cases remain in the full reports.
Old native8also runs after16s idle in both headings;old12can still stall at
1.6mps after idle. The earlier controller is NOT a complete start/stop solution.

Verified code distinction: native source uses native-mb-velocity-online-v2,
explicit terminal VX/VZ and heading conditioning. JointTerrainController calls
sparse_soft_intent_conditions,clearing future masks and using learned command
embeddings;later command adapters additionally affect Pose/decoder. The current
diagnostic Root config tag is soft-intent-v3 after constructor mutation,while
the original is velocity-online-v2. Both weights and conditioning
changed;do not falsely isolate one as the full regression cause. Next terrain
work must preserve/test the old flat capability,not count Root travel as gait.

Independent local recovery:http://127.0.0.1:28940/.
scripts/soma_recovered_flat_server.py and launch_recovered_flat_trial.ps1 load
unchanged original bundle/native backend,NOT Joint SDF or any recent candidate.
Original native profile is default:8tokens,4passes,24frame generated-future
buffer,0route refinements. Existing packet/command invalidation retained;not
interpolation,IK,foot locking or root edits. Sliders remain available. Page starts
paused;firstW starts generation. It is labelled experimental/terrain unqualified.
All28936/28937/28938/28939services/pages preserved. FinalPID50572 is an identity
hint only;verify cmdline before any action. No new automation.

Source4frames unchanged. Original flat.npz42x42/.2m domain was too small for
sequential compiler probes;startup correctly failed unknown-ground check before
serving. Kept log20261013-221016. Created an EXPLICITLY AUTHORED100m zero plane,
not guessed missing real terrain:flat_recovery_20261013/flat100m.npz SHA
1da8e1148ba412ece65fa45037f400a980e2df0efad430156a5022ba8cf012d3;
receipt retains seed/source hash and source_history_modified=false. No model
training used this scene. Initial eager launch was slow~13fps;8frame-refresh
compiled live reached~27fps with boundary waits. Neither was declared30fps.
Final uses original24frame buffer. Ten compiled/native probes preserve active
tokens;FKmax5.232e-6m. Actual W run6s produced179frames,29.7fps,0underruns;
final process firstW4s produced122frames,29.7fps,0underruns. A/D/S then respond
with changed positions;29.5..29.6fps and3cumulative underruns after idle/turn
testing. This is short operation evidence,NOT sustained30fps/collision approval.
running-original-profile.jpg is actual officialSOMA,debug overlays off. Old
version still drifts at rest and has terrain/contact limits. Restoring it does
not solve the outstanding terrain-aware WASD controller task.

## Current Work: Corrected Gradients And Genuine Restart Pilot

FINAL STATUS: both arms COMPLETE,16actual updates/1556generated primitives/
64sourceAE/64auxiliary source actor calls EACH. Exact baseline/source/yaw/exposure
parity and all frozen hashes verified. No active training or enlarged budget.
Neither qualifies for deployment.62focused and existing online tests pass.
Critical yawPI,16s idle then8s requested1.6mps,prior/control/phase:
- idle drift1.26/.25/.27m;forward11.10/14.24/15.51m;side1.53/1.55/1.40m;
- whole8s close-feet27.5/19.6/7.5percent (see exact JSON for unrounded values);
- no stationary sole-material proxy79.6/83.3/93.8percent;left near-plane material
  speedP901.94/1.97/1.98mps. This is NOT force contact or airborne percentage.
- skin peaks9.11/4.34/8.55cm;stairs0/12in all. Greater forward distance and separated
  feet did not fix skating. New genuine restarts still progress only-.02..36cm
  in2.4s in phase arm;two held families~8/7cm. No end-to-end controller approval.
Both arms exhibit long-feedback gradient amplification (preclip maxima1014.65
and4871.96);original.5clip remains. Do not mistake finite clipped updates for
stable supervision or automatically add more training.

Read-only restart_history_gap_v1 replays all54published MOVE blocks of final
phase model with maxerror0. Replacing each input with actual source history
offline yields last5mean speeds.65..1.69mps, versus generated-history-.01..21mps
and source.78..1.77mps. This teacher-history diagnostic is NOT an online actor.
Important countercheck restart_clean_boundary_v1 initializes ONCE at real source
stopped history then runs all9MOVE blocks continuously. Forward distances
1.188/.749/.022/.618/.002/.054m versus original generated-idle
.161/.363/.094/-.021/.078/.067m. Thus genuine history helps several training
cases but does not solve all starts,especially either held source. Cannot blame
only idle drift or claim a working locomotion prior is established. No source
motion stitched,root translated or future target fed to actor in either audit.

restart_comparison_v1 exported3predeclared flat clips/9official SOMA recordings,
all6restart/16flat/12stairs metrics and material-point mechanics.30receipt-listed
files privately downloaded and verified;archiveSHA
5586402228d05923d7a6a05017e96a9ca0baa009af7e21ea0b0037c1a1bb3d98.
Viewed actualframe540/660 in background28936,frame660.jpg saved;left is PRIOR
ORANGE command model,not deployed28939;right is the new phase-prior trial.
http://127.0.0.1:28936/web/decoder_pilot/?data=/out/parc_rugged32_20261013_v2/restart_comparison_v1/visuals/&clip=idle16s-forward-yaw180&left=before&right=phase
No model weights downloaded or live server restarted. New WASD deliverable is
still outstanding. Next decision must address robust rest->gait initiation and
support,with clean-prefix versus generated-prefix evidence and temporal-gradient
stability,not another blind full budget or a runtime phase/root/foot correction.

User authorized the reviewed repairs and requests an actual WASD result. Old
approved28937/28938, experimental28939 and review28936 remain unchanged. No new
model is approved or deployed. This section supersedes the review-only status
below; the numerical findings there remain valid.

Implemented SoftCommandState(history_gradients=True), retaining the release-time
anchor graph during later STOP frames. Removed the unconditional zero-valued
current-root gradient shift in the two local training callers. Runtime default
still detaches; forward values unchanged. Implemented source_support_masks with
separate frame-position and certified velocity-interval masks, in prepared and
legacy geometry losses. Forty focused tests pass across control Jacobians,
source-support interval fixtures, command modules, real-source command replay
and checkpoint declarations. Tests establish implementation properties, not gait.

source_intent_semantics_v1 revealed both old start cases are from
bones:turn_start_walk_0315, with maximum requested/demonstrated facing differences
147.14/132.80 degrees. Long-clip net travel and preserved initial facing were not
valid labels for these local turn-start windows. Both excluded from new training.
Local episode_controls now uses local start travel and travel-facing, with a
regression test; completed snapshots/data are unchanged.

rest_restart_data_v2/cases.pt SHA
2de74cfd3b08db79827636777135ff40868ec998bc50ce58214977c425ee5ad4
contains6real uninterrupted rest->move recordings,4train and2family holdouts
for THIS probe, not unseen pretraining. Real rest1.87..3.67s; no invented16s
rest, resets, padding or pose stitching. Sources pass unchanged2cm full-skin
screen; actual maxima.40..1.15cm. Zero input retains initial facing; MOVE uses
post-rest planar weak speed/travel-facing, permitting genuine turn-and-start.
Future body only screens source validity and supplies loss targets, never actor
conditions. Earlier selectionv1 failed an incorrect initial-facing assumption;
preserved. v2corrected command semantics,not collision thresholds. Weak labels
are not recorded keys. Three selected restarts turn79/33/137degrees; one3.7deg.

Bounded matched experiments restart_control_v1/restart_phase_v1 completed under
task/parc_rugged32_20261013_v2, immutable code_bones_control_repair_v1. Each16actual
updates,not batch32/50k. Both start from unchanged orange command_all_v1 checkpoint
SHA43711ba1a93246f49e6221eb7a1172c323e31b5472d5787520a0bd4dd346a80e.
Fresh Adam5e-5,clip.5,unchanged coefficients; only Root command and Pose/decoder
command adapters train. Native networks,codec/vocabulary and terrain/SDF frozen.
Each update:one true long idle,one continuous restart,one consistent walk/run,
one original PARC training anchor,deterministic paired yaw on flat cases.
Old inconsistent start cases and first walk excluded; second walk retained.
No command reset at MOVE. Both arms compute the same auxiliary real-source actor
and sourceAE window. Phase arm reallocates native supervision in restart episodes
to mean(first plan,real restart plan); control keeps first plan only. No fake
source-token targets attached to far-drifted generated histories. Existing full
episode geometry still uses fixed-time references; its known feasibility risk
has NOT magically disappeared and must be assessed by outcomes.

restart_smoke_v1 one-update preflight completed:future-target,training-ST/runtime
forward errors0; full temporal gradients,all three trained modules and frozen
hashes pass.99generated primitives,one diagnostic optimizer update,not part of
either16update run. Initial launcher used a relative working path and failed
before process creation; corrected absolute path launch succeeded. GPUs2/7 were
freshly checked free; PID644831/646270 are identity hints only. No other's job
touched. Baseline and final6restart/16flat/12stairs reports retained automatically.

Live loader now installs declared command modules before strict checkpoint load;
legacy checkpoints retain the old graph. Missing/mismatched declarations fail.
code_bones_live_check_v1 is a separate snapshot. command_live_parity_v1 used the
unchanged orange weights,CPU,13real online backend publications acrossW/release/
A/release/S/W restart. Direct training-protocol and published motion/history
match exactly0 in every block,using default knobs. No future source or correction.
This is interface parity,not new candidate quality or a browser/30fps test.
No new weights downloaded or live server restarted. Final fitting results and
material-point skating checks are above; do not substitute root progress.

## Current Review: Incorrect Stop Jacobian And Missing Transition Supervision

Latest user says orange is better but still often cannot step and skates; asks
for careful principle/code REVIEW. No new training or model/runtime changes.
Only read-only audits added. Current28939 and all old artifacts remain unchanged.

P1 CONFIRMED TRAINING-GRAPH BUG: pilot_bones_command_alignment.py49 adds
history_current-history_current.detach() to goalXZ UNCONDITIONALLY. For MOVE,
goal=current+requested_velocity*horizon,so that history derivative is appropriate.
For an already retained STOP anchor,goal must NOT follow the current history.
Inside terrain_model_v3._predict,goal-current consequently has its stop-position
feedback derivative canceled. Same forward values do not mean same backward
dynamics. command_gradient_review_v1.json uses actual SoftCommandState and exact
shift/asinh expressions:anchorX0,currentX.4,zero request yields autograd derivative0
but numerical forward derivative-.928476690881. MOVE gives0/0 as expected.
This is one canceled control-feedback path,NOT zero gradients throughout model.
The shift was inherited from moving-only PARC whole-passage training and is not
valid unconditionally for the new long-idle cases. Future repair must distinguish
moving goals,release-time anchor capture,and retained anchors;not detach all
goals or move the anchor on every STOP step. No repair has yet been trained.

P1 VERIFIED DISTRIBUTION GAP: eight BONES source sequences and four PARC anchors
have ZERO within-episode stopped->moving request changes. episode_loss starts
every case from source[:4] and a fresh SoftCommandState. True source-starts exist,
but the generated16s-rest state is never continued with MOVE during training.
Separate continuous idle and start episodes do not train this composed event.
Our previous language about restoring continuous feedback was insufficiently
specific. Data/actor review results must distinguish clean-source starts from
restarts after self-generated rest;no fake pose stitching/reset is allowed.

P1 SUPERVISION/EVALUATION GAP: native_losses is called only for the first32frame
plan in each episode. Long idle has60later plans without that native motion/Root/
masked-token loss; whole-episode FK,source sole XYZ/velocity and factual-prefix
objectives still supervise those outputs. Do NOT claim completely unsupervised
later motion. SourceAE is a separate source reconstruction,not a Pose token loss
on each generated continuation. Native skate loss also gates with thresholded
predicted contacts; it is not an independent support certificate. Do not simply
add time-aligned source-token targets to far-drifted histories: earlier audits
proved such targets can demand impossible corrections. Need feasible continuous
supervision or explicitly declared causal locomotion goals/priors.

READ-ONLY MECHANICS: mechanics_review_v1/complete.json and mechanics.png reconstruct
official sole material points from the downloaded immutable recordings. In the
orange forward8s,foreaft feet-close fraction27.5%(not just last32's9.375%),longest
close run1.333s,first2s close68.3%. Near-plane sole material speed medians.227/.206mps,
P901.94/2.08mps(L/R).79.6% of intervals have neither foot with>=3points within2cm
of plane and speed<=.1mps. This is a kinematic support proxy,not measured force
contact;do not call it79.6% airborne. First2s forward.23m,last2s4.24m. Generated
contact-positive channel fraction40.2%,not an actual contact measurement. Previous
root progress and tail-leg-spacing evidence did not establish natural gait.

P2 CONFIRMED SUPPORT-MASK BUG: terrain_joint_objective.py75(and legacy192)
reconstructs velocity intervals from the UNION of supported endpoint frames.
This can bridge an uncertified middle interval. Actual prepared_losses fixture
with source X[0,0,.01,.01] and stationary predicted foot has certified[T,F,T],
but used[T,T,T],giving slip loss.1mps rather than0 on certified intervals.
This is a synthetic scalar loss test,never model training or fake motion input.
Unrotated actual source audit finds0bridges in both idle/walk examples,1in one
start,56/85in two runs,and5/7/6/34in stair anchors;at most about3% of used material
intervals. Real bug,not sufficient explanation of all visible skating. Position
support masks and velocity support intervals must not be conflated.

FALSIFIED SHORTCUT HYPOTHESIS: command_decoder_path_review_v2 ran read-only CPU
inference(A100 GPUs occupied,none touched) on six factual recorded8-row histories.
It holds final tokens,Root external conditions,history and other modules fixed,
and disables only the new decoder-latent command residual. Maximum mean forward
speed change .00195mps;foot foreaft RMS change<=.00227m. Replayed actor versus GPU
recording maxfeature error<=2.25e-5;decoder restoration and re-enable check pass,
model tensors unchanged. Does NOT support new decoder command branch directly
pushing Root as the principal cause in those samples. Does not exonerate the
whole decoder or identify Pose as the unique cause. V1 import failed before any
model call;v2 inlines the existing native coordinate restoration,not a motion fix.

Principle correction: commanded movement is not a gait target or support law.
Root/body-relative foot velocity must combine to a near-stationary world foot
during actual stance. Model can move its Root and separate its feet yet skate.
MB's authored locomotion keyframes/timing provide more than a broadcast MOVE bit;
explicit hand-coded phase is not inherently required. Do not infer from this
that IK,Root replacement,output smoothing or zeroing both feet is acceptable.
Next fix graph/mask correctness with numerical tests,then valid continuous
generated-state transitions and full-motion/support criteria BEFORE new fitting.
All new audit jobs complete;no new optimizer updates/deployment/automation.
Small JSON reports copied bySSH;no weights/animations transferred over publicSSH.

## Current Work: Matched Causal Command-Interface Probe

User agreed to repair the control-interface/training alignment, then small A/B.
Implemented locomotion_command_adapter.py: desired asinhVX/VZ, requested heading,
horizon and explicit MOVE flag condition native Pose queries and decoder latents.
The stale retained release-position delta is NOT a body-style feature. No future
body/height/keyframe truth supplied. This is a learned causal adaptation inspired
by the paper's explicit locomotion intent, NOT a literal Smart Locomotion replica.
Initial residual outputs exactly zero; no output projection/smoothing/IK or gate
that holds still frames. Existing live28939 and approved28937/28938 unchanged.

New controller_probe_data_v1 COMPLETE,sha05ce01dfc0c3bfa3394576bc338abddb820eb28608777eb6dce17a062b1b4c9d.
Eight real training sequences:two513frame idle, two97frame stationary-start,
two97frame efficient walk,two97frame run. No padding/pose stitching. Native
training split and prior pilot-reserved families maintained. Actual complete
source skins pass existing2cm peak gate. Initial body/travel angle plus source
planar weak speed/direction labels supply commands; future body/Y mutation does
not change them. Source controls are declared weak labels, not actual keyboard.
Four original PARC training anchors included separately; original6case stair
review retained. Long idle uses61continuous8frame primitives,others9; staircase
primitive count comes from each real case. Both arms use identical random yaw
within cardinal quadrants on BONES against unchanged zero plane; PARC source/
scene remain paired without augmentation in this bounded probe.

COMPLETE matched arms command_root_v1 and command_all_v1 under
task/parc_rugged32_20261013_v2, immutable code_bones_command_v1.
Script scripts.pilot_bones_command_alignment,8actual updates per arm. Root-only
arm trains existing Root command MLP only. All-stages arm additionally trains new
Pose/decoder command modules. All original Root/Pose/codec,encoder/vocabulary,
terrain/SDF networks remain frozen. Same deployed candidate parenteb631648...,
fresh Adam5e-5,clip.5,unchanged original losses,full generated-history episode
gradients and persistent SoftCommandState. One long idle +one start +one walk/run
+one PARC episode per update; one sourceAE and additional source actor call each.
Actual actor exposure is recorded,not batch32 or50k terminology. Initial/future
target/training-ST/runtime errors all0 for BOTH real-model preflights. Baseline
flat metric JSONs byte-identical7e77accc5d951f1749b2b514a46deae7da96ad40af852c4be1f90de908f9ba8a.
First-step all-stage gradients Root2.7750,Pose1.8446,decoder1.5580; temporal gradients
positive and finite. Fixed original tensor hashes checked after every update.
17focused local tests pass,including both-stage hooks and future-body isolation.

Physical GPU5/6 were freshly checked before launch. Both jobs finished; do not
operate on recycled PID148731/149391. No duplicate/restart or automation.
Final review repeated the actual large-map4frame idle seed on pure flat:
four headings x1.6/5mps,each immediate-start and16s-idle->8s-move; actual720frames
retained for idle trials,240for immediate. Exact full-skin metrics and gait are
separate from root progress. Original6stairs x2speeds also complete. Do not claim
fix from training loss or model wiring. No new weights transferred/deployed.
The existing world seed was privately transferred andhashe19876a3...checked.

RESULT:8actual updates EACH,688continuous generated primitives,32sourceAE uses
and32additional source actor preparation calls. Source/yaw/primitive schedules,
exposures and complete initial flat/stair metrics match exactly. All original
network/terrain/vocabulary tensor hashes remain unchanged. No failed case removed.
Previously stalled yawPI after16s idle,then8s forward at requested1.6mps:
- current deployed parent:.49m forward,1.00m lateral,last speed.10mps,last32feet-close100%;
- Root-command-only:.05m forward,.66m lateral,last speed.014mps,feet-close100%;
- all-stage command:11.10m forward,1.53m lateral,last speed2.39mps,feet-close9.375%.
This supports the all-stage command path for THIS startup failure at this budget;
does not establish that Root-only can never learn or that native keyframes were
reproduced. All-stage16s idle still drifts1.26m at yawPI; moving skin peak9.11cm.
All four1.6mps restart headings are moving by the tail, but maximum lateral error
is1.53m and flat peaks range4.91-9.11cm. Stair quality0/12 for before and both arms.
User-approved up0.7 scene peak worsens from current mixed8.50cm to all-stage14.00cm;
root-only7.73cm is still worse than accepted pure-PARC.69cm. Therefore NOT adopted.

All local JSON reports preserved as command_root_v1-report.json and
command_all_v1-report.json. command_comparison_v1 contains original/root-only/
all-stage official SOMA for the predeclared yawPI failure and yaw0 controls,
plus ALL16flat/12stair metrics. Three clips,unmodified720/240frame durations.
29receipt-listed files privately downloaded/verified,archiveSHA
f2834963597603d7727a35f04fd3e40f7f30352ecfc277671bfe5f3a6cde7df3.
Use compatible equal-length viewer:
http://127.0.0.1:28936/web/decoder_pilot/?data=/out/parc_rugged32_20261013_v2/command_comparison_v1/visuals/&clip=idle16s-forward-yaw180&left=before&right=all_stages
Actual screen reviewed at frame660 and saved frame660.png. Kept background tab,
paused at frame480(16s,first forward input); user28939 page not touched. Initial
attempt with variable-length terrain_route_review failed missing per-arm frame
metadata; used the matching decoder_pilot format without editing motion/receipts.
Next work must address idle drift, lateral fidelity and support/terrain retention
using the new causal evidence, not automatically scale this failed candidate.

## Current Work: Flat Restart Failure And Original MB Control Review

Latest user reports intermittent inability to walk and drifting on FLAT ground,
asks to check train/inference alignment, and explicitly reminds us to study the
original MotionBricks paper. This supersedes expanding terrain data/training now.
No new optimizer updates, model deployment, runtime correction or automation in
this audit. User is actively trying28939; do not reset or overwrite their session.
Browser initially showed speed5m/s with trained defaults8tokens/4passes/1refinement,
12mfar/2mhalfwidth/8frame refresh. Later user changed speed to1.6. We did not change
their controls. Failure was separately reproduced at1.6, so do not blame5m/s alone.

Local live_parity_v1 and live_parity_idle_v2 COMPLETE under
out/parc_rugged32_20261013_v2/. Script audit_bones_live_parity.py loads exact deployed
42b4eb49... checkpoint and existing world's factual4frame BONES seed, places it on
a genuine zero plane, then feeds explicit commands only. No future reference.
Direct actor versus JointTerrainOnlineBackend: five move/move/stop/stop/restart
blocks have world/history/stop-goal maxdifference0. Serial versus committed-history
prefetch: four further move/stop/stop/restart blocks maxdifference0. This validates
these same-model backend paths, NOT original-MB control semantics or all UI timings.

Four rotated initial headings x1.6/5mps,240played frames each, demonstrate marked
direction-dependent response. After60zero-command blocks (16s) at yawPI, drift
already1.22m. Then1.6mps request for8s produces only.49m forward and1.00m sideways;
last-second forward speed.10mps,all steps<.2mps,last32close-foreaft-feet fraction1.
Same heading without prior idle produces11.37m forward and2.03m sideways in8s.
The stalled case is a learned/history-dependent failure even without terrain;
not missing browser packets, and not proof that all headings or all starts fail.
Other headings also have gait/side-drift issues; raw eight-case NPZs remain local.
No output frame padded/modified. These tests are not a terrain or quality pass.

CPU replay of actual800BONES samples complete: bones_training_input_audit_v2.json.
Each category100samples, but start has only29stationary-to-moving examples;
only3/800requests>=5mps. All newly fitted histories are source, no generated-history
training and no persistent stop-state training. Root/position/rotation re-encoding
differences <=1.44e-6; redundant local_vel max5.20mps differs at window boundaries,
but EXTERNAL_POSE_FEATURE_MODE uses positions/rotations and factual-prefix loss
excludes redundant velocities. Do NOT misreport that unused channel as an actor
leak or measured cause. Auditv1 failed a global-vs-local feature-key assumption,
v2 corrected the query, no training/model changed.13focused control/sampler/history
tests pass. All audit exec sessions complete. Live services intentionally remain.

Re-read primary paper https://arxiv.org/html/2604.24833v1 sections3/4/6.1/AppendixC
and current official raw demo/controllers.py and demo/full_agent.py from
NVlabs/GR00T-WholeBodyControl. Also inspected tracked local source at HEADab1add2
and the12-line SOMA extraction diff; it does not add the missing primitive.
Original MB converts commands to target keyframes through Smart Locomotion,
using an input-side spring proposal, style-pose selection and learned timing/Root
refinement. Paper training uses random yaw, not heading canonicalization. The
G1 demo separately has optional runtime canonicalization. Do not conflate these.
Paper recommends3-9frame replanning; our8frame interval is within that range.

Concrete current divergence: sparse_soft_intent_conditions masks all future
Root/pose slots; terrain_model_v3 feeds seven scalar command features primarily
to Root through a learned embedding, with fixed8tokens. It does not instantiate
NativeSmartLocomotion/NativeReferenceClips or provide native idle/walk target
poses. pilot_rugged_transfer bypasses native batch preparation and has no random
yaw augmentation, despite unfreezing base Root/Pose/decoder. Source-history-only
fit replaces the accepted parent's whole-passage feedback phase. These are
verified interface/training differences, not a proved single sufficient cause.
Earlier20261010/stop-study notes already recognized the smart-primitive gap;
do not present that background as a wholly new discovery or repeat failed stop
gated-MLP trials blindly. New evidence is this exact deployed-model parity and
long-idle flat restart failure, with explicit input coverage audit.

Next: resolve the intentional-control interface before another generic fit.
Study a matched input-side locomotion condition/phase prior, exact causal command
state and rigid yaw augmentation, plus generated-history idle/start sequences
with real source coverage. Fixed library style keyframes are authored goals,
not the actual unknown future, but restoring them changes the current declared
targetless-pose interface: needs an explicit new contract and matched evaluation,
not disabling existing validators or injecting held-out/source-time futures.
Do not copy G1 source_root_realignment, output qpos blending, repeated idle tail,
or hold-last-frame behavior to conceal drift. Springs on requested inputs and
output smoothing are distinct, but no input change is adopted without testing.
First pass long idle->start in four headings,turn/release/restart,flat footing and
accepted stairs; preserve all failures. New training was deferred for this paper/
source review. The world NPZ alone was staged through owned privateDrive at
private-store:motionbricks/parc_rugged32_20261013_v2/live_alignment/terrain-world.npz
for potential A100 diagnostics; no public SSH animation/model transfer occurred.

## Latest User Override: Experimental BONES Large-World Trial

User explicitly requests replacing the trial with the BONES candidate on the
large map to try it. This authorizes an EXPERIMENTAL interactive preview despite
the known stair regression, NOT a quality-pass claim. Selected rugged_bones50_v1
(fixed SDF), not the jointly adapted arm: better flat skin peak3.69cm versus5.58cm
in the direct command test, and slightly less accepted-stair peak8.50 versus8.94cm.
No weights retrained or output correction added. Approved28937/28938 remain intact.

New localhost28939 is running the native eager actor on existing terrain-world.npz:
256x256m,36test zones plus central plaza, factual BONES4-frame reset seed. The
authored heightfield/mesh hashes are verified; this map was NOT added to training.
Source bundle0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d.
Original candidate checkpointeb631648f5144d5353384d74c85059d1ba2bbe05379492233f7a9a8a0bfb7cde.
Portable checkpoint42b4eb4976ca580433cc4c83ffc1a519cfcc7296bd80cc93758e0318da20e8d6;
tensor digest72c7664781243faf117b0741610d20a90f60edd16d6f780ee7ba619166432cb7.
Only optimizer/RNG/cursors removed; all model tensors and stage contract unchanged.
Private owned-client Drive upload/download complete; all receipt files verified.
Local export out/parc_rugged32_20261013_v2/bones_world_live_v1. Packaging helper
package_bones_world_trial.py and launcher launch_bones_world_trial.ps1. Local PID33476
is only an identity hint; inspect cmdline before operations. No automation created.

Actual browser W/D/S/A tested, native official SOMA rendered, no browser errors.
W advances from(-20,-112) to(-19.73,-110.16)XZ in56played frames. Direction keys
reach438played frames before pause, but this is wiring verification, NOT complete
directional/terrain quality validation. Display FPS29.5-30.0, not stable30fps
certification. Region reset to Rock field6cm and W44frames works,display29.4fps,
zero reported underruns. Returned to Stairs16cm, reset paused at entrance; W starts.
Screenshot world-ready.png, rough-trial.png and browser-check.json saved with export.
Visible label BONES + PARC trial / EXPERIMENTAL. Controls, sliders and world map
retained. User-facing deliverable http://127.0.0.1:28939/. Failed quality remains
recorded; user trial approval is not broad terrain or collision clearance.

## Current Work: BONES Seed Mix And Interactive WASD Goal

Latest user explicitly requests BONES seed in training and reiterates that the
delivery MUST remain an actual WASD character controller, not recorded clips or
an automatic route demonstration. This overrides the older PARC-only new-fitting
restriction. Approved local28937/28938/ae558ee8... baseline remains untouched.
No new candidate is approved or deployed merely because inputs reach the model.

Completed rugged_bones50_v1 under parc_rugged32_20261013_v2. Same accepted parent,
same200updates/Adam5e-5/losses as archived rugged_fit_v1. Every batch retains
2new PARC+2original PARC windows and ADDS4BONES:800PARC+800BONES=1600actor samples,
plus1600sourceAE uses. Batch8 versus previous4; same PARC exposure, NOT identical
compute or optimizer trajectory. PARC exposure equality verified. BONES categories
idle/walk/run/start/stop/backward/lateral/turn each receive100real33-frame windows.
Start/stop sample real beginning/middle/end, no padded or synthesized frames.
Source manifest c250f57dfd13ee7907a6ae0239dfbba26f8655b2dd64c5ec8f84ae937cb75de7,
normalization hashes unchanged. Original held splits retained. Backward/lateral
lack an original held split, so entire jog_backward_loop/jog_sideway_left_loop
families were excluded from THIS pilot's training for diagnostic evaluation;
they are not claimed unseen by inherited pretraining. Other1914eligible train
records remain. Sampling/control implementation bones_seed_rehearsal.py.

BONES requests use weak planar trajectory labels; facing uses only actual observed
history and requested travel, never future source body facing/height. During
rollout, observed history is generated history, not later source poses. Fixed
zero plane is preserved/extended as an infinite flat domain, no pose/floor fit.
Actual BONES-only gradient preflight Root9.8686,Pose11.5911,decoder7.1265 confirms
these samples reach trainable base predictors despite SDF flat cancellation.
Initial PARC rollouts match archived control max0. Future-target mutation0,
runtime difference0,ST difference2.002716e-5. Frozen hashes pass. Root/Pose/decoder
trainable, SDF/encoder/vocabulary frozen in this completed arm.

Observed changes: BONES run endpoint error3.59->1.54m,turn6.29->1.21m versus
accepted baseline; idle drift.37->.30m. Not all improve: walk overshoots, stop
tail speed remains1.16m/s. Original stair screens3/12->0/12; accepted up0.7 clip
skin peak.00687->.08504m. Not a replacement for the approved controller.

bones_position_review_v1 also COMPLETE: same known XZ route, input turn triggered
by actual generated position, no teacher future height/pose, no output changes.
Approved baseline reaches0/11goals; BONES arm reaches5/11 (3/7source-qualified).
Root milestone traversal was required, not just endpoint proximity. Penetration
still about9-25cm, so these are NOT full quality passes. This is an automated
planar-input diagnostic, NOT human WASD or learned navigation. Serial controller
tests do not by themselves certify the browser's prefetched command latency.
Helper review_position_joystick.py has tests for future-body isolation, no mutation
of history, and no skipping an unvisited waypoint.6focused local tests passed.

COMPLETE: rugged_bones50_joint_v1, code_bones_rugged_joint_v1, unique script
pilot_bones_rugged_joint_v1, --train-terrain. Fresh same accepted parent, same200
updates/batch8/windows/losses/LR as frozen-SDF BONES arm; additionally trains SDF
encoder/joint embedding/Root/Pose/decoder adapters. Encoder/vocabulary/old control
remain frozen. Initial baseline and all800BONES draws match; frozen hashes pass.
Future-target mutation0,runtime difference0,ST difference1.859665e-5. Actual SDF
gradients are nonzero. Old stairs0/12; accepted up0.7 peak8.94cm versus8.50cm in
fixed-SDF arm and0.69cm in accepted baseline. No replacement. Flat endpoint errors
improve in some categories but not all. Completed joint position-input review
reaches5/11goals, with10-25cm penetration, NOT five full quality passes.
Metadata caveat: inherited parc_only_control_bones field/folder in JOINT report
actually contains the fixed-SDF BONES arm, selected by --reference-run. Do not
mislabel that reference as pure PARC. Raw completed reports are retained.

COMPLETE: bones_wasd_backend_v2, code_bones_wasd_review_v2. Approved/fixed-SDF/
joint each generate480actual frames under60eight-frame blocks,9input phases:
W,release,A,release,D,release,S,release,Wrestart. Requested speed1.6m/s; release
velocity exactly0. Only four initial factual frames from the held BONES walk
are used; NO future source speed/path/pose drives these commands. Generated
history and persistent SoftCommandState feed the real JointTerrainOnlineBackend.
Runtime defaults, joint_terrain_live.py, native_mb_online_live.py and core
joint_sdf_controller.py match local code hashes. Serial fully-played blocks do
NOT certify browser-prefetch latency, physical keyboard input or30fps.

All three finish without map errors; no output changes or padding. Release A/D/S
tail speed P90: accepted1.16/1.22/1.27m/s; fixed-SDF BONES.28/.47/.26;
joint.26/.35/.21. Restart tail velocity along requested direction: accepted
-.41m/s versus+.94/+.86. This supports improved response in THIS single continuous
test, not general controller qualification or instant stopping. First W release
still accumulates.55/.44m drift AFTER the first second for mixed arms, despite
later slowing. Flat full-skin peaks4.53/3.69/5.58cm. Terrain regression remains,
so neither candidate is deployed. Slow stair movement alone is not failure;
penetration and wrong-direction/continued unrequested movement are separate.

Diagnostic v1 exited at import before model inference: runtime settings module
missing in inherited snapshot; it also contained stale online backend code.
v2 is a separate hardlink snapshot: old backend entries were renamed BEFORE
copying exact current local files, preserving all completed training snapshots.
Import check and backend/core hashes checked before launch. Failed v1 log remains.
No training was restarted. All new mixed training/position/backend jobs complete.
Six focused sampler/control/waypoint tests pass locally. Do not act on recycled
process IDs or reuse GPU occupancy snapshots. Approved localhost28937/28938
listeners remain unchanged. No model/animation transfer over public SSH.

Next investigate accepted-stair retention and generated-state recovery alongside
these direct control tests; do not blindly enlarge another source-history fit.
Keep failures/actual frame counts. Only a qualified candidate
gets a NEW local live port, hash verification, official SOMA, actual WASD, start/
turn/stop/terrain-transition and frame-rate checks. Preserve old pages. No IK,
foot locks, collision projection, Root replacement, output smoothing or future
teacher body conditions. Large artifacts only owned private Drive/NAS/internal.
No new automation. Detailed completed JSONs local under
out/parc_stair64_20261013/: rugged_bones50_v1-report.json,
rugged_bones50_joint_v1-report.json, bones_position_review_v1-report.json,
bones_joint_position_review_v1-report.json and bones_wasd_backend_v2-report.json.

## Current Work: User-Approved Live Baseline And Rugged32 Expansion

FINAL STATUS OF THIS ROUND: collection, dataset build, source audit and
rugged_fit_v1 are COMPLETE. No active generation/training/audit expected; verify
real processes before any action. Expanded model NOT adopted; user-approved
live ports/weights unchanged.200actual updates/800actor primitives plus800AE
completed. Future-target mutation error0,runtime error0,ST difference1.621246e-5;
frozen hashes preserved. Source-history adaptation is NOT feedback recovery.

Original6stairs x2request scales:3/12geometry screens before,2/12after. Exact
user-approved generated_afe4ea1b1d92b874-up-speed0.7 regresses:40->48frames,
skin peak0.006870m->0.153227m,soleP990->0.145588m. Do not replace that visual
baseline just because training loss decreases. New11-case recorded-direction
replay:0before/1after finish the input duration without a map-query error, but
NO complete ordered route; that remaining case covers only2/30root milestones.
This is fixed-time offline INPUT replay, not a free-WASD verdict: model/expert
speed differences can make90-degree commands arrive before/after the model's
platform arrival. All failures/actual lengths remain in baseline_new/final_new;
source eligibility flags identify4unqualified sources. No trained rugged
controller is certified. Next separate command timing with position-triggered
PLANAR route requests (no future height/pose or output corrections) from generated-
history drift, and explicitly measure accepted-motion retention. More source-
history fitting alone is not a demonstrated solution. Full small report is local
out/parc_stair64_20261013/rugged_fit_v1-report.json. No candidate weights or
animations were transferred over public SSH.

LATEST EXECUTION: all32candidates generated.12native full-route candidates,
11complete physical/SOMA preliminary admissions. Full18056vertex/8substep exact
source audit passed7: train stairs1/rough1/mixed1,validation stairs1,test ramp2/
mixed1. All11audited and all32candidate outcomes retained; no failed records
deleted or thresholds loosened. Training-ramp layout2 and validation-rough5
had0complete native routes. Do NOT move the test ramps into training.

dataset_v3/parc_native completed and verify_dataset passed. It preserves538old
PARC anchor records and adds11new preliminary records, NOT549new samples. Both
mean/std SHA256 match the approved model's mixed_dataset exactly. Original
generated-only conversion was rejected by the existing both-PARC-origins policy;
the working incremental conversion keeps the original author-released/generated
PARC anchors, with no BONES newly mixed in and no validator bypass. dataset_v1
stopped on the last-worker GPU-occupancy guard, dataset_v2 on source-origin policy;
both failure records remain. Last-layout physics was retried ONLY after a fresh
idle check, using recording_v3/ to preserve the empty failed recording_v2/.
No motion was regenerated. All collection workers are now finished.

Completed model pilot: parc_rugged32_20261013_v2/rugged_fit_v1, script
scripts.pilot_rugged_transfer in code_rugged32_v2. Supervisor/runner PID3204878 is
an identity clue only; inspect cmdline/state/progress. From accepted ae558ee8...
checkpoint,200actual updates,batch4,800actor primitives+800sourceAE uses. Each
batch has2new audited source-history windows +2original four training PARC stair
sources. Only3new source IDs train; held layouts and failed source skins do not.
All original losses/coefficient scales,Adam5e-5,clip.5 retained. Root/Pose/decoder
trainable;SDF/codec encoder/vocabulary frozen. Source-history-only adaptation,
NOT a claim of new generated-history recovery training. Cruise magnitude is an
explicit weak estimate from the real flat approach, recorded directions are unit
vectors, initial facing offset retained; no future teacher height/pose enters
the actor. Every25updates immutable-output checkpoints, frozen hash checks;
actual first-step target mutation/runtime/ST preflight is in preflight.json.
All200updates completed normally. Initial/final new-terrain review is OFFLINE
recorded-direction replay with inferred cruise, NOT free WASD. Root waypoint
coverage rejects merely standing at the goal. All11new cases/failures stay in
review, with source eligibility flags. Existing6stair cases are regressed at
both request scales. No model deployed or large neural budget started.
6focused geometry/provenance/control/ordered-route tests passed locally.

2026-10-13 latest user says the live result looks good and requests expanded
training toward traversing rugged terrain. This authorizes a controlled new
data/learning round; it is NOT permission to replace the accepted model silently
or to call its remaining support/reverse-direction issues resolved. Keep local
28937/28938 and accepted ae558ee8... parent unchanged. Local portable SHA
8b75107a69b23abf17fdf1377d2e61258d21288aaa9cc10b0cacf145c20884aa reverified.

New bounded collection: task/parc_rugged32_20261013_v2.32candidate motions across
8distinct16m layouts,4candidates per layout. Families stairs,rough,ramp,mixed;
first round uses controlled low rise profiles across the lane, not arbitrary
2D rock fields, cliffs or overhangs. Two genuine90-degree turns are on a real
level platform, with flat entry/exit. Four cardinal global orientations. Whole
geometry splits assigned beforehand:0..3train,4..5validation,6..7test. No STOP,
mid-sequence history reset, pose stitching, source motion edits or runtime IK.
Future actor conditions remain factual history,current planar request,actual
terrain and model predictions, not future teacher height/pose.

build_parc_rugged_course.py geometry tests2passed; all8actual native PARC terrain
preflights matched their routes, unique hashes. Initial test exposed an endpoint
in the flat-platform assertion that already belonged to descent; segment bounds
were corrected before any generation. Corecourse geometry did not change later.
Record native direction commands before diffusion with exact source frame clocks;
these are UNIT directions, NOT m/s. Do not silently treat them as speed labels.

Active snapshot code_rugged32_v2; new recovery runner scripts/run_parc_rugged_v4.py,
generator generate_parc_rugged_v2.py, probe_parc_rugged_physics_v2.py, admission
module scripts.admit_parc_rugged_v1. A100GPU0/2 newly verified free at every launch;
GPU1 belongs to someone else. Supervisors3042060/3042685 are identity clues only.
Read actual cmdline and collection/worker_0.json + worker_2.json first. Do not
duplicate or overwrite healthy snapshots. Each worker locks its assigned GPU and
rechecks occupancy before stages. Old finished full-budget campaigns stay off.

Recovery history is preserved: V1 stopped before generation due to a missing
wrapper dependency,0candidates. V2 generated4+4 candidates, but the missing
physics entrypoint stopped the next stage. V3 reused those same outputs and hit
a legacy-report compatibility issue BEFORE simulation: missing sampling_profile.
The audited legacy generator hash4623d414... and unchanged official config
hash26406180... prove actual DDIMstride5/CFG.65 matches official_data. New helper
parc_generation_profile.py verifies both hashes and actual settings, refuses an
unknown generator or altered sampler, and records provenance WITHOUT modifying
raw generation reports/motion.2targeted tests passed. No silent guessed profile.
V4 reuses completed generation by hash, preserves failed preparation in recording/,
and writes fresh physical records under recording_v2/. Old logs/receipts retained.
Later imports and CLI entrypoints preflighted before continuation.

Historical first8 candidates:5 complete native routes (2train,3validation),3failures retained.
Prefix maxerrors about1.8e-7, below existing3e-5 criterion. Official physical
tracking of those5 produced4whole records and1partial; partial excluded from full
route admission. Existing SOMA screens admitted2train +2validation,20.6seconds
per split. These are preliminary teacher admissions, not proof of learned MB
traversal. All eight layouts have since finished as recorded above.
Post-tracker route coverage, exact full-skin
source/support/continuity review and command/time/coordinate checks remain before
the next bounded fit. Keep all source/candidate failures and actual lengths.

Expanded model must regress against the user-approved stair clips and use held
layouts/continuous requested directions, not a copied source timing schedule.
No blind geometry-weight scaling or adoption of rejected continuity/Pose-support
trials. All large animations/data/models stay on A100 or owned privateDrive/NAS;
no public SSH binary transfer, shared Google client or new automation.

## Current Work: Approved Baseline Live Trial And Support Review

2026-10-13 latest human paused Unreal's realtime viewport at our request. Do not
operate or close Unreal. GPU contention was real: the SAME down model warmup
fell from54.6seconds to620ms after resources became available. This does NOT
explain the model's raw8-frame replan discontinuity. Both must stay separate.

The user-approved pure_parc_fit_v1 episode checkpoint ae558ee8... is preserved.
Private approved_live_v1 portable weights SHA8b75107a69b23abf17fdf1377d2e61258d21288aaa9cc10b0cacf145c20884aa
remove only optimizer state; model tensor digest unchanged. Seven receipt files
verified. The live scene loads only actual4-frame history plus terrain, NOT the
separate offline-cases.pt. No runtime IK, projection, Root edits or smoothing.

Live trial URLs: up http://127.0.0.1:28937/ and down http://127.0.0.1:28938/.
Own current processes125488(down),176876(up), identity clues ONLY. Check full
command before any operation. Native eager, committed-prefetch,start-paused,
exact approved model, original trained defaults. Old28934/28936 kept untouched.
Startup officially skinned full vertices versus77-rig transport max error0.
approved_live_probe_v2 stream versus serial max feature error0 for40up/64down
frames; local versus A100 Root max1.907e-6/3.099e-6m. No claimed bitwise CUDA
cross-machine equality. Only our immutable already-published model frames may
serve as early next-block history; actual played history waits for the ack.
SoftCommandState is copied independently; no source future supplied.

Actual browser W tests completed, nonblank SOMA verified. Rates vary26.2-29.8fps,
NOT a stable30fps pass. Latest recorded42up/56down frames were26.16/26.32fps,
with8/10 commit-wait ticks despite0 ordinary queue-underruns; do not hide these
waits. Actual packet latency about243-312ms for8frames. A/D/S requests reach the
model and alter motion. HOWEVER S still displaces the initial down body from
(-.15,-.50) to(.26,-1.17)XZ over36played frames despite command(-1.102,+.215)m/s:
reverse response is not correct yet. This is a LIVE TRIAL, not a complete WASD
controller. STOP remains outside this fitting round; manual Pause is available.
Browser evidence approved_live_v1/browser-check-v1.json and live-up-frame24.jpg.
The approved visual baseline is explicitly labelled separately from unvalidated
interactive behavior; inherited robotics/display-only safeguards remain.

Playback review: UI was holding a short recording until the longer source ended
and looping both; it now ends explicitly, retaining actual lengths. Paused route
viewer used to redraw both bodies every display refresh. New renderDirty path
redraws on actual frame/camera/resize changes only, not by modifying motion.
Browser count stayed72 across successive paused observations, frame25 shown;
recorded end48 remained without automatic wrap. Screenshots saved in
continuity_export_v1/approved-vs-boundaries-frame25.jpg. User tabs not closed.

continuity_control_v1 FAILED atupdate3 due to an unknown generated terrain query;
only2updates/56actor primitives retained. Reviewed/exported without restart or
reset. continuity_all_boundaries_v1 completed8updates/224actor primitives+32AE,
geometry screen12/16 vs accepted baseline3/16, but NOT adopted. Held-up0.7 raw
boundary mean deltaV1.711->1.371m/s,max2.892->2.055; allsoles>5cm is100% of48frames
vs72.5% of40baselineframes, source10.1% of89. Lower penetration is not grounded
walking. Geometric gap is NOT force-contact truth. This is not matched8vs8
because the control failed. Private continuity_export_v1 has160verified files.

continuity_root_v1 completed8updates with additional native-normalized decoded
Root height/incoming velocity supervision on actual published frames. Two tests
check loss/gradient coverage. It remains a diagnostic candidate, NOT deployed.
support_gradient_v1 performed0updates: source has2442/2338/1580/2257 valid support
vertex-frames in the four cases, so support labels are not absent. Existing
support gradients are weak on Pose compared with other losses, but often strong
on Root/decoder. Global geometry-weight amplification is not justified.

New pose_support_v1 from SAME accepted parent:8updates/224actor primitives+32AE,
same four pure PARC passages,seed,Adam5e-5,all-boundary+publishedRoot objectives.
Only additional normalized support gradients target Pose; Root/decoder still
train on original objectives. Position/gap scales.02m,velocity.1m/s reflect the
existing support criteria. Existing GLOBAL gradclip.5 retained: stronger Pose
gradients can indirectly alter other groups' clipping, not an identical-update
claim. Actual A100 first-step non-Pose extra-gradient difference0; unit synthetic
coupled graph tests passed. Snapshot code_pose_support_v1, unique script
pilot_pose_support_v1;PID2584624 only identity clue, GPU0 checked idle at launch.
Training AND final review finished, no active training/export remains. Both
root and Pose-support trials reached all16 goal planes, geometry screens13/16
and5/16 respectively. Neither adopted: held-up0.7 allsoles>5cm fractions are
97.92%(root) and85.42%(Pose), versus72.5% acceptedbaseline. Median lowest-sole
gaps17.43/16.20cm vs14.11cm. Pose held-up peakpenetration4.253cm,P993.371cm,
although endRootheight error only.236cm. Pose maximum replan deltaV1.941m/s is
lower than2.892baseline, but support/regression remain unacceptable. This is
evidence against treating the targeted gradient change as a fix.

support_candidates_export_v1 privately downloaded and138files hash-verified,
archive a2194c40a8cc91821a80c5489b9a495013fbedd29c52d794cac0f4cfcf053964.
Contains unchanged source, acceptedbaseline, root and Pose outputs with their
individual real lengths. Viewed zoomedframe25 on28936; saved
support_candidates_export_v1/approved-vs-pose-frame25.jpg. No model swapped.
Read-only screen check measured25.56% changed canvas pixels between initial and
actual W-frame24; nonblank official mesh and actual staircase visually checked.
Attempted mobile viewport override did not change the IAB's1280x720 viewport;
it was reset. Do NOT claim mobile verification from this attempt.
Two live trial pages are reset/paused for W startup; manual Pause is necessary,
not a learned STOP guarantee. New UI ready-state telemetry is refreshed after
reset completes. Final TCP check:28936/28937/28938 listening onlocalhost;
28934 was not listening, was NOT stopped/restarted by this work.

No extra full budget or mass generation yet. Next: resolve learned support and
requested-direction response with genuinely command-conditioned PARC examples,
retain the accepted model, and measure repeated real-time behavior; do not scale
the same straight-route targets or geometry weights blindly. Numeric geometry
screen counts are not natural motion/controller acceptance. Never duplicate the
completed pilots or replay an old full-budget command.

41focused local tests pass for metadata, live history/prefetch/WebSocket contract,
publishedRoot,boundaries and restricted Pose gradients. Both edited JS files
parse. Not a full-suite or motion-quality claim. No automation was recreated.

## Historical: Approval And Initial Continuity Review

2026-10-13 user APPROVES the orange feedback arm of pure_parc_export_v2,
generated_afe4ea1b1d92b874-up-speed0.7, with occasional sudden-motion/stutter
still requiring attention. Approved checkpoint is pure_parc_fit_v1/episode-weights.pt,
SHAae558ee88542191b6cbe72659bd96f1482711ea2d8beec6c1641a0ae5b814934.
This is now an accepted visual baseline; do not keep calling this exact candidate
rejected. Original leg_head_sequence_v2 also remains preserved. Historical
automated geometry/support metrics remain unchanged. User authorizes full review
from training to inference, fixes, then expanded pure PARC learning and actual
WASD controller. No STOP requirement has been reintroduced. No IK/output smoothing,
projection,Root replacement or future teacher-body conditions.

Read-only approved_continuity_v1: actual40frame accepted recording has average
Root velocity change1.711m/s at8frame replans vs.419inside (about4.1x), max2.892
atpublished frame8. Raw motion therefore has a genuine boundary issue. Separate
viewer behavior also holds the40frame arm until the89frame source ends, then
auto-wraps both. Viewer now ends playback without automatic wrap; no frames
changed or interpolated. Existing duration/end labels remain.

Review found whole-passage geometry boundary_mps supervises only the episode's
first transition after concatenation. Internal primitive starts were not included
in that specific loss (FK/support and factual-prefix losses still give other
gradients). New all_replan_boundaries uses SAME coefficient, actual published
adjacent frames and source velocity labels at every8frame boundary.2targeted tests
check internal/two-sided gradients and first-primitive/source equality.
Two matched8episode continuations are running: continuity_control_v1 and
continuity_all_boundaries_v1, each224actor primitives+32sourceAE, same new approved
parent,seed,cases,freshAdam5e-5,Root/Pose/decoder trainable,SDF/codec encoder and
vocabulary frozen. Snapshot code_continuity_review_v1, scriptpilot_replan_boundary.
GPUs0/2 freshly checked; PIDs2133975/2134557 only identity clues. Read true cmdline,
state/progress/complete before operations. No duplicate run or large training yet.

Live review found inherited leg_head_fit metadata takes precedence and rejects
the new86step checkpoint as not20steps. Local joint_checkpoint_progress now reads
the active pure_parc_pilot/stair_training_stage before inherited stages and uses
retained exposure counts. Tests confirm488actor examples/344sourceAE for86updates.
6initial targeted tests passed. Full local live/default inference parity and
30fps/browser-WASD validation still pending. Fresh-inference scheduling currently
waits until the fully played boundary, so latency must be addressed before claims
of smooth30fps. Preserve old28934/28936pages and all current model snapshots.

## Current Work: Pure PARC Continuous Stair Imitation Pilot

FINAL STATUS OF THIS ROUND: both bounded arms and causal audit COMPLETE; no active
training/diagnostic/export process after anchored Python-command check. Neither
candidate adopted; no mass generation/full budget/live WASD started. Latest user
goal stays natural PARC-like reciprocal UP/DOWN in the requested WASD direction,
actual terrain recognition, no penetration. STOP is outside this round.

pure_parc_joint_v2 matches first-arm baseline NPZs exactly (max0),80+6updates/
488actor primitives+344sourceAE,source/runtime mutation error0,ST1.752e-5,
frozen hashes/approved parent checks pass. Final16/16goal planes reached but0/16
geometry screens. Held UP peak4.08/6.47cm; held DOWN5.97/9.66cm; approved-UP
regression16.30/15.90cm. Co-adapting terrain is NOT a standalone cure. Window loss
first10->last10 means: fixedSDF3.63->0.68,joint3.62->0.73; batches vary, this is
only a training-loss trend, NOT motion quality evidence.

pure_parc_causal_v1 COMPLETE,0updates,3models x6factual source approaches.
Each runs actual terrain vs synthetic flat from the IDENTICAL4frames/request,
then independently generated history for the same declared primitive count;
also four requested travel directions,body facing held independently. FixedSDF
officialUP netRootY+82.97cm vs flat-1.66cm; officialDOWN-106.89 vs flat-1.98cm.
Joint heldDOWN-133.25 vs flat+1.44cm. Confirms terrain-dependent response, not a
memorized unconditional rise; does NOT prove anticipated proper footholds, and
includes the actor's factual ground-relative coordinate frame. Direction commands
change forecasts, but some reverse forecasts still travel forward within the
first0.93s; not a complete WASD/turn acceptance. No future teacher pose/height,
Root replacement,IK or motion modification used in the audit.

Post-export pure_parc_heldup_footfalls_v1 audits the clean0.69cm held ascent using
the unchanged source and actual40-frame candidate. Both0same-tread catch-up
events,but SOURCE5 vs CANDIDATE0 strict stationary sole episodes:>=3material
vertices within2cm,<=.1m/s,>=3frames. Kinematic support evidence,NOT force-contact
truth; unequal actual durations89/40 are preserved. Missing stable episodes
cannot be declared success just because no same-tread event was detected. This
reinforces that3/16geometry screens are NOT full natural-controller passes.

Private pure_parc_export_v2 uploaded/downloaded,212receipt hashes verified,
archive SHA25685ad0f18aab2490c3b55a040d9f84b7d64dc8e9327bb12cc81d9cb3a0e9c1fae.
Contains source/before/window/full-history/joint-terrain and SEPARATE flat clips,
each original30Hz length,no padding/retiming. V1 export failed on an incorrectly
specified stats path, preserved; V2 uses verified long_route_mixed_10k_ab_v1/
mixed_dataset/stats. No training was repeated for this export error. New local
early stats-existence guard prevents that failure before creating future exports.
8focused tests (selection/taller stairs,no source mutation/split leakage,existing
leg masks/history and WASD case contracts) pass; not a full-suite claim.
Official SOMA source vs fixedSDF candidate viewed UPframe25/DOWNframe32 on new
background tabs, saved up-source-vs-candidate-frame25.png and
down-source-vs-candidate-frame32.png in the export,marked deliverable. Old pages
untouched. Approved parent independently rehashed unchanged at final handoff.
Review28936/web/terrain_route_review/?data=/out/parc_stair64_20261013/pure_parc_export_v2/visuals/
with clips generated_afe4ea1b1d92b874-up-speed0.7 and
official_dc03d6484b176db0-down-speed1.0,left=source,right=feedback.
These orange bodies are NEW rejected/unapproved candidates,not the original
user-approved orange. Continue evidence-based learned foothold/support repair
and retain regression tests BEFORE expanding diverse PARC generation/training.

LATEST ACTUAL STATUS: pure_parc_fit_v1 COMPLETE,86actual updates/320source-window
actor primitives+168generated primitives+344sourceAE examples. Frozen hashes and
approved parent file pass. Actor future-target/runtime differences0; train-ST
difference1.502e-5. Baseline14/16goal planes reached and0/16geometry screens;
window-only16/16and0/16; complete-history16/16and3/16. These screens do NOT certify
reciprocal footfalls or live WASD. One training ascent peaks0.96/0.90cm, one pilot-
holdout ascent at0.7request peaks0.69cm; held descent4.38/4.85cm,approved-UP
regression13.66/13.37cm versus its preserved parent. Do NOT replace approved model
or expand data on these partial results. Median peak baseline4.39cm vs final4.41cm:
do not claim across-the-board penetration improvement from reaching all goals.

Completed controlled second arm pure_parc_joint_v2, SAME parent, six cases, seed,
80window+6episode updates and coefficients; additionally trains SDF encoder,
joint embeddings and Root/Pose/decoder SDF adapters along with native predictors.
This tested terrain/motion coadaptation, NOT a proven unique cause or weight tuning.
Codec encoder/vocabulary/command/old terrain modules still frozen. Archived first
arm baseline NPZ parity checked before any updates. New immutable code snapshot
code_pure_parc_v3, unique script pilot_pure_parc_stairs_joint_v2.py; GPU2 freshly
checked free at launch; process943077 ONLY identity clue. Inspect real cmdline,
state/progress/complete before operations. Do not duplicate or overwrite active
code. Once collected, compare both arms and export honest source/before/candidates
via private owned-OAuth Drive, preserving every actual duration and failure.

2026-10-13 latest user explicitly requests terrain-aware MotionBricks trained
only on PARC ascending/descending stair motion, WITHOUT stopping. Goal: causal
WASD travel direction plus actual terrain produces natural reciprocal steps,
not penetration. After a small version is verified, generate more diverse PARC
stairs/motions on A100 and screen before expansion. This supersedes the STOP
focus for this round, NOT the no-IK/projection/root-edit/fake-frame constraints.
The approved orange parent remains immutable:
ef806c919c29d73b5492667a100151449bf939d4366eeb3112ad1861de5fc9bf.

All STOP trials are COMPLETE, none adopted. stop_transition_v1:64 updates,
128actor primitives,0/2 screens; upSTOP tailP90 .3077m/s/bodymax10.70cm,
down .31857/5.32cm. stop_speed_review_v1:0updates,8explicit1/1.5m/s trials;
480/512-frame allowance completes all8 routes but STOP still0/8. Slow pace alone
is NOT failure. Private stop_study_export_v1 has57hash-verified local files.
It compares rejected STOP models, not the original approved orange gait.
No active STOP training, no new WASD deployment or automation exists.

Preparation pure_parc_data_v1 performed0updates and stopped with only2UP/3DOWN
eligible sources: its coarse selector excluded the taller accepted stair risers.
Preserve this diagnostic. V2 selects regular3+risers,5-34cm, with unchanged
full-skin<=2cm source gate. No source frames, terrain, pose or times are edited.
pure_parc_data_v2 COMPLETE, cases SHA256
43292058878925fb2de993ca9ca5ea9388832b943578f3122c732b61049486e9.
Train:official_6d8935e4152e794f-up81f,official_dc03d6484b176db0-down81f,
generated_b8b67bec426d92e2-up65f,generated_1a9586e26463b55f-down97f.
Pilot holdouts:generated_afe4ea1b1d92b874-up89f,
generated_eebe1044e47fe89f-down97f. Source peaks0.75-1.63cm,18,056vertices/
8substeps,0unknown/uncertain. Holdout IDs are separate for THIS fitting, not
claimed unseen by inherited mixed-data training. Controls are weak fixed approach
velocity/heading, not recorded WASD. All new optimizer samples are PARC; existing
pretrained weight provenance is NOT erased or misrepresented as PARC-only.

New bounded runner scripts.pilot_pure_parc_stairs:80source-window updates,batch4
(320actor primitives),then6whole-passage updates with4source passages accumulated
per update (168generated primitives),plus344sourceAE examples. Fresh Adam5e-5,
original coefficients; Root/Pose/codec.decoder train jointly, encoder/vocabulary,
terrain adapters and command module freeze. True initial4 only in feedback phase;
actual generated histories retain temporal gradients without reset. Fixed-time
whole-passage imitation is a bounded probe, not proof of valid off-source recovery.
Runtime structure/defaults unchanged; no STOP branch installed. Six source cases
plus original approved up/down regression cases: held cruise at1/.7request scale,
max480frames with each actual duration retained, complete route/height/lateral/
skin/gait measurements; same-body same-command flat terrain counterfactual is
diagnostic, NOT alone a causal quality pass.7local focused tests pass.

Snapshot:parc_stair64_20261013/code_pure_parc_v2; new filenames staged only,
existing hardlinked snapshot files never overwritten. Planned output:
parc_stair64_20261013/pure_parc_fit_v1. Inspect actual state/progress/complete and
cmdlines before any action, never repeat an active run or restart old full budgets.
After complete, compare baseline/window/full-history stages and publish official
SOMA visuals through user-owned private Drive. No full-scale expansion until real
continuous up/down and terrain/direction tests show reliable improvement. Preserve
old28936pages and accepted checkpoint; no runtime corrections or failed deployment.

## Completed Paper/Source Control Review And Continuous Restart Pilot

2026-10-13 latest user: consider multiple causes, solve from papers and source.
No old full-budget campaign or64-generation cohort was restarted. The user-
approved orange checkpoint stays immutable. All new changes are learned command-
conditioned network branches, not IK, foot locks, output smoothing, or Root edits.

Primary review: MotionBricks https://arxiv.org/html/2604.24833v1 sections4.2/6.1,
official full_agent.py in NVlabs/GR00T-WholeBodyControl; CAMDM
https://arxiv.org/html/2404.15121v1 section3.2 and official AIGAnimation/CAMDM;
PFNN author's https://theorangeduck.com/media/uploads/other_stuff/phasefunction.pdf;
PARC https://arxiv.org/html/2505.04002v1 sections5-6 and official mshoe/PARC.
MotionBricks deliberately lets its decoder refine Root, so different stage
velocities do NOT prove a broken interface. Its smart primitive chooses pose
style/keyframes as well as Root intent; source full_agent uses idle mode0.
Our soft controller removes future pose keyframes, supplying commands mainly to
Root. PFNN uses gait labels and phase; CAMDM separates control conditions. These
support testing explicit learned stop style, not copying their runtime heuristics
or adding authored future body poses to this actor. PARC physics tracking a moving
reference does not by itself establish a desired-stop policy.

`stop_decoder_factors_v1` COMPLETE,0updates,6 factual moving/STOP states x10
offline interventions,60 records. Real decoder VX/VZ fields agree with Root
finite differences <=3e-4m/s; future body-condition mask false. Baseline8 published
frames replay their recording. Merely zeroing future external velocity OR merely
substituting genuine BONES rest tokens is insufficient. Together,prototype0
reduces forecast-tail mean speed to.021up/.031down m/s, but reconstructed past
XZ error~5-7cm and some up-skin intersections remain. Reference tokens are NOT
the true future of these stair states and remain strictly offline diagnostic.
No oracle became a controller or a passing complete sequence.

`stop_decoder_factors_v2` COMPLETE,0updates,30 records. Compare decoder INPUT
past velocity/height with actual observed intervals0..2/heights0..3, never future
truth. Input past velocity max error .324up/.428down m/s. Supplying factual past
conditions reduces max reconstructed historyXZ error5.29->4.27cm up and6.42->
5.06cm down, barely changes forecast-tail speed. It is NOT an adopted repair.
Root and Pose/decoder issues are coupled; neither one-factor fix is sufficient.
All raw records retained; small JSON reports are local, animations stay private.

`stop_stage_pose_v1` and `stop_stage_decoder_v1` COMPLETE,50updates/batch4/
200primitives EACH,from matched Root-only stop50 checkpoint. Only one new
STOP-gated query/control MLP is trained; prior Root-only branch,all original
weights,encoder/vocabulary and SDF modules remain frozen. Inputs are desired
VX/VZ,heading,horizon and current layer queries,NOT stale release-position delta.
Same8 true braking/rest windows +factual decoder-prefix loss, original native
coefficients. Training corruption RNG2333 after baseline; constructor hidden
weights were not seeded explicitly in v1, so do NOT claim a seed-controlled
exclusive architectural causal comparison. Both baselines match all four
archived recordings exactly. Future-target mutation error0, fixed hashes pass.
Both FAIL0/2 stairs: Pose tailP90=.641up/.557down,body maxima12.44/5.06cm;
decoder=.661/.428,body maxima13.10/3.91cm. Pose native loss3.41->~1.5 does NOT
prove motion improvement. All MOVE72/144frames remain exactly unchanged.

`stop_feedback_v1` COMPLETE:40actual updates/80actor primitives (one continuous
generated stair primitive +one real-source primitive/update),four genuine prior
MOVE initial states across up/down and2/3 initial MOVE blocks. Truncated feedback
uses actual generated/re-encoded histories, no source-time future pose/Root
targets. Only new Root/Pose/decoder stop modules learn. Futures are desired
velocity0,actual terrain collision and delayed standing support/foot velocity,
with a1s braking grace. Original native Root scaling/geometric coefficients and
source prior retained. No failures or unknown-point deletion.0/2 stop screens;
tailP90=.640up/.368down,body maxima13.14/4.46cm;MOVE exactly unchanged.

`stop_feedback_v2` COMPLETE with SAME initial model/seed2333,episode order/budget
40updates/80primitives,not bitwise identical on-policy histories after updates.
Quality-scaled task: speed.15m/s,support gap.02m,planted foot velocity.1m/s,
penetration scale.02m,collision top1% rather than diluting a local defect.
Scales come from existing control/support screens; NO quality gate changed.
Published velocity includes the observed-history->first-new-frame boundary,
not just decoded within-block velocity. Source/native objectives unchanged.
Gradient audit on SAME state proves original task signal can be much weaker than
source CE: first state native-task Pose norm.00676 versus prior6.4795; scaled task
norm.61416. Root norms.63296/58.3444 versus prior1.855,decoder.30474/27.5833 vs
prior.02283. This is a scale/conditioning diagnosis, not evidence of quality.
Final0/2:up tailP90 .27094,half-second drift4.80cm,stop outside flight,resume only
9.11cm,body max8.66cm. Down .15976,drift1.65cm,stop inside flight,resume23.99cm,
body max4.32cm,and route not finished after actual192frames. NO near-pass rounding
or gate relaxation. Moving72/144frames exactly unchanged. Not adopted.
Candidate SHA2317ca38ab54f2cb3c6483aa6aead0ee7225cec26622d48490f2f2ae04d059c6.

Current new bounded `stop_transition_v1`: permits learned modules during genuine
restart events,then returns to unchanged MOVE branch after>=2plans and observed
speed>=.3m/s. This is command-history gating,NOT a motion speed cap. Continuous
MOVE with no precedingSTOP does not activate restart even during slow steps.
Cold-start at observed speed<=.15 also activates learned startup. Real restart
windows from the SAME4 BONES sources are screened before admission to this pilot.
Four actual continuously generated episodes each have10STOP+6resume primitives;
max64updates/128primitives including one real prior/update. Source-time future
body/Root supervision remains restricted to real source histories. Actual stair
feedback uses velocity/support/collision goals only. Resume target is minimum
meaningful forward progress,not forcing3m/s up stairs. No Source-body transplant.

Code snapshot:parc_stair64_20261013/code_stop_transition_v1.
Command:scripts.pilot_stop_transition;run:parc_stair64_20261013/stop_transition_v1.
Launcher receiptstop-transition-v1-process.json;PID772819 only an identity clue,
inspect actual cmdline/progress/complete before any operation. GPU0 freshly checked
idle at launch;GPU1 belongs to others. Do NOT duplicate or modify running snapshot.
This pilot is NOT yet claimed complete or usable. First collect its exact results,
frozen hashes/continuous-MOVE parity and actual up/downstop/resume quality. Preserve
failures/actual lengths. If viable, export private official SOMA and broader start/
stop/heading cases before local WASD/30fps. No new live server deployed yet.

New tests:STOP-stage exact MOVE preservation/no stale-goal dependence,actual
publication-boundary velocity gradient,no velocity cap,causal restart gating.
Privacy remains private owned-OAuth Drive/NAS/internal for all bodies/models.
Old28936/approved orange pages untouched. No automation created.

## Completed Stop Supervision Audit And Root-Only Pilot

2026-10-13 latest user asks how to solve the mid-stair stop/resume failure.
The original64 cohort is finished and still0/64 admissible. No old campaign was
restarted, no larger dataset admitted, no new WASD server or automation created.
Approved orange weights remain the baseline. The work below is bounded research,
not a working controller delivery or permission to lower the stop/support gates.

Read-only `parc_stair64_20261013/stop_supervision_audit_v2` COMPLETE. Actual pinned
expert cfg (not the local default): XY_DIR, displacement threshold.1m,
use_target_loss=true, w_target=.02, w_vel_root_pos=.5,30Hz,.5s,2 history frames.
Its optional direction target loss is a negative target-direction/displacement
dot product: zero target gives zero loss AND zero gradient. Reconstruction and
velocity losses still exist; this is not proof of absent velocity learning or a
unique cause. Source direction labels are inferred from future motion, not
recorded requests to start braking. The original expert's entire training corpus
has NOT been audited; do not generalize our paired-source inventory to it.

All available unchanged paired SOMA sources were checked,0 read errors. PARC
526train/51validation/49test (626total) have NO root-quiet run>=24frames at
<=.12m/s even WITHOUT the contact requirement. BONES train2003clips contain415
moving-to-rest runs,90 with later restart; validation has9 and1 respectively.
Quiet supported runs additionally require one active channel on each foot side.
These are real-motion weak command candidates, not recorded keys or certified
mid-stair stops. Four DISTINCT BONES sources yielded8 genuine33-frame braking/
rest windows with exact18,056-vertex/8substep prism max.36-.56cm,0 unknown or
uncertain samples. Their source bodies/Root were not translated or altered.

Matched native history-guidance endpoints COMPLETE: layouts3/6,up/down,
cfg.65 vs1.0,8 continuations total. All0/8 stop inside the flight and resume.
With cfg1, stopping still takes5.13-6.07s and3.53-4.08m; boundary maximum joint
angle grows to63-81degrees versus default29-58degrees. Not adopted. v1 retained
four completed layout3 rows but stopped at a quaternion sign assertion on layout6;
q and-q are physically identical. v2 compares signs ONLY in the assertion,
does NOT alter output rotations, and completes the remaining layout6 cases.
Physical prefix error<=1.20e-7; both logs/raw recordings retained. This is not a
relaxed motion-quality threshold. Reports are local stop_history_guidance_v1.json
and stop_history_guidance_v2.json; v1 is partial, not a falsely completed run.

`real_stop_control_supervision_v1` is a0-update feasibility/gradient audit,4
windows from ONE real BONES sequence, not4 independent demonstrations. Existing
command adapter gradients are finite/nonzero, actor future-target mutation error0,
training forward error0, frozen gradients empty, weights unchanged. These checks
do not establish stair quality.

New `stop_command_pilot_v1` COMPLETE:50 actual Adam updates,batch4,200 source
primitives (6.25batch32-equivalent),fresh Adam5e-5,clip.5,existing native motion/
Root/token losses unchanged. Only a new STOP-gated residual in the seven-feature
COMMAND EMBEDDING is trainable; all approved weights/codec vocabulary are frozen.
It is not an output Root correction. MOVE outputs are verified EXACTLY unchanged
for the72-frame ascent and144-frame real descent. Future target mutation and
training-forward errors0; all fixed tensors remain unchanged. Four screened
sources train braking AND genuine quiet continuation, not copied standing frames.
Source history only in this pilot; no generated-history quality was claimed.

Actual continuous2.667s release/resume reviews still FAIL BOTH stairs. Up tail
speedP90 .9169->.6256m/s,drift28.2->25.8cm,body max12.38->12.21cm,stop outside
flight. Down tailP90 .51394->.51248m/s,drift18.7->21.0cm,body max3.64->6.81cm.
Resume reaches endpoints but that does NOT make the stop acceptable. Do NOT adopt.
Parent SHAef806c919c29d73b5492667a100151449bf939d4366eeb3112ad1861de5fc9bf.
Rejected candidate SHA8bd03d3d069c3b160f22f652fc7dc51a2bda1a73e86c8a10f692cc346fcb9571.
Immutable training snapshot:parc_stair64_20261013/code_stop_command_v1.

`stop_stage_audit_v1` COMPLETE:40 factual STOP-history replays,ALL output error0.
Root and decoded outputs are distinct. Last3 stop blocks of the new candidate:
up planned mean.1818m/s vs decoded.4509;down.2314 vs.4434. These within-block
means exclude cross-block speed impulses and differ from tailP90. They are NOT
an additive error decomposition and do NOT mean Root has fully stopped. Code
review confirms explicit command embedding enters Root Transformers; Pose and
decoder receive command effects indirectly through predictions/observations,
not a direct desired-speed/STOP token. This is a concrete next conditioning
surface to test, NOT proof that one hidden layer is the only cause.

`stop_goal_probe_v2` COMPLETE,0 updates: STOP asks for the current observed XZ
instead of returning to the original release point; speed still0, no trajectory
overwrite. Original/candidate up/down all0/4 stop screens. TailP90 remains
.9655-1.0675m/s,although some collision depths decrease. Not adopted. v1 failed
on a helper variable-name collision before recording; v3 helper fixes that,
without editing immutable training code. All four MOVE recordings still match
the archived baseline exactly. A goal-semantics mismatch is therefore NOT a
sufficient explanation or a verified repair.

Before/after official SOMA export COMPLETE,private Drive downloaded and22 receipt
hashes verified:out/parc_stair64_20261013/stop_command_export_v1. Archive7,436,181
bytes SHA595321ac24c6d223a59c1ae220d827078f62a9eb36c5062b77363b7544224a3d.
No padding/retiming; before/after happen to share actual durations. Rejected trial
label is explicit.28936 background review was visually checked at up STOP frame95,
saved up-stop-frame95.png and marked deliverable; user's old pages are untouched.
URL:28936/web/terrain_route_review/?data=/out/parc_stair64_20261013/stop_command_export_v1/visuals/&clip=up-release_resume&left=before&right=candidate.
10 focused existing/new tests pass (course/WASD source cases,STOP gating/frozen
gradients,quaternion comparison with no input mutation). Local archive helper's
Python-version hash API failure was repaired; full private receipt check passed.

Next controlled work: STOP intent must reach Pose/decoder as well as Root, with
factual-history preservation and generated-feedback braking/settling objectives.
Keep MOVE branch immutable and compare stage-specific small adapters, not blind
unfreezing or loss-weight escalation. Genuine source rest preserves motion prior;
it is NOT automatically a stair-standing teacher. If teacher data is needed,
first validate a few native-physics brake/stand/restart episodes from EXACT native
moving states with explicit desired-stop objectives and real support, rather than
assuming a tracker following a failed moving reference will learn to stop. No
SOMA->PARC inverse shortcut, fake static references, transplanted source poses,
Root replacement,IK,foot locking,output smoothing,or future true body/height input.
Expand64 data/student training only after these control/support checks succeed.
Do not restart original full budgets or adopt any rejected configuration above.
Final anchored Python-command lookup found no active training/diagnostic process.
Approved parent file was independently rehashed unchanged. No PID-only operation.

Primary research consulted: https://arxiv.org/html/2505.04002v1 sections5-6 and
https://github.com/mshoe/PARC. PARC's physically tracked recordings are an OFFLINE
data-generation loop, not justification for prohibited runtime correction.

## Completed 64-Generation And Stop Condition Screen

The authorized64-candidate cohort is COMPLETE, not running.8 independent
layouts x8 different initialized positions. Each has causal MOVE/STOP/resume
commands in one generated passage, both up and down, without history resets.
42/64 satisfy ordered geometric route coverage. All128 requested stop intervals
were retained;0 individual intervals pass the combined mid-flight stop+resume
screen, and0/64 full sequences pass.9 intervals have a stationary tail, but not
the full requested mid-stair behavior. This is a NATIVE COMMAND-RESPONSE screen,
not full physics/SOMA clearance. No candidate was admitted, no larger training
started, and costly physics/SOMA stages were not run on these control failures.
Do not call42 routes42 clean/admissible training samples.

Important correction to prior interface discussion: PARC training DOES encode
zero direction for short-horizon near-stationary motion (`mdm.py`
get_dir_from_canonicalized_pos_and_rot). A unit-direction interface has no metric
speed control, but "no stop input exists" would be wrong. Zero direction was
issued before diffusion, and the target token was captured as zero with its
mask enabled. The issue is unreliable braking/settling from moving stair states,
not a missing key binding or absent target. We have not proved a unique training
root cause or that no PARC configuration/model could learn this.

Read-only matched `stop_condition_v1` COMPLETE: factual pre-stop native histories
from layouts0/6, same seed, move vs stop vs stop without CFG,4 candidates per call.
For layout6 input2.246m/s,13 future frames end at1.28-1.68m/s with MOVE versus
.978-1.203m/s with STOP: input has an effect but is not a reliable complete stop.
Disabling CFG did not improve this case (1.151-1.219m/s). No weights changed.
`held_stop_v2` COMPLETE: keep STOP until measured settling or6.067s, then resume,
without editing poses. Layout3 moves3.347m, remains .152m/s P90 after6.067s and
leaves the flight. Layout6 settles after4.2s but moves3.577m and also leaves the
flight. Neither solves the requested mid-flight stop. These are two conditional
continuation diagnostics, not additional claimed full64-dataset successes.

All generation/diagnostic processes finished, checked by actual process lookup.
Approved orange checkpoint rehashed unchanged:
`ef806c919c29d73b5492667a100151449bf939d4366eeb3112ad1861de5fc9bf`.
No new WASD server or model deployment. No automation created/recreated.
Next work must establish reliable braking/standing/restart teacher data or
control supervision before expanding student training; do NOT teach the failed
zero-input movement as successful stopping, loosen gates, hold output frames,
or introduce IK/Root replacements. Existing approved ascent remains the baseline.

All64 raw candidates, commands, layouts, logs and immutable generation code are
privately backed up: `private-store:motionbricks/parc_stair64_20261013/`.
Archive `stair64_candidates_v1.tar.gz`,15274489bytes, SHA256
`cd7e604a94561942eba1b9785b12ef0aad68b7841d5c382f3032d31733c63ffe`.
Downloaded through existing7897 proxy with owned OAuth, hash verified, safely
extracted to local `out/parc_stair64_20261013/raw_v1`; all64 raw pickle files
counted. Local screening_v1.json/png/pdf retain every stop event; diagnostic
reports stop_condition_v1.json and held_stop_v2.json are local. Raw diagnostic
animations stay remote unless transferred through private storage.

## Completed 64-Candidate Generation Contract

2026-10-13 user explicitly authorized generating64 up/down stair candidates,
screening them, then expanding training only after passing, preserving the
approved orange model and ultimately delivering actual WASD stairs. Latest
refinement REQUIRES stops/resumes MID-FLIGHT in the SAME continuous sequence,
with different initial positions. Not separate pasted stop/walk clips.

New cohort root on A100:
`model_terrain_v3_20261011/parc_stair64_20261013/batch_v1`.
Immutable staged code:`parc_stair64_20261013/code_v1`. Supervisor
`run_parc_stair64_batch.py`, launch PID247524 only an identity hint; inspect actual
cmdline before action. It first verifies layout0, then schedules remaining7
layouts on freshly checked physical GPUs2/3/4/0, never GPU1 or occupied cards.
Eight distinct regular-riser layouts times8 candidates =64 raw candidates,
NOT a claim of64 admissible training sequences. Layouts0..5 train,6 validation,
7 test, with distinct geometry hashes. Each route has approach/ascent/platform/
descent/exit. Eight initial flat-ground offsets are applied BEFORE first PARC
generation, not to an existing generated body. Mid-ascent and mid-descent issue
exact-zero direction requests for1.6/2.0/2.4/2.8s, then resume ordinary path input.
No history resets, copied still frames, reversed motion, or runtime correction.

PARC expert pinned EMA checkpoint a03a9d4cc7b3a43b217d67116429d241c92e31e7de93e02849832b0371860e2e,
official DATA sampling stride5/cfg.65 (not the faster evaluation stride10).
Isolated predict_x0 wrapper restores the two previous-state flags after every
call, with every conditioned native2-frame prefix checked<=3e-5; vendor unchanged.
Capture commands AFTER causal command scheduling and BEFORE diffusion; zeros
mean requested stop, unit vectors mean move direction, not desired metric speed.
Native pre-screen measures final0.5s speedP90<=.15m/s, drift<=.08m, stays inside
the requested flight, resumes>=.25m, and requires both stops. This is NOT SOMA
contact/physics approval. Twelve local course/flag/schedule/provenance tests pass.
First live log progressed through7.77s of continuous generation; no completion
or stop success claimed yet. Do not duplicate or overwrite running code/results.

Next: complete generation inventory, inspect stop responses, then official
physical recording and unchanged SOMA conversion/screens for eligible candidates.
Local physics helper now supports explicit short-course min-duration and full-
route/stop checks, defaults unchanged; it is NOT yet staged/running for this cohort.
Local admission helper has optional terrain-family stairs instead of mislabeled
mixed. New data must also pass full-skin exact-prism/contact/continuity and
reciprocal moving-footfall checks. Same-tread settling WHILE STOPPING can be
legitimate; do not reject standing with both feet on a tread as moving step-to,
nor let that exception hide repeated same-tread stepping under MOVE input.
All raw failures and actual lengths stay. No controller training started yet.
No recurring automation recreated. Orange checkpoint/local pages untouched.

## Completed Stair WASD Probe And PARC Interface Audit

2026-10-13 user scope: first make normal UP/DOWN stairs work under their WASD,
not a whole general-purpose controller. Latest user asks whether to generate lots
of PARC stair data before training. Generator interfaces were inspected, but NO
new mass-generation/training campaign has started. Existing PARC XY_DIR converts
target displacement to a unit direction (`gen_util.py`94..97); path-end truncation
is not a stop example, and target distance is not a requested speed channel.
The generic diverse "stairs" builder can create1.2-4m broad platforms, so use
actual regular risers/treads for this user's staircase task, not family names
as proof of coverage. Existing generation->official tracker->SOMA screens are
available, but all generated motions remain candidates until screened. Earlier
native prefix-flag side-effect audit must be respected when preparing new runs.

Accepted model was privately uploaded and downloaded byte-for-byte, SHA256
`ef806c919c29d73b5492667a100151449bf939d4366eeb3112ad1861de5fc9bf`.
Local artifact `out/stair_wasd_20261013/approved_v1` contains weights, checked
export.json, up/down scene+exact4 observed frames, and SEPARATE offline-cases.pt.
Live scene files have no future motions. Bundle remains a100_live_20261010/
a100-terrain.pt SHA2560672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d.
Private destination `private-store:motionbricks/stair_wasd_20261013/approved_v1/`.
Remote preparer `task/prepare_stair_wasd_trial_v1.py` COMPLETE; no training run.
Source dataset native_manifest hashc250f57d... matches bundle, not relabeled.
Up source8030 at214, exact skin max1.033cm. Real DOWN source054c2050e9d10bb0
at490, regular descending flight with source max1.9965cm, all points known;
not reversed ascent or stitched motion. Two source-selection tests pass.

Local RTX3080 eager REAL online-backend probes completed in `local_probe_v1`
and `local_probe_v2` under the same stair_wasd_20261013 root. No actual browser
WASD yet and no new server port started; only existing28936 review is serving.
First-block default controller vs online backend error0 for up/down. Keyboard-
equivalent3m/s input reaches up72f with max3.680cm, down144f with max3.391cm.
Down endpoint Root height differs18.2cm from reference, so low collision alone
does NOT prove natural descent; visual/footfall assessment still required.
Median generation~240ms per8 frames, P95~250-259ms mostly, with a334ms outlier.
This is NOT a30fps browser result; current fresh-inference-at-played-boundary
scheduler can add stalls and must be measured, not hidden by padding/smoothing.

Stop test4 blocks first showed deceleration, so travel during braking alone was
not called failure. Extended stop10 blocks (2.667s) still leaves mean final-block
speed0.530m/s up and0.339m/s down. Up stop/resume max12.82cm; initial zero input
then start also drifts and max10.36cm. These are genuine behavior limitations,
not merely key binding. All generated frames/failures retained. Original
approved upward gait is untouched; no runtime motion corrections or new weights.
Scripts: prepare_stair_wasd_trial.py and probe_stair_wasd_live.py. Online progress
metadata now counts leg adaptation20 episode updates/860 examples rather than
mistakenly inheriting batch384; six live/control tests pass. No model forward
or control semantics changed by that metadata patch.

Next decision: a bounded32-64-candidate regular-stair generation/quality pilot,
then expand only useful admitted coverage. Need up/down, approach/departure,
different geometry and genuine control transitions. Do not random-relabel
straight motion as stop/turn, assume the PARC->SOMA bridge supplies identical
generated-state recovery, or launch another large budget before data validation.
No automation exists/recreated. Other user's GPU1 left untouched.

## User-Approved Orange Stair Gait

LATEST USER DECISION (2026-10-12): the user explicitly APPROVED the orange right
arm on `leg_head_comparison_v2`, clip `reported-stairs`, arm `sequence`.
Preserve this as the accepted stair-gait baseline. Do not keep describing this
specific visual as rejected by the user, or silently substitute the earlier
Root-only/100-step/source-prefix candidates. This acceptance is of the rendered
stair gait; it does not claim other routes or live WASD/30fps were tested/passed.
The user considers3.68cm potentially acceptable visually. Keep measured values
and original automated results; do not invent a new numeric threshold or rewrite
the historical gates. No IK, projection, foot locking, timewarp or Root override.

Accepted checkpoint: `joint_sdf_20261012/leg_head_sequence_v2/weights.pt`, SHA256
`ef806c919c29d73b5492667a100151449bf939d4366eeb3112ad1861de5fc9bf`.
Main actual72-frame recording SHA256
`528aa994d34701c935ac9e395467d2f6507d30f18d916d8ae658baee926444c7`.
20 episode updates/860 actor primitives+80 sourceAE, from Root-head candidate;
48 leg rotation output rows only. All other head rows exact, every non-head state
hash unchanged. Baseline all5 recordings max difference0, future/runtime/training
forward differences0, late-to-first gradient.22996. Nine local focused tests pass.
Main landing sequence R0 -> L.2869 -> R.5737 -> Ltop, then Rtop, preserved.
Strict stationary runs1->4, including R frames33..37 ON stairs, not only top.
Main soleP99 2.909->2.291cm, max3.6047->3.6786cm. Three full-route peaks
5.513/9.950/3.679cm; all complete routes, automated varied1/17 (idle only), walk
still fails. These measured limitations remain open separately from visual approval.

Both small probes are COMPLETE; no active training/diagnostic jobs remain.
No large campaign or automation restarted. Latest43-file private export downloaded
and hashes verified: `leg_head_comparison_v2`,7663562bytes, SHA256
`18bb4eb9f95efe3138ee17de4148eae4b5fbd06fdd548826cca613b0e74f764f`.
Official SOMA viewed frames34 and24 (transparent ground), screenshots in export.
Viewer labels use a separate `visuals/review-status.json` acceptance sidecar;
original receipt-listed manifest/rigs/metrics/recordings remain byte-unchanged.
28936 serves the visible before/after; no weights loaded into live WASD yet.
One SSH connection closed after footfall export; process status was checked,
completed footfalls retained and only remaining peak audit/private packaging
continued. No training duplicated. Earlier failed V1 snapshot/output details below.

## Completed Leg-Output Probes

LATEST: `leg_head_v1`100-update source-window fit is COMPLETE and rejected.
Three actual route peaks4.97/10.12/4.60cm (before8.88/10.77/3.60cm), all fail.
Main strict stationary runs1->3 occur on TOP landing, not restored stair support.
Main near-surface events still L.2869->R.5737->Ltop, BUT the prior R level0 event
is absent. Thus do not claim preserved COMPLETE landings from a no-repeat count.
Exact peak locator puts main worst depths at right-foot frame17 and left-foot23.
Worst vertex absent from sparse set, but sparse set still sees4.497cm versus full
4.603cm; sparse sampling does NOT explain away this failure or all gradients.

Now bounded full-passage leg-only probe `leg_head_sequence_v2`,20 episode
optimizer updates,860 generated primitives+80 sourceAE uses. SAME initial
Root-head checkpoint (not the failed100-step leg fit), same48 trainable rows,
same coefficients. Each update accumulates3 complete10-primitive stair routes
and one13-primitive idle/walk route. Actual generated histories retain temporal
gradients without resets; source first4 only at start, source future only labels.
No compute-equivalence claim against100 source-window updates/400 primitives.
Runner `task/pilot_joint_sdf_leg_sequence_v2.py`, new immutable code snapshot
`code_joint_sdf_leg_sequence_v2` derives from verified sequence-credit snapshot
plus source-sole helper. Six local row/history-gradient tests passed. All5
baseline recordings must match archived Root-head results before updates;
actual future/runtime/temporal-path checks and immutable hashes are required.
GPU0 freshly rechecked; launch PID4151238 is only an identity clue.

`leg_head_sequence_v1` FAILED before any optimizer update: the older
`code_joint_sdf_support_xyz_v1` does not implement history_gradients. Do not
assume snapshot chronology implies that support. Failure log retained. Its
mistyped output path is `${PRIVATE_STORAGE}/motionbricks-20261011/joint_sdf_20261012/leg_head_sequence_v1`,
containing diagnostic baseline/state, not trained weights; no user data removed.
V2 correct output is under the usual task/joint_sdf_20261012 root. No healthy
job restarted, old snapshots not modified, no automation recreated.

User resumed with "continue fixing" after the earlier interruption. Fresh process
check found no active Joint SDF jobs. `render_head_fit_v1` is COMPLETE, not running.
Root-only head fitting restored the main route's actual alternating landing
sequence: R level0 at18..20, L.286867 at22..26, R.573734 at33..39, then L/R top
landing. No interior same-tread catch-up; only1 strict stationary run, so support
is still insufficient. Three-route exact skin peaks8.88/10.77/3.60cm fail the2cm
gate. Root+rotation fitting peaks2.83/6.10/3.16cm but loses middle landing evidence
on the main route, and is NOT preferred just for lower penetration. Both fail
flat walking. Neither is adopted. Root checkpoint SHA256
`a2b168d5d7428c25ae836ebb1bb7d04d9ff42182ee9e270801581bb15fdf68d8`.

The28-file `render_head_comparison_v1` export is privately downloaded and fully
hash verified; archive6315534bytes, SHA256
`630d473e6fc65f5009b1735e396fbb6f77c2e36199ab994e33d20de77ae23d25`.
Static28936 had no listener and was restored, not repurposed. Background page
`web/terrain_route_review/?data=/out/model_terrain_v3_20261011/joint_sdf_20261012/render_head_comparison_v1/visuals/&clip=reported-stairs&left=previous&right=root`
was inspected at frames26/37. Original motion clocks and distinct durations
remain. Old model/pages are not replaced.

New bounded probe `leg_head_v1`:100 actual updates, batch4,400 actor windows plus
400 sourceAE uses, from Root-only candidate. Only48 final Conv1d rotation output
rows for bilateral Leg/Shin/Foot/ToeBase train. Root output rows, upper body,
entire decoder trunk, Root/Pose/encoder/vocabulary/SDF all remain immutable.
Fresh Adam5e-5, zero decay, row gradient mask; every step checks frozen rows
exactly and8/final steps hash every non-head state. Root WEIGHTS are fixed, not
Root TRAJECTORY: changed rotations may change subsequent history/queries.
No inference override or postprocessing. Native/sourceAE/factual/physical losses
retain existing coefficients, source-supported XYZ at original FK coefficient;
real source contact/velocity, not weak frozen-teacher phase masks.
Five local tests pass for exact row isolation and factual source sole objective.
Actual A100 future-label/runtime/gradient checks precede final evaluation.

Versioned runner outside immutable snapshot: `task/pilot_joint_sdf_leg_head_v1.py`,
using `code_joint_sdf_support_xyz_v1`; state inside `leg_head_v1/state.json`.
Fresh GPU0 launcher PID4103328 only an identity clue; inspect cmdline before any
action. GPU1 belongs to another user and was not touched. Normal final/varied/
15s route reviews run automatically; then audit actual landing order/planting.
Do not adopt from training loss or missing footfall events. No full budget,
automation or new runtime model started.

## Completed Decoder Output-Head Precision Fit

All SGD probes below are COMPLETE and failed the full motion gates. No failed
candidate is adopted. Most important new evidence: `source_fit_logits_v2`
shows100% ACTOR token accuracy on all30 factual-prefix stair windows after the
1000-step source fit (actor CE~.005), yet Root mean error5.5-5.9cm and foot support
position errors5-6cm remain. Do NOT blame incorrect token selection in that
conditional test. Free-history token accuracy is lower and this is not a global
claim about Pose. Raw source RIC versus displayed FK was also checked: mean
~1e-6 to3.8e-6m, max1.30e-5m, so inconsistent source geometry is NOT the cause.

`trained_codec_oracles_v2` completed offline only. With factual source history,
source tokens equal actor tokens exactly. Replacing predicted Root conditioning
with true source Root only reduces mean Root RMS5.892->5.478cm; relative-joint
error1.980->1.941cm. Thus a major residual is decoded Root reconstruction even
with correct tokens and Root conditions. Generated-history Root RMS19.525cm
(contains time-alignment differences), all-source oracle17.429cm; not a runtime
repair. V1 failed on a missing diagnostic-module import before any inference;
V2 loads a versioned copied reference script outside immutable snapshots.

Now running a bounded output-head fit, no more large/SGD budget:
`joint_sdf_20261012/render_head_fit_v1`, supervisor
`task/run_joint_sdf_render_head_v1.py`, state`joint-sdf-render-head-fit-state.json`.
Fresh GPU7 launch PID4009304 is only an identity clue; check real cmdline/state.
Fitter`task/fit_joint_sdf_render_head_v1.py` uses `code_joint_sdf_source_continue_v2`
without modifying that snapshot. Parent is source1000 checkpoint SHA256
`7d97f19fbc609e3676e97afeb950d8c335ce3d71f8fc1a3891d9a1ab479b6306`.
Only existing last Conv1d output rows change: arm root fits height/XZ velocity
(3 rows), arm rendered also fits180 global-rotation rows. All other head rows
and every non-head state tensor are hash/equality checked. No output Root/pose
replacement, IK, lookup trajectory, timewarp or inference postprocessing.

Features are captured from the ACTUAL causal actor with source first4 observed
frames, ordinary command and scene, using its predicted Root/tokens. Future
source motion is used only as supervised labels after the actor call. Last-layer
convolution is reconstructed as its exact linear design and checked<=1e-4.
Ridge fits a delta around the old weights with multipliers1e-4/1e-3/1e-2/1e-1,
selecting normalized validation MSE on block%3==1 held-out prefix WINDOWS.
These are overlapping motions already seen by the parent, not unseen-data
generalization. No optimization-step count is invented for the linear solve.
Two tests pass for exact Conv1d design and held-out fitting. The head changes
the model's proposal queries too, so static-feature fit alone is NOT a quality
claim: final, varied, conditional-fit and actual15s route reviews run afterward.

Completed intervening probes (not adopted): source200 decoder/coordinated pair;
20-episode detached/through-history pair (860 primitives+80 AE uses per arm);
source continuation200->1000 preserving Adam/RNG/cursors; then matched source
contact control/XYZ1000->1200,200 updates each. XYZ keeps the original source
contact mask exactly and adds 3D position at existing FK coefficient.328, without
removing vertical loss; it did NOT improve actual output. Source1200 control
three-route peaks9.05/11.28/11.55cm, XYZ13.55/10.66/13.86cm. Conditional main foot
error5.67 vs7.32cm. No further XYZ-weight sweep is justified.
Through-history route peaks8.52/17.35/6.93cm, detached11.17/14.22/12.94cm: both
fail. Source1000 peaks14.85/17.16/12.44cm; lower training loss was NOT improvement.

Source continuation restored all434 Adam counters at200, original LR5e-5,
cursors[10,10,10,5]/indices[0,1,2,4] and Torch CPU/CUDA RNG. A CPU next-step
model/Adam exact-match test passed; no unsupported CUDA bitwise claim. The source
pair visualization archive is now local:38 files verified,8651762bytes, SHA256
`98fd96d19ba7013f0723cc7b1554574abfa8d186b1ab23393a0ca4ebe00f9f1c`.
Later probe exports are not yet delivered. Automation remains absent, not
recreated. Old accepted model/pages remain untouched.

## Completed Whole-Passage Temporal-Credit Probe

Source-imitation pair is COMPLETE. Joint Root/Pose/decoder improves main-route
peak to3.90cm versus decoder-only15.89cm and V6's7.93cm, but NOT solved: only one
strict stationary run, missing middle-stair landings, flat0/2. No same-tread
event is NOT a pass when contacts are absent. Source-pair three-route joint peaks
11.40/8.30/3.90cm; decoder-only6.34/15.03/15.89cm. Both native runtime and future
isolation passed, frozen hashes passed; no deployment.38-file source comparison
is privately uploaded; local download may still be in progress, verify receipt.
`reciprocal_source_footfalls_v1` contains the actual landing evidence.

Next controlled small loop is running, NOT a new full budget:
`sequence_credit_detached_v1` versus `sequence_credit_through_history_v1`.
Both initialize the same finished coordinated source checkpoint.20 episode
optimizer updates each, each accumulating3 full10-primitive stair passages and
one13-primitive idle/walk passage:860 actor primitives and80 source-AE examples
per arm, NOT860 optimizer updates. Fresh Adam5e-5, original physical coefficients,
Root/Pose/decoder train; vocab/encoder/all terrain adapters frozen.

Both use the SAME real initial4 source frames, then exclusively their generated
published history for each complete passage. Physical loss compares the complete
same-start trajectory to the real unmodified source, with real contact/velocity
targets. Native supervised first-window loss and rotating real source AE retain
the vocabulary; known-history prefix remains. No moving sources, source-time
resets, invented intermediate poses, future Root in actor, or runtime corrections.
The only arm difference is gradient propagation across played history.

Training-only `history_gradients=True` retains the graph through observed Root,
velocity/pose input packing and factual skeletal reencoding. DefaultFalse has
unchanged inference values. The first local wiring test failed because sparse
input packing still detached observed slots; this was corrected for the opt-in
path only, not by replacing any generated output.19 targeted tests now pass,
including nonzero future-to-earlier Root/rotation gradients, full two-primitive
parameter backward, unchanged default forward and future-target isolation.
Hard tokens still use the existing straight-through estimator; no claim of an
exact gradient through discrete sampling. Default runtime is used for reviews.

Immutable snapshot:`code_joint_sdf_sequence_credit_v1`; supervisor
`scripts.run_joint_sdf_sequence_credit --arm detached|through_history`.
States:`joint-sdf-sequence-credit-detached-state.json` and through_history variant.
Fresh idle GPU4/7 launches PID3559099/3560514 only identity clues. Check actual
cmdlines/state/progress; don't duplicate or alter running snapshots. First full
episode explicitly checks whether late loss reaches the first generated state;
all updated weights need original-clock full-route/footfall/flat review.
No automation exists and none was recreated. Old accepted model and pages stay.

## Completed Source-Imitation Capacity Probe

User explicitly insists on a real fix. New bounded pair (not a restarted full
campaign): `reciprocal_source_decoder_v1` versus `reciprocal_source_coordinated_v1`,
each200 actual updates/batch4 from accepted `factual_prefix_aux_v1`. Snapshot
`code_joint_sdf_reciprocal_source_v1` is immutable. Controls/data/seed/LR/losses
match; the only arm difference is decoder-only versus Root+Pose+decoder training.
Encoder, codebook, all SDF and legacy conditioning adapters stay frozen in both.
This is an explicit capacity test, not a claim that unfrozen models are better.

Training uses REAL source first4 frames for every real33-frame source window,
not generated history paired with a different source time. Native noisy Pose
token/Root/motion supervision and actual source FK/sole supports/material
velocities use the original geometry coefficients. Source-codec AE retained;
factual Root independently normalized, body prefix at old scale. No frozen-model
weak phase/support labels, guessed zero-velocity labels, moved source, generated
recovery targets or future truth in actor. Source future Root only enters the
separate auxiliary autoencoder as before. Actor inference is unchanged.
This teacher-forced fit is NOT recovery training; closed-loop reviews must use
actual generated history without resets and can still fail despite good fit.

Three original prepared source passages were audited locally: all have regular
seven-level stair flights and no interior opposite-foot same-tread events under
the current evidence test. Bottom/top platforms and source cross-tread strides
are retained, not relabeled. The source includes some multi-tread strides; this
does not supervise an exactly-one-riser-per-step requirement. Source contact
labels are measured kinematic evidence, not force sensors.

11 local targeted tests pass: future-target mutation actor difference0,
straight-through forward agreement, intended gradient groups, frozen codebook,
and prior factual-prefix/footfall tests. Runner checks actual A100 initial
accepted replay, future/runtime parity,8-step gradients/frozen hashes, then
200-step final checks, varied reviews and original-clock15s route review.
Supervisor:`scripts.run_joint_sdf_reciprocal_source --arm decoder|coordinated`.
State files:`joint-sdf-reciprocal-source-decoder-state.json` and coordinated
variant. GPUs4/7 were freshly checked idle; launch PIDs3460355/3461490 only
identity clues. Do not duplicate/restart; read actual states before actions.
Every arm has800 actor source-window examples plus800 auxiliary AE uses.
FLOPs/parameter counts differ; do not claim identical compute. No deployment.

## Reciprocal Footfall Requirement And Audit

Latest user rejects V6's left/right feet successively going to the SAME tread.
Required gait is reciprocal forward ascent: the next swing foot passes the
supporting foot and lands on a higher tread while inside the stair flight.
Do not redefine this as success merely because Root never steps backward.
Natural slowing remains allowed; stance duration/cadence and step length must
remain consistent with actual treads. Catch-up on the TOP LANDING is allowed,
so do not blindly require every footfall everywhere to increase height.

Local read-only `reciprocal_footfall_audit_v1` measures official98-vertex soles
per foot, unchanged80-frame source versus120-frame V6. The actual scene contains
seven regular tread levels0/.1434/.2869/.4303/.5737/.7172/.8606m. V6 left foot
has near-surface evidence frames76..79 on level4(.573734m), then right85..87
on the SAME level4, with higher treads remaining. This is an interior step-to
event. Source has no corresponding interior repeat; source L/R catch-up on
.8606m is the top landing and explicitly not rejected. Do not confuse projected
foot positions during swing with landing labels: earlier swing maxima are often
10-40cm above the surface.

Near-surface evidence requires>=3 sole vertices within2cm for>=3 frames. Stricter
stationary evidence also requires the same material vertices moving<=.1m/s in
both neighboring intervals. Source has6 such runs; V6 has0. These are kinematic
checks, NOT force measurements or a proof that the source is flawless. Missing
evidence cannot count as a valid landing or correct alternation. Several V6
right-foot swings have no sustained near-surface event; retain that uncertainty.

Implementation gap confirmed: `teacher_support_gap` only supervises height at
the selected sole XY; inside a tread its XZ gradient is0 and it does not penalize
sliding or encode left/right order/next tread. New regression test demonstrates
identical loss for static and sliding feet on the same plane. Existing body
distillation plus history continuity does not close this gap. Correct Root
continuity is retained, but V6 is still NOT the requested stair gait.

Do not simply add zero world-velocity loss to the old weak phase mask: audit on
the true source selected4206 intervals, mean speed.341m/s, median.081m/s,
P90.967m/s,16.38% above.5m/s and9.58% above1m/s. It includes legitimate foot
transitions, so treating every selected vertex as a stationary contact would
corrupt the training target. No such unvalidated loss or new training was launched.
Next repair needs verified stance/landing supervision and cadence/foothold
coordination, not another collision/history coefficient sweep or a runtime
footstep/Root correction. Preserve source geometry and no future truth in actor.

New script:`scripts/audit_joint_sdf_stair_footfalls.py`, tests in
`tests/test_stair_footfall_audit.py`. Report and inspected plot/PDF are local in
`out/model_terrain_v3_20261011/joint_sdf_20261012/reciprocal_footfall_audit_v1`.
No model, old recording or runtime changed this turn; no training running.
Automation is absent and was not recreated. Prior before/after remains a failed
experimental comparison, never a deployment approval.

## V6 Backward-Seam Repair Results

Both V6 probes completed200 updates each (400 total actual small-trial updates,
1600 actor/1600 teacher/1600 source-AE examples, counted separately). States are
complete; no training or export process remains. No model was deployed for WASD.
All source/control schedules match archived controls with hash
`23278c5f69e0c9c4d09d1c4dba1b5811f50ad67fe8529b51b35246f38a1ba1cd`.
Initial replay/teacher/future-target/runtime differences0; first8-step checks and
final200-step frozen hashes/decoder-only gradients passed.15 local tests pass.
Snapshot remains immutable. Main script hash
`c9bf95c7218e264d4abeff9515782d50af31fb68ed3013f28cca1cdb26fa6658`;
loss hash`b686f5295c308372527e741aa26b3ec54cf3eaf3ff7639949c34ed3acf17f867`.

Isolated support V6 versus support V3 reduces main-route accumulated backward
travel2.115m to0.269m and removes backward seams over5cm, but has residual small
backward motion,1.122m lateral deviation and unverified terrain points. It is
NOT adopted. Its three routes peak17.22/18.53/10.83cm; flat screen0/2.

Direction V6 (matched single-change control is direction V5) completes all three
routes in5.07/5.33/4.00s with ZERO backward frames, route lateral endpoint error
2.64/9.16/3.58cm and no unknown geometry. Main120-frame net/path lengths are
5.336/5.346m. Mid/late full gait cycles advance61.46/52.19/49.30cm and reduce
fore-aft foot excursion79.05/63.90/56.12cm, instead of slow V3's~19cm net cycles
with~28cm backward corrections. No minimum forward speed was added. Main route
remains slower than accepted gait2.13s, without the old visible backward tug.
Do not attribute V3->direction V6 entirely to Root normalization: this includes
the direction-control change; the archived direction V5 already had no backward
frames, completed in2.13s and had27.59cm main penetration. Speed alone is not a failure.

Quality STILL FAILS. Full-skin8-substep exact peaks for direction V6 are
20.18/22.40/7.93cm. Main improves versus slow V3's17.23cm and matched direction
V5's27.59cm, but other routes worsen relative to direction V5. Flat idle drifts
12.00cm (>10cm screen); walk passes, so flat1/2. Body silhouettes and cycle
progress improved locally, not certified real contact or universal naturalness.
Direction V6 checkpoint SHA256
`9bb90c74f83bfa114f1d789d36ed8e43197297022012a04870a18c75a18125d0`.
Old accepted checkpoint/page remains untouched. Do not deploy this candidate
or describe the entire terrain/stepping problem as solved.

`natural_factual_comparison_v6` includes all five arms across three routes,
their original recordings/metrics/contracts and official SOMA visual exports.
Downloaded via existing7897 proxy and user-owned private Drive client;
45 receipt files verified. Archive15206276bytes SHA256
`40cfd983f20d6582c6e30f870a146f1e5b43ac063d9be28598f3d02b25d8c846`.
All15 rig/root lengths checked, no padding or retiming. Local viewer inspected
at frame73: left slow failure, right direction V6. Desktop/mobile screenshot
pixel checks pass; viewport restored; delivery tab playing in background.
Screenshots:`natural_factual_comparison_v6/before-after-frame73.png` and
`mobile-review.png`. A viewer-only clock bug was corrected: loop length now uses
the TWO SELECTED arms, not a hidden longer arm. Verified176->120->176 when
switching left arm. Existing decoder_pilot and motion data unchanged.
URL:http://127.0.0.1:28936/web/terrain_route_review/?data=/out/model_terrain_v3_20261011/joint_sdf_20261012/natural_factual_comparison_v6/visuals/&clip=reported-stairs&left=slower&right=factual_direction

Remaining work is contact/stance consistency and flat idle, not more blind
history-weight tuning. Weak support proxy is not contact truth. Preserve allowed
natural slowdown, all failures and existing geometric gates; no IK/runtime
corrections. No additional training was launched after this failed full review.
Automation note: at final synchronization the app returned that `joint-sdf`
does not exist (possibly manually deleted). It was NOT recreated or resumed.
Earlier references to PAUSED describe the prior configuration, not current app state.

### V6 Experiment Contract

User explicitly requested fixing the treadmill failure. Two bounded200-update,
batch4 decoder-only trials are complete: `contact_factual_support_v6` and
`contact_factual_direction_v6`. Immutable code:`code_joint_sdf_contact_factual_v6`.
Both start from accepted `factual_prefix_aux_v1`, not from the regressed V3.
They match archived support V3 / direction V5 respectively in baseline, source
windows, command schedule, exposure, LR and all other losses. Only the13 factual
Root-channel loss normalization changes: remove its old28*365-feature contribution
and instead average over13 known channels, using the existing native Root
coefficient1. Body prefix normalization is unchanged. No runtime graph change,
no output replacement, no invented future or direct future speed target added.
The direction arm retains zero penalty at any nonnegative collinear future pace.
This is a significant Root-gradient rescaling (~786x), disclosed and isolated;
it is NOT automatically proof of quality improvement.

15 local tests pass, including exact unchanged body gradients, zero future and
unknown interval3 velocity gradient, decoder-only gradients and unchanged actor.
Each runner checks first8 real updates, frozen hashes and accepted output before
training, then final full/varied reviews and original-clock15s route review.
New quality check includes recurrent backward seams; no minimum forward speed.
Supervisor:`scripts.run_joint_sdf_factual_contact --arm support|direction`;
states:`joint-sdf-contact-factual-support-state.json` and direction variant.
Started on freshly checked GPUs4/7, supervisors2950436/2951595 only identity clues.
Read actual state/cmdline before acting; no duplicate launch. Existing automation
remains PAUSED. Do not overwrite training snapshots or accepted visual model.

## Confirmed Treadmill Regression

Latest user correction: the contact candidate repeatedly steps in place. Do NOT
defend this as natural slowdown or as acceptable merely because it crosses the
goal eventually. Natural slowing remains allowed. Retain the accepted orange
`factual_prefix_aux_v1` unchanged; no contact candidate is adopted.

Read-only `stair_treadmill_v1` audits the actual original-duration main-route
recordings (before64frames, slower176frames, source80frames), without retiming,
future substitution or new training. Slower V3 has19 backward frame steps,
ALL at8-frame replanning boundaries (frames24,32,...,168). Max backward step is
15.514cm; accumulated backward travel2.115m. First-recorded-frame-to-last net
forward travel5.283m versus XZ path length9.582m; do not mix this reference origin
with the prior5.348m goal-axis progress from initial history. Accepted before
and true source have ZERO backward steps on this clip.

Legs keep cycling at about0.533s/full left-right cycle. Frames66..82 and82..98
advance only19.91/18.78cm net, despite48.20/46.68cm gross forward travel, because
28.28/27.90cm is undone by backward jumps. Root rises-0.50/+15.18cm respectively;
do not call every low-XZ lift an idle failure. Fore-aft separation excursions
remain1.063/1.116m. These are displayed FK phase measures, NOT force contacts.

A100 `stair_treadmill_replay_v1` reproduced BOTH saved recordings with max
feature difference0,0 optimizer updates. Decomposition verified every block:
published step = error reconstructing the3 factual past Root intervals + decoded
new interval. Frame72: factual past+12.440cm, reconstructed past-2.871cm,
past error-15.310cm; outgoing step-0.204cm gives published-15.514cm. Frame88:
past error-14.751cm plus outgoing-0.638cm gives-15.390cm. This is an actual
decoder-history regression, not a viewer interpolation effect. Earlier accepted
prefix model has much smaller offsets and no reversed published steps.
No runtime Root replacement or offset compensation was applied.

The current factual-prefix loss is still present, but its13 observed Root
channels share the full28*365-feature denominator with body reconstruction.
That is a code-level weakness in protecting factual Root continuity, not proof
that adding an arbitrary large weight solves gait/contact. Same-input frozen
teacher pose preservation and weak foot support do not couple actual forward
advance to each stance cycle; future speed matching is NOT the remedy. The next
bounded repair should distinguish factual past consistency from optional future
slowdown and measure cycle progress/contact together. Do not blindly repeat
V4 speed preservation, add minimum runtime velocity, or launch another full run.

The previous4-second/<5cm stagnation count was too coarse and missed ALL these
short cycles. The mean norm of Root displacement also counts backward motion
as travel; do not present it as net forward speed. Local future route review now
records full-cycle progress, gross/backward travel and8-frame seams, and rejects
3 or more backward seam jumps over5cm. This explicit regression screen imposes
NO minimum forward speed and is not complete naturalness certification.
Three tests pass, including valid1mm/frame slow forward motion and artificial
backward seam detection. Completed remote snapshots/reports remain untouched.

Artifacts: `stair_treadmill_v1/complete.json`, `replay-report.json`,
`diagnosis.png`/`diagnosis.pdf`; plot inspected. No new training this turn.
No model has been fixed or redeployed by this audit. Heartbeat remains PAUSED.

## Previous Contact Trials

### Latest User Correction: Natural Stair Slowdown Is Allowed

The user explicitly corrected the pace criterion: slower movement on stairs is
acceptable and realistic. Do NOT require the same displacement in80 frames,
the original cruise speed, equal clip duration, or equal-time pose alignment.
Prior statements rejecting V3 merely for slow80-frame progress are withdrawn.
Do not equate slower travel with stopping or bypassing the stairs. Penetration,
route completion, direction, sustained floating and actual stationary control
remain separate checks. Collision/contact tolerances are NOT relaxed.

`natural_stair_time_v1` now evaluates before/V3/V4 on the same three routes with
an explicit15-second safety horizon independent of requested speed. It retains
original30Hz frames and ends each recording only at the fixed goal or the bound.
V3 reaches the three goal planes in5.07/11.73/5.87seconds; this does NOT establish
absence of local stepping in place (see current audit). The main route's
displacement-norm speed is1.53m/s versus before2.67m/s; this includes backtracking
and is not net advance. Additional failures are17.23cm
penetration and96.6cm endpoint lateral deviation; the second route also ends at
the wrong height. V4 reaches the main goal in2.13s with10.37cm peak and8.3cm
lateral deviation, still not penetration-free. A timeout is not proof of
incapacity at arbitrarily slow speed. Full report is local.

New matched `contact_direction_v5`, immutable `code_joint_sdf_contact_direction_v5`,
starts from the same accepted orange checkpoint,200updates,batch4. It replaces
speed matching with normalized lateral-velocity and mean-backward penalties;
ANY nonnegative collinear pace, including0 for a preparation window, has0 loss.
Idle commands still target0 velocity. Frozen-teacher body rotation/relative-FK
preservation, source autoencoder, factual history, exact penetration and weak
support remain. No minimum forward-speed clamp or runtime correction exists.
V3 is the matching reference, with equal baseline/source/command exposure.
One new unit test verifies zero loss AND gradient for speeds0,.05,.5,1,3m/s under
the same forward request, while sideways/reverse/idle drift incur loss;6 tests pass.
V5's preferred interpretation is natural control, not restoration of the old
speed constraint. It and its long-duration route review are now COMPLETE.
No training/review/transfer process remains active. Do not restart these trials.
Script hash`d6c0c9c29998ca8c6f5b975f537105ea1ecf6de245759b14f34908d88ac1d4c6`;
loss hash`82821d07ca7f7ee60da3831c306a0100b9c5b34d157fbc45db6f1214b8370cf1`.

### Current Results And Delivered Comparison

No contact candidate is approved for deployment. On the main route, the
accepted baseline's8-substep skin peak is19.09cm; V3 slower17.23cm; V4 speed-
preserving10.37cm; V5 direction-only27.59cm. V4 improves this one route but not
the other stairs and is NOT preferred merely for being faster. V5 completes
all three longitudinal goal crossings in2.40/2.40/2.13seconds with lateral
errors23.5/19.7/28.3cm, but penetration15.06/9.02/27.59cm still fails. Its idle
drift3.80cm and walk pass the existing flat screen; that does not fix stairs.
Do not describe the new direction loss as an empirically successful contact fix.
The preceding fixed80-frame results remain diagnostics, not natural-speed gates.

This request ran five200-update arms (two V2, V3, V4, V5), each800 actor examples,
800 frozen-teacher calls and800 source-autoencoder examples. Failed V1 runs made
zero updates. These are separate small experiments, not a new large campaign.
Eight targeted unit tests passed, including explicit no-penalty/no-gradient for
forward slowdown. All initial accepted-recording/future-target/runtime/teacher
parities and frozen-gradient checks passed; quality remains a separate failure.

`natural_contact_comparison_v5` was privately downloaded and hash-verified:
archive9041399bytes SHA256
`85cb6b24d8027a9013b350f9a857b158c0aed327c0f8d33748a3ea688e5e21f7`,
40 receipt-listed files. All12 original rigs satisfy their own actual frame
counts. `web/terrain_route_review/` is a separate viewer; the old decoder_pilot
and accepted gait page are unchanged. Unequal-length arms keep original30Hz,
own durations, no retiming or appended recording frames. The shorter panel
displays its final pose with an explicit Ended label while the longer one runs;
this UI end state is not fabricated extra motion data. Common timeline length
is only a viewer clock. Before-main64frames/2.13s and slow-main176frames/5.87s
were verified at wall-clock4.03s. A camera framing issue was corrected only in
the new viewer. Desktop/mobile screenshots and blue/orange canvas-region pixel
checks passed; temporary viewport was reset. Direct WebGL context access is not
exposed by the browser DOM tool, so pixel checks used actual screenshot pixels.
Screenshots:`natural_contact_comparison_v5/original-duration-desktop.png` and
`original-duration-mobile.png`.
Delivered original-duration comparison:
http://127.0.0.1:28936/web/terrain_route_review/?data=/out/model_terrain_v3_20261011/joint_sdf_20261012/natural_contact_comparison_v5/visuals/&clip=reported-stairs&left=before&right=slower

The user's requirement is natural safe traversal with allowed slowdown, NOT
fixed speed. Next work must use that corrected criterion and actual contact
evidence; the weak phase proxy is not ground truth and does not justify more
blind weight/step sweeps. Retain the accepted stair gait checkpoint unchanged.
No IK, collision projection, Root replacement or output smoothing was used.
The existing heartbeat remains paused; do not silently resume it.

Latest user authorization: preserve the current orange stair gait, first repair
terrain penetration with small-sample training and matched before/after tests;
NO IK or runtime corrections. The accepted starting model is
`factual_prefix_aux_v1/weights.pt`, SHA256
`b4faa6a64defb4d7f76320f5a4003ade7a88038a2a527a61f9ed72d064df0f35`.
It remains unchanged and is a visual gait baseline, not broadly quality-approved.
No large training budget or change to the paused heartbeat is authorized.

New matched runs:`contact_preserve_control_v2` and `contact_preserve_skin_v2`,
under`joint_sdf_20261012/`. Immutable code:`code_joint_sdf_contact_preserve_v2`,
runner`scripts.pilot_joint_sdf_contact_preserve --arm control|contact`.
Each is bounded at200 actual updates,batch4:800 actor examples,800 same-input
frozen-teacher calls and800 source-autoencoder examples, counted separately.
Only codec.decoder trains; Root/Pose/encoder/codebook/all adapters are frozen.
The entire initial orange actor is frozen separately as the style teacher.
Histories are actual current student-generated8-frame outputs, detached between
updates; resets only at genuine episode boundaries, not source-time alignment.
Both arms use the same source windows/commands/optimizer/other losses, although
their generated histories naturally diverge. Do not claim identical histories
or bitwise training. Baselines/schedule/exposure hashes must match.

Three real stair passages extend to105 real frames/10 complete33-frame windows,
with their original prepared prefixes checked identical; no padding or moved
sources. Flat idle/walk retain13 steps. Stair training commands use each exact
constant approach velocity and body yaw. The primary8030 baseline must reproduce
the accepted80-frame recording before any optimizer update. Standard varied-goal
reviews remain distinct from this exact recording and retain their old commands.

Both arms retain native source-codec reconstruction and factual past Root/pose
loss. Same-input frozen-teacher distillation retains normalized rotation and
planar Root velocity plus pelvis-relative FK, NOT absolute Y. RIC Y is absolute
in native MB, so blindly distilling RIC Y would preserve the existing penetration;
this implementation uses actual FK relative to the pelvis instead. Distillation
is a training loss, never output replacement or runtime trajectory correction.
The separate autoencoder's future source Root remains training-only.

Only the contact arm adds`terrain_skin_tail_loss.py`: official full18056 skin
vertices on the actual8 published frames plus4 linear sweep substeps from the
last factual frame. Exact negative heightfield-prism distance, existing5mm margin,
worst1% mean squared excess to avoid dilution by nonpenetrating body points.
The original collision coefficient8.42715518 and FK coefficient0.32805658 are
retained, not searched/reweighted. This is a new tail-risk reduction, not a claim
that the old training used approximate SDF (it already used exact negative SDF).
No collision or force-contact labels are fabricated. Unknown/uncertified queries
fail explicitly. Full17-case review and exact80-frame skin peak, floating-foot
clearance, gait, travel, sideways/height change and idle are reported afterward.
Low penetration while stopping, detouring, floating or losing the accepted gait
must not be called success. No failed case is dropped, no frames are filled.

Three unit tests passed: plane and step-wall gradient signs/finite difference,
zero loss when clear, factual history detached, future tensors unchanged, and
unknown terrain rejected. First8 A100 updates additionally check active skin
gradient, no unpublished-frame output gradient, actor future-target/runtime
parity, same initial teacher output and frozen hashes. Gradient may correctly be
zero on a clear first step; an active sample must be seen by8 updates.

V1 aborted at initialization, BEFORE baseline/optimizer updates, because of a
missing`state_digest` import. Both failed run directories and immutable snapshot
are retained. V2 only fixes that import, no objective/budget change. Script hash
`8aa177839f9f5ebb048bf9ef0d46a63bbc481cb8e6756085b506f4861352af6e`;
loss hash`67778ba4c63c2c332b164ed1bc42fc9839d71d6be5d402433dff680d1a0e3635`.
LaunchGPU0/2 were freshly checked idle; PID2512770/2513325 are identity hints only.
Both V2 runs have now completed200 updates and all reviews. The collision-only
arm is explicitly REJECTED for floating despite low penetration: main8030 peak0,
but90% of frames have both soles above10cm and all80frames above5cm. Other two
stairs have86-89% above10cm. Mean Root-Y increases17.6/21.8/28.3cm. This is learned
model output, not an applied Root offset, but still fails the user's requirement.
The no-collision control retains18.10cm main-route peak. Current accepted orange
baseline full8-substep peak is19.089cm; its earlier discrete-frame peak19.047cm.
V2 initial recording/teacher/future-target/runtime differences all0, gradient
and frozen checks pass. Files and failed floating candidates are preserved.

### Second Contact Loop: Prevent Floating With Weak Stance Evidence

First audited the original stationary-sole rule (two consecutive <=.1m/s world
intervals, bottom2cm). The accepted main-route model has0/80 eligible frames,
while the unchanged source has57/80. Thus silently pretending predicted feet
have measured stable contacts would be wrong. A separate weak phase proxy uses
two backward intervals relative to the pelvis (<=-.1m/s along current command),
a foot at least5cm lower than the other, and only that foot's bottom2cm vertices.
Idle uses the original stationary rule. Masks come ONLY from the frozen accepted
teacher and are detached; student output cannot choose its supervision mask.

On actual source80frames the phase proxy selects64frames/5009 vertex samples,
mean absolute gap1.68cm;5.23% of selected samples are>5cm above terrain, so it is
imperfect weak evidence, NOT force/contact ground truth. On the accepted actor
it selects72frames/5131samples, mean absolute gap6.83cm. Preserve these limitations;
do not turn proxy success into motion acceptance or use it for runtime foot locks.

New`contact_support_v3` runs200updates from the SAME accepted orange checkpoint,
not the rejected floater. Immutable`code_joint_sdf_contact_support_v3`. Same
schedule, teacher, losses and collision coefficient as V2 contact, plus actual
predicted-XY terrain height gap at proxy-selected sole vertices. Its coefficient
is the original support_gap_m=0.353079742, not retuned. No future source pose/height
enters the actor, no output modification, and no dropped previous losses. The
completed V2 contact arm is the matching reference; baseline and final exposure
hashes are asserted against it. New actor histories may diverge, as expected.
Five contact tests now include detached proxy masks, false isolated-apex rejection,
backward/lower-foot selection, and support gradients only on selected future points.
V3 script hash`0274f07b62e087fedbcd0b82dc0f234838b37c5ace822cabf4319a4497665fca`;
loss hash`70712bc24364213fc0b31c3758a38dcd0e1492dfed3ac77983669914b8b57f21`.
V3 is complete; its old equal-time interpretation below is SUPERSEDED by the
user's natural-slowdown correction at the top. Main-route80-frame progress falls from7.33m to3.05m
(other routes7.05->2.50,6.59->1.07), with17.23cm main peak remaining. Floating is
reduced. A later longer review confirms that it does cross the goal plane;
reject its penetration/lateral failures, not its pace. Idle improves18.20->2.65cm and
walk passes, yet these do not compensate for failed stairs. All17 reviews and
raw outputs are retained. V3 export107 files/28.65MB has been downloaded privately
and its archive+internal hashes verified; it is not a repaired candidate.

### Third Loop: Explicit Planar Root Preservation

Code review of the preservation term shows3 normalized Root channels averaged
with180 rotation channels and then divided by the365-feature representation.
This assigns each Root component only3/365 of the independently normalized
Root-head scale. It was too weak to preserve travel under contact/support loss;
this is a diagnosed loss-balancing design problem, not an inference Root bug.
`contact_support_v4` uses the unchanged rotation term plus independently averaged
3-channel Root preservation at the existing native local_root_loss_coeff=1.
Collision/support coefficients remain unchanged. V3 is the matched200-update
reference; the new run starts again from the same accepted orange model, NOT
from its slowed or floating predecessors. Same windows, commands, seed and
budget; on-policy histories may differ. No absolute world position target or
runtime displacement constraint is introduced.

Immutable`code_joint_sdf_contact_support_v4`; runner flag
`--phase-support --separate-root-preservation --reference-trial .../contact_support_v3/complete.json`.
Script hash`8368a7809be3e8374806b6c59b85a13bc8a472a038fbedfdd6e3083bd5d27007`.
PID2602762 only a hint; GPU0 freshly checked idle. Inspect V4 state/progress/
preflight/complete and actual cmdline; never restart healthy training. Do not
claim success on low penetration if travel, gait or sustained contact fails.
Source/models/animations continue to use only user-owned private Drive/NAS;
old comparison28936 files and user tabs remain intact, no live candidate loaded.

## Completed: Pose History Sensitivity And Recovery Feasibility

### User Visual Acceptance And Remaining Contact Problems

The user explicitly says the orange prefix candidate now looks natural going
upstairs and asks what problems remain. Preserve this as positive qualitative
feedback on this clip, not a request to reject its gait style or a blanket
deployment approval. No new model/training was started for this follow-up.

Fresh LOCAL full-skin audit of the EXACT current `reported-stairs`80-frame
recording (not the different goal-ended/held-direction reviews) found static
frame penetration over2cm on46/80 generated frames, versus0/80 source frames.
Maximum generated penetration is19.047cm at zero-based frame18 (0.60s/display19),
vertex17965 at[11.78978,-0.19047,-0.12884]m. Source discrete-frame maximum0.637cm.
Every query is known/certified. These are discrete original30Hz frame measures;
do not substitute them for swept metrics or claim original triangle-mesh CCD.
They use the same official18056-vertex skin and nearest-cell heightfield-prism
solid as the displayed terrain, whose mesh cell boundaries were checked in
`web/decoder_pilot/app.js`. This is not the earlier coarse voxel approximation.

The exact frame was inspected with opaque AND transparent ground, in a separate
background tab without changing the user's existing comparison. The orange
foot is partly hidden by the opaque terrain, while the source foot remains
visible. Screenshot:`factual_prefix_export_v1/source-vs-prefix-contact-frame18.png`.
Temporary inspection views are not new live models. Other verified remaining
issues: idle18.20cm drift in104frames (3.47s); same-route20%-faster goal-ended
test has28.10cm full-skin peak; original-stair default start has48.2cm sideways
endpoint deviation. Those belong to SEPARATE test commands/routes, not the
exact current80-frame recording. Ordinary walk passes and two BONES runs retain
alternating gait. Prioritize contact/terrain matching, then stationary control,
then speed/start robustness; do not keep treating all visual motion as broken.

The user said to continue after the hopping diagnosis. The read-only audits below
have ZERO optimizer updates. Their measured rotational prefix failure motivated
one bounded decoder capacity A/B, now COMPLETE and rejected after the full reviews
below. No training, audit, export or transfer process remains active. No runtime
modification or deployment approval.
The completed160万 main budget and previous small trials must not be restarted.
The frozen snapshot `code_joint_sdf_history_root_v1` remains unchanged; new audit
scripts live beside it in the task directory. Preserve28934/28936 and user tabs.

### Root Planning Versus Pose History

`pose_condition_v2` captures the final actual Pose/decoder calls and replays
unmodified tokens exactly. Each intervention starts from a recursive independent
snapshot; original tensors are checked unchanged and baseline tokens reproduced
again AFTER all substitutions. Both generated and source-history modes retain
their actual80 frames. The normal actor still matches the user's recording with
max feature difference0. Correct-source-history substitution has input difference0
and leaves its outputs unchanged, validating coordinates/normalization.

| Generated-history intervention | Last32 close-feet fraction | Relative joint error |
| --- | ---: | ---: |
| Original actor |84.375%|15.47cm|
| Source Root plan into Pose only |59.375%|14.52cm|
| Source past pose into Pose only |6.25%|11.31cm|
| Source Root and past pose into Pose |18.75%|11.08cm|
| Source Root plan into Pose AND decoder |100%|12.70cm|

Decoder played-history conditions and actual scene/SDF memory remain unchanged.
These isolate upstream token selection from decoder input replacement. The
source-past sensitivity is stronger here than merely replacing the Root plan,
but neither is a causal runtime correction or proof of feasible recovery.

`pose_components_v3` splits the past Pose input into hip height, RIC positions
and6D rotations. Generated-history close-feet fractions: height81.25%, positions
78.125%, rotations3.125%, body excluding height9.375%, all6.25%. Source-history
controls are all unchanged. Thus rotation-history channels carry most of this
measured sensitivity, not simply accumulated pelvis height. Mixing source
rotations with generated positions may describe an inconsistent skeleton:
these are channel-sensitivity probes, NOT artificial training poses or labels.

Important invalid experiment: `pose_condition_v1` used a shallow tensor helper
on dictionaries, causing shared-condition mutation. Its results are rejected,
not compared or exported. Files/logs remain. V2 uses recursive snapshots;
3 local diagnostic tests cover nested isolation, capture/restoration on failure,
and SO(3) angle identity/90-degree cases. Original runtime code was never changed.

### Recovery Candidates Fail Geometry And Boundary Checks

`recovery_candidates_v1` measures each offline8-frame output from the factual
last played frame, with all18056 skin vertices and8 linear sweep substeps. This
is exact distance to the measured heightfield prism, NOT dynamic validation or
continuous collision against the original triangle mesh. All samples in this
audit are known/certified. The original actor has13.20cm future skin peak;
source-token oracle12.97cm; source-past-Pose oracle15.13cm. None are admitted.

The unchanged source with its own history has1.033cm peak, maximum boundary
joint speed9.54m/s, maximum finite-difference joint acceleration118.41m/s2.
The same fixed-time source futures joined to actual generated history require
43.33m/s and1381.61m/s2, have3 colliding boundary intervals and2 backward8-frame
targets. These quantities are diagnostics, not newly loosened quality gates.
They reaffirm that forcing a same-time source future is not a feasible recovery
target once the actor's state has drifted. Low future collision alone misses it.

### Representation And Native Sampling Checks

`rotation_recovery_v1` uses the same actual argmax history and Root/terrain to
sample16 native stochastic MaskGIT token sequences at each predeclared block
0,3,6,8, plus its argmax reference. Each first random sample is reproduced with
the same seed; all original conditions and argmax tokens are unchanged afterward.
All68 candidates and full32-frame predictions/actual4-frame histories are saved.
No candidate is fed back or selected as a training label. Final actor replay0.

Past-state reencoding retains rendered FK: maximum position error9.54e-7m,
rotation error<0.00006deg. Source6D rotations are orthonormal to floating-point
precision. Raw decoder6D axes have norm errors0.216-0.340 and dot products
0.103-0.200 in the sampled generated histories; their canonicalized observed
versions are orthonormal while preserving displayed motion. Decoder redundant
RIC positions disagree with displayed FK by mean5.76-6.03cm/max24.95-33.69cm.
These are output consistency errors, NOT evidence that reencoding rotates the
body incorrectly. Repeated feature reencoding differs by<=5.54e-5 (includes
derivative channels), so do not claim bitwise idempotence of every feature.

Only the existing2cm future-skin/known-map/positive-progress conditions are used
as a GEOMETRY-ONLY screen. Pass counts including argmax:15/17,14/17,4/17,0/17 at
blocks0,3,6,8. At block6, geometry-passing candidates still have boundary joint
speed>=12.36m/s and acceleration>=342.82m/s2, above the source control. At block8
even the least penetrating candidate has7.82cm peak. This bounded sample rules
out claiming that a random seed switch is a demonstrated fix; it does NOT prove
that no possible token sequence can recover. No candidate is quality-approved.

### Current Artifacts And Next Decision

Local JSON reports:`pose_condition_v2-report.json`, `pose_components_v3-report.json`,
`recovery_candidates_v1-report.json`, `rotation_recovery_v1-report.json`, under
`out/model_terrain_v3_20261011/joint_sdf_20261012/`.
`pose_diagnostic_export_v1` packages all68 native samples and same80-frame
condition probes with explicit offline labels, using user-owned private Drive.
All158 receipt-listed files were downloaded and SHA256-verified. The rotation
channel comparison was visually inspected at frame55; screenshot
`pose_diagnostic_export_v1/rotation-history-frame55.png`. It is an inconsistent
input diagnostic, not a corrected controller or an approved motion.

The static review service had stopped: neither28934 nor28936 was listening when
checked. Only28936 was restored, bound to127.0.0.1 with the same motionbricks
document root; HTTP200 and official skin rendering were verified. PID157136 is
a local identity hint only. Old files/URLs were not changed. Old28934 was NOT
replaced/restarted and failed candidate weights were not loaded for live use.

### Decoded Past Rotations Reveal Boundary Inconsistency

`rotation_boundary_v1` is complete. It uses only saved original predictions,
not altered motions. The SO(3) product identity for the observed-to-published
rotation decomposes into observed-to-decoded-past and decoded-past-to-future;
maximum matrix residual across68 candidates is2.21e-6. Angles are NOT added.
At block6 the argmax RightShin decoded-past error is52.05deg, its own next-step
rotation4.04deg, and the actual published jump55.91deg. At block8 these are
58.21deg,1.34deg,56.88deg. Both shins are affected. RightToeBase reconstructed-past
position error reaches45.77/53.87cm at those blocks. This locates a severe mismatch
between the actual history and the decoder's internally smooth window, beyond
the already identified Root integration offset. Unlike Root velocity integration,
future global rotations are decoded directly: the SO(3) identity alone does NOT
prove that changing predicted past rotations would causally change future ones.
The controlled prefix-loss trial below tests that proposed training repair and
does not find robust incremental improvement. The ordinary decoded reconstruction
mask discards all first4 pose frames,
even though those predictions set the decoder's internal continuation boundary.
Conditioning on the history does not guarantee that its reconstruction equals it.

### Completed Factual Prefix Capacity A/B

An isolated opt-in loss `decoded_factual_history.py` supervises decoded factual
Root intervals0..2, heights0..3 and RIC/rotation pose0..3. Contacts, redundant
velocities, the unobserved outgoing interval and every future channel are excluded.
It uses native normalized SmoothL1 and the same per-feature scale as the original
28-frame future reconstruction; no prefix averaging amplification or tuned weight.
Two local tests verify zero at correct history, exact13Root+1068body channels,
no future gradient, decoder-only gradient and unchanged actor output. Three
diagnostic-harness isolation/rotation tests also passed.

Completed runs:`factual_prefix_control_v1` and `factual_prefix_aux_v1`, each under
`joint_sdf_20261012/`, each exactly100 updates,batch4. BOTH arms trained only the
codec decoder and keep original native SOURCE AUTOENCODER reconstruction. The
auxiliary arm adds the factual-prefix loss; the control computes/logs it but
does not optimize it. Actual generated-history banks are collected once from
the same initial SDF-only A with the existing commands, no artificial prefix,
reset, source translation or future Root replacement. Bank/schedule hashes and
complete baseline reports were compared and match. Only own-source history/Root enter
the separate source autoencoder. No fixed-time source future is imposed as an
actor recovery target in this capacity test. This is NOT continuation of the
previous joint actor objective or evidence that removing its losses fixes motion.

Each100-update arm has400 observed-prefix exposures AND400 source-autoencoder
examples. Do not misreport this as400 examples of all kinds or a new main budget.
Same fresh Adam5e-5, clipping0.5, prefix coefficient1 in native per-element scale.
All Root/Pose/SDF/old adapters, encoder and codebook are frozen and hash-checked.
No ST estimator or token changes are needed to train the decoder. Eight-update
and final100-update preflight checks pass: actor source-future mutation difference0,
runtime difference0, finite decoder-only gradients and frozen hashes unchanged.
Full17-case source/held/speed/idle/flat baseline/final and original8-case staircase
tests are complete, plus both BONES runs and reported/counterfactual80-frame runs.

Immutable snapshot:`code_joint_sdf_factual_prefix_v1`. Runner
`scripts.pilot_joint_sdf_factual_prefix --arm control|prefix`; each run has its
own`state.json`, `progress.json`, `preflight.json`, `weights.pt`, `complete.json`.
Fresh idleGPU0/2 were checked at launch. Training and post-review processes have
completed. Do not restart these trials or extrapolate into a large training run.
Script SHA256`11f2852f6f0ec0cb08a6a48f8cde6c4fba2aa8d45c925a3c46b6c4edee75802b`;
objective SHA256`d95f809edb10ffb68416a4723fdd49e4da229b2e0fafff216b569024fe0ee970`.
No new full training budget, runtime pose replacement, smoothing or projection.
All prior failures and quality thresholds remain unchanged.

### Capacity A/B Result: Not Adopted

Final screen: control0/17, auxiliary1/17 (walk only), stairs0/15 for BOTH. Official
stair start/speed review0/8 for both. Idle drifts17.06/18.20cm, failing10cm. The
source-reconstruction control also violates flat-walk2cm skin peak (2.34cm);
auxiliary walk peak1.16cm passes. No candidate is deployable.

The exact reported constant-command80 frames show less close-leg posture:
original84.375%, source-reconstruction control15.625%, prefix auxiliary12.5%.
However, on the same flat counterfactual it is original100%, control56.25%,
auxiliary78.125%. The incremental prefix benefit is NOT robust. Both real BONES
runs retain alternating gait; those two examples do not certify general control.
At the exact reported block6, RightShin decoded-past error is original52.05deg,
control6.83deg, auxiliary6.91deg; at block8 it is58.21,5.55,4.49deg. Published
block8 jumps are original56.88,control10.07,auxiliary16.00deg. The source-codec
control achieves most of the rotational-prefix improvement. It cannot be
attributed to the new loss, nor does it remove penetration/control failures.
These are equal-command/equal-time rollouts whose generated states and progress
also differ, not a held-state estimate of the decoder parameter change alone.

Last20-batch mean prefix losses are0.00254 control versus0.000775 auxiliary;
source reconstruction means0.02948 versus0.02871. The factual bank/schedule match,
but lower loss did not imply robust autoregressive improvement. This experiment
does not justify increasing the prefix weight, running longer or deploying a
past-rotation/root overwrite. Training-bank pasts are factual but from the fixed
initial model; this is not proof of coverage of every updated actor history.
Remaining work must distinguish future Root/Pose/codec errors under the now-more-
consistent decoder, with real feasible labels and source/flat preservation,
before another narrowly defined experiment. No accepted recovery labels exist.

All post-review JSONs are local:`factual_prefix_control_post_v1-report.json` and
`factual_prefix_aux_post_v1-report.json`. `factual_prefix_export_v1` contains
same80-frame official SOMA comparisons and ALL32 distinct full-review recordings,
metrics/contracts/hashes. It was uploaded as one private tar.gz to reduce small-file
API overhead; no model weights were transmitted. Archive24633663bytes SHA256
`3f219d32c8807ad93e86d6fe5696428040d6183fc1b0a227b20fce24c0c781cb`
and all95 internal receipt hashes were verified locally. Extraction paths were
validated to remain within the named local artifact directory. Real initial vs
prefix candidate at frame55 was inspected in the official SOMA viewer; screenshot
`factual_prefix_export_v1/initial-vs-prefix-frame55.png`. The new candidate has
separated legs at this frame but still fails quality, with no output corrections.
Background deliverable:
http://127.0.0.1:28936/web/decoder_pilot/?data=/out/model_terrain_v3_20261011/joint_sdf_20261012/factual_prefix_export_v1/visuals/&clip=reported-stairs&left=initial&right=prefix
The existing heartbeat was found PAUSED and was left PAUSED while updating its
context. Do not silently reactivate it. Current explicit continue work does not
authorize changing the paused schedule.

## Completed: Close-Leg Gait Collapse And Replanning Root Offset

The latest explicit user request is to find the root cause of close-leg hopping
in the right side of the8030 source-context comparison. That recording uses the
OLDER SDF-only `decoder_ab_A` model, not the latest bounded trial. Its hash is
`62090d9b78ce01ac07073d84b5d2481a5478ea7409ef7e080e91439ba8a22cda`.
No full campaign was restarted. All diagnostics and the one bounded trial below
are complete; no candidate is approved or deployed. Preserve28934 and28936.

### Reproduction And Two Distinct Mechanisms

`audit_joint_sdf_bilateral_hops.py` reproduced all80 original recorded frames
with feature and Root difference0 in `bilateral_hop_audit_v2`. GPU-side command
calculation matters: the earlier v1 CPU scalar route differed by2.57e-5. Actual
command is[3.026415348,-0.063034154]m/s. Do not claim a different checkpoint or
the visualizer caused this recording. Root displacement and displayed FK are
already present in the unmodified saved actor output.

1. Generated-history feedback loses alternating gait. Over the last32 frames,
   source foot fore-aft separation RMS is24.56cm, versus7.66cm generated;
   separation<10cm occurs in34.375% versus84.375% of frames. These are gait
   descriptors, NOT force contacts or proof of a physical airborne jump.
2. The decoder reconstructs the first three already-observed Root velocity
   intervals and integrates them from the first history position. Their error
   shifts the first newly published frame at every8-frame replan. The exact
   identity is `new_step = decoded_past_displacement - factual_past_displacement
   + decoded_outgoing_velocity/30`. At played frame56 the forward-X components
   are12.743cm =3.183cm past error +9.560cm outgoing step. It is not literally
   counting the correct physical displacement twice; the reconstructed past
   error is carried into the future. No offset has been subtracted at runtime.

Code: `local_root_local_body.py:113` integrates velocities;
`motion_inference.py:480` decodes/integrates the entire local window.
`terrain_decoder_pilot.py:62` masks all first4 decoded frames from direct
reconstruction, consumed by `motion_vqvae.py:235`. Future world/geometry losses
can still indirectly constrain prefix errors; do not claim zero total gradient.
The omission concerns decoded Root, not the separate Root planner loss.

### Factorial Offline Tests Pinpoint The Gait Failure

`audit_joint_sdf_gait_chain.py` / remote `audit_joint_sdf_gait_chain_v2.py`
compares source versus actual generated history, then separately substitutes
source tokens or source Root in the captured final decoder request. Default
decoder replays are checked against actor outputs. All oracles are independent
diagnostic windows and NEVER advance the actor history or enter deployed code.

The full real source supplies113 frames from source frame214. Its first97
frames match `context_pilot_data_v3` exactly, so final oracle windows use real
tail frames rather than padding. The generated80 frames still exactly match the
user's recording. `gait_stair_v2/complete.json` reports:

| History and decoder inputs | Last32 close-feet fraction | Last32 fore-aft RMS | Mean relative joint error |
| --- | ---: | ---: | ---: |
| Continuous actor |84.375%|7.66cm|15.47cm|
| Source history each window, actor tokens/Root |12.5%|24.32cm|7.39cm|
| Actor history/Root, source tokens |21.875%|28.95cm|8.94cm|
| Actor history/tokens, source Root |100%|3.72cm|13.78cm|
| Actor history, source tokens and Root |53.125%|12.44cm|7.06cm|

This establishes a large contribution from pose-token prediction under drifted
history. The decoder CAN express alternating poses with different tokens under
the same actor history/Root. Root correction alone does not restore alternation.
It does not prove that only Pose weights are wrong: upstream conditions, token
selection and decoder conditioning remain coupled. Source histories are not a
usable live reset. Source-token truth requires future motion and is not a fix.
Fixed-time reference error also includes phase/progress differences; a lower
close-feet fraction or MPJPE alone is not a terrain quality pass.

`gait_flat_v2` replays the prior flat counterfactual exactly as well. The same
stair-approach initial body/command produces100% close feet in the last32 frames
on a measured-height plane. This excludes staircase sensing as a NECESSARY
trigger in this case. Its reference remains a STAIR source; source-token/Root
substitutions there are not validated flat demonstrations or quality tests.

### Why SDF-Only Training Cannot Repair The Flat Base

`gait_chain_v1` uses two actual BONES train run records, selected only from source
speed/straightness among23 eligible rows. They are
`jog_ff_loop_180_R_002__A048` and `jog_ff_loop_180_R_003__A052`, mean2.56/2.40m/s,
NOT exactly3m/s. Neither reproduces the late close-leg collapse (3.125%/9.375%
close feet), so do not generalize the failure to every run or every start state.

Both prove the flat cancellation contract: SDF memory equals flat memory exactly;
enabled/disabled actor difference0; encoder/embedding gradients0 and largest
SDF attention parameter gradient7.39e-9, consistent with floating-point backward
cancellation. `JointSDFController._condition` adds `condition(actual)-condition(flat)`
and `freeze_base()` freezes all other parameters. Flat examples therefore supply
no meaningful learning signal to repair the frozen flat controller. Previous
idle/walk preservation checks did not establish recovery of this3m/s start state.
This is a training-scope limitation, not evidence that the user selected a bad
slider or that more SDF-only steps will solve it.

### Completed Bounded Decoded-History Root Trial

`decoded_history_root_v1`:300 updates,batch4,1200 examples, same SDF-only A start,
same source-history schedule as archived `context_source_v1`. Only SDF modules
train. `decoded_history_root_objective.py` adds loss on the first3 factual Root
intervals and4 known heights, never outgoing velocity3 or future/body/token
replacement. It reuses the native local-Root coefficient, not retuned geometry.
Two local tests and8-update A100 preflight pass; source-target mutation/actor
parity0, ST forward9.66e-6, schedule/exposure/baseline and frozen hashes verified.
Snapshot `code_joint_sdf_history_root_v1` is immutable; state complete. Do not rerun.

Final full screen only3/17 (1/15 stairs plus2/2 flat); retained official start
review0/8. Held8030 still has sole
P99=12.59cm and body peak12.16cm. This candidate is NOT adopted. In the exact
reported80-frame command, last32 close-feet fractions are original84.375%,
matched control15.625%, new loss9.375%. Only15.625->9.375 is the incremental
comparison; do not attribute the entire original-to-new gain to the auxiliary
loss. Mean past-offset norms0.973cm control versus0.933cm new are also modest.

### Artifacts And Next Work

Local JSON reports are under `out/model_terrain_v3_20261011/joint_sdf_20261012/`:
`bilateral_hop_audit_v2-report.json`, `reported_hop_compare_v1-report.json`,
`decoded_history_root_v1-report.json`, `gait_chain_v1-report.json`,
`gait_stair_v2-report.json`, `gait_flat_v2-report.json`.
Plots `bilateral_hop_audit_v2.png/pdf` and `gait_stair_v2.png/pdf` were rendered
and visually inspected. They are diagnosis, not evidence of a usable new model.
`export_joint_sdf_hop_reviews.py` packages original/source/matched-trial and
explicitly labelled offline oracle SOMA animations as `hop_export_v1`, exclusively
through user-owned private Drive. All51 receipt-listed files were downloaded
using the existing7897 proxy and hash-verified. The remote export SSH session
closed before its success line was returned, but the exporter/rclone processes
had finished and the complete local receipt verifies the uploaded content.
No recovery/re-upload was needed, no public/shared OAuth client was used.

Local official SOMA comparison was inspected at zero-based frames55/59
(display56/60). The actor shows close-leg crouching, while the offline source-token
arm can separate the legs under the same actor history/Root conditions. Both
retain visible geometry problems; it is NOT a usable repaired actor. Screenshots:
`hop_export_v1/actor-vs-offline-tokens-frame55.png` and `-frame59.png`.
The new background tab is a marked deliverable; existing user tabs are preserved.
Viewer:
http://127.0.0.1:28936/web/decoder_pilot/?data=/out/model_terrain_v3_20261011/joint_sdf_20261012/hop_export_v1/visuals/&clip=reported-stairs&left=actor&right=offline_source_tokens

Next work must target phase-preserving recovery in the causal history->Root/Pose
tokens->decoder chain, including direct supervision of decoded factual Root
intervals. First establish feasible recovery targets for actual generated
states and measure the interaction of pose tokens with Root conditioning.
Do not infer that unfreezing Pose alone is sufficient: earlier bounded Root/Pose
trials failed broad checks. Do not invent runtime Root anchoring, smooth the
output, increase SDF weights, or restart a large campaign. Preserve original
stairs, continuous flat starts/runs and complete terrain traversals in testing.
User's restored7897 proxy works; never repeat the rejected27891 proxy action.

## Completed: PARC Recovery Bridge Audit And Local Visuals Available

The user answered that a working global proxy is enabled. All FOUR pending
private Drive exports are now downloaded locally and every receipt-listed file
SHA256 is verified:backbone_export_v1(21),chain_context_export_v1(41),
context_data_export_v1(9),context_export_v1(76), total147 files. Earlier connection
probes still timed out transiently, but the rclone transfers completed through
the existing7897 proxy. No temporary proxy was started, no system settings were
changed and no public/shared OAuth client was used. Do not keep treating the
old proxy question as pending or ask the user to enable it again.

The new audits below are complete. No MB optimizer updates, new training or
deployment occurred. Old28934 WASD and28936 review service remain available.

### Visual Inspection

Actual official SOMA meshes loaded and were inspected in the local browser:

- Latest two trained arms,8030 route idle-start72-frame recording, frames43/59:
  teacher_history and generated_history are visibly distinct and both retain
  crouched/close-leg poses. Screenshot:`context_export_v1/idle-comparison-frame59.png`.
- Raw source vs SDF-only A constant-input8030 route, frames35/59: source has a
  stepping pose near the platform while generated motion remains more crouched
  with closer legs; differences in temporal progress are also visible. This
  qualitative observation is not a claim to infer contact force from a screenshot.
  Screenshot:`chain_context_export_v1/source-vs-constant-frame59.png`.

All locations above are under local
`out/model_terrain_v3_20261011/joint_sdf_20261012/`. The user closed the initial
temporary comparison tabs while returning to old WASD; a source-comparison tab
was subsequently opened in the background and marked as a deliverable. Do not
close or replace their active WASD tab. Linkable review pages:

http://127.0.0.1:28936/web/decoder_pilot/?data=/out/model_terrain_v3_20261011/joint_sdf_20261012/chain_context_export_v1/source_context_visuals/&clip=generated_8030ae7a2a7ad640&left=source&right=constant_approach

http://127.0.0.1:28936/web/decoder_pilot/?data=/out/model_terrain_v3_20261011/joint_sdf_20261012/context_export_v1/visuals/&clip=generated_8030ae7a2a7ad640-idle-72f&left=teacher_history&right=generated_history

Context comparisons with DIFFERENT actual lengths remain separate recordings,
not aligned/padded as if they shared a source clock. Select a same-length entry
such as the inspected idle72 case for a direct two-arm display. Raw source has
its own reference entries; teacher_history is a trained model, NOT source playback.

### SOMA To PARC Is Not An Exact History Interface

`audit_parc_recovery_bridge.py` inverts the existing calibration's position and
orientation task equations analytically and checks the official15-body PARC FK
and joint DOFs. It uses no IK, root fitting, terrain correction or projection.
Test on deliberately constructed native PARC task data recovers FK at<1e-6m and
orientation/DOFs<0.001deg (1 test passed), ruling out a simple transpose/scale bug
in this analytic task inversion.

On actual SOMA histories from all three source clips and context-feedback model,
simultaneously preserving mapped orientations yields mean position-task/FK
residual4.2-5.1cm, maximum9.86-10.71cm, with the feet at the maximum. Generated
late-history DOF residuals reach10.71deg; source-retargeted histories also have
nonzero residual. Fifteen additional SOMA articulations have no independent PARC
body state. These are inverse-task/FK incompatibilities, NOT measured optimized
retarget roundtrip errors and NOT proof that no learned bridge could exist.
Current four predicted SOMA foot channels also do not supply fifteen factual
PARC body-contact channels. Do not copy same-time source contacts onto a different
generated pose or label an approximate native body as the identical played state.

Report:`parc_recovery_bridge_v1/complete.json`, local corresponding`-report.json`.
All raw analytic history candidates remain remote and are NOT admitted as labels.

### Bounded Native PARC Continuation Check

The source's actual native recording is available at
`base/parc_long_route_batch32_finished_v1/recording/export/recordings/0000_long_raw_c002_dm.pkl`,
SHA256`8cf3e5a06f181db8b2a9936178776c3e6ea7203435b6475f065429ae498fb640`.
`prepare_parc_native_recovery_probe.py` selected original frames240/241 and their
real15-body contacts; same-time recorded direction is[0.999603,-0.028168]. The
unmodified paired heightfield is used. Future native frames are diagnostic
references only. Prefix input SHA256`fed1fb0b2f90658305d562a6107e2c9252215047c023fbc0ca3b180bba8da856`.

Pinned checkpoint:`base/parc_runtime_v1/wide_v1/out/parc_legacy_models_20261008/demo_iter3_verified.ckpt`,
SHA256`a03a9d4cc7b3a43b217d67116429d241c92e31e7de93e02849832b0371860e2e`.
The actual cfg/extra contract is30Hz,2 history frames,seq_len15 (13 NEW frames),
EMA,XY_DIR,relative-to-root Z. Do not infer16 frames from0.5s plus an endpoint.
Generation uses official DDIM stride10/CFG0.65 with a single four-row noise draw.
This is a native kinematic capability check, NOT recovery from the SOMA actor.

An upstream side effect is verified: MDM.predict_x0 changes both
PREV_STATE_NOISE_IND_KEY and PREV_STATE_FLAG_KEY to false for the unguided branch
and does not restore them. gen_sequence then reads the mutated flag and returns
regenerated history despite the caller requesting factual history. Raw prefix
root-position discrepancy is4.08mm; joint quaternion-component discrepancy0.145
and contact-component discrepancy0.0888. Do not confuse quaternion components
with rotation degrees or treat this as the cause of MB's terrain failures.

In ISOLATED diagnostic wrappers only, preserving/restoring these flags around
predict_x0 gives prefix Root/contacts difference0 and quaternion-component
differences5.96e-8. Future generation agrees exactly with an unmodified REPEAT
run. Both differ from the initial run by the same floating-point scale
(Root<=9.54e-7m); thus do not claim strict bitwise determinism to the first run.
The installed PARC vendor, prior datasets and MB runtime are unchanged.

`probe_parc_native_continuation.py` output`parc_native_continuation_v4` contains
all three matched calls (released, released_repeat, preserved flags),four
candidates per call, and the raw source. Each native future advances1.096-1.183m
and rises0.414-0.494m across13 new frames. Native304-point surface peaks on
CERTIFIED queries are0-3.32cm, with only78.6-90.1% nearest-surface certificates;
these partial sparse metrics are NOT a complete collision pass. No simulation,
SOMA skin conversion, candidate selection or training admission was performed.
Report and all arrays are retained; local`parc_native_continuation_v4-report.json`.

Failure records:v1 rejected the incorrect16-frame preflight before generation;
v2 discovered prefix alteration and stopped before persisting its outputs; v3
saved both raw outputs but strict zero-difference future assertion failed at
floating-point scale. V4 retains raw outputs BEFORE assertions, adds unmodified
repeat control and records the difference honestly. Do not count failed-script
attempts as accepted labels or silently overwrite their logs.

### Next Work

PARC can continue its own compatible native history, but the tested direct
SOMA-to-PARC inverse is not an acceptable identical-state recovery teacher. Do
not silently fit/project the played body, invent unobserved contacts, apply
future Root replacement or expand recovery generation into a dataset. Any
future bridge must measure the actual two-way SOMA prefix/skin continuity.

The original issue remains native MB learning with off-reference feedback.
Review a SOMA-space recovery objective or conditional generator that preserves
the factual history and observed terrain, with a bounded demonstrable feasible
target before training. Keep fixed-time and source-history baselines, complete
routes, flat controls and old official staircase failures. The prior optional
factual Root-target helper and its parity tests remain available but unadopted;
it is not sufficient evidence to restart training. Use the now-available local
visuals to inspect candidate changes before further architecture/loss changes.
No new full-budget authorization and no deployment quality approval.

## Completed: Reference Matching, Codec Support And Factual Root Targets

All four new audits below are COMPLETE. No optimizer updates, new training or
deployment occurred. The earlier matched300-update arms remain failed and must
not be restarted. Immutable snapshots remain unchanged; a separate
`code_joint_sdf_target_audit_v1` holds only the optional Root-target helper.

### Reference-Time Matching Does Not Yet Supply Recovery Targets

`audit_joint_sdf_reference_alignment.py` replayed the initial SDF-only A and
finished context-feedback checkpoints on the same three stair passages. Actor
commands remain at the original clock. Offline candidate selection scores only
the factual four-frame history against unchanged world-space source windows;
selected source indices NEVER enter actor inputs or affect the commands.
Every actor call was repeated after matching with exact output parity. All
26 factual source-history windows select their original index (both models).

Compared fixed time, nearest Root history, nearest full-pose history, monotonic
full-pose matching with at most8-frame advance, and nearest full pose with an
additional32 real source frames. All windows retain33 actual frames. There is
no source translation, pose interpolation, invented tail or actor correction.

For the feedback checkpoint, next-step displacement mismatch mean (cm) and
missing PUBLISHED reference target frames through the goal:

| Route suffix | Fixed time | Nearest pose | Advance<=8 pose |
| --- | --- | --- | --- |
|415ccb3cd0a84f27|12.14 /0 missing|6.63 /2 missing|7.37 /0 missing|
|babca93468f737e6|26.34 /0 missing|19.84 /4 missing|21.05 /3 missing, goal not covered|
|8030ae7a2a7ad640|33.23 /0 missing|17.31 /12 missing|22.18 /0 missing|

The last capped case still has86.32cm maximum step-displacement mismatch.
Adding real tail candidates lowers that case's mean to12.42cm but leaves13
reference frames uncovered, including its goal frame. These are target-coverage
measurements, not a change to actual physical traversal. Nearest-time matching
therefore is NOT adopted as a fix or permission to omit difficult motion. The
uncapped matcher is a diagnostic lower-distance comparison, not a trainer.
Source-history identity, monotonic bounds and invalid-input tests:3 passed.

Remote:`joint_sdf_20261012/reference_alignment_v1/complete.json`.
Local report:`out/model_terrain_v3_20261011/joint_sdf_20261012/reference_alignment_v1-report.json`.
Local inspected PNG/PDF:`reference_alignment_v1.png` /`.pdf` in that directory.
The chart is an offline label diagnostic, not a new model-motion visualization.

### First Token Is Not Pure History

`audit_joint_sdf_codec_time_support.py` uses the actual trained encoder and
decoder. Structural convolution/residual/norm support covers all32 input frames
for each of the8 tokens. Measured first-token gradients are nonzero at every
future frame4..31 in all three windows. Replacing ONLY source local frames24..31
with another real window's features changes3/1/1 of the first token's8 code
heads. This splice is an OFFLINE perturbation test, never a training sample.
Changing input history0..3 can also change later tokens, as expected from the
bidirectional convolutional receptive field.

The native decoder, replayed on its actual already-conditioned latent and
original masks, exactly matches its saved output. Its published frames4..11
have nonzero derivatives with respect to latent position0 (12.4-18.6% of summed
position gradient norms in this probe). This is a sensitivity measure, not a
quality or causal-attribution score. The existing first token must not be
blanket-masked as a supposedly pure-history label, and replacing only the source
body prefix before encoding would affect future token labels as well.
Theoretical-support code was checked against real strided/dilated convolution
gradients (1 passed). Actor output replay is unchanged.
Remote:`codec_time_support_v1/complete.json`; local corresponding`-report.json`.

### Optional Factual-Past Root Target Correction

New helper:`motionbricks/data/terrain_root_supervision.py`, version
`factual-global4-local3-v1`. It constructs a LOSS-SIDE Root batch using observed
global Root frames0..3 and observed local intervals0..2. All future global labels
from4 onward and local labels from3 onward remain byte-identical to the original
source targets. It does not change source pose/token labels, transform source
geometry, create a recovery transition or alter inference. It is OPT-IN and is
NOT wired into existing runners or any deployed model; a future experimental
runner must explicitly record its version in the training contract.

Four tests cover None/pelvis/terrain vertical frames, source-history loss AND
gradient equivalence, changed factual history, future-target preservation, no
mutation/label gradient and actor parity. All pass. Actual A100 verification
`audit_joint_sdf_factual_root_targets.py`, output`factual_root_targets_v1`, checks
all26 real windows: source-history loss difference0, Root-output gradient
difference0, actor replay difference0, optimizer updates0. Thus the helper is
implemented and verified, but has NO demonstrated learned-motion improvement.
It deliberately leaves the future-boundary recovery problem unresolved.

### Geometry Objective Compatibility

`audit_joint_sdf_feedback_objectives.py`, output`feedback_objectives_v2`, uses the
finished feedback checkpoint and original coefficients. For the first new frame,
absolute FK has zero error at`source_joint[4]`; the boundary-speed term has zero
error at`played_joint[3] + source_joint[4] - source_joint[3]`. Their separation is
exactly`played_joint[3] - source_joint[3]`. It is approximately0 at factual starts
and mean9.27-78.14cm across the six later feedback probes (max joint88.92cm).
V1 stopped at a3.8e-6m float32 cancellation in a50m scene; V2 verifies this vector
identity in float64 while retaining original float32 model/loss computation.
This is numerical diagnostic repair, not a change to geometry gates.

The two terms' gradients at the actual first output frame are NOT consistently
opposed: cosine ranges0.054..0.989 on the six later probes. With existing weights,
boundary output-gradient norms are7.5-23.6x the first-frame FK norms; some
collision/boundary gradients oppose. These are OUTPUT-space derivatives, not
parameter-space magnitudes or evidence to blindly rescale/remove a term. The
different zero-loss endpoints show incompatible exact recovery targets, but do
not establish that a compromise cannot be learned or that one term is the sole
cause. Keep original weights and baseline until a recovery objective is tested.
Remote:`feedback_objectives_v2/complete.json`; local corresponding`-report.json`.

Next review feasible recovery supervision from the actual generated state,
including whether a scene-conditioned source/expert can produce a real recovery
continuation. First demonstrate target feasibility and prefix/body/terrain
consistency in a bounded case. Do not replace this with nearest-frame skipping,
root-shifted source, synthetic boundary teleports, a pure-history token mask,
unvalidated gradient balancing or another full-budget train. Any source/feedback
comparison must retain complete physical routes, flat cases and the old official
failed staircase. Read the current generated-history loss path and source
generator contracts before another training change. The optional Root correction
is available for that controlled experiment, not evidence that it is sufficient.

Local7897 proxy still times out; no rejected27891 proxy action was retried and
no duplicate user request was made. Four earlier visual exports remain on private
Drive awaiting local download/receipt verification/inspection. All new raw motion
history NPZs remain remote; only compact JSON diagnosis came over SSH. Old28934
and28936 services remain intact. No new WASD candidate is approved.

## Completed: Full-Context Pair; Feedback Target Consistency

`context_source_v1` and`context_feedback_v1` both completed300 actual updates,
batch4,1200 primitive examples per arm. Training took134.8/136.8 seconds excluding
baseline/final reviews. Both supervisors completed the new-route and retained
official-stair reviews. No active training or diagnosis remains; no deployment.
Do not restart them or the earlier160万-example campaign.

### Controlled Result

Snapshot:`code_joint_sdf_context_v1`, unchanged after launch. Entry points:
`scripts.run_joint_sdf_context_pilot --arm source|feedback` and
`scripts.pilot_joint_sdf_full_context`. State files:
`joint-sdf-context-source-state.json` /`joint-sdf-context-feedback-state.json`.
Both initialize from`decoder_ab_A/weights.pt`; only SDF adapters are trainable.
Three real stairs occupy three batch rows; the fourth alternates original
BONES idle/walk episodes. This targeted75/25 primitive mix is NOT the broad
campaign's50/50 episode distribution. Per-row cursors traverse8/9/9/13 primitives
independently. All source windows, command values, cursor schedules and exposure
counts match between arms (verified schedule hash). The sole intervention is
whether the next history comes from the source or the actual generated motion.

Commands use the CURRENT recorded direction times fixed weak approach speed.
The initial observed body/travel offset is retained with native yaw sign; no
future source pose sets facing. Held-forward,0.8x/1.2x speed and idle starts are
separate evaluations. These remain weak speed labels, not recorded m/s input.
Two command tests passed, including clockwise heading and future-pose mutation.
The existing future-target/inference-mode test also passed. Actual A100 preflight
after8 updates verified future Root target mutation changes actor output by0,
runtime/no-grad difference0, train-ST forward maximum difference9.66e-6,
finite SDF gradients and no frozen-module gradients. Final frozen hashes match.
Both initial batch outputs are exactly equal; complete baseline metrics agree.

| Evaluation | Source-history arm | Generated-history arm |
| --- | --- | --- |
| New stair goal planes reached |15/15|15/15|
| New complete stair quality screen |0/15|0/15|
| Flat controls |2/2|2/2|
| Original official stair screen |2/8|0/8|

The new screen retains the P99 criteria AND requires all18056 skin vertices'
sampled exact-prism maximum<=2cm. This is stricter than the old P99-only screen;
old official metrics/gates are unchanged. Counts are not interchangeable.
Every actual trajectory and failed case is retained. Reaching a goal does not
override penetration/cross-track/height/unknown-map failure.

Held-forward results (cm):

| Route suffix | Baseline sole P99 | Source-history P99 / body peak | Generated-history P99 / body peak |
| --- | --- | --- | --- |
|415ccb3cd0a84f27|4.55|3.61 /4.59|6.82 /7.48|
|babca93468f737e6|9.22|0 /0.69|5.86 /9.40|
|8030ae7a2a7ad640|5.06|2.60 /6.24|12.02 /12.97|

The low-penetration source-history middle case deviates44.7cm sideways and
therefore still fails. Flat outputs/metrics are unchanged. Completing the real
tail and adding full generated-history exposure did NOT solve terrain response;
do not adopt either arm, expand its budget, or declare coverage the sole cause.
Local pair report:`out/model_terrain_v3_20261011/joint_sdf_20261012/context_pair_summary.json`.

### Read-Only Loss / Target Audit

`audit_joint_sdf_root_history_loss.py` ran on the finished feedback checkpoint,
output`root_history_audit_v2` (v1 retained). It replays three real trajectories
and inspects positions0,3,last. Native Root loss includes all32 source frames,
while decoded-motion supervision masks history. At generated histories, observed
Root and source past can differ by6.3-68.4cm. This is an inconsistent past target,
but its sdf_root gradient norm is only0.071-0.458% of the future-global-Root term
on the six off-source probes. Do not exaggerate this term into the main cause.
Gradient comparisons cover only these parameter/term subsets, not the total loss.

The stronger target-feasibility signal is fixed SOURCE TIME versus actual motion
progress. For route8030..., feedback position3 asks the next frame to be55.6cm
ahead of current Root (the real source step is10.0cm). At position8 the next
source target is69.8cm BEHIND current Root and the end of the next8-frame block
is50.4cm behind, despite a forward command; the genuine source step is+2.37cm.
Routebabca... also has a next-frame target7.03cm behind at its last position.
These are target locations, not replacement runtime Root positions or a claim
that the actor literally moves backward. This establishes a recovery/temporal
pairing problem to investigate, not proof that automatic time alignment fixes it.

Also reviewed the native MaskGIT loop: it replaces ALL token IDs from each pass's
logits, including the final pass. The hypothesis that retained early IDs were
being differentiated through unrelated last-pass logits is not supported here.

Next investigate feasible feedback supervision and reference-time alignment with
the current fixed-time baseline retained. Any proposed alignment must be training
only, use real source frames at unchanged world coordinates, expose selected
indices/coverage, preserve the full physical route and never reset evaluation
or alter generated Root. Do not silently move source geometry with prediction,
label corrective teleports as motion, infer first-token temporal support without
checking codec receptive fields, or launch a new large budget. Repair the past
Root-target inconsistency only with source-history equivalence and causal tests;
its small measured gradient means it is not sufficient evidence for another
standalone training campaign.

`context_export_v1` completed private Drive upload:35 final comparison entries
and22 initial/source entries, all official SOMA, actual durations and hash receipt.
Different durations remain separate recordings; no stretching or last-frame fill.
Teacher-history model label is`teacher_history`, distinct from raw`source`.
All four new exports remain undownloaded locally because existing7897 proxy
still times out. The pending user proxy question is unchanged; no rejected
temporary proxy action was retried. Preserve28934/28936 and do not call these
results locally visually verified. Compact JSON reports were retrieved via SSH.

## Completed: Exact Surface Audit And Full-Context Pilot Preparation

The exact-surface audit and real-tail data preparation are COMPLETE. No new
training was launched. Root/Pose and every earlier trial remain completed and
unapproved. Preserve all original failures and do not restart the160万 budget.

### Correction To The Source Penetration Finding

`audit_joint_sdf_exact_surface.py` evaluates identical18056-vertex skin and
8-substep linear sweeps using the existing distance volume AND exact prism SDF.
The distance-volume result reproduces the archived nine source/actor measurements
within1e-5. Negative-only exact queries agree with full signed queries on a
deterministic verification subset. Every sample is within known terrain; no
interior distance needs fallback. The original raw NPZs and metrics are unchanged.

| Source suffix | Source peak: volume / exact | Constant-command actor: exact P99 / peak |
| --- | --- | --- |
|415ccb3cd0a84f27|7.26 /1.07cm|2.89 /8.58cm|
|babca93468f737e6|4.55 /0.96cm|3.50 /9.07cm|
|8030ae7a2a7ad640|4.38 /1.03cm|8.91 /13.20cm|

All three source exact P99 values are0. Therefore the earlier4-7cm source-skin
claim is NOT confirmed by exact geometry; most of it was volume approximation
bias. `TerrainDistanceVolume` uses a10cm vertical grid, with interpolation around
riser edges. This does NOT explain away the actor failures, which remain large
under exact measurement. The completed Joint SDF training and small pilots use
exact negative-prism collision losses already, not this volume approximation.
No loss-weight change or completed-training restart follows from this finding.

The source-speed-schedule actor has exact peaks2.69/1.73/0cm; the first was
previously missed entirely by the volume. Its already-recorded goal-plane
shortfalls remain unchanged, so it is still not a complete traversal success.
Exactness is relative to the observed piecewise-constant heightfield solid,
not the original PARC triangle scene; sampled linear skin sweeps are not
continuous collision detection. Report:`joint_sdf_20261012/exact_surface_v1/complete.json`,
local`out/model_terrain_v3_20261011/joint_sdf_20261012/exact_surface_v1-report.json`.

### Concrete Next Pilot Data

`prepare_joint_sdf_context_pilot.py` produced`context_pilot_data_v3/cases.pt`:
three true stair passages plus the original BONES idle/walk references. Every
stair training history can now reach the source-defined end-of-plateau goal,
while its prediction still has a full33-frame factual target window. No repeat,
padding, retiming, terrain modification or invented post-ascent pose was used.

| Source suffix | Real full-source interval, end exclusive | Added real tail | Segments | Last published source frames |
| --- | --- | --- | --- | --- |
|415ccb3cd0a84f27|323..412|7 frames|8|383..390 (goal387)|
|babca93468f737e6|874..971|5 frames|9|942..949 (goal948)|
|8030ae7a2a7ad640|214..311|11 frames|9|282..289 (goal282)|

The entire extended source, including initial history and added tail, passes
the exact18056-vertex/8-substep source audit with peaks1.07/0.96/1.03cm and no
samples over2cm, unknown columns or uncertified interior distances. This is
geometry validation, not a visual or actor-quality approval.

Recorded direction, contact and encoded motion SHA256 identities were checked;
intent/contact source-frame arrays match exactly, with matching30Hz lengths.
The incremental dataset writer stores frame IDs in sidecars, not the motion
shard. Preparation V1/V2 assumed absent metadata and failed before creating
cases; their logs are retained. V3 follows the actual schema. No labels changed.
Recorded direction changes by5.08/13.77/4.89deg across these passages; especially
the middle case is not perfectly constant direction. Directions have unit norm,
not m/s. Weak approach speeds2.95/2.64/3.03m/s compare with actual ascent medians
2.87/2.20/2.32m/s. Never equate these weak estimates with recorded speed intent.

Planned bounded comparison: identical initial SDF-only A checkpoint, full real
source windows, episode cursors, commands, example/update counts and original
losses; compare source history against actual generated-history feedback over
the complete passage. Keep Root/Pose/codec frozen for this isolated test. Retain
the original failed official85-frame staircase as additional evaluation. Review
the differing recorded directions before fixing the weak control protocol, and
verify target/actor separation and default inference parity in a small preflight.
Do not automatically expand to full-model or full-budget training.

`context_data_export_v1` contains three official SOMA source clips, prepared cases,
exact reports and SHA256 receipt. Export and upload to the user's private Drive
completed. All three new exports (`backbone_export_v1`, `chain_context_export_v1`,
`context_data_export_v1`) remain undownloaded/unviewed locally. Existing7897 proxy
still timed out in this follow-up; the already-rejected27891 helper was not
retried. User's existing asynchronous global-proxy question remains pending;
do not ask again or block independent remote research on the download.

## Previous Work: Feedback Coverage And Command / Motion Timing

All Root/Pose probes and the read-only diagnostics below are COMPLETE. There is
no new training run. Do not restart the160万 campaign or automatically start a
joint full-model run. The original objective remains unresolved; no deployment.

### Native Root / Pose Probe Results

Both300-update/3600-example arms pass only1/8 start/speed cases and1/4 original
pilot stair cases. Root passes left20cm; Pose passes idle-default. Frozen-module
hash checks and initial-baseline parity completed. Flat idle/walk sole P99 is
0.17/0.55cm for Root and0.17/0.43cm for Pose. Neither is adopted.
`backbone_export_v1` is uploaded to the user's private Drive, but has NOT been
downloaded or visually reviewed locally in this follow-up.

### Read-Only Chain Trace

`scripts/trace_joint_sdf_chain.py` ran against immutable
`code_joint_sdf_backbone_v1`, output`joint_sdf_20261012/chain_trace_v2`.
One refinement directly calls the archived actor; its output is exactly equal
to saved SDF-only A, Root and Pose final rollouts (maximum difference0).
The earlier`chain_trace_v1` failed on snapshot/slider interface mismatch; its
log is retained. No live controller or training snapshot was patched.

| Model | Query-to-final mean, 1 / 2 refinements | Sole vertical P99, 1 / 2 |
| --- | --- | --- |
| Completed160万 model |3.89 /0.45cm|45.68 /45.37cm|
| SDF-only A |5.32 /2.23cm|1.70 /23.07cm|
| Root probe |3.19 /1.14cm|3.37 /8.02cm|
| Pose probe |3.72 /1.28cm|7.59 /17.52cm|

Extra re-query reduces proposal mismatch but does not repair stair motion. Do
not adopt it. Exact vector identities separate inherited root error, new Root
plan error and decoder/Root discrepancy; RMS values are NOT additive. With the
default actor, Root-to-decoded-root RMS is6.68-10.00cm across these four models,
and relative-foot RMS against the same-time reference is11.48-31.26cm. These
reference errors include timing differences, not just geometric invalidity.
Original final model records72/80 actual frames and the unknown-map failure;
other arms record80. No frames were filled or hidden.
Local compact report:`out/model_terrain_v3_20261011/joint_sdf_20261012/chain_trace_v2-report.json`.

### Factual Feedback Coverage Gap

`audit_joint_sdf_feedback_coverage.py` exactly replays the pilot's seed2333 window
draws:150 single-primitive updates and150 seven-primitive updates,3600 examples.
For the85-frame official stair, source-clock history coverage ends at55;
generated-history coverage spans8..55. Evaluation blocks7/8/9 consume histories
56..59,64..67,72..75, none used as training history. Geometry target coverage is
4..79, with ZERO geometry supervision at80..83. The first riser is at47.
This is a real coverage limitation of these small trials, not proof of the sole
failure cause and not an audit of the earlier large campaign. Do not invent or
pad source tails. Report:`joint_sdf_20261012/pilot_feedback_coverage.json`.

### Longer Real Source / Unchanged Actor Review

`audit_joint_sdf_stair_sources.py` screened all526 training PARC full references
for measured flat approach, repeated rises and real post-ascent context.
This broad geometry screen includes boxes/platforms and is NOT a stair-quality
certificate. Three generated seven-riser routes were then examined with the
unchanged SDF-only A model in`source_context_v2`:

| Source suffix | Constant-input sole P99 | Source-speed-schedule sole P99 |
| --- | --- | --- |
|415ccb3cd0a84f27|5.85cm|0.93cm|
|babca93468f737e6|6.34cm|0.47cm|
|8030ae7a2a7ad640|12.74cm|0cm|

Source-speed schedule is an OFFLINE weak planar control diagnostic, not recorded
WASD and not autonomous speed adaptation. No future height or pose enters actor.
Additional source-defined goal-plane audit uses the end of a16-frame plateau:
all constant-command runs cross it but penetrate; all schedule runs remain
20.5-37.3cm short at the recording endpoint. Do not call the low-penetration
schedule a completed-route success. Originals and actual generated durations
are retained. Goal report:`source_context_goals.json`.

The18056-vertex,8-swept-substep volume audit also finds source animation peak
penetration7.26/4.55/4.38cm (P992.37/0.90/2.30cm). These are distance-volume
measurements with heightfield fallback, not exact continuous-mesh SDF proof.
Next check exact prism distances and inspect source visuals before using these
as clean training exemplars. The recorded unit-direction signals still lack
verified m/s semantics. Do not turn outcome-derived speed into claimed intent.

`review_joint_sdf_source_context.py` and`audit_joint_sdf_context_goals.py` are
read-only. V1 review failed on an unavailable pilot import before running;
V2 uses explicit verified bundle loading and completed all nine source/actor
recordings. `export_joint_sdf_chain_reviews.py` creates source/model official
SOMA comparisons and hashes under`chain_context_export_v1` and transfers them
only via private Drive. Export and upload completed successfully (19MiB, six
chain clips including separate incomplete recordings, three source-context
clips). Remote destination:`private-store:motionbricks/joint_sdf_20261012/chain_context_export_v1`.
Additional`source_context_goals.json` is saved separately, including locally.

Local transfer limitation: existing127.0.0.1:7897 times out to Drive. The separate
localhost27891 proxy launch was rejected by automatic approval with only
`blocked by policy`; do not retry that rejected action through a wrapper. No
temporary proxy was started. The stalled local rclone was stopped after checking
its exact cmdline. An asynchronous request asks the user to enable a working
global proxy. Remote diagnostics and private Drive uploads are unaffected.
Preserve28934/28936. New visuals are NOT yet locally verified.

Next: exact source-geometry verification, honest visual inspection after proxy
recovery, then a bounded controlled experiment that actually covers the full
stair passage with real tail context and distinguishes requested speed from
executed speed. Keep the original failed official staircase in all comparisons.
Do not automatically retrain or declare the sample-coverage finding a cure.

## Completed: Native Root / Pose Capacity Test Setup

The source-reconstruction-objective A/B completed200 updates/2400 primitive
examples per arm. Both arms pass only1/8 start/speed cases and0/4 original pilot
stair cases. Auxiliary reconstruction reduces default moving sole P9928.43cm
to7.67cm, but idle start worsens2.31cm->8.67cm and slow worsens7.43cm->12.70cm.
Both still have unacceptable local penetration. Neither is adopted or deployed.
Official SOMA visuals, receipts and complete metrics are downloaded/hash-verified
at`out/model_terrain_v3_20261011/joint_sdf_20261012/sourcecodec_export_v1/`.
The source/auxiliary viewer was inspected at frame59; screenshot is
`sourcecodec-frame59.png`. Temporary download proxy is closed; old WASD is intact.

Viewer:
http://127.0.0.1:28936/web/decoder_pilot/?data=/out/model_terrain_v3_20261011/joint_sdf_20261012/sourcecodec_export_v1/visuals/&clip=stairs-source_history_start&right=auxiliary

### Recorded Intent Audit

`audit_parc_intent_semantics.py` checked86 recorded-intent training clips, every
sidecar SHA256 against both accepted records and the artifact index,30Hz lengths,
strictly increasing source-frame indices, recorded unit-vector norms and the
existing PARC->native coordinate transform. On18,614 moving windows, median
direction difference from the outcome-derived velocity is3.68deg and P9521.91deg.
Restricting to9223 windows with <=5deg command variation gives median2.45deg,
P9513.62deg. The5048 stable-command windows with >10cm source root-support relief
have median2.43deg, P9512.70deg, and no differences >45deg. Therefore widespread
direction reversal is not supported as the dominant failure in this subset.

This says nothing about intended SPEED: the recorded signal is unit direction,
not m/s. Source-derived speed can still confound the desired action with the
executed result. Recorded commands are not yet substituted into training.
The generator capture code records normalized target-minus-history-root before
diffusion; its retained chunk lengths match the inspected upstream append logic.
The builder metadata alone does not prove every source's capture timing.
Report:local`joint_sdf_20261012/recorded_intent_audit.json`; remote same relative
path. No datasets or labels were rewritten during this audit.

### Active Controlled Test

Previous small experiments froze both native Root and Pose backbones. The current
test changes one parameter group at a time, using the completed SDF-only
`decoder_ab_A`300-update run as the matched reference:

- `backbone_root_v1`: SDF adapters plus native Root trainable; Pose and full codec frozen.
- `backbone_pose_v1`: SDF adapters plus native Pose trainable; Root and full codec frozen.

Both start from the identical completed4167-update checkpoint and use the same
seed, selected cases, optimizer/lr, loss coefficients, primitive schedule and
300 updates/3600 examples. No decoder unfreezing, source-codec auxiliary objective
or lateral augmentation. Both initial baseline reports match the archived
SDF-only reference in all fields except timing, with1e-5 float tolerance. Actual
first-step gradients into the selected backbone are required. A new focused test
passes for each arm and confirms the other backbone/codec receive no gradients.
Frozen-module hashes are checked after training.

This is a test of native model capacity, not a claim that unfreezing a backbone
solves terrain adaptation. The inference graph is unchanged: no Root substitution,
projection, IK, foot lock or source future input. Root/Pose continue to learn
their existing native losses and the existing differentiable geometry objective.
This does not reproduce separate-stage official MB training wholesale.

Remote snapshot:`task/code_joint_sdf_backbone_v1`.
Supervisors:`task/run_joint_sdf_backbone_probe.py --arm root|pose`.
Training script:`task/pilot_joint_sdf_backbone.py`.
Outputs:`task/joint_sdf_20261012/backbone_root_v1` and`backbone_pose_v1`.
States:`task/joint-sdf-backbone-root-state.json` and`joint-sdf-backbone-pose-state.json`.
Launch PIDs1097671/1098033 are identity hints only. Physical0/2 were freshly
checked idle. Both arms passed startup parity and completed with finite
gradients. Their supervisors completed the original pilot and8-case start/speed
review. These are historical launch details; do not restart completed runs.

Next collect both results, distinguish improvements in planned motion from
actual decoded motion, check flat retention, body maximum as well as P99 and
full traversal. Export every result through private Drive. Broaden validation
before any deployment, and do not immediately launch another full-budget run.

## Completed: Native Source Reconstruction Objective

All decoder and lateral A/B runs below are complete. Nothing is approved for
live deployment. The lateral200-update arms scored4/8(control) and1/8(augmented)
on the start/speed screen. Augmentation improved the right20cm case but worsened
default/slow cases; it is not adopted. Both arms' final source/generated visuals
and reports are downloaded and hash-verified under`lateral_export_v1/`. Temporary
download proxies are closed; original WASD28934 remains unchanged.

The32-frame native-length versus max-length padded decoder hypothesis is ruled
out on all four diagnostic windows: metric differences are floating-point scale.
Direct`codec(source)` native autoencoding matches the token+Root oracle result.
Bypassing quantization improves some errors but still fails geometry: at start40,
sole P9918.06cm->9.63cm; at start48,17.10cm->12.40cm. This is not evidence that a
continuous controller would solve the problem, and no quantization bypass is live.

A separate native reconstruction capacity probe completed200 updates/600 examples.
It used the existing MotionVQVAEModel.loss on the official encoder/quantizer/decoder
source-reconstruction path, trained decoder only and checked every other module
unchanged. Source Root is an oracle codec condition in this OFFLINE diagnostic.
Its schema is`motionbricks.native_codec_capacity_diagnostic.v1`, not a deployable
Joint SDF checkpoint. At window48 source-token+source-Root reconstruction improves
relative joint error7.20cm->0.97cm, root RMS12.88cm->3.52cm, sole P9917.10cm->4.84cm.
Actual generated-token motion in the same diagnostic still has7.06cm sole P99.
Do not mistake source reconstruction success for causal control success.

Code review found the previous joint decoder adaptation only applies the native
motion reconstruction loss to GENERATED-token outputs. It omits the original MB
source-encoder/quantizer/decoder reconstruction path while changing the decoder.
The new bounded test asks whether preserving that source-code meaning helps;
it does not assume decoder unfreezing alone was the missing fix.

Completed arms: `sourcecodec_control_v1` and `sourcecodec_auxiliary_v1` under the
remote task's`joint_sdf_20261012/`. Both start from the same completed4167-update
checkpoint, with fresh Adam5e-5,200 updates/2400 primitive examples, train SDF+
decoder, and preserve Root/Pose/encoder/codebook. The auxiliary arm additionally
reconstructs those same2400 source examples with the existing native codec loss
at coefficient1.0; no geometric weight change. Compute is therefore not identical.
No lateral augmentation. Source Root/pose futures enter only the training-side
autoencoder reference path; actor generation still sees only actual history,
command, measured scene and model-predicted future. Its own first4 source frames
define the auxiliary boundary, avoiding a generated/source history mismatch.

New focused test passes: auxiliary objective ignores a deliberately corrupted
actor-history boundary, leaves the generated output unchanged, trains decoder
gradients and leaves Root/Pose/encoder/quantizer without gradients. The source
path is based on MotionVQVAEModel.training_step, not a new runtime controller.

Code snapshot:`task/code_joint_sdf_sourcecodec_v1`; supervisors
`task/run_joint_sdf_sourcecodec_pilot.py --arm control|auxiliary`; training script
`task/pilot_joint_sdf_sourcecodec.py`; helper`data/joint_sdf_codec_objective.py`.
State files:`joint-sdf-sourcecodec-control-state.json` and
`joint-sdf-sourcecodec-auxiliary-state.json`. Launch PIDs861694/862361 are only
identity hints. Physical0/2 were freshly checked idle. Inspect actual cmdline,
progress and complete state before any action. Each supervisor automatically
runs the original pilot review and8-case start review. Do not duplicate or expand.

Next compare both arms' honest generation, flat retention, peak as well as P99
penetration and perturbed starts; retain failed cases. Export source/generated
visuals through private Drive. If useful, test other clips before deployment.
No full-budget retrain, no IK/projection/root replacement, no relaxed quality gate.

## Latest Findings: Decoder A/B Complete

Both300-update arms completed3600 primitive examples. SDF-only A is better than
SDF+decoder B on the training staircase. This does not support adopting decoder
unfreezing as the next general fix, and the earlier oracle finding must not be
presented as proof that frozen decoder weights caused the live failure.

| Pilot stair mode | Initial sole P99 | A: SDF only | B: plus decoder |
| --- | ---: | ---: | ---: |
| Moving start, constant approach |45.68cm|1.70cm|7.64cm|
| Idle start |25.29cm|2.19cm|4.38cm|
|0.8x speed |41.55cm|1.07cm|1.90cm|
|1.2x speed |36.45cm|18.67cm|25.71cm|

A passes3/4 original pilot stair screens, B1/4. A flat output remains identical
to its baseline. B changes flat output: idle sole P990.11cm, walk0.58cm versus
baseline0.34/0.41cm. These two flat samples alone cannot certify B flat retention.

A broader unchanged20-case evaluation passes4/20, versus the previous5/20.
Validation boxes worsen from63.0 to132.5cm vertical sole P99; the corresponding
body SDF P99 worsens19.5->27.8cm. A is not approved for deployment.

An additional8-case fixed-command start/speed review stops at the first published
8-frame block endpoint crossing a source-defined goal plane, or a bounded timeout.
It does not truncate/pad motion to a desired frame count or change actor motion.
Initial model0/8, A3/8, B1/8 meet its combined root/path/sole/body/known-map screen.
Even A's default numerically passing case has6.63cm sampled-body maximum
penetration; P99 near zero must not be advertised as collision-free. A shifted
20cm sideways fails, with sole P99 up to21.28cm. One idle failure has a tiny
unknown-body fraction; it remains a failure. These are still the training scene.

All A/B visuals and reports have been downloaded via private Drive and every
exported-file SHA256 verified. Local directory is
`out/model_terrain_v3_20261011/joint_sdf_20261012/decoder_ab_export_v1/`.
Browser source/A checked at frames49 and59; screenshot `stairs-A-frame59.png`.
Viewer:
http://127.0.0.1:28936/web/decoder_pilot/?data=/out/model_terrain_v3_20261011/joint_sdf_20261012/decoder_ab_export_v1/visuals/&clip=stairs-source_history_start&right=A

### Completed Controlled Loop: Lateral Reference Augmentation

Read-only gradient audit at source-history starts24/40/48 found weighted collision
gradient/native-gradient ratios0/8.14%/0.63%. At start24 collision is actually zero.
This is a local measurement, not justification for multiplying loss weights.
No loss or weight changes were made.

Terrain-fixed source translation admission was first checked using566 sampled
skin vertices, then independently confirmed on all18,056 vertices over85 frames.
World-Z offsets[-0.2,-0.1,0,0.1]m preserve every sampled terrain-height lookup and
known occupancy exactly. Offsets-0.3,+0.2,+0.3 alter correspondence and are rejected.
The check concerns these source paths on this raster, not universal scene symmetry.
The source trajectory, source joints and target skin translate together; generated
motion is never translated or corrected during inference.

Two bounded arms start from A's identical weights62090d9b78ce01ac07073d84b5d2481a5478ea7409ef7e080e91439ba8a22cda.
Each uses200 optimizer updates/2400 primitive examples, same data indices, crop
RNG, mask RNG, Adam5e-5 and original coefficients. Only the augmented arm samples
the admitted lateral offsets. Both train SDF only and preserve all base weights.
The augmentation uses its own seeded NumPy RNG to avoid changing sample/mask RNG.

Remote snapshot:task/code_joint_sdf_lateral_v1. Original snapshots are unchanged.
Outputs:task/joint_sdf_20261012/lateral_control_v1 and lateral_augmented_v1.
Supervisors:task/run_joint_sdf_lateral_pilot.py --arm control|augmented.
State:task/joint-sdf-lateral-control-state.json and joint-sdf-lateral-augmented-state.json.
Launch PIDs748266/748830 are hints only; verify current cmdline. Physical0/2 were
freshly checked idle. Each supervisor runs the original pilot evaluation and then
the same8-case start/speed review automatically. Do not duplicate or expand budget.
One new admission test passes and rejects changed support heights/unknown space.

Next: inspect both complete.json and start_review/complete.json, export all
outcomes, compare sensitivity and flat retention, then decide the next change
from evidence. Do not turn training-scene memorization into a generalization claim.
The completed160万 campaign must remain stopped. Old WASD28934 is unchanged.

## Result

The requested new sample budget completed:4167 actual optimizer updates,
1,600,128 primitive examples, equivalent in sample count to50,004 batch32 steps.
This is not optimization equivalence. The model did not achieve reliable terrain
traversal and must not replace the working V8 WASD controller.

The final20-case screen passes5 cases; the SDF-disabled baseline also passes5.
All eight constant-approach tests fail the current screen. Some of those source
routes turn, so constant-direction endpoint error is not a fair universal route
completion score. The straight official staircase independently fails on clear
penetration and map exit; the overall failure does not depend on that caveat.

| Case / control | Baseline sole vertical P99 | SDF sole vertical P99 | Observation |
| --- | ---: | ---: | --- |
| Official stairs / source-derived speed schedule |22.8cm|20.8cm|Full80 frames; body penetration P99 worsens11.0->14.1cm |
| Official stairs / constant forward approach |51.2cm|40.1cm|Only72/80 frames; exits known scene |
| Climbing / source-derived speed schedule |40.0cm|17.9cm|Improved, still fails |
| Validation boxes / source-derived speed schedule |74.5cm|63.0cm|Still fails; body penetration worsens |
| Train/validation flat controls |Unchanged|Unchanged|Frozen flat residual preserves baseline output |

Vertical sole gap is not Euclidean penetration depth. Body SDF uses566 sampled
official skin vertices, with a conservative vertical fallback when the nearest
surface cannot be certified within observed map bounds. Neither metric certifies
all body vertices or continuous swept collision. Baseline means the same final
weights with SDF disabled; the frozen non-SDF weights match the prior controller.

## Delivered Evidence

Local directory:
`motionbricks/out/model_terrain_v3_20261011/joint_sdf_20261012/train_1600k_v1/`.

- `export/evaluation.json`: complete final/disabled-SDF metrics, including failures.
- `export/receipt.json`: sample count, model and source identity.
- `export/weights.pt`: final inference weights, SHA256
  `024065ecd6475310f05e0c8aa829955cfdd2cf2c598847a2e99384b79690a9f2` verified locally.
- `export/visuals/`: official SOMA77 rigs and18,056-vertex/eight-weight skin.
  All61 referenced files exist; export's official-skin round-trip error is zero.
- `stairs-frame59.png`: browser screenshot, source left and trained model right.
- `input_audit.json`: all-rank exposure and preserved source-intent inventory.
- `oracle_report.json`: four offline source-history windows and interventions.

Viewer:
http://127.0.0.1:28936/web/decoder_pilot/?data=/out/model_terrain_v3_20261011/joint_sdf_20261012/train_1600k_v1/export/visuals/&clip=train-stairs-source_velocity_schedule&right=sdf

The visible viewer was checked at frames49 and59 with source/SDF selected.
The body rises but the timing/placement of feet differs from the real reference.
These are recorded neural outputs, not live WASD, with no IK or corrections.
The old V8 controller at28934 remains available. No new controller was deployed.

Download used only the user-owned private Drive OAuth client. Local Clash was
in direct mode, so a separate localhost27891 proxy used the already configured
HKG A node. System settings were unchanged and the temporary proxy was stopped
after download. Models/animation never traversed public SSH.

## Data Audit

All three rank counters sum exactly to1,600,128 primitive examples. Only35,894
examples used the inferred constant approach label, about2.24% of the total.
3,164 examples have an explicitly stairs-named family, about0.20%. Mixed scenes
can also contain stairs; this count must not be called total stair exposure.

Train corpus:437 official PARC records and3 older generated records store causal
velocity proxies;86 generated records store recorded intent sidecars;2003 BONES
records have no such sidecar. The current trainer ignores those recorded sidecars.
The examined recorded vectors have unit norm and describe direction, not velocity
in m/s. The long-route generator records a pre-generation target-relative command;
capture provenance and per-frame alignment must be traced before reuse. Do not
blindly substitute its norm as requested speed. The official staircase has only
a velocity proxy, so recorded intent alone cannot fix that example.

## Decoder Attribution

Four33-frame windows start at0,24,40,48 of official_df3f1541e287fcb3. Each uses
the true four-frame past and evaluates24 future frames. Offline interventions
replace predicted tokens, decoder Root conditions, or both with source-derived
values. These are explicitly oracle experiments, never controller inference.

The original captured decoder call was replayed without replacement; every
window has max reconstruction difference0. This checks coordinate restoration.
The saved diagnostic's `replay_generated` row accidentally marks future truth
as true; that row actually uses the unchanged captured predicted inputs. The
script metadata is corrected locally; preserve the original evidence file.

At start40, generated sole vertical P99 is9.2cm; oracle tokens plus oracle Root
still gives18.1cm, with relative joint error7.6cm and root RMS15.9cm. At start48,
oracle tokens plus Root gives17.1cm. Therefore giving Pose a correct discrete
answer is insufficient under the current decoder/conditioning pipeline. This
does not prove the codebook itself is incapable: decoder adaptation and the
joint token/Root interface still need controlled tests.

## Active Bounded A/B

Two arms start from the same completed4167-update checkpoint with fresh Adam5e-5,
the same seed, original loss coefficients, and the same real official85-frame
staircase plus BONES idle/walk.300 optimizer updates per arm, batch3, first150
single-primitive updates then150 seven-primitive accumulated updates:3600
primitive examples per arm, batch32-equivalent112.5.

- A: train the new SDF modules only.
- B: additionally train the existing MB codec decoder; Root/Pose, encoder and
  vocabulary remain frozen and checked. No runtime correction or oracle input.

Remote script:task/pilot_joint_sdf_decoder_ab.py, dependencies code_joint_sdf_v1.
Outputs:task/joint_sdf_20261012/decoder_ab_A and decoder_ab_B.
Logs:task/joint-sdf-decoder-ab-A.log and joint-sdf-decoder-ab-B.log.
Launch receipts:matching `-process.json` files. GPU0 and2 were freshly checked
idle. PIDs642702 and643304 are hints only; inspect current cmdline before action.

Both arms save baseline/final/disabled-SDF rollouts. In B, disabling SDF retains
the fine-tuned decoder and is not the original baseline. Compare A/B final to
their saved identical initial baseline and check flat retention. The fixed slow
mode screen may conflate traversal speed with source-clock completion; retain
its failure and report progress/penetration separately rather than weakening it.

Do not restart the completed160万 run. Finish this small A/B, inspect all outcomes,
and only make further changes supported by measured failures. A numerical win on
one training staircase is not broad or live success. Future work must include
full-length paths, independent control, and local30fps before deployment.
