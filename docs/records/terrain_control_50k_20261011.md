# Real Terrain Traversal and the 50k Trial

## Interactive Final Trial (2026-10-12)

The fully trained weights5709 now also run at
http://127.0.0.1:28934/?mode=terrain with the previous runtime sliders enabled.
Services28931/28932/28933 were preserved. Interactive PID71140 is only an
identity hint; verify its command before any process action.

Launch with scripts/launch_joint_terrain_trial.ps1 -Interactive
-UnreviewedTrial -Execution cuda-graph, the wasd_trial_final export directory,
and out/a100_live_20261010/terrain-world.npz. Source/model hashes and training
totals are unchanged. No additional training or offline quality review ran.

Controls reach the real joint model: command speed0.3-12m/s, horizon6-16 tokens,
denoising1-8 passes, survey distance0.8-20m or automatic, half-width0.3-6m,
automatic speed/horizon gain0-4, base distance0-10m, replan8-56 frames, and
path refinement0-2 passes. Replan duration is capped by complete eight-frame
blocks of decoded future; changes invalidate buffered continuations at the
played boundary. Automatic scan uses max(command speed, measured speed),
horizon, gain and base. Near observation retains the trained metric scale;
survey controls alter the far branch. These runtime ranges do not imply all
combinations were represented in training or improve motion quality.

Default8-token/4-pass/1-refinement inference retains a checked CUDA graph with
dynamic survey inputs. Other shapes use native eager model inference with
the same compiled SOMA geometry/history path. Real GPU parity checked seven
commands, including automatic and20m/6m scans: active native tokens matched;
maximum world-feature error1.2994e-5. Browser checks exercised all slider
families, automatic/manual modes, long-view profile, continuous WASD and
buffered24-frame replanning. Default generation was about51-57ms per8 frames;
12-token/2-pass/2-refinement generation was about198ms, with buffered blocks
reporting0ms generation. These are interface checks, not motion-quality gates.

Nineteen targeted tests passed before startup; browser testing then found and
fixed the service's batched-heading input compatibility. The five affected
control/live tests passed again using the actual nested command format.
The reported feet-together hopping behavior is still unresolved; restoring
controls is not evidence that its underlying cause has been repaired.

## Delivered Final WASD Trial

Final weights5709 are downloaded and hash-verified locally under
out/model_terrain_v3_20261011/wasd_trial_final. Successful final upload/download
used the user's owned OAuth client mb-private; no new client or secret was created.
Full training totals:10709 actual cumulative updates,1600512 examples,
batch32-equivalent50016. No offline quality evaluation was run.

Local service:http://127.0.0.1:28933/?mode=terrain. PID155832 is an identity hint;
verify its command before future process actions. It serves the existing256m
terrain world with official SOMA skin, camera-relative controls and path overlays.
Both source and final model SHA256 were verified. The HTTP config reports the
expected final model hash and30Hz motion. Compiled active tokens matched native
inference; maximum recorded world-feature deviation1.2994e-5. Browser inspection
showed ONLINE, the skinned character and terrain; a held W input generated moving
body poses. Observed generation latency53.9ms per8 frames, skin1.4ms. This is an
interface/startup check, not a motion-quality or long-duration stability approval.
The page is labeled V8 trial (not evaluated). Local receipt:
out/model_terrain_v3_20261011/wasd_trial_final/local_trial_receipt.json.

Old services28931/28932 were not stopped or replaced. The browser's temporary tab
later disappeared from the tool inventory, so the durable delivery is the live URL.
The app reports terrain-5 no longer exists; no monitor was recreated. Training is
complete and no additional offline review or new research run should be started.

## Owned Google API Client

The user explicitly forbids shared Google API credentials. All shared-client
transfers were stopped. Reused the user's existing desktop OAuth client named
"rclone config" in project hybrid-reactor-423804-p0. Its existing configuration
was found in the4090D account's vggt remote; no new client, key or permission
was created. Only required fields were copied into private configs, and the
original dedicated MotionBricks transfer folder binding was retained.

Use remote name mb-private and configs:
- Local:${USER_HOME}/.codex/secrets/motionbricks-drive/private-client.conf
- A100:${PRIVATE_STORAGE}/motionbricks-20261010/drive-auth/private-client.conf

Never print tokens or secrets. The original configs are preserved but must not
be used for this task's Drive API calls. Owned-client listing of the existing
private final-export directory succeeded. Current final upload log is
final-trial-upload-private-client.log; export PID2627224 is only an identity hint.
The uploader validates the specific owned client before any network call.
Final cloud path is private-store:runs/terrain-control-50k-20261011/v8-final-trial.

The app reported automation terrain-5 no longer exists; no duplicate monitor
was created. Finish the requested WASD delivery in this active task.

## Final Checkpoint Only

The latest user instruction is to use only the fully trained model. Do not launch
the earlier5350 snapshot. Full training has completed at5709:10709 actual
cumulative optimizer updates,1600512 primitive examples,50016 reference-batch
equivalent steps. There are zero quality-review directories; evaluation was
skipped at the explicit user request.

Final source:control_sample_v8_live/weights-005709.pt, SHA256
8f70da0f9db03e84af86dfd09e5fce63b7d7e761b23ed9a42aa7fd72a9bea9cc.
Private Drive:mb-transfer:runs/terrain-control-50k-20261011/v8-final-trial.
Local target:out/model_terrain_v3_20261011/wasd_trial_final. Use only the weights,
export.json and contract.json; the optimizer last.pt is not needed for inference.
The uploader was corrected to filter those files. Transfer progress now uses
final-trial-upload-direct.log (A100 HTTPS directly to private Google Drive,
8MiB chunks,10-minute timeout; no direct SSH payload). The user enabled the local
global proxy, so downloads use localhost7897. The earlier5350 artifact may remain
on disk but is not the requested trial.

## Direct WASD Trial Override

The user explicitly cancelled offline quality evaluation and requested immediate
WASD access. This supersedes prior final-review gates for the requested trial.
Do not present it as motion-quality approved. Hash checks, finite inference,
rendering and short interface checks remain necessary to make the page usable.
Do not run another offline quality evaluation or launch a new research campaign.

An immutable checkpoint at model-step5350 was retained as
wasd-trial-input-005350.pt, exported under wasd_trial_005350 and private Drive
mb-transfer:runs/terrain-control-50k-20261011/wasd-trial-005350. It has10350 actual
cumulative updates,1324800 examples, batch32-equivalent41400 steps. Source SHA256
is0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d; weights SHA256
3a17fe47da34531b8b5e780ad4f272fd899e0451886e01323dfe72650254c328.
Use scripts/launch_joint_terrain_trial.ps1 -UnreviewedTrial after private download.
Its explicit trial option overrides the automatic model-quality gate; do not
forge model_checks_passed. The model label says it was not evaluated.

The remaining authorized sample budget continues in control_sample_v8_live,
code_control_v8_live, state control-sample-v8-live-state.json, supervisor2447732,
torchrun2447824 (identity clues only), GPUs4/7. It resumes exact saved optimizer,
RNG and episode state at5350 with an explicitly amended execution policy:
--skip-quality-evaluation --amend-execution. Final weights at5709 export directly
to mb-transfer:runs/terrain-control-50k-20261011/v8-final-trial without reviews.
The old control_sample_v8 supervisor was stopped; do not restart it.

The existing local proxy was in direct mode and prevented Google Drive access.
The user has now enabled global proxy mode. Use localhost7897 and the existing
A100 reverse proxy27890. No standalone proxy was started. All large artifacts
continue through private Google Drive, never direct SSH file transfer. Check
wasd-trial-upload-restored-proxy.log for the active initial snapshot upload.

## Current Delivery Priority

The user now prioritizes making this current version usable if no major issue
remains. Finish V8 and deliver its actual local SOMA/WASD behavior before expanding
the research scope. The diagnosed13-primitive batching repair is deferred to a
later training revision; do not restart healthy V8 or introduce another training
campaign solely for it. Preserve the single final evaluation and disclose the
limitation. Readiness still needs actual motion/control/performance checks; major
drift, penetration or lost control cannot be called usable.

Local deployment entrypoint is scripts/launch_joint_terrain_trial.ps1. It checks
both source and model hashes against export.json, preserves existing services,
chooses a new free local port starting28933, and starts the official SOMA viewer
with the joint checkpoint. Use -VerifyOnly to verify exported files first. The
source dependency already exists at out/a100_live_20261010/a100-terrain.pt;
download the final weights/visuals through private Drive only. Runtime metadata
now distinguishes actual optimizer updates from reference-batch equivalent steps.
Starting a process does not prove a working page: inspect startup logs, native
versus compiled inference, actual WASD, visuals and30fps before declaring delivery.

## Latest Analysis Finding

Read terrain_v8_loss_analysis_20261011.md. A read-only CPU audit confirmed that
the batch-wide minimum source length prevents the intended13-primitive feedback:
163 reconstructed batches averaged4.27 primitives, maximum6, zero at13. Short
65-frame sources truncate the whole global batch. This is a batching defect,
not a finite-number/runtime failure. The user asked for analysis; the running
contract was not changed or restarted and no interim quality review was run.
Complete the current bounded sample-budget run as authorized, disclose this
limitation in its final review, and fix sequence batching before the next training
revision. Do not describe V8 as proven long-feedback training. The same audit
also finds sparse/rank-local loss logs and changing horizons; coarse loss means
are1.756,1.927,1.759,1.726, so persistent divergence is not established.

## Current Sample-Budget Authorization

The user explicitly selected "accelerate using the original total sample count".
This supersedes the previous requirement for 50000 actual optimizer updates at
the enlarged batch. Original budget:32*50000=1600000 repeated primitive examples.
Read this section before older V7/V6 instructions. Do not restart either run.

V7 was stopped after verifying its owned process identities. Its last observed
step was4201, but it had not saved a new checkpoint; at least201 large-batch
updates are unretained and are not credited to the new model or budget. Restore
the complete model AND Adam state from control-v7-input-004000.pt:9000 cumulative
actual updates and288000 retained examples. The output/logs remain preserved.

V8 code:code_control_v8; output:control_sample_v8; supervisor:
scripts.run_sample_budget_terrain; state:control-sample-v8-state.json; plan:
control-sample-v8-plan.json. Actual production supervisor2234442 / torchrun2234544
(identity clues only). GPUs4/7, per-card batch384, global768. Other users took
additional cards during launch, so the completed six-rank bootstrap was restored
with an explicit DDP repartition to two currently free cards. Model, Adam and
sample clock are retained; active episodes begin afresh on the new ranks and this
is not claimed to be bitwise continuation. GPUs0/1/2/3/5/6 belong to other jobs
at this snapshot; always recheck current ownership before any action. Budget
remainder1312000 examples requires1709 further actual updates, ending at model
step5709 / cumulative10709 actual optimizer updates. Final1600512 examples equals
50016 original-batch updates; the512 extra examples are batch rounding (0.032%).
Do not call this50000 actual optimizer updates or imply equal convergence.

LR and feedback probability use the sample-exposure clock, retaining the original
5e-5 peak and2e-6 floor; no blind linear LR multiplication. Adam moments and
its actual update counters are preserved. Quality evaluation is only at budget
completion; finite-number checks remain. Complete checkpoints every50 updates
use asynchronous persistence. The four-update bootstrap wrote a resumable checkpoint
under control_sample_v8_bootstrap, then the supervisor continues that same
checkpoint; those updates count because their trained model and optimizer state
are retained. That checkpoint has six saved rank states and Adam step4004. The
new two-rank sampler partition is declared in the production contract.
Do not launch another trainer while bootstrap or production is running.

Latest benchmark gains and the physical loss equivalence from V7 remain valid;
80GB occupancy is no longer the optimization objective. Measure completion time
at fixed sample exposure. Other users, old demos and private transfer rules are
unchanged. Final review still determines whether an actual WASD trial is viable.

Verified production progress at model-step4026:9026 actual cumulative optimizer
updates,307968 retained primitive examples,9624 reference-batch-equivalent updates.
The actual sample-clock LR was4.92115e-5 and feedback probability.9; both advance
with exposure rather than the unscaled update clock. No intermediate review
directories were created. Recorded code hashes match all15 active files and
finite loss/gradient checks passed. Initial throughput with only GPUs4/7 available
was252 examples/s including episode preparation, median update1.73 s. At that
early rate the remaining budget is about85 minutes plus final review; remeasure
after more updates instead of reusing the obsolete60-hour V7 estimate. Targeted
sample-clock, sampler, checkpoint and supervisor regressions:8 passed.

## Latest User Override

The user explicitly requested all available A100 compute and memory, cancellation
of intermediate quality evaluations, and evaluation only at cumulative 50000
updates. This supersedes the older two-card limit and 5000-new-update pilot gate.
Other users' jobs must remain untouched. On the latest check GPU 1 is occupied;
0/2/3/4/5/6/7 are eligible for this task after verifying availability.

V6 was stopped after verifying its process identities and retaining an immutable
hard link to its complete 4000-step optimizer checkpoint:
model_terrain_v3_20261011/control-v7-input-004000.pt. Cumulative completed updates
are 9000. Receipt: control-v7-migration-input.json. Do not restart V6 or apply its
old pilot gate. V7 execution optimization is being benchmarked in code_control_v7
and execution_benchmark_v7. The model, losses, source data and LR schedule remain
fixed; batch/world repartition is an explicitly recorded execution migration,
preserving Adam state and update count, not a claim of identical sample replay.
See the newest control-50k-v7-state.json when it exists. Until then, inspect
benchmark processes and this chat's current work; do not launch a duplicate.

The sections below retain earlier V6 evidence and are superseded where they
conflict with this explicit user override.

## V7 Execution Configuration

Production command: scripts.run_final_terrain_50k --base <A100 base> --gpus
0 2 3 4 5 6 7 --batch-per-gpu 1024. Snapshot code_control_v7, output
control_50k_v7, state control-50k-v7-state.json. Seven DDP ranks, global batch
7168, micro batch 1024 per card, one loader worker per rank with 2 GiB cache.
GPU 1 stays untouched. The launcher rechecks processes, memory and utilization.
It raises only its own file-descriptor limit to 65536 (large tensor batches
exceeded the default 1024 during the first diagnostic).

Restore the v6 step-4000 model AND Adam moments. The existing schedule continues
at 4000 toward 45000, plus 5000 inherited updates, ending at 50000 cumulative.
The changed world/batch begins new factual episodes with a recorded sampling
cursor and reseeded rank RNG; this is explicit migration, not bitwise old-run
continuation. Batch size increases 224-fold. Sample throughput increases, but
50000 updates now process far more samples and can take longer in wall time.
Do not claim that filling memory automatically speeds optimizer updates.

No intermediate quality reviews, initial review, stop-at-5000 or pilot gate.
Keep finite-loss/gradient checks. Full optimizer/RNG/episode checkpoints every
1000 updates use bounded asynchronous CPU snapshots; pending writes finish
before normal exit. Final evaluation runs once at step 45000. The supervisor
exports source/generated visualizations even if final quality fails, and sends
large artifacts only via private Drive. Failed quality never becomes deployment
approval. The old 28931/28932 demos stay unchanged.

Performance changes preserve the objective: source FK/skin cached per episode;
batched static SDF and full heightfield queries; per-sample denominators retained;
fused Adam; unnecessary DDP unused-parameter search removed when all modules train;
scene packing avoids padding every row to the global largest scene; streamed
replacement sampling preserves the exact seeded index stream.

Verification: CPU query/loss/gradient equivalence and source/motion/scene tests
passed. On an actual CUDA batch128, loss and point-gradient max absolute
difference versus the original geometry objective were both 3.7253e-9.
Single-card batch128 token8 improved from 1.546 s to .508 s/update (83 to252
samples/s). Batch1024 token16 reached61.79 GiB allocated. Real seven-card
batch7168 smoke completed eight optimizer updates, saved all seven rank states,
preserved Adam update counts, generated no quality reviews and reported zero
unknown surface points. Median update3.29 s; 57.26 s for8 updates including two
large episode setups; rank0 peak60,850 MiB. nvidia-smi observed roughly65-73 GB
per GPU and99-100% utilization during compute. These short measurements do not
include every future episode shape, checkpoint or data-loading stall.

Evidence is under execution_benchmark_v7 and execution_smoke_v7_final. Those
bounded diagnostic updates are NOT added to production's cumulative count.
Training quality and unseen control remain unproven until the final review.

Production verified running: supervisor2068219, torchrun2068237 (identity clues
only). At step4026/cumulative9026, 26 production updates took137.14 s including
episode setup: 1358.94 samples/s, median update2.975 s; all losses/gradients finite,
zero unknown-surface queries, generated-history examples present, zero review
directories. Latest code hashes match the saved14-file contract. Targeted final
local tests:28 passed. Recent device snapshots showed about65000-73000 MiB of
81920 MiB/card and89-100% utilization during compute. This is early throughput;
41000 remaining updates at this wall-time average would take about60 hours,
not a speedup in fixed-update completion time. Keep exposing both sample throughput
and total wall-time estimates, because the global batch grew224-fold.

Current status: v6 launched on physical A100 GPUs 2/3 (NVLink) after the matched
feedback experiment and verified native yaw augmentation. Supervisor PID 1656690
is only an identity clue; verify its current command before any process action.
State: model_terrain_v3_20261011/control-50k-v6-state.json; immutable code:
code_control_v6; run: control_50k_v6. Read the actual state/progress/reviews before
reporting results. No v6 motion-quality improvement or deployment is claimed yet.

Verified after launch: the complete initial review contains 21 real terrain
cases, 16 short comparisons, 4 flat controls and 4 causal directional tests.
The inherited model passes 1/21 traversal, 0/4 flat and 0/4 directional tests;
this is the retained baseline, not a training-process error. Optimization then
started and was observed at new step 326 (cumulative 5326), about 0.55 seconds
per update during the initial teacher-history phase. Read current progress
instead of reusing this snapshot or extrapolating it to final feedback speed.

V4 completed 5000 updates and stopped because train_complete_cases_retained
failed. Its weights-005000.pt initializes v6 with fresh Adam and the original
physical loss coefficients. V6 has 45000 new updates, making 50000 cumulative.
Do not restart v4, deploy the diagnostic overfit models, or relabel warmstart
updates as new training. Detailed evidence: terrain_feedback_goal_repair_20261011.md.

## Current Objective

The user's corrected priority is motion through terrain within the real training
distribution. Climb a traversable obstacle, ascend/descend stairs, and cross a
platform as demonstrated in PARC. Stopping before these tasks is not success.
Do not substitute artificial-wall avoidance for terrain-conditioned motion.

Use the MB Root/Pose/token vocabulary/decoder architecture and existing native
losses. Preserve model-side geometry objectives, actual FK-history feedback,
and shared online command construction. No runtime collision projection, root
replacement, IK, foot locking or output speed clamp. Future source height and
pose remain targets only; inputs are observed history, planar intent/facing and
bounded current local terrain along intent and predicted route.

## Corrected Experiment

- 50% real PARC motion/terrain pairs, weighted by available source duration.
- 50% BONES flat locomotion, balanced across existing control categories.
- Zero artificial wall pairs in the main run. Files remain for independent
  auxiliary diagnostics. They do not set physical loss scales, drive checkpoint
  selection, or determine long-training eligibility.
- Record sampling probability per family and actual sampled counts. The 50/50
  mixture applies to episode draws; separately report effective per-domain
  primitive exposure because complete source episodes have different lengths.
  Preserve original source splits and geometry hashes.
- Model schema: motionbricks.joint_terrain_controller.v4.
- Control version: explicit-planar-request-independent-facing-v2.

Keep movement velocity and facing independent. Construct six neural task features
from planar controls and actual generated history. Source commands are inferred
weak planar velocity/facing labels, not recorded keyboard input. Reference-path
supervision also retains the requested endpoint XZ independently of generated
drift. Free WASD uses causal rolling requests from the actual body and joystick;
evaluate this separately. Future height and pose never become conditions.
Stop-labelled files include moving approaches,
so window terminal velocity determines their weak labels. Start timing labels
still need review if initiation fails: clip-wide cruise speed is not measured
user intent.

## Actual Terrain Evaluation

terrain_traversal_metrics.py declares real-paired-terrain-traversal-v1.
For every available PARC terrain family, choose a reference using source data
only. Review up to ten seconds of consecutive generated motion (short clips
remain shorter), including source-defined vertically active windows. All input
horizons have real endpoints; no padding or invented future pose. After the
initial four observed frames, history comes from generated FK only.

Report training-source fit separately from held-out source/scene-group validation.
Missing families fail coverage. A family without held-out examples has no
generalization evidence, even if its training example works. Fixed windows
demonstrate the recorded route segment, not completion of a multi-minute course
or arbitrary unseen obstacle handling.

Score ordered three-dimensional route milestones, endpoint error, height error,
full-body FK error and all 18,056 official skinned vertices with swept collision
samples. Store the unchanged reference alongside every rollout, plus its own
physical residuals. A still character, lateral bypass, wrong-height route or
endpoint teleport must fail ordered traversal. Declared trial tolerances
(35 cm milestone radius, 50 cm endpoint, 20 cm mean root-height error, 3 cm body
penetration P99) are research screening criteria, not perceptual or physical
certification. Visual review and local control testing are still required.

Retain BONES rest/start/stop evaluations and native reconstruction checks.
Optional wall tests cannot compensate for failed traversal. No claim of PFNN
superiority follows from these tests.

## Execution

Run control_50k_v6, immutable code code_control_v6, launcher
run_terrain_control_50k.py --revision v6 --initialize control_50k_v4/weights-005000.pt,
state control-50k-v6-state.json. Source bundle and mixed dataset remain unchanged.
Global batch 32, micro batch 16 per GPU, LR warmup 1000, adapter warmup 0,
generated-history start 1000 and ramp 4000. Up to 13 consecutive primitives from
257 actual source frames, with one optimizer update per primitive and detached
generated history retained between updates. Use the native random yaw transform
for source motion and controls, invert the same transform for terrain and losses.
Unknown generated centers restart only affected training rows at the same source
time; the preceding failed prediction remains penalized. Evaluation never resets.
Checkpoint every 250 updates; evaluate every 1000. At new update 5000 (cumulative
10000), spending the remaining budget requires real-terrain
learning evidence and retained train/validation paths, reconstruction and flat
control. This is distinct from readiness for deployment. A passing budget review
continues from the exact optimizer/RNG/episode checkpoint to 45000 new updates.
Continuation removes the model-only initialization flag before resuming. Final delivery also
requires traversal and flat-control acceptance, visual review and local WASD.
Failures require diagnosis, not relaxed gates.

V6 verification: 45 targeted component tests passed plus the supervisor
continuation regression. A real two-process CPU comparison matched the complete
5-update checkpoint exactly between uninterrupted execution and a 3+2 resume,
including optimizer, RNG, requests, generated history and counters. Real two-A100
global-batch-32 smoke completed three updates with two generated-history steps,
finite gradients and zero unknown surface queries (0.611 s median per update,
4.69 GB peak allocated on rank 0). These are implementation checks, not evidence
of improved motion. Files: primitive_replay_cpu_v6/replay_verification.json and
primitive_smoke_gpu_v6/progress.json. No resource-wait queue is needed now: the
user reported availability and fresh GPU/process/topology checks confirmed it.

Only use two available cards; do not affect others' jobs. Large artifacts go
through private Google Drive/NAS/internal networking. Preserve the existing
28931 web demo and deploy a validated candidate on another local port.

## Preserved History

The v2 supervisor was deliberately stopped after inspecting PID 311871 and its
owned command. Its child pilot is bounded to 2000 updates and may finish its
checkpoint/evaluation. It cannot start the old wall-gated 50k continuation.
V2 mixed 50% PARC / 25% BONES / 25% artificial wall-stop pairs and reviewed only
32 generated PARC frames. Its results are historical controls, not evidence of
successful real-terrain traversal. All previous outputs are retained.

The existing terrain-5 automatic follow-up was updated to this objective.

## V4 Verification History

Local targeted regression: 35 passed. A100 targeted subset: 17 passed.
The real two-rank smoke run completed two full optimizer updates, wrote its
optimizer/RNG checkpoint, and completed 21 traversal cases (15 training-source
families, 6 held-out families), 16 short comparisons and 4 flat controls.
Missing family lists are empty. The shorter smoke review emits 2.13-2.93 seconds
per traversal; the main run uses up to 9.87 seconds. Initial motion quality fails
traversal/flat acceptance, as recorded, and is not a deployment candidate.

The two-update smoke remains in traversal_smoke_v3, separate from the main run.
The old v2 pilot reached 2000, failed its gates and exited; its supervisor had
already been stopped. V3 initially launched on free physical GPUs 2/3 (supervisor
549567, torchrun 549648), but its ten-second initial review exposed a generated
rollout leaving observed terrain in the mixed family. The existing vertical-frame
guard correctly rejected an invented ground height. Evaluation mistakenly aborted
the whole job, before any optimization. That failed run remains untouched.

V4 changes the evaluator to record this specific predicted-rollout failure,
retain only emitted frames, and continue reviewing other fixed cases. Invalid
initial source conditions and other exceptions still raise. Failed cases remain
in coverage and score progress against the entire fixed source route. Completion
is a separate mandatory criterion. Prefix errors are labeled partial; the budget
review compares complete paths on unchanged clocks and cannot drop cases. Source
and full learned sequences share comparison clips; interrupted initial sequences
remain explicitly shorter separate recordings, never padded.

The additional partial-rollout regression passes; the full targeted suite now
has 36 passing tests. V4 uses the same model, losses, real-source mixture and 50k
schedule as v3. Actual V4 execution: supervisor PID 590856, torchrun PID 590946,
physical GPUs 2/3. Its initial review completed all 21 terrain cases plus 16
short comparisons and four flat controls. Two cases explicitly terminated when
the prediction left known terrain: train mixed (232/296 frames), validation
misc_mdm (160/192). Both fail traversal; neither was padded or dropped.

After that complete baseline, optimization started successfully and reached
update 26 in the latest checked progress record. Losses and gradients were finite,
artificial-wall sampling remained zero, and actual sampled counts included PARC
and the separate BONES control categories. Initial gates are expected to fail;
do not restart a healthy run merely because its untrained baseline fails. The
5000-step budget review and 50000-step deployment review are different gates.
No learned quality improvement or successful WASD result is claimed yet.

## Subsequent Memorization Diagnostic

The user requested testing whether this strategy can overfit existing data.
See terrain_overfit_probe_20261011.md. The separate overfit_v1 run starts from
weights-002000.pt and completes 600 teacher-history plus 600 two-primitive
feedback updates on four fixed train clips without changing model or loss.
All four teacher-history tests meet the 5 cm Root/FK screen; generated-history
stairs/climbing still miss the 15 cm screen (Root RMSE 16.43/19.54 cm). Body-relative
joint error is only 1.70/2.37 cm, while per-primitive plan and decoder displacement
errors accumulate. These selected examples descend, and do not prove ascent.
This is a capacity/feedback diagnosis, not a new deployment or held-out success.
Keep the main bounded run and its existing budget gates unchanged. Do not describe
this experiment as proving the overall training strategy or stable terrain control.
