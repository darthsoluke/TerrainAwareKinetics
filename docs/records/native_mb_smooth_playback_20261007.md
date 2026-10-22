# Native MB playback and foot sliding

## Separate causes

The current frozen SOMA30 bundle contains Tokenizer/Root/Pose checkpoints at
50,000 updates each. SHA256:
`a17027e3cf8fc4958437329fa91a0889324949f37458a87af36c8d9c824e963c`.
No weights, joint transforms, or root trajectories have been smoothed or locked.

Three problems must remain separate: uneven UE/native frame scheduling,
discontinuities between fresh neural generations, and sliding already present
in reconstructed motion. An average frame rate alone cannot approve the first
two, and a render/frame-clock fix cannot approve the third.

## Runtime changes

The server now samples consecutive frames from previously generated motion
until the command changes, history becomes incompatible, or the buffer is low.
It drops the last four predicted frames and excludes decoded context from
playback. Publication remains immutable eight-frame packets. Minimum predicted
duration is four tokens: four context, eight published, four dropped frames.
The allowed durations remain a subset of the upstream trained range.

History is not overwritten by the backend. Cached motion may receive only the
constant planar coordinate translation already imposed by gameplay collision;
body, vertical root, and nonuniform root changes invalidate it. New commands
invalidate unpublished cached motion. The existing committed protocol still
introduces its bounded queued-block input latency; this is not an exact copy of
the unreleased official UE implementation.

The native step clock allows only a tiny floating-point boundary tolerance.
Regression reproduces `0.03333333f` causing 299 rather than 300 native steps.
Per-tick zero/one/multiple-step counters now expose presentation cadence, with
underruns reported separately. No pose interpolation or camera lag was added.

Authority:
- [Paper Algorithm 1 and Appendix G](https://arxiv.org/html/2604.24833v1#A7):
  reuse generated motion until new targets are needed.
- [Appendix C](https://arxiv.org/html/2604.24833v1#A3): overly frequent replanning
  can degrade discrete-latent motion details; drop-frame/command-change policy.
- Shipped `motion_backbone/demo/full_agent.py`: spring and target placement
  methods are still invoked directly. The public G1 preview's controller timing
  is not claimed identical to the paper's UE scheduling.

## Frozen-weight comparison

90 packets at native30Hz, 24s of idle/walk/run/turn/backward/stop commands, same
weights/seed, FP32 highest and TF32 disabled. No postprocessing. Official SOMA
FK and same numbered skinned sole vertices are used for measurements.

| Metric | Regenerate every eight frames | Reuse motion buffer |
| --- | ---: | ---: |
| Neural generations |90|47|
| Raw FK acceleration mean (m/s2) |12.022|8.745|
| Raw FK acceleration P95 (m/s2) |53.421|34.486|
| Maximum local joint step (degrees) |102.535|105.198|
| Left own-mask sole speed mean (m/s) |0.115|0.105|
| Right own-mask sole speed mean (m/s) |0.234|0.171|
| Right penetration mean (m) |0.0129|0.0172|

This reduces generation frequency and the overall acceleration diagnostic, but
does NOT eliminate forearm jumps or approve foot sliding. Own geometric masks
have different coverage and gait phases. The fixed baseline mask is generated
motion, not capture truth; neither is sufficient evidence of a learned sliding
improvement. Penetration also worsens in this comparison. Endpoint-skipping and
source-realignment-off diagnostic variants were NOT selected for deployment.

Artifacts: `out/native_mb_lafan_paper_20261006/playback-buffer-review-v2/`.
Reproduce with `scripts/review_native_soma_playback.py`.

## Confirmed model bottleneck

The existing six-window subject4 validation system review already measures
source, Tokenizer reconstruction, and Root/Pose output with the same
source-derived material-vertex masks. In low-slip source windows:

| Window / foot | Source speed (mm/s) | Tokenizer speed (mm/s) | Generated speed (mm/s) |
| --- | ---: | ---: | ---: |
| walk start0 / Left |1.2|73.4|94.8|
| walk start0 / Right |1.8|60.1|62.9|
| walk start7081 / Left |1.9|122.7|106.4|
| walk start7081 / Right |1.8|76.1|97.7|
| run start0 / Left |2.7|97.5|86.8|
| run start0 / Right |2.6|60.8|91.7|

Other moving-source windows themselves have nonzero sliding and much smaller
measured masks. These are relative diagnostics, not universal contact truth.
Evidence: `pose-efficiency-v1/assessment-50000/system-review-50000/report.json`.

Thus improving only UE or Pose cannot repair the current encoding/decoding
floor. The official Tokenizer already has foot-sliding and joint-velocity
losses (release coefficients0.01 and2.0). They remain unchanged. No candidate
sole-region custom loss or force loss is added to the native training route.

## Training dependency

Next convergence checkpoint is200k total updates, retaining Adam, batch256,
the original2M cosine schedule,10k warmup, and original keyframe curricula.
This is an intermediate review, not paper-budget completion or a guarantee of
natural motion. At50k Root/Pose have only one keyframe per end; Tokenizer has up
to two pose keyframes in total. Dense four-frame live context must not be fed
to these weights. At200k the configured training curricula are fully exposed.

Tokenizer and Root can continue independently with full optimizer/scheduler
state. A changed Tokenizer changes the motion codebook, so the old Pose cannot
be resumed against it. New Pose training must bind to the newly frozen
Tokenizer. Live deployment remains on the old snapshot until the new pipeline
passes source-masked slip, raw joint continuity, control and PIE checks.

## Verification so far

All669 native Python regressions passed before the training-runtime precision
follow-up. Editor build succeeds. The first buffer PIE run measured29.986FPS,
34.59ms P95 and zero steady underruns; its two new clock-test implementation
errors were fixed. Second run's six tests passed but performance deteriorated
to22.789FPS /52.06ms P95 /three underruns. It is NOT a smoothness approval.

`RunNativeSomaLocomotion.ps1 -RequireContinuousPlayback` now makes the existing
PIE continuous-playback gate mandatory, not just the gross-overspeed gate.
Both failed/successful evidence directories are retained. Inference precision
and render coexistence are being measured rather than hiding the failed run.

The latest strict trial `native-lafan50k-buffer-profile-v3` passed all six UE
tests with both gross-overspeed and continuous-playback gates required:
30.001FPS,34.482ms frame-time P95,zero steady underruns,361 measured ticks with
exactly one native frame each,zero skips/blends/camera lag. Its raw retarget
switch still peaks at100.891degrees. The user-visible Editor was launched only
after those mandatory tests passed. This does not explain the intermittent
second-run slowdown or establish long-session performance.

Evidence:
`F:/dev/GASP_57/Saved/MotionBricks/native-lafan50k-buffer-profile-v3/` and
`F:/dev/GASP_57/Saved/MotionBricks/soma_control_pie_causal_native-lafan50k-buffer-profile-v3.json`.

Standalone RTX3080, same90-packet buffered sequence, excluding the first four
packets: highest matmul neural-generation mean36.70ms/P9538.86ms; official high
mean36.62ms/P9537.95ms. Numerical outputs differ despite the same weights.
This is insufficient speed benefit to select a new live numerical policy; the
server remains highest, matmul/cuDNN TF32 disabled. These timings exclude UE,
packet processing and rendering and must not be substituted for concurrent PIE
response times. Artifacts: `rtx3080-buffer-{highest,high}-v1` under the native
LAFAN output root.

## Continuation launched

Remote MB root:
`${PRIVATE_STORAGE}/motionbricks-preparation/GR00T-WholeBodyControl/motionbricks`.
Outputs: `out/native_mb_lafan_paper_20261006/continuation-200k-v1/`.
Tokenizer PID218415 on GPU0, Root PID218646 on GPU1 are launch observations;
recheck command lines and process identity before sending any later signals.
No unrelated job on GPU5 was stopped or reused.

Both runs restored the selected50k full Lightning checkpoint and have executed
new optimization beyond50,500. Target is200k TOTAL updates, not200k additional.
Seed7, input batch256, official factor2 augmentation, original2M schedule,
native models/objectives/coefficients and the dataset manifest are unchanged:
`b68ad5a0d7b028dc1b5e3075ae84cfa4e5c4dc4fd20fc9aefbf63835a8167d46`.
Data remains the accepted LAFAN SOMA30 flat-locomotion training subset, not a new
BONES run. Validation/test recordings do not enter training.

The two single-GPU runners now apply the released `matmul_precision: high`
setting already declared in their configs, as the official entrypoint does.
Only these runner files have an explicit, recorded Root resume exception;
model/data/objective source identities remain strict. Prior numerical policy
was not recorded in the old contracts, and is marked unknown rather than
fabricated. Training weights remain float32, not BF16 or strict IEEE matmul.

Evaluation temporarily switches to highest matmul precision and disables
cuDNN TF32, then restores training policy, including exception paths. The
previous legacy CUDA flag mutation conflicted with the explicit matmul setting
under PyTorch2.11. Tests now set/restore that same matrix-precision interface,
not contradictory global state. All678 native Python regressions pass; a real
remote CUDA matrix-operation/precision-restore smoke check also passes.
[PyTorch precision API guidance](https://docs.pytorch.org/docs/2.11/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices).

Every optimizer update checks finite gradients; every5000 steps preserves a
checkpoint; final upstream reconstruction/control review runs automatically.
RNGs/workers are reseeded at resume, so exact uninterrupted sample order is NOT
claimed. The old model remains selected for UE. No new quality benefit is yet
established, and fresh Pose training for the new Tokenizer is NOT launched or
queued by this turn. That stage must use the verified new codebook and the
existing DDP path, followed by the same source-masked sole and raw continuity
review before deployment.

Local copies of both launch contracts are under the same continuation output
root. The remote `tokenizer.log` and `root.log` contain restored-state evidence
and step losses. Early variable-window training losses are not sliding metrics
and are not reported as a visual improvement.
