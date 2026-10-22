# Native MotionBricks SOMA Baseline

## Scope

Use the repository's shipped MotionBricks training implementation as the baseline,
not the custom Bones control, contact-repair, or rollout models. This is a SOMA
adaptation of the public preview implementation in this checkout, not a claim to
reproduce the UE human demo or the paper's full training budget.

Three GPT-6 Luna subagents implemented the data adapter, native model factories,
and independent upstream parity tests. The parent integrated them, corrected
integration issues, ran the full suite, and owns the remote training/evaluation.

## Native Contract

- Original `VQVAE` and `MotionVQVAEModel`, with native decoder root output.
- Native Root/Pose model factories, not `BonesControlRootModel` or `BonesPoseModel`.
- Shipped feature modes, eight-head EMA-reset quantization, AdamAtan2, losses,
  random keyframes, and token masking. No custom geometry or rollout losses.
- The preview configs use 6-16 tokens, four frames per token, and 30 Hz. This
  baseline therefore uses 24-64 local frames (0.8-2.13 seconds); it does not claim
  that the paper's stated minimum of 12 frames equals the preview setting.
- Preserve 10,000-step LR warmup, 200,000-step keyframe warmup, and the original
  2,000,001-step scheduler horizon. A bounded stop does not compress these.
- Pose token focus is cosine-sampled per batch. It is not an all-MASK training
  replacement or a newly invented step-based masking curriculum.
- Root and Pose are constructed at original backbone sizes. Neither is trained
  by this first tokenizer run.

Necessary adaptations: SOMA30 skeleton and derived feature dimensions, BONES data
and train-only normalization, source resampling to 30 Hz, fixed-plane input
(`floor_estimation=None`), one available GPU and batch size 32. The Pose factory
also freezes supporting tokenizer parameters explicitly: the upstream constructor
assigns to a method name instead of freezing the parameters. No upstream file was
changed to make this mechanical correction.

## Data

Source: `${PRIVATE_STORAGE}/motionbricks-bones/pilot-v7`.
New data: `${PRIVATE_STORAGE}/motionbricks-bones/native-mb-soma30-v1`.

All 191 shards converted: 152 train, 19 validation, 20 test. The original source,
splits, and old statistics are untouched. Offline anti-aliasing and SLERP resample
rotations/root positions, then the official FK and feature extraction regenerate
velocities and contact features at 30 Hz. This is data-rate conversion, not output
motion smoothing or foot correction. The old fixed ground calibration is retained.

Global inputs have 366 channels, local outputs 365, and dual statistics 370.
Training reads normalized global features. Official conversion supplies the local
features inside each sampled window. Statistics use train-only 65-frame windows
and four yaw orientations (3,800 augmented windows; 247,000 frames).

`native_manifest.json` binds converted shards, accepted split records, statistics,
and conversion source hashes. Training verifies these before model construction.
The original contact feature detector is used; contact accuracy is not established
by this change, and no region labels are presented as new physical ground truth.

## Entry Points

- `scripts/build_native_mb_dataset.py`: create a new immutable dataset version.
- `scripts/train_native_mb.py`: from-scratch native tokenizer training and fixed
  validation reconstruction. Stop steps and schedule horizon are separate.
- `scripts/render_native_mb_review.py`: saved source/reconstruction skeletons,
  no IK, smoothing, root replacement, or other motion correction.

The native runner does not accept old model weights. Prior experiments and UE's
current backend are unchanged. `--resume-run` accepts only a completed native run
with matching config, data hashes, seed, and batch size. Lightning restores model,
optimizer, scheduler and loop state; `--steps` is the total step target, not an
additional budget. This does not promise bitwise-identical uninterrupted sampling:
the RNGs are reseeded, but the step-based curricula are not restarted.

## Verification

The 12,000-step delivery's combined local suite passed 372 tests, including 19
native-path tests. The continuation work passes all 407 local tests, including
independent checkpoint review, Root/Pose backward integration, source-identity
resume checks, and evaluation precision restoration/rejection regression tests.
Behavioral parity tests load committed upstream code and compare stock loss values,
prediction gradients, and seeded window sampling against the current base path.
Native factories enforce 30 Hz and test original classes/dimensions, root decoder
gradients, and frozen supporting tokenizer behavior.

The remote 10-step smoke completed training, checkpointing, and evaluation on 57
fixed held-out windows under both no-keyframe and context4 conditions. It is an
execution check, not a trained-quality baseline. Test-set motions are not scored.

## Training Status

The 12,000-update run completed from scratch on freshly checked idle GPU 0 at
`out/native_mb_soma30_20261006/tokenizer-12000` in the remote repository.
Training took 528.2 seconds, with 715.9 MiB peak allocated GPU memory. Results and
the completion state are in `training_result.json`, `run_contract.json`, and
`validation/metrics.json`. Reports and reconstruction arrays were copied into the
same relative local output directory; full checkpoints remain on g3.

Fixed validation windows, equal weighting per window, scoring local frames 4-63:

| Condition | Windows | FK MPJPE | Root error | Joint velocity error |
| --- | ---: | ---: | ---: | ---: |
| No keyframes | 57 | 75.24 mm | 53.05 mm | 0.1990 m/s |
| Four context frames | 57 | 73.70 mm | 51.41 mm | 0.2003 m/s |

All ten entries of each of the eight codebook heads were used on these windows.
Maximum normalized-feature difference between the training forward path and
token-index decoding was 1.91e-6. These are early reconstruction measurements,
not a direct A/B against previous 60 Hz experiments and not locomotion acceptance.

Resume was also exercised on the separate smoke run, from step 10 to step 12:
exactly two additional updates. Saved scheduler step advanced from 10 to 12;
saved LR advanced from 2.0e-7 to 2.4e-7 rather than resetting. The 12,000-step
checkpoint has scheduler step 12,000 and LR 0.0001999995115. Its SHA256 is
`2f3b1c0b400530b6ccc27cd82dccb272b41063a4f7ddfb44a77a97626558d2ad`.
The exact pre-resume runner used for the main experiment is preserved alongside
its reports as `train_native_mb.executed.py`.

This budget reaches the end of LR warmup, but not the full keyframe curriculum.
At 12,000 steps the official keyframe sampler still permits at most one keyframe.
It is therefore explicitly premature to call this a converged multi-keyframe
tokenizer or to freeze it as a finished interactive locomotion backend.

## Native Continuation Protocol

The next bounded run resumes the 12,000-step checkpoint to a total of 45,000
optimizer updates, without changing training data, losses, network size, batch
size, optimizer, or the original 2,000,001-step LR schedule. The tokenizer's
original keyframe curriculum first permits two keyframes at 40,000 updates;
45,000 is an observation milestone, not a convergence criterion. The full
200,000-step keyframe warmup has not been shortened.
The four-context-frame review is therefore ahead of the trained keyframe-count
curriculum at both 12,000 and 45,000 steps; it is a fixed diagnostic probe, not
evidence that four-frame continuation conditioning is already trained. Four
keyframes first become possible at 80,000 steps under the unchanged curriculum.

Root/Pose integration tests now exercise their original `training_step` and
backward paths with SOMA30 features. The full backbones remain intact; only the
test sampling horizon is restricted to the original six-token minimum. Root's
continuous decoder and duration head receive gradients; Pose's original masked
and incorrect-token training mixture receives gradients while its supporting
tokenizer stays frozen. These tests do not constitute training either stage.

Independent review uses the same 57 held-out windows and scores frames 4-63,
with interval endpoints inside that range. In addition to world-space error,
it reports root-relative FK error, local joint rotation error, and root-relative
joint velocity error. Root-relative scoring removes translation only, not
orientation or pose error. The dataset hash and exact window signature must
match before comparing checkpoints.

The executed 45,000-step runner is preserved inside that run as
`train_native_mb.executed.py` (SHA256
`50ba2ad985488900f7835929a245939e6fee831cc673ade4e553b8f0c026d08d`).
The native data/model adapters, upstream tokenizer model, VQ-VAE module, and
training sampling helper match the 12,000-step run's recorded hashes on g3.
The local VQ-VAE hash differs only through CRLF/LF line endings. Future resume
validation checks these source identities and rejects semantic changes; it
does not require orchestration-only runner edits to be byte-identical.

### Evaluation Precision Diagnosis

The initial CUDA 25,000-step review failed the unchanged 1e-5 normalized-feature
token-decoder parity check. The official straight-through expression
`x + (quantized - x).detach()` differs from direct code lookup by at most
1.1921e-7 in the measured latent vectors. Direct VQ output and code lookup
matched exactly; repeating the same forward call also matched exactly.

cuDNN's default TF32 convolution amplified this rounding difference. On the
worst feature-error window (run shard `9b6f23b5ecc3b442302b_30hz`, start 20,
context4), disabling only evaluation's TF32 reduced output-feature discrepancy
from 0.00263649 to 3.8147e-6 and maximum joint-position discrepancy from
0.000101219 m to 4.3393e-7 m. Training settings, quantization, model weights, and
the parity threshold are unchanged. Evaluation now temporarily disables TF32
for CUDA matmul and cuDNN and restores the previous settings on exit.

## 45,000-Step Result

The continuation completed normally at exactly 45,000 updates, adding 33,000
updates in 1,457.15 seconds (24.3 minutes). Peak allocated GPU memory was
712.97 MiB. Final checkpoint SHA256:
`a7a1f2aa35e107f44104db3686fcee2408315d240acd82b05c24b6c5a04bbecc`.
The original runner's validation completed successfully, and independent FP32
CUDA review also passed. Maximum token-decoding discrepancy was 6.676e-6;
all ten codes in each of eight heads were used on validation windows.

Same 57 validation windows, no-keyframe condition, full-precision evaluation:

| Optimizer step | World FK MPJPE | Root error | Root-relative FK MPJPE | Joint velocity error |
| --- | ---: | ---: | ---: | ---: |
| 12,000 | 75.24 mm | 53.05 mm | 46.25 mm | 0.1990 m/s |
| 25,000 | 67.30 mm | 40.75 mm | 46.84 mm | 0.1988 m/s |
| 45,000 | 60.74 mm | 26.05 mm | 51.02 mm | 0.2066 m/s |

The fixed context4 probe shows the same mixed trend: world FK error
73.70 -> 59.74 mm and root error 51.41 -> 25.25 mm, but root-relative FK error
45.78 -> 50.61 mm and joint velocity error 0.2003 -> 0.2069 m/s.
This is improved root reconstruction, not demonstrated improvement of limb
motion. The contact/slip diagnosis is explicitly not assessed by these metrics.
Do not approve or deploy the 45,000-step checkpoint as an interactive backend.

Artifacts are under `out/native_mb_soma30_20261006/`: immutable FP32 reviews at
`review-{12000,25000,45000}-fp32/metrics.json`, completed run records at
`tokenizer-45000/`, and five source/reconstruction skeleton GIFs in that run's
`validation/` directory. They use source-encoded tokens and no correction;
they are not WASD-generated clips. Local reports and videos are copied back;
full optimizer/model checkpoints remain on g3.

### Training Versus Validation

Same evaluator, CPU float32, identical precision and scoring policy, fixed
windows within each split (456 train windows and 57 validation windows):

| Metric, no keyframes | Train 12k | Train 45k | Validation 12k | Validation 45k |
| --- | ---: | ---: | ---: | ---: |
| Root-relative FK MPJPE | 21.78 mm | 11.77 mm | 46.25 mm | 51.02 mm |
| Root-relative joint velocity error | 0.1027 m/s | 0.0638 m/s | 0.1810 m/s | 0.1980 m/s |
| Normalized Smooth L1 | 0.02218 | 0.00640 | 0.09050 | 0.10694 |

Every one of the five validation behavior groups has worse root-relative FK
error at 45k, while the corresponding training groups improve. This is a
generalization gap consistent with overfitting or insufficient distribution
coverage, not evidence that longer training alone will fix limb motion. The
current run is single-seed; it does not isolate the sole causal explanation.
Training and validation contain different recordings, so the comparison does
not claim they have identical difficulty. Test-set clips remain unscored.

The larger training-split CUDA review still hit the strict 1e-5 absolute
token-decoder guard with TF32 disabled; its failed attempt has no accepted
metrics. Both splits and both checkpoints were then evaluated on CPU without
loosening the guard, and all passed. The matching-backend reports live in
`review-{train,validation}-{12000,45000}-cpu-fp32/`. No CUDA training-split
parity approval is claimed; the trained checkpoints were not altered.

## Next Training Gate

Do not simply extend the same 152-clip pilot and assume the remaining curriculum
will remove this gap. The next training version should expand flat-ground
locomotion coverage from the already available BONES source, using the existing
SOMA conversion/FK path and recording-group split protection. Preserve the pilot
validation/test identities and keep related recordings/variants out of training.
Version the dataset and train-only normalization; changed statistics must not
be silently loaded through this run's strict resume path.

Keep the native architecture, training objective, and keyframe schedule as the
baseline. Root training is independent of the tokenizer and can proceed in
parallel on the compatible expanded dataset. Pose training should not freeze
this pilot tokenizer as a quality-approved final model. Keep automatic held-out
and training-split reconstruction review attached to future checkpoints.
Reconstruction from source-encoded tokens is not WASD generation; no result here
establishes that UE twitching or foot sliding has been solved. UE is unchanged.

## Expanded Corpus Execution

The approved expansion uses the frozen pilot candidate inventory, not new
filename rules or relaxed geometry thresholds. It preserves every one of the
191 pilot shard files and the full validation/test record dictionaries. New
clips come only from training families. Selection rotates between behaviors
and recording/propagation families, with deterministic ordering within each
family. Content hashes of all 1,278 reserved validation/test candidates also
block renamed duplicate motions from entering training.

The 60 Hz intermediate at `${PRIVATE_STORAGE}/motionbricks-bones/expanded-planar-2048-v1`
completed with 2,003 train, 19 validation, and 20 test clips. There are 155
training families and 4.7007 hours of training motion. Of 1,896 added candidates,
1,851 passed and 45 were rejected for being shorter than the unchanged 129-frame
minimum. Training behavior counts: backward 314, idle 348, lateral 348, run 301,
turn 346, walk 346. Conversion took 568.7 seconds after the protected-content
preflight.

This controlled expansion deliberately retains the pilot's `position_only_v0`
fixed-plane proxy geometry, verified against its saved offsets and mesh hash.
Two existing training clips (walk and run) were reconverted first: every saved
array matched exactly, including rotations, roots, features, and confidence.
This is not a new approval of the proxy contact geometry. The native 30 Hz
builder still regenerates the original MB contact features independently.

The native builder binds the completed expansion report and checks copied pilot
identities before conversion. New statistics are train-only; new training starts
from scratch rather than loading old weights against incompatible statistics.
The comparison budget remains 12k updates followed by resume to 45k, seed 7,
batch size 32, original networks/losses/optimizer and unchanged keyframe/LR
schedules. Test motions remain unscored and UE remains unchanged.

The evaluator now has a separate physical-only cross-dataset comparison. It
requires identical unnormalized source windows, skeleton names/hierarchy/neutral
geometry, FPS, conditioning and scoring ranges. Different statistics cannot
justify comparing normalized Smooth L1, which is explicitly excluded from this
comparison. Old reports must be reevaluated to acquire actual source identities.
The latest full local suite passes 442 tests. This includes source-identity
comparison, split-protected expansion, and three-panel renderer regressions.

The native 30 Hz build completed in 310.3 seconds. All fields in all 39 held-out
shards match the previous native dataset exactly. There are 2,003 usable training
clips (up from 152), 155 training families (up from 50), and 4.7007 training hours
(up from 0.3045). Train-only normalization uses 59,196 augmented windows; validation
and test statistics are never fitted. Dataset manifest SHA256:
`b2cff543f93f7a3f04a177ae6f7aacbced45c556da07db02a23ec92e45fa29e2`.

### Expanded 12,000-Step Result

The expanded run completed 12,000 updates in 437.36 seconds, with 715.90 MiB
peak allocated GPU memory. It uses the same 23,647,725-parameter native tokenizer,
configuration, seed and optimizer-update budget as the original pilot. Checkpoint
SHA256: `816e472e0d4c3e7874e2184b79b7094d57686350b98d72e713b97aa62e4f86e9`.
Independent CPU FP32 review passed token-decoder parity and the physical-only
source-identity comparison against the reevaluated pilot checkpoint.

| Metric, no keyframes, 57 fixed validation windows | Pilot 12k | Expanded 12k |
| --- | ---: | ---: |
| World FK MPJPE | 75.24 mm | 73.38 mm |
| Root error | 53.05 mm | 58.10 mm |
| Root-relative FK MPJPE | 46.25 mm | 32.91 mm |
| Joint velocity error | 0.1990 m/s | 0.1557 m/s |
| Root-relative joint velocity error | 0.1810 m/s | 0.1380 m/s |
| Local rotation geodesic error | 0.1206 rad | 0.1011 rad |

This is better held-out limb reconstruction at the same update count, with a
small root-error regression. It is not a slip/penetration result or an
interactive-generation result. The context4 probe has the same trend, but four
keyframes remain beyond the original curriculum at this checkpoint. Three-panel
source/pilot/expanded comparisons are saved separately from the immutable input
reviews. They use shared camera/bounds and contain no postprocessing.

### Expanded 45,000-Step Result

The continuation completed normally at exactly 45,000 total updates, adding
33,000 updates in 1,192.23 seconds (19.9 minutes). The two training stages used
1,629.59 seconds (27.2 minutes) in total, excluding data conversion and reviews.
Peak allocated GPU memory was 712.97 MiB during continuation. GPU 0 was checked
again for availability before this stage. Final checkpoint SHA256:
`eb2600665e18a2a2c1bd585fdcb2159dfb60cb008d9cccaf603856c1c75855ce`.

The run, independent CPU FP32 review, strict physical-only source-identity
comparison, and token-decoder parity all completed successfully. Maximum
normalized-feature decoding discrepancy was 2.3842e-6; all ten entries of each
of the eight codebook heads were used. Configuration SHA256 is unchanged from
the pilot: `96e7f20967fdb4677f8b00511c3d03c69d51f5ae2abd9cba55a1caeb22822f76`.

| Metric, no keyframes, same 57 windows | Pilot 45k | Expanded 12k | Expanded 45k |
| --- | ---: | ---: | ---: |
| World FK MPJPE | 60.74 mm | 73.38 mm | 39.73 mm |
| Root error | 26.05 mm | 58.10 mm | 23.33 mm |
| Root-relative FK MPJPE | 51.02 mm | 32.91 mm | 27.69 mm |
| Joint velocity error | 0.2066 m/s | 0.1557 m/s | 0.1245 m/s |
| Root-relative joint velocity error | 0.1980 m/s | 0.1380 m/s | 0.1186 m/s |
| Local rotation geodesic error | 0.1218 rad | 0.1011 rad | 0.0823 rad |

Unlike the small pilot, held-out limb reconstruction improves between 12k and
45k on the expanded corpus. Against the same-budget pilot, root-relative FK
error is 45.7% lower and joint velocity error is 39.7% lower. These numbers show
that the expanded training data and its train-only normalization help this
native baseline; they do not isolate the sole cause of the earlier defects.

| Behavior, no keyframes | Windows | Pilot / expanded FK error (mm, root-relative) | Pilot / expanded velocity error (m/s) |
| --- | ---: | ---: | ---: |
| walk | 9 | 42.86 / 20.64 | 0.1503 / 0.0798 |
| run | 12 | 86.60 / 43.21 | 0.5312 / 0.2881 |
| turn | 12 | 40.37 / 24.55 | 0.1650 / 0.1167 |
| idle | 12 | 42.13 / 24.99 | 0.0265 / 0.0292 |
| backward | 12 | 41.12 / 23.29 | 0.1460 / 0.0977 |

Every behavior improves in root-relative FK error, but idle velocity error
regresses slightly. Do not claim that every motion metric improves. The context4
probe gives world error 39.43 mm, root error 24.27 mm, root-relative FK error
26.59 mm and velocity error 0.1334 m/s. Four keyframes are still ahead of the
unchanged training curriculum at 45k; the probe is not continuation acceptance.

### Artifacts and Next Gate

Reports and five source/pilot/expanded comparison GIFs are copied back under
`out/native_mb_soma30_expanded_20261006/`. `review-45000/metrics.json` is the
independent physical comparison, `tokenizer-45000/` contains the completed run
records and reconstruction arrays, and `comparison-45000/` contains the visual
comparisons. Full model/optimizer checkpoints remain on g3. The executed runner
is preserved in each run directory; prior data and experiments are untouched.

This is a single-seed tokenizer result. No Root or Pose model was trained in this
experiment. No UE changes or deployment occurred. Neither foot slip/penetration
nor long-horizon interactive control is established by this reconstruction
review. There is no IK, foot locking, root replacement or output smoothing.

The result supports proceeding with the native training route, rather than
freezing the old pilot or inventing a repair layer. The next staged experiment
should continue the expanded tokenizer through the unchanged 80k four-keyframe
milestone and review conditioned reconstruction there; do not call 45k converged.
Native Root training can proceed independently on this same versioned dataset.
Pose training should use an explicitly selected, frozen native tokenizer after
its relevant reconstruction gate. Controlled generation and continuous rollout
remain separate acceptance steps before returning to UE.

## Native Control Training Protocol

The user approved training the remaining Root/Pose stages. The first staged
execution continues the same expanded tokenizer from 45k to 100k, not merely
stopping at the first 80k update that permits four keyframes. The 100k milestone
includes training experience with four-frame conditioning; it still does not
complete the original 200k keyframe warmup or 2,000,001-step training horizon.

Root trains independently on the identical expanded dataset. Pose binds one
completed tokenizer checkpoint by dataset/config/checkpoint hashes, freezes its
parameters and eval mode, and verifies its tensor digest before/after training.
The supporting tokenizer is stored outside the native Pose state_dict, exactly
as in the upstream model, so both checkpoints are required for deployment.

`scripts/train_native_mb_control.py` wraps the original Root/Pose Lightning
classes without subclassing their training steps, losses or condition sampling.
The original schedule horizon remains 2,000,001 updates, LR warmup 10k, and
keyframe warmup 200k. The bounded first control milestone is 12k updates, with
continuation decided from held-out measurements and measured runtime. Networks
retain their original dimensions; batch size is 32 on one freshly checked GPU
per job. Only training clips with at least 65 global frames are used. The shipped
Root off-range sampling probability is zero; the runner checks larger minimum
lengths if that setting is ever explicitly changed.

Automatic checks count actual optimizer callbacks, reject skipped training
batches/nonfinite gradients, bind all dependencies, and preserve full Lightning
resume state. Root 10-step training and 10->12 resume both passed on g3; Pose's
10-step smoke passed with unchanged frozen tokenizer state. These smoke runs
explicitly use `trained_not_reviewed` status, not quality approval. Their Pose
dependency is the existing 45k model only for execution testing; the formal
Pose run will record its separately selected tokenizer dependency.

Held-out review must distinguish sparse-root prediction, pose-token generation
given a source root trajectory, and any supplied target-pose condition. No
source-encoded future tokens may be passed off as generated motion. Start-only
and endpoint-pose probes must be labeled separately because the shipped Pose
training always samples start/end pose conditions. No IK, motion smoothing,
root replacement, or UE changes are part of this training stage.

### Expanded Tokenizer 100,000-Step Result

Continuation from 45k added exactly 55,000 updates in 1,997.03 seconds, with
714.25 MiB peak allocated memory. The original configuration, LR schedule,
keyframe curriculum, dataset and optimizer state were retained. Checkpoint:
`770758d08148f409adaee8511b78e848a05edc5c2ad6b48f9a8e13cac9038d56`.
Independent CPU FP32 evaluation uses the identical 57 validation windows and
strict same-dataset comparison. The executed runner was snapshotted only after
matching its LF-normalized hash to the run contract.

| Metric | 45k, no keyframes | 100k, no keyframes | 100k, context4 |
| --- | ---: | ---: | ---: |
| World FK error | 39.73 mm | 30.72 mm | 29.62 mm |
| Root error | 23.33 mm | 13.64 mm | 14.48 mm |
| Root-relative FK error | 27.69 mm | 25.54 mm | 24.43 mm |
| Joint velocity error | 0.1245 m/s | 0.1176 m/s | 0.1211 m/s |
| Local rotation error | 0.0823 rad | 0.0739 rad | 0.0718 rad |

The 100k tokenizer is selected as the immutable dependency for the first formal
Pose run. This remains source-encoded reconstruction, not a contact-physics or
interactive-control acceptance result. Four keyframes now have actual training
exposure. Eight keyframes are still outside this tokenizer's current curriculum.

### Root Curriculum Diagnostic

The first native Root run completed 12,000 updates in 439.87 seconds with
1,040.12 MiB peak allocated memory. Its final checkpoint was preserved when an
initial evaluation import failed; the missing evaluator was deployed and the
independent CPU review completed without restarting training.

The original Root curriculum has trained at most one keyframe per end at this
milestone. Evaluation at the same weights and same 57 windows found:

| Root condition | Position error | Velocity error |
| --- | ---: | ---: |
| Root only, four frames at each end | 1.926 m | 2.454 m/s |
| Root only, one frame at each end | 0.228 m | 0.424 m/s |
| Global root, local root and pose, one frame at each end | 0.128 m | 0.306 m/s |

The last case supplies a real target body pose, matching the shipped early
in-between task; it is not WASD. These are distinct probes, not interchangeable
scores. No sampler or output repair was changed to conceal the mismatch.
The Root continuation to 45k retains model/data/loss/sampling hashes and full
optimizer/scheduler state. An explicitly recorded entrypoint-only revision
fixes direct-script import support and verified Pose-tokenizer load bookkeeping.
Sampling RNGs are reseeded on resume, so the run is not claimed bitwise identical
to uninterrupted sampling.

### Completed Root/Pose Milestones

Root reached exactly 45,000 updates with zero skipped batches. Its 12k->45k
continuation took 1,218.01 seconds and 1,041.43 MiB peak allocated memory.
Checkpoint: `295ad84e3346a4e6f5c5ff7a0d6d5eee6003003b9ec6e44d1f5f4358077ac1a9`.
On the same native one-keyframe-per-end task, position error fell from
128.09 to 62.29 mm, and velocity error from 0.3060 to 0.1978 m/s. This task
supplies global root, local root and body conditions at source frames 3 and 63.
Root-only one-frame error is 199.06 mm; the four-frame root-only stress probe
remains poor at 1.713 m. These results do not establish command-only control.

Pose completed 12,000 updates from scratch in 607.24 seconds, with zero skipped
batches and 3,186.92 MiB peak allocated memory. It contains 136,582,512 trainable
backbone parameters; the separately loaded tokenizer is frozen. Checkpoint:
`b9648c8dd101c882f7837161661fd16322285abf1d6b1f10acede34c1bba9203`.
Frozen tokenizer tensor digest, unchanged before/after training and on reload:
`48c4f595842e6a457a33a254c0999d265776fa23d6b7ec5612820270faf233c7`.

Post-training Pose evaluation initially exposed Lightning's teardown behavior:
the registered Pose backbone moved to CPU while the upstream plain-dict
tokenizer remained on CUDA. The evaluator now invokes the original
`move_supporting_nets_to_device` hook before decoding. CPU/CUDA regression
tests cover this mismatch. Existing weights were reloaded, not retrained;
the successful independent CPU FP32 result is `pose-12000/review-v2` and the
first failed attempt is retained in the run's recovery record.

| Pose-only condition, known source root trajectory | World FK error | Root-relative FK error | Joint velocity error |
| --- | ---: | ---: | ---: |
| One source pose at each end | 97.83 mm | 73.90 mm | 0.3243 m/s |
| Four source poses at each end, stress probe | 110.71 mm | 87.58 mm | 0.3614 m/s |
| First four poses only, stress probe | 119.68 mm | 96.29 mm | 0.3377 m/s |

Pose errors measure distance to one possible source motion. Conditional motion
is non-unique, so these are not sufficient naturalness or contact criteria.
The four-per-end probe also exceeds the frozen tokenizer's trained keyframe
count. No source future tokens are encoded in any of these generation probes.

### End-to-End Generation and Failed Command Probe

`scripts/evaluate_native_mb_inbetween.py` runs the original `motion_inference`
Root -> Pose -> native decoder path over all 57 validation windows. Only source
keyframes 3 and 63 supply global root, local root and body; frame 0 supplies a
coordinate reference, not a body condition. Local endpoint velocities use
intervals [3,4] and [63,64], explicitly recorded in metadata. Intermediate root
trajectories and every pose token are model predictions. Interior scoring is
frames 4..62, excluding the supplied endpoint. Full-body world error is
130.92 mm, root error 91.45 mm, root-relative error 79.16 mm, and joint velocity
error 0.3894 m/s. This is conditional in-between generation, not reconstruction
and not WASD. Five source/generated comparison arrays and GIFs use truthful
model-predicted-root captions and no postprocessing.

The separate command-only probe uses four initial source frames, then only
played generated history. Commands are idle, forward, turn right, stop, and
backward. Thirty replans commit eight frames each, producing eight seconds.
It uses the original inference path with no IK, foot locking, output blending,
root replacement or trajectory smoothing. **It fails qualitatively and is not
approved for UE:** mean velocity-command error is 88,824.42 m/s and minimum
root height is -8,039.98 m. Finite tensors are not evidence of plausible motion.
Do not suppress these values, auto-scale a video to hide the divergence, or
report successful execution as successful locomotion.

The original interpretation emphasized the early keyframe curriculum and the
missing target body pose. The runtime investigation below supersedes the proposed
response of merely extending training or changing condition-sampling probabilities.
The omitted target pose is supplied by the original smart primitive at runtime;
it is not future ground truth and does not require velocity-command supervision.
These milestones do not complete the original 2,000,001-update horizon or the
200k keyframe curriculum, but that fact alone does not explain the numerical
divergence or justify changing the native objective.

The measured command-probe median model-call latency is 14.40 ms (P95 15.36 ms)
on g3's RTX4090. This is neither a stable-motion benchmark nor an RTX3080/UE
PIE30 acceptance result. Foot slip/penetration and terrain remain unassessed.

### Delivery State

The final local suite passes 506 tests. Training runs and evaluations completed;
all full checkpoints remain on g3 under `out/native_mb_control_20261006/` and
`out/native_mb_soma30_expanded_20261006/tokenizer-100000/`. Contracts, exact run
configurations, executed runners, metrics and small motion arrays were copied
locally. In-between visual comparisons are in
`out/native_mb_control_20261006/inbetween-root45k-pose12k/`; the failing command
probe is retained separately in `command-root45k-pose12k/`. The tokenizer
45k/100k comparison is in `out/native_mb_soma30_expanded_20261006/comparison-100000/`.
Old experiments and held-out data are unchanged. No new model was deployed to UE.

## Runtime Reproduction Correction

The frozen-weight investigation follows the [paper's Section 6.1 and Appendix G](https://arxiv.org/html/2604.24833v1)
and the official checkout at commit `ab1add20491c8093c696be27fb2e8eeb31001016`.
The public implementation is the G1 preview, not the unpublished UE human runtime.

Three concrete differences invalidate interpreting the old command-only probe as
a complete MotionBricks smart-locomotion reproduction:

1. It omitted the smart primitive. The official path computes a spring-based
   input root goal, places sparse reference body poses on that goal, then calls
   Root -> Pose -> decoder. Reference poses are input constraints, not output
   playback and not source-encoded future tokens.
2. `skip_ending_target_cond` clears only the decoder's ending pose constraints.
   Root and Pose still receive target poses in the official demo. Clearing future
   body conditions before those modules changes the task.
3. The shipped G1 playback uses `DEFAULT_PRED_OFFSETS=4`: generation resets the
   playback cursor to zero; at cursor eight its context getter selects the next
   unplayed frames 8..11. The old probe instead used the last four played frames
   and skipped prediction frames 0..3. Appendix G describes a past-pose buffer
   for UE, so neither policy may be silently equated to both implementations.

`native_mb_locomotion.py` directly binds the original spring, pose-placement and
pure inference-packing methods. The only G1-specific loader replacement builds
SOMA reference transforms from train-only clips. It cancels the G1 helper's
documented target-speed multiplier so SOMA velocity commands retain m/s units.
Input realignment is recorded; no generated output is realigned, filtered,
blended, locked, clamped or replaced. The pure helper was mechanically extracted
from `_generate_inbetween_frames` before this investigation; the upstream Root,
Pose and decoder computations are unchanged.

The comparison also distinguishes four-context/four-target, four-context/one-
target (supported by the paper), and one-per-end early-curriculum diagnostic
conditions. Native timing remains masked and is predicted within the recorded
allowed-token mask; fixed-eight timing is a separate ablation.

The old probe remains as a negative control. Read-only Root/decoder tracing
reproduced its saved eight-second divergence at identical weights. Early moving
plans already show replanning boundary speeds exceeding within-clip speeds;
height drift alone is therefore not an established first cause. Traces are saved
at `out/native_mb_control_20261006/runtime-trace-legacy-20261006/trace.json`.

### Frozen Runtime Results

Seven same-weight command cases and two module-specific input-mask ablations
completed on freshly checked idle g3 GPU 4. The report is
`out/native_mb_control_20261006/smart-locomotion-frozen-20261006/REPORT.md`.
Corrected four-per-end smart locomotion still diverges: public-buffer playback
reaches a peak played root speed of 282,997 m/s. Sparse one-per-end diagnostics
remain bounded for eight seconds but are slow and do not establish naturalness.

Keeping full Pose/decoder inputs while reducing only Root masks gives a bounded
1.000..1.083 m pelvis height and 1.152 m/s peak root speed. Keeping Root's full
masks while reducing only Pose/decoder inputs still reaches 60,082 m/s. This
localizes the dominant numerical failure to Root's multi-keyframe use in this
sample. It does not prove that training length is the only cause.

Root45k and Pose12k have only one-per-end training exposure under the unchanged
200k curriculum. Continue the native Root/Pose curriculum with the selected
tokenizer fixed, and review full smart-primitive closed loops automatically at
training milestones. Do not deploy sparse-mask ablations, invent output repair,
or change the training task to command-only as a substitute. No training or UE
deployment occurred in this investigation. The final stable local test suite is
539 passed; all remote inference experiments completed and agents were closed.

## Native Curriculum Continuation Started

The continuation now runs on g3; this is execution, not a new training method.
The remote output directory is
`${PRIVATE_STORAGE}/motionbricks-preparation/GR00T-WholeBodyControl/motionbricks/out/native_mb_control_20261006/continuation-official-20261006/`.

- Root resumes the complete 45k checkpoint on GPU 0; Pose resumes the complete
  12k checkpoint on GPU 3. Optimizer and scheduler states are restored.
- Both stages retain batch size 32, seed 7, the 2,000,001-update scheduler,
  native network/loss/sampling configuration and the existing SOMA dataset.
  Resume preflight found no changes to the checkpoint-bound training sources.
- The expanded SOMA30 dataset retains 2,003 train clips (4.7007 hours), with
  validation and test separated. Its manifest SHA256 is
  `b2cff543f93f7a3f04a177ae6f7aacbced45c556da07db02a23ec92e45fa29e2`.
- The native tokenizer100k remains frozen, including its code space and
  parameters; Pose does not train against a moving tokenizer dependency.
- Milestones are 100,000 and 220,000 completed optimizer updates per stage.
  At exactly 200,000 completed updates, no actual update has used the
  four-per-end eligibility threshold (steps 0..199999). By 220k, 20k updates
  are eligible to sample four among counts 1..4; not every update uses four.
  This remains a partial native training horizon, not a convergence claim.
- RNGs are reseeded at resume. Data-loader cursor/RNG restoration is not
  bitwise-equivalent to uninterrupted training; this limitation is recorded.

`scripts/continue_native_mb_control.py` only schedules the unchanged native
training entry point and checks each completed milestone before continuing.
Training retains finite-gradient checks, actual-update accounting and the
frozen-tokenizer tensor digest check. A failed stage is recorded as failed;
no batch skipping or silent success is introduced.

`scripts/review_native_mb_continuation.py` waits for both stages' completed
contracts at each milestone, then runs held-out Root -> Pose -> decoder
in-between generation and the existing same-weight Smart Locomotion cases,
including the full native conditions and diagnostic playback/mask controls.
It checks GPU 4 is still idle before each GPU evaluation. The renderer receives
the generated-token, model-predicted-root review rather than reconstruction
tokens. Numeric reports precede rendering; failures preserve logs and do not
become quality approval. The reviewer process is detached alongside training.

Remote `launch.json`, `root_state.json`, `pose_state.json` and
`review_state.json` record commands, processes, stages and outputs. Local copies
are snapshots, not live status. Reports will appear under `review-100000/` and
`review-220000/`. Automatic review completion is not approval of natural motion,
foot contact, RTX3080 latency or UE PIE30 performance. No new model has yet
been deployed to UE; there is no new motion-quality result from this running
continuation.

At launch verification, Root advanced beyond 55,500 updates and Pose beyond
19,500 with finite logged losses; both wrappers and training processes were
alive. The automatic reviewer was alive in `waiting_for_training`, targeting
the paired 100k milestone. The full local test suite passes 627 tests (10
warnings); orchestration tests cover waiting, paired completion, command
execution, rendering input, report persistence and failure propagation. All
three evaluation/render entry points also passed their CLI smoke checks on g3.
These are launch/integration checks, not evidence of improved motion quality.

## Budget Gate and Performance Measurement

The user subsequently requested keeping the current training configuration,
while checking that longer training is worth its cost. The 100k milestone is
now a budget gate: current Root/Pose training continues normally, but the two
continuation wrappers alone are held with SIGSTOP. The independent paired
reviewer stays running. This prevents automatically launching the 220k stage
before reviewing 100k evidence, without changing or restarting a training child.
The gate is recorded in remote/local `continuation_gate.json`.

Only the wrapper PIDs were signaled, after validating their Linux command lines,
stage and output directory; neither native training children nor process groups
were signaled. While held, the wrapper state JSON files remain at `running` even
after a training child exits. Inspect each milestone's `run_contract.json`,
`training_result.json` and `review_state.json` for actual completion. Releasing
the verified wrapper PIDs with SIGCONT resumes normal verification and then
the existing unchanged 220k command. Do not release them merely because loss
fell or a report file exists. Preserve checkpoint/optimizer/scheduler state.

At 100k, review the held-out generation and the one-per-end cases that match
actual training exposure, against the frozen45k/12k reference with the same
inputs, precision, seed and scoring. Check bounded full-body/root motion,
command response, replan discontinuities and the available generated-motion
videos; execution success and finite values alone do not imply naturalness.
The four-per-end cases at this milestone are out-of-curriculum stress tests,
not a reason to demand training exposure that only starts at 200k. If the
in-curriculum evidence supports progress, the unchanged 220k stage is the next
test of increased-keyframe exposure; it is not proof of final convergence.
If evidence is poor, investigate a concrete data/condition/training-path defect
before spending more GPU time. Contact quality and UE/RTX3080 acceptance remain
separate unproved requirements. The single-GPU batch32 SOMA pilot is not the
same training scale as the shipped multi-node batch128 configurations.

On idle GPU4, `profile_native_mb_control.py` injected Lightning's SimpleProfiler
into short, isolated clones of the unchanged native entry point. Each resumed
the original full Root45k/Pose12k checkpoint for 200 actual updates, with the
same batch32, seed7, schedule and frozen tokenizer. These timing clones are
not production checkpoints or motion-quality experiments. No production
training source, hyperparameter, loss, sampler or optimizer was modified.

| Stage | Workers | Steady training batch CPU wall time | Steady loader wait |
|---|---:|---:|---:|
| Root | 2 | 36.52 ms | 0.97 ms |
| Root | 8 | 36.92 ms | 1.16 ms |
| Pose | 2 | 48.75 ms | 1.16 ms |
| Pose | 8 | 47.77 ms | 1.32 ms |

The loader-only test improves from 27.09 to 6.62 ms with eight workers, but this
does not translate to substantial end-to-end gain: prefetch already overlaps
loading with training. Differences in the short training test are too small
to justify a configuration change. Measurements are single-run, on a shared
host, with profiler overhead and worker-dependent random sampling streams.
Profiler regions overlap. Gradient-check wall timings include CUDA queue waits
and must not be described as their removable, exclusive GPU cost. Preserve
finite-gradient and frozen-tokenizer checks. The sampler's CPU/NumPy round trip
is part of the official path, not permission to change its sampling rules.

Performance records are under
`out/native_mb_control_20261006/continuation-official-20261006/performance/`.
Unused GPU memory and instantaneous utilization percentages alone are not
evidence that DDP or a larger batch would improve wall time. GPUs1/2 belong to
other active jobs; no other owner's process was stopped or changed.

## Paired 100k Milestone Reviewed and Budget Released

Both native stages completed 100,000 actual optimizer updates, including their
held-out stage reviews, with zero skipped batches. Root added 55k updates in
2,019.93 seconds; Pose added 88k in 4,406.25 seconds. The unchanged tokenizer
dependency and complete resume state were verified again before continuation.

The 57 paired in-between validation windows have identical source-window hashes,
seed, conditions, scored frames, precision and inference sampling configuration
to the Root45k/Pose12k reference. Conditional generation is non-unique: these
metrics measure deviation from one held-out reference, not proof of naturalness.

| Paired generation metric | Root45k/Pose12k | Root100k/Pose100k |
|---|---:|---:|
| World FK MPJPE | 130.92 mm | 68.90 mm |
| Root-relative FK MPJPE | 79.16 mm | 42.62 mm |
| Root FK error | 91.45 mm | 49.91 mm |
| Joint velocity error | 0.3894 m/s | 0.2791 m/s |

Native one-per-end stage validation also improves: Root position error is
62.29 -> 48.28 mm; Pose known-root generation error is 97.83 -> 41.24 mm.
Smart Locomotion comparisons retain the same source bank, seed and commands.
The one-per-end eight-second rollout remains bounded at pelvis height
0.9817..1.0110 m, with peak root planar speed 0.9818 m/s. Raw turn-command
velocity deviation improves, while forward/stop deviation does not. Seam
joint-speed mean decreases but its peak increases. These mixed results are
not acceptance of continuity, foot contact or responsive locomotion.

Full four-per-end Smart Locomotion still diverges, with extreme finite root
positions and speeds, and is not approved for UE. At 100k it remains outside
the native keyframe curriculum. That fact motivates the next bounded exposure
test, but does not establish training length as the sole cause or promise a fix.

The verified learning signal and preserved in-curriculum boundedness support
the unchanged 220k experiment. Fresh idle-GPU checks, exact checkpoint hashes,
configuration/seed/batch/scheduler bindings, frozen tokenizer and source-bound
resume preflight passed for both stages. Only the verified held wrapper PIDs
were released with SIGCONT; they launch the original 220k commands. The budget
decision is in `gate_review_100000.json`; `continuation_gate.json` records
`released_for_native_curriculum_to_220k`. `quality_approved` and `ue_deployment`
remain false. No hyperparameter, model, sampler, loss or output was changed.

The remote renderer had an older hard-coded source-token caption despite correct
generated-token arrays/metadata. The tested local renderer was synchronized and
rerun into `review-100000/inbetween-render-v2/`, preserving original artifacts.
The copied motion NPZ hashes are unchanged. These corrected source/generated
videos are keyframe-conditioned in-between examples, not WASD demonstrations.
The captions now explicitly identify generated tokens and model-predicted root.
The original `inbetween/` numerical report remains valid; its old rendered
captions must not be used. Renderer/continuation/gate regression tests pass 28
tests; this artifact-only correction does not change model quality or training.

## Full Basic Planar Scope and Paper Optimization

The user's subsequent direction supersedes the pilot's manual milestone gates:
continue automatically across all known basic flat-locomotion candidates and use
the optimization settings disclosed in MotionBricks section 7.4. This is not a
claim that every possible locomotion record in the full BONES release has been
identified. The frozen candidate list contains 11,634 training candidates;
conversion retains numerical quality exclusions rather than accepting bad data
to meet a count. The original 19 validation and 20 test records remain held out.

Source expansion is running on g3 as PID 3661760 into
`${PRIVATE_STORAGE}/motionbricks-bones/expanded-planar-all-v1`. Native conversion targets
`${PRIVATE_STORAGE}/motionbricks-bones/native-mb-soma30-all-planar-v1`. Official feature
conversion, FK, contact representation and training-only normalization remain
in use. The source floor inherits the previously documented proxy-based
calibration; it is not a new surface-ground calibration or physics truth.

The `paper` profile uses Adam, learning rate 5e-5, 10,000 warmup updates,
cosine decay to 2e-6, batch 256 per GPU, 12..64 frames at 30 Hz and 2,000,000
actual optimizer updates per stage. Adam betas, epsilon and zero weight decay
are standard PyTorch defaults, not hyperparameters explicitly reported in the
paper. The original public training_step, losses, tokenization, keyframe
curriculum and inference classes are retained. The `release` profile remains
the default and does not silently acquire these paper scalar settings.

The public executable architecture is the authority where prose is ambiguous:
the paper says Root width 512 with 12 heads, while the public implementation
requires divisibility and uses 16. The paper's tokenizer width/depth description
also differs from its shipped hparams. These differences, SOMA30 versus the
paper's human27 representation, the smaller flat-motion dataset and fewer
physical GPUs are explicit deviations. Batch 256 on one GPU is not the paper's
32-GPU global batch. No exact reproduction or equivalent compute claim is made.
The paper source is https://arxiv.org/html/2604.24833v1#S7.SS4.

Automatic batch-256 checks first exposed a genuine shortest-window defect:
`sample_keyframes` asserted that all eight configured middle poses could fit
even when a 12-frame window provides four middle slots. The existing sampler
already caps its sampled count to available slots. Removing that obsolete
assertion activates the existing cap without changing long-window randomness
or the count distribution. Tests cover empty/short/long slots, invalid
probabilities, long-window RNG parity and the actual 12-frame official Pose
forward/backward path with a frozen tokenizer. Including source preparation,
the final local suite passes 711 tests (11 warnings).

The failed technical check is preserved under
`out/native_mb_paper_soma_20261006/technical-checks-b256-frames12-v2`.
The corrected check is under `technical-checks-b256-frames12-v3`, manager
PID 3695590. It runs 200 real updates per stage solely to validate the actual
training path and batch memory; those weights are not production models.
All three stages passed 200 actual optimizer updates with Adam and no skipped
batches. Tokenizer/Root/Pose training took 49.48/51.55/57.33 seconds and peak
allocated GPU memory was 2,830.55/4,268.00/9,894.04 MiB. The workflow's artifact
verifier independently accepted the completed checks. These short, early-warmup
measurements establish executable training and memory headroom, not convergence,
final throughput or natural motion.

`scripts/run_paper_mb_workflow.py` queues native conversion, fresh Tokenizer and
Root training, fresh Pose training bound to the final frozen Tokenizer, and
fixed held-out generation plus Smart Locomotion review/rendering. It verifies
actual updates, Adam identity, no skipped batches, dataset/config/source hashes
and the frozen tokenizer tensor digest. GPU availability is checked again at
each launch. Failed stages stop only owned children and publish failures;
finite values or completed reports do not become motion-quality approval.

The active manager is PID 3698685 under
`out/native_mb_paper_soma_20261006/full-basic-planar-v2`. Tokenizer and Pose are
queued for GPU3, Root for GPU0, final review for GPU4. It waits for preparation,
the corrected technical checks and idle hardware; it does not interrupt the
older owned pilots or other users' jobs. There are no manual 100k/220k gates in
this new workflow. At this snapshot the full 2M stages have not started, and
neither new motion quality nor UE deployment is approved.

The earlier release-profile full manager and the first paper manager were
stopped after verifying their exact owned process identities. Their artifacts
remain intact; the continuing source expansion and unrelated training were
not stopped. `workflow-v2-launch.json` and `workflow_state.json` are the current
execution records, not the historical manager receipts.

## LAFAN Resolved Supplementary Source

The user supplied https://github.com/orangeduck/lafan1-resolved. The author's
source describes a 60 Hz MotionBuilder re-solve onto the Geno/ZeroEGGS skeleton
with toe motion, not a SOMA motion release. Raw source has been downloaded once
to `${PRIVATE_STORAGE}/motionbricks-sources/lafan1-resolved-v1` on g3. The BVH archive
is 1,064,494,185 bytes, contains 77 BVH files and has SHA256
`94d209e2eff61751e372f7451312db643a1732924079dc4f6ea826fd7b50b5d1`.
The observed motion hierarchy has 75 named joints. Filename selection yields
16 walk/run candidates, not automatically approved flat-motion segments.

`scripts/prepare_lafan_resolved_source.py` successfully CRC-checked the full ZIP
and staged those 16 candidates into `locomotion-source-v1`. Their declared motion
length is 231,636 frames, approximately 64.34 minutes. All candidate joint names,
hierarchies and channel order match the supplied Geno bind BVH. The nominal
source rate is 60 Hz; serialized Frame Time 0.016667 gives 59.9988 Hz when read
literally. The source-only manifest and receipt have local snapshots under
`out/lafan_resolved_source/`; the raw motion remains on g3.

The actual export uses six channels on every joint, including non-root local
translation. Its `offsets` are `bone_local_positions[0]`, not immutable bind
geometry. The audit therefore preserves each clip's header offsets and measures
their difference instead of rejecting legitimate root positions or overriding
them. The largest observed non-root first-frame offset difference from bind is
0.2236 raw distance units. This is a header comparison, not a retargeting error.
The source gate requires preserving animated per-joint translations and native
Euler ordering; header parsing alone does not verify full animation import.
No bind-alignment fudge or double addition of OFFSET to exported positions is
approved. The exporter source is
https://github.com/orangeduck/lafan1-resolved/blob/main/export_bvh.py.

This is independent supplementary source preparation, not a mid-run change to
the BONES dataset. Geno-to-SOMA retargeting, unit/reference-frame validation,
source-contact preservation and leakage-free train/held-out splits must be
verified before any such data enters training. No joint-name-only rotation
copying, unverified mixed skeletons, runtime IK or output correction is added.
The candidate inventory explicitly keeps `training_eligible=false` and
`SOMA_retarget_verified=false`. The source uses CC-BY-NC-ND 4.0, not a commercial
dataset grant; adapted motion is not approved for public redistribution.

## LAFAN Priority Update

The user subsequently prioritized training on LAFAN after correct retargeting.
The old small continuation/reviewer and BONES paper production queue have
been stopped; their checkpoints are preserved. BONES source expansion has
completed, but its production training is no longer the active priority.

LAFAN has now been decoded and retargeted to SOMA using unmodified GMR/Mink.
Nine of sixteen recordings pass the declared engineering checks: 35.51 minutes
total, including 19.43 train, 8.05 validation and 8.03 test. The seven rejected
recordings are not used to inflate the dataset to the raw 64.34 minutes.
See `lafan_mb_training_20261006.md` for source identities and limitations.

A real batch-size issue was also found: the earlier dataset loader could
yield fewer than 256 rows when the number of source files was smaller than
the batch. Earlier technical-check memory/timing measurements therefore do
not establish actual batch256 headroom. The paper loader now samples real
random crops with replacement, asserts actual batch size and records it;
the original factor-two MB augmentation remains unchanged. Fresh GPU checks
use the new contract before the LAFAN 2M-stage runs.
