# LAFAN Resolved -> SOMA -> Native MotionBricks

## Scope

Prioritize the user's LAFAN flat walk/run source over the BONES production
queue. Keep the original MotionBricks Tokenizer, Root, Pose, objectives,
keyframe sampling curriculum and inference. No runtime IK, foot locking,
root overwrite, output smoothing or custom sole-region objective is used.
Offline data retargeting uses the unmodified GMR/Mink solver with an explicit
Geno-to-SOMA adapter. This is not an official NVIDIA Geno converter.

## Source and Separation

Source: https://github.com/orangeduck/lafan1-resolved

- Archive SHA256: `94d209e2eff61751e372f7451312db643a1732924079dc4f6ea826fd7b50b5d1`.
- 16 walk/run recordings, 231,636 raw frames, approximately 64.34 minutes total.
- Geno has 75 named joints; this is not the SOMA skeleton or UE mannequin.
- Nominal 60 Hz; serialized Frame Time 0.016667 reads as 59.9988 Hz.
- Subjects 1/2/3 are eligible for train; subject4 is validation, subject5 test.
- Statistics are generated only from accepted train recordings. Neither
  validation nor test recording windows enter gradient updates.
- CC-BY-NC-ND 4.0 source: no permission for public adapted-data redistribution
  or commercial use is claimed.

## Retargeting Contract

The author's exporter writes six channels on every joint: animated absolute
local X/Y/Z translations and intrinsic Z/Y/X rotations. OFFSET is not added
again. The decoder preserves non-root animated translations and checks the
hierarchy against the supplied Geno bind. The bind is not forced to a T-pose.
Bone-local anatomical axes, leg-length scaling, units and a proper Y-up/Z-up
rotation define the mapping. MuJoCo provides only kinematics/Jacobians;
placeholder inertias are not a biomechanical SOMA model.

GMR commit: `bb1bbe40774794fceb2a7c579a3464a28e68c844`.
The upstream retargeting core is checked against that Git commit before use.
SOMA FK is cross-checked in non-neutral poses. The current conversion uses
feet/toe position cost 500, with the same settings for every recording.
Official SOMA selected-vertex LBS matches full-mesh region heights exactly
on the parity samples; fixed single-point foot proxies are not substituted.

The reference and tracking bounds are declared in the conversion report.
They are engineering acceptance limits, not claims of captured force/contact
truth. Stationary source foot/toe anchors are only kinematic evidence. The
one fixed recording plane estimated from SOMA sole surfaces is not a measured
capture plane. No per-frame vertical correction is performed.

## Current Accepted Dataset

Remote source: `${PRIVATE_STORAGE}/motionbricks-lafan-resolved/soma60hz-v3`.
Accepted manifest SHA256:
`727afce64c1533742114e8355a5d25e3c447e7b4a913d18b03b60069768c9187`.

| Split | Recordings | Minutes |
| --- | ---: | ---: |
| Train | 5 | 19.434 |
| Validation | 2 | 8.050 |
| Test | 2 | 8.031 |
| Total Accepted | 9 | 35.514 |

Seven complete recordings are rejected and retained for diagnosis. Five
have excessive newly introduced angular steps, one exceeds orientation and
sole-gap bounds, and one exceeds the sole-gap bound. For example, a rejected
walk contains a source leg step of about 1 degree versus a retargeted step
of about 27 degrees. Thresholds are not relaxed to advertise all 64 minutes.
No failed recording is listed in accepted.json. No stitching or replacement
of bad poses is used. Conservative source calibration-pose trimming is
recorded per clip; it is not asserted to remove every static introduction.

The local report snapshot is
`out/lafan_soma_20261006/retarget-preview-v3/retarget_review.json`.
Retargeting acceptance does not approve a learned model or prove zero foot slip.

## Training and Automatic Checks

Target native data: `${PRIVATE_STORAGE}/motionbricks-lafan-resolved/native30hz-v1`.
The converter resamples rotations/root offline and regenerates official MB
features, contacts and train statistics at 30 Hz. FK/feature roundtrips and
all accepted shard hashes are checked. Custom contact labels are not used.

Paper scalar profile: Adam, learning rate 5e-5, 10,000-update warmup, cosine
decay to 2e-6, 2,000,000 actual updates per stage, 12-64 local frames at 30 Hz,
256 input rows per GPU. Architecture follows executable release hparams
where the paper's prose is inconsistent; SOMA30 versus paper27 and a smaller
dataset/fewer GPUs are explicit deviations, not exact paper reproduction.
Reference: https://arxiv.org/html/2604.24833v1#S7.SS4

A real loader defect was found before starting this run: nominal batch256
with fewer recording files yielded only the file count. Paper-profile data
now uses PyTorch replacement RandomSampler and independent random real crops,
with drop_last. CPU caching is bounded to 256 MiB per worker and stores only
un-normalized source features. The original MB factor-two augmentation is
unchanged: 256 input rows yield 512 internally augmented windows. This is
disclosed, not silently claimed to equal the paper's multi-GPU global batch.

Every training batch asserts the actual input size, finite gradients, real
optimizer-update counts and zero skipped batches. Pose checks its frozen
Tokenizer state. Each run records observed batch sizes, sampling settings,
configuration/source/data identities and peak GPU memory.

The automatic workflow first executes 200 updates on each real training
path to validate memory and contracts. Root/Pose technical checks skip the
BONES-specific 57-window evaluator, not the training objective. These
technical weights are isolated and never initialize the fresh 2M runs.
Tokenizer and Root then train independently; Pose starts after Tokenizer
finishes and binds its immutable final checkpoint. Availability is rechecked
before every GPU launch. Failures publish a receipt and stop only owned jobs.

Final review uses fixed subject4 walk/run windows and the original MB
in-between pipeline. Source-encoded reconstruction and generated-token/root
motion are kept separate. Continuous WASD, UE deployment and latency are
not yet assessed or approved by this training run.

## Previous Jobs

The old small BONES continuation and reviewer were stopped without deleting
checkpoints. The BONES paper production queue was stopped to prioritize
LAFAN. BONES source expansion finished independently and remains available.
Earlier batch256 technical-check memory/timing numbers used fewer actual
input rows; they must not be cited as proof of true batch256 memory headroom.

The active launch and runtime outcome are recorded separately after execution,
not inferred from the presence of this document.

## Execution Snapshot

The guarded LAFAN workflow was launched on g3 with manager PID `3846540`.
Run directory:
`${PRIVATE_STORAGE}/motionbricks-preparation/GR00T-WholeBodyControl/motionbricks/out/native_mb_lafan_paper_20261006/full-planar-v1`.
The 30 Hz native conversion completed with split counts 5/2/2 and manifest
SHA256 `b68ad5a0d7b028dc1b5e3075ae84cfa4e5c4dc4fd20fc9aefbf63835a8167d46`.

All three isolated GPU technical checks passed. Each completed exactly 200
real Adam optimizer updates, observed input batch sizes `[256]`, and skipped
zero batches. Pose kept its Tokenizer frozen.

| Stage | Training Seconds | Peak Allocated MiB |
| --- | ---: | ---: |
| Tokenizer | 21.10 | 2823.94 |
| Root | 25.23 | 4268.01 |
| Pose | 39.51 | 9873.26 |

These early-warmup times are technical-check measurements, not full-run ETA
or quality evidence. At launch, fresh 2M Tokenizer training ran on GPU3 and
fresh 2M Root training ran on GPU0. Pose was queued for GPU3 after Tokenizer,
and final validation for GPU4. This plan is superseded by the bounded
50k-stage handoff below.
No technical-check weights initialize these production runs.

The local full test run passed 786 tests. A subsequent focused rerun of the
final workflow/evaluator/renderer/paper loader/check guards passed 48 tests.
No LAFAN production model is yet claimed trained, converged, UE-ready or
natural merely because the jobs are running.

## User-Selected 50k Stage Handoff

The user requested selecting Tokenizer/Root at 50,000 optimizer updates,
then confirmed a fresh Pose budget of 50,000 updates. This is a bounded
intermediate experiment, not completion of the paper's 2M-stage training.
The paper scalar profile, original MB model classes/objectives and 2M
learning-rate schedule remain unchanged; no runtime IK or smoothing is added.

Remote workflow:
`out/native_mb_lafan_paper_20261006/stage50k-v1`.
Manager PID at launch: `4006605`.

Tokenizer's exact saved 50k checkpoint was selected after its old process
had advanced beyond 50k. Root continued to its own exact saved 50k checkpoint.
The exporter byte-copies weights, full Adam moments, scheduler and Lightning
loop state, verifies actual optimizer updates and hashes, and retains source
run/config/data provenance. Original 2M run directories are marked
`stopped_at_user_request`, not completed. No checkpoint or dataset is deleted.
Selected artifacts use `completed` only for their bounded checkpoint export;
their contract explicitly disclaims original-long-run completion and quality
approval. Intermediate timing/peak-memory/skipped-batch callback metrics are
not invented when unavailable.

| Stage | Selected Updates | Run | State at Handoff |
| --- | ---: | --- | --- |
| Tokenizer | 50,000 | tokenizer-50000 | Checkpoint selected |
| Root | 50,000 | root-50000 | Checkpoint selected |
| Pose | Target 50,000 | pose-50000 | Fresh training on GPU3 |

Pose PID at launch: `4011176`. The frozen Tokenizer checkpoint SHA256 is
`51e4df8d3f8100431d3a61d767bdfeeeb1ad42b60ccab47055fb950eb13a4771`.
Pose still uses a single GPU in this launched run. Official Pose release
configuration supports 8-device DDP, but the current audited native runner
uses one device; a multi-GPU launcher requires explicit rank-safe data and
artifact handling before it can be claimed active. The GPU availability
snapshot is transient; other users' processes must not be terminated.

After Pose completes, the workflow verifies 50k actual updates, zero skipped
batches and unchanged frozen Tokenizer tensors, then reviews subject4 walk/run
through the native Tokenizer reconstruction and Root/Pose in-between pipeline.
The reviewer accepts an explicit 50k budget, validates all three actual saved
steps and does not publish a 2M-completion claim. Subject5 stays untouched;
continuous WASD, UE latency and natural-motion quality remain unapproved until
separately demonstrated.

The focused snapshot/handoff/budget/evaluator/control regression run passed
143 tests. Both real selected checkpoints also passed the enhanced integrity
check: Tokenizer has 90 initialized Adam parameter states; Root has 149 out
of 150 declared parameters. Its one uninitialized optimizer state is valid
lazy Adam behavior, not a lost moment tensor. Every initialized state and
the optimizer loop record exactly 50,000 updates. The isolated integrity
receipt records this distinction. Pose subsequently logged successful steps
500, 1,000 and 1,500; this is execution evidence, not convergence evidence.

## DDP Continuation Supersedes Single-GPU Pose

On 2026-10-07 the user requested original DDP multi-GPU continuation without
model/loss changes. The single-GPU Pose checkpoint was preserved at exactly
5,000 updates. A guarded five-device torchrun/Lightning DDP continuation is
now active on GPU0-4, targeting 50,000 total Pose updates with the same 2M
schedule and per-GPU batch256. GPU5 remains assigned to another process.
The active workflow is `out/native_mb_lafan_paper_20261006/pose-ddp-v1/training-v2`.
See `docs/native_mb_pose_ddp_20261007.md` for exact checkpoint lineage,
automatic checks, runtime source versions and delivery boundaries.
