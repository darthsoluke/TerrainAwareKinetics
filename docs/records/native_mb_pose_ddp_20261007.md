# Native MotionBricks Pose DDP Continuation

This five-rank run was superseded at its saved15k checkpoint by the
user-authorized fixed-global-batch efficiency continuation. Current execution
and measured speeds are in `native_mb_pose_efficiency_20261007.md`. The
following record preserves the original five-rank configuration and results.

## Scope

User request: use original DDP multi-GPU training, with no model or objective
changes. The runtime uses torchrun plus Lightning DDP/NCCL on a single host.
Official release Pose configuration specifies DDP; its model, losses,
optimizer and schedule remain authoritative. No IK, smoothing, new contact
objective or root replacement is introduced.

References:
- Official release: `out/motionbricks_pose/version_1/config.yaml`.
- Standard launcher: https://lightning.ai/docs/pytorch/stable/accelerators/gpu_intermediate.html

## Active Run

Remote root:
`${PRIVATE_STORAGE}/motionbricks-preparation/GR00T-WholeBodyControl/motionbricks`.

Active workflow:
`out/native_mb_lafan_paper_20261006/pose-ddp-v1/training-v2`.
Manager PID at launch: `4052447`.
Production torchrun PID at launch: `4054533`.
Pose run: `training-v2/pose-50000`.

The launch availability check found GPU0-4 idle with no existing compute
processes. These five RTX4090 devices train one Pose model together. GPU5
retains another process and is not used. No unrelated process was stopped.
During successful production updates, GPU0-4 were approximately 92-94%
utilized with approximately 16,106 MiB resident memory each. This is a
transient observation, not a reservation or guaranteed future state.

## Budget and Resume

Tokenizer and Root use the selected exact 50,000-update checkpoints in
`out/native_mb_lafan_paper_20261006/stage50k-v1`.

The single-GPU Pose run was stopped after exporting its exact saved 5,000
update checkpoint, preserving weights, Adam moments, scheduler and loop state.
Its snapshot is `pose-ddp-v1/prepare/pose-resume`.

The first DDP check reached 5,200 updates. An aggregation-guard unit test
then found that different self-consistent per-rank batch declarations were
not rejected. The guard was fixed; the already saved 5,200 checkpoint was
preserved. An initial continuation with no later saved checkpoint was stopped,
and its unsaved updates are not counted in the selected continuation lineage.
The adapter-only revision is explicitly recorded in the resumed run contract;
changes to original model/objective/data source files remain forbidden.

The reviewed DDP adapter completed a further 200 actual updates to 5,400.
Production continues from this complete checkpoint to 50,000 total updates:
5,000 single-GPU + 200 first DDP + 200 reviewed DDP + 44,600 production.
The target is not 50,000 additional updates or five independent models.

All five ranks restored the same global step, initialized Adam update count
and scheduler last_epoch. The scheduler horizon stays 2,000,000 with the
original 10,000-update warmup, Adam 5e-5 and final learning rate 2e-6. There is
no automatic learning-rate multiplication for more GPUs. Sampling RNGs are
reseeded on continuation; this is not bitwise uninterrupted replay.

## Sampling

Per-GPU input batch: 256. Global input batch: 1,280. Original factor-two MB
augmentation produces 512 windows per rank and 2,560 globally per update.
The larger global batch is an explicit consequence of DDP, not equivalence
to the paper's original GPU count. Processing more samples per update does
not imply five-fold faster completion of 50,000 updates.

The train split still contains five accepted LAFAN recordings. Standard
Lightning replacement of a RandomSampler would instead shard the five-file
dataset directly, losing its replacement window budget. The DDP adapter
explicitly uses Fabric's DistributedSamplerWrapper around PyTorch's
replacement RandomSampler, and disables a second automatic replacement.
Each rank shards draw positions from the same seeded sequence, then fetches
independently cropped real recording windows with rank-aware worker seeds.
Recording IDs can repeat; draw positions are sharded. Each rank retains 256
batches per bookkeeping epoch. No generated motions enter training.

## Automatic Checks

Every rank checks device/sampler mapping, true input batch size, finite
gradients, real optimizer updates, and the supporting tokenizer's eval mode,
frozen parameters and absent gradients. At completion every rank publishes
a separate receipt. Checkpoint calls remain collective; only rank0 writes
shared configuration, contract, result and checkpoint bytes.

The 5,200-to-5,400 check verified:
- 200 actual updates per rank and zero skipped batches.
- Input batches exactly [256] on every rank.
- Restored Adam/scheduler/global step exactly 5,200 on every rank.
- Identical final backbone tensor digests across all five ranks.
- Unchanged frozen tokenizer digest on every rank.
- Original configuration/data and all original model/loss source hashes
  identical to the single-GPU snapshot.

Frozen tokenizer tensor digest:
`2d9cda1e1b5da399c9cef48d9a8e676adcd48e4f22c860a00705dce9f91d66c4`.
Reviewed technical checkpoint SHA256:
`75b3513d7a6a422fc54b87c898a1f15ebee94e0df5ad02c0ff362ed06b7943db`.

Each technical check took roughly 49 seconds for 200 updates; maximum
allocated GPU memory was approximately 10,393 MiB per device. This differs
from nvidia-smi resident memory. Production logs observed roughly 4.5-5.1
shared updates/second after startup; variable window lengths affect timing.

Full local regression suite: 998 passed. Additional restored-state receipt
aggregation checks were replayed on the real five-rank technical result.
The active remote training entrypoint is kept unchanged at its launch version;
that version already enforces exact restoration in every rank's startup
callback. Its executed source copy and hash remain in each run directory.
The later local aggregation-only validation enhancement does not justify
restarting or replacing a running training program. Future adapter revisions
must be explicit and cannot authorize original model/loss source changes.

## Delivery Boundary

The workflow will verify all five final receipts and frozen tokenizer, then
run native Tokenizer reconstruction and Root/Pose in-between validation on
subject4 walk/run windows. Subject5 remains held out. The 50k budget is
recorded explicitly; this is not completion of 2M paper training. Model
quality, continuous WASD, UE deployment and RTX3080 latency are not approved
by DDP launch success or declining training loss.

Local technical result:
`out/native_mb_lafan_paper_20261006/pose-ddp-technical-v2-result.json`.
