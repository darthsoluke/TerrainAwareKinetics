# PARC Expansion and Mixed Training, 2026-10-10

Current result and resource policy: `parc_long_route_results_20261010.md`.
The user now reserves GPU0/7; use only GPU1..6. The old short-course workflow
below has been stopped in favor of continuous long-route collection.

The finite plan and launch PIDs below are historical. The user's expanded
duration request is now handled by `parc_duration_campaign_20261010.md`: rolling
diverse scenes, a one-hour minimum milestone, a two-hour new-training-motion
target, per-scene admission and a gated training receiver. The latest instruction
allows all idle A100 cards for collection, superseding the original two-card cap.

## Active Instructions

Generate broad, varied PARC motion/terrain pairs first, then train together with
the existing BONES-SEED flat locomotion. Use all idle A100 cards for collection,
checking occupancy before each dispatch. The original allocation was GPU2/3
because GPU0 was occupied before generation launch.
Live `nvidia-smi topo -m` confirmed GPU2<->GPU3 is NV12 on NUMA0;
`nvidia-smi nvlink --status -i 2,3` reported all12 links active on both devices,
25GB/s per link. PARC's two independent scene workers are actually generating
on this pair. Subsequent MB DDP uses the same pair; independent generation does
not need gradient communication. Do not infer training communication throughput
from nominal link speeds or GPU utilization.
Do not use subagents. Preserve original MB/PARC methods, and keep local-PC-only
interactive inference. Large transfers use private Google Drive or NAS, never
public-IP SCP. Small source scripts and a49KB packaging-tool wheel used SSH;
the3.5GB runtime package uses Drive.

## Original Finite Generation Plan (Superseded)

`scripts/build_parc_diverse_course.py` defines96 distinct scene groups:
16 each for stairs, platforms, obstacle blocks, gaps, blocky rough surfaces and
mixed continuous routes. The mixed layouts combine detours, gap traversal,
stairs, ledges and rough patches, with explicit geometry variation. Requested
terrain family is not a claim that the generated motion demonstrated a skill.
Climbing/hand support must be established from actual recorded motion.

All96 scenes passed CPU geometry preflight;96 unique heightfields and no
cross-split duplicates were observed. Each family is assigned12 train,
2 validation and2 test layouts before generation or scoring. All candidates
and retries from one layout retain that split.

The generator is the already verified official iteration3 EMA checkpoint,
with released CFG/DDIM settings,128 candidates per layout attempt, up to3
attempts. At most2 candidates per scene proceed through the released selection
thresholds,3000-step official motion optimization, official Isaac Gym tracker,
full-frame recording and official export.192 is a maximum selected count,
not a promised number of accepted training clips. No thresholds were loosened.
This generation is offline; no user-facing realtime claim is made for PARC.

The varied terrain builders call upstream `terrain_util` and upstream A*.
The exact source method inserts interpolated guidance across jump edges;
those intermediate points are not falsely asserted to be ground supports.
Both start/end support heights remain checked. Random geometry and A* routing
are seeded independently of diffusion, preserving layouts across retries.

## Runtime Migration and Jobs

The first4090D collection under
`${PRIVATE_STORAGE}/motionbricks-parc-physics-20261009/diverse_20261010_v1`
generated and optimized initial platform/rough candidates. Two integration
issues were found and corrected before admission:

- Include the existing physics environment's bin directory, including Ninja,
  in the recorder subprocess PATH.
- The upstream terrain builder computes `int(padding/dx)`. With float32
  dx=0.40000000596 and padding=0.4 it truncates the intended1 padding cell to0
  and fails at the mask slice. `align_terrain_spacing` now expresses padding
  as the validated integer count times the actual source spacing. The0.2m
  case also retains its intended2 cells. Motion, tracker, policy rays and
  terrain surface heights are not modified. Regression tests cover both cases.

Other workloads then occupied both4090D GPUs. Our workers were stopped without
signaling unrelated tasks. The initial outputs and failure reports remain for
review. The new collection migrates to the two available A100 cards using a
packed copy of the existing Python3.8/Isaac Gym environment and official assets.
Do not resume the old4090D workers in parallel with this new collection.

Historical remote roles and initial launch PIDs (superseded by the duration campaign):

| Role | Location | PID at Launch |
| --- | --- | ---: |
| Runtime packing/Drive upload |4090D `runtime_transfer_20261010_v1.log` under physics NAS base|3814228|
| Upload retry supervisor after the initial upload process |4090D `runtime_transfer_20261010_v1.retry.log`|3852022|
| Runtime receipt, relocation, two-card PARC collection |A100 `parc_collection_a100_v2`|2710999|
| CPU SOMA admission and incremental native dataset build |A100 `diverse_dataset_a100_v1`|2659235|
| Local dataset receipt, BONES merge, matched training and evaluation |A100 `diverse_mixed_a100_v3`|2711000|

A100 task base is `${PRIVATE_STORAGE}/motionbricks-20261010`.
Its deployment code is `code_feedback_fix_v1`; migrated PARC runtime is
`parc_runtime_v1`. The collection bootstrap verifies package SHA256, runs
conda-unpack, checks Isaac Gym/Torch loading, then rechecks the two physical cards
before launch. Relocated Python3.8, Torch2.2.0+cu118 and Isaac Gym loaded on A100.
The finalizer is CPU-only on A100. Its existing MuJoCo3.8.0 environment was
completed with the same Mink0.0.13, qpsolvers4.13.0 and DAQP0.10.3 as4090D.
Generation, physics, retargeting, dataset construction and MB training all
remain on A100 storage. Generated datasets are not exported back to4090D.

Private Drive paths:

- `mb-transfer:environments/parc-20261010`
- `mb-transfer:runs/parc-diverse-mixed-20261010-v1`

Only the initial existing environment/model migration and final local-PC
artifacts use Drive. The old cloud-roundtrip finalizers/receivers were stopped;
their waiting-state files are historical, not the active jobs. The private
Drive shared application briefly returned403 project-wide request quotas;
uploads and receivers retry without switching to public-IP file transfer.
The476-byte checksum receipt was relayed separately to A100 so the verified
environment download can start while the asset upload finishes; both large
archives still travel exclusively through Drive.

## Dataset Admission and Training

`finish_parc_diverse_dataset.py` consumes full physics recordings only. It
retains exact source/export/report hashes and existing SOMA geometry/continuity
screening, excludes failures and partial suffixes, and keeps all old official
PARC splits unchanged. Incremental native conversion verifies every existing
artifact and preserves old shard bytes and records without requiring the old
raw files from4090D. Only new generated records are retargeted/encoded. The new
scene splits are also preserved. Exact scene
duplicates spanning splits reject construction instead of being reassigned.

The existing native converter is reused with a verified-statistics option.
Its affine statistics remain byte-identical to the source checkpoint's PARC
train statistics. This is checkpoint compatibility, not a claim that they were
re-estimated over the new corpus. The final A100 merge reuses all accepted
BONES flat clips and preserves their original splits and source bytes.

After the local dataset is ready, the queued experiment trains two matched arms,
10000 updates per stage in each arm, using the native MB objectives. Tokenizer
and Root run independently on GPU2/3 at input batch256; Pose uses both cards,
batch1024/GPU, BF16 student withFP32 labels/frozen Root. Adam5e-5, warmup10k,
cosine2M and clipping0.5 remain explicit. This is a bounded comparison, not a
claim of completed paper-scale training. Periodic checkpoints and fixed
source/category validation are retained.

The two arms compare the current pelvis-origin coordinates and a new observed
terrain-origin option. Both are evaluated for60s rest/start-stop and24 held-out
terrain windows with actual generated-FK feedback. No new bundle is
automatically installed in the local browser. Flat success alone does not
approve parkour, and native loss values across different coordinate systems
must not be compared as the primary motion-quality result.

## Underlying Drift Evidence

The deterministic four-pass argmax probe is
`feedback_mechanisms_v2/mechanisms.json` on A100; v1 used default stochastic
sampling and is superseded for paired comparisons.

On two held-out quiet seeds,12 generated eight-frame blocks reduced pelvis Y
from0.99361 to0.92084m and0.99423 to0.94929m. Teacher-history controls ended
near0.98961/0.99018m. Re-encoding changed root position by less than1e-9m;
cropped-vs-padded decoder height differed by less than4e-7m. The repeated
downward bias originates in model prediction, not observed-history integration
or padding. These two seeds locate an error mechanism, not every motion defect.

Shifting only the body by +/-0.1m while keeping ground fixed shifted the
decoded future by almost the same amount. The existing pelvis-origin scheme
removes pelvis clearance from the tokenizer's coordinate frame; only terrain
adapters can supply it indirectly. The candidate instead subtracts the actual
current near-patch center height from every world-Y feature and restores the
same constant after decoding. It preserves the observed body's height above
ground, leaves all future heights predicted, and never projects output to
terrain. Missing current ground remains an error rather than an invented plane.
It is an explicit hypothesis requiring retraining and paired validation.

The candidate's original three training paths,20-update two-card integration,
strict bundle export/reload and short rollout execution passed. The20-update
smoke model still floats and fails the motion-quality gate, as expected for
an untrained coordinate change; it is not a fix or a deployment artifact.

Verification:108 focused tests passed before the added terrain-frame translation
and incremental-conversion cases; both additions also passed their focused suites.
Checks include varied-scene
determinism, official padding-cell semantics, source statistics hash checks,
online unknown-future masks and native gradient paths. The incremental integration
test removes access to old raw inputs and confirms unchanged prior records,
shard bytes and statistics. Full generation,
new-data training and motion-quality acceptance remain running or queued.
