# A100 Mixed Training Review, 2026-10-10

## Decision

The native MB stage objectives are a valid motion-learning baseline. They do
not establish a terrain-aware closed-loop controller by themselves. The latest
50k mixed run failed the four 60-second rest/start-stop cases. Do not deploy
it, or treat lower reconstruction loss as evidence that drift is fixed.

The performance work below preserves this experiment's objectives. It makes
larger experiments practical and corrects checkpoint selection and validation;
it is not a motion-quality result. No live browser weights were replaced.

## Official Method and Our Extensions

Reference code: upstream checkout commit `b307976`, with local diffs inspected.
The paper's sections 4.1-4.3 and 7.4 were read directly on 2026-10-10:
https://arxiv.org/html/2604.24833v1

| Component | Retained from MB | Explicit extension/adaptation |
| --- | --- | --- |
| Body representation | Native dual global/local Root and body feature pipeline | SOMA30 and official SOMA FK/skin, with derived feature dimensions |
| Tokenizer | Multihead EMA VQ, four frames/token, conditional decoder, reconstruction/commit/contact-velocity/joint-velocity objectives | Shared observed-pelvis vertical coordinates; four required observed decoder frames; joint-velocity boundary uses actual context FK |
| Root | Native trajectory regression, global/local losses and token-duration classification | Current local terrain, component availability masks, terminal planar velocity/heading goal, known horizon |
| Pose | Native masked/corrupted token prediction and cross-entropy | Terrain attention and predicted relative Root height; frozen Root predictions replace teacher trajectories as conditions |
| Inference | Root -> masked-token Pose -> native conditional decoder; decoder retains Root refinement | Only played FK enters the next history; no future pose or height supplied as observation |

Terrain uses two local 9x9 XYZ/valid patches, encoded by convolutions into 18
continuous tokens and gated cross-attention. This is not a second VQ terrain
tokenizer or a visual RGB encoder. Near support stays fixed; forward extent
depends on commanded speed and horizon. Heightfields cannot describe multiple
surfaces at the same XZ, overhangs or arbitrary climbing handholds.

The original paper trains with variable authored keyframe constraints. Our
velocity-only online interface is a harder and different conditional task.
Source terminal velocity is a hindsight goal label, not recorded WASD intent.
No claim is made that this is an unchanged reproduction of the official demo.

Optimization retains Adam5e-5, 10k warmup, 2M cosine horizon and clipping0.5.
The 50k run was a weights-only adaptation with a fresh optimizer and global
input batch256. It was not the paper's 2M updates at batch256/GPU over32 GPUs.
At50k the retained cosine schedule is still almost at its peak learning rate.

## Evidence Against Simply Extending the Existing Run

Run: `${PRIVATE_STORAGE}/motionbricks-20261010/mixed_velocity_50k_v1`.
Bundle SHA256: `de0aa8c351dea859339422acf151af83cbada3b86ab47babf7022243502c64a8`.

- Tokenizer validation:0.07168 ->0.03595.
- Root: best0.06996 at10k, final0.07755.
- Pose: best1.84744 at5k, final2.10848.
- Original validation was five equally weighted batch means, including a
  smaller final batch. It did not report source/category losses separately.
- Original checkpoint callback retained only the latest periodic checkpoint;
  early validation-best checkpoints were already absent on disk. This cannot
  be repaired retroactively by relabeling the final checkpoint as best.
- Four60s closed-loop tests all failed. Stationary cases drifted about1.40m.
  P99 penetration was0.338-0.740m across rest/start-stop cases. These are real
  failures in the generated output, not frame-rate measurements.

This training uses source histories and short12-64-frame windows. It has no
multi-replan generated-history training objective. Pose conditions on predicted
Root, but that only addresses one part of the train/runtime distribution gap.
The tokenizer still samples native random future keyframes in addition to the
four forced observed frames; runtime supplies no future poses. MB's original
curriculum supports variable constraints, so this is a hypothesis to test by
a matched continuation ablation, not proof of the dominant drift cause.

## Actual Training Corpus

| Origin | Train Clips | Train Hours |
| --- | ---: | ---: |
| BONES-SEED |2003|4.70532|
| PARC official |437|0.50668|
| PARC generated |4|0.01796|

The native dataset remains `mixed_control_full_v3`, manifest SHA256
`88f7023556e8f32334c5b7a8c5ed18d8bf176f062b31390e9891915eda94574a`.
All records are already available on A100; no large public-IP transfer occurred.
PARC includes only four dedicated stairs-category training clips, four climbing,
three jumping and three vaulting; mixed terrains may contain additional actions.
More repeats of this roughly5.23h corpus are not more independent training data.

The67 validation clips include19 BONES records in idle/start/stop/walk and48
PARC records. There is no dedicated validation stairs category, or BONES
run/turn/backward/lateral category. Repeated fixed crops improve temporal
coverage, but do not create new independent clips or fill those missing skills.

## Performance Changes Implemented

1. Global/per-GPU batch and training precision are explicit run-contract fields.
   The workflow supports2/4/6/8 available GPUs. Pose may use BF16 while other
   stages retainFP32. Frozen dependencies and shared statistics remain bound.
2. Source coordinate transforms, frozen token labels and frozen Root conditions
   remainFP32 under student AMP. CPU regression tests and actual A100 hashes
   verify identical targets/conditions between precision profiles. The corrected
   paired Pose probe has0.123% initial-loss difference and0.9977 sampled-gradient
   cosine; this is numerical screening, not long-training quality equivalence.
3. Large terrain batches now use bounded packed chunks. Previously the16M-cell
   limit switched the entire batch to sequential source queries. Query coordinates,
   invalid cells, units and output ordering remain unchanged. Ragged-map tests
   cover both range policies and forced chunk boundaries.
4. Pose DDP removes unused-parameter graph traversal and enables gradient bucket
   views. Real six-rank training completed100 updates without unused-grad errors.
5. Validation uses fixed multiple crops, source/category reports and sample-weighted
   batch means. Evaluation batches are partitioned across ranks with per-batch
   deterministic seeds. Missing control categories are explicitly reported.
6. All periodic checkpoints are retained. Completed stages select a validation
   candidate and preserve the final weights separately. Pose and exported bundles
   load the exact selected, frozen Tokenizer/Root dependencies. A real integration
   run selected Root step50 over step100 and used that checkpoint for Pose.

## Measurements

Single-card worst-length benchmark:64 frames, five warmup plus15 timed updates,
Adam, backward and clipping included. Data are cached real CPU mixed batches
including host-to-device copies; these timings exclude cold disk loading and
are not an end-to-end convergence comparison. Batch refers to input clips;
native window sampling expands it by2.

| Profile | Before Packed Chunks | After Packed Chunks | Allocated Peak |
| --- | ---: | ---: | ---: |
| Root FP32, batch1024 |1870ms/update|495ms/update|18.3GiB|
| Pose BF16, batch1024 |2064ms/update|575ms/update|22.8GiB|

The paired input digests, initial targets, and printed update0/update10 losses
match across the chunk change. Corresponding sample throughput is2063/s for
Root and1771/s for Pose. Increasing Pose to batch2048 reaches1817/s but uses
43.6GiB: approximately2.6% more throughput for almost twice the activation memory.
It is not a reason to keep increasing batch just to occupy80GB.

Additional profiles: Root FP32 batch256 was2226 input samples/s; Tokenizer FP32
batch512 was1909/s. Root batch1024 is not faster per example than batch256.
Final per-stage settings must follow throughput and validation, not equal batch
sizes or equal memory occupancy across differently sized models.

Remote benchmark receipts: `benchmarks/a100-scale-v1/` under the A100 task base.
Integration receipts: `smoke_scale_v1` (20 updates/stage, two-rank Pose) and
`smoke_scale_six_v2` (100 updates/stage, three-rank independent stages then
six-rank BF16 Pose). These are mechanics/performance runs, not approved weights.
Six-card batch256 Pose measured2159 input samples/s; sampled GPU utilization
was94-100%, which includes communication activity and is not a tensor-core
utilization certificate.

The additional six-rank Pose run `pose_scale_six_b1024_v1` completed100 updates
at batch1024/GPU, global6144, BF16 student andFP32 teachers. Median update was
0.986s and throughput5752 input clips/s (11504 native sampled windows/s),
allocated peak23.4GiB/rank0. This is2.66x the measured six-card batch256
throughput. The two short DDP runs used different matched dependency snapshots;
this is a capacity/throughput result, not a controlled convergence comparison.
Total stage time including validation and checkpoint I/O was150.4s. Neither
this timing nor an nvidia-smi snapshot establishes time-to-quality speedup.

The practical measured profile is TokenizerFP32/batch512, RootFP32/batch256,
and PoseBF16/batch1024 per GPU, with the shared normalization and teacher
conditions keptFP32. The existing stage CLI expresses exact global batch and
precision; the workflow can override Pose precision independently. Changing
global batch must remain explicit and must not automatically multiply learning
rate. Only currently free devices may be used. GPU2 was left to its other task.

Verification:71 focused local tests passed. The complete six-card staged run
exported and strictly reloaded its matching bundle, SHA256
`4b33449b1f9ded30bf8cb02191cc41a120d62821a8b3b3e473f538f927e6ab80`.
This smoke artifact is not a deployment candidate. No new long training job was
started after the failed50k closed-loop evaluation.

## Next Training Decision

Keep inference on the local PC and the failing new bundle out of the live app.
Use the improved runner for bounded, matched experiments that distinguish
Root error, body-token error and decoder/played-history accumulation. Compare
both source-history and actual generated-FK-history rollouts before committing
to another long training budget. Expand admitted paired PARC behaviors and
independent validation coverage; BONES supplies real standing and starts/stops,
but its fixed plane is an explicit source assumption, not measured terrain.

Preserve native objectives as the baseline. Any causal-context or feedback
adaptation must be isolated and compared against it, without IK, output
projection, smoothing, manufactured future observations or source split leakage.
