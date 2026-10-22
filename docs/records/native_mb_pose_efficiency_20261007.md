# Native MotionBricks Pose Efficiency Continuation

## Authorized Scope

The user selected two changes: restore the official `matmul_precision: high`
runtime setting, and split a fixed global input batch256 across four GPUs
(64/rank). BF16, model/objective changes, learning-rate scaling, schedule
compression, motion postprocessing and UE changes are not included.

The original MB entrypoint `motionbricks/exp_setup/experiment.py` applies
`torch.set_float32_matmul_precision(conf.matmul_precision)`. The previous
external DDP adapter saved this configuration but omitted that runtime call.
The new adapter applies and verifies it before building the Pose model.
Trainer precision remains32; internal matrix computation follows the official
high setting, not a claim of bitwise FP32 equivalence.

The fixed-global-batch profile is an explicit efficiency deviation from paper
batch256 per GPU. Original factor-two augmentation remains unchanged:
four ranks each consume64 real windows and construct128 augmented windows,
for global256 inputs /512 augmented windows per optimizer update. Equal rank
batches use ordinary DDP mean reduction. Adam/LR/2M schedule remain unchanged.

## Preserved Checkpoint

Remote root:
`${PRIVATE_STORAGE}/motionbricks-preparation/GR00T-WholeBodyControl/motionbricks`.

Workflow root:
`out/native_mb_lafan_paper_20261006/pose-efficiency-v1`.

Selected full Pose checkpoint:15,000 optimizer updates. SHA256:
`b13c664adc06408eab29a622c5774f6934dc2be07d3cc0ae85e558c8314ff297`.

The byte-identical snapshot is in `prepare-v2/pose-resume`. Export verifies
weights, Adam moments, optimizer-loop counter and scheduler identity. Frozen
Tokenizer50k and Root50k are unchanged. No training starts from random weights.

The first handoff attempt observed the manager before asynchronous SIGSTOP
was reflected in `/proc`; it failed and resumed the old coordinator. The
guard now waits for the verified suspended state before proceeding. The
second attempt exported the complete15k checkpoint and stopped only the owned
torchrun process group and coordinator. Other GPU jobs were not touched.
Old logs reached15,700 updates; updates after the selected15k checkpoint were
not saved and are not counted in the resumed lineage. Historical run outputs
and the failed preparation record remain available.

## Execution And Measurement

`training` first runs isolated200-update timing forks from exactly the same15k
checkpoint: single-rank highest precision, then single-rank official high.
Their checkpoints are marked benchmark-only and cannot initialize production.

The four-rank high technical continuation also executes200 real updates and
is the selected production predecessor at15,200. Production then continues
to50,000 total updates, not50,000 additional updates. All ranks must prove
the same restored counters, finite gradients, zero skipped batches, actual
batch64, frozen Tokenizer identity and final backbone tensor digest.

Timing excludes the first25 updates and includes the remaining native training
work plus Lightning end-of-training checkpoint/shutdown overhead. It is an
end-to-end short-run timing, not isolated GPU-kernel or steady-state timing.
Native varying token lengths are retained and their histograms recorded;
comparison reports whether workloads have matching length histograms. No
claim that isolated timing forks measure convergence speed or motion quality.

Shared source/config/data/tokenizer hashes stay strict. Only the DDP adapter
and batch transition have an explicit audit exception. Selected GPUs are
rechecked for other compute jobs before each launch. Ordinary inference
review follows the final50k checkpoint; naturalness, slip and UE readiness
remain separate acceptance criteria.

## Verified Results

All three timing runs finished200 optimizer updates from the same15k full
checkpoint. Measurements exclude25 warmup updates and score175 updates.
Native token-length histograms match across all three runs and all four ranks.

| Execution | Global input batch | Updates/s | Input samples/s |
| --- | ---: | ---: | ---: |
| Single rank, highest |256|5.002|1280.4|
| Single rank, official high |256|5.908|1512.4|
| Four ranks, official high |256|9.023|2309.8|

Official high improves this single-rank timing by1.181x. Four ranks improve
over single-rank high by1.527x. Combined improvement over single-rank highest
is1.804x, not5x. This is a short-run wall-time measurement, not a convergence
or motion-quality claim.

Production subsequently observed approximately10.6-10.7 updates/s in100-update
log intervals, which exclude end-of-run checkpoint overhead. These intervals
are not substituted for the controlled three-run comparison above.

The official objective remains the mean of each rank's focused-token CE mean.
Equal input batch sizes do not make this identical to a pooled single-rank
token mean when focused-token counts differ; no token-count reweighting was
introduced. Official high matrix precision can also change frozen Tokenizer
assignments near a quantization boundary: frozen weights are not a promise of
bitwise identical outputs across numerical policies.

The selected four-rank technical checkpoint is15,200. SHA256:
`b48821fe281860a452c866b4a0e558b88ac0d4f4682bfd2fa24448a56f62cc90`.
All ranks restored optimizer/scheduler/global step15,000, then completed200
updates with batch64 and zero skips. Their final backbone tensor digest is
`58de3dd28cdd669ddca1dc9af76d0e2b1c9a77230ae9db193520730663741299`;
frozen Tokenizer digest remains
`2d9cda1e1b5da399c9cef48d9a8e676adcd48e4f22c860a00705dce9f91d66c4`.
Maximum allocated GPU memory is4688.49MiB per rank. Runtime receipts on all
four ranks verify precision32, effective matmul high, no BF16 and no override.

An independent read-only replay of the strengthened receipt gate verified the
actual15,200 artifact: every rank's observed batch binds to64, world4 times64
equals global256, and the saved checkpoint byte hash matches its receipt.
The active training entrypoint was not overwritten for this additional audit.

Formal continuation is running from15,200 to50,000 total updates on GPUs0-3.
GPU4 is free; GPU5's pre-existing unrelated process is untouched. Launch
manager PID4116691; production torchrun PID4121204. These IDs are observations,
not authorization to signal a future process without rechecking its identity.

Full local suite:1088 passed,11 warnings before the final five receipt-gate
regressions were added. The final efficiency sidecar has95 passing tests,
including batch-only resume exceptions, strict source/config/data bindings,
GPU selection, benchmark isolation and asynchronous-stop regression checks.
Those five additions verify successful4x64 receipts and reject contradictory
observed batches, invalid global arithmetic and contract/plan disagreement.
At final observation, production reached approximately20,000 updates and
continued normally at about10.6-10.8 updates/s.

Local copies: `pose-efficiency-v1/speed_comparison.json` and
`pose-efficiency-v1/ddp-technical-result.json` under the native LAFAN output
root. Live status is in
`training/workflow_state.json`; timing comparison is
`training/speed_comparison.json`.

Residual failure-path risk: coordinator hard-kill cleanup relies on torchrun
to terminate its separately grouped workers. The completed handoff explicitly
confirmed old GPU workers disappeared; no such orphan failure occurred. Launch
guards reject GPUs with any existing compute process instead of killing an
unrelated or unverified process. This limitation does not affect the current
successful continuation, but should be tested before unattended failure retries.
