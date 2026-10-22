# Reactive Prefix Training

## Current Work

13:39 FINAL: first authorized training pilot and long-control evaluation are
complete. Both64-update endpoints fail the control screen; neither is promoted
to28944/28950. All A100 workers and the single persistent SSH session exited;
GPU0/1were verified free. No additional optimizer work is queued.

| Metric | Original parent | Source-history training | Generated-history training |
| --- | ---: | ---: | ---: |
| Short validation Root RMS, m |0.128721|0.108175|0.124022|
| Forward mean speed, target1.2m/s |0.8463|0.8650|1.1946|
| Idle16s endpoint displacement, m |0.2696|0.2959|1.1793|
| Start-stop stop onset below0.1m/s, s |0.933|not reached|2.200|
| Maximum full-skin penetration across3trials, m |0.026477|0.019410|0.035548|
| Complete control+geometry trials passed |0/3|0/3|0/3|

The generated-history arm improves forward/sideways progress but worsens idle,
braking and contact. Source-history training improves the short reconstruction
metric without producing stable independent-direction control. Increasing this
same objective's budget is not the selected next action. The next design focus
is an explicit current-intent condition, removal of the terminal-condition
interpretation for the new actor, and long stationary/braking training objectives
with retained gait/contact priors. That architecture is not yet trained.

All9authored-arena trials completed, totaling4500genuine emitted frames. All
18,056vertices and8swept substeps were assessed; actual-body unknown fraction
and uncertain-inside count are0. Checkpoints remained unchanged during eval.
Full scene routes/source-stance acceptance and new-checkpoint RTX3080 browser
timing were not run because these basic control trials failed.

Small reports downloaded and extracted at
`out/pfnn_mixed_20261015/reactive_prefix_training_20261021_v1/`.
Archive526267bytes/SHAcd232830e7b837588e2e122cef81a8ff94c1bacdf91193e1d45ba9e7c1dbd8d0.
Model weights and raw animation remain on private remote storage. Formal38.4M
exposures stay unchanged; this pilot adds128Adam,1024source-window exposures,
7168supervised prefixes. The failedv1attempts and all preflights add0Adam.

13:30: long-controlv1 retained early termination on source-plane sensor bounds.
v2uses an explicitly authored321x321/0.2m/64m plane at the same seed ground height,
same seed and fixed commands for all3models. Sensor validity masks follow live
inference; full-skin unknown/uncertain counts must still be0. This separates
long-control performance from a small source capture's finite map. No training
sample, model or full-skin threshold changed. New evaltimeout1844929/GPU0,
18CPU tests pass locally/remotely; evaluation code SHAa6f6de743f489c4d883c623c178b653cf0fbf2d8aa670b637f0b7e028fc5bb60.

13:22: both v2arms completed64Adam/512windows/3584prefixes. Frozen hashes
unchanged; identical sample-sequence SHAaf48fd533d63f916129648e682c6f8f84688ff7da7367518d28ac7239a4115af.
Validation baseline RootRMS0.1287208m; source-history0.1081747m;
generated-history0.1240222m. Mean losses2.1756458 ->1.9978516/2.1272213.
Source endpoint SHA77e7b7ad680ef63c7eb7bbbb71794c3448eb3d1fcca587823f4c7fedea05a359;
generated endpoint SHAa3e9dcdcf3532a3b2cc0bd2384a068ca97a3d0b06dd10b221b5a9d39d2033c06.
No extra optimizer budget launched. Long control evaluation starts onGPU0,
timeoutPID1834898, separate `code_reactive_prefix_evaluation_v1` source and
`reactive_prefix_control_evaluation_20261021_v1` output.16CPU evaluation tests
pass locally/remotely. Original weights remain on the user page.

13:05: v2actual training confirmed. GPU0source-history worker1816312 and
GPU1generated-history worker1816314 each completed3/64Adam updates,
24window exposures/168prefixes. Sample-sequence digests match. First Adam's
frozen-scope and finite-model checks passed. Each arm retains outer3600s cap.

13:00: v1both arms stopped in baseline evaluation with0Adam updates because
a rotated minimum-range sensor norm fell marginally below0.8m. The sensor now
restores its declared trained range after the FP32 norm; emitted motion is
unchanged. New source `code_reactive_prefix_pilot_v2` and output
`reactive_prefix_pilot_20261021_v2` retain the same64update budget. v1failures
are preserved. New zero-update preflight timeoutPID1813395, GPU0.

2026-10-21: user explicitly authorized training-side changes and A100 training
to repair the unstable pure kinematic controller. New experiment only; the
completed38.4M formal exposure ledger and original viewer weights are unchanged.
Remote A100 is reachable through one persistent SSH connection. Initial scan
foundGPU0/1 idle; GPUs2..7 have other users' processes. Remote data5 has4.1TB free.

Implementation: `configs/reactive_prefix_pilot_20261021.json`,
`scripts/train_reactive_prefix_pilot.py`, and the two new data helpers
`reactive_prefix_training.py` / `reactive_training_data.py`.
79CPU tests pass locally and on A100. Source dependency synchronization completed.
Zero-update preflight timeoutPID1802956 started onGPU0, bounded900seconds.
The two training arms have not started yet.

Remote run is `pfnn_mixed_20261015/reactive_prefix_pilot_20261021_v1`, isolated
source snapshot `code_reactive_prefix_pilot_v1`. All remote Python uses data5
TMPDIR. Only small code archives travel through the existing SSH connection.

## Fixed Pilot

Both arms start from final Pose checkpointb960fbe9 and the same source draws.
Each uses64fresh Adam updates, batch8/micro1,4PFNN+2PARC+2BONES windows per update.
Learning rate1e-5 cosine to1e-6, gradient clip0.5. Per-arm outer timeout3600s;
no automatic restart or budget extension. Exactly512source-window exposures
and3584supervised prefixes per completed arm. Each prefix has4executed frames.
Both arms combined:128Adam,1024windows,7168prefixes,28672executed frame exposures.

The96selected q0 training windows and24validation windows are source-stratified,
group-diverse, and retain the original50/25/25domain quota. Available idle,
start, stop, strafe, reverse and forward categories are counted explicitly.
Missing categories remain visible. No test outcome is used for selection.
Every selected source/scene/contact binding is checked against its admitted cache;
no data, retarget, geometry admission or cache is rebuilt.

Trainable: Root, Pose, codec decoder and their Root/Pose/decoder/height adapters.
Frozen: shared scene encoder, codec encoder, quantizer and EMA. All modules use
eval-mode forward behavior; gradients use a hard-forward token estimator.

The source arm supplies genuine source history at each call. The generated arm
supplies its own last4executed frames, officially FK-reencoded and detached.
Seven calls supervise source4..31 without padding unavailable future targets.
The source stays in a fixed world frame. Terrain is re-queried from the actual
input history and predicted Root path at each call, including during training.

Losses: RootXYZ SmoothL1/0.05m (weight1), Root-relative body FK SmoothL1/0.05m
(1), global-rotation matrix MSE(0.1), boundary-inclusive joint velocity
SmoothL1/0.5m/s(0.25). Velocity targets come from source motion, so moving legs
are supervised to move. This pilot adds no full-skin collision objective.

Command labels use the average of the next4source intervals and final heading.
The existing native velocity-v2 input architecture remains, including its local
terminal-velocity condition at31. This run tests prefix supervision and generated
history within that architecture; a separate immediate-intent input is not yet
implemented. The design review documents that later option.

## Evaluation And Publication

Use the fixed64update endpoint. Baseline/final evaluations always use generated
history on the same24validation windows. Report actual updates, source-window
draw digest, prefix counts, frozen hashes, Root error and motion loss. Verify
the first actual Adam update changes only the intended scope and stays finite.
Code, plan, parent, bundle, selected data and parameter names bind the preflight.

These short source continuations precede fixed-command long idle/start/stop,
independent-facing strafe/reverse, actual analog trace, full-skin terrain checks
and RTX3080render/latency checks. Export eligibility and quality approval remain
false. Original28944 and experimental28950 remain sharedPID11492 with their
original trained weights; no new checkpoint is sent to either page automatically.

## Code Transfer Receipts

- Initial code23KB: f853415dac7a74ca31bd18eb42b8cd4757ff1a8f852b1bdaf20af9b9d8674403.
- Runner/contract/static helper: ace06b4505157f43a7547b1d884b08ade4529516ee25f59f6f25177a979ed382.
- Matching live-query dependencies: 2d6a1e62bf0160a53d2bb07be504110bf06eb2aacd87da0eba5239f96ab0689d.

Original design and implementation review: [training review](reactive_training_design_review_20261021.md).
