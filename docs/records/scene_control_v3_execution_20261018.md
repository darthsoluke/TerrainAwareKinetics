# Scene Control V3 Execution Status - 2026-10-18

## Current Status

Selected backend: CPU eager, two threads, H1024, preview64 at 30 Hz,
8,665,966 parameters, FP32/highest matmul precision/TF32 off. The actor is an
**UNTRAINED architecture-capacity probe**. Optimizer updates are zero; formal
38.4M exposures, old v2 services and old quality diagnostics remain untouched.

The final query policy is `committed_root_preview_256_128_128_v2`: 256 current
official surface points, 128 along the previous committed predicted Root path,
128 immediate latest-intent probes. Initial Root plans use observed history
velocity only. Native-heading ego/world conversion and plan ages are explicit;
only the matching rendered ACK commits Root/hidden/body points/plan once. No
extra NN or queries, output Root substitution, smoothing or IK were added.

## Measured Evidence

Final receipt: `out/scene_control_v3_20261018/performance/final_integration_v2/receipt.json`.
It records current-policy CPU measurement, the historical CUDA failed budget,
browser evidence, source hashes and the reproduced first interruption separately.

Exactly one selected-profile CPU remeasurement completed: H1024, four scenes,
30 warmup/100 measured steps per scene; preflight 93 passed/3 GPU tests skipped.

| Scene | Necessary P99 ms |
| --- | ---: |
| PFNN011 | 12.218413 |
| PFNN007 | 10.469507 |
| Flat | 10.301757 |
| True3D | 11.491062 |

All pass the 20 ms necessary-compute budget. Full18056 and sparse512 official
skin parity errors are zero. The prior CUDA H256 pilot P99 was 26.773601 ms,
dominated by native rig/history work; it remains `failed_budget`. No GPU full
sweep, compilation or new-policy GPU benchmark followed. The old three-width
CPU envelope is retained as historical, not reused as final-policy timing.

Parent browser run: PID179228, predeclared 300-frame limit, 300 committed frames,
last index299, no runtime error, approximately ten seconds. Retained frames44..299
give 29.994001 Hz and necessary P99 14.465700 ms. Formal command/display evidence
is **18 associated samples**, P95 67.900 ms; 29 submissions include initial/force
and unassociated entries. Render-submit P95 is 0.300 ms. This measures client
render submission, not scanout. It is a bounded capacity segment, not a long-run
or learned-quality acceptance. All18056/eight-influence browser CPU-mirror
reference max-absolute error is 3.024308e-7 m. Parent pixel checks show nonblank,
moving desktop rendering and no mobile horizontal overflow.

## Reproduced Interruption

The earlier PID129724 session closed at 383/900 frames. Its artifact is retained
as failed/incomplete, not browser full-pass evidence. The original saved JSON is
truncated at 200,011 bytes and remains invalid. The parent recovered its actual
seven command switches from the original CUA AX snapshot; that separately saved
transcription, not the later 300-frame commands, drove exactly one CPU replay.

Actual CapacityRuntime/H1024/seed20261018 reproduced the first exception at
frame_index383 with 383 committed frames. Root was [1.130063,4.863434,1.680202] m.
All 512 query points were finite but outside the unchanged known domain
[-4,-1,-4]..[4,4,4]: minimum query Y=4.013572 m exceeded its 4 m upper bound.
All three query-source valid counts were zero, scene_known remained true and
input_valid became false. The publication guard raised "Unknown scene or invalid
actor input: no capacity frame published". Thus untrained upward Root/body drift
caused known-domain exit and rejection at the exact browser frontier. The old
server lost that exception on session removal; D's bounded archive/visible error
fix preserves future diagnostics. No model, bounds, source or budget changed.

Evidence: `out/scene_control_v3_20261018/performance/root_cause_repro_383_v1.json`
and `forensic_383_parent_switches_v1.json`. The replay stopped on that first
exception; no further experiment or performance run was performed.

## Next Milestones

1. Preserve the reproduced known-domain rejection and explicit closed-session
   diagnostics as the incident baseline; do not hide it by expanding bounds or
   substituting Root motion.
2. Bind approved real-data SOMA/geometry/contact lineage and split-safe targets
   for the causal actor and preview head; keep unknown geometry explicit.
3. Obtain a separate training budget, then train the new actor with separately
   recorded exposures/updates. Do not resume or relabel the old formal run.
4. Evaluate learned command response, continuity, contact/slip and full-body
   collision quality with full18056, eight substeps and 0.02 m peak-penetration
   gates unchanged. Retest selected-backend latency and longer bounded sessions
   before claiming usable learned WASD or sustained long-run operation.
