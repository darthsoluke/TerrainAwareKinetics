# Variable-Horizon Root-Path Acceleration

2026-10-17. Three-horizon acceleration deployed and user-tab reviewed. Near30fps; not fully stable30fps. Zero optimizer updates.

## Contract

One parameterized Root graph -> current-map native predicted-path observations -> four-pass Pose/decoder graph. Supported real-time horizons are8/12/16tokens, with32/48/64total frames,24/40/56usable future frames and0.80/1.33/1.87seconds of usable motion. Four factual-history frames,4-frame replanning,8-frame immutable transport and30Hz playback remain unchanged.

Only dimensions/control flow select a cached graph. Scene geometry and query extent/width/gain/base are runtime observations; no map pointers enter the graph. Each horizon must set the native controller's decoder height/time memory length to4*num_tokens, and sample the corresponding Root future interval. Startup builds a bounded three-profile cache, never during playback.

## Memory And State

Tracing retains shared frozen CUDA tensor storage instead of serializing/reloading complete neural constants for every stage/profile. Constant storage identities and allocator peaks will be recorded on RTX3080 10GB. No third model process; baseline28943 is preserved. Owned28944 stays live until CPU preparation is complete.

Supported horizon-only transitions configure a private future backend. Mutable runtime/inference dictionaries are copied before configuration. The candidate binds num_tokens, runtime_controls and inference_config together after predecessor ACK. The existing independent speed-governor candidate state contract remains intact. Unrelated controls retain their existing boundary policy.

## Bounded Checks

Per horizon: original native token/FK/official skin/history checks at fixed factual and generated histories, including changed query controls; four-frame rolling and8->12->16->8serial/prefetch/ACK identity. Check one map rebind without graph recreation. Moving-packet timings exclude capture/warmup and include publication/serialization; separate CUDA-event stages are read once at packet end. Browser checks each horizon and live transitions without repeating route-quality evaluation.

## Initial State

Owned28944PID86712 and baseline28943PID60524 command lines verified. InitialGPU usage8390MiB/10240MiB. The available IAB review tab is still007/3m/s/8tokens/4passes/4refresh/governorOFF/paused; its settings are preserved in `horizon_predeployment_v1.json`. Latest parent/user request specifies final005/3m/s; final deployment follows that requested map, without mistaking the old review tab for the user's current selection.

## First-Case Numerical Investigation

`horizon_acceleration_v1/failure.json` retains the first startup failure. Single-case evidence `horizon_idle_case_v1..v3` identifies005/8tokens/zero velocity/query[-1,.6,1,.4], field `points`: normalized Root difference3.5763e-7 became1.18585e-4m in extended fine-query coordinates. Native Root path length is.010208m and its final segment.000193716m. Supplying the identical native Root to the new query implementation gives exactly0coordinate difference. The short final tangent amplifies small Root roundoff; query coordinates are not frozen.

Disabling freeze's optional numerical optimizations alone (v2) did not change the discrepancy. Disabling the JIT optimizing executor while warming/capturing the shared graphs (v3) makes graph Root bitwise equal to dense eager. The remaining native-nested/dense difference produces8.58307e-6m observation coordinate error and0SDF error, within the unchanged original gates. No path threshold, smoothing, direction substitution or rounding was added. Shared8-token stage constants use759.25MB unique storage, of which758.60MB points to the existing model buffers/parameters; peak allocated977.42MB in that isolated case.

The subsequent full startupv2passed8/12but retained a16/idle `points` failure3.91006e-5m, normalized Root4.76837e-7andFK1.91664e-6m. Root's second transformer has a fixed valid prefix of9+num_tokens for this batch-one protocol. Capture now verifies that padding pattern, runs the original encoder on that unpadded prefix without a padding mask, and restores the native zero tail. `horizon_idle_case_v4` shows16-token Root/path/fine coordinates/SDF all bitwise identical to native NestedTensor. This changes execution packing only, not sensor math.

## GPU Checks Completed

`horizon_acceleration_v3/complete.json` passes all three native/accelerated startup profiles, each at idle/default query and forward/custom query. Each horizon's two8-frame packets contain four active-token sets, all equal; rolling FK maxima8/12/16are1.045e-6/1.952e-6/3.462e-6m. Own-motion history errors are at most4.620e-7against the original1e-5gate; ACK history identity is bitwise. Root prefix17/21/25padding is checked during trace construction, not in replay.

8->12->16->8prefetch changes, combined with governorOFF/ON changes, match same-policy serial motion/history/state bitwise. Parent runtime/inference/governor state remains unchanged before ACK. Candidate commits num_tokens, runtime_controls and inference_config consistently.005->007->005rebinding keeps all six graph identities; no graph is map-bound.

## Server Timing And Memory

RTX308010GB,005,requested3m/sforward,4-frame refresh,4Posepasses,governorOFF. Each horizon has3discarded warmup packets,12uninstrumented moving packets, then3separate CUDA-event packets. Every server sample includes session planning, publication/FK/skin, JSON serialization and final CUDA completion; network/browser waiting is measured separately. CUDA events are recorded per stage and read once after the packet. Stage samples occur later along each actual generated trajectory, so stage medians should not be subtracted from the earlier whole-packet medians.

| Tokens | Whole8-frame packet P50/P95 ms | Coarse queries P50 ms | Root graphs P50 ms | Path queries P50 ms | Pose/decoder graphs P50 ms |
|---|---:|---:|---:|---:|---:|
|8|193.00 /220.96|35.30|6.73|29.27|21.92|
|12|200.77 /209.55|78.12|6.74|62.82|22.51|
|16|206.99 /212.33|84.14|6.79|80.36|22.33|

Each row contains two Root and two four-pass Pose invocations. Geometry/observer event medians are4.77/1.46ms,10.05/1.46ms and8.87/1.46ms. Longer-horizon queries cover different future geometry; the measured neural cost is nearly constant because the native backbone uses its maximum padded token layout. No query points, passes, precision or replanning frequency were reduced.

The six graph stages reference760,551,064bytes of unique CUDA constants, with758,597,944bytes shared directly with model parameters/buffers. Model constants are not copied six times. Steady allocated956,455,936bytes and reserved1,264,582,656bytes across all three timing conditions. Including startup and one additional map binding, process allocator peaks1,098,799,616allocated /1,417,674,752reserved bytes. Whole-device usage includes the preserved baseline and unrelated Windows/browser graphics and is reported separately at deployment.

## Actual Browser Results

The first browserv1timing file is retained but invalid as a three-profile WASD comparison: after radio selection, the iframe retained keyboard focus and12/16received no movement events. `horizon_browser_v1-invalidated.json` records this. Subsequent runs explicitly focus the canvas, and every per-horizon window has8new physical key events (W/A/S/D press and release), verified using browser `changed_at` timestamps.

All valid rows use005/requested3m/s/4refresh/4Pose passes, the same reset seed per arm, startup and warmup excluded, and no screenshot during the timed interval. Persistent UI settings had governorON in browserv2; v4explicitly restores the initial capturedOFFsetting. These conditions are recorded separately rather than merged.

| Browser record / governor | Tokens8 fps / ACK wait ticks | Tokens12 fps / ACK wait ticks | Tokens16 fps / ACK wait ticks |
|---|---:|---:|---:|
|v2 / ON,4s warmup |29.656 /4|29.627 /3|29.835 /1|
|off_v4 / OFF,3s warmup |29.742 /6|30.000 /0|29.999 /0|

All six windows have0empty-buffer underruns. Actual applied profiles report `split-cuda-graphs`; displayed Root arrays contain24/40/56frames, with the same active trajectory renderer. `horizon_browser_switch_v3.json` covers uninterrupted8->12->16->8->12with governorON:29.823fps,0empty-buffer underruns,0deferred prefetches,2ACK wait ticks. This is approximately30fps with measured occasional waits, not strictly jitter-free playback.

`horizon_wait_analysis_v1.json` filters existing request rows to their browser measurement windows. v2maximum server compute is289.84/285.99/280.35ms and request-to-binary304.60/301.40/295.60ms. Switching maximum316.54/331.20ms. These slow requests already use the earliestprefix0with266.67msavailable; recorded queue times are around.03-.14ms. The existing sequential WebSocket handler processes ACK after its awaited prefetch finishes, so over-budget packets also produce ACK wait. No horizon change was deferred or discarded. InOFFv4,8has a303.44mscompute/318.20msarrival peak;12/16max arrivals246.70/257.70msand no ACK waits. General scene-query/host scheduling variation remains; no SDF-engine expansion, lower refresh rate or frame deletion was used.

## Deployment

Owned servicePID166916, baseline28943/PID60524preserved. Current URL: `http://127.0.0.1:28944/?map=005&speed=3`. Review page selected12tokens/4refresh/4Pose/governorOFF and is paused, leaving inference capacity for the real user page. Parent located the actual user IABtab8and refreshed it ONLINE with005/3/12/4; final user-page screenshot/readback belongs to that parent review.

```text
"D:\.anaconda\envs\motionbricks\python.exe" -u -m scripts.soma_pfnn_maps_server --bundle ${PROJECT_ROOT}/out/a100_live_20261010/a100-terrain.pt --catalog ${PROJECT_ROOT}/out/pfnn_mixed_20261015/pfnn_map_catalog_v1/catalog.json --port 28944 --root-path-observation --initial-map 005 --readiness-report ${PROJECT_ROOT}/out/pfnn_mixed_20261015/horizon_live_v1-readiness.json
```

Final device snapshot7862MiB/10240MiBincludes baseline/Windows/browser workloads, not just this process. Full source hashes, process identity and raw evidence hashes: `horizon_delivery_v1.json`. Six focused CPU tests passed; no training, optimizer updates, full-route or full15-map evaluation. Original model quality decision remains unchanged. Non-supported experimental token/pass settings remain visibly marked native/unaccelerated; accelerated8/12/16never silently takes that path.

Parent's actualtab8delivery is saved in `horizon_parent_delivery_v1.json` and `.png`:005/12tokens/40Rootframes/4refresh/CUDA confirmed; frame731showed30.0fps. Later screenshot atframe1184shows `Confirming played frames`,28.6/28.7fps and0empty-buffer underruns. User page remains running with canvas focus; child review remains paused. Therefore this delivers usable accelerated horizon switching near30fps, while strict stable30fps is still unmet because of occasional long packets/ACK waits. These actual user-page artifacts are included in the manifest, not replaced by the child's paused screenshot.
