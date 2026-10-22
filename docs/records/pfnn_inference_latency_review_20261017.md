# Inference Latency Review

2026-10-17. Status: implemented, reviewed and live. Zero optimizer updates.

## Scope

The owned 28944 predicted-Root-path viewer uses native Root/Pose/decoder and compiled FK/skin/history. The preserved 28943 viewer uses its existing reusable model graph. The current 4-frame setting generates two native plans for each immutable 8-frame packet. Default model weights, 8 tokens, 4 Pose passes and scene-query math stay fixed.

## Measurement Plan

- Use the same 007 scene, factual initial history, commands and generated-history sequence in one process.
- Keep uninstrumented whole-packet runs separate from CUDA-event stage samples. Record events without per-layer synchronization; read at the end of each packet.
- Include final publication/FK/skin and serialization in server durations. Record send duration separately. Browser timing uses its own performance clock from input version to first displayed frame carrying that version.
- Report sample counts, cold startup separately, warmup exclusion, P50/P95, real replans per second and observation ages. Playback FPS is a separate metric.
- Keep the 8-frame immutable transport and factual played-history accounting. Profile waiting due to prefetch separately from computation.

## Implementation Plan

1. Capture the native Root prefix and Pose/decoder suffix as separate scene-independent graphs. Query the predicted path between them using the current native geometry.
2. Compare active tokens, final-model FK (3e-5 m), official full skin and history against the native path before deployment.
3. Measure the same packet workload before/after, then inspect browser control and steady playback through an isolated session.
4. Publish the optimized owned viewer, preserve the original viewer and update the main handoff documents.

## Initial Inspection

Both servers' complete command lines were checked: 28943 PID60524 and 28944 PID36260. RTX3080 10GB had 8485MiB in use; no third full model will be loaded concurrently. Native execution and duplicate generation at the 4-frame setting are confirmed from source. Existing web INFER excludes final publication and therefore does not measure the complete packet.

## Measured Server Result

Same RTX3080, map007, frozen model, 8 tokens and 4 Pose passes. Each setting has 3 discarded warmups, 20 uninstrumented complete-session samples, then 6 separate CUDA-event samples. Fixed factual and native-generated histories and command sequences are reused. Complete-session samples include final publication, official FK/skin/history and JSON encoding, but exclude socket and browser waiting.

| Refresh | Native P50/P95 | Split P50/P95 |
| --- | --- | --- |
| 4 frames | 281.20 / 294.73 ms | 137.97 / 144.54 ms |
| 8 frames | 141.19 / 154.00 ms | 69.43 / 75.57 ms |

Raw measurements are in `out/pfnn_mixed_20261015/root_path_latency_v3/failure.json` under `report.settings`. The performance run finished; its subsequent cross-candidate history assertion failed and is retained. Earlier independent v1 and v2 measurements also remain in their own folders.

The split graph contains one Root evaluation, native post-Root queries, then four Pose passes and decoding. Each graph includes its own SharedTerrainEncoder. Its steady 4-frame packet uses two such plans. Native encoder timing is now recorded explicitly rather than assigned to Python overhead. Body and actual-velocity samples are reused only within the same prediction, where their coordinates/history are unchanged; all 1536 physical model inputs are retained and reused/direct samples compare exactly.

Native SharedTerrainEncoder intervals for a4-frame packet totalP50/P95=2.829/3.269ms (6separate event samples). The earlier event categories do not imply that all remaining time is Python. Split Root and Pose/decoder stage intervals include their respective encoders, model operations and stream/launch waits.

## Rolling History Review

Startup checks compare native/compiled active tokens, FK <=3e-5m, official 18056-vertex skin, fine height points/world coordinates/validity and native SDFs. Fine height diagnostics now describe the actual Pose input.

The initial 4-frame rolling/prefetch comparison passed all four active-token checks, then a newly introduced absolute comparison between histories of two independently generated trajectories failed at 2.145767e-5 in a motion feature. The failure is preserved in v3. The follow-up checks each committed history against native re-encoding of its own published frames at absolute1e-5, and records cross-candidate feature drift separately. Native/compiled physical FK continues to use the original3e-5m gate. The v4 rolling check passed four active-token comparisons, maximumFK2.83187e-6m; each candidate's actual-frame history errors were3.12924e-7/2.38419e-7. The final v5 check also adds per-candidate serial/prefetch bitwise comparison and exact ACK identity, without repeating the unchanged server performance run.

Finalv5passes: each candidate serial/prefetch bitwise; both ACK commits retain the exact published history; first/second packet histories vs native re-encoding have maximum errors4.47035e-7/3.12924e-7. Cross-candidate maximum is feature336, `local_vel`/`body`, at frame1: native-0.1781380177 versus split-0.1781594753. This is recorded separately from protocol identity and same-motion history gates.

## Queue And Observation Semantics

Static-map geometry does not become stale solely with wall-clock time. Logical observation age is relative to the history/predicted path used for a generated prefix. Wall-clock prefetch waiting is a different quantity. Browser telemetry echoes each command version and its browser timestamp, then records the first actual frame rendered with that version. Request receipt or response arrival is not counted as display. Server queue, complete session, JSON encoding and socket enqueue durations are kept separate from browser timestamps.

The existing immutable8-frame packet is retained. No full-route geometry evaluation or training is part of this review.

## Controlled Browser Comparison

Initial exploratory v1/v2 browser records lacked complete settings snapshots and allowed render-follow updates to inflate command-change counts. They remain as exploratory evidence and are not used for the prefetch conclusion.

The accepted fixed comparison is `root_path_latency_browser_v3-fixed.json` and its summary. Each condition starts from map007's identical factual scene seed, then exactly64 native playback frames. All three measurement-start roots were bitwise identical in browser float32 values: [-0.0312756896,1.0037008524,0.0093974080]. Every plan records actual settings; all match8tokens/4Pose passes/4-frame refresh, original query controls and split CUDA execution. Each script has12 explicit WASD holds of650ms, each followed by500ms release,24 physical key events total. The iframe's stored settings were excluded only in the temporary fixture. That fixture is archived and removed from production JS.

| Prefix | First displayed key events | Input-event-to-render P50/P95 | Binary-to-render P50 | Server session P50/P95 |
| --- | --- | --- | --- | --- |
| 0 | 23/24; final release outside capture | 381.30 /495.13ms |127.10ms |131.15 /156.92ms,53packets |
| 2 |24/24 |381.45 /494.16ms |92.75ms |132.70 /159.74ms,53packets |
| 4 |24/24 |294.70 /400.15ms |30.70ms |129.14 /147.73ms,54packets |

Timestamps are all from the same browser clock. Render means the CPU timestamp immediately after the WebGL render call, not physical photon presentation. Records include superseded/unobserved event versions. The full-window frame-rate arithmetic in the summary includes the brief automatic pause at the64-frame warmup boundary; active final120-frame cadence was30fps. New live validation records the active interval directly. Empty-buffer underruns were0 in all three fixed samples; commit-wait ticks were0/1/0.

The current hardware is shared RTX3080 10GB. Process scans during these cases found only the preserved60524 model server and the owned28944 server, with about9240MiB total graphics memory in use. Windows graphics/browser clients remain present. This is a local deployment measurement, not exclusive-GPU timing.

## Production Scheduling

The viewer chooses a prefix once when each immutable packet starts. It keys recent request-to-binary samples by the current map and complete requested runtime settings. It uses the maximum of the last12 neural-generation responses plus18ms reserve, chooses the latest feasible0/2/4 prefix and keeps that decision for the packet. Fewer than3 compatible samples choose0. Map/settings changes therefore begin conservatively, including1/2-frame modes. Measured response times include server queue, geometry, JSON, socket and browser parse/arrival; no cross-clock subtraction is used.

Available time before the next packet's first playback tick is8ticks for prefix0,7ticks for2,5ticks for4. The reserve covers commit/ACK and timer jitter. A sudden load increase may still delay a frame; the protocol never replaces published frames. `prefetch_decisions` and actual request indices are logged. A manual `prefetch` query remains available for controlled comparison.

## Camera Follow

Held-W checks used explicit4/8/24-frame settings,3.2s holds, no mouse input, identical map and frozen model. Rendering follow created14/3/25 command changes while held, with maximum heading variation4.16e-17rad. The backend's float32 command conversion already removed this noise: actual Root counts over about3.2s were4/12/24, or1.25/3.75/7.48Hz. It did not cause extra model evaluation in this sample.

Movement direction now comes from a camera basis refreshed on genuine look/Orbit gestures and camera reset. Pure render following no longer updates movement intent. Held-W after the edit has constant heading and the same actual Root counts/rates. Input presses/releases are still handled immediately, and genuine look events refresh the movement basis. Raw before/after records and summary are `root_path_camera_held_*`.

This is a browser intent/timing-noise correction, not a speedup source. Actual pointer motion was checked through the existing drag handler using a temporary left-button mapping because the automation API cannot hold the right button during its drag primitive. Eight `user-look` updates changed camera azimuth from-3.04614 to2.86205rad; the followingWcommand changed toVX/VZ[-0.8277585,2.8835422], heading-0.2795462rad. The production right-button filter was restored immediately. Evidence:`root_path_mouse_direction_v1.json`.

## Final Service And Live Check

Owned service PID186648, listener127.0.0.1:28944. Preserved28943PID60524 is unchanged. Actual command:

```text
D:/.anaconda/envs/motionbricks/python.exe -u -m scripts.soma_pfnn_maps_server --bundle ${PROJECT_ROOT}/out/a100_live_20261010/a100-terrain.pt --catalog ${PROJECT_ROOT}/out/pfnn_mixed_20261015/pfnn_map_catalog_v1/catalog.json --port 28944 --root-path-observation --initial-map 007 --readiness-report ${PROJECT_ROOT}/out/pfnn_mixed_20261015/root_path_split_v6-readiness.json
```

URL:`http://127.0.0.1:28944/?map=007&speed=3`. Refresh an existing viewer to load the new JS. The isolated review tab is paused after verification so it does not compete with the user's controls.

Clean production startup releases temporary initialization allocator caches and does not run the latency harness. The preceding diagnostic-servingv5process's first request stalled while total GPU usage reached9791MiB; its raw zero-frame/in-flight state is preserved in`root_path_adaptive_live_v5.json`. It was stopped after checking its full command line. Cleanv6startup and live playback use about5845MiB total graphics memory at the final snapshot.

Finalv6actualW/A/S/D+idle check: measured active interval29.94117fps,0empty-buffer underruns,491frames at the immediate capture. All actual prefetch request indices equal their packet-locked choices; all three choices0/4/2were observed under real response variation. Original settings8tokens/4Pose/4refresh were confirmed. Raw state, every frame interval and scheduling decisions:`root_path_adaptive_live_v6.json`; screenshot:`root_path_adaptive_live_v6.png`. Eight synthetic scheduler boundary/identity cases pass in`root_path_prefetch_boundaries.json`. Python/JS syntax passes. No broad test suite or full-route evaluations were run.

Complete service command, source SHA256 values and evidence references are in`root_path_latency_delivery_v6.json`.

## Remaining Work

High-speed motion can still fail terrain contact because the fixed model's planning/decoding has geometric residuals. This engineering review changes execution and scheduling only. The4-frame profile has an observed7.5Hz internal replan budget; keyboard commands still enter immutable8-frame packets. Other scenes and1/2-frame settings can take longer. No new training, IK, collision projection or output smoothing was introduced.
