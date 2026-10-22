# Anticipatory Terrain Speed Review

2026-10-17. Optional experiment delivered, default OFF. Bounded comparisons and final toggle continuity checks complete. Zero optimizer updates.

## Strategy

Only the planar target velocity sent to Root is changed. The requested heading, model, frame clock, inference passes, geometry and packet contract remain as configured. Controls are a separate `speed_governor` group, never added to the eight-key model runtime profile.

Each replan reads four factual/published-prefix Root frames, the raw user intent, and the previous Root-path snapshot belonging to this backend. It scans the requested direction, actual velocity direction and an aligned previous predicted curve. A changed intent always has its own fresh corridor. Previous curves are projected forward from the current position and ignored when stale, distant or opposite to the new intent. No shared controller/observer path is used as session history.

Fixed budget: three routes,257 longitudinal stations,5 lateral offsets within +/-0.35m,3855 nearest-height samples maximum. Heights retain original map values. Forward extent is at least2m and covers the estimated braking requirement where possible, capped at16m. Unknown samples are treated as a boundary requiring a stop; reaching that boundary is reported separately from successful terrain traversal.

Local signed height changes over about0.18m, slope and lateral height spread determine a bounded risk score. Both upward and downward transitions contribute. Current v2entry speed is `max(minimum_speed, requested_speed * (1-(0.35+0.30*strength)*risk))`. The strongest known-terrain risk reduces the request by35%/50%/65% across the strength slider. Flat zero-risk samples impose no speed cap.

For hazard distance d and entry speed ve:

```text
t_reaction = 0.40 + 0.20*strength + refresh_frames/30 seconds
a_decel = 2.0 m/s^2
margin = 0.25 + 0.20*strength metres
v_plan = max(v_actual, v_requested)
d_needed = v_plan*t_reaction + max(v_plan^2-ve^2,0)/(2*a_decel) + margin
v_cap = sqrt(ve^2 + 2*a_decel*max(d-v_plan*t_reaction-margin,0))
```

The smallest route/station cap controls the target. Initial/stopped starts apply the cap directly, so flat starts retain the request and near-obstacle starts never first accelerate toward the unconstrained request. Thereafter target deceleration is bounded by2m/s^2 per simulated replan interval; recovery defaults to1m/s^2 and only continues a real risk-limited state. User release resets that recovery state. A0.4s simulated hold retains a recently entered rough-region cap while feet clear it. If the16m scan cannot cover the requested braking horizon, a separately reported visibility cap solves the same distance equation for speed; it is active even on apparently flat visible ground.

## State And UI

An immutable governor state contains the last target, finite risk hold, simulation frame counter and session-bound previous path. Packet construction works on a copy. Prefetch forks the state after the current published packet. ACK commits only the corresponding packet state. Serial/prefetch and disabled-output comparisons are required before deployment.

Terrain tab: enable toggle, speed-versus-motion-margin strength, minimum target speed and recovery acceleration. Requested, governed and actual played speeds plus risk/braking state are shown separately. A risk point uses the existing trajectory-layer renderer. The animation clock stays30Hz.

## Predeclared Checks

- CPU flat/up/down/boundary/turn/state-copy cases and governor timing.
- Real final model: disabled matches the prior execution; enabled serial/prefetch state and motion agree, actual ACK state identity holds; no extra Root pass.
- Before model comparisons, static-map scan found the original right ray flat through10m. That geometry-only draft is preserved asCPUv1. FinalCPUv2declaration uses forward[0,1] and diagonal[-sqrt(.5),sqrt(.5)], both from the original factual spawn[0,0]. First8cmchanges are1.70m and2.90m; fixed endpoints are3.0m and3.9m, respectively, beyond those same transitions. Each arm has at most240generated frames. Requested3m/s,8tokens/4Pose passes/4-frame refresh, off/balanced(strength.5)/cautious(strength1), minimum.6m/s and recovery1m/s^2are fixed. Full packets are saved; reported common-progress geometry ends at the first endpoint crossing. Exceeding.75m lateral deviation is a corridor departure. Incomplete/stalled arms retain their partial progress and failures rather than receiving a completed-traversal score.
- Heavy skin metrics are offline only. Browser verifies fixed settings, active cadence, one-click enable/disable and speed readouts. No whole-route campaign or new training.

## First Bounded Result And One Revision

`anticipatory_speed_review_v1` completed all six arms, with unchanged model tensor hashes and passing disabled/serial-prefetch/ACK/native-split checks. The first reciprocal entry formula requested0.75m/s at balanced/full risk and0.6m/s at cautious/full risk. Forward off/balanced took2.285/6.996s to3m, with full-skin peaks0.2577/0.2654m. Diagonal off/balanced took2.485/7.213s to3.9m, peaks0.2958/0.1588m. Cautious reached only0.387m and2.180m within8s and is incomplete. It is not counted as a low-penetration success.

The policy is revised once, before the next model run, to the bounded fractional reduction above. At requested3m/s, balanced/full-risk entry becomes1.5m/s and cautious1.05m/s. Route origins, directions, endpoints, time limits, model settings and all metrics remain the same. The purpose is to preserve actual traversal while exposing the speed/geometry trade-off. All v1results remain. Actual braking demand/remaining margin use actual speed; requested-scan braking demand and scan_required_extent are separate fields.

## Final Bounded Comparison

All six v2 arms completed their predeclared longitudinal endpoints. Same007 map, factual origin, requested3m/s,8tokens,4Pose passes,4-frame replanning,8-frame packets. Skin uses all18056 official vertices and8 swept substeps; soles use196 unique official vertices. Unknown/uncertain counts are zero. Geometry stops at the same longitudinal crossing; lateral deviation is retained.

| Direction / policy | Progress m | Time to endpoint s | Actual mean m/s | Max lateral m | Body peak cm | Sole peak cm | Sole >2cm fraction | Near-surface sole slide m/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Forward / off |3.0300|2.28485|1.33963|.31905|25.7677|25.7677|.1051094|.4672325|
| Forward / balanced |3.03957|4.40258|.695534|.19497|27.1299|26.5821|.1810107|.2597774|
| Forward / cautious |3.00035|5.09960|.596519|.16451|29.3610|29.3610|.1478812|.2502132|
| Diagonal / off |3.94205|2.48504|1.599408|.34754|29.5799|29.5545|.1456355|.4174128|
| Diagonal / balanced |3.91575|4.61692|.861215|.13141|28.9628|28.9628|.1089553|.3340753|
| Diagonal / cautious |3.90232|5.89486|.678142|.33530|13.2939|13.2939|.1198926|.2871698|

Forward becomes slower and penetrates more deeply. Diagonal cautious reduces the sole peak from29.55cm to13.29cm while increasing traversal time2.49s to5.89s. Therefore the controller ships OFF, with an immediate in-window experiment switch. The original2cm quality gate remains failed. No further policy search or training was performed.

Initial braking is about1.69m before the forward transition and2.87m before the diagonal transition. Actual-speed braking demands are about.352m/.452m for balanced/cautious at the initially near-stationary history; requested-speed scan demands are separately3.9375m/4.624375m. The Root model tracks the target imperfectly: actual-target RMSE is1.8623/.8615/.5409m/s forward and1.7588/.83775/.45333m/s diagonal for off/balanced/cautious.

Governor overhead P95 is.84-.96ms per replan, two replans per4-frame-mode packet, with zero extra Root passes. Moving-trajectory packet P95 spans202-266ms; the earlier fixed-state144.54ms latency result is not substituted for those positions.

## Evidence And Corrections

- `anticipatory_speed_review_v2/complete.json` and `proofs.json`: frozen tensor hashes unchanged, disabled motion/history/payload bitwise against the prior backend; enabled serial/prefetch and ACK identities pass. Actual governed inputs retain native/split active tokens and FK gates. Future proof entry points explicitly enable the policy and assert a reduced target, despite production default OFF.
- `anticipatory_speed_soles_v1/invalidated.json`: the first supplement was invalid because provider reconstruction ignored the copied region attributes and returned18056 vertices. Its results remain preserved and are not used above.
- `anticipatory_speed_soles_v2/complete.json`: corrected explicit official-vertex selection and shape assertions; zero new model generations/full-body distance queries. The six v2 denominators are109172,210308,243628,118776,220696,281652, each196 times its swept-time count.
- `anticipatory_speed_cpu_v6/complete.json`:10 CPU checks, including initial near-risk limiting, flat restart, requested-speed scan extent, bounded recovery, unknown boundary and passed-path-end rejection. The previous-path terminal guard prevents reverse extension after the character passes the old path end. Saved v2 telemetry has155 limiting steps from current/requested corridors and one previous-path limiter whose risk point projects2.654m forward; this guard does not alter those recorded targets.
- `anticipatory_speed_browser_before_toggle_fix_v1.json`: stable OFF30.0012fps and stable ON30.0276fps, each0underruns. The earlier ON29.7382fps window includes screenshot readback. Switching OFF produced one boundary underrun; it triggered the scheduling repair below.

## Governor Toggle Handoff

Previously a differing governor config returned `prefetch_deferred`, forcing generation to wait for ACK. Now the committed-history prefetch copy applies the requested governor before preparing its next complete packet. The real backend's config/state, current motion and current packet state remain unchanged. The candidate stores both its starting governor state and its post-packet state. After ACK of the predecessor, candidate commit adopts its config/starting state; after ACK of the candidate itself, its post-packet state advances the simulation clock by exactly8frames.

An already prepared old-config packet remains valid and may play once more. The next prefetch sees the latest requested config. No published packet is modified or discarded to apply a toggle. The original eight-key runtime-controls change policy still defers exactly as before.

Focused CPU checks: `tests/test_governor_prefetch.py`,4 passed, covering0/2/4prefixes, repeated OFF/ON/OFF, private candidate state, serial equality, ACK history and unchanged model-controls deferral. Live protocol/browser results are recorded separately in the final delivery manifest.

`anticipatory_speed_toggle_transport_v1/complete.json`: actual deployed server,5packets and4OFF/ON transitions, fixed007/3m/s/4-frame setting. Each mid-packet candidate is received before ACK. Full official rig payloads are bitwise equal to the corresponding serial sequence; source-frame progression is0/8/16/24/32, config/readout and split-cuda-graphs execution match. Both sequences finish their40-frame ACK chain. No additional model-quality run.

`anticipatory_speed_toggle_browser_v2.json`: after a4s warmup, four actual checkbox transitions during W/release playback,328frames over10.9289s =30.01217fps,0new underruns,0deferred prefetches. All four applied config readbacks match. Screenshot capture occurs after the measurement and pause. Final page007/3m/s/4-frame mode, governorOFF, Terrain tab open with the corrected minimum-target label. Previous one-underrun evidence is preserved in the pre-fix record.

## Deployed Service

Owned28944/PID86712, URL `http://127.0.0.1:28944/?map=007&speed=3`; baseline28943/PID60524 retained. Complete command:

```text
"D:\.anaconda\envs\motionbricks\python.exe" -u -m scripts.soma_pfnn_maps_server --bundle ${PROJECT_ROOT}/out/a100_live_20261010/a100-terrain.pt --catalog ${PROJECT_ROOT}/out/pfnn_mixed_20261015/pfnn_map_catalog_v1/catalog.json --port 28944 --root-path-observation --initial-map 007 --readiness-report ${PROJECT_ROOT}/out/pfnn_mixed_20261015/anticipatory_speed_live_v2-readiness.json
```

Startup5seeds retain equal active tokens, maxFK1.171e-6m, official18056skin and rig error0. GPU is local RTX3080 10GB. Browser check uses this server alone for inference, with the preserved baseline idle. A Windows socket-close callback emitted WinError10054 during client teardown; subsequent protocol and browser checks completed normally. No suppression or network fallback was added.

Final source/evidence hashes and actual process identity are in `out/pfnn_mixed_20261015/anticipatory_speed_delivery_v2.json`. Final screenshot: `anticipatory_speed_default_off_final.png`. The remaining issue is the model's mixed geometric response to slower target motion, documented above; this implementation does not revise weights or the2cm gate.
