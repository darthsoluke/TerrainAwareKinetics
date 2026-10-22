# Long-Horizon Running Continuity

2026-10-17. Dual-horizon execution deployed after parent review. Zero optimizer updates.

The user reports a1-2second run/pause/run pattern at12/16tokens, distinct from render stutter. Parent readback found007, actual3m/s (URL6was stale),4Pose passes/4refresh, and later governorON. Requested/applied governor mismatch in an earlier snapshot was a normal packet transition. The user's live controls are left unchanged.

## Fixed Diagnostic

Independent sessions on the existing28944server, same007factual seed, constant world[0,3]m/s and heading0,8/12/16tokens,4Pose passes,4refresh,governorOFF.24eight-frame packets per arm =6.4seconds of simulated motion. Requests are spaced on wall time to limit contention with the user; rendering FPS is not inferred from this offline protocol collection.

Record every published Root/official joint/rig frame, packet IDs, actual profile and neural-generation count, each4-frame observation/history source, full Root prediction speed profile, and requested/governed intent. Distinguish Root slowdown from low root-relative leg/foot movement while translating. Initial descriptive flags: Root speed<.35m/s or root-relative leg RMS velocity<.15m/s for at least4frames after the first second. They identify inspection windows, not geometry-quality gates.

## Source Evidence

The formal training path in `unified_full_training.py` prepares8tokens,32frame Root teachers and32frame Pose/height conditions. `source_terminal_goal` labels velocity from source interval31->32 and heading31. `train_unified_full_corpus.py`records tokens8/played24. Longer runtime horizons therefore change the actual learned time/duration conditioning beyond the formal training length. This establishes a distribution difference, not yet the cause of the reported stop/run behavior.

No retraining, pose smoothing, interpolation, foot locking, Root replacement or lower replan rate is authorized. A changed long-Root/short-Pose semantic policy requires explicit parent review of time alignment before deployment.

## Existing-Server Reproduction

`gait_protocol_v1/complete.json` records all192frames per arm with fixed3m/s,governorOFF,2real replans per8-frame packet, exact simulation sourceframes0/4/8/...188 and constant requested/governed commands. No model buffer reuse occurred. Mean actual Root speed8/12/16is2.4147/.2302/.08523m/s. The corresponding mean first4Root-prediction speeds are2.4037/.2867/.1301m/s, versus final4prediction speeds2.8397/2.0391/2.2271m/s. The long-horizon Root repeatedly puts acceleration near the far end rather than in the executed prefix. The physical motion, not render delivery, is slow.

The8arm has no post-first-second low-speed/leg-still episode under the predeclared descriptive thresholds.12has Root<.35m/s atframes31-151, with root-relative low leg motion over42-80,94-116and118-124.16has Root<.35m/s over31-192, with low leg movement102-136andseveral later shorter spans. No translating-with-frozen-legs event was detected; Root slowdown is already present before interpreting the foot motion.

`gait_warm_handoff_v1`: the first64frames of8-token acceleration match the saved8baseline Root and joints bitwise. Switching this moving history to12/16for128moreframes yields mean1.4547/1.4504m/s, versus the continuing8reference already saved.16eventually drops below.35m/s for the last19frames(.633s). This supports a moving-handoff regression as well as poor initial acceleration. It does not yet establish a clean periodic1-2soscillation across repeated cycles.

## Startup Metrics From Saved Frames

`gait_summary_v1.json` adds these measurements with0new model calls. Progress is signed worldZ. Sustained attainment requires15consecutive raw30Hzsamples above the threshold, without filtering or altering motion.

| Horizon |1s progress m|2s progress m|First sustained .5m/s start s|First sustained1m/s start s|First sustained2m/s start s|
|---|---:|---:|---:|---:|---:|
|8|.4321|2.3510|.700|.900|2.333|
|12|.1329|.2131|5.267|not reached|not reached|
|16|.0681|.1494|not reached|not reached|not reached|

Instantaneous2m/sfirst occurs at1.367sfor8; the stronger sustained measure starts2.333s. Both instantaneous and sustained values are retained for every threshold. The long profiles' slow-start evidence matches the user's newer report.

## Reviewed Single Candidate

Schema would explicitly separate preview and execution horizons. From the same4factual frames and the same world-velocity/heading command, nativeRoot8 supplies all32frame conditions to nativePose8anddecoder. For12/16preview, a second nativeRootN call supplies genuine farther predicted query positions. The terminal task constraint is independently at31forRoot8and47/63forRootN; the long terminal is never moved into a short sequence.8mode retains its one-Root behavior.

To make long preview causally affect the actor without splicing output roots, use the fixed8longitudinal stations of the existing3DSDFcorridor as two observation sets:4on the short execution path and4at actual long-Root times afterframe31. Query measured scene geometry at those positions; preserve body and actual-velocity regions. The short2Dheight patch follows the shortRoot. The combined physical SDFobservation enters the Pose/decoder terrain encoder. No Root positions are joined or blended into an actor trajectory; the two predicted paths remain separately labelled. Record near-path disagreement and far sample coordinates rather than treating them as one guaranteed future route.

Per replan:8mode1Root+4Posepasses;12/16candidate2Roots+4Posepasses, with one8-token decoder. The design was reviewed before implementation. Full-length12/16remains a distinct policy for comparison. Results and deployment are recorded below.

A cumulative-progress task would be a different command protocol. The current trained sparse interface supplies only a terminal planar velocity/heading and accepts no corresponding progress constraint. Adding or repurposing intermediate position conditions is not an execution-equivalent fix and is not being mixed into this candidate.

## Implemented Dataflow And Proof

Both Roots use the identical original8-horizon coarse observation snapshot. The short Root's global/local tensors, terminal condition at31and decoder height/time features remain one32-frame chain. Actor2Dheight follows that short Root. The3Dcorridor uses exact short frames4/12/20/27 and long frames32/36/40/43or32/41/50/59; body and actual-velocity regions are retained. No generated Root arrays are spliced, stretched or smoothed. Production `dual_horizon_execution.py` is separate from `scripts/dual_horizon_reference.py`, which contains native reference and far-removal tests only.

`dual_horizon_candidate_v1/complete.json`: preserved8mode's64frame Root/official rig/joints bitwise against saved baseline; original full16native/compiled maxFK1.9805e-6m, proving the slow long behavior was not introduced by tracing. Candidate four native/compiled cases have equal active tokens, maxFK1.27595e-6m, official18056skin error0 and own-history error<=4.7684e-7. Prefetch8->12->16->8keeps parent policy/state private untilACK and commits execution8/count32and previewN together. Weight hashes unchanged.

At identical history/goal/short Root, masking only256far corridor samples changes actual actor memory by max.4097-.6578 and final Pose logits by.3127-1.7617; active code changes are3/3/0/2across the four cases. Short Root/local conditions and all32height/time features remain bitwise identical between the ablation arms. Thus long observations are consumed by the actor, not just drawn.

Long-time samples are not guaranteed farther forward in space. Cold12/16have9/192and18/192far stations behind the short endpoint; minima-.217m/-.384m. Warm12/16have1/128and7/128, minima-.083m/-.128m. Maximum near-path disagreement is.393/.437m cold and.371/.442m warm. Actual world positions and signed forward distances are retained in each packet and `dual_horizon_candidate_summary_v1.json`.

## Startup And Continuity Result

Same007/3m/s/governorOFF. Original baselines are reused; only the64frame8prefix needed to reconstruct exact hot-handoff history was regenerated and checked bitwise. Cold candidate runs are192frames, hot handoffs128frames. No post-first-second Root-slow or leg-still episode occurred in either candidate under the predeclared flags.

| Mode |1s progress m|2s progress m|Sustained1m/s start s|Sustained2m/s start s|Mean speed m/s|
|---|---:|---:|---:|---:|---:|
|Original8|.432|2.351|.900|2.333|2.415|
|Original full12|.133|.213|not reached|not reached|.230|
|Original full16|.068|.149|not reached|not reached|.085|
|Dual preview12 / execute8|.455|2.348|.867|2.067|2.350|
|Dual preview16 / execute8|.445|2.358|.867|2.067|2.289|

Hot-handoff candidate means are2.948/3.022m/s, compared with original full12/16means1.455/1.450m/s. Candidate progress over4.267s is12.561/12.880m; original6.167/6.128m. This fixes the reproduced planning-delay/startup regression by changing the declared execution policy, not by claiming full-length Pose has been trained for12/16.

## Uninstrumented Cost

The first candidate's `packet_ms` included reference history re-encoding, assertions and CPUworld snapshots. It is labelled `instrumented_with_reference_checks` in its preserved `timing_scope.json` and is not production timing.

`dual_horizon_latency_v1`: three identical saved factual histories,3warmups then9uninstrumented session.plan+JSON+CUDA-completion samples per policy. Native re-encoding/assertions are outside this timed region. Separate3-sample CUDA-event packets are read only at packet completion.

| Policy |Whole8-frame packet P50/P95 ms|Extra preview Root P50 per packet ms|Coarse query calls per packet|
|---|---:|---:|---:|
|Original8|212.64 /274.09|0|2|
|Dual12|226.38 /239.46|6.59|2|
|Dual16|231.99 /243.84|7.53|2|

Each packet still has two4-frame replans and two four-pass Pose/decoder evaluations. Production core matches the saved candidate first packets bitwise.007->005->007keeps graph identities and binds a fresh policy view to the current scene/field. `proof.json` binds current production source hashes, weights and device; original-full proofs are not substituted for candidate proofs.

## UI And Live Handoff

Fixed the stale draft regression: an unedited panel now follows settings from the same viewer. Horizon radio buttons send a partial request; the main viewer merges it into its current live settings, changing only preview and prescribed4refresh/4Pose. Thus a stale panel cannot reset top-bar speed, Shift, governor or camera values. Three CPU migration cases preserve6m/s/Shift1.6/governor/camera while converting old full8/12/16settings to execute8+previewN. Fixed execution token/pass/refresh/preset controls are disabled in dual mode.

`dual_ui_regression_v1.json`: top-bar3->6then immediate preview12->16retains6,Shift1.6andgovernorON. Live metadata is execute8/32frames, Root24frames, preview40/56frames,2Rootpasses per replan. UI explicitly labels preview versus near execution. `dual_scene_session_v1.json`: one live007->005->007switch rejects the old session before use, and a new16preview session binds007with execute8/count32and correct factual origin.

The short browser movement check used requested3/governorON, preview12,4refresh: about30.002fps,0newempty-buffer underruns/ACK waits. At the end of theWhold the governor target was1.5m/sand played speed1.177m/s; this is not mixed with the governorOFFstartup table. Screenshot `dual_browser_12_v1.png` shows actor and separate Root/preview layers. Child review is paused, with preview8requested to preserve the last user selection; parent owns final actual-user-tab refresh and parameter preservation.

## Deployment Identity

Owned28944PID115796, baseline28943PID60524preserved. URL `http://127.0.0.1:28944/?map=007&speed=3`; actual UI values, not an old URLspeed, are authoritative. Command:

```text
"D:\.anaconda\envs\motionbricks\python.exe" -u -m scripts.soma_pfnn_maps_server --bundle ${PROJECT_ROOT}/out/a100_live_20261010/a100-terrain.pt --catalog ${PROJECT_ROOT}/out/pfnn_mixed_20261015/pfnn_map_catalog_v1/catalog.json --port 28944 --root-path-observation --initial-map 007 --planning-policy dual --dual-proof ${PROJECT_ROOT}/out/pfnn_mixed_20261015/dual_horizon_latency_v1/proof.json --readiness-report ${PROJECT_ROOT}/out/pfnn_mixed_20261015/dual_horizon_live_v1-readiness.json
```

Legacy full-length behavior remains an explicitly selected experiment: use the same entry with `--planning-policy full` and omit `--dual-proof`; it is never an automatic fallback. Final source/evidence/process manifest: `dual_horizon_delivery_v1.json`. Original training budget and full-terrain quality decision are unchanged. No new training or output postprocessing; occasional shared-machine packet-time spikes remain possible.

Parent actualtab8final review is complete: `dual_parent_delivery_v1.json/.png`. ONLINE, applied new schema,execute8/32,preview8,4Pose/4refresh, briefidle30fps/0underruns. Parent restored the user's original paused state and factual spawn, preserving actual3m/s,Shift1.6,governorONandpreview8. The retained URLspeed6is explicitly not treated as the current setting. Both parent artifacts are bound in the final manifest; no further diagnostics were run.
