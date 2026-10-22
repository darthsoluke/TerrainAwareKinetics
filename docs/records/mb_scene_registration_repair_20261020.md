# Scene Registration And Foot Orientation Repair

## Current Work

2026-10-20 23:44 +08:00: final retained code passes373targeted tests. v52 restores
all15PFNN maps on28948, server27336/planner18108. Normal Xbox/WASD page is Ready,
Urban selected; simulated input is disabled. Its Urban83s trace failed at33.859s
with physical torso contact, at30.016fps and0display drops. Therefore stable
terrain control remains failed. No additional training, filters or forces.

2026-10-20 23:07 +08:00: v49 is preparing on port28948, server13988.
It restores the learned Root path and repairs terrain/reference registration
and native foot banking. The fixed64m flat arena and the frozen MB/ProtoMotions
weights are unchanged. Physical evaluation is pending; no training is added.

## Isolated Integration Defects

The old scene adapter queried terrain around the actual Root but then added
the actual/reference displacement to local patch coordinates and recomputed
volume features around the displaced reference. Saved offsets reach6-9m.
Local terrain therefore appeared metres away to the model.

A CPU experiment translated only the reference chart by[8,0,-6]m while keeping
physical terrain, pose and intent fixed. Old mapping changes tokens and gives
0.7982955085m maximum joint-position difference. Correct registration retains
identical tokens, identical patch/volume inputs, and1.63424e-6m maximum joint
difference. Physical query error is4.77e-7m. Evidence:
`out/mb_sonic_e0_e3_20261018/scene_registration_probe_v2.json`.

The corrected contract preserves local neural features, expresses query points
in the immutable native reference chart, and translates the whole generated
source into the physical scene before G1 retargeting. Geometry audits explicitly
map chart points back to the physical scene. Native history remains in its own
chart. Continuous streams use schema/policyv3.

The old retarget discarded rotation about the ankle-to-toe axis. A10degree
native foot-bank perturbation produced exactly zero G1 change. The material-axis
mapping now transports the native Foot rotation in its calibrated neutral gauge.
The review and remaining metric-stride issue are recorded in
[the contact contract review](mb_g1_contact_contract_review_20261020.md).

The earlier Root/stopping diagnosis was confounded by these invalid terrain
inputs. v49 removes the manual planar Root governor and evaluates the learned
path with the corrected inputs.371targeted Python tests pass before deployment.

## Runtime Record

v48 was stopped by console interrupt; server21148, planner12884 and logging
child21076 were confirmed absent at23:05:51.728+08:00. Its failed49.983s PFNN000
rollout and browser evidence are retained. v49 began at23:06:29.2748067+08:00,
with the same explicit development GPU power scope. Results will be appended.

## v49 Physical Result

The83s simulated Xbox trace completed; the run then stopped at91.74s after
three rejected continuation results (browser last displayed91.70s). All91,640
recorded controlled substeps pass the independent physical audit: zero falls,
joint/motor limit violations or unknown ground; peak scene penetration7.468mm.
Display30.016fps, zero dropped samples. ChainP9989.709ms and first visualP95187ms.

Stop-phase mean speed is0.04647m/s, final neutral0.00713m/s and post-trace neutral
0.00965m/s. Right/reverse strafe progress remains0.13985/0.11189m/s against
0.72270m/s intent; backward0.17534m/s. Full records remain in
`registered_scene_foot_v49/browser_receipt.json`, `gamepad_review.json`,
`physical_audit.json` and the episode1 numeric log.

The three rejected results occur at91.48/91.58/91.70s. GPU graph execution rises
from its prior23-25ms to55.99-63.57ms, and provider work reaches97.14-104.64ms.
The100ms state/wall-age bounds reject them. The prior520ms continuation reserve
then runs out. The unexplained GPU timing jump remains recorded.

v50 prepares one native24/30s segment ahead for held commands. Its total reserve
is1.2s for the current tracker. New commands still replace the uncommitted
future at the original handoff; age bounds and control/render rates are unchanged.
Regression explicitly verifies prefix preservation and changed-command handling
with more than1.2s already buffered. No extra frame or tail hold is introduced.

v50 also enables the [metric source-foot candidate](mb_metric_foot_candidate_20261020.md)
for a separate physical comparison and uses equivalent batched quaternion
conversion.250scoped integration tests pass. v49 processes were confirmed absent
by23:21:04.938; v50 server32508 began23:23:40.8537661+08:00. Physical outcome pending.

v50 completed the full83s trace and136.53s total with no coverage failure, at
30fps.136,432controlled substeps pass mechanical checks. However, fullXYZ foot
anchoring progressively crouches the robot (minimum Root0.15464m) and worsens
several direction metrics. That candidate is rejected. Its source and evidence
remain in `metric_scene_foot_v50`. Current v51 tests horizontal metric anchoring
with original embodied height targets; it passed78scoped CPU checks and began
at23:30:41.9584008+08:00, server5548. v50 processes are confirmed stopped.

v51 horizontal-only candidate fails with pelvis/torso scene contact at9.346s.
All9,246recorded controlled substeps were audited; failure is retained in
`metric_planar_foot_v51`. Both foot-placement candidates are rejected and removed
from the production solver. Native Foot banking, chart registration, equivalent
quaternion batching and proactive genuine-reference renewal remain. The failed
experiments demonstrate that source-foot displacement matching alone does not
produce a compatible G1 gait. No additional training was run.

## Retained v52 Runtime

`registered_pfnn_v52` binds the final production source and restores all15maps.
Urban episode1 records33,759controlled substeps. Original joint/motor limits,
finite values, known ground, zero auxiliary force and continuous clocks all
pass. The body contacts the world at33.859s; peak terrain penetration is15.167mm.
The actual Root is[2.61005,-1.12068,0.31843]m with ground height0.15362m at failure.
The retained screenshot shows the low obstacle near the staircase. This failure
is physical tracking, with no reference exhaustion or interval exception.

Display30.016fps/0drops; chainP9991.277ms, first visualP95207.4ms, body onset
serverP951239.87ms. Input responsiveness and terrain stability still need work.
UnrealEditor PID26648 began23:33:46.769+08:00 and shared this GPU during v52.
Before v52 launch the GPU had6168MiB occupied, with no Python compute process.
This does not explain v49's earlier timing spike; its cause remains unmeasured.

Final sources match package hashes,373tests pass, and no candidate foot-position
option remains in the production solver or runtime. Existing source-foot stride
loss and missing direct target Root translation/velocity in the frozen actor
remain the next integration issues. The two rejected foot-fitting experiments
show that minimizing marker error alone is insufficient to resolve them.

The normal page was manually reset and reloaded, with15available PFNN entries,
Urban selected, no error and no simulated input. Screenshot:
`registered_pfnn_v52/normal_page_ready.png`. The service stays available for
inspection; quality/stable-gamepad approval remains false.
