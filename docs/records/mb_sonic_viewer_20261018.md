# MB + SONIC G1 Physics Lab

## Final Snapshot: 2026-10-19

Current STATUS：`../out/mb_sonic_e0_e3_20261018/task_artifacts/STATUS.md` and
E3 evidence index：`../out/mb_sonic_e0_e3_20261018/task_artifacts/artifacts/E3/index.json`:
E1v4 foreground30Hz/full-chainP9915.597689ms PASS; PID55588, tab2 Ready to Start
after actual Reset PASS. E3v3 FAIL at future G1 ankle-limit admission and
full-chainP9988.61218ms; cmd2 consumed, no cmd2 physical onset before termination.
Final CPU suite257passed/7CUDA opt-in skips/44warnings,21.21s. Earlier run
descriptions below retain their historical scope.

## Ownership And Launch

E3 viewer: `scripts/mb_sonic_lab_server.py`, `web/mb_sonic_lab/`,
`tests/test_mb_sonic_lab_protocol.py`, and this document. Runtime implementation
belongs to `motionbricks.mb_sonic.runtime`. Port **28947** is reserved by the
parent. Only the parent launches the real runtime, server, and browser.

## Runtime Interface For C

The server imports `create_lab_runtime(config_path, allocation_id=...)` from
`motionbricks.mb_sonic.runtime`. The returned synchronous, thread-safe object has:

| Method | Contract |
| --- | --- |
| `config()` | JSON-ready configuration described below. No simulation step. |
| `start()` | Arm the independent worker; an explicit episode gate may keep physics waiting. |
| `begin_episode()` | Begin the armed finite episode only after the explicit Start action. No automatic replay. |
| `stop()` | Stop the loops and preserve terminal failure/state. |
| `submit_intent(command)` | Atomically replace the single newest intent. Never queue it. |
| `select_scene(scene_id)` | Explicit user scene selection, never automatic reset. Return current config. |
| `snapshot()` | Return latest immutable JSON-ready state, or `None` before first state. Never step. |
| `asset(asset_id)` | Immutable bytes for the content-addressed packed mesh described below. |

Provider errors propagate and remain visible. No mock, replay, random actor,
alternate physics engine, or fake state is substituted. Runtime failure must not
reset the robot. In intent-enabled mode, disconnect submits an explicit zero-velocity intent; it does not
change qpos/qvel or reset the physics state.
Official-reference read-only mode submits no input on connection or disconnect.

### Configuration: `motionbricks.mb_sonic.lab.v1`

- `schema`: the string above; `optimizer_updates`: 0.
- `physics_dt_s`: 0.005; `controller_hz`: 50; `world_up`: `z`.
- `control_mode`, `intent_enabled`, `start_required`, `start_action`. The actual
  E1 adapter declares `official_reference_read_only`, false, true,
  `begin_episode`. The reference layer is labeled Official reference, not MB.
  WASD, gamepad polling, speed and joystick are disabled; received intents are
  rejected by the server. `command=None` and unknown onset remain unchanged.
- `scene_catalog`: list of `{id, label, available, unavailable_reason}` covering
  all 15 PFNN maps and the true-3D-object scenes. Unavailable entries remain
  visible and disabled. Do not invent a scene list in the viewer.
- `scene`: `{manifest_id, manifest_sha256, geometries}`. These geometries come
  from the same canonical SceneManifest as physics and queries.
- `robot`: `{name, source, geometries}` for official G1 visual geometry, with
  geom IDs matching the live MuJoCo state.
- Each geometry: `{id, kind, rgba, ...}`; `mesh` has
  `mesh:{asset_id, vertex_count, index_count}`. `asset_id` is the SHA256 of its
  binary payload: little-endian Float32 XYZ (`vertex_count*3` values), then
  little-endian Uint32 triangle indices (`index_count` values). No static mesh
  numeric arrays go into JSON config or snapshots. The browser fetches each
  asset once from `/api/mb-sonic/assets/{asset_id}` and caches by hash. Assets
  retain full source resolution unless an explicit renderer-only LOD is
  identified by the provider. Display LOD never changes physics or query data.
  `box` has MuJoCo half-extent `size[3]`; `sphere` has `size[1]`; `capsule` and
  `cylinder` have `size[radius, half_length]`. Static scene geoms also carry
  `position[3]`, `quaternion_xyzw[4]`. Finite planes must be exported as the
  manifest's explicit mesh. Mesh local transforms/scales are already baked in.
- `reference`: `{version, source, coverage_start_s, coverage_end_s, parents}`.
  Primary display uses the released adapted G1 14-body order in both modes.
  `parents` is derived from the compiled G1 hierarchy using the nearest included
  ancestor. MB labels read `MB -> G1 reference`; E1 stays `Official reference`.
  Runtime `reference.joint_positions` and Root XYZ error use the same adapted G1
  reference and timestamp. Raw SOMA NPZ remains available for offline three-layer
  auditing; there is no second raw-SOMA overlay in this T0 page. The primary
  display rejects a raw30-body SOMA hierarchy instead of relabeling it as G1.
- `initial_camera`: `{position[3], target[3]}` in the same world coordinates.

### Runtime Snapshot

`{frame_id, sim_time_s, server_monotonic_ns, scene_manifest_id,
scene_manifest_sha256, status, actual, reference, contacts, timing, errors}`:

- `actual`: `{root_position[3], geom_poses:[{id, position[3],
  quaternion_xyzw[4]}], body_onset_command_id, body_onset_session_id,
  body_onset_sim_time_s, body_onset_server_ns}`.
  `body_onset_*` is null until the runtime measures physical response.
- `reference`: null when unavailable, otherwise `{version, time_s,
  joint_positions:[[x,y,z],...], command_id}`. Time coverage and the original
  version are never extended by the viewer.
- `command`: null before consumption, otherwise the **actually consumed**
  intent including `command_id`, `client_input_ms`, `server_received_ns`,
  `command_seq`, `velocity_xy_mps[2]`, `yaw_rate_radps`, `input_frame=world_z_up`.
  Runtime-derived `derived_heading_rad` is separate; `effective_intent` is null
  until actual consumption. The renderer never derives yaw from velocity.
- `contacts`: real measured rows `{geom1, geom2, position[3], normal_force_n}`.
- `timing`: `{chain_ms, physics_steps, controller_steps, realtime_factor,
  tracking_error_m, heading_error_rad}`. Unmeasured entries are null, not zero.
- `status`: provider status; `errors`: persistent list of actual runtime errors.

## Transport And Clocks

HTTP config: `/api/mb-sonic/config`; diagnostics: `/api/mb-sonic/diagnostics`;
WebSocket: `/api/mb-sonic/ws`. Loopback/same-origin only, one controlling session.

Client messages use the server-issued `session_id`:

- `intent`: increasing `command_id=command_seq`, browser `client_input_ms`,
  world-frame `velocity_xy_mps`, and independent raw `yaw_rate_radps`.
  Device sampling is100Hz with unchanged commands deduplicated; keyboard/pointer
  events send immediately. Q/E, gamepad right X and a spring-return slider turn
  independently of forward/backward/strafe. C integrates the raw yaw rate at
  native50Hz from actual initial world yaw; it must not overwrite raw intent.
- `begin_episode`: explicit Start button after geometry loading and the first
  actual state draw. Server calls `runtime.begin_episode()` under the scene lock;
  this action creates no input ID or command-motion association.
- `scene`: `scene_id`. This is an explicit scene transition, never a fall reset.
- `reset_episode`: explicit Reset icon between runs, selecting the current
  canonical scene as a new episode. Disabled while running; never sent by a
  timer, failure handler or reconnect. Existing receipts/failures remain and a
  separate Start is required. The runtime always arms the newly created episode,
  including after a completed/dead prior worker, without stepping physics.
- `rendered`: `packet_id`, `frame_id`, `command_id`,
  `client_render_started_ms`, `client_render_completed_ms`. A render receipt is
  telemetry only; it never advances or commits simulation.
- `receipt`: request the retained diagnostics.

The server prefetches the native latest state immediately when a render ACK frees
the one transport slot. If there is no new `frame_id`, it polls at the configured
native physics dt; there is no server30Hz gate or queue. Physics/SONIC never wait
for ACKs. The only30Hz presentation deadline belongs to browser `DisplayLedger`;
the outer RAF has no second gate. During an explicitly started episode, RAF also
submits the current scene between unique physics draws to keep the compositor
active. These `paint_only` submissions never create a physics frame, ACK, sample
or command association. Ready/Completed repaint only for camera changes or a
new actual state. Hidden documents do not draw; physics remains independent.

Visible FPS is `active_unique_display_hz`, measured first-to-last successful
render submissions with physics_steps>0 in the current scene revision. Ready
idle is excluded from this one metric. All retained frame rows, status/tick
counts, and the old all-sample `unique_display_hz` remain in the receipt. The
original v1 receipt remains unchanged. Local lucide icons are initialized once
at DOMContentLoaded before connection startup.

Configuration reads, scene transitions and snapshot publication share a lock.
Every display packet/config envelope carries `scene_revision`. An explicit
scene transition invalidates the old transport slot; a late old-scene render
receipt is identified as invalidated, not a runtime failure. New-scene config
is sent before any new-scene display packet. `select_scene` must return only
after `config()` and `snapshot()` both refer to the selected manifest.

First visual latency is the first rendered snapshot whose consumed command ID
matches the input. Actual-body-onset latency is the first displayed snapshot
carrying that command's runtime-measured onset ID; onset simulation time is
retained separately. Both are browser input-to-render-submission measurements,
not display scanout. Unknown onset stays unknown. Targets: necessary-chain P99
<=20 ms; command-associated display P95 <=100 ms; unique display 30 Hz.
The separate body-onset-server P95 uses the runtime's actual onset monotonic
timestamp minus that intent's server-receive timestamp; it remains unknown when
the provider has not measured onset. Session ID must match as well as command ID.

## Verification

Protocol unit fixtures are isolated to tests. No server, browser, SSH, transfer,
or GPU work is started by this viewer worker. Independent-frequency tests must
show that polls, missing ACKs, and render receipts never call a simulation step
or control tick. Browser framing/pixels and real control integration remain
parent-owned acceptance checks.

### Worker Evidence

From the `motionbricks` workspace:

```powershell
& D:/.anaconda/envs/motionbricks/python.exe -m pytest tests/test_mb_sonic_lab_protocol.py -q
node --test web/mb_sonic_lab/protocol.test.mjs
Get-Content web/mb_sonic_lab/app.js -Raw | node --input-type=module --check
```

Observed after the pacing/yaw/diagnostic/Reset/G1-ghost/Follow patch: **25 Python tests passed**, **13 Node
protocol tests passed**, JavaScript syntax check passed. The Python suite invokes
the same13 Node tests once; they are not additional Python cases. Scene-race coverage invokes in-memory
WebSocket handlers with a blocked runtime snapshot and late old-scene ACK;
it opens no listener. These checks establish transport/clock isolation and do
not measure real MuJoCo, SONIC, WebGL pixels, latency, or real-time factor.

The parent's actual first browser run is preserved as **FAIL** in
`out/mb_sonic_e0_e3_20261018/e1_browser_analysis_v1.json`:112active unique frames,
21.4991284Hz, render-periodP9568.7ms/max69ms,0display drops. The3.9063Hz old HUD
included Ready idle. Full250controller-tick chainP99 is9.792443ms; applied
external body/generalized forces are0. This was a transport failure. Actual
the worker's virtual-clock tests are separate from the parent's real browser
measurements below.

Actual v2 is also **FAIL**, retained in `e1_browser_receipt_v2.json`:24active
frames,3.9099687Hz over5882.4ms,0drops, while physics remains about1x. Six normal
roughly33ms intervals were followed by1010-1031ms gaps interspersed with normal
intervals;0.4-0.7ms render submission CPU time does not explain those gaps.
Document/native-host scheduling must be measured in a foreground rerun.

The next receipt's `browser_diagnostics` records actual
`document.visibilityState`, `document.hidden`, `document.hasFocus()`, focus/blur/
visibility/page events, raw RAF timestamps/deltas, frame socket-delivery and
processing timestamps, render CPU and receive-to-render delay, ACK submission/
roundtrip, and same-server-clock publish-to-ACK/ACK-to-next-publication/snapshot
wall time. Counters separate compositor-only paints from unique frames. Raw
events (capacity4096) and lifetime counts are retained; active metrics exclude
Ready idle but do not hide active hidden/unfocused samples. Native host occlusion
is not inferred from document visibility. No timer is used as a second rendering
clock and no browser30Hz result is declared before an actual foreground rerun.

The old IAB tab closed before the initial v3 attempt. The parent subsequently
opened owned tab2, replaced its older-handler service190760 with140812 using
`e1_live_config_v3.json`, and performed one actual foreground measurement.
V3 display is **PASS** in
the receipt：`../out/mb_sonic_e0_e3_20261018/e1_browser_receipt_v3.json`:
157active samples/5200.4ms,29.99769248Hz,sim.005->5.18,0drops. The
desktop screenshot：`../out/mb_sonic_e0_e3_20261018/e1_browser_desktop_v3.png`
is retained. No browser is operated by this worker.

V3 necessary-chain timing is **FAIL**: receipt sampledP9927.4647ms; the complete
250policy-tick physical log gives P9927.249933ms,max27.7333ms. It remains
official-reference read-only,0commands and no command-associated latency result.

The v3 receipt captures `browser_surface`: actual body/canvas computed
styles, viewport/canvas bounds and backing size, enabled/visible G1 mesh counts,
and actual Root camera NDC. These are DOM/render-state observations, not a pixel
visibility or native-host occlusion claim. Desktop/mobile v2 evidence remains
preserved. V3 records visible/focused document and50/50actual G1 meshes visible.
The old process returned `Unknown lab message` for Reset before the explicit
parent restart; no fresh-handler Reset click is recorded in the v3 lifecycle.
The late Reset mobile toolbar uses a two-row grid; post-patch mobile/Reset
verification remains **NOT_RUN**.

Follow is an initially checked checkbox in the wrapping layer bar. Every new
actual snapshot translates both camera position and OrbitControls target by the
actual G1 Root delta; orbit/zoom offsets are preserved. There is no smoothing,
reference-Root substitution or actor/state modification. Configure resets the
previous-Root baseline; disabled Follow keeps the baseline current without
moving the camera. The manual Focus command remains available. Receipts retain
Follow state and camera/target world positions. Actual desktop v3 framing is
**PASS** with Root camera NDC X/Y approximately0; post-patch mobile wrapping
remains **NOT_RUN**. A static HTML/CSS test
checks the initially checked checkbox belongs to the width-constrained wrapping
layer bar; it is not a rendered mobile-layout test.

The pinned G1-mode translation-sensitivity diagnostic proves reference world
Root/body XYZ is not consumed directly; the Root XYZ readout is a diagnostic
error. It does not imply the default controller tracks an absolute Root path.

The actual `SonicLabRuntime` and `create_lab_runtime` adapter are implemented by
C. The healthy viewer service remains E1 official-reference read-only. Separate
actual B1 physical smokes v1 and v2 have status **FAIL**.

Current evidence:

- B1 failure analysis：`../out/mb_sonic_e0_e3_20261018/e3_b1_failure_analysis_v1.json`
- Execution/source lock：`../out/mb_sonic_e0_e3_20261018/e3_b1_physical_smoke_v1/execution.json`
- Actual smoke report：`../out/mb_sonic_e0_e3_20261018/e3_b1_physical_smoke_v1/report.json`
- Raw physical/plan JSONL：`../out/mb_sonic_e0_e3_20261018/e3_b1_fullplan_v1/episode_0001_000.jsonl`
- v2 analyzed report：`../out/mb_sonic_e0_e3_20261018/e3_b1_physical_smoke_v2/analyzed_report.json`
- v2 execution/source lock：`../out/mb_sonic_e0_e3_20261018/e3_b1_physical_smoke_v2/execution.json`
- v2 raw JSONL：`../out/mb_sonic_e0_e3_20261018/e3_b1_fullplan_v2/episode_0001_000.jsonl`

The numerical model admission is PASS, maxFK3.937178917e-6m with tensor/RNG
equality. Actual policy execution is6ticks/24substeps after.18s warmup, ending
at sim.30s on trusted-reference exhaustion. Only initial zero was issued;
WASD/turn program and browser/response validation remain NOT_RUN.
The1.0s packet has100ms usable margin after.9s required future; first publication
to next ready is144.5844ms. The second provider's81.2116ms includes28.1122ms
history snapshot. Full-plan graph34.4838ms is also above20ms.

V2 executed the uniform1.12s reserve and still failed:12ticks/48policy substeps
after.18warmup,sim.42,initial zero only. History snapshots28.6254/31.4003ms and
providers81.4406/86.4201ms led to consumer100ms wall/snapshot-age discards then
reference exhaustion. B's exact CPU history optimization is in progress;
next physical retry, WASD/turn response and E3 browser remain NOT_RUN.
All prior graph failures and raw JSONL/NPZ evidence remain. E1v3 display PASS
and its chain FAIL are independent of E3.

The parent's E1 launch command remains:

```powershell
$env:PYTHONPATH='.;out/vendor/kimodo'
& D:/.anaconda/envs/motionbricks/python.exe scripts/mb_sonic_lab_server.py `
  --config out/mb_sonic_e0_e3_20261018/e1_live_config_v3.json `
  --allocation-id e1-live-cpu-v3 `
  --evidence out/mb_sonic_e0_e3_20261018/e1_live_transport_v3.json
```

The reserved local URL is `http://127.0.0.1:28947/`. No other lab port is used.
