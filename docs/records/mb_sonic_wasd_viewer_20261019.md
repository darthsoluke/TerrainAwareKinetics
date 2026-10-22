# G1 WASD Viewer Handoff

Source frozen:2026-10-19 Asia/Singapore. CPU protocol status:PASS.
Actual new WASD/browser/physics/30Hz acceptance:NOT_RUN by this worker.

## Independent Launch

Parent owns launch, browser and GPU allocation. Preserve baseline PID55588,
28947 and its existing user page. New allocated endpoint:28948 only.
From `${PROJECT_ROOT}`, parent command shape:

```powershell
$env:PYTHONPATH='.;out/vendor/kimodo'
& D:/.anaconda/envs/motionbricks/python.exe -u -m scripts.mb_sonic_lab_server `
  --config configs/mb_sonic_wasd_v1.json --port 28948 `
  --allocation-id <parent-allocation> --evidence <parent-evidence-path>
```

New URL:`http://127.0.0.1:28948/`. Default remains28947. Each port checks its
own loopback Host/Origin and has one controlling WebSocket. There is no service
discovery, takeover, automatic launch, reconnect or reset.

## Runtime Binding

C owns runtime and the supplied `mb_sonic_wasd_v1.json`. The inspected config
declares interactive300s, native dt.005,50Hz SONIC and explicit process-scoped
Windows1ms clock. All15PFNN IDs are000..007,009..014,urban; no008/rigidboxes.
Each map has its actual `source_scene.npz` path/hash. UI availability follows
the runtime catalog; it never substitutes another map or factual seed.

Configuration exposes original integer `episode_id`. Every sent raw intent has
`session_id`, increasing `command_id==command_seq`, `scene_id`,
`scene_manifest_sha256`, `episode_id`, worldXY velocity and independent yaw rate.
Server and runtime reject stale map/hash/episode scopes. Runtime binds the first
real input session until explicit scene/reset; UI never silently takes ownership.

Start is enabled only after an actual awaiting-start snapshot. A manual click
sends a scoped zero, then `begin_episode`; movement controls enable on its ACK.
Scene selection explicitly stops the previous episode, invalidates old display
and pending input, loads the selected manifest/seed, then waits for manual Start.
Reset is separate and explicit. Failure records remain in receipts; a runtime
snapshot failure parks the pump and leaves the connection available for explicit
Reset/scene selection, without retrying physics or holding a reference tail.

## Controls And Display

Existing same-window map, speed, yaw slider, joystick, WASD and Q/E controls are
retained. Right gamepadX controls yaw independently of translation; velocity is
camera-relative input converted once to raw worldXY, not forced facing.
WASD first actual snapshot after initial map/reset aligns the camera behind
`actual.root_heading_rad`, preserving view distance/elevation. Explicit Focus
does the same using the latest actual heading. Other frames only translate
Follow by actual Root displacement; user Orbit/zoom is not chased by heading.
E1 initialization is unchanged; its explicit Focus retains prior behavior.
Unavailable actual heading fails explicitly rather than inferring mesh axes.
Blur/pagehide/hidden or focus entering an editable control clears held inputs and
sends zero. A held gamepad must return to neutral after blur/map/reset before
rearming. A one-value client mailbox replaces blocked unsent input; input IDs
are created only when transmitted. Received IDs are associated only with scoped
actual consumption/first visual/actual onset records, not socket ACKs.

Actual physics200Hz and controller50Hz remain runtime-owned. The browser keeps
the existing single30Hz unique-state gate with active-only compositor paints;
no render call advances physics. Large packed mesh validation, normals and
bounding-sphere construction execute in a module Worker. Its entry is
`/app.js?mesh-worker`, supported even by the old baseline's static route; it
imports Three by an absolute local URL and does not require a Worker importmap.
Binary buffers transfer once per loaded asset. Old-map browser geometry is
disposed after explicit configuration changes; physics/query resolution is
unchanged. Loading/planning/warmup/terminal states are shown explicitly.

## Verification And Remaining Acceptance

Only CPU/Node checks were executed:

```powershell
& D:/.anaconda/envs/motionbricks/python.exe -m pytest tests/test_mb_sonic_lab_protocol.py -q
Get-Content web/mb_sonic_lab/app.js -Raw | node --input-type=module --check
```

Result:29Python cases PASS in0.71s, including19Node cases once. Node includes
the actual Worker entry and real Three CPU geometry with no DOM/importmap/GPU.
Covered: latest-only input, stale episode rejection, delayed ACK/config race,
independent200/50/30clocks, two-port same-origin isolation, explicit reset after
failure, neutral gamepad rearm, lifecycle labels and command association.
Heading-direction coverage checks that camera-relative W agrees with actual
forward at -pi/2,0,+pi/2,pi and that Follow preserves the camera offset.
No listener, browser, runtime/model, physics rollout, GPU or old service/page
was launched or operated by this worker.

Parent actual acceptance still needs: new28948 interactive startup, all15map
availability and explicit switch/Start flow, direct WASD/stop/strafe/QE/gamepad,
failure/Reset recovery, actual G1-vs-G1 reference,30unique frames/s, native RTF,
command-associated display/body onset and complete necessary-chain timing.
Gauss owns G1 constraint correction; C owns runtime/seeds and Zeno geometry
performance. This source handoff does not mark those physical outcomes PASS.
