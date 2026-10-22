# Reset / Controller Session Repair

The reported error was `Controller session changed; an explicit manual Reset is
required` immediately after opening the control page, at simulation time0.

`LabSession.select_scene` correctly reset the runtime and retained command
diagnostics. `disconnect` then treated those retained commands as authority over
the new episode. It stamped a disconnect-stop command with the new episode ID,
binding that episode to the departing socket. The next socket's first command
was therefore rejected.

The disconnect handler now sends a stop only when the latest accepted command
belongs to the current scene, manifest and episode. Historical commands remain
in diagnostics. A current episode still receives its zero-input stop when the
socket disconnects; controller takeover during an episode still requires Reset.

Three new regression cases cover Reset/scene-change followed by disconnect, and
disconnect after new-episode input.210protocol/runtime tests pass. The runtime
motion/controller source hashes equal v43. Packagev44 additionally pins and
copies the server source. No training or model/planning changes.

Evidence is under `out/mb_sonic_e0_e3_20261018/interactive_session_fix_v44/`.
`before_receipt.json` and `before.png` retain the reported UI error. Live browser
verification confirms Start -> Reset -> refresh leaves episode2 with zero
accepted commands until the new socket starts. That socket successfully starts
with its own ID; the controller-session error does not recur.

The v44 second episode later stopped at6.96s with an independent reference
coverage error. Three expired plans were retained in the log, including two
consecutive wall-age rejections. The successful new-session start and subsequent
coverage failure are both retained in `after_receipt.json`.

The continuous stream now schedules steady-intent extensions with three planning
budgets of reserve: two expired attempts plus one fresh attempt, including native
20ms dispatch boundaries. For the current configuration, renewal starts with
520ms coverage remaining instead of400ms. Changed input still requests immediately.
Snapshot/wall-age limits,30fps display,50Hz control, trusted coverage and stop-on-
exhaustion remain unchanged. One regression reproduces the two-expiration timeline;
215protocol/runtime/continuation tests pass. No saved tail or extrapolation is used.

v45 completes the full83s simulated input trace and148.851s total with0runtime
failures/0expired results,29.99365fps and0display drops. Reset -> refresh again
leaves the new episode with zero accepted commands until Start. Its next episode
later encounters a separate numeric-log queue overflow at16.34s: a153.82ms
persistence pause exceeds the four-record(80ms) queue. Complete overflow evidence
is retained; this is not a controller-session rejection.

The final logging queue holds at most16tick records(320ms at50Hz), with matching
disk-reserve admission and the existing record/total-disk limits. No frames are
discarded and no control commands wait for logging. A deterministic blocked-writer
test verifies all nine records survive an eight-tick persistence pause and drain
in order. Queue overflow remains an explicit failure beyond the declared bound.
v46's incomplete admission configuration was rejected before model loading;
its error remains saved. v47 includes the matching queue cap and storage reserve.

Final candidate: `interactive_session_ready_v47`, server17324/planner21296/28948.
Live repeated-session verification is pending. All previous failures are retained.
