# Standalone SOMA Motion Lab

The browser stage runs Three.js 0.164.1 with local assets. Python runs the existing
native MB Root/Pose/VQ-VAE bundle and the pinned official Kimodo SOMASkin. There
is no Unreal dependency in the viewer. The skin retains all of the shipped LBS
influences, including vertices with more than four influences.

From the motionbricks directory, in the existing motionbricks Conda environment:

```powershell
python -m scripts.soma_lab_server --port 28780
```

Open http://127.0.0.1:28780. Optional web dependencies are available through
`pip install -e ".[soma-lab]"`. The server adds the repository-pinned Kimodo source
to its import path. Three.js and Lucide assets and their licenses are vendored
under web/soma_lab/vendor, so the page does not need a runtime CDN connection.

## Motion Ownership

Live mode uses the 50,000-update native SOMA bundle. Camera-relative WASD intent
and heading enter NativeSomaLiveBackend. Every response publishes eight
immutable 30 Hz frames. The backend's next history is the last four frames of
that committed block. The client must play the block before consuming the next
one; pauses retain the queue, and resets explicitly reset the session.

The server applies official SOMASkin to the model's joint transforms. Each
binary frame carries root XYZ followed by all skinned vertex XYZ values as
little-endian float32. The viewer uses those positions directly. It neither
adds a second root translation nor changes joint poses. Rendering has its own
clock; animation uses the native 30 Hz samples without interpolation, IK, foot
locking, smoothing, or skipped committed frames. A late block stalls animation
instead of inventing or dropping motion.

Terrain playback now supports bounded played-prefix prefetch. A request after
two actual frames combines those observations with the preceding two played
frames; it never promotes the unplayed remainder to history. Cached native
future can be prepared in advance, and a fresh candidate is aligned to the next
block by excluding overlapping uncommitted predictions. Full-block ACK still
gates publication-state promotion and playback of the queued block. The native
timer preserves phase and does not accumulate callback lateness.
See [causal playback review](native_mb_online_causal_playback_20261009.md).

The orbit camera follows horizontal root translation with a fixed target
height. Hip height and individual joint movements do not drive the camera.

The TPS camera now starts behind the commanded heading. Clicking the stage
captures the mouse for free look; Escape releases it and clears movement.
Right-button drag remains available when an embedded browser denies capture.
Touch orbit and wheel/pinch zoom continue using the vendored OrbitControls.
Mouse Y can be inverted with the toolbar toggle. A held WASD direction updates
immediately when camera yaw changes; its right axis is camera-forward cross up,
not the reversed axis. Diagonal input is normalized before applying speed.

The view rig remains independent of model-generated pose/heading. Only the
camera-relative planar intent and requested heading enter the backend; camera
look never directly rotates the character mesh or changes generated frames.
Camera/terrain collision is deliberately disabled at the user's request: no
terrain probes, boom shortening, or camera floor projection are applied.
Lighting uses a warm shadow-casting key, cool fill, rim light and hemisphere
fill, with tighter 4K shadow coverage around the character. This is presentation,
not a motion/contact correction.

`python -m scripts.check_soma_tps --output out/soma_tps_check` checks real browser
mouse capture/release, held-key camera changes, cardinal/diagonal commands,
vertical inversion, zoom, desktop/mobile framing and nonblank rendering.

## Read-Only Motion Overlays

The route icon opens eight independently selectable diagnostic layers:

- Orange dashed: current planar input preview. It is drawn at observed Root Y;
  that Y is not a future height target and is never sent back to the model.
- Gray dashed: the planar endpoint actually submitted for the current generation.
- Blue: the actual Root backbone prediction, unnormalized and translated back
  to world XZ using the native inference convention, with no extra yaw rotation.
- Magenta: the actual final decoded future Root trajectory, including its Y.
- Green: at most five seconds of Root positions already played by the client.
- Cyan and ochre: the exact two 9x9 near-support and intent-direction terrain
  patches supplied to the model. Unknown points and their grid edges are omitted.
- Violet: the official SOMA30 hierarchy using the same FK positions used for skinning.

The prediction lines include the generated horizon, not merely the published
eight-frame block. They shorten on actual playback and advance with the same
buffer slices as motion. Cached blocks retain the generation-time observation
snapshot; its displayed age is explicit. Input changes do not relabel an old
prediction as a newly generated one: the submitted goal remains separate.
Resets and mode/clip switches clear the old overlays. Offline replay only adds
played history; it is never labeled as a neural future prediction.

Diagnostics travel in bounded JSON metadata. Binary float32 mesh geometry and
the playback-acknowledged history contract remain unchanged. The skeleton reuses
the existing inverse/FK result rather than running a different kinematic model.
No diagnostic value enters inference, modifies motion, locks feet or supplies
future terrain heights. The blue/magenta separation is intentionally visible;
it can expose remaining Root/decoder disagreements rather than hide them.

`python -m scripts.check_soma_diagnostics --output out/soma_diagnostics_check`
checks actual model traces, observation masks, layer visibility, paused history,
reset cleanup, desktop/mobile nonblank rendering and browser errors. Automated
transport and buffering regressions also check that diagnostics do not alter
the returned motion or geometry. These checks are not motion-quality approval.

The eye toggle opens a local depth preview captured at 10 Hz. Its sensor follows
the last nonzero movement intent, independently of the body heading and orbit
camera. Only static floor/terrain geometry is visible to that sensor. Three.js
depth packing and perspective-depth conversion produce a linear distance preview
(white near, black at the 14 m far plane). This is an observation debug view,
not an input to the currently loaded flat-ground model.

## Physics Replay

Replay mode shows three fixed PARC physics recordings retargeted offline to
SOMA: terraces, platforms, and three stair flights. Their NPZ hashes must match
the source screening report. The saved source heightfield is transformed into
the same Y-up frame using parc_heightfield_to_native, including its axis order,
cell centers, and spacing. The rendered terrain mesh comes directly from PARC's
convert_heightfield_to_voxelized_trimesh, transformed by the same shared basis;
zero and negative terrain heights remain present. The infinite live floor is
hidden during replay. Character geometry is streamed in bounded chunks through
the same official skinner as live mode.

These are recorded motions, not WASD-generated terrain locomotion. Their
screening report remains preflight_complete_not_training_approved. Live mode
currently uses the flat-ground model; the proposed terrain-conditioned motion
decoder is a separate training task.

## Protocol

- GET /api/config: mesh topology, initial pose, native clock, and replay catalog.
- WS /ws: ready, then sequential plan requests; reset starts a new initial context.
- Each plan response: JSON metadata followed by one float32 binary block.
- GET /api/clips/{id}: replay metadata and paired terrain.
- GET /api/clips/{id}/frames?start=N&count=8: a bounded replay geometry block.

Inference and skinning run on one worker to preserve GPU operation ordering.
The service binds to loopback. Geometry transport, native context handoffs,
official skinning, command validation, and websocket reset sequencing have
automated regressions in tests/test_soma_lab.py.

Run `python -m scripts.check_soma_lab --output out/soma_lab_check` against the
running service for desktop/mobile screenshots, nonblank WebGL/depth checks,
actual WASD motion, reset/pause and replay checks. It uses installed Microsoft
Edge through Playwright. Passing these checks is not motion-quality approval;
the viewer deliberately exposes the model's raw pose, including any remaining
upper-body or contact defects.
