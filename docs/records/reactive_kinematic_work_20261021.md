# Reactive Kinematic Implementation

## Current Work

2026-10-21 12:04: [Gait repair](reactive_gait_repair_20261021.md) supersedes
the03:56status. Analog input bypassed4frame cadence. v4fixes scheduling,
but live2310frame/30.0106fps trial still fails stopping/drift. Original28944
restored;28950research only; sharedPID11492. No new training.

Final03:56status and measured results: [Reactive Kinematic Results](reactive_kinematic_results_20261021.md).
The entries below preserve implementation chronology; the linked result defines
the current v3contract and live state.

03:47 Asia/Singapore: v3preparing, PID25772, shared28944legacy/28950reactive.
v1/v2processes were stopped; all evidence is retained. The unchanged post-Root
pipeline fused admission passed7CUDA cases in v1/v2, active tokens identical,
maximumFK9.07993e-7m, official18056vertex skin difference0, weights/RNG retained.
v3additionally uses the existing source-bound exact sphereBVH proof and compares
its actual coarse/path queries against the previous field at identical points.

v1suffix-cropping asynchronous proposals is rejected: episode13rendered5054
frames at29.999fps but reached22.80685m Root disagreement and failed. v2uses
immutable accepted prefixes and exact first-frame handoffs.4-frame mode passes
4962rendered frames/165.4s at30.0048fps; maximum Root handoff disagreement.03695m,
P95.00366m.2-frame mode exhausts its genuine forecast atframe56 after repeated
missed deadlines; recorded computeP95112.47ms, max155.32ms. Failure retained.

v3separates requested update interval4/2/1from measured compute lead0..6frames.
Recent compute maximum plus18ms presentation allowance sets the lead; confirmed
delivery misses raise its minimum. Accepted prefixes remain immutable. New plans
still enter at their exact first frame; missed candidates are discarded. UI shows
actual UPDATEHz and LEADframes alongside30Hz display. Selecting1frame does not
establish30fresh predictions/sec.45backend and37frontend checks pass before
launch. Model weights remain frozen;0optimizer updates.

The paragraphs below describe the initial v1implementation; the exact-boundary
and separately labelled committed-prefix contract above supersedes suffix cropping.

2026-10-21: user authorized implementation after the framewise-control review.
No new training. Existing trained pure SOMA viewer28944/PID11952 remains active
while the prototype is built. Latest observed map is Urban. External UnrealEditor
PID35220 shares the RTX3080; it is not part of this task and is left untouched.

The new protocol keeps a24-frame native forecast but acknowledges EACH rendered
frame. Only those played samples enter the next four-frame history. Joystick
intents travel independently from forecast requests. One inference runs at a
time and captures the latest input/history when dispatched. Target refresh
intervals are4/2/1frames, with actual generation/adoption intervals logged.

New forecasts retain absolute frame indices and replace only future display
samples. An asynchronous result may arrive after its source frame: its age and
same-time disagreement against the old forecast are measured explicitly.
No interpolation, tail hold, pose filter, IK or Root correction is introduced.
All returned motion frames originate in the frozen native model.

## Implementation

- `motionbricks/data/reactive_kinematic.py`: causal playback state and native engine.
- `scripts/soma_reactive_server.py`: independent intent reception, serialized
  model work, forecast/played journal and raw native forecast records.
- `web/full_corpus_lab/app.js`: opt-in reactive client, actual render ACKs,
  absolute-time handoffs,4/2/1selector and preserved Xbox/legacy control.
- `scripts/review_reactive_kinematic.py`: frame rate, compute/input latency and
  Root/joint handoff measurements from saved actual playback records.
- `motionbricks/data/reactive_path_execution.py`: in-progress fusion of the
  unchanged coarse query -> Root -> predicted-Root query ->4Posepasses -> codec
  -> official24-frame rig/history calculations. GPU admission still pending.

The server is designed to keep28944 as the original viewer and expose28950 as
the experiment, sharing one frozen model and one inference worker. It will need
one explicit service handover after CPU preparation, rather than loading another
large model alongside the current viewer and Unreal.

## 3D Geometry Foundation

`motionbricks/data/scene_geometry_field.py` supplies known static3D box scenes,
oriented surfaces, distance/normal/validity/face IDs and matching renderer meshes.
Its34CPU checks pass, including bridge top/bottom and low ceilings. It is an
independent component; the live controller still uses its trained PFNN field.
See `scene_geometry_field_20261021.md` for exact union-distance semantics.

## Verification So Far

43CPU checks pass for playback state, asynchronous latest-intent reception,
server regression and the3D field.28frontend checks pass for the new protocol,
default legacy transport, Xbox mapping and tuning. The30-test server run also
passes. These sets overlap; their counts are not summed.

The next required work is numerical GPU admission, then real browser4/2/1
comparisons with the same frozen checkpoint and input patterns. Thirty displayed
frames per second and thirty fresh model predictions per second are reported
separately. Original28944 and all failed/new evidence are preserved.
