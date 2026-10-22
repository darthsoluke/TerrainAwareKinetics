# Reactive Kinematic Results

2026-10-21 12:04 correction: user rejects the v3 gait. See
[gait repair and failed candidates](reactive_gait_repair_20261021.md).
Cadence bypass is fixed inv4; all motion output remains native. The v4browser
trial reaches30.0106fps over2310frames but still drifts after stopping, so it
is not a stable controller release. User page restored to28944;28950research
only. Both sharePID11492;25772stopped. Historical measurements below are retained.

2026-10-21 03:56 Asia/Singapore. The zero-training prototype runs on28950 with
4/2/1-frame requested replanning intervals, Xbox and WASD. It uses the same
trained checkpoint as28944. Both ports share one frozen model/process, PID25772,
with Urban selected and15PFNN maps. The experiment is paused at its seed with
2frames selected;28944retains the original control protocol.

The new pipeline works, but high-frequency regeneration accumulates idle drift,
and1-frame compute deadlines are not met consistently. No training, IK, filtering,
synthetic frames, tail holds or external forces were added.

## Implementation

- Native8tokens,4Posepasses and24playable future frames are retained. Joystick
  messages arrive independently of inference; one prediction runs at a time.
- Every rendered frame is acknowledged. Factual history contains those samples.
  Planning uses a separately labelled immutable accepted prefix, selected by
  measured compute/delivery latency, at most6frames. New plans enter at their
  exact first frame; missed candidates are discarded without suffix cropping.
- Request interval and computation lead are independent. The page displays
  actual UPDATEHz and LEADframes alongside the display rate.
- History re-encoding, coarse query, Root, predicted-Root-path query, four Pose
  passes, codec and official SOMA rig share one captured execution graph and one
  CPU transfer. Unused full-skin computation is removed from the publication path.
- Existing exact sphereBVH querying is source-bound and matches the previous
  field at21,504actual query points, including distances, normals and validity.

Core files: `data/reactive_kinematic.py`, `data/reactive_path_execution.py`,
`scripts/soma_reactive_server.py`, `scripts/review_reactive_kinematic.py`, and
the opt-in branch in `web/full_corpus_lab`. The original server now accepts an
optional shared executor, retaining one serialized model worker across ports.

## Actual Comparison

Evidence: `out/pfnn_mixed_20261015/reactive_control_20261021_v3/review_v3.json`.
Same Urban seed, speed1.2m/s, native key events W2s,D1s,S2s,A1s followed by idle.
Trials have different total durations. External UnrealEditor PID35220 shared the
RTX3080; normal GPU power was used.

| Request interval | Played frames | Display fps | Adopted update Hz, approx. | Compute P50 / P95 | Input-to-visible P50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4frames | 742 | 30.031 | 7.5 | 27.51 / 41.94ms | 113.3ms |
| 2frames | 1192 | 29.999 | 15.0 | 25.42 / 44.51ms | 85.3ms |
| 1frame | 2284 | 30.002 | 27.4 | 27.57 / 45.98ms | 96.4ms |

Visible-latency sample counts are6/5/6 consumed inputs, including startup;
P95s are286.6/113.6/140.2ms. Unconsumed coalesced inputs remain in the journal.
Only64.34% of1-frame predictions meet the complete33.33ms request-to-ready budget;
2-frame meets99.67% of66.67ms budgets. Every played frame retains its absolute
index. Existing genuine forecasts supply playback while new predictions compute.

All2,878saved conditioning histories were reconstructed from rendered samples
plus their referenced accepted prefixes. Maximum difference0. The3runs contain
no server failure or forecast exhaustion. Every forecast, acceptance, rejection,
intent and played-frame receipt is retained.

## Remaining Quality Issue

After the final stop input plus1second, net idle drift is0.03116m/s over17.80s
(4frames),0.05017m/s over32.73s (2frames), and0.13131m/s over69.16s (1frame).
The last displacement is[8.163,3.980]m inXZ. Thus faster regeneration alone does
not resolve motion quality.

Decoded history endpoints differ from the conditioned Root endpoint by
millimetres; these deltas are recorded per prediction. Summing them accounts
for part of the idle displacement. Next work must isolate the decoded integration
origin and native stopping behavior before promoting1-frame regeneration.

Same-time old/new Root disagreement P95 is18.12/7.11/7.30mm for4/2/1; maxima
85.12/156.90/33.52mm. These compare alternative forecasts; actual adjacent-frame
Root/joint dynamics are recorded separately. Full18056vertex terrain penetration
was not evaluated in these trials.

## Verification

Seven GPU cases retain identical active tokens. Maximum FK error9.07993e-7m
is below3e-5m. Official18056vertex skin error0;24-frame rig also matches legacy
8-frame geometry chunks exactly. Queries compare identical captured inputs;
independent-trajectory differences are recorded separately. Model tensors and
RNG states remain unchanged.

45backend and37frontend checks pass, including factual/committed history,
latest-intent reception, exact handoffs, deadline lead, Xbox and original transport.
Counts overlap earlier server runs and are not summed.

## 3D Foundation

`data/scene_geometry_field.py` supplies oriented static solid boxes, distance,
normal, validity, face IDs and matching renderer meshes from one manifest.
Low-ceiling/uneven-ground, side-wall/step and bridge-above/below fixtures pass34
CPU checks. This independent component is ready for integration; the live model
continues to use its trained PFNN observations. Exact distance-composition
semantics are documented in `scene_geometry_field_20261021.md`.

## Retained Versions

v1 asynchronous suffix cropping failed with22.80685m Root disagreement after
5054rendered frames. v2 exact-boundary4-frame mode ran4962frames at30.0048fps;
2-frame exhausted atframe56 when its fixed lead could not cover compute
P95112.47ms/max155.32ms. v3 uses measured lead independently of update frequency.
All failed sources, logs and screenshots remain in the respective output folders.
v1/v2processes are stopped. Current accounting is
`reactive_control_20261021_v3/resource_ledger.json`; optimizer updates remain0.
