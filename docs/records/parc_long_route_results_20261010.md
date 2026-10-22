# Continuous PARC Data: Measured Result

The user has now stopped further generation and approved training with the
existing corpus. Continuation is running; see
`long_route_mixed_training_20261010.md` for data identities, live startup
evidence and automatic evaluation. Do not restart a collection campaign.

## Resource Rule

The latest user instruction reserves physical A100 GPU0 and GPU7 for other users.
Only GPU1..6 may be allocated to this project. Do not resume the previous
all-idle-card policy. The short-course coordinator and its obsolete waiting
training receiver have been stopped; source files and completed outputs remain.

## One-Hour Benchmark

The first simultaneous cohort used GPU2..6, seeds 20261012..20261016 and 32
candidates per scene. Each scene has one continuous 111.2 m route through three
independently varied mixed-terrain sections. Motion history is carried through
section boundaries without resets or replayed/concatenated motion clips.

| Measurement | Verified result |
| --- | ---: |
| New complete physical recordings passing SOMA screens | 62 |
| New training motion | 3631.8 s = 60 min 31.8 s |
| Clip duration minimum / mean / maximum | 50.9 / 58.5774 / 65.3667 s |
| Generation start to completed motion screening | 1413.28 s = 23 min 33.28 s |
| Generation start to native MB dataset ready | 1745.40 s = 29 min 5.40 s |
| Existing native records preserved | 538 |
| Native records after this cohort | 600 |

The timing is the corrected production run, not the earlier investigation,
implementation or pilot experiments. It includes generator startup, generation,
physical recording, SOMA screening and native conversion. Prior official PARC,
BONES, earlier long-route pilots, the late GPU1 batch, raw candidates and partial
physical recordings are excluded from the benchmark's new-motion duration.
Duration is `(frames - 1) / 30`, never overlapping training windows or replay time.

Native dataset on A100:
`${PRIVATE_STORAGE}/motionbricks-20261010/parc_long_route_scale_v1/parc_native`

Manifest SHA256:
`a4480219520afb8714a4dd86d75c92496f7c075236b59e1c9516e33facd7624a`

The complete artifact hashes, native/source contracts, new train split and
duration were independently checked with `verify_dataset` and the actual
`accepted.json` records. The small receipt is mirrored locally at
`motionbricks/out/parc_long_route_20261010/dataset_receipt.json`.

`launch.json` subsequently includes a late GPU1 batch (seed 20261017, eight
candidates). The original five-job benchmark remains explicit via
`--cohort-size 5`; its start time and first successful screening timestamp were
preserved. Additional data is merged into a separate dataset version.

The consolidation has now completed and passed `verify_dataset`: 85 newly
generated long training clips totaling 5031.8667 s (1 h 23 min 51.8667 s), plus
three separate long validation clips totaling 176.0333 s. Including the 538 old
native records, the consolidated dataset has 626 records. This larger total is
not substituted into the original five-card/30-minute benchmark.

Consolidated dataset: `parc_long_route_complete_v1/parc_native` under the A100
task base. Manifest SHA256:
`f0ed1d707eb4be576afe063e779b1b13865810c92f58ff068d6691f1449c708e`.

## What Changed

The old 3-5 second outputs principally resulted from short start-to-end routes,
not a model length limitation or a fixed clip splitter. The prior pipeline then
spent 3000 optimization updates on each small retained clip. This was a poor
match for collecting long continuous traversals quickly.

The new course builder constructs a large terrain and one connected route.
Each constituent section varies stairs, ledges, detours, gaps and rough patches.
It uses the existing PARC terrain construction and route conventions; the
generator remains the verified official iteration-3 EMA checkpoint.

Sampling uses the published training-evaluation DDIM stride of 50, from
`PARC/data/configs/parc_1_train_gen_default.yaml`, instead of the data-generation
default of 5. This is an explicit sampling-profile change, not a claim that
every default in the original generation YAML remained identical.

The new experimental collection path tracks generated references directly with
the official controller and simulator. It does not run per-clip offline
kinematic optimization; upstream `scripts/parc_2_kin_gen.py` also makes that
stage optional through `use_opt`. Complete recording from the first frame and
the existing SOMA geometric/continuity screens remain required. Suffix-only
recordings are excluded. Raw heuristic scores and route diagnostics are retained.

This is offline autoregressive reference generation followed by real physical
tracking. It is not an implemented replacement for the official interactive
demo's generator/simulator feedback loop. No newly trained MB controller or
local live model deployment is claimed here.

SOMA validation had another length-dependent cost: the official box-union SDF
was evaluated against the whole large heightfield on CPU. An optional CUDA
backend now runs the same PARC function, with unchanged geometry and bounds.
Real long-clip CPU/A100 comparisons differed by at most 2.98e-8 m. Independent
clips are retargeted in parallel; no motion, feet, roots or terrain are adjusted
to pass the screen.

A packaging defect was fixed: different batches all had a directory named
`recording` and reused names such as `long_raw_c005`. Upstream identities now
include the immutable physics-report SHA256 plus candidate key. The strict
conflicting-identity validator remains enabled; motion files were not changed.

## Evidence and Limits

The first long pilot generated four 54.87-61.4 s candidates in 144.50 s. Official
tracking took 138.78 s and produced three whole recordings plus one suffix.
All three whole recordings subsequently passed SOMA screening, totaling 176.03 s
assigned to validation. A second single-card batch admitted 20 train clips,
1212.37 s. Neither pilot is counted in the one-hour benchmark above.

These are accepted paired geometric teachers under the current screens.
Exact passage through every authored waypoint and demonstration of every named
skill remain unverified. `complete_planned_route_verified` remains false; scene
names must not be presented as proof of climbing or obstacle-specific success.
SOMA retargeting is not a new rigid-body physics simulation of the SOMA mesh.

Validation included 10 long-route/generation tests, 15 retarget/SDF admission
tests including CUDA parity, and 18 catalog/native-dataset tests, plus actual
A100 generation, simulation, retargeting and native dataset verification.

## Artifacts

All bulk data remains on A100 under
`${PRIVATE_STORAGE}/motionbricks-20261010`:

- `parc_long_route_probe_v2`: first continuous-reference pilot.
- `parc_long_route_physics_probe_v1`: its complete/suffix physical recordings.
- `parc_long_route_soma_probe_v2`: accepted diagnostic validation clips.
- `parc_long_route_batch32_finished_v1`: the preceding 20-clip train pilot.
- `parc_long_route_scale_v1`: the measured cohort, per-stage logs and native data.
- `parc_long_route_complete_v1`: incremental consolidation of additional qualified
  long clips; separate from the benchmark dataset and timing receipt.

Do not restart `parc_duration_campaign_v1` or `diverse_mixed_a100_v4`: they belong
to the superseded short-course workflow. The next training run must consume the
new long-route dataset with the accepted BONES flat locomotion, retain native MB
objectives and validate generated-feedback motion before local deployment.
