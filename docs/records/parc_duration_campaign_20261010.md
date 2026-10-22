# Duration-Driven PARC Collection

This workflow is superseded by `parc_long_route_results_20261010.md`. The new
measured run produced 60 min 31.8 s of screened long training motion and a native
dataset in 29 min 5.4 s. GPU0/7 are now reserved; only GPU1..6 may be allocated.
The coordinator and waiting trainer described below are historical and stopped.

## Target and Counting

The new request is at least one hour of diverse new motion. The campaign targets
two hours of new admitted **training** motion, with one hour recorded as the
minimum milestone. Validation and test motion are additional; existing official
PARC and BONES data do not count towards these hours.

Count `(frames - 1) / source_fps` only after complete official physics recording,
official export and unchanged SOMA geometry/continuity screening. Reject partial
recordings, raw candidates, duplicate source identities and duplicate canonical
SOMA motion. Motion fingerprints remove horizontal translation, quantize root
positions to 1 mm and local rotation matrices to 1e-4. This is duplicate
detection, not proof that two motions are semantically different.

Credit at most 120 seconds from one exact terrain geometry towards the target.
Keep additional unique admitted clips in the corpus, but do not let a single
scene satisfy the duration goal. Train needs at least 300 credited seconds and
six distinct geometries per family. Each holdout split needs at least 30 seconds
and two geometries per family. The final training receipt reconciles duration
and splits against the actual native dataset records.

## Distribution

Each wave preflights 24 layouts, four each of stairs, platforms, blocks, gaps,
rough surfaces and continuous mixed courses. The first wave has 24 distinct
heightfields, verified identically in the native Python 3.11 environment and
the Python 3.8 PARC generation runtime.

Variation includes map size, stair riser/width/count, platform and obstacle
density/height/orientation, gap depth and route count, roughness and reverse
routes. Mixed courses include detours, gaps, stairs, ledges and rough sections;
ledge height varies, including high-ledge challenge layouts. Use the official
terrain builders and A*. Family names describe requested geometry. Hand support,
climbing, jumping and other successful skills are not certified by those names.

Scene IDs and splits are deterministic and assigned before scoring. Layout
index modulo 10 gives validation at 0, test at 1, and train otherwise. All
candidates/retries for a layout keep its split. Exact terrain copies cannot
cross splits. New waves prioritize underrepresented families and stop adding
holdout layouts once the corresponding holdout coverage is sufficient.

## Changes to Corpus Selection

Preserve official iteration-3 EMA weights, released CFG/DDIM, 128 candidates per
attempt, released score thresholds and the 3000-step motion optimizer. Generate
at most three attempts per layout. Up to 16 passing candidates now proceed,
instead of two. After official score and route gates, farthest-point selection
on body-relative FK poses, body-relative velocity and root velocity spreads
retained samples. Velocities become displacements over 0.1 and 0.25 seconds,
respectively, so phase-normalizing sample length does not erase cadence/speed
differences. Near duplicates below 2 cm descriptor RMS do not fill the quota.
This is a corpus selection extension,
not a modification to the generator, physics tracker or quality thresholds.

The rollout time budget is explicitly versioned: `ceil(path_m / 0.6 + 8)`, with
a minimum of 15 seconds and a maximum of 45 seconds for single-family layouts
or 90 seconds for mixed courses. This is not an upstream default. Timing-out
candidates do not become accepted motion. Do not splice, loop or time-stretch
clips to increase credited hours.

## Running Pipeline

Host: A100, `${PRIVATE_STORAGE}/motionbricks-20261010`.

- Latest allocation instruction supersedes the two-card limit: use all idle
  A100 cards. The scheduler scans physical indices 0..7 before dispatch.
- Campaign: `parc_duration_campaign_v1`, coordinator PID 3087229 at the latest
  launch. Read `state.json.coordinator_pid` and `elastic_handover_*.json` for
  current ownership and logs; always verify current PIDs before signaling.
- Training receiver: `diverse_mixed_a100_v4`, PID 2822192 at launch. It waits for
  the campaign receipt and rejects receipts missing the duration/coverage gate.
- At scale-out verification, actual active devices are 1, 2, 3 and 4. Devices
  0, 5, 6 and 7 carry other users' workloads. They remain eligible when idle;
  memory occupancy excludes a card even if instantaneous utilization is zero.
- Existing GPU2/3 workers were preserved. New GPU1/4 workers generate distinct
  wave_001 training scenes; the first wave and its immutable splits were not
  repartitioned while running. The MB training receiver still awaits admission.
- Code: `code_feedback_fix_v1`; official generation runtime: `parc_runtime_v1`.
- Local inference/UI deployment remains unchanged.

The old finite coordinator and its waiting finalizer/trainer were superseded.
In-flight GPU children were preserved, allowed to finish, and their outputs
resumed under the original immutable plan. Legacy stairs, platforms and blocks
remain included in the campaign ledger with their original validation splits.
The old remaining 93 unstarted layouts do not keep a competing GPU queue.

CPU admission runs per scene alongside generation. `state.json` records live
accepted seconds, train-credit seconds, per-family/split coverage and deficits.
`catalog.json` binds actual admitted/rejected sources. `wave_*/plan/plan.json`
and scene receipts retain geometry, seeds, selection and failures.

The elastic scheduler reserves every remaining scene in each adopted worker's
queue, so extra cards cannot duplicate those jobs. Other idle cards receive one
explicit scene each and immediately become available for the next scene when
finished. New waves are preflighted before the queue drains rather than waiting
for a wave-wide GPU barrier. Per-card locks and a fresh occupancy check protect
dispatch; a lock/occupancy race returns code 75 and requeues the scene. Other
process errors or missing receipts remain explicit failures. Eight scheduler
regressions cover allocation, busy cards, legacy ownership, duplicate IDs,
completed-job exclusion, PID reuse and retry/error separation; the combined
scheduler and admission-budget suite passed 22 distinct tests.

After reaching the two-hour target with all coverage gates, build the native
PARC dataset incrementally, preserve existing source statistics and splits,
then merge with BONES and run the previously scheduled matched MB experiments.
No live model is automatically replaced. The quality of terrain-aware control
and long-term drift still requires the existing rollout evaluations.

The campaign stops visibly if three entire waves admit no motion, disk free
space drops below 100 GiB, an infrastructure stage fails, or 64 waves cannot
satisfy the target. These conditions do not produce a training-ready receipt.
This prevents spending indefinitely on a broken pipeline or counting failures
as successful collection.

## Verification and First End-to-End Result

45 focused tests pass, covering duration accounting, split leakage, duplicate/capped
credit, immutable layouts, official score parity and incremental conversion.
The first 24 layouts also pass preflight in the actual generation environment.

The first real SOMA pass exposed missing GMR Git metadata on A100. The pinned
upstream Git objects were restored (less than 1 MB, code metadata only); core
revision and file-byte validation remain enabled. Runtime failures now raise
an execution error rather than becoming motion-quality rejections. The failed
attempt is archived, and the same physical recordings were retried.

Two stair recordings then passed SOMA screening: 5.2 seconds total, both
validation. A real incremental native conversion then preserved all 538 prior
records and added these two to validation, with unchanged source statistics.
They are not training hours. The three legacy scenes subsequently admitted six
clips, 20.0333 seconds in total, all validation. The expanded first wave is now
actually generating stairs_40000000 and platforms_40100000 on GPU2/3 (worker
PIDs 2877569/2877570 at launch). One or two hours have not yet been produced.

To replace a live coordinator, use `scripts.scale_parc_campaign` with the
verified coordinator PID. It freezes only the coordinator to snapshot ownership,
preserves GPU workers, then launches an elastic owner with that snapshot. Do not
signal those GPU workers. To resume an exited owner, pass `--elastic`, the same
base/output and `--adopt-workers` referencing verified surviving worker identities.
Omit the old one-time `--retry-scenes` option. Never launch duplicate coordinators
or bypass the campaign/per-card locks and occupancy checks.
