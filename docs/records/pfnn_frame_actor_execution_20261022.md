# PFNN full-data 3D frame actor execution

2026-10-22. User authorized the preceding full-PFNN plan and requested two A100s.
Main design: [accepted plan](pfnn_all_data_3d_plan_20261021.md).

**Latest user change: execute on PRO6000.** Follow [migration work](pfnn_pro6000_execution_20261022.md). Do not launch the pending A100 P1 command below. Data synchronization must use private Google Drive or another route excluding the A100 public IP; public SSH is control/status only.

**13:15 update:** P1 reached its8h wall limit and exited. Saved endpoint and all
zero-update comparisons were reviewed in [P1 review](pfnn_frame_actor_p1_review_20261022.md).
One-step fit improves, but closed-loop control/skin fails. No deployment or P1
resume. P2 preparation is next; no P2 training has started. GPU3 only.

**2026-10-22 04:48: GPU3 only.** GPU0/1/2 are reserved for other programs.
Private-Drive migration and all archive hashes are complete;198 PRO CPU checks,
GPU3 zero-update preflight and2-update Adam smoke passed. P1 has now launched
on PRO6000 physical GPU3, timeout PID2482846, under
`${PRIVATE_STORAGE}/motionbricks-pfnn-frame-20261022/runs/pfnn_frame_gpu3_v1/p1`.
Single-card global256/micro256; original10epoch/602900-window/8h caps remain.
By04:50, worker2482964 completed4 updates/1024 windows/8192 next-frame targets;
preflight/smoke/P1 bindings and initialization match. No failure files. The sole
SSH connection was closed normally; detached P1 continues. No repeat launch or
automatic resume.
P0 totals4 actual updates/1024 windows across two hosts; P1 is recorded separately.
The A100 launch instructions and two-card implementation descriptions below are
historical. Current source is `code_pfnn_frame_gpu3_v1`; follow the migration record.

## Scope and budget

- Train a genuine one-frame SOMA controller with persistent GRU state and explicit current movement/facing intent.
- Reuse our three 8x8x8 volumetric regions and original shared environment encoder; initialize only that encoder from the final parent. Root/path/Pose/contact heads are new.
- Root predicts eight path stations at 0.1--0.8 seconds. Pose receives a fresh corridor observation along those predictions. No future source pose chooses query locations.
- Use all 48,232 PFNN train rows once per coverage epoch, then 12,058 category-balanced extras. Existing train/validation/test families remain fixed.
- P1: at most 10 coverage epochs, 602,900 window exposures, 8 supervised next-frame transitions per window. Two-card global batch256, micro128; real tail counts and summed gradients preserve every sample. AdamW actor1e-4/environment1e-5, exposure cosine, clip1.
- Original phase caps: P0 4, P1 16, P2 24, P3 12, comparisons8 aggregate GPU-hours. P1 additionally has an external 28,800-second wall timeout. No automatic retry/resume/budget extension.
- P0 zero-update forward/backward proof precedes a separately counted two-update Adam smoke. Only a passing local frame-budget receipt and matching source proofs permit P1.
- No automatic export or model replacement. Formal original 38.4M exposure accounting remains separate.

## Full-index audit

Remote `pfnn_frame_actor_v1/coverage/audit.json` passed, with source artifacts hashed.
All 80 original/mirrored files have admitted data; 16,195 paired windows and 64,780 augmented rows; 34 source families.

| Split | Rows | Families | >=2s same-scene spans | >=4/8/16s spans |
| --- | ---: | ---: | ---: | ---: |
| Train | 48,232 | 28 | 672 | 0/0/0 |
| Validation | 14,132 | 4 | 216 | 0/0/0 |
| Test | 2,416 | 2 | 34 | 0/0/0 |

Train categories: idle1544, start1352, stop1340, strafe8736, reverse2476, forward32784.
These counts are augmented windows, not independent recordings or joystick events.
Long source imitation must respect this coverage; longer control objectives need explicit task episodes.
No retarget, scene admission, or cache was rebuilt.

## Implementation and verification

- `pfnn_full_training_data.py`: complete index, scene-chart recovery, real contact clock, full-coverage sampler, source-derived stop labels.
- `pfnn_3d_scene.py`: terrain prisms plus authored 3D solids, batch queries and predicted-path observations.
- `pfnn_frame_actor.py`: one recurrent update, one next-pose prediction, official FK history update.
- `train_pfnn_frame_actor.py`: two-rank weighted gradients, bounded exposure ledger, fixed per-category validation, source-bound proofs.
- `launch_pfnn_frame_actor.py`: fresh all-GPU/PID scan; select exactly two empty A100s; external timeout.

138 local CPU tests passed, including31 runner/partial-batch/causality/budget checks. Remote106 model/data/geometry tests passed initially; CPU export test hit a PyTorch2.5 CUDA capability probe under its no-CUDA fixture. Rerunning that test with CUDA hidden passed (no model or contract change). All31 runner checks also passed remotely.
Composite geometry is exact for exterior distance; intersecting-solid interior distance uses min-component semantics. Independent final full-skin penetration checks retain their own geometry gate.

## P0 results

- v1 zero-update preflight: two A100s, two global256 batches; model/rank hashes identical; every required branch has nonzero finite gradients. Each batch took16.2--16.7s; tensor peak0.55GB/card.
- v2 increased micro16 to128, vectorized the eight factual history re-encodings, and tightened proof/accounting checks. Same two input batches/losses/gradient norms;10.2--10.3s/batch, peak3.36GB/card.
- v3 fixes zero-work timeout finalization. Its separate zero-update proof passed. Shared-server elapsed times vary; no fixed throughput guarantee is inferred from two batches.
- `adam_smoke_v3` passed: exactly2 AdamW updates,512 train-window exposures,4096 supervised next-frame targets;192 validation windows before and after, no validation training exposure or test use. Two-rank weights agree. Initial model digest `b1da12f6d1b860493b6d660a470754f31b6a06f01825de4f538f47691df6cb0b`, final `8c4d1404ce32de2a0580dffcc0b6579a4290e7a583ac8bbb668ba1dcdad1917b`.
- P1 will initialize afresh from the declared original environment encoder, not continue this smoke checkpoint. Its budget excludes the separately recorded512 smoke exposures.

Local RTX3080 graph execution includes both fresh geometry queries, Root/Pose and official FK, with100 warmup and200 timed frames per case:

| Scene | Complete-frame P99 ms |
| --- | ---: |
| Existing64m flat |8.013 |
| PFNN007 rough |9.523 |
| Low ceiling/uneven |10.282 |
| Side wall/step |10.915 |
| Bridge below |7.020 |
| Bridge above |8.543 |

All six compute-budget gates passed. Additional144 causal transitions and24 nonstatic skeletal-history cases passed graph/eager parity; max FK difference4.89e-7m. This timing used new untrained Root/Pose heads and the original environment encoder; trained long behavior and browser timing remain subsequent gates.
Reports: local `out/pfnn_frame_actor_benchmark_20261022/graph_v2/complete.json` and `parity_review_v1/REVIEW.md`.

## Current work, 01:09

**P0 is complete. P1 has not started because all eight A100s are occupied by other users.**
The launcher refused before creating a P1 process or launch receipt. The last scan again found compute PIDs on all eight cards, including GPU4 when its utilization had earlier read0%.
No our-training processes remain. Do not repeat completed preflights or smoke when resources become available.

Final immutable source snapshot: `$RUN/code_pfnn_frame_actor_v3`.
Run root: `$RUN/pfnn_frame_actor_v1`, where `RUN=${PRIVATE_STORAGE}/motionbricks-20261010/pfnn_mixed_20261015`.
Plan SHA256: `0a699f80d45ece2eef00652a24c2a4f35429abd8c8d01a7f52821278ad5bb73d`.
Previous snapshots/proofs are retained. New work totals2 optimizer updates/512window exposures, all from smoke; formal original38.4M accounting is unchanged.

All small reports were downloaded through the same SSH connection to local
`out/pfnn_mixed_20261015/pfnn_frame_actor_20261022_v1/reports_v3/`.
Archive `reports_v3.tar.gz` is1,092,908bytes, SHA256 `79985bc26b2fb9d78b713a045f2665f92d3da887ad416b7dcf92c8f0df823cb8`.
No model/animation/data download occurred. The persistent SSH connection is closed after this handoff; a future follow-up opens only one connection.

The old `pfnn-wasd` automation no longer existed. Current active heartbeat ID is `pfnn`, every30minutes, quiet unless progress/failure/action is meaningful.
The user's training/two-card authorization remains valid. Once two A100s are idle, verify current files still match the v3 proofs and execute this single launch from the v3 source root:

```bash
export TMPDIR=${PRIVATE_STORAGE}/mbtmp
export MBBASE=${PRIVATE_STORAGE}/motionbricks-20261010
export MBRUN=$MBBASE/pfnn_mixed_20261015
export PYTHONPATH=.:out/vendor/kimodo
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
cd "$MBRUN/code_pfnn_frame_actor_v3"
"$MBBASE/env/bin/python" -m scripts.launch_pfnn_frame_actor \
  --mode execute --output "$MBRUN/pfnn_frame_actor_v1/p1" \
  --plan configs/pfnn_frame_actor_v1.json \
  --parent "$MBRUN/full_training_v16/pose/weights.pt" \
  --cache "$MBRUN/cache_pfnn_world_v2" --pfnn "$MBRUN/dataset_world_v2" \
  --stats "$MBBASE/long_route_mixed_10k_ab_v1/mixed_dataset/stats" \
  --audit "$MBRUN/pfnn_frame_actor_v1/coverage/audit.json" \
  --preflight "$MBRUN/pfnn_frame_actor_v1/preflight_v3/complete.json" \
  --smoke "$MBRUN/pfnn_frame_actor_v1/adam_smoke_v3/complete.json" \
  --realtime local_frame_budget.json
```

Before launch check `p1.launch.json`, `p1/state.json` and actual process command lines to avoid duplicates. A resource refusal before a launch receipt is not a failed training run. After a real launch, no automatic resume/retry is authorized. External timeout or partial coverage must retain actual counts and last checkpoint.

Runtime target remains 30Hz, full key-path P99<=20ms, independent left-stick translation/right-stick facing.
Quality still requires long idle/stop/strafe/reverse/turn and terrain-switch cases, original stance, full18,056skin vertices/8substeps, <=2cm penetration and zero unknown/uncertain body samples.
