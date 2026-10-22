# SOMA continuous-control execution, 2026-10-05

## Status

The planned Root/Pose training, continuous execution, paired history experiment,
fixed validation, complete videos, and local GPU measurement have been executed.
This is a working experimental kinematic backend, NOT a quality-approved UE model.
No UE files, held-out test data, or existing experiments were replaced.

## Implemented Path

Executed four-frame SOMA30 history and requested root goals feed the official MB
Root and Pose backbones, the frozen human tokenizer decoder, and official SOMA FK.
The initial Pose configuration is 512-wide, eight layers, eight attention heads.
At runtime, plan 64 frames, publish four, and replan from the published history.
Published frames are immutable. There are no future source-body inputs, source
resets, foot IK, foot locking, post-decoder root correction, or ground snapping.

Human adapters live in `motionbricks/data/bones_control.py`, `bones_rollout.py`,
and `bones_history.py`; upstream animation algorithms remain the foundation.
The pose packet contains root XYZ, SOMA30 local rotation matrices, and decoded
contact channels. Contact outputs are not certified force or support truth.
Publication timestamps start at zero; the initialization history precedes them.

## Training And Identity

All runs use pilot-v7: 152 training clips and 57 fixed validation windows.
Test was not used. Teacher/generated continuations start from identical Pose
weights, use identical budgets, and preserve the tokenizer parameters and EMA.

| Stage | Optimizer updates | Final remote run |
|---|---:|---|
| Root | 10,000 | `runs/soma-control-20261005/root` |
| Pose baseline | 20,000 | `runs/soma-control-20261005/pose-complete` |
| Teacher history | 10,000 | `runs/soma-history-20261005-v3/teacher` |
| Generated history | 10,000 | `runs/soma-history-20261005-v3/generated` |

Remote run paths above are relative to `${PRIVATE_STORAGE}/motionbricks-preparation`.
The generated arm ramps generated history probability to 0.5 over 5,000 updates.
Both arms retain the same source token targets after a common planar rigid
transport; this does not make generated root errors supervised physical truth.
Full training checkpoints, optimizer state, configuration, contracts, and exact
source snapshots remain on the server. State restoration has an automated test;
an interrupted/resumed full-budget training comparison was not performed.

- Dataset SHA256: `54c5f61c6765d630a6c3e8813848bcef015c5905d123b62139c438db5bf38b8b`
- Frozen tokenizer SHA256: `134ed76c3bc2fcdcbe087c945aac15d42ba4735f7ec785cb1e4f71c8415435f5`
- Root checkpoint SHA256: `fa0105b65ba1271aa1f6441a1c53dd3f8164762aa90c4bf7b3ea083c5f5fa20a`
- Baseline Pose checkpoint SHA256: `cca098dac64caa25b8e80b0ae1b804d3aacca7e1736fefbeab198ab1998160c1`
- Portable bundle SHA256: `995e93ece7539db84bdab50d8fa5decdbbaac1d31b543a041686c4c06ff8fae7`

## Defects Repaired During Execution

1. Future-encoded first tokens cannot be given as history input. All Pose input
   tokens are MASK, and every target token, including the first, is supervised.
2. Tokenizer freezing calls the actual method and keeps codebook EMA in eval.
3. Short-horizon decoding passes the active-token mask; padded tokens cannot
   affect valid poses through the tokenizer's non-causal decoder.
4. Human inference uses the actual decoder root-conditioning representation,
   supports batches, and preserves only already-observed root frames.
5. Stop requests generate braking goals ahead of the observed motion, rather
   than asking a moving body to return to its current position with zero velocity.
   The 3 m/s^2 limit shapes requested keyframes ONLY, not generated root output.
6. Detached evaluation resolves subprocess paths independently of login cwd.

Superseded experiments and source snapshots were preserved. Review v1 used an
unsupervised first context token. Review v2 uses the final weights but the old
goal construction. Review v3 uses the corrected input goals and is current.
The input repair did not demonstrate a global improvement in motion quality.

## Current Evidence

Each arm has the same 57 paired windows and three full 60-second command runs.
The command sequence includes idle, walk, run, stop, 90/180-degree turns,
backward, lateral, and restart. No future recorded body is supplied.

| Arm | Source-root MPJPE, mm | Actual-root MPJPE, mm | Free velocity error, m/s | Near-ground material speed, m/s |
|---|---:|---:|---:|---:|
| Baseline | 103.2 | 139.2 | 0.630 | 0.329 |
| Teacher | 93.3 | 136.7 | 0.610 | 0.330 |
| Generated | 94.5 | 136.8 | 0.668 | 0.346 |

Source-root is an oracle diagnostic, not ordinary control. The free point-speed
metric is geometric evidence, not a known load-bearing contact label. Measured
vertex-interval coverage is 28.4/29.2/31.7 percent respectively. Mean penetration
is below 1 mm, but worst penetration is 54.3/39.1/56.2 mm: averages hide failures.
Boundary joint steps average 35.8/34.4/31.3 mm versus 16.9/16.8/17.1 mm inside
blocks. Idle drift and transition bursts remain visible in the full videos.

Generated history training did not outperform the paired teacher arm on overall
velocity error or this sliding proxy. Do not promote it as a solution. Incorrect
body prediction persists even with source root, while block boundaries and root
control also fail. These observations do not identify one unique cause.

## RTX 3080 Measurement

FP32 eager inference used the actual final baseline weights, not dummy networks.
Root -> Pose -> decoder -> SOMA FK -> CPU packet, 900 calls and 3,600 native frames:

- Mean: 41.28 ms per four-frame block; P95: 63.87 ms; worst: 110.92 ms.
- Model-only mean: 33.20 ms. Peak allocated CUDA memory: 346.3 MiB.
- Four-frame budget: 66.67 ms. 34/900 calls exceeded it.
- Measured average capacity: 96.9 native pose frames/s, not 96.9 model calls/s.
- This was a desktop run, not an isolated GPU benchmark. UE rendering, input
  latency, async scheduling, and buffer underruns were not measured.

Thus mean throughput is sufficient, but stable 60 Hz and UE 60 FPS are NOT proven.
No interpolation or IK was used to claim either result.

## Artifacts And Verification

Current local artifacts: `out/soma_control_v3/`.
`REPORT.md`, `summary.json`, and per-arm `metrics.json` contain the measurements.
Each arm has `rollouts_full.mp4` (all three seeds, full 60 seconds), an official
SOMA-skinned worst-case excerpt, and raw motion/history/goal arrays.
`baseline_backend.pt` is an inference-only portable bundle with trained stats.
`rtx3080_benchmark.json` records weights and runtime hashes; the corresponding
NPZ has 3,600 finite packets and monotonic native 60 Hz publication timestamps.
All evaluation/render jobs exited zero. Local tests: 214 passed, 10 warnings.

Run the portable measurement from the motionbricks directory, with project and
Kimodo on PYTHONPATH and the configured motionbricks environment:

```text
python -m scripts.portable_bones_control benchmark --bundle out/soma_control_v3/baseline_backend.pt --output out/soma_control_v3/new_benchmark.json --iterations 900
```

Next work should target root/body agreement and four-frame boundary continuity,
with the same raw no-IK acceptance protocol. Do not blindly extend training,
add unverified mechanics losses, or migrate failed source motion into UE as a fix.
