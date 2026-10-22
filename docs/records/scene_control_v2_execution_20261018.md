# Scene Control V2 Execution

2026-10-18, Asia/Singapore. Machine-readable state: `out/scene_control_v2_20261018/execution_status.json`.

## Current State

- 28945/PID67404 still runs the old source. Tab10/011 is Stopped at frame39120, CMD1, scene6/settings0, ACK9780/39120, packet9781/4. The new pending-packet failure fix is CPU-verified and **pending deployment**; UI/map/service were not reset.
- Desktop body/terrain pixels were seen before failure; temporary tab9 W press/release advanced CMD1 to3, then tab9 closed. On011, transient11.5Hz and later idle29.5-29.7Hz are separate observations; loaded sustained playback is unmeasured.
- Initial spawn has sufficient known coverage. Failure-time Root/query coordinates and traceback are missing, so the all-unknown geometry cause remains unresolved. Original screenshots and stderr are retained.
- Boundaries: existing model assets and the646-frame PARC simulation retarget are real artifacts; region fixtures and the small optimizer diagnostic are synthetic engineering; contact head and8x128 codebook are untrained. The provider is engineering-only; quality approval remains false.

## Actual Checks

| Check | Result |
| --- | --- |
| Final CPU combination | **172 passed, 2 CUDA cases skipped**, 9 upstream warnings |
| Node protocol | **19 passed** |
| Real provider, CPU2threads | Three CLI batches12/16/8; zero Adam; model/codec/EMA unchanged |
| Provider coordinates | Global366, T+1 endpoints; root3.8147e-6m / FK5.8010e-6m, gates1e-5/3e-5m |
| Provider observations | Native prismSDF, trained0.4m 2D spacing, path-tangent corridor |
| Labels | All valid codes including latent0; independent motion-prefix mask; normalization-bound cache rejection |
| Runtime source fix | 33 focused CPU cases; ACK/played before deferred candidate failure, send-failure/close handling |
| Previous-source GPU parity | 4x8,4x12,8x16 passed; not rerun after source patch |

Conditional CUDA wrapper check stopped at measured52% GPU utilization (threshold40%):0cases,0allocation,0retry. Evidence is `cuda_wrapper_vjp_v1.json`. Runtime and provider proofs are in the same output directory.

CLI mapping: `python -m scripts.train_scene_control_v2 --config configs/scene_control_v2_real_provider_cpu.json --output <fresh-relative-output> --mode forward --factory motionbricks.scene_control_v2.provider:make_real_engineering_factory --seed 2338`.

## Budget

- Unique optimizer diagnostic remains4 Adam, batch2,8 engineering examples; its five original files are unchanged.
- Provider forwards, conversion/codec checks and runtime proofs add0 training exposures. New formal Adam/exposures:0; completed legacy38.4M stays separate.

## Specific Next Steps

1. After remote access, choose the existing target project root and verify the overlay's relative-path/SHA manifest, including required dirty-base core dependencies. Weights, data and toy checkpoints are excluded; baseGitHEAD alone is insufficient. No remote root or GPU allocation is assumed.
2. Deploy the pending runtime patch only under a new instruction; capture failure-time Root/query/traceback before attributing geometry cause.
3. Complete record/asset licensing, full-body contact surfaces/events and matched recovery teachers before formal data admission or new training allocation.
4. Verify four-direction/moving4/8 matrix, mobile, strict input latency and full18056/8-substep/0.02m/contact/route quality gates. No further GPU wrapper attempt is scheduled.
