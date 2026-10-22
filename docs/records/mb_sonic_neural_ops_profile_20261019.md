# Frozen Native Neural Operator Profile

Date: 2026-10-19. Script: `scripts/profile_mb_sonic_neural_ops.py` (119 lines).

## Allocation And Dependencies

Only the parent runs this CLI, once, under its external 300-second cap. It requires
an explicit allocation ID, fresh output directory, `--device cuda`, and existing
artifact/bundle/scene-manifest paths. Dependencies are the existing motionbricks
environment (Python 3.10.20, Torch 2.11.0+cu128, NumPy and native project
dependencies), plus functioning CUDA/CUPTI profiling. No additional installation,
TensorRT dependency, server, SSH connection or training is needed.

From `${PROJECT_ROOT}`, after the parent assigns the ID:

```powershell
$env:PYTHONPATH='.;out/vendor/kimodo'
& 'D:/.anaconda/envs/motionbricks/python.exe' scripts/profile_mb_sonic_neural_ops.py `
  --allocation-id $env:MB_SONIC_ALLOCATION_ID --device cuda `
  --artifact out/pfnn_mixed_20261015/final_acceptance_flat_v1 `
  --bundle out/a100_live_20261010/a100-terrain.pt `
  --manifest out/mb_sonic_e0_e3_20261018/scenes/000/scene_manifest.json `
  --output out/mb_sonic_e0_e3_20261018/neural_ops_v1
```

## Measurement

`TrainedMBReferenceAdapter.load` checks the original trained asset identities and
sets highest FP32 precision with TF32 disabled. The script reads source `000`'s
real four-frame initial history without resampling and supplies native controls
`(vx,vz,cos_heading,sin_heading)=(0,0.5,1,0)`. It calls the unchanged
`controller.forward` three times for warmup, then profiles one 8-token invocation.
There is no second autoregressive block, added FK, skinning or optimizer.

Only these outer modules receive temporary `record_function` hooks:

- `controller.root`, the actual `inference._root_model.backbone_net`: one call.
- `controller.pose`, the actual `inference._pose_model.backbone_net`: four passes.
- `controller.codec.decoder`, the actual pose VQ-VAE `DoubleCondDecoder`: one call.

Pre-hooks are prepended so existing terrain-adaptation hooks remain inside the
region. The original hooks retain their relative order and inputs/outputs. No
Transformer layer or child gets a diagnostic hook, and no Transformer fastpath,
network parameter or model implementation is changed. Hooks are removed on exit.

The report uses the locally inspected Torch 2.11 APIs `device_time_total`,
`self_device_time_total`, `DeviceType`, and `key_averages(group_by_input_shape=True)`.
It requires positive real CUDA activities and positive device totals with exactly
the declared region counts. Missing CUPTI/kernel evidence or changed counts
produce `BLOCKED` and exit code 2, never GPU timings inferred from CPU values.
Region counts/totals select CPU events only. CUDA user annotations are excluded
from the raw device-activity sum; they describe overlapping spans, not kernels.

## Outputs

- `report.json`: allocation ID, old model identity hashes, model-state and source
  hashes before/after, precision, zero optimizer steps, region inclusive device
  totals, per-region operator-family self device time, and op shapes/call counts.
- `operators.txt`: complete shape-grouped raw profiler table, sorted by self
  device time. Its CPU columns remain CPU measurements.
- `trace.json`: Chrome trace from CPU+CUDA profiling with shape recording.

All timing fields in JSON end in `_us`. CUDA activity sums include device-side
kernel/copy durations, not Python idle or elapsed wall time. Family sums use CPU
operator events' attributed **self** device time only; raw CUDA events are not
added again. Region inclusive totals are a separate view and are not added to
family totals. Fused Transformer kernels are explicitly grouped with attention,
not presented as isolated attention FLOPs. `outside_neural` means outside the
three named regions; it also contains native terrain encoders and query/setup ops.
These measurements characterize the native eager forward, not CUDA-graph latency.
Use the resulting conv/GEMM/fused-attention mix and shapes to select a kernel or
export route after actual GPU measurement.

## Verification

Worker command: `python -m pytest tests/test_mb_sonic_neural_ops_profile.py -q`.
Result: **PASS**, 25 passed in 2.33 s. These are isolated CPU parser, accounting,
schema, hook-cleanup and local-profiler-API tests; they load no model assets and
do not initialize CUDA. GPU operator profiling by this worker: **NOT_RUN**.

## Retained V2 Offline Review

The parent's `mb_neural_operator_profile_v2/execution.json` records exit 2,
18.9813269 seconds allocated wall time, and identical source hashes before/after.
The retained `result/report.json` stays unchanged with its original `BLOCKED`
status. The aggregation failure was the region dictionary accepting CUDA
`gpu_user_annotation` rows, overwriting CPU counts (Root 5 versus 1; decoder 3
versus 1). It was not a missing-CUPTI failure: the trace records CUPTI 26 and
2,957 real kernel events on RTX 3080. A source-only fix now selects CPU region
rows and excludes CUDA user annotations from the activity sum. The duplicate-name
CPU regression includes all three names. Updated CPU tests: **PASS**, 26 passed
in 2.11 s. No GPU profiling was repeated.

Offline analysis reads the retained Chrome trace's `cat=kernel` durations. Every
kernel links by `args["External id"]` to one CPU op; that op's host start position
within the six CPU user-region intervals determines its owner. All 2,957 kernels
link successfully. These are **linked kernel-duration sums**, not recreated
`FunctionEvent` objects, profiler self-device metrics or host annotation spans.

| Native Region | CPU Calls | Linked Kernel Sum (ms) |
|---|---:|---:|
| Root backbone, including its root decoder | 1 | 5.396546 |
| Pose backbone | 4 | 8.412942 |
| Pose codec decoder | 1 | 4.227428 |
| Outside the three named regions | - | 1.865512 |
| All 2,957 kernels | - | 19.902428 |

Memcpy and memset categories add 0.339032 and 0.041084 ms respectively; including
these gives 20.282544 ms of non-annotation CUDA activity. Neither that sum nor
the six host-region spans is the unprofiled full-plan elapsed time.

Dominant linked operator families are convolution **7.773607 ms** and GEMM
**6.745573 ms**, together 73.0% of the kernel sum. Pose accounts for 6.077138 ms
of GEMM. Root and pose-decoder convolution account for 3.900948 and 3.836625 ms.
All attention kernels combined account for 1.112654 ms.

The first narrow optimization target is the existing FP32 decoder convolutions:
`[1,512,1,L]` input, `[512,512,1,3]` weights, `L=16/32/64`. Their dilation-1,
stride-1, same-padded cases alone consume **5.159108 ms** across both decoders.
The original operator table shows 2.495/2.012/1.863 ms self CUDA for the complete
shape groups at L=32/16/64, including the other recorded dilations. Existing-stack
fixed-shape algorithm selection or an exact FP32 execution replacement is a more
targeted candidate than a backend reinstall. Keep original weights, bias,
padding/dilation, masks and residual order; admit native token/FK parity before
using any replacement.

Second is Pose's small-M FP32 GEMM: QKV `[8,1024]@[1024,3072]` costs 1.779406 ms;
FFN `[8,2048]@[2048,1024]` costs 1.425752 ms; fused FFN input
`[8,1024]@[1024,2048]` costs 1.350164 ms. These are useful fixed-shape GEMM/epilogue
targets without reducing four Pose passes, changing horizon, enabling TF32 or
changing precision. Root's GEMM/attention/norm together are only about 0.741 ms,
so Root-prefix is not the dominant opportunity in this profile.

Retained inputs are unchanged. SHA256 of `trace.json`:
`f6c16bea93b461ee1c069fd9c25a910435d7c1df591ae57ee6c1af9b6a14286f`;
`operators.txt`: `853abf82baaec9f33cf608fd029421ec65a93616f482618cb8cfbf10046ad2f40`;
`report.json`: `055543a289a6d55a207f90134aecbb7b31dd25ffc4681299f53f9911f89f0071`.
