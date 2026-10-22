# Native Geometry Execution Sidecar

Date: 2026-10-19. Implementation: `motionbricks/mb_sonic/geometry_execution.py`.

## Scope

The frozen trained MotionBricks + official GEAR SONIC + MuJoCo G1 pipeline remains
the owner of neural inference, scheduling, reference packets and physics. This
sidecar captures existing native geometry calls, independently of the network.
It performs no training, resampling, alternate SDF evaluation, skinning, motion
correction, Root reanchoring or physical state writes.

The existing `mb_hot_profile_v1/report.json` measured whole-proposal P99
103.822504 ms. This work prepares the query/FK/observed-history dispatch reduction;
it does not supply a new CUDA timing result.

## API

```python
query = SceneQueryExecution(
    engine.scene_inputs, (history, goal, origin),
    scene_id=manifest.scene_id,
    source_sha256=manifest.source["scene"]["sha256"],
    bindings=dict(scene=engine.scene, field=engine.field,
                  sampler=engine.controller.sampler),
)
fk32 = SomaFKExecution(inference.global_motion_rep, native_frames32)
fk52 = SomaFKExecution(inference.global_motion_rep, native_frames52)
observe = ObservedHistoryExecution(inference.global_motion_rep, played_frames4)
```

- `SceneQueryExecution` accepts a native tuple-of-13 callback, with fixed tensor
  signatures. Current B1 arguments are history `[1,4,366]`, goal `[1,4]`, and
  explicit SOMA query origin `[1,3]`; a two-argument native callback is also
  supported. All arguments remain dynamic tensor inputs. B1 translation and
  relative-frame feature construction stay in `CompiledMBNeuralExecution.scene_inputs`.
- `SomaFKExecution` supports one fixed `[1,32,366]` or `[1,52,366]` shape per
  instance. It calls `representation.inverse(features, False,
  joint_positions_from="global_rot_data", return_all=False)` and returns only
  `root_pos`, `local_joint_rots`, `global_joint_rots`, `posed_joints`.
- `ObservedHistoryExecution` takes `[1,4,366]` and calls the existing
  `reencode_played_history` with `tensor_checks`. Its dictionary contains `history`,
  `position_error`, `rotation_error`. The next autoregressive context is the
  returned `history` from generated frames `world[:,24:28]`, not their raw codec
  channels. Contacts retain native forecast semantics.

Each instance exposes `.function`, a raw frozen TorchScript function, for outer
full-plan composition. Use these raw functions inside a larger trace/capture;
standalone `__call__` already owns a `StaticCudaCall` and must not be nested in
another capture. Construction warms/captures on allocated FP32 CUDA examples.
Standalone replay requires `torch.inference_mode()`. Its returned tensor snapshots
are cloned by the existing `StaticCudaCall` before subsequent reuse.

Tracing uses existing `trace_fixed_call(shared_cuda_constants=True)`, disabling
freeze numerical optimization and retaining original static tensor storage. Warmup,
tracing, capture and admission use `full_precision_evaluation`: highest FP32 matmul
precision, TF32 disabled. The helper does not change model weights or statistics.

## Binding And Admission

Retain the exact scene/field/sampler and any native tree handles in `bindings`.
The scene ID, source SHA256, full-resolution tensors, storage pointers, shapes,
strides, dtypes/devices and available tensor versions are bound at preparation.
`assert_binding(scene_id=..., source_sha256=..., bindings=...)` checks the current
binding at an admission/rebind boundary. Recreate the instance after scene/source,
field/tree identity or geometry storage changes. Inference tensors lack version
counters, so their captured storage must remain immutable. The callback owner's
scene, field and sampler must remain the retained objects.

Owner-supplied admission cases must include changed history, commands and actual
query origin XYZ. The helpers compare both raw trace and standalone capture to
the original native function:

- `admit_scene_queries(query, [inputs0, inputs1, ...])`: all 13 tensors, exact
  validity masks and integer IDs, floating absolute tolerance `1e-5`, `rtol=0`.
- `admit_soma_fk(fk, [frames0, frames1, ...])`: root/joint Euclidean maximum error
  `3e-5` m and all four output tensor element errors `3e-5`, `rtol=0`.
- `admit_observed_history(observe, [frames0, frames1, ...])`: re-encoded tensor
  parity `1e-5`, `rtol=0`, and original position/rotation reconstruction gates
  `2e-5`. Preserve these returned checks in the outer plan admission.

No gates are widened. Nonfinite outputs fail admission. Unknown/invalid geometry
keeps the original native masks and zero sentinels. Native Python input exceptions
are evaluated during preparation/admission; replay is for the admitted finite
fixed-signature domain. Admission, metrics and identity checks are outside the hot
call. There is no eager fallback or automatic recapture.

## Verification

Worker CPU command, from `${PROJECT_ROOT}`:

```powershell
$env:PYTHONPATH='.;out/vendor/kimodo'
& 'D:/.anaconda/envs/motionbricks/python.exe' -m pytest tests/test_mb_sonic_geometry_execution.py -q
```

First CPU run: **PASS**, 19 passed, 7 skipped, 29 warnings in 9.22 s. Warnings are
PyTorch deprecations from the reused TorchScript helpers. Tests exercise the real
source `000` 500x500 map without resampling, authored stepped/masked geometry,
varying native history/goal/query origin, exact native FK for both shapes, and
played-history re-encoding after stale redundant position channels. They check
source input immutability, exact masks/IDs, frozen representation state and RNG.
CPU trace tests do not initialize CUDA. They are not CUDA performance evidence.

Focused regression command adds `tests/test_mb_sonic_reference.py` to the same
invocation: **PASS**, 51 passed, 7 skipped, 29 warnings in 11.72 s.

CUDA admission: **NOT_RUN** by this worker. Seven real-CUDA tests are opt-in for
the parent after allocating the runtime:

```powershell
$env:MB_SONIC_GEOMETRY_CUDA_TESTS='1'
$env:PYTHONPATH='.;out/vendor/kimodo'
& 'D:/.anaconda/envs/motionbricks/python.exe' -m pytest tests/test_mb_sonic_geometry_execution.py -q
```

Parent validation must also compare real trained model token IDs and full-plan
FK, unchanged model/statistics digests and source identity, then measure the
complete required chain. Standalone helper success does not substitute for those
integration gates.
