# Native MB LAFAN 50k UE trial

Historical failed trial. Its dense-keyframe runtime has been replaced by the
checkpoint-aligned sparse navigation runtime. See
[native_mb_ue_alignment_20261007.md](native_mb_ue_alignment_20261007.md) for the
same-weight diagnosis, corrected PIE results and remaining limitations. The
failure evidence below is retained, not reinterpreted as quality approval.

## Loaded Model

- Tokenizer, Root, Pose: 50,000 optimizer updates each; not completed paper budget.
- Native MotionBricks factories and motion_inference, official SOMA30, native 30 Hz.
- Bundle: out/native_mb_lafan_paper_20261006/ue-trial50k-v1/native-soma-50k.pt
- Bundle SHA256: a17027e3cf8fc4958437329fa91a0889324949f37458a87af36c8d9c824e963c
- Root checkpoint: 832505fed4865a34616202bc63e95b891086f41acd7bd52c08e970ddf270fa9c
- Pose checkpoint: 509b5c9e58f1aed0740e296227e5c1ed583e5a73b8adf96c9f2698f8f9edc49e
- Tokenizer checkpoint: 51e4df8d3f8100431d3a61d767bdfeeeb1ad42b60ccab47055fb950eb13a4771
- Full source/normalization bindings and real train-only reference selections are in the bundle receipt.

Reference poses are input conditions, not played clips. Idle is a genuine quiet
30-frame segment of walk1_subject1 at frame 28, walk is its 120-frame segment
at 7560, and run is run1_subject2 at 1320. No held-out source motion, reversed
clips, stationary synthesis, IK, foot locking, root overwrite or output smoothing.

## Runtime

Python: scripts.native_soma_ue_live_server on 127.0.0.1:28769, RTX 3080, FP32,
TF32 disabled. NativeSmartLocomotion calls the existing official spring,
target-pose placement and inbetween packing methods directly. It uses the
existing four-context/four-target primitive, argmax tokens and fixed reference
selection seed 2333. The output context is discarded and generated frames 4:12
are published as eight immutable 30 Hz frames. Subsequent context is the last
four committed generated frames, with observed planar collision offset only.
This playback integration is a candidate, not NVIDIA's unreleased UE runtime.

UE: F:/dev/GASP_57, MotionBricks SomaControl map, existing CharacterMovement,
Enhanced Input and source-to-Manny hierarchy. Native packet rate initializes
the playback clock and cannot change inside a session. Legacy v1 remains 60 Hz;
v2 supports 30/60 Hz. Presentation copies the retargeted pose without filtering.

Relaunch after closing this Editor/server:

```powershell
& F:\dev\GASP_57\Scripts\RunNativeSomaLocomotion.ps1 `
  -Bundle ${PROJECT_ROOT}\out\native_mb_lafan_paper_20261006\ue-trial50k-v1\native-soma-50k.pt `
  -TrialName native-lafan50k-next
```

The launcher preserves prior evidence and rejects an occupied port; it does not
kill unrelated servers. The trial Editor is visible; background inference is hidden.

## Verification And Failure Evidence

- Local focused Python regressions: 44 passed.
- GASP_57Editor control runtime/editor modules: build succeeded.
- MotionBricks.SomaControl automation: 3 functional checks passed, including
  actual Enhanced Input, requested model identity, native packet rate, retarget
  segment lengths, zero pose blending and two PIE sessions.
- Functional tests are NOT motion quality approval. The optional continuous
  performance gate was not required and its recorded result is false.
- Actual PIE: 50 plans, 391 applied native frames, zero skipped committed frames,
  one warmup underrun and zero steady underruns.
- Measured game-loop rate: 25.374 FPS; frame-time P95: 47.894 ms.
- Forward displacement over the 3-second input phase: 40.807 m, grossly above
  the commanded 1.2 m/s. Final sequence displacement: 130.476 m.
- Maximum applied planar step: 1.072 m per native frame.
- Maximum raw retarget pose switch: 177.809 degrees.
- Screenshot visibly leaves the finite test floor, consistent with excessive
  displacement. This is a failed continuous locomotion quality result.

Evidence remains in:
F:/dev/GASP_57/Saved/MotionBricks/native-lafan50k-20261007 and
F:/dev/GASP_57/Saved/MotionBricks/soma_control_pie_causal_native-lafan50k-20261007.json.

Do not infer an integration-only root cause or a model-only root cause from
these data. Four-keyframe conditioning at both ends is also beyond the Pose
50k one-keyframe curriculum schedule. Compare packed conditions and generated
Root/Pose/decoder stages against the native evaluator before changing anything.
Do not clamp root motion, blend poses, lock feet or change weights to conceal
the failure. No training, data conversion or old bundle replacement occurred.
