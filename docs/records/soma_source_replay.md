# SOMA Source Replay in UE 5.7

This is M0 recorded-source diagnosis, not trained MotionBricks inference or WASD-controlled generation.

## Source and Processing

- Four real BONES-SEED recordings from the corrected V7 pilot: `idle_loop_002__A065`, `walk_forward_start_002__A023`, `jog_forward_stop_003__A021`, `turn_walk_270_003__A070`.
- Official Kimodo SOMA77 conversion, SOMA30 reduction and standard reference assets at revision `58e781898b3d7e328a676a75d3e338c45dce3ad9`.
- 60 Hz, anti-aliased resampling, boundary calibration removal, one inferred support plane per clip and MB canonical heading/translation. No trained weights, G1 conversion or generated walk cycle.
- Reference root height uses the official mesh-derived sole proxies. Heel/toe contacts are automatic confidence-masked estimates, not manual ground truth.
- Versioned fixture includes ordered names/parents, reference positions, local XYZW rotations, root translation, Python global FK references, source/shard digests and explicit `source_replay` state.

## UE Route

Project: `F:/dev/GASP_57/GASP_57.uproject`.

Map: `/MotionBricksGroot/SomaReplay/L_SomaSourceReplay`.

Assets: `/MotionBricksGroot/SomaReplay/soma30_kimodo_standard`.

The replay Actor loads each fixture once, validates all source frames against official Python FK, samples local rotations with SLERP and computes FK before native `FIKRetargetProcessor` execution. Global positions are not linearly interpolated; source and target bone lengths stay fixed.

The target uses a SkeletalMeshComponent and native AnimInstance proxy. UObject reads and value snapshots happen in PreUpdate on the game thread. Animation evaluation consumes only the copied pose buffer with compact-to-mesh bone mapping. No file I/O, inference, foot IK, inertialization or pose smoothing is hidden in this diagnostic.

SOMA skeletons and Manny meshes appear side by side for idle, start, jog-stop and turn. Planar translation is removed only for this comparison view; source motion and FK validation retain the actual root path. The details panel can disable `Display In Place` or pause `Playing`; Blueprint functions provide exact seeking. Loop resets are recording resets, not learned motion transitions.

## Reproduce

From the repository, with the motionbricks Conda environment and Kimodo on PYTHONPATH:

```powershell
$env:PYTHONPATH=(Resolve-Path 'motionbricks/out/vendor/kimodo').Path
python motionbricks/scripts/export_soma_ue_replay.py --dataset motionbricks/out/bones_pilot_reports/pilot-v7 --output motionbricks/out/soma_ue_replay
```

The local review dataset contains the accepted manifest, skeleton contract and only the four required NPZ shards, not the whole training corpus. Copy the generated fixtures and hierarchy folder into `F:/dev/GASP_57/Plugins/MotionBricksGroot/RuntimeData/SomaReplay`.

```powershell
& 'F:/dev/GASP_57/Scripts/RunSomaSourceReplay.ps1' -Build -Author -Test
```

This launches the dedicated map in the editor after building the MotionBricks plugin modules, authoring and testing. Existing game-module binaries are required. The script uses a process-scoped writable local DDC fallback because this machine's Zen/default cache was unavailable. It does not change global cache settings.

Reports are saved under `F:/dev/GASP_57/Saved/MotionBricks/soma_replay_authoring.json` and `F:/dev/GASP_57/Saved/Automation/SomaReplay`. PIE screenshots: `F:/dev/GASP_57/Saved/MotionBricks/soma_source_replay_pie.png` and `soma_walk_replay_closeup.png` in the same directory.

## Boundary

Passing source FK, bone lengths and visible-pose parity does not certify animation aesthetics, target sole contact, trained reconstruction, live controls or 60 Hz inference. Those remain M1-M5 gates. The human gameplay Character/Enhanced Input path is separate from these four recording actors; existing G1 gameplay defaults are not relabeled as SOMA.

Known editor diagnostic: the source skeletal-mesh carrier contains only a reference hierarchy and no render LOD. UE may emit a Min LOD warning for this specific asset; the source is drawn as bones, not rendered through that mesh. The fixture test explicitly recognizes only that warning. Unexpected asset/runtime warnings still fail the launcher test gate.

Concurrent project changes added unrelated character/cloth code with invalid UE 5.7 calls (`SetDisableClothSimulation`, `SetWind`). A full-project build is currently blocked by those files, which this work did not modify. The MotionBricks module build and PIE route are verified separately against the existing game-module binaries; this is not a claim that all current project source compiles.

## Numerical Evidence

The 1,358 recorded frames have a maximum UE/Python source FK discrepancy of 0.00007245 cm and maximum global-rotation discrepancy below 1.4e-7 radians. Target segment lengths differ from the target reference by less than 3e-13 cm. These are geometry/replay parity measurements, not neural reconstruction scores.

Native reference validation additionally passes torso facing, seven limb/body chain probes, transformed coordinate-basis consistency, root translation response and rejection of wrong identity/count, NaNs, scales and modified references/parents. Anatomical retargeting maps a 100 cm source displacement to 94.91 cm target root displacement; the later live movement adapter must consume the original model root trajectory exactly once, not mistake this anatomical scale for model control accuracy.

The Python checks cover reflected-axis/quaternion conversion, 90-degree yaw, isolated left/right limb FK, official reference consistency and explicit replay state. UE automation checks all source frames, subframe bone-length preservation, identity/FK corruption rejection, visible target parity and two PIE sessions with live playback/resume.

Final checks on 2026-10-04: 17 focused Python tests passed; the two `MotionBricks.SomaReplay` UE tests passed with zero unexpected warnings/errors on the local RTX 3080/D3D12. MotionBricks plugin-module compilation passed. The unrelated full-project build limitation above remains open. No V7 optimization steps or new model weights were produced.
