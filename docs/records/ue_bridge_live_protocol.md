# MotionBricks UE Live Protocol

This document defines the Step 2 live bridge protocol boundary between UE and MotionBricks. It is intentionally small: it describes JSON messages and UE-side data structs, but it does not define sockets, threading, retry behavior, or playback/retarget implementation.

## Ground Truth

- MotionBricks demo control inputs are `control_signals` with `movement_direction`, `facing_direction`, integer `mode`, and optional `allowed_pred_num_tokens`. The directions are documented in `full_agent.generate_new_frames(...)` as global MuJoCo-coordinate vectors.
- The WASD controller emits horizontal MuJoCo directions with `z = 0`; idle derives movement from recent qpos velocity and facing from root quaternion using scalar-first order.
- `mujoco_qpos` is 36 elements: root xyz `[0..2]`, root quaternion `[3..6]` in `[w, x, y, z]`, and 29 hinge DOF `[7..35]`.
- MotionBricks Python generation is 30 Hz (`base_controller._FPS = 30`) and regenerates every 8 frames by default. The deployment boundary keeps planner updates separate from control: planner thread 10 Hz, generated planner qpos interpreted as 30 Hz, resampled to 50 Hz control-rate motion.
- UE 5.7 local headers define `FVector` as `UE::Math::TVector<double>` and `FQuat` as `UE::Math::TQuat<double>`. `FQuat` stores `X, Y, Z, W`; incoming MotionBricks root quaternions remain protocol fields named `root_quat_wxyz` and must be reordered only when placing them into `FQuat` storage.
- UE JSON support is from `Runtime\Json` (`FJsonObject`, `FJsonValue`, `TJsonReaderFactory`, `FJsonSerializer`). This protocol header does not depend on network modules.

## Transport Boundary

Step 2 may carry each message as either:

- one UTF-8 JSON object per line over a stream, or
- one UTF-8 JSON object per UDP datagram.

Transport code is outside this protocol module. The receiving side should treat malformed JSON or schema mismatch as a dropped packet.

All live messages use:

```json
"schema_version": "motionbricks_ue_live.v1"
```

`sequence_id` is monotonic per UE client command. MotionBricks should echo the latest command sequence in frame packets. UE runtime should be able to ignore older frame packets; Step 2 can start with a single "last accepted sequence id" check and tolerate packet drops.

## UE to MotionBricks Command

Required fields:

```json
{
  "schema_version": "motionbricks_ue_live.v1",
  "sequence_id": 42,
  "client_time_s": 12.345,
  "movement_direction_mj": [1.0, 0.0, 0.0],
  "facing_direction_mj": [1.0, 0.0, 0.0],
  "mode": "walk"
}
```

Optional fields:

```json
{
  "mode_index": 0,
  "allowed_pred_num_tokens": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  "request_frames": 8
}
```

Field rules:

- `movement_direction_mj`: `[x, y, z]` unit or near-unit vector in MuJoCo world basis. For locomotion commands, `z` must be `0`.
- `facing_direction_mj`: same basis and horizontal rule as movement.
- `mode`: stable human-readable mode string such as `idle` or `walk`.
- `mode_index`: optional backend-specific integer. Python MotionBricks currently consumes an integer tensor mode, so the server may map `mode` to this value.
- `allowed_pred_num_tokens`: optional mask compatible with MotionBricks root model token selection.
- `request_frames`: optional requested frame count. The server may clamp to its generation/token boundaries.

## MotionBricks to UE Frame Packet

Recommended packet:

```json
{
  "schema_version": "motionbricks_ue_live.v1",
  "sequence_id": 42,
  "server_time_s": 12.367,
  "source_hz": 30.0,
  "frames": [
    {
      "time_s": 12.367,
      "qpos": [0.0, 0.0, 0.72, 1.0, 0.0, 0.0, 0.0],
      "root_xyz_m": [0.0, 0.0, 0.72],
      "root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
      "bone_global_positions_m": [[0.0, 0.0, 0.72]]
    }
  ]
}
```

`qpos` must always be length 36 in real packets; the shortened example above only illustrates layout. `root_xyz_m` and `root_quat_wxyz` should duplicate `qpos[0..2]` and `qpos[3..6]` for parser clarity. `bone_global_positions_m` is optional and intended for debug visualization/joint-cloud preview.

The server may send one frame per packet or a short array of future frames. UE should consume packet order by `sequence_id` first, then frame `time_s` within the packet.

## Skeleton / Hello Packet

Skeleton metadata should be sent once at startup or repeated occasionally in frame packets:

```json
{
  "schema_version": "motionbricks_ue_live.v1",
  "message_type": "schema",
  "source_hz": 30.0,
  "qpos_length": 36,
  "root_quat_order": "wxyz",
  "skeleton": {
    "bone_names": ["pelvis_skel"],
    "parents": [-1],
    "dof_names": ["left_hip_pitch_joint"]
  }
}
```

The canonical G1 debug skeleton has 34 bones. The canonical MuJoCo qpos has 29 hinge DOF after the 7 root elements.

## Coordinate Rules

MotionBricks qpos and live commands use MuJoCo world basis:

- MuJoCo: `X forward`, `Y left`, `Z up`
- UE: `X forward`, `Y right`, `Z up`

UE input must be converted to global MuJoCo XY before sending. For example, a UE debug vector in centimeters can become a MuJoCo horizontal direction by unscaling, flipping Y, zeroing Z, and normalizing:

```text
movement_direction_mj = normalize([UE.X, -UE.Y, 0])
```

For Step 2 debug visualization only, qpos and bone positions may use this candidate conversion:

```text
FVector UE_cm = FVector(MJ.x * 100, -MJ.y * 100, MJ.z * 100)
```

Quaternion basis change is not done in Step 2 unless explicitly implemented and tested. Reordering `[w, x, y, z]` into UE `FQuat(x, y, z, w)` is only a storage conversion; it does not convert MuJoCo basis to UE basis and must not be treated as a correct UE skeletal rotation.

## UE Module Boundary

The UE protocol structs live in:

- `Source/GASP_57/Public/MotionBricksBridge/MotionBricksBridgeProtocol.h`
- `Source/GASP_57/Private/MotionBricksBridge/MotionBricksBridgeProtocol.cpp`

They provide:

- schema/version constants,
- `FMotionBricksLiveCommand`,
- `FMotionBricksLiveFramePacket` and nested frame/skeleton/quat structs,
- simple validation helpers,
- one debug-only MuJoCo meters to UE centimeters position conversion.

They deliberately do not include Sockets, EnhancedInput, component ticking, JSON reader/writer implementation, playback, retargeting, or Build.cs changes.
