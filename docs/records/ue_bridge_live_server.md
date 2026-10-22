# MotionBricks UE Bridge Live Server

`scripts/ue_bridge_live_server.py` is a small standalone Python server for UE
debug playback. It does not change MotionBricks core code and does not use
raylib. Motion is still produced through the official path:

1. `navigation_demo(args)`
2. `full_agent.get_next_frame()`
3. context from `get_context_mujoco_qpos()` by default
4. `full_agent.generate_new_frames(control_signals, controller_dt * generate_dt)`

## Protocol Choice

This worker implements UDP JSON packets, not TCP JSONL.

Reason: live UE control usually wants the newest command and newest frame. UDP
keeps the bridge stateless, easy to restart, and tolerant of UE reconnects. The
server drains all pending command datagrams every tick and uses only the latest
valid command. If the Live-Protocol worker later standardizes TCP JSONL, the
command and frame objects below are the fields to preserve.

Default ports:

- Server listens for UE commands on `127.0.0.1:9010`.
- Server sends frame packets to `127.0.0.1:9011`.
- If `--send_port 0` is used, frames are sent to the most recent command
  sender instead.

## Command Packet

UE sends one JSON object per UDP datagram:

```json
{
  "type": "control",
  "mode": "walk",
  "movement_direction_mj": [1.0, 0.0, 0.0],
  "facing_direction_mj": [1.0, 0.0, 0.0],
  "allowed_pred_num_tokens": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
}
```

Fields:

- `mode`: MotionBricks clip mode string. The server maps it through
  `full_agent._clip_holder.CLIPS`. Unknown modes fall back to `idle`, then
  `walk`, then the first available mode, and print a warning once.
- `movement_direction_mj`: MuJoCo-basis `[x, y, z]`, converted to torch
  float32 shape `[1, 3]`.
- `facing_direction_mj`: MuJoCo-basis `[x, y, z]`, converted to torch float32
  shape `[1, 3]`. A zero vector falls back to `[1, 0, 0]`.
- `allowed_pred_num_tokens`: optional. If missing, the server uses
  `controller.get_default_allowed_pred_num_tokens(mode_id)`.

Aliases `movement_direction` and `facing_direction` are accepted for early
prototyping, but `*_mj` is the preferred spelling.

If no command has arrived yet, the server uses this idle command:

```json
{
  "type": "control",
  "mode": "idle",
  "movement_direction_mj": [0.0, 0.0, 0.0],
  "facing_direction_mj": [1.0, 0.0, 0.0]
}
```

## Frame Packet

The server sends one JSON object per UDP datagram:

```json
{
  "type": "frame",
  "protocol": "udp_json",
  "schema_version": "motionbricks_ue_live.v1",
  "source_step": 12,
  "time": 0.4,
  "qpos": [0.0],
  "root_xyz_m": [0.0, 0.0, 0.0],
  "root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
  "bone_global_positions_m": [[0.0, 0.0, 0.0]],
  "bone_global_rotations_wxyz": [[1.0, 0.0, 0.0, 0.0]],
  "skeleton": {
    "bone_count": 34,
    "bone_names": ["pelvis_skel"],
    "parents": [-1]
  },
  "schema_header": {
    "qpos_length": 36,
    "root_quat_order": "wxyz",
    "schema_hash": "..."
  },
  "command": {
    "mode": "walk",
    "mode_id": 1,
    "context": "context_mujoco_qpos"
  }
}
```

Important invariants:

- Current live packets use schema version `motionbricks_ue_live.v1`. The older
  `motionbricks_ue_bridge.live.v1` value is historical and should only be
  treated as legacy compatibility.
- `qpos` length is 36.
- `root_quat_wxyz` is normalized and ordered `[w, x, y, z]`.
- `bone_global_positions_m` shape is `[34][3]`.
- `bone_global_rotations_wxyz` shape is `[34][4]`.
- Positions and rotations are in MuJoCo world basis:
  `X forward / Y left / Z up`.
- UE candidate position conversion is `(x, -y, z) * 100`.
- Rotations require a basis change; do not only reorder quaternion components.

## CLI

Smoke test without UE:

```powershell
conda run -n motionbricks python scripts\ue_bridge_live_server.py --smoke_test --max_steps 5
```

Run the live server:

```powershell
conda run -n motionbricks python scripts\ue_bridge_live_server.py `
  --host 127.0.0.1 --port 9010 `
  --send_host 127.0.0.1 --send_port 9011 `
  --tick_hz 30
```

Supported live options:

- `--host`, `--port`: UDP command bind address.
- `--send_host`, `--send_port`: UDP frame destination.
- `--max_steps`: `0` means run until interrupted.
- `--tick_hz`: live server tick rate.
- `--print_packets`: print outgoing frame JSON.
- `--smoke_test`: run local idle/walk commands and validate output.
- `--use_qpos`: defaults to `1`, so generation uses `context_mujoco_qpos`.

## Smoke Validation

`--smoke_test` runs a few ticks with idle then walk commands, validates:

- qpos length is 36.
- bone count is 34.
- root quaternion norm is within `1e-3` of 1.

It writes the final packet to:

```text
motionbricks/out/ue_bridge_live/smoke_last_frame.json
```

## Assumptions For Protocol Alignment

- UDP JSON datagram size is acceptable for local UE/MotionBricks development.
  If this becomes fragile over a routed network, switch the same objects to TCP
  JSONL.
- Commands are absolute MuJoCo-world intent vectors, not UE-basis vectors.
- The server sends schema/skeleton on every frame for simple consumers. A later
  protocol can split this into a one-time `schema` packet plus lean `frame`
  packets if needed.
