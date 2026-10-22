# MotionBricks UE Shared-Memory Protocol

This is the single canonical shared-memory protocol for the MotionBricks UE
bridge. Python must match the UE C++ ground truth exactly:

- `F:\dev\GASP_57\Source\GASP_57\Public\MotionBricksBridge\MotionBricksSharedMemoryBridgeBackend.h`
- `F:\dev\GASP_57\Source\GASP_57\Private\MotionBricksBridge\MotionBricksSharedMemoryBridgeBackend.cpp`
- `F:\dev\GASP_57\Source\GASP_57\Public\MotionBricksBridge\MotionBricksBridgeBackendTypes.h`

The UDP server remains a separate transport and is not part of this protocol.

## Win32 Objects

Default object names from `FMotionBricksBridgeRuntimeConfig`:

| Object | Default name |
| --- | --- |
| Command map | `Local\MotionBricks_Command` |
| Frame map | `Local\MotionBricks_Frames` |
| Command mutex | `Local\MotionBricks_CommandMutex` |
| Frame mutex | `Local\MotionBricks_FrameMutex` |
| Command event | `Local\MotionBricks_CommandReady` |
| Frame event | `Local\MotionBricks_FrameReady` |

Default sizes and capacity:

| Field | Value |
| --- | ---: |
| Command memory size | 4096 bytes |
| Frame memory size | 1048576 bytes |
| Frame packet capacity | 32 slots |
| Shared memory timeout | 0.005 seconds |

The Python runner uses Win32 APIs through `ctypes`:
`CreateFileMappingW`, `OpenFileMappingW`, `MapViewOfFile`, `CreateMutexW`,
`OpenMutexW`, `CreateEventW`, `OpenEventW`, `WaitForSingleObject`,
`ReleaseMutex`, and `SetEvent`.

If maps or sync objects already exist, the runner reuses them. Otherwise it
creates them.

## Constants

| Constant | Value |
| --- | ---: |
| Protocol version | `1` |
| Command magic | `0x4D42434D` (`MBCM`) |
| Frame magic | `0x4D424652` (`MBFR`) |
| Max mode bytes | 64 |
| Max name bytes | 64 |
| Max allowed token count | 32 |
| Max qpos elements | 64 |
| Max skeleton bones | 64 |
| Max DOF names | 64 |
| Max frames per packet | 4 |
| Expected qpos length | 36 |
| Expected bone count | 34 |

All structures use the UE C++ default packing/alignment on Windows x64.

## Structures

### FSharedMemoryHeader

```cpp
uint32 Magic;
uint32 Version;
uint32 HeaderSizeBytes;
uint32 TotalSizeBytes;
uint32 Sequence;
uint32 Flags;
double LastWriteSeconds;
```

Python/UE size: 32 bytes.

### FCommandBuffer

```cpp
FSharedMemoryHeader Header;
int32 CommandSequenceId;
double ClientTimeSeconds;
float MovementDirectionMj[3];
float FacingDirectionMj[3];
ANSICHAR ModeUtf8[64];
int32 ModeIndex;
int32 RequestFrames;
int32 AllowedPredNumTokensCount;
int32 AllowedPredNumTokens[32];
```

Python/UE size: 280 bytes.

UE writes commands while holding `CommandMutex`, increments
`Header.Sequence`, then signals `CommandEvent`. Python consumes a command when
`Header.Sequence` changes.

### FFixedFrame

```cpp
double TimeSeconds;
int32 QposCount;
float Qpos[64];
float RootXyzM[3];
float RootQuatWxyz[4];
int32 BoneGlobalPositionCount;
float BoneGlobalPositionsM[64][3];
```

Python/UE size: 1072 bytes.

Shared-memory v1 stores qpos, root, and global bone positions. The current UE
ground-truth structure has no rotation field; rotations emitted by
MotionBricks are therefore not serialized in this transport.

### FFramePacketSlot

```cpp
uint32 SlotSequence;
uint32 PacketFlags;
int32 PacketSequenceId;
double ServerTimeSeconds;
float SourceHz;
ANSICHAR SchemaVersionUtf8[64];
int32 BoneNameCount;
ANSICHAR BoneNamesUtf8[64][64];
int32 ParentCount;
int32 Parents[64];
int32 DofNameCount;
ANSICHAR DofNamesUtf8[64][64];
int32 FrameCount;
FFixedFrame Frames[4];
```

Python/UE size: 12848 bytes.

Python writes one `FFramePacketSlot` per generated packet. It fills skeleton
names, parents, and DOF names from the MotionBricks packet when available.

### FFrameRingBuffer

```cpp
FSharedMemoryHeader Header;
int32 Capacity;
int32 SlotSizeBytes;
uint32 WriteSequence;
uint32 LatestSlotIndex;
int32 ExpectedQposLength;
int32 ExpectedBoneCount;
int32 MaxFramesPerPacket;
```

Python/UE size: 64 bytes.

The frame map layout is:

```text
FFrameRingBuffer
FFramePacketSlot[Capacity]
```

Required size for default capacity 32:

```text
sizeof(FFrameRingBuffer) + 32 * sizeof(FFramePacketSlot)
= 64 + 32 * 12848
= 411200 bytes
```

The default frame map is 1048576 bytes, leaving room for the default ring.

## Write Flow

Python frame writer:

1. Wait for `FrameMutex`.
2. Validate or initialize `FFrameRingBuffer`.
3. Compute `WriteSequence + 1`.
4. Write slot index `(WriteSequence - 1) % Capacity`.
5. Set `SlotSequence` to the new write sequence.
6. Fill packet metadata, skeleton metadata, and up to 4 fixed frames.
7. Set `WriteSequence`, `LatestSlotIndex`, `Header.Sequence`, and `Header.LastWriteSeconds`.
8. Release `FrameMutex`.
9. Signal `FrameEvent`.

UE reader:

1. Polls or waits on `FrameEvent`.
2. Holds `FrameMutex`.
3. Reads all slots with `SlotSequence` between its last read sequence and
   `WriteSequence`.
4. Drops overwritten or stale packets according to UE runtime config.

## Smoke Test

Run from the `motionbricks` directory:

```powershell
conda run -n motionbricks python scripts\ue_bridge_shared_memory_runner.py --smoke_shared_memory --max_steps 1 --command_map "Local\MotionBricks_Command" --frame_map "Local\MotionBricks_Frames" --command_mutex "Local\MotionBricks_CommandMutex" --frame_mutex "Local\MotionBricks_FrameMutex" --command_event "Local\MotionBricks_CommandReady" --frame_event "Local\MotionBricks_FrameReady" --frame_capacity 32 --command_memory_size 4096 --frame_memory_size 1048576
```

Smoke behavior:

1. Creates or opens all canonical UE map/mutex/event names.
2. Writes a default idle command into `FCommandBuffer`.
3. Reads the command back through the same structure.
4. Calls `MotionBricksLiveServer.step(command, 0)`.
5. Writes one `FFramePacketSlot` into `FFrameRingBuffer`.
6. Reads the latest slot back and validates:
   - `qpos_len == 36`
   - `bone_count == 34`
   - `abs(norm(root_quat_wxyz) - 1) <= 1e-3`

Expected success line:

```text
shared-memory smoke ok: write_sequence=1 slot_index=0 qpos_len=36 bone_count=34 root_quat_norm=1.000000
```

## Live Runner

Run from the `motionbricks` directory with UE default names:

```powershell
conda run -n motionbricks python scripts\ue_bridge_shared_memory_runner.py --command_map "Local\MotionBricks_Command" --frame_map "Local\MotionBricks_Frames" --command_mutex "Local\MotionBricks_CommandMutex" --frame_mutex "Local\MotionBricks_FrameMutex" --command_event "Local\MotionBricks_CommandReady" --frame_event "Local\MotionBricks_FrameReady" --frame_capacity 32 --command_memory_size 4096 --frame_memory_size 1048576 --tick_hz 30 --use_qpos 1 --planner default --clips G1
```
