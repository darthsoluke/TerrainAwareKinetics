# MotionBricks Shared Memory Protocol

Status: canonical UE implementation contract for `BackendType=SharedMemory`.

Ground truth:

- `Source/GASP_57/Public/MotionBricksBridge/MotionBricksSharedMemoryBridgeBackend.h`
- `Source/GASP_57/Public/MotionBricksBridge/MotionBricksBridgeBackendTypes.h`

This is not the older single-region design. UE now uses two Windows file mappings: one command buffer written by UE, and one frame ring buffer written by Python.

## Backend Names

Default names from `FMotionBricksBridgeRuntimeConfig`:

| Purpose | Default |
|---|---|
| Command mapping | `Local\MotionBricks_Command` |
| Frame mapping | `Local\MotionBricks_Frames` |
| Command mutex | `Local\MotionBricks_CommandMutex` |
| Frame mutex | `Local\MotionBricks_FrameMutex` |
| Command event | `Local\MotionBricks_CommandReady` |
| Frame event | `Local\MotionBricks_FrameReady` |

Default sizes and timing:

| Setting | Default |
|---|---:|
| `CommandMemorySizeBytes` | `4096` |
| `FrameMemorySizeBytes` | `1048576` |
| `FramePacketCapacity` | `32` |
| `SharedMemoryTimeoutSeconds` | `0.005` |
| `StalePacketTimeoutSeconds` | `0.5` |
| `bDropStalePackets` | `true` |
| `bLaunchProcess` | `false` |

`bLaunchProcess` is reserved in UE and currently only logs a warning.

## Magic And Version

Namespace: `MotionBricksSharedMemoryBridgeProtocol`.

| Constant | Value | Meaning |
|---|---:|---|
| `ProtocolVersion` | `1` | Shared-memory protocol version |
| `CommandMagic` | `0x4D42434D` | ASCII `MBCM` as little-endian `uint32` |
| `FrameMagic` | `0x4D424652` | ASCII `MBFR` as little-endian `uint32` |

The Python writer/reader must reject or reinitialize buffers whose `Header.Magic` or `Header.Version` do not match.

## Binary Encoding

- Platform: Windows shared memory.
- Layout: native UE/MSVC C++ struct layout from the header, not packed JSON.
- Endianness: little-endian.
- Strings: fixed UTF-8 `ANSICHAR` arrays, NUL-terminated when shorter than the fixed field.
- Time fields named `Seconds`: `double`, UE `FPlatformTime::Seconds()` style.
- Vectors: `float[3]`, MuJoCo basis, meters where positions are used.
- Root quaternion: `float[4]` in `wxyz`.

Python should implement these with `ctypes.LittleEndianStructure` or equivalent and native-compatible alignment. Do not use the previous offset table from the single-region draft.

## Limits

| Constant | Value |
|---|---:|
| `MaxModeUtf8Bytes` | `64` |
| `MaxNameUtf8Bytes` | `64` |
| `MaxAllowedPredNumTokens` | `32` |
| `MaxQposElements` | `64` |
| `MaxSkeletonBones` | `64` |
| `MaxDofNames` | `64` |
| `MaxFramesPerPacket` | `4` |

Current G1 contract carried inside those maxima:

- Expected qpos length: `36`.
- Expected skeleton bones: `34`.
- Expected hinge DOFs after free root: `29`.
- Source rate default: `30 Hz`.

## Shared Header

Every command and frame mapping begins with `FSharedMemoryHeader`:

```cpp
struct FSharedMemoryHeader
{
    uint32 Magic;
    uint32 Version;
    uint32 HeaderSizeBytes;
    uint32 TotalSizeBytes;
    uint32 Sequence;
    uint32 Flags;
    double LastWriteSeconds;
};
```

`Sequence` is a monotonic writer counter for the buffer. `LastWriteSeconds` is written from the writer process clock and is used only as local status/debug data, not cross-process clock sync.

## Command Mapping

UE owns and writes `FCommandBuffer` into `CommandMemoryName`.

```cpp
struct FCommandBuffer
{
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
};
```

UE write behavior:

1. Validate `FMotionBricksLiveCommand`.
2. Wait for `CommandMutexName` up to `SharedMemoryTimeoutSeconds`.
3. Reinitialize the command buffer if magic/version mismatch.
4. Increment `Header.Sequence`.
5. Write `Header.LastWriteSeconds`.
6. Copy command payload.
7. Release the command mutex.
8. Signal `CommandEventName`.

Python read behavior:

- Wait or poll `CommandEventName`.
- Lock `CommandMutexName` before reading.
- Treat a changed `Header.Sequence` as a new command.
- If no new command arrives, Python should keep its last valid command briefly, then fall back to idle.

Command validation mirrors the UDP live protocol:

- `MovementDirectionMj` and `FacingDirectionMj` must not contain NaN.
- Direction Z components should be horizontal, approximately `0`.
- Either `ModeUtf8` is non-empty or `ModeIndex != -1`.
- `RequestFrames >= 0`.

## Frame Mapping

Python owns and writes `FFrameRingBuffer` plus `Capacity` `FFramePacketSlot` values into `FrameMemoryName`.

```cpp
struct FFrameRingBuffer
{
    FSharedMemoryHeader Header;
    int32 Capacity;
    int32 SlotSizeBytes;
    uint32 WriteSequence;
    uint32 LatestSlotIndex;
    int32 ExpectedQposLength;
    int32 ExpectedBoneCount;
    int32 MaxFramesPerPacket;
};
```

UE initializes or validates this header with:

- `Header.Magic = FrameMagic`.
- `Header.Version = ProtocolVersion`.
- `Header.HeaderSizeBytes = sizeof(FFrameRingBuffer)`.
- `Header.TotalSizeBytes = FrameMemorySizeBytes`.
- `Capacity = FramePacketCapacity`.
- `SlotSizeBytes = sizeof(FFramePacketSlot)`.
- `ExpectedQposLength = RuntimeConfig.ExpectedQposLength`.
- `ExpectedBoneCount = RuntimeConfig.ExpectedBoneCount`.
- `MaxFramesPerPacket = 4`.

Python should preserve those values unless it is the first creator and must initialize an empty mapping to the same values.

## Frame Packet Slot

```cpp
struct FFramePacketSlot
{
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
};
```

```cpp
struct FFixedFrame
{
    double TimeSeconds;
    int32 QposCount;
    float Qpos[64];
    float RootXyzM[3];
    float RootQuatWxyz[4];
    int32 BoneGlobalPositionCount;
    float BoneGlobalPositionsM[64][3];
};
```

For G1, Python writes:

- `SchemaVersionUtf8 = "motionbricks_ue_live.v1"`.
- `BoneNameCount = 34`.
- `ParentCount = 34`.
- `DofNameCount = 29`.
- `FrameCount` from `1` to `4`; current normal live behavior is `1`.
- Each frame `QposCount = 36`.
- Each frame `BoneGlobalPositionCount = 34`.

UE converts each slot to `FMotionBricksLiveFramePacket` and then applies the existing frame validation in the live client path.

## Ring Behavior

Python writer procedure for each packet:

1. Lock `FrameMutexName`.
2. Increment `FFrameRingBuffer::WriteSequence`.
3. Compute `SlotIndex = (WriteSequence - 1) % Capacity`.
4. Fill `FFramePacketSlot` at `sizeof(FFrameRingBuffer) + SlotIndex * sizeof(FFramePacketSlot)`.
5. Set `SlotSequence = WriteSequence`.
6. Set `LatestSlotIndex = SlotIndex`.
7. Update `FFrameRingBuffer::Header.Sequence` and `Header.LastWriteSeconds`.
8. Release `FrameMutexName`.
9. Signal `FrameEventName`.

UE reader behavior:

1. Polls `FrameEventName` with zero timeout in `Tick`.
2. Locks `FrameMutexName` up to `SharedMemoryTimeoutSeconds`.
3. Reads `WriteSequence`.
4. If `WriteSequence <= LastReadFrameSequence`, no new packets are available.
5. If more packets are available than ring capacity, UE counts the overwritten packets as dropped and starts from the oldest still available sequence.
6. For each sequence, UE calculates `SlotIndex = (Sequence - 1) % Capacity`.
7. If `SlotSequence != Sequence`, UE drops that packet.
8. If stale dropping is enabled and the frame buffer is stale, UE drops that packet.
9. Otherwise UE converts the slot to a live frame packet.
10. UE sets `LastReadFrameSequence = WriteSequence`.

`DroppedPacketCount` is internal to the backend. `UMotionBricksLiveClientComponent` owns the public `PacketsDropped` compatibility counter.

## Mutex And Event Strategy

The canonical implementation uses Windows mutexes for both read/write consistency and buffer initialization:

- Command writes are protected by `CommandMutexName`.
- Command reads on Python side must also use `CommandMutexName`.
- Frame writes on Python side must use `FrameMutexName`.
- Frame reads on UE side use `FrameMutexName`.
- `CommandEventName` is auto-reset and signaled after UE writes a command.
- `FrameEventName` is auto-reset and signaled after Python writes a frame packet.

There is no seqlock in the canonical UE implementation. Do not implement the older odd/even write sequence protocol for this backend.

## Stale And Drop Logic

UE stale status is based on `LastFrameReceivePlatformSeconds`, a local UE timestamp updated only when a frame slot converts successfully.

`IsFrameBufferStale(NowSeconds)` is true when:

- the frame buffer has been initialized,
- `StalePacketTimeoutSeconds > 0`,
- `LastFrameReceivePlatformSeconds > 0`,
- `NowSeconds - LastFrameReceivePlatformSeconds > StalePacketTimeoutSeconds`.

If `bDropStalePackets` is true and the buffer is stale, UE drops candidate packets instead of converting them. If false, UE may still consume old packets.

UE also drops packets when:

- frame mutex wait times out,
- frame ring magic/version/capacity is invalid,
- ring overwrite means unread sequences are no longer available,
- a slot's `SlotSequence` does not match the requested sequence,
- slot conversion fails,
- converted frame data fails later live-frame validation.

Python stale command recommendation:

- If `Header.Sequence` has not changed for `max(0.25s, 3 / SendRateHz)`, continue last valid command.
- If it has not changed for `1.0s`, fall back to idle:
  - mode `idle`
  - movement `[0, 0, 0]`
  - facing `[1, 0, 0]`

## Startup Order

Either side may create the named mappings first. In practice:

1. UE starts `FMotionBricksSharedMemoryBridgeBackend`.
2. UE creates/open mappings, mutexes, and events.
3. UE initializes command and frame headers if they are missing or stale.
4. Python opens the same names and verifies magic/version.
5. Python waits for commands and publishes frame slots into the ring.

Python must not assume the frame mapping is a single packet; it is `FFrameRingBuffer` followed by `Capacity` slots.

## Python Startup Commands

Shared-memory smoke test with UE default names:

```powershell
Set-Location ${PROJECT_ROOT}
conda activate motionbricks
python -m pip install pywin32
python scripts\ue_bridge_shared_memory_runner.py --smoke_shared_memory --max_steps 1 --command_map "Local\MotionBricks_Command" --frame_map "Local\MotionBricks_Frames" --command_mutex "Local\MotionBricks_CommandMutex" --frame_mutex "Local\MotionBricks_FrameMutex" --command_event "Local\MotionBricks_CommandReady" --frame_event "Local\MotionBricks_FrameReady" --frame_capacity 32 --command_memory_size 4096 --frame_memory_size 1048576
```

Shared-memory live runner matching UE defaults:

```powershell
Set-Location ${PROJECT_ROOT}
conda activate motionbricks
python -m pip install pywin32
python scripts\ue_bridge_shared_memory_runner.py --command_map "Local\MotionBricks_Command" --frame_map "Local\MotionBricks_Frames" --command_mutex "Local\MotionBricks_CommandMutex" --frame_mutex "Local\MotionBricks_FrameMutex" --command_event "Local\MotionBricks_CommandReady" --frame_event "Local\MotionBricks_FrameReady" --frame_capacity 32 --command_memory_size 4096 --frame_memory_size 1048576 --tick_hz 30 --use_qpos 1 --planner default --clips G1
```
