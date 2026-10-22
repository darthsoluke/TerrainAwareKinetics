"""Windows shared-memory runner for the MotionBricks UE bridge.

The binary protocol in this file mirrors the UE C++ ground truth:

- MotionBricksSharedMemoryBridgeBackend.h
- MotionBricksSharedMemoryBridgeBackend.cpp
- MotionBricksBridgeBackendTypes.h

It intentionally keeps the UDP server untouched and reuses
MotionBricksLiveServer.step(command, step_idx) for MotionBricks generation.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ue_bridge_live_server import (  # noqa: E402
    DEFAULT_IDLE_COMMAND,
    QPOS_LEN,
    QUAT_NORM_TOL,
    MotionBricksLiveServer,
    _build_arg_parser as _build_live_arg_parser,
    _finalize_args as _finalize_live_args,
)


PROTOCOL_VERSION = 1
COMMAND_MAGIC = 0x4D42434D  # MBCM
FRAME_MAGIC = 0x4D424652  # MBFR

MAX_MODE_UTF8_BYTES = 64
MAX_NAME_UTF8_BYTES = 64
MAX_ALLOWED_PRED_NUM_TOKENS = 32
MAX_QPOS_ELEMENTS = 64
MAX_SKELETON_BONES = 64
MAX_DOF_NAMES = 64
MAX_FRAMES_PER_PACKET = 4

EXPECTED_QPOS_LENGTH = 36
EXPECTED_BONE_COUNT = 34
DEFAULT_FRAME_CAPACITY = 32

COMMAND_MAP_NAME = r"Local\MotionBricks_Command"
FRAME_MAP_NAME = r"Local\MotionBricks_Frames"
COMMAND_MUTEX_NAME = r"Local\MotionBricks_CommandMutex"
FRAME_MUTEX_NAME = r"Local\MotionBricks_FrameMutex"
COMMAND_EVENT_NAME = r"Local\MotionBricks_CommandReady"
FRAME_EVENT_NAME = r"Local\MotionBricks_FrameReady"

COMMAND_MEMORY_SIZE_BYTES = 4096
FRAME_MEMORY_SIZE_BYTES = 1048576
SHARED_MEMORY_TIMEOUT_SECONDS = 0.005

PAGE_READWRITE = 0x04
FILE_MAP_ALL_ACCESS = 0xF001F
SYNCHRONIZE = 0x00100000
MUTEX_ALL_ACCESS = 0x001F0001
EVENT_ALL_ACCESS = 0x001F0003
WAIT_OBJECT_0 = 0x00000000
WAIT_ABANDONED = 0x00000080
WAIT_TIMEOUT = 0x00000102
INFINITE = 0xFFFFFFFF
ERROR_FILE_NOT_FOUND = 2
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class FSharedMemoryHeader(ctypes.Structure):
    _fields_ = [
        ("Magic", ctypes.c_uint32),
        ("Version", ctypes.c_uint32),
        ("HeaderSizeBytes", ctypes.c_uint32),
        ("TotalSizeBytes", ctypes.c_uint32),
        ("Sequence", ctypes.c_uint32),
        ("Flags", ctypes.c_uint32),
        ("LastWriteSeconds", ctypes.c_double),
    ]


class FCommandBuffer(ctypes.Structure):
    _fields_ = [
        ("Header", FSharedMemoryHeader),
        ("CommandSequenceId", ctypes.c_int32),
        ("ClientTimeSeconds", ctypes.c_double),
        ("MovementDirectionMj", ctypes.c_float * 3),
        ("FacingDirectionMj", ctypes.c_float * 3),
        ("ModeUtf8", ctypes.c_char * MAX_MODE_UTF8_BYTES),
        ("ModeIndex", ctypes.c_int32),
        ("RequestFrames", ctypes.c_int32),
        ("AllowedPredNumTokensCount", ctypes.c_int32),
        ("AllowedPredNumTokens", ctypes.c_int32 * MAX_ALLOWED_PRED_NUM_TOKENS),
    ]


class FFixedFrame(ctypes.Structure):
    _fields_ = [
        ("TimeSeconds", ctypes.c_double),
        ("QposCount", ctypes.c_int32),
        ("Qpos", ctypes.c_float * MAX_QPOS_ELEMENTS),
        ("RootXyzM", ctypes.c_float * 3),
        ("RootQuatWxyz", ctypes.c_float * 4),
        ("BoneGlobalPositionCount", ctypes.c_int32),
        ("BoneGlobalPositionsM", (ctypes.c_float * 3) * MAX_SKELETON_BONES),
    ]


class FFramePacketSlot(ctypes.Structure):
    _fields_ = [
        ("SlotSequence", ctypes.c_uint32),
        ("PacketFlags", ctypes.c_uint32),
        ("PacketSequenceId", ctypes.c_int32),
        ("ServerTimeSeconds", ctypes.c_double),
        ("SourceHz", ctypes.c_float),
        ("SchemaVersionUtf8", ctypes.c_char * MAX_NAME_UTF8_BYTES),
        ("BoneNameCount", ctypes.c_int32),
        ("BoneNamesUtf8", (ctypes.c_char * MAX_NAME_UTF8_BYTES) * MAX_SKELETON_BONES),
        ("ParentCount", ctypes.c_int32),
        ("Parents", ctypes.c_int32 * MAX_SKELETON_BONES),
        ("DofNameCount", ctypes.c_int32),
        ("DofNamesUtf8", (ctypes.c_char * MAX_NAME_UTF8_BYTES) * MAX_DOF_NAMES),
        ("FrameCount", ctypes.c_int32),
        ("Frames", FFixedFrame * MAX_FRAMES_PER_PACKET),
    ]


class FFrameRingBuffer(ctypes.Structure):
    _fields_ = [
        ("Header", FSharedMemoryHeader),
        ("Capacity", ctypes.c_int32),
        ("SlotSizeBytes", ctypes.c_int32),
        ("WriteSequence", ctypes.c_uint32),
        ("LatestSlotIndex", ctypes.c_uint32),
        ("ExpectedQposLength", ctypes.c_int32),
        ("ExpectedBoneCount", ctypes.c_int32),
        ("MaxFramesPerPacket", ctypes.c_int32),
    ]


def _assert_ue_compatible_sizes() -> None:
    expected_sizes = {
        "FSharedMemoryHeader": 32,
        "FCommandBuffer": 280,
        "FFixedFrame": 1072,
        "FFramePacketSlot": 12848,
        "FFrameRingBuffer": 64,
    }
    actual = {
        "FSharedMemoryHeader": ctypes.sizeof(FSharedMemoryHeader),
        "FCommandBuffer": ctypes.sizeof(FCommandBuffer),
        "FFixedFrame": ctypes.sizeof(FFixedFrame),
        "FFramePacketSlot": ctypes.sizeof(FFramePacketSlot),
        "FFrameRingBuffer": ctypes.sizeof(FFrameRingBuffer),
    }
    mismatches = [f"{name}: expected {expected_sizes[name]}, got {size}" for name, size in actual.items() if size != expected_sizes[name]]
    if mismatches:
        raise RuntimeError("ctypes layout does not match UE C++ default packing: " + "; ".join(mismatches))


def required_frame_memory_size(capacity: int) -> int:
    return ctypes.sizeof(FFrameRingBuffer) + int(capacity) * ctypes.sizeof(FFramePacketSlot)


def _encode_fixed_utf8(value: str, size: int) -> bytes:
    encoded = str(value).encode("utf-8", errors="replace")[: max(size - 1, 0)]
    return encoded + b"\0" * (size - len(encoded))


def _decode_fixed_utf8(value: bytes | ctypes.Array[Any]) -> str:
    raw = bytes(value)
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def _fixed_utf8_array(value: str, size: int) -> ctypes.Array[Any]:
    array_type = ctypes.c_char * size
    return array_type.from_buffer_copy(_encode_fixed_utf8(value, size))


class Win32:
    def __init__(self) -> None:
        if platform.system() != "Windows":
            raise RuntimeError("MotionBricks shared-memory runner is Windows-only.")

        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        self.kernel32.CreateFileMappingW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_wchar_p,
        ]
        self.kernel32.CreateFileMappingW.restype = ctypes.c_void_p
        self.kernel32.OpenFileMappingW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
        self.kernel32.OpenFileMappingW.restype = ctypes.c_void_p
        self.kernel32.MapViewOfFile.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_size_t]
        self.kernel32.MapViewOfFile.restype = ctypes.c_void_p
        self.kernel32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
        self.kernel32.UnmapViewOfFile.restype = ctypes.c_int
        self.kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self.kernel32.CloseHandle.restype = ctypes.c_int

        self.kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
        self.kernel32.CreateMutexW.restype = ctypes.c_void_p
        self.kernel32.OpenMutexW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
        self.kernel32.OpenMutexW.restype = ctypes.c_void_p
        self.kernel32.ReleaseMutex.argtypes = [ctypes.c_void_p]
        self.kernel32.ReleaseMutex.restype = ctypes.c_int

        self.kernel32.CreateEventW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p]
        self.kernel32.CreateEventW.restype = ctypes.c_void_p
        self.kernel32.OpenEventW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
        self.kernel32.OpenEventW.restype = ctypes.c_void_p
        self.kernel32.SetEvent.argtypes = [ctypes.c_void_p]
        self.kernel32.SetEvent.restype = ctypes.c_int

        self.kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.kernel32.WaitForSingleObject.restype = ctypes.c_uint32

    def last_error(self) -> int:
        return ctypes.get_last_error()

    def create_or_open_mapping(self, name: str, size: int) -> ctypes.c_void_p:
        handle = self.kernel32.OpenFileMappingW(FILE_MAP_ALL_ACCESS, False, name)
        if not handle:
            if self.last_error() != ERROR_FILE_NOT_FOUND:
                raise OSError(self.last_error(), f"OpenFileMappingW failed for {name}")
            handle = self.kernel32.CreateFileMappingW(INVALID_HANDLE_VALUE, None, PAGE_READWRITE, 0, int(size), name)
        if not handle:
            raise OSError(self.last_error(), f"CreateFileMappingW failed for {name}")
        return handle

    def create_or_open_mutex(self, name: str) -> ctypes.c_void_p:
        handle = self.kernel32.OpenMutexW(MUTEX_ALL_ACCESS, False, name)
        if not handle:
            if self.last_error() != ERROR_FILE_NOT_FOUND:
                raise OSError(self.last_error(), f"OpenMutexW failed for {name}")
            handle = self.kernel32.CreateMutexW(None, False, name)
        if not handle:
            raise OSError(self.last_error(), f"CreateMutexW failed for {name}")
        return handle

    def create_or_open_event(self, name: str) -> ctypes.c_void_p:
        handle = self.kernel32.OpenEventW(EVENT_ALL_ACCESS | SYNCHRONIZE, False, name)
        if not handle:
            if self.last_error() != ERROR_FILE_NOT_FOUND:
                raise OSError(self.last_error(), f"OpenEventW failed for {name}")
            handle = self.kernel32.CreateEventW(None, False, False, name)
        if not handle:
            raise OSError(self.last_error(), f"CreateEventW failed for {name}")
        return handle

    def map_view(self, handle: ctypes.c_void_p, size: int) -> int:
        ptr = self.kernel32.MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, int(size))
        if not ptr:
            raise OSError(self.last_error(), "MapViewOfFile failed")
        return int(ptr)

    def close_handle(self, handle: ctypes.c_void_p | None) -> None:
        if handle:
            self.kernel32.CloseHandle(handle)

    def unmap_view(self, ptr: int | None) -> None:
        if ptr:
            self.kernel32.UnmapViewOfFile(ctypes.c_void_p(ptr))

    def wait_mutex(self, handle: ctypes.c_void_p, timeout_seconds: float, context: str) -> None:
        timeout_ms = max(0, int(round(float(timeout_seconds) * 1000.0)))
        result = self.kernel32.WaitForSingleObject(handle, timeout_ms)
        if result not in (WAIT_OBJECT_0, WAIT_ABANDONED):
            if result == WAIT_TIMEOUT:
                raise TimeoutError(f"Timed out waiting for mutex while {context}.")
            raise OSError(self.last_error(), f"WaitForSingleObject failed while {context}")

    def release_mutex(self, handle: ctypes.c_void_p) -> None:
        if not self.kernel32.ReleaseMutex(handle):
            raise OSError(self.last_error(), "ReleaseMutex failed")

    def set_event(self, handle: ctypes.c_void_p) -> None:
        if not self.kernel32.SetEvent(handle):
            raise OSError(self.last_error(), "SetEvent failed")

    def wait_event_zero(self, handle: ctypes.c_void_p) -> None:
        self.kernel32.WaitForSingleObject(handle, 0)


class SharedMemoryContext:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.win32 = Win32()
        self.command_mapping = None
        self.frame_mapping = None
        self.command_mutex = None
        self.frame_mutex = None
        self.command_event = None
        self.frame_event = None
        self.command_ptr: int | None = None
        self.frame_ptr: int | None = None

    def __enter__(self) -> "SharedMemoryContext":
        self.command_mutex = self.win32.create_or_open_mutex(self.args.command_mutex)
        self.frame_mutex = self.win32.create_or_open_mutex(self.args.frame_mutex)
        self.command_event = self.win32.create_or_open_event(self.args.command_event)
        self.frame_event = self.win32.create_or_open_event(self.args.frame_event)
        self.command_mapping = self.win32.create_or_open_mapping(self.args.command_map, self.args.command_memory_size)
        self.frame_mapping = self.win32.create_or_open_mapping(self.args.frame_map, self.args.frame_memory_size)
        self.command_ptr = self.win32.map_view(self.command_mapping, self.args.command_memory_size)
        self.frame_ptr = self.win32.map_view(self.frame_mapping, self.args.frame_memory_size)
        self.initialize_command_buffer_if_needed()
        self.initialize_frame_buffer_if_needed()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.win32.unmap_view(self.command_ptr)
        self.win32.unmap_view(self.frame_ptr)
        self.win32.close_handle(self.command_mapping)
        self.win32.close_handle(self.frame_mapping)
        self.win32.close_handle(self.command_mutex)
        self.win32.close_handle(self.frame_mutex)
        self.win32.close_handle(self.command_event)
        self.win32.close_handle(self.frame_event)

    @property
    def command_buffer(self) -> FCommandBuffer:
        if self.command_ptr is None:
            raise RuntimeError("Command memory is not mapped.")
        return FCommandBuffer.from_address(self.command_ptr)

    @property
    def frame_ring(self) -> FFrameRingBuffer:
        if self.frame_ptr is None:
            raise RuntimeError("Frame memory is not mapped.")
        return FFrameRingBuffer.from_address(self.frame_ptr)

    def frame_slot(self, slot_index: int) -> FFramePacketSlot:
        if self.frame_ptr is None:
            raise RuntimeError("Frame memory is not mapped.")
        offset = ctypes.sizeof(FFrameRingBuffer) + int(slot_index) * ctypes.sizeof(FFramePacketSlot)
        return FFramePacketSlot.from_address(self.frame_ptr + offset)

    def initialize_command_buffer_if_needed(self) -> None:
        self.win32.wait_mutex(self.command_mutex, self.args.shared_memory_timeout, "initializing command buffer")
        try:
            command = self.command_buffer
            if command.Header.Magic != COMMAND_MAGIC or command.Header.Version != PROTOCOL_VERSION:
                ctypes.memset(self.command_ptr, 0, self.args.command_memory_size)
                command.Header.Magic = COMMAND_MAGIC
                command.Header.Version = PROTOCOL_VERSION
                command.Header.HeaderSizeBytes = ctypes.sizeof(FCommandBuffer)
                command.Header.TotalSizeBytes = int(self.args.command_memory_size)
                command.Header.LastWriteSeconds = time.perf_counter()
                command.ModeUtf8 = _encode_fixed_utf8(self.args.idle_mode_name, MAX_MODE_UTF8_BYTES)
                command.MovementDirectionMj[:] = (1.0, 0.0, 0.0)
                command.FacingDirectionMj[:] = (1.0, 0.0, 0.0)
                command.ModeIndex = -1
        finally:
            self.win32.release_mutex(self.command_mutex)

    def initialize_frame_buffer_if_needed(self) -> None:
        required_size = required_frame_memory_size(self.args.frame_capacity)
        if self.args.frame_memory_size < required_size:
            raise ValueError(
                f"Frame memory size {self.args.frame_memory_size} is too small for capacity "
                f"{self.args.frame_capacity}; requires {required_size} bytes."
            )

        self.win32.wait_mutex(self.frame_mutex, self.args.shared_memory_timeout, "initializing frame buffer")
        try:
            ring = self.frame_ring
            if (
                ring.Header.Magic != FRAME_MAGIC
                or ring.Header.Version != PROTOCOL_VERSION
                or ring.Header.TotalSizeBytes < required_size
                or ring.Capacity != self.args.frame_capacity
                or ring.SlotSizeBytes != ctypes.sizeof(FFramePacketSlot)
            ):
                ctypes.memset(self.frame_ptr, 0, self.args.frame_memory_size)
                ring.Header.Magic = FRAME_MAGIC
                ring.Header.Version = PROTOCOL_VERSION
                ring.Header.HeaderSizeBytes = ctypes.sizeof(FFrameRingBuffer)
                ring.Header.TotalSizeBytes = int(self.args.frame_memory_size)
                ring.Header.LastWriteSeconds = time.perf_counter()
                ring.Capacity = int(self.args.frame_capacity)
                ring.SlotSizeBytes = ctypes.sizeof(FFramePacketSlot)
                ring.ExpectedQposLength = EXPECTED_QPOS_LENGTH
                ring.ExpectedBoneCount = EXPECTED_BONE_COUNT
                ring.MaxFramesPerPacket = MAX_FRAMES_PER_PACKET
        finally:
            self.win32.release_mutex(self.frame_mutex)


def _read_fixed_float_array(values: Any, count: int) -> list[float]:
    return [float(values[i]) for i in range(count)]


def _read_command_from_buffer(command_buffer: FCommandBuffer) -> tuple[int, dict[str, Any] | None]:
    if command_buffer.Header.Magic != COMMAND_MAGIC or command_buffer.Header.Version != PROTOCOL_VERSION:
        return 0, None

    command: dict[str, Any] = {
        "type": "control",
        "sequence_id": int(command_buffer.CommandSequenceId),
        "client_time_s": float(command_buffer.ClientTimeSeconds),
        "movement_direction_mj": _read_fixed_float_array(command_buffer.MovementDirectionMj, 3),
        "facing_direction_mj": _read_fixed_float_array(command_buffer.FacingDirectionMj, 3),
        "request_frames": int(command_buffer.RequestFrames),
    }

    mode = _decode_fixed_utf8(command_buffer.ModeUtf8)
    if mode:
        command["mode"] = mode
    elif int(command_buffer.ModeIndex) != -1:
        command["mode"] = int(command_buffer.ModeIndex)
    else:
        command["mode"] = "idle"

    if int(command_buffer.ModeIndex) != -1:
        command["mode_index"] = int(command_buffer.ModeIndex)

    allowed_count = max(0, min(int(command_buffer.AllowedPredNumTokensCount), MAX_ALLOWED_PRED_NUM_TOKENS))
    if allowed_count > 0:
        command["allowed_pred_num_tokens"] = [int(command_buffer.AllowedPredNumTokens[i]) for i in range(allowed_count)]

    return int(command_buffer.Header.Sequence), command


def _write_smoke_command(ctx: SharedMemoryContext, sequence: int = 1) -> None:
    ctx.win32.wait_mutex(ctx.command_mutex, ctx.args.shared_memory_timeout, "writing smoke command")
    try:
        command = ctx.command_buffer
        if command.Header.Magic != COMMAND_MAGIC or command.Header.Version != PROTOCOL_VERSION:
            ctx.initialize_command_buffer_if_needed()
            command = ctx.command_buffer

        command.Header.Sequence = int(sequence)
        command.Header.LastWriteSeconds = time.perf_counter()
        command.CommandSequenceId = int(sequence)
        command.ClientTimeSeconds = 0.0
        command.MovementDirectionMj[:] = (0.0, 0.0, 0.0)
        command.FacingDirectionMj[:] = (1.0, 0.0, 0.0)
        command.ModeUtf8 = _encode_fixed_utf8("idle", MAX_MODE_UTF8_BYTES)
        command.ModeIndex = -1
        command.RequestFrames = 0
        command.AllowedPredNumTokensCount = 0
        for idx in range(MAX_ALLOWED_PRED_NUM_TOKENS):
            command.AllowedPredNumTokens[idx] = 0
    finally:
        ctx.win32.release_mutex(ctx.command_mutex)
    ctx.win32.set_event(ctx.command_event)


def _packet_skeleton(packet: dict[str, Any]) -> tuple[list[str], list[int], list[str]]:
    skeleton = packet.get("skeleton", {})
    bone_names = list(skeleton.get("bone_names") or packet.get("bone_names") or [])
    parents = [int(x) for x in list(skeleton.get("parents") or packet.get("parents") or [])]
    dof_names = list(skeleton.get("dof_names") or packet.get("dof_names") or [])
    schema_header = packet.get("schema_header", {})
    if not dof_names and isinstance(schema_header, dict):
        dof_names = list(schema_header.get("dof_names") or [])
    return bone_names[:MAX_SKELETON_BONES], parents[:MAX_SKELETON_BONES], dof_names[:MAX_DOF_NAMES]


def _fill_name_array(dest: Any, values: list[str], max_count: int, item_size: int) -> int:
    count = min(len(values), max_count)
    for idx in range(max_count):
        dest[idx] = _fixed_utf8_array("", item_size)
    for idx in range(count):
        dest[idx] = _fixed_utf8_array(values[idx], item_size)
    return count


def _identity_rotations(count: int) -> np.ndarray:
    rotations = np.zeros((count, 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    return rotations


def _fill_fixed_frame(frame: FFixedFrame, packet_frame: dict[str, Any]) -> None:
    qpos = np.asarray(packet_frame.get("qpos", []), dtype=np.float32).reshape(-1)
    if qpos.shape != (EXPECTED_QPOS_LENGTH,):
        raise ValueError(f"Expected qpos_len={EXPECTED_QPOS_LENGTH}, got {qpos.shape}.")

    root_xyz = np.asarray(packet_frame.get("root_xyz_m", qpos[:3]), dtype=np.float32).reshape(-1)
    root_quat = np.asarray(packet_frame.get("root_quat_wxyz", qpos[3:7]), dtype=np.float32).reshape(-1)
    bone_positions = np.asarray(packet_frame.get("bone_global_positions_m", []), dtype=np.float32)
    if root_xyz.shape != (3,):
        raise ValueError(f"Expected root_xyz_m shape (3,), got {root_xyz.shape}.")
    if root_quat.shape != (4,):
        raise ValueError(f"Expected root_quat_wxyz shape (4,), got {root_quat.shape}.")
    if bone_positions.shape != (EXPECTED_BONE_COUNT, 3):
        raise ValueError(f"Expected bone_global_positions_m shape ({EXPECTED_BONE_COUNT}, 3), got {bone_positions.shape}.")
    if abs(float(np.linalg.norm(root_quat)) - 1.0) > QUAT_NORM_TOL:
        raise ValueError(f"Root quaternion norm must be 1, got {float(np.linalg.norm(root_quat)):.6f}.")

    frame.TimeSeconds = float(packet_frame.get("time_s", packet_frame.get("time", 0.0)))
    frame.QposCount = EXPECTED_QPOS_LENGTH
    for idx in range(MAX_QPOS_ELEMENTS):
        frame.Qpos[idx] = float(qpos[idx]) if idx < EXPECTED_QPOS_LENGTH else 0.0
    for idx in range(3):
        frame.RootXyzM[idx] = float(root_xyz[idx])
    for idx in range(4):
        frame.RootQuatWxyz[idx] = float(root_quat[idx])
    frame.BoneGlobalPositionCount = EXPECTED_BONE_COUNT
    for bone_idx in range(MAX_SKELETON_BONES):
        for axis_idx in range(3):
            frame.BoneGlobalPositionsM[bone_idx][axis_idx] = float(bone_positions[bone_idx, axis_idx]) if bone_idx < EXPECTED_BONE_COUNT else 0.0

    # The UE ground-truth FFixedFrame currently has no rotation storage. When
    # MotionBricks emits rotations we intentionally do not serialize them here;
    # consumers should treat shared-memory v1 as qpos + position debug data.
    _identity_rotations(EXPECTED_BONE_COUNT)


def _write_packet_to_ring(ctx: SharedMemoryContext, packet: dict[str, Any]) -> tuple[int, int]:
    ctx.win32.wait_mutex(ctx.frame_mutex, ctx.args.shared_memory_timeout, "writing frame packet")
    try:
        ring = ctx.frame_ring
        if ring.Header.Magic != FRAME_MAGIC or ring.Header.Version != PROTOCOL_VERSION or ring.Capacity <= 0:
            ctx.initialize_frame_buffer_if_needed()
            ring = ctx.frame_ring

        write_sequence = int(ring.WriteSequence) + 1
        slot_index = (write_sequence - 1) % int(ring.Capacity)
        slot = ctx.frame_slot(slot_index)
        ctypes.memset(ctypes.addressof(slot), 0, ctypes.sizeof(FFramePacketSlot))

        slot.SlotSequence = write_sequence
        slot.PacketFlags = 0
        slot.PacketSequenceId = int(packet.get("sequence_id", packet.get("source_step", write_sequence)))
        slot.ServerTimeSeconds = float(packet.get("server_time_s", packet.get("time", 0.0)))
        slot.SourceHz = float(packet.get("source_hz", packet.get("sample_rate_hz", 30.0)))
        slot.SchemaVersionUtf8 = _encode_fixed_utf8(str(packet.get("schema_version", "motionbricks_ue_live.v1")), MAX_NAME_UTF8_BYTES)

        bone_names, parents, dof_names = _packet_skeleton(packet)
        slot.BoneNameCount = _fill_name_array(slot.BoneNamesUtf8, bone_names, MAX_SKELETON_BONES, MAX_NAME_UTF8_BYTES)
        slot.ParentCount = min(len(parents), MAX_SKELETON_BONES)
        for idx in range(MAX_SKELETON_BONES):
            slot.Parents[idx] = parents[idx] if idx < slot.ParentCount else 0
        slot.DofNameCount = _fill_name_array(slot.DofNamesUtf8, dof_names, MAX_DOF_NAMES, MAX_NAME_UTF8_BYTES)

        source_frames = packet.get("frames")
        if isinstance(source_frames, list) and source_frames:
            frames = source_frames[:MAX_FRAMES_PER_PACKET]
        else:
            frames = [packet]
        slot.FrameCount = len(frames)
        for frame_idx, packet_frame in enumerate(frames):
            _fill_fixed_frame(slot.Frames[frame_idx], packet_frame)

        ring.WriteSequence = write_sequence
        ring.LatestSlotIndex = slot_index
        ring.Header.Sequence = write_sequence
        ring.Header.LastWriteSeconds = time.perf_counter()
        return write_sequence, slot_index
    finally:
        ctx.win32.release_mutex(ctx.frame_mutex)


def _read_latest_slot(ctx: SharedMemoryContext) -> FFramePacketSlot:
    ctx.win32.wait_mutex(ctx.frame_mutex, ctx.args.shared_memory_timeout, "reading latest frame packet")
    try:
        ring = ctx.frame_ring
        if ring.Header.Magic != FRAME_MAGIC or ring.Header.Version != PROTOCOL_VERSION or ring.Capacity <= 0:
            raise RuntimeError("Frame ring buffer is not initialized.")
        slot_index = int(ring.LatestSlotIndex)
        slot = ctx.frame_slot(slot_index)
        copied = FFramePacketSlot()
        ctypes.memmove(ctypes.addressof(copied), ctypes.addressof(slot), ctypes.sizeof(FFramePacketSlot))
        return copied
    finally:
        ctx.win32.release_mutex(ctx.frame_mutex)


def _slot_validation_summary(slot: FFramePacketSlot) -> dict[str, Any]:
    if int(slot.FrameCount) <= 0:
        raise RuntimeError("Latest frame slot contains no frames.")
    frame = slot.Frames[0]
    qpos = np.asarray([float(frame.Qpos[i]) for i in range(int(frame.QposCount))], dtype=np.float64)
    root_quat = np.asarray([float(frame.RootQuatWxyz[i]) for i in range(4)], dtype=np.float64)
    bone_count = int(frame.BoneGlobalPositionCount)
    bone_positions = np.asarray(
        [[float(frame.BoneGlobalPositionsM[i][axis]) for axis in range(3)] for i in range(bone_count)],
        dtype=np.float64,
    )
    result = {
        "ok": bool(
            qpos.shape == (EXPECTED_QPOS_LENGTH,)
            and bone_count == EXPECTED_BONE_COUNT
            and bone_positions.shape == (EXPECTED_BONE_COUNT, 3)
            and math.isclose(float(np.linalg.norm(root_quat)), 1.0, abs_tol=QUAT_NORM_TOL)
        ),
        "slot_sequence": int(slot.SlotSequence),
        "packet_sequence_id": int(slot.PacketSequenceId),
        "frame_count": int(slot.FrameCount),
        "qpos_len": int(frame.QposCount),
        "bone_count": bone_count,
        "root_quat_norm": float(np.linalg.norm(root_quat)),
        "schema_version": _decode_fixed_utf8(slot.SchemaVersionUtf8),
        "bone_name_count": int(slot.BoneNameCount),
        "dof_name_count": int(slot.DofNameCount),
    }
    if not result["ok"]:
        raise RuntimeError(f"Shared-memory slot validation failed: {result}")
    return result


class MotionBricksSharedMemoryRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        _assert_ue_compatible_sizes()
        self.args = args
        self.server = MotionBricksLiveServer(args)
        self.last_header_sequence = 0
        self.latest_command = dict(DEFAULT_IDLE_COMMAND)

    def run(self) -> None:
        with SharedMemoryContext(self.args) as ctx:
            print(
                "MotionBricks UE shared-memory runner "
                f"command_map={self.args.command_map} frame_map={self.args.frame_map} "
                f"command_event={self.args.command_event} frame_event={self.args.frame_event} "
                f"tick_hz={self.args.tick_hz}",
                flush=True,
            )
            period = 1.0 / max(float(self.args.tick_hz), 1e-6)
            step_idx = 0
            while int(self.args.max_steps) <= 0 or step_idx < int(self.args.max_steps):
                loop_start = time.perf_counter()
                ctx.win32.wait_event_zero(ctx.command_event)
                ctx.win32.wait_mutex(ctx.command_mutex, self.args.shared_memory_timeout, "reading command")
                try:
                    header_sequence, command = _read_command_from_buffer(ctx.command_buffer)
                finally:
                    ctx.win32.release_mutex(ctx.command_mutex)

                if command is not None and header_sequence != self.last_header_sequence:
                    self.latest_command = command
                    self.last_header_sequence = header_sequence

                packet = self.server.step(self.latest_command, step_idx)
                _write_packet_to_ring(ctx, packet)
                ctx.win32.set_event(ctx.frame_event)

                step_idx += 1
                sleep_s = period - (time.perf_counter() - loop_start)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

    def run_smoke(self) -> dict[str, Any]:
        with SharedMemoryContext(self.args) as ctx:
            _write_smoke_command(ctx, sequence=1)
            ctx.win32.wait_event_zero(ctx.command_event)
            ctx.win32.wait_mutex(ctx.command_mutex, self.args.shared_memory_timeout, "reading smoke command")
            try:
                header_sequence, command = _read_command_from_buffer(ctx.command_buffer)
            finally:
                ctx.win32.release_mutex(ctx.command_mutex)

            if command is None or header_sequence != 1:
                raise RuntimeError("Smoke command was not readable from UE-compatible FCommandBuffer.")

            packet = self.server.step(command, 0)
            write_sequence, slot_index = _write_packet_to_ring(ctx, packet)
            ctx.win32.set_event(ctx.frame_event)
            slot = _read_latest_slot(ctx)
            result = _slot_validation_summary(slot)
            result["write_sequence"] = write_sequence
            result["slot_index"] = slot_index
            result["command_map"] = self.args.command_map
            result["frame_map"] = self.args.frame_map
            print(
                "shared-memory smoke ok: "
                f"write_sequence={result['write_sequence']} slot_index={result['slot_index']} "
                f"qpos_len={result['qpos_len']} bone_count={result['bone_count']} "
                f"root_quat_norm={result['root_quat_norm']:.6f}",
                flush=True,
            )
            return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = _build_live_arg_parser()
    parser.description = "MotionBricks G1 Windows shared-memory live runner for UE."
    parser.add_argument("--command_map", type=str, default=COMMAND_MAP_NAME)
    parser.add_argument("--frame_map", type=str, default=FRAME_MAP_NAME)
    parser.add_argument("--command_mutex", type=str, default=COMMAND_MUTEX_NAME)
    parser.add_argument("--frame_mutex", type=str, default=FRAME_MUTEX_NAME)
    parser.add_argument("--command_event", type=str, default=COMMAND_EVENT_NAME)
    parser.add_argument("--frame_event", type=str, default=FRAME_EVENT_NAME)
    parser.add_argument("--command_memory_size", type=int, default=COMMAND_MEMORY_SIZE_BYTES)
    parser.add_argument("--frame_memory_size", type=int, default=FRAME_MEMORY_SIZE_BYTES)
    parser.add_argument("--frame_capacity", type=int, default=DEFAULT_FRAME_CAPACITY)
    parser.add_argument("--shared_memory_timeout", type=float, default=SHARED_MEMORY_TIMEOUT_SECONDS)
    parser.add_argument("--idle_mode_name", type=str, default="idle")
    parser.add_argument("--smoke_shared_memory", action="store_true")
    return parser


def main(args: argparse.Namespace) -> None:
    runner = MotionBricksSharedMemoryRunner(args)
    if args.smoke_shared_memory:
        runner.run_smoke()
    else:
        runner.run()


if __name__ == "__main__":
    main(_finalize_live_args(_build_arg_parser().parse_args()))
