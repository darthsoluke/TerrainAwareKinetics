"""Serialized model execution with independently received intent and factual ACKs."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol

import torch


RUNTIME_SCHEMA = "motionbricks.scene_control_v2.runtime.v1"
FPS = 30
REPLAN_FRAMES = 4


@dataclass(frozen=True)
class RuntimeSettings:
    packet_frames: int = 4
    lookahead_tokens: int = 8
    dense_debug: bool = False

    def __post_init__(self):
        if type(self.packet_frames) is not int or self.packet_frames not in (4, 8):
            raise ValueError("packet_frames must be 4 or 8")
        if type(self.lookahead_tokens) is not int or self.lookahead_tokens not in (8, 12, 16):
            raise ValueError("lookahead_tokens must be 8, 12 or 16")
        if type(self.dense_debug) is not bool:
            raise ValueError("dense_debug must be boolean")

    def contract(self):
        return dict(asdict(self), schema=RUNTIME_SCHEMA, fps=FPS, replan_frames=4,
                    pose_passes=4, execute_tokens=8, history_frames=4,
                    postprocessing=None, quality_approved=False)


@dataclass(frozen=True)
class Command:
    command_id: int
    velocity_xz: tuple[float, float]
    heading: float
    received_at: float

    @property
    def intent(self):
        return self.velocity_xz, self.heading


@dataclass(frozen=True)
class RuntimeState:
    history: torch.Tensor
    model_state: Any = None

    def clone(self):
        return RuntimeState(self.history.detach().clone(), copy.deepcopy(self.model_state))


@dataclass(frozen=True)
class StepResult:
    motion: torch.Tensor
    state: RuntimeState
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EncodedMotion:
    payload: bytes
    metadata: Mapping[str, Any]


class RuntimeBackend(Protocol):
    def step(self, state: RuntimeState, command: Command, settings: RuntimeSettings) -> StepResult: ...
    def encode(self, motion: torch.Tensor, *, dense_debug: bool) -> EncodedMotion: ...
    def commit(self, state: RuntimeState) -> None: ...


class SerializedWorker:
    """Share one instance across all sessions using the same model/controller."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scene-control-v2")

    async def run(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(self._executor, function, *args)

    def close(self):
        self._executor.shutdown(wait=True, cancel_futures=False)


@dataclass(frozen=True)
class _Candidate:
    state: RuntimeState
    encoded: EncodedMotion
    base_frame: int
    scene_revision: int
    settings_revision: int
    frame_count: int
    commands: tuple[Command, ...]
    diagnostics: tuple[Mapping[str, Any], ...]
    compute_ms: float


@dataclass(frozen=True)
class PublishedPacket:
    packet_id: int
    start_frame: int
    end_frame: int
    scene_revision: int
    settings_revision: int
    metadata_json: str
    payload: bytes
    state: RuntimeState


def _revision(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class SceneControlDispatcher:
    """At most one immutable published packet and one replaceable candidate.

    Every four-frame model call snapshots the latest command. Calls already in
    progress finish; intent traffic never restarts an in-progress packet. A ready
    but unpublished candidate may be replaced while its predecessor is playing.
    This bounds command staleness without starving publication under input load.
    """

    def __init__(self, backend: RuntimeBackend, initial_state: RuntimeState,
                 send: Callable[[str, bytes | None], Awaitable[None]], worker: SerializedWorker,
                 *, scene_revision=0, settings=RuntimeSettings(), clock=time.monotonic):
        self.backend, self.committed_state = backend, initial_state
        self.send, self.worker, self.clock = send, worker, clock
        self.scene_revision = _revision(scene_revision, "scene_revision")
        self.settings, self.settings_revision = settings, 0
        self.latest: Command | None = None
        self.pending: PublishedPacket | None = None
        self.candidate: _Candidate | None = None
        self.committed_frame = self.packet_id = 0
        self.sent_complete_at: float | None = None
        self.publish_started_at: float | None = None
        self.failures: list[dict] = []
        self.discarded_candidates = 0
        self._wake = asyncio.Event()
        self._published = asyncio.Event()
        self._ack_lock = asyncio.Lock()
        self._task = None
        self._closed = False
        self.paused = False
        self._generation_epoch = 0
        self._failure = None
        self.failure_reported = False
        self._ack_in_flight = False
        self.transport_failed = False
        self.aborted_publication = None
        self._building_base_frame = 0
        self._building_predecessor_id = None
        self._building_base_state = initial_state
        self._building_step_state = initial_state
        self._building_phase = "step"
        self._building_replan_index = 0
        self._computing_command_id = None
        self._computing_revisions = (self.scene_revision, self.settings_revision)

    def start(self):
        if self._task is not None:
            raise RuntimeError("Dispatcher already started")
        self._task = asyncio.create_task(self._pump())

    def _check_revisions(self, message):
        if (_revision(message.get("scene_revision"), "scene_revision") != self.scene_revision
                or _revision(message.get("settings_revision"), "settings_revision") != self.settings_revision):
            raise ValueError("Stale scene/settings revision")

    def receive_command(self, message):
        if self._closed:
            raise ValueError("Dispatcher is closed")
        if message.get("type") != "command":
            raise ValueError("Expected command message")
        self._check_revisions(message)
        command_id = _revision(message.get("command_id"), "command_id")
        velocity, heading = message.get("velocity_xz"), message.get("heading")
        if (not isinstance(velocity, (list, tuple)) or len(velocity) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in velocity)
                or math.hypot(*velocity) > 12.0):
            raise ValueError("velocity_xz must contain finite values with speed <=12 m/s")
        if type(heading) not in (int, float) or not math.isfinite(heading):
            raise ValueError("heading must be finite radians")
        command = Command(command_id, tuple(float(v) for v in velocity), float(heading), self.clock())
        if self.latest is not None and command_id <= self.latest.command_id:
            if command_id != self.latest.command_id or command.intent != self.latest.intent:
                raise ValueError("command_id must increase; repeated IDs require identical intent")
            duplicate = True
        else:
            duplicate = False
            changed = self.latest is None or self.latest.intent != command.intent
            self.latest = command
            if changed and self.candidate is not None and self.pending is not None:
                self.candidate = None
                self.discarded_candidates += 1
            self._wake.set()
        return dict(type="command_received", schema=RUNTIME_SCHEMA, command_id=command_id,
                    scene_revision=self.scene_revision, settings_revision=self.settings_revision,
                    duplicate=duplicate)

    def update_settings(self, settings: RuntimeSettings, revision: int):
        if not isinstance(settings, RuntimeSettings):
            raise TypeError("Expected RuntimeSettings")
        if _revision(revision, "settings_revision") != self.settings_revision + 1:
            raise ValueError("Expected the next settings revision")
        self.settings, self.settings_revision = settings, revision
        self.latest = None
        self.candidate = None
        self._wake.set()

    def set_paused(self, paused: bool):
        if type(paused) is not bool:
            raise ValueError("paused must be boolean")
        if paused != self.paused:
            self.paused = paused
            self._generation_epoch += 1
            self.candidate = None
            self._wake.set()

    def replace_scene(self, backend: RuntimeBackend, state: RuntimeState, revision: int):
        if self.pending is not None:
            raise ValueError("Drain every published packet before replacing the scene")
        if _revision(revision, "scene_revision") != self.scene_revision + 1:
            raise ValueError("Expected the next scene revision")
        self.backend, self.committed_state, self.scene_revision = backend, state, revision
        self.committed_frame = 0
        self.latest = None
        self.candidate = None
        self._wake.set()

    @staticmethod
    def _step(backend, state, command, settings):
        with torch.inference_mode():
            if (state.history.ndim != 3 or state.history.shape[:2] != (1, 4)
                    or not torch.isfinite(state.history).all()):
                raise ValueError("Expected four finite factual history frames")
            result = backend.step(state.clone(), command, settings)
            if (result.motion.shape != (1, REPLAN_FRAMES, state.history.shape[-1])
                    or not torch.isfinite(result.motion).all()
                    or result.state.history.shape != state.history.shape
                    or not torch.isfinite(result.state.history).all()):
                raise ValueError("Backend violated its four-frame finite step contract")
            return StepResult(result.motion.detach().clone(), result.state.clone(),
                              copy.deepcopy(result.diagnostics) if settings.dense_debug else {})

    @staticmethod
    def _encode(backend, parts, settings):
        with torch.inference_mode():
            encoded = backend.encode(torch.cat(parts, 1), dense_debug=settings.dense_debug)
            if not isinstance(encoded.payload, bytes) or not encoded.payload:
                raise ValueError("Backend must encode immutable nonempty bytes")
            return EncodedMotion(encoded.payload, copy.deepcopy(encoded.metadata))

    @staticmethod
    def _failure_roots(factual, pending, build, replan):
        # A small serialized device read only after failure, never a scene query.
        states = [factual, build, replan]
        if pending is not None:
            states.append(pending.state)
        with torch.inference_mode():
            roots = torch.cat([state.history[..., :5] for state in states], 0).detach().cpu().tolist()
        return dict(factual_root5=roots[0], build_input_root5=roots[1],
                    failed_replan_input_root5=roots[2],
                    pending_end_root5=roots[3] if pending is not None else None)

    async def _build(self):
        backend, settings = self.backend, self.settings
        revisions = self.scene_revision, self.settings_revision, self._generation_epoch
        self._computing_revisions = revisions[:2]
        state = self.pending.state if self.pending is not None else self.committed_state
        base_frame = self.pending.end_frame if self.pending is not None else self.committed_frame
        self._building_base_frame = base_frame
        self._building_predecessor_id = self.pending.packet_id if self.pending is not None else None
        self._building_base_state = state
        parts, commands, diagnostics = [], [], []
        started = self.clock()
        for index in range(settings.packet_frames // REPLAN_FRAMES):
            command = self.latest
            self._building_phase = "backend.step"
            self._building_replan_index = index
            self._building_step_state = state
            self._computing_command_id = command.command_id
            result = await self.worker.run(self._step, backend, state, command, settings)
            if self._closed or revisions != (self.scene_revision, self.settings_revision, self._generation_epoch):
                self.discarded_candidates += 1
                return None
            parts.append(result.motion)
            commands.append(command)
            diagnostics.append(result.diagnostics)
            state = result.state
        self._building_phase = "backend.encode"
        encoded = await self.worker.run(self._encode, backend, parts, settings)
        if self._closed or revisions != (self.scene_revision, self.settings_revision, self._generation_epoch):
            self.discarded_candidates += 1
            return None
        return _Candidate(state, encoded, base_frame, *revisions[:2], settings.packet_frames,
                          tuple(commands), tuple(diagnostics), (self.clock() - started) * 1000)

    async def _pump(self):
        while not self._closed:
            await self._wake.wait()
            self._wake.clear()
            if self._closed:
                break
            if self._failure is not None:
                if (not self.failure_reported and not self._ack_in_flight and self.pending is None
                        and self.committed_frame >= self._failure["dependency_frame"]):
                    try:
                        await self.send(json.dumps(self._failure), None)
                        self.failure_reported = True
                    except (ConnectionError, OSError) as error:
                        self.transport_failed = True
                        self.stop()
                        failure = dict(type="runtime_failure", schema=RUNTIME_SCHEMA,
                                       phase="transport.send_failure", committed_frame=self.committed_frame,
                                       message=f"{type(error).__name__}: {error}",
                                       traceback=traceback.format_exc())
                        self.failures.append(failure)
                        logging.getLogger(__name__).error(json.dumps(failure))
                continue
            while not self._closed and self.latest is not None and not self.paused:
                try:
                    if self.candidate is None:
                        self.candidate = await self._build()
                    if self.candidate is None:
                        continue
                    if self.pending is not None or self._ack_in_flight:
                        break
                    candidate = self.candidate
                    if candidate.base_frame != self.committed_frame:
                        raise RuntimeError("Candidate does not follow the acknowledged frame frontier")
                    self.packet_id += 1
                    metadata = dict(candidate.encoded.metadata)
                    metadata.update(type="plan", schema=RUNTIME_SCHEMA, packet_id=self.packet_id,
                                    scene_revision=candidate.scene_revision,
                                    settings_revision=candidate.settings_revision,
                                    start_frame=candidate.base_frame,
                                    end_frame=candidate.base_frame + candidate.frame_count,
                                    frame_count=candidate.frame_count, fps=FPS, replan_frames=4,
                                    pose_passes=4, command_ids=[c.command_id for c in candidate.commands],
                                    commands=[dict(command_id=c.command_id, velocity_xz=c.velocity_xz,
                                                   heading=c.heading) for c in candidate.commands],
                                    compute_ms=candidate.compute_ms, quality_approved=False,
                                    command_age_ms=[(self.clock() - c.received_at) * 1000
                                                    for c in candidate.commands],
                                    postprocessing=None)
                    if self.settings.dense_debug:
                        metadata["debug"] = list(candidate.diagnostics)
                    encoded_json = json.dumps(metadata, allow_nan=False, separators=(",", ":"))
                    self.pending = PublishedPacket(self.packet_id, candidate.base_frame,
                                                   candidate.base_frame + candidate.frame_count,
                                                   candidate.scene_revision, candidate.settings_revision,
                                                   encoded_json, candidate.encoded.payload, candidate.state)
                    self.candidate = None
                    self._published.clear()
                    self.sent_complete_at = None
                    self.publish_started_at = self.clock()
                    self._building_phase = "transport.send_plan"
                    try:
                        await self.send(encoded_json, self.pending.payload)
                    except Exception:
                        self.transport_failed = True
                        self.aborted_publication = self.pending
                        self.pending = None
                        self.stop()
                        self._published.set()
                        raise
                    self.sent_complete_at = self.clock()
                    self._published.set()
                except Exception as error:
                    scene_revision, settings_revision = self._computing_revisions
                    factual, pending = self.committed_state, self.pending
                    failure = dict(type="runtime_failure", schema=RUNTIME_SCHEMA,
                                   command_id=self._computing_command_id,
                                   scene_revision=scene_revision,
                                   settings_revision=settings_revision,
                                   dependency_frame=self._building_base_frame,
                                   predecessor_packet_id=self._building_predecessor_id,
                                   committed_frame=self.committed_frame,
                                   factual_history_role="last_full_ack_at_failure_capture",
                                   root5_layout=["x_m", "y_m", "z_m", "heading_cos", "heading_sin"],
                                   pending_packet=(dict(packet_id=pending.packet_id,
                                                        start_frame=pending.start_frame,
                                                        end_frame=pending.end_frame,
                                                        history_role="published_not_fully_acknowledged")
                                                   if pending is not None else None),
                                   build_history_role=("published_not_acknowledged_at_build_start"
                                                       if self._building_predecessor_id is not None
                                                       else "factual_at_build_start"),
                                   replan_history_role=("unpublished_candidate_prefix"
                                                        if self._building_replan_index else "build_input"),
                                   phase=self._building_phase,
                                   replan_index=self._building_replan_index,
                                   traceback=traceback.format_exc(),
                                   message=f"{type(error).__name__}: {error}")
                    if self.aborted_publication is not None:
                        aborted = self.aborted_publication
                        failure["aborted_packet"] = dict(packet_id=aborted.packet_id,
                                                         start_frame=aborted.start_frame,
                                                         end_frame=aborted.end_frame,
                                                         publication_status="transport_send_failed")
                    self.failures.append(failure)
                    self._failure = failure
                    try:
                        failure["roots"] = await self.worker.run(
                            self._failure_roots, factual, pending,
                            self._building_base_state, self._building_step_state)
                    except Exception as diagnostic_error:
                        failure["root_capture_error"] = repr(diagnostic_error)
                    logging.getLogger(__name__).error(json.dumps(failure))
                    self._wake.set()
                    break

    async def acknowledge(self, message):
        async with self._ack_lock:
            pending = self.pending
            if message.get("type") != "played" or pending is None:
                raise ValueError("No published packet awaits this ACK")
            expected = dict(packet_id=pending.packet_id,
                            frame_count=pending.end_frame - pending.start_frame,
                            scene_revision=pending.scene_revision,
                            settings_revision=pending.settings_revision)
            if any(type(message.get(key)) is not int or message[key] != value
                   for key, value in expected.items()):
                raise ValueError("ACK must match the entire published packet and its revisions")
            # The first frame can be presented immediately. The last frame is
            # presented (N-1)/30 seconds later. Send-start is an earlier clock
            # bound than browser binary receipt, even under socket backpressure.
            received_at = self.clock()
            earliest = self.publish_started_at + (expected["frame_count"] - 1) / FPS
            if received_at + 1e-9 < earliest:
                raise ValueError("Early playback ACK")
            await self._published.wait()
            if self.transport_failed:
                raise ValueError("Plan transport failed; this packet cannot be acknowledged")
            self._ack_in_flight = True
            try:
                await self.worker.run(self.backend.commit, pending.state)
                self.committed_state = pending.state
                self.committed_frame = pending.end_frame
                self.pending = None
                receipt = dict(type="played", schema=RUNTIME_SCHEMA, **expected,
                               committed_frame=self.committed_frame,
                               send_complete_at=self.sent_complete_at,
                               ack_received_at=received_at)
                # Publication/failure release is gated by the actual played
                # receipt send, not just by committing the ACK on the worker.
                await self.send(json.dumps(receipt, allow_nan=False), None)
            except (ConnectionError, OSError) as error:
                self.transport_failed = True
                self.stop()
                failure = dict(type="runtime_failure", schema=RUNTIME_SCHEMA,
                               phase="transport.send_played", committed_frame=self.committed_frame,
                               packet_id=pending.packet_id, message=f"{type(error).__name__}: {error}",
                               traceback=traceback.format_exc())
                self.failures.append(failure)
                logging.getLogger(__name__).error(json.dumps(failure))
                raise
            finally:
                self._ack_in_flight = False
            self._wake.set()
            return receipt

    async def close(self):
        self.stop()
        if self._task is not None:
            await self._task

    def stop(self):
        self._closed = True
        self._wake.set()


class SOMADualBackend:
    """Real native SOMA/dual engine adapter, without dense debug serialization."""

    def __init__(self, runtime):
        from motionbricks.data.dual_horizon_execution import DualHorizonExecution
        from motionbricks.data.root_path_split_execution import SplitRootPathExecution

        self.runtime = runtime
        self.inference = runtime.inference
        runner = self.inference._terrain_execution
        if not isinstance(runner, SplitRootPathExecution):
            raise ValueError("SOMA v2 requires the existing split Root-path engine")
        self.engine = DualHorizonExecution.from_runner(runner)
        self.history = runtime.terrain_initial_history.detach().clone()
        self.scene_revision = runtime.scene_revision

    def initial_state(self):
        return RuntimeState(self.history.detach().clone())

    @torch.inference_mode()
    def step(self, state, command, settings):
        from motionbricks.data.native_mb_online_live import online_command_goal

        if self.runtime.scene_revision != self.scene_revision:
            raise ValueError("SOMA scene changed; drain and create a new adapter")
        history = state.history
        engine = self.engine
        goal = online_command_goal(history[..., :5], command.velocity_xz,
                                   command.heading, 8, engine.protocol_version)
        query = history.new_tensor([[-1., .6, 1., .4]])
        with engine.controller.context(history, engine.scene):
            if settings.lookahead_tokens == 8:
                result = engine._prediction(history.contiguous(), goal.contiguous())
            else:
                prepared = engine.prepare_dual(history, goal, query, settings.lookahead_tokens)
                result = engine.compiled_suffix(history, goal, query, prepared)
        if not torch.isfinite(result["motion"]).all() or not (result["count"] == 8).all():
            raise ValueError("SOMA prediction violated its finite 8-token horizon")
        motion = result["motion"][:, 4:8].detach().clone()
        observed = engine.observe(motion).detach().clone()
        diagnostics = {}
        if settings.dense_debug:
            engine._describe_prediction(history, goal, diagnostics, result, 8)
        return StepResult(motion, RuntimeState(observed, copy.deepcopy(state.model_state)), diagnostics)

    @torch.inference_mode()
    def encode(self, motion, *, dense_debug):
        from motionbricks.data.soma_lab import pack_geometry_frames

        roots, vertices = self.runtime.skin_features(motion, self.inference)
        if vertices.shape != (motion.shape[1], 18056, 3):
            raise ValueError("SOMA transport requires all 18056 official skin vertices")
        payload = pack_geometry_frames(roots, vertices)
        metadata = dict(encoding="float32-le", geometry_format="soma-full-vertices-v1",
                        vertex_count=18056, frame_stride_floats=3 + 18056 * 3)
        return EncodedMotion(payload, metadata)

    def commit(self, state):
        self.history = state.history.detach().clone()
