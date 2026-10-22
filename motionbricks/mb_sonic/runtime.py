"""DDS-free SONIC PD control of a canonical scene's real MuJoCo G1 model."""

import copy
from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from functools import partial
import hashlib
import json
from multiprocessing import get_context
import sys
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter_ns
from typing import Callable
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .reference import (ExecutedHistory, G1_JOINT_NAMES, NativeSomaHistory, OFFICIAL_BODY_NAMES,
                        ReferenceCoverageError, SONIC_FPS, SONIC_FUTURE_OFFSETS, load_official_reference, load_g1_reference)
from .scene import load_scene, mujoco_to_soma, sha256
from .physics_numerics import configure_physics_numerics
from .cached_forward_step import cached_forward_step
from .contact_batch import ContactBatch, capture_contacts
from .numeric_compression import encode_npz
from .static_heap import StaticHeap
from .sonic_policy import (BODY_JOINT_NAMES, DEFAULT_ANGLES, FrozenSonicPolicy, HISTORY_FRAMES, KD, KP,
                           ObservationHistory, POLICY_DT, Proprioception)
from .windows_clock import WindowsTimerResolution, wait_until


def configure_thread_schedule(clock):
    if "python_thread_switch_interval_s" in clock:
        interval = float(clock["python_thread_switch_interval_s"])
        if not np.isfinite(interval) or not 0 < interval <= .005:
            raise ValueError("The explicit Python thread interval must be positive and at most 5ms")
        sys.setswitchinterval(interval)
    return sys.getswitchinterval()


def load_physics_scene(manifest_path, robot_xml_path, *, contact_pair_binding=None, flat_ground_primitive=None):
    """Compose only the canonical scene and original G1 robot, with no extra floor."""
    scene = load_scene(manifest_path)
    robot_path = Path(robot_xml_path).resolve(strict=True)
    root = ET.parse(robot_path).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((robot_path.parent / compiler.attrib["meshdir"]).resolve(strict=True)))
    ground_implementation = "full_resolution_hfield"
    if flat_ground_primitive is not None:
        if flat_ground_primitive != "exact_finite_box_v1":
            raise ValueError("Unknown exact flat-ground collision representation")
        if np.ptp(scene.height) == 0:
            include = ET.parse(scene.scene_include_path).getroot()
            name = next(row["geom_name"] for row in scene.manifest.objects if row["purpose"] == "terrain")
            geom = include.find(f"./worldbody/geom[@name='{name}']")
            half_x, half_y, _, depth = scene.manifest.heightfield["size_m"]
            x, y, top = scene.manifest.heightfield["center_xyz_m"]
            geom.attrib.pop("hfield")
            geom.set("type", "box")
            geom.set("size", f"{half_x:.17g} {half_y:.17g} {depth/2:.17g}")
            geom.set("pos", f"{x:.17g} {y:.17g} {top-depth/2:.17g}")
            for element in include:
                root.append(element)
            ground_implementation = "exact_finite_box_v1"
        else:
            ET.SubElement(root, "include", file=str(scene.scene_include_path))
    else:
        ET.SubElement(root, "include", file=str(scene.scene_include_path))
    contact_receipt = None
    if contact_pair_binding is not None:
        path = Path(contact_pair_binding["path"]).resolve(strict=True)
        if sha256(path) != contact_pair_binding["sha256"]:
            raise ValueError("Robot contact-pair template hash differs")
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        ground = next(row["geom_name"] for row in scene.manifest.objects if row["purpose"] == "terrain")
        slide, torsion, roll = scene.manifest.friction["coefficients"]
        pairs = ET.parse(path).getroot()
        for pair in pairs:
            if pair.tag != "pair" or pair.attrib["geom2"] != "floor":
                raise ValueError("Expected an explicit foot-to-scene contact template")
            pair.set("geom2", ground)
            pair.set("friction", " ".join(map(str, (slide, slide, torsion, roll, roll))))
            pair.set("condim", str(scene.manifest.friction["condim"]))
            contact.append(pair)
        contact_receipt = dict(contact_pair_binding, count=len(pairs), ground_geom=ground,
                               friction_source="canonical_scene_manifest", extra_ground=False)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return scene, model, dict(path=str(robot_path), sha256=sha256(robot_path),
                              contact_pair_binding=contact_receipt,
                              ground_collision_implementation=ground_implementation,
                              hands="original_collision_and_dynamics_of_declared_robot_XML_preserved")


def load_mb_seed_history(settings, scene):
    scene_id = scene.manifest.scene_id
    if "scene_seeds" in settings:
        entry = settings["scene_seeds"][scene_id]
        source = scene.manifest.source["scene"]
        path = Path(entry["path"]).resolve(strict=True)
        if path != (scene.root/source["file"]).resolve(strict=True) or entry["sha256"] != source["sha256"]:
            raise ValueError("Per-scene MB seed must be the canonical manifest's exact source asset")
        expected_hash = entry["sha256"]
    else:
        path = Path(settings["mb_seed_path"]).resolve(strict=True)
        expected_hash = settings["mb_seed_sha256"]
    if sha256(path) != expected_hash:
        raise ValueError("Frozen MB source seed SHA256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        values = archive["initial_history"]
        if values.shape != (1, 4, 366) or values.dtype != np.float32:
            raise ValueError("The scene asset must contain one actual FP32 four-frame native seed")
        features = values[0].copy()
    return NativeSomaHistory((np.arange(4)-3)/30, features, scene_id,
        dict(kind="source_seed", file=str(path), sha256=expected_hash,
             scene_manifest_sha256=sha256(scene.manifest_path), source_key="initial_history",
             spatial_transform="none"))


def _reference_body_parents(model, body_names):
    body_ids = [model.body(name).id for name in body_names]
    body_indexes = {body_id: index for index, body_id in enumerate(body_ids)}
    parents = []
    for body_id in body_ids:
        parent_id = int(model.body_parentid[body_id])
        while parent_id and parent_id not in body_indexes:
            parent_id = int(model.body_parentid[parent_id])
        parents.append(body_indexes[parent_id] if parent_id in body_indexes else -1)
    return parents


@dataclass(frozen=True)
class ContactForce:
    geom_ids: tuple
    body_ids: tuple
    position_world: np.ndarray
    frame_contact_from_world: np.ndarray
    wrench_contact: np.ndarray
    distance: float


@dataclass(frozen=True)
class PhysicsState:
    time_s: float
    captured_at_monotonic_ns: int
    state_version: int
    qpos: np.ndarray
    qvel: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    body_positions_world: np.ndarray
    body_quat_wxyz: np.ndarray
    actuator_controls: np.ndarray
    actuator_forces: np.ndarray
    generalized_actuator_forces: np.ndarray
    root_qfrc_actuator: np.ndarray
    qfrc_applied: np.ndarray
    xfrc_applied: np.ndarray
    solver_warnings: dict
    root_ground_known: bool
    root_ground_height_m: float | None
    contacts: tuple
    terminal_reason: str | None


@dataclass(frozen=True)
class RuntimeTick:
    scene_id: str
    command_id: str
    phase: str
    reference_time_s: float
    issued_at_monotonic_ns: int
    raw_reference: dict
    adapted_reference: object
    policy_command: object
    physics_states: tuple
    timing_ns: dict

    @property
    def actual(self):
        return self.physics_states[-1]


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot record {type(value).__name__}")


class IntentMailbox:
    """One newest intent, not a FIFO; arrival never means policy consumption."""

    def __init__(self):
        self._lock = Lock()
        self._revision = 0
        self._command = None
        self._unapplied = deque()

    def replace(self, command):
        with self._lock:
            self._revision += 1
            self._command = copy.deepcopy(command)
            self._unapplied.append((self._revision, command["server_received_ns"]))
            return self._revision

    def latest(self):
        with self._lock:
            return self._revision, copy.deepcopy(self._command)

    def pending_since_ns(self, applied_revision):
        with self._lock:
            while self._unapplied and self._unapplied[0][0] <= applied_revision:
                self._unapplied.popleft()
            return self._unapplied[0][1] if self._unapplied else None


def _root_heading(qpos):
    forward = Rotation.from_quat(np.asarray(qpos)[[4, 5, 6, 3]]).apply([1, 0, 0])
    if np.linalg.norm(forward[:2]) < 1e-7:
        return None
    return float(np.arctan2(forward[1], forward[0]))


def _angle_delta(target, origin):
    return float(np.arctan2(np.sin(target-origin), np.cos(target-origin)))


class YawIntentIntegrator:
    """Integrate the newest raw yaw rate on native 50Hz ticks, independent of XY motion."""

    def __init__(self, initial_actual_heading_rad):
        if initial_actual_heading_rad is None:
            raise ValueError("Initial desired yaw requires an observable actual root heading")
        self.desired_heading_rad = float(initial_actual_heading_rad)
        self.tick = 0
        self._tick_time_s = None
        self._tick_origin_heading = self.desired_heading_rad

    def for_tick(self, command, time_s):
        if self._tick_time_s is None or not np.isclose(time_s, self._tick_time_s, atol=1e-9, rtol=0):
            if self._tick_time_s is not None and not np.isclose(time_s-self._tick_time_s, POLICY_DT, atol=1e-8, rtol=0):
                raise ValueError("Yaw-rate integration must advance on native 50Hz ticks")
            self._tick_origin_heading = self.desired_heading_rad
            self._tick_time_s = time_s
            self.tick += 1
        self.desired_heading_rad = _angle_delta(
            self._tick_origin_heading+float(command["yaw_rate_radps"])*POLICY_DT, 0.)
        return self.desired_heading_rad


class BodyOnsetObserver:
    """First sustained, command-directed root velocity/heading change after consumption."""

    def __init__(self, *, linear_threshold_mps, heading_threshold_rad, sustained_samples, sample_hz):
        if linear_threshold_mps <= 0 or heading_threshold_rad <= 0 or sustained_samples < 1 or sample_hz <= 0:
            raise ValueError("Explicit positive body-onset thresholds and persistence are required")
        self.linear_threshold_mps = float(linear_threshold_mps)
        self.heading_threshold_rad = float(heading_threshold_rad)
        self.sustained_samples = int(sustained_samples)
        self.sample_hz = float(sample_hz)
        self.command = None
        self.onset = None

    def consume(self, command, state, *, desired_heading_rad, consumed_at_monotonic_ns):
        identity = (command["session_id"], command["command_id"])
        if self.command is not None and identity == self.identity:
            self.desired_heading_change = (_angle_delta(desired_heading_rad, self.baseline_heading)
                                           if self.baseline_heading is not None else None)
            return
        self.identity, self.command = identity, copy.deepcopy(command)
        self.baseline_time_s = state.time_s
        self.baseline_velocity = state.qvel[:2].copy()
        self.baseline_heading = _root_heading(state.qpos)
        self.desired_velocity_change = np.asarray(command["velocity_xy_mps"])-self.baseline_velocity
        self.desired_heading_change = (_angle_delta(desired_heading_rad, self.baseline_heading)
                                       if self.baseline_heading is not None else None)
        self.consumed_at_monotonic_ns = consumed_at_monotonic_ns
        self.onset, self._candidate, self._consecutive = None, None, 0

    def observe(self, state):
        if self.command is None or self.onset is not None or state.time_s <= self.baseline_time_s:
            return self.onset
        requested_change = float(np.linalg.norm(self.desired_velocity_change))
        directed_velocity_change = None
        if requested_change >= self.linear_threshold_mps:
            directed_velocity_change = float(np.dot(state.qvel[:2]-self.baseline_velocity,
                                                     self.desired_velocity_change/requested_change))
        directed_heading_change = None
        actual_heading = _root_heading(state.qpos)
        rate = float(self.command["yaw_rate_radps"])
        heading_direction = np.sign(rate) if rate != 0 else None
        if (heading_direction is None and self.desired_heading_change is not None
                and abs(self.desired_heading_change) >= self.heading_threshold_rad):
            heading_direction = np.sign(self.desired_heading_change)
        if heading_direction is not None and self.baseline_heading is not None and actual_heading is not None:
            directed_heading_change = float(heading_direction * _angle_delta(actual_heading, self.baseline_heading))
        changed = ((directed_velocity_change is not None and directed_velocity_change >= self.linear_threshold_mps)
                   or (directed_heading_change is not None and directed_heading_change >= self.heading_threshold_rad))
        if not changed:
            self._candidate, self._consecutive = None, 0
            return None
        if self._candidate is None:
            self._candidate = dict(sim_time_s=state.time_s, sample_monotonic_ns=state.captured_at_monotonic_ns,
                                   directed_velocity_change_mps=directed_velocity_change,
                                   directed_heading_change_rad=directed_heading_change)
        self._consecutive += 1
        if self._consecutive >= self.sustained_samples:
            self.onset = dict(self._candidate, command_id=self.command["command_id"],
                              session_id=self.command["session_id"],
                              detected_at_monotonic_ns=state.captured_at_monotonic_ns,
                              consumed_at_monotonic_ns=self.consumed_at_monotonic_ns,
                              baseline_time_s=self.baseline_time_s,
                              baseline_velocity_xy_mps=self.baseline_velocity.tolist(),
                              baseline_heading_rad=self.baseline_heading,
                              requested_yaw_rate_radps=self.command["yaw_rate_radps"],
                              criterion="sustained_command_directed_root_velocity_or_heading_change",
                              linear_threshold_mps=self.linear_threshold_mps,
                              heading_threshold_rad=self.heading_threshold_rad,
                              sustained_samples=self.sustained_samples, sample_hz=self.sample_hz,
                              sustained_span_s=state.time_s-self._candidate["sim_time_s"])
        return self.onset


class RecordingBudget:
    """Shared physical-byte account; reserve drains bounded queues after stop."""

    def __init__(self, maximum_bytes, reserve_bytes, max_json_record_bytes):
        if not 0 < reserve_bytes < maximum_bytes or max_json_record_bytes <= 0:
            raise ValueError("Recording needs a positive total byte limit and explicit drain reserve")
        self.maximum_bytes, self.reserve_bytes = maximum_bytes, reserve_bytes
        self.max_json_record_bytes = max_json_record_bytes
        self.used_bytes, self.exhausted = 0, False
        self._lock = Lock()

    def reserve(self, byte_count):
        with self._lock:
            if self.used_bytes+byte_count > self.maximum_bytes:
                raise RuntimeError("Recording hard byte cap exhausted; required record cannot be persisted")
            self.used_bytes += byte_count
            if self.used_bytes > self.maximum_bytes-self.reserve_bytes:
                self.exhausted = True

    def check(self):
        with self._lock:
            if self.exhausted:
                raise RuntimeError("Recording storage budget exhausted; bounded pending records are draining")


class PhysicsRecorder:
    """Append actual state and both references; no resampling or render synthesis."""

    def __init__(self, path, metadata, *, storage_budget=None):
        self._lock = Lock()
        self.storage_budget, self.tick_writer = storage_budget, None
        self.tick_format = "numeric_npz"
        self.stream = Path(path).open("x", encoding="utf-8", newline="" if storage_budget is not None else None)
        self.write(dict(kind="metadata", **metadata))

    def write(self, record):
        line = json.dumps(record, default=_json_value, allow_nan=False) + "\n"
        if self.storage_budget is not None:
            size = len(line.encode("utf-8"))
            if size > self.storage_budget.max_json_record_bytes:
                raise RuntimeError("JSON control receipt exceeds its explicit byte bound")
            self.storage_budget.reserve(size)
        with self._lock:
            self.stream.write(line)
            self.stream.flush()

    def tick(self, tick):
        if self.tick_writer is not None:
            record_id = f"tick:{tick.actual.state_version}"
            if self.tick_format == "columnar_npz_v1":
                from .columnar_physics import pack_physics_states

                values = {field.name: getattr(tick, field.name) for field in fields(tick)
                          if field.name != "physics_states"}
                payload = dict(kind="full_policy_tick_columnar", tick=values,
                               physics=pack_physics_states(tick.physics_states))
            else:
                payload = dict(kind="full_policy_tick", tick=tick)
            self.tick_writer.submit(record_id, payload)
            self.write(dict(kind="policy_tick_binary", numeric_record_id=record_id,
                scene_id=tick.scene_id, command_id=tick.command_id, reference_time_s=tick.reference_time_s,
                actual_time_s=tick.actual.time_s, state_version=tick.actual.state_version,
                captured_at_monotonic_ns=tick.actual.captured_at_monotonic_ns))
            return
        self.write(dict(kind="policy_tick", **asdict(tick)))

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class NumericLogBackpressure(RuntimeError):
    pass


def _numeric_snapshot(value):
    """Own read-only array copies; never share mutable pending data with a writer."""
    if isinstance(value, np.ndarray):
        copied = np.array(value, copy=True, order="C")
        copied.flags.writeable = False
        return copied
    if is_dataclass(value):
        return {field.name: _numeric_snapshot(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: _numeric_snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_numeric_snapshot(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


class NumericArtifactWriter:
    """Bounded lossless NPZ logging, with one reserved fatal-overflow record."""

    def __init__(self, directory, recorder, *, pending_plans, storage_budget=None, max_record_bytes=None,
                 compression_executor=None):
        if pending_plans < 1:
            raise ValueError("Required numeric logging needs an explicit positive queue capacity")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.recorder = recorder
        self.storage_budget, self.max_record_bytes = storage_budget, max_record_bytes
        self.compression_executor = compression_executor
        self._queue = Queue(maxsize=pending_plans)
        self._stop, self._lock = Event(), Lock()
        self._overflow, self._failed_item, self._failure = None, None, None
        self._thread = None

    def start(self):
        self._thread = Thread(target=self._run, name="mb-numeric-log", daemon=False)
        self._thread.start()

    def submit(self, record_id, record):
        self.check()
        started_ns = perf_counter_ns()
        snapshot = _numeric_snapshot(record)
        item = (record_id, snapshot, perf_counter_ns()-started_ns)
        try:
            self._queue.put_nowait(item)
        except Full:
            with self._lock:
                self._overflow = item
                self._failure = "required_numeric_log_backpressure; overflow_record_retained"
            self._stop.set()
            raise NumericLogBackpressure(self._failure)

    def check(self):
        with self._lock:
            if self._failure is not None:
                raise RuntimeError(self._failure)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.check()

    def _persist(self, item):
        record_id, record, snapshot_ns = item
        arrays = {}

        def encode(value):
            if isinstance(value, np.ndarray):
                key = f"array_{len(arrays):04d}"
                arrays[key] = value
                return dict(array=key, shape=list(value.shape), dtype=value.dtype.str)
            if isinstance(value, dict):
                return {key: encode(child) for key, child in value.items()}
            if isinstance(value, tuple):
                return [encode(child) for child in value]
            return value

        started_ns = perf_counter_ns()
        metadata = dict(schema="motionbricks.mb_sonic.numeric_record.v1", record_id=record_id, record=encode(record))
        metadata_json = json.dumps(metadata, allow_nan=False, separators=(",", ":"))
        artifact_id = hashlib.sha256(record_id.encode()).hexdigest()
        path = self.directory / (artifact_id+".npz")
        payload = dict(metadata_json=np.asarray(metadata_json), **arrays)
        if self.compression_executor is None:
            encoded = encode_npz(payload)
        else:
            encoded = self.compression_executor.submit(encode_npz, payload).result(timeout=10)
        if self.max_record_bytes is not None:
            if len(encoded) > self.max_record_bytes:
                raise RuntimeError("Numeric required record exceeds its explicit single-record byte bound")
            self.storage_budget.reserve(len(encoded))
        with path.open("xb") as stream:
            stream.write(encoded)
        self.recorder.write(dict(kind="numeric_artifact_written", record_id=record_id,
            path=str(path.resolve()), sha256=sha256(path), bytes=path.stat().st_size,
            array_count=len(arrays), snapshot_copy_ns=snapshot_ns,
            persistence_ns=perf_counter_ns()-started_ns, completed_at_monotonic_ns=perf_counter_ns()))

    def _run(self):
        item = None
        try:
            while True:
                try:
                    item = self._queue.get(timeout=.05)
                except Empty:
                    if self._stop.is_set():
                        break
                    continue
                self._persist(item)
                self._queue.task_done()
                item = None
            with self._lock:
                overflow, self._overflow = self._overflow, None
            if overflow is not None:
                item = overflow
                self._persist(item)
                item = None
        except Exception as error:
            with self._lock:
                self._failed_item = item
                self._failure = f"numeric_log_failure:{type(error).__name__}:{error}"
            self.recorder.write(dict(kind="numeric_log_failure", error=self._failure,
                                     unpersisted_record_id=item[0] if item is not None else None,
                                     monotonic_ns=perf_counter_ns()))


@dataclass(frozen=True)
class MBReferenceUpdate:
    intent_revision: int
    command: dict
    history_time_s: float
    derived_heading_rad: float
    conditioning_history: object
    estimated_actual_history: object
    proposal: object
    adapted_reference: object
    timing_ns: dict
    clocks_ns: dict
    packet: object


@dataclass(frozen=True)
class MBGenerationSnapshot:
    simulation_time_s: float
    history_time_s: float
    state_version: int
    integrated_intent_revision: int
    actual_root_xyz_m: np.ndarray
    derived_heading_rad: float
    reference_history: object
    executed_history: object
    captured_at_monotonic_ns: int
    timing_ns: dict
    reference_source: object = None
    previous_reference_packet: object = None


@dataclass(frozen=True)
class ReferencePacket:
    schema_version: int
    rig_id: str
    robot_model_sha256: str
    scene_manifest_sha256: str
    joint_names: tuple
    units: dict
    frame: str
    issued_from_sim_time: float
    generated_from_command_seq: int
    plan_id: str
    state_version: int
    future_timestamps: np.ndarray
    root_pose: np.ndarray
    body_joint_positions: np.ndarray
    body_joint_velocities: np.ndarray
    valid_until: float
    conditioning_mode: str
    conditioning_history_times_s: np.ndarray
    query_root_xyz_m: np.ndarray
    query_origin_soma_xyz: np.ndarray
    query_origin_frame: str
    morphology_root_z_offset_m: float
    clocks_ns: dict
    episode_id: int
    session_id: str

    def use_receipt(self, start_time_s, *, policy_action_ready_ns, first_motor_control_ns,
                    reference_offsets_s=None):
        offsets = np.arange(10)*.1 if reference_offsets_s is None else np.asarray(reference_offsets_s)
        requested = start_time_s+offsets
        if requested[0] < self.future_timestamps[0]-1e-8 or requested[-1] > self.valid_until+1e-8:
            raise ReferenceCoverageError("Policy use exceeds the ReferencePacket's actual future window")
        indices = np.searchsorted(self.future_timestamps, requested-1e-9)
        if not np.allclose(self.future_timestamps[indices], requested, atol=1e-8, rtol=0):
            raise ReferenceCoverageError("Policy use does not lie on the packet's real 50Hz reference grid")
        return dict(plan_id=self.plan_id, generated_from_command_seq=self.generated_from_command_seq,
                    episode_id=self.episode_id, session_id=self.session_id,
                    scene_manifest_sha256=self.scene_manifest_sha256,
                    state_version=self.state_version, used_time_window=self.future_timestamps[indices].tolist(),
                    policy_action_ready_ns=policy_action_ready_ns, first_motor_control_ns=first_motor_control_ns,
                    state_age_sim_s=start_time_s-self.issued_from_sim_time)


def next_controller_grid(time_s):
    nearest = round(time_s*50)/50
    return float(nearest if abs(time_s-nearest) <= 1e-8 else np.ceil(time_s*50)/50)


def calibrated_native_query_origin(actual_robot_xyz, root_z_offset_m):
    origin = np.array(mujoco_to_soma(actual_robot_xyz), dtype=float, copy=True)
    origin[1] -= root_z_offset_m
    return origin


def consumer_horizon_contract(settings):
    required_future = (.16 if settings.get("tracker_backend", "sonic") == "protomotions" else
                       float(SONIC_FUTURE_OFFSETS[-1])/SONIC_FPS)
    horizon = float(settings["consumer_horizon_s"])
    adoption_interval = float(settings.get('mb_adopt_interval_s', settings['mb_replan_interval_s']))
    if not np.isfinite(adoption_interval) or adoption_interval < 0:
        raise ValueError('Reference adoption interval must be finite and nonnegative')
    minimum = required_future+max(settings['mb_replan_interval_s'], adoption_interval)+settings["max_plan_wall_age_s"]+POLICY_DT
    if not np.isfinite(horizon) or not np.isclose(horizon/POLICY_DT, round(horizon/POLICY_DT), atol=1e-8, rtol=0):
        raise ValueError("Consumer horizon must be a finite native-50Hz duration")
    if horizon < minimum-1e-9:
        raise ValueError(f"Consumer horizon {horizon}s is shorter than the declared scheduling requirement {minimum}s")
    return dict(consumer_horizon_s=horizon, required_future_s=required_future,
        replan_period_s=settings["mb_replan_interval_s"], adoption_interval_s=adoption_interval,
        maximum_plan_wall_age_s=settings["max_plan_wall_age_s"],
        controller_period_s=POLICY_DT, minimum_required_horizon_s=minimum,
        usable_window_s=horizon-required_future, dense_sample_count=round(horizon/POLICY_DT)+1)


def reference_adoption_due(update, active, time_s, interval_s):
    changed = active is None or (update.command['session_id'], update.command['command_seq']) != (
        active.command['session_id'], active.command['command_seq'])
    return changed or time_s-active.packet.issued_from_sim_time >= interval_s-1e-8


def continuous_reference_replan_due(active, revision, time_s, lookahead_s, wall_budget_s):
    # Prepare one native 24/30s segment ahead. Changed intents still replace the
    # uncommitted future immediately; this reserve is generated motion, not a hold.
    reserve = lookahead_s+24/30+2*(wall_budget_s+POLICY_DT)
    return (active is None or revision!=active.intent_revision
        or active.adapted_reference.trusted_until_s-time_s<=reserve+1e-8)


def _admission_limits(mode, settings):
    return ({name: settings[name] for name in ("max_generation_state_age_s", "max_plan_wall_age_s",
                                              "max_pending_intent_age_s")} if mode == "mb" else None)


def _scene_query_options(backend, proofs, scene):
    if backend == "native":
        return dict(scene_query_backend=backend, scene_query_proof=None)
    if backend not in ("warp_bvh_exact", "warp_bvh_sphere_exact"):
        raise ValueError("Unknown explicit MB scene-query backend")
    entry = proofs[scene.manifest.scene_id]
    path = Path(entry["path"]).resolve(strict=True)
    if sha256(path) != entry["sha256"]:
        raise ValueError("Pinned scene-query proof SHA256 mismatch")
    return dict(scene_query_backend=backend, scene_query_proof=path)


class FrozenMBProvider:
    """Real B generation plus Gauss calibration; no reference-as-execution substitution."""

    def __init__(self, generator, retargeter, *, conditioning_mode, estimate_actual_diagnostics,
                 robot_model_sha256, scene_manifest_sha256, rig_id, reference_history_builder,
                 consumer_horizon_s, constraint_solver=None, maximum_adoption_age_s=None, planning_lead_s=None,
                 native_blocks=2, required_future_s=.9, yaw_goal_lead_s=0., inference_rng_seed=None,
                 state_observer_limb_source='reference', reference_stream_policy=None):
        if type(native_blocks) is not int or native_blocks not in (1, 2):
            raise ValueError("Select one or two real native AR8 calls")
        self.native_blocks = native_blocks
        self.required_future_s = required_future_s
        self.inference_rng_seed = inference_rng_seed
        self.state_observer_limb_source = state_observer_limb_source
        if reference_stream_policy not in (None,'continuous_native_AR_v3'):
            raise ValueError('Unknown explicit reference stream policy')
        if reference_stream_policy is not None and conditioning_mode != 'B1':
            raise ValueError('Continuous motion memory uses explicit forecast-context conditioning')
        self.reference_stream_policy = reference_stream_policy
        self._reference_stream = None
        self.yaw_goal_lead_s = float(yaw_goal_lead_s)
        if not np.isfinite(self.yaw_goal_lead_s) or self.yaw_goal_lead_s < 0:
            raise ValueError("Yaw goal lead must be a declared finite nonnegative horizon")
        self._static_heap = None
        if conditioning_mode not in ("B1", "B2"):
            raise ValueError("Select explicit B1 reference history or B2 measured error-state feedback")
        self.generator, self.retargeter = generator, retargeter
        self.conditioning_mode = conditioning_mode
        self.estimate_actual_diagnostics = estimate_actual_diagnostics
        self.robot_model_sha256, self.scene_manifest_sha256 = robot_model_sha256, scene_manifest_sha256
        self.rig_id = rig_id
        self.reference_history_builder = reference_history_builder
        self.consumer_horizon_s = consumer_horizon_s
        self.constraint_solver = constraint_solver
        self.maximum_adoption_age_s = maximum_adoption_age_s
        self.planning_lead_s = planning_lead_s
        if constraint_solver is not None and (
                maximum_adoption_age_s is None or not np.isfinite(maximum_adoption_age_s)
                or planning_lead_s is None or not np.isfinite(planning_lead_s)
                or not 0 < planning_lead_s <= maximum_adoption_age_s
                or not np.isclose(planning_lead_s/POLICY_DT, round(planning_lead_s/POLICY_DT), atol=1e-8, rtol=0)):
            raise ValueError("Constrained references require an explicit native-grid lead within the admission window")
        self.request_evidence = None
        self.failure_evidence = None
        self.execution, self.preparation_report = "native", None
        self._history_observer = None
        self._state_observer = None
        self._scene_adapters = OrderedDict([(scene_manifest_sha256, generator)])
        self._scene_preparations = {}
        self._scene_cache_capacity = 1
        self._scene_query_backend, self._scene_query_proofs = "native", {}

    @classmethod
    def load(cls, settings, *, canonical_scene):
        execution = settings["mb_execution"]
        if execution not in ("native", "trace", "compiled", "full_plan"):
            raise ValueError("Explicit native, trace, compiled, or full_plan MB execution is required")
        configure_thread_schedule(settings["clock"])
        capacity = settings["scene_cache_capacity"] if "scene_cache_capacity" in settings else 1
        if type(capacity) is not int or not 1 <= capacity <= 2:
            raise ValueError("MB scene cache must explicitly fit one or two scene slots")
        consumer_contract = consumer_horizon_contract(settings)
        # Check the explicit calibration asset before allocating any motion model.
        calibration = Path(settings["calibration_path"]).resolve(strict=True)
        if sha256(calibration) != settings["calibration_sha256"]:
            raise ValueError("MB calibration asset SHA256 mismatch")
        from motionbricks.data.native_mb_data import build_native_representation
        from .mb_reference import (TRAINED_BUNDLE_SHA256, TRAINED_WEIGHTS_SHA256,
                                   TrainedMBReferenceAdapter)
        from .retarget import CalibratedSomaG1Adapter

        if (settings["mb_artifact_sha256"] != TRAINED_WEIGHTS_SHA256
                or settings["mb_bundle_sha256"] != TRAINED_BUNDLE_SHA256):
            raise ValueError("Explicit MB weight/bundle pins differ from the admitted trained pair")
        retargeter = CalibratedSomaG1Adapter.load(calibration, robot_model_path=settings["robot_model"],
            representation=build_native_representation(),
            limb_rotation_source=settings.get('retarget_limb_rotation_source', 'markers'))
        constraint_options = {}
        if "reference_constraints" in settings:
            from .constrained_reference import ConstrainedReferenceSolver

            options = settings["reference_constraints"]
            if options["schema"] == "motionbricks.mb_sonic.constrained_reference.v1":
                solver = ConstrainedReferenceSolver(retargeter.kinematics, iterations=options["iterations"])
            elif options["schema"] == "motionbricks.mb_sonic.trajectory_reference.v1":
                from .trajectory_reference import TrajectoryReferenceSolver

                if settings['reference_root_constraints_authorized'] is not True:
                    raise ValueError('Reference Root constraints require explicit authorization')
                solver = TrajectoryReferenceSolver(retargeter.kinematics, **{
                    key: options[key] for key in ('iterations', 'acceleration_time_s', 'jerk_time_s',
                                                  'overlap_time_s', 'overlap_weight')},
                    pose_task_frame=options.get('pose_task_frame','world'))
            else:
                raise ValueError("Unknown reference constraint policy")
            constraint_options = dict(constraint_solver=solver,
                maximum_adoption_age_s=settings["max_generation_state_age_s"],
                planning_lead_s=options["planning_lead_s"])
        generator = TrainedMBReferenceAdapter.load(
            Path(settings["mb_artifact"]), Path(settings["mb_bundle"]), canonical_scene=canonical_scene,
            root_reference_acceleration_mps2=settings.get('root_reference_acceleration_mps2'),
            device=settings["mb_device"], **_scene_query_options(
                settings["scene_query_backend"] if "scene_query_backend" in settings else "native",
                settings["scene_query_proofs"] if "scene_query_proofs" in settings else {}, canonical_scene))
        provider = cls(generator, retargeter, conditioning_mode=settings["mb_conditioning_mode"],
                   estimate_actual_diagnostics=settings["estimate_actual_diagnostics"],
                   robot_model_sha256=sha256(settings["robot_model"]),
                   scene_manifest_sha256=sha256(canonical_scene.manifest_path),
                   rig_id=ET.parse(settings["robot_model"]).getroot().attrib["model"],
                   reference_history_builder=None,
                   consumer_horizon_s=consumer_contract["consumer_horizon_s"],
                   native_blocks=settings.get("mb_native_blocks", 2),
                   required_future_s=consumer_contract["required_future_s"],
                   inference_rng_seed=settings.get('inference_rng_seed'),
                   state_observer_limb_source=settings.get('state_observer_limb_source', 'reference'),
                   reference_stream_policy=settings.get('reference_stream_policy'),
                   yaw_goal_lead_s=settings.get("yaw_goal_lead_s", 0.), **constraint_options)
        provider.execution = execution
        if settings.get("heap_strategy") == "freeze_static":
            provider._static_heap = StaticHeap()
        provider._scene_cache_capacity = capacity
        provider._scene_query_backend = settings["scene_query_backend"] if "scene_query_backend" in settings else "native"
        provider._scene_query_proofs = copy.deepcopy(settings["scene_query_proofs"]) if "scene_query_proofs" in settings else {}
        return provider

    def bind_scene(self, scene):
        from .mb_reference import TrainedMBReferenceAdapter

        if self._static_heap is not None:
            self._static_heap.release()
        key = sha256(scene.manifest_path)
        if key in self._scene_adapters:
            adapter = self._scene_adapters.pop(key)
            adapter.activate_scene()
        else:
            evicted = False
            while len(self._scene_adapters) >= self._scene_cache_capacity:
                old_key, old = self._scene_adapters.popitem(last=False)
                old.release_scene()
                self._scene_preparations.pop(old_key, None)
                del old
                evicted = True
            if evicted:
                import gc
                import torch

                # Scene switching is quiescent; collect bound-method graph cycles before allocation.
                gc.collect()
                torch.cuda.empty_cache()
            adapter = TrainedMBReferenceAdapter(self.generator.controller, canonical_scene=scene,
                identity=self.generator.identity,
                **_scene_query_options(self._scene_query_backend, self._scene_query_proofs, scene))
            if adapter.controller is not self.generator.controller:
                raise ValueError("A scene switch must retain the same controller and original model weights")
        self._scene_adapters[key] = adapter
        self.generator, self.scene_manifest_sha256 = adapter, key
        self.preparation_report = None

    def cache_description(self):
        import torch

        device = torch.device(self.generator.controller.inference.device)
        allocator = (dict(device=str(device), memory_allocated_bytes=torch.cuda.memory_allocated(device),
            memory_reserved_bytes=torch.cuda.memory_reserved(device), captured_at_monotonic_ns=perf_counter_ns(),
            scope="this_process_pytorch_allocator; excludes Warp/context/other_processes")
            if device.type == "cuda" else None)
        return dict(capacity=self._scene_cache_capacity, shared_controller=True,
            manifests=list(self._scene_adapters),
            accounted_cuda_torch_storage_bytes=sum(adapter.accounted_bytes for adapter in self._scene_adapters.values()),
            storage_scope="unique scene/field Torch storages only; excludes graph arenas/JIT constants/Warp BVH",
            torch_cuda_allocator=allocator,
            warp_bvh_bytes=None, graph_arena_bytes=None, graph_constant_bytes=None)

    def prepare(self, history, *, actual_root_xyz_m, desired_heading_rad, initial_qpos=None):
        """Explicit deployment initialization/parity, never a consumed joystick command."""
        self.preparation_report = None
        if self.inference_rng_seed is not None:
            import random
            import torch

            if type(self.inference_rng_seed) is not int or self.inference_rng_seed < 0:
                raise ValueError('Inference RNG seed must be a nonnegative integer')
            random.seed(self.inference_rng_seed)
            np.random.seed(self.inference_rng_seed)
            torch.manual_seed(self.inference_rng_seed)
        if self._static_heap is not None:
            self._static_heap.release()
        from .mb_reference import prepare_reference_history_b1, reference_history_b1, world_xy_command_to_soma

        prepared_now = self._history_observer is None
        if prepared_now:
            self._history_observer = prepare_reference_history_b1(self.retargeter.rep, history.features)
            self.reference_history_builder = partial(reference_history_b1, observer=self._history_observer)
        observer_report = dict(cpu_history_observer=copy.deepcopy(self._history_observer.preparation),
                               cpu_history_observer_prepared_now=prepared_now)
        if self.reference_stream_policy is not None:
            from .reference_stream import ContinuousReferenceStream

            self._reference_stream = ContinuousReferenceStream(self)
            observer_report['reference_stream'] = self._reference_stream.schema
        if self.conditioning_mode == 'B2':
            from .state_feedback import FixedPairStateObserver

            self._state_observer = FixedPairStateObserver(history, initial_qpos, self.retargeter,
                limb_source=self.state_observer_limb_source)
            observer_report['state_observer'] = self._state_observer.schema
            observer_report['state_observer_limb_source'] = self.state_observer_limb_source
        if self.scene_manifest_sha256 in self._scene_preparations:
            self.preparation_report = dict(copy.deepcopy(self._scene_preparations[self.scene_manifest_sha256]),
                                           **observer_report, scene_preparation_cache_hit=True)
            return self._freeze_prepared_heap()
        if self.execution == "native":
            self.preparation_report = dict(execution="native", conditioning_mode=self.conditioning_mode,
                                           **observer_report, scene_preparation_cache_hit=False)
            self._scene_preparations[self.scene_manifest_sha256] = copy.deepcopy(self.preparation_report)
            return self._freeze_prepared_heap()

        velocity, heading = world_xy_command_to_soma([0., 0.], desired_heading_rad)
        query_origin = calibrated_native_query_origin(actual_root_xyz_m, self.retargeter.root_z_offset_m)
        plan_options = dict(plan_blocks=self.native_blocks) if self.native_blocks != 2 else {}
        neural_report = self.generator.prepare_compiled(history,
            velocity_world_xz_mps=velocity, heading_world_rad=heading, execution=self.execution,
            conditioning_mode=self.conditioning_mode, query_origin_soma_xyz=query_origin, **plan_options)
        self.preparation_report = dict(neural_report, **observer_report, scene_preparation_cache_hit=False)
        self._scene_preparations[self.scene_manifest_sha256] = copy.deepcopy(self.preparation_report)
        return self._freeze_prepared_heap()

    def _freeze_prepared_heap(self):
        if self._static_heap is not None:
            self.preparation_report["static_heap"] = self._static_heap.prepare()
        return self.preparation_report

    def propose(self, snapshot, *, command, intent_revision, requested_at_ns, current_sim_time):
        if self._reference_stream is not None:
            return self._reference_stream.propose(snapshot,command=command,intent_revision=intent_revision,
                requested_at_ns=requested_at_ns,current_sim_time=current_sim_time)
        from .mb_reference import world_xy_command_to_soma

        started = perf_counter_ns()
        self.request_evidence = None
        self.failure_evidence = None
        estimated_actual = None
        if self.estimate_actual_diagnostics:
            estimated_actual = self.retargeter.estimate_history(
                snapshot.executed_history, end_time_s=snapshot.history_time_s)
        history = snapshot.reference_history
        if self.conditioning_mode == "B2":
            history = self._state_observer.history(snapshot.executed_history,
                end_time_s=snapshot.history_time_s, observer=self._history_observer)
        estimated = perf_counter_ns()
        expected_kinds = (("actual_g1_error_state",) if self.conditioning_mode == "B2"
                          else ("source_seed", "legacy_reference_history"))
        if history is None or history.provenance["kind"] not in expected_kinds:
            raise ValueError("MB conditioning history does not match the declared baseline")
        terminal_heading = _angle_delta(snapshot.derived_heading_rad+float(command["yaw_rate_radps"])*self.yaw_goal_lead_s, 0.)
        velocity, heading = world_xy_command_to_soma(command["velocity_xy_mps"], terminal_heading)
        if self.preparation_report is None:
            raise RuntimeError("Explicit MB deployment preparation must complete before online generation")
        query_origin = calibrated_native_query_origin(snapshot.actual_root_xyz_m, self.retargeter.root_z_offset_m)
        proposal = self.generator.generate(history, velocity_world_xz_mps=velocity,
                                            heading_world_rad=heading,
                                            command_id=str(command["command_id"]), blocks=self.native_blocks,
                                            conditioning_mode=self.conditioning_mode, query_origin_soma_xyz=query_origin)
        generated = perf_counter_ns()
        if self.constraint_solver is None:
            consumer_start = next_controller_grid(current_sim_time())
        else:
            lead = 0. if snapshot.previous_reference_packet is None else self.planning_lead_s
            consumer_start = next_controller_grid(snapshot.simulation_time_s+lead)
        consumer_end = consumer_start+self.consumer_horizon_s
        self.request_evidence = dict(proposal=proposal, conditioning_history=history,
            command=copy.deepcopy(command), snapshot=snapshot, consumer_start_s=consumer_start,
            consumer_end_s=consumer_end, consumer_horizon_s=self.consumer_horizon_s, requested_at_ns=requested_at_ns)
        if consumer_end > proposal.source.trusted_until_s+1e-9:
            raise ReferenceCoverageError("The explicit consumer horizon exceeds genuine MB future coverage")
        constraint_options = {}
        if self.constraint_solver is not None:
            previous = snapshot.previous_reference_packet
            boundary = None
            if previous is not None:
                if (previous.robot_model_sha256 != self.robot_model_sha256
                        or previous.scene_manifest_sha256 != self.scene_manifest_sha256
                        or previous.episode_id != command["episode_id"]
                        or previous.session_id != command["session_id"]):
                    raise ValueError("Reference constraint boundary belongs to a different robot/session")
                boundary = dict(timestamps_s=previous.future_timestamps,
                    joint_positions=previous.body_joint_positions, plan_id=previous.plan_id,
                    adoption_until_s=snapshot.simulation_time_s+self.maximum_adoption_age_s)
                if self.constraint_solver.identity['schema'] == 'motionbricks.mb_sonic.trajectory_reference.v1':
                    boundary['root_pose'] = previous.root_pose
            elif self.constraint_solver.identity['schema'] == 'motionbricks.mb_sonic.trajectory_reference.v1':
                initial = snapshot.executed_history.samples[-1]
                boundary = dict(initial_qpos=initial['qpos'], initial_qvel=initial['qvel'],
                                initial_time_s=initial['timestamp_s'], plan_id='measured_startup_state')
            constraint_options = dict(constraints=self.constraint_solver, boundary=boundary)
        if self.required_future_s != .9:
            constraint_options["required_future_s"] = self.required_future_s
        reference = self.retargeter.retarget_packet(proposal.source, start_time_s=consumer_start,
            consumer_horizon_s=self.consumer_horizon_s, **constraint_options)
        ready = perf_counter_ns()
        clocks = dict(requested=requested_at_ns, provider_started=started, history_ready=estimated,
                      mb_ready=generated, reference_ready=ready)
        packet = ReferencePacket(2, self.rig_id, self.robot_model_sha256, self.scene_manifest_sha256,
            BODY_JOINT_NAMES, dict(length="m", time="s", angle="rad"), "mujoco_world_z_up",
            snapshot.simulation_time_s, command["command_seq"], f"{intent_revision}:{requested_at_ns}",
            snapshot.state_version, reference.timestamps_s, reference.qpos[:, :7],
            reference.joint_pos, reference.joint_vel, reference.trusted_until_s, self.conditioning_mode,
            history.timestamps_s, snapshot.actual_root_xyz_m.copy(), query_origin, "soma_world_y_up",
            self.retargeter.root_z_offset_m, clocks, command["episode_id"], command["session_id"])
        return MBReferenceUpdate(intent_revision=intent_revision, command=copy.deepcopy(command),
            history_time_s=snapshot.history_time_s, derived_heading_rad=snapshot.derived_heading_rad,
            conditioning_history=history, estimated_actual_history=estimated_actual, proposal=proposal,
            adapted_reference=reference,
            timing_ns=dict(history_estimation=estimated-started, mb_generation=generated-estimated,
                           retarget=ready-generated, provider_total=ready-started), clocks_ns=clocks, packet=packet)


class OnlineMBWorker:
    """Native-control-boundary dispatch with one in-flight request and latest result."""

    def __init__(self, provider, intents, history_snapshot, recorder, *, numeric_writer,
                 simulation_time, replan_interval_s,
                 max_state_age_s, max_plan_wall_age_s, required_future_s=.9):
        if replan_interval_s < POLICY_DT:
            raise ValueError("Periodic MB replans cannot exceed the native 50Hz request cap")
        self._replan_ticks = round(replan_interval_s/POLICY_DT)
        if not np.isclose(self._replan_ticks*POLICY_DT, replan_interval_s, atol=1e-9, rtol=0):
            raise ValueError("Periodic MB replanning must use whole native control ticks")
        self.provider, self.intents, self.history_snapshot = provider, intents, history_snapshot
        self.simulation_time = simulation_time
        self.recorder, self.replan_interval_s = recorder, replan_interval_s
        self.numeric_writer = numeric_writer
        self.max_state_age_s, self.max_plan_wall_age_s = max_state_age_s, max_plan_wall_age_s
        self.required_future_s = required_future_s
        if max_state_age_s <= 0 or max_plan_wall_age_s <= 0:
            raise ValueError("Positive snapshot and plan wall-age bounds are required")
        self._wake, self._stop = Event(), Event()
        self._lock = Lock()
        self._ready, self._failure, self._thread = None, None, None
        self._published_ns = None
        self._last_request_ns = None
        self._last_attempt_ns = None
        self._last_dispatch_tick, self._last_control_tick = None, None
        self._pending, self._busy, self._dispatch = False, False, None

    def start(self):
        self._thread = Thread(target=self._run, name="motionbricks-reference", daemon=False)
        self._thread.start()

    def request(self):
        with self._lock:
            self._pending = True

    def control_tick(self, current_time_s, *, intent, initial_wait=False):
        """Only the physics owner can dispatch; held startup keeps the 50Hz wall cap."""
        tick = round(current_time_s/POLICY_DT)
        if not np.isclose(current_time_s, tick*POLICY_DT, atol=1e-8, rtol=0):
            raise ValueError("MB dispatch requires an actual native control boundary")
        with self._lock:
            if self._last_control_tick is not None and tick < self._last_control_tick:
                raise ValueError("MB control ticks cannot move backwards without a new worker")
            self._last_control_tick = tick
            if self._last_dispatch_tick is None or tick-self._last_dispatch_tick >= self._replan_ticks:
                self._pending = True
            if self._stop.is_set() or self._busy or self._ready is not None or not self._pending:
                return False
            requested_ns = perf_counter_ns()
            if (self._last_dispatch_tick == tick and not initial_wait) or (
                    initial_wait and self._last_attempt_ns is not None
                    and requested_ns-self._last_attempt_ns < round(POLICY_DT*1e9)):
                return False
            self._busy, self._pending = True, False
            self._last_dispatch_tick, self._last_attempt_ns = tick, requested_ns
        revision, command = intent
        history, capture_error = None, None
        if command is not None:
            try:
                history = self.history_snapshot()
            except Exception as error:
                capture_error = error
        history_ready_ns = perf_counter_ns()
        with self._lock:
            if command is None or self._stop.is_set():
                self._busy = False
                return False
            self._dispatch = dict(revision=revision, command=command, history=history,
                capture_error=capture_error, requested_ns=requested_ns,
                history_ready_ns=history_ready_ns, control_tick=tick, initial_wait=initial_wait)
        self._wake.set()
        return True

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join()

    def take_ready(self, current_time_s):
        self.numeric_writer.check()
        with self._lock:
            if self._failure is not None:
                raise RuntimeError(self._failure)
            future = self._ready
            if (future is not None and current_time_s < future.adapted_reference.timestamps_s[0]-1e-8
                    and round(current_time_s*1e9)-round(future.packet.issued_from_sim_time*1e9)
                    <= round(self.max_state_age_s*1e9)
                    and perf_counter_ns()-future.clocks_ns["requested"] <= round(self.max_plan_wall_age_s*1e9)):
                return None
            update, self._ready = self._ready, None
            published_ns = self._published_ns
        if update is None:
            return None
        clocks = dict(update.clocks_ns, reference_published=published_ns)
        update = replace(update, clocks_ns=clocks, packet=replace(update.packet, clocks_ns=clocks))
        # Accumulated physics ticks must not turn an exact 100ms age into a false overrun.
        state_age_ns = round(current_time_s*1e9)-round(update.packet.issued_from_sim_time*1e9)
        reason = None
        if state_age_ns > round(self.max_state_age_s*1e9):
            reason = "maximum_generation_snapshot_age_exceeded"
        elif perf_counter_ns()-update.clocks_ns["requested"] > self.max_plan_wall_age_s*1e9:
            reason = "maximum_plan_wall_age_exceeded"
        elif (current_time_s < update.adapted_reference.timestamps_s[0]-1e-8 or
              current_time_s > min(update.packet.valid_until, update.adapted_reference.timestamps_s[-1])
              -self.required_future_s+1e-8):
            reason = "trusted_future_coverage_expired"
        if reason:
            self.recorder.write(dict(kind="mb_result_discarded", reason=reason,
                                     command_id=update.command["command_id"],
                                     intent_revision=update.intent_revision, current_time_s=current_time_s,
                                     ready_deadline_ns=update.clocks_ns["requested"]+round(self.max_plan_wall_age_s*1e9),
                                     clocks_ns=update.clocks_ns, timing_ns=update.timing_ns))
            self.request()
            return None
        return update

    def _run(self):
        try:
            while not self._stop.is_set():
                self._wake.wait()
                self._wake.clear()
                if self._stop.is_set():
                    return
                with self._lock:
                    dispatch, self._dispatch = self._dispatch, None
                if dispatch is None:
                    continue
                try:
                    self._process_dispatch(dispatch)
                finally:
                    with self._lock:
                        self._busy = False
        except Exception as error:
            with self._lock:
                self._failure = f"{type(error).__name__}: {error}"
            self.recorder.write(dict(kind="mb_provider_failure", error=self._failure,
                                     planner_process_failure=self.provider.failure_evidence,
                                     monotonic_ns=perf_counter_ns()))
            evidence = self.provider.request_evidence
            if evidence is not None and not isinstance(error, NumericLogBackpressure):
                self.numeric_writer.submit(f"failed:{evidence['requested_at_ns']}",
                    dict(kind="mb_failed_request_evidence",
                    raw_source=evidence["proposal"].source,
                    raw_blocks=evidence["proposal"].raw_blocks,
                    conditioning_history=evidence["conditioning_history"],
                    raw_command=evidence["command"], consumer_horizon_s=evidence["consumer_horizon_s"],
                    planner_process_failure=self.provider.failure_evidence,
                    consumer_start_s=evidence["consumer_start_s"], consumer_end_s=evidence["consumer_end_s"],
                    state_version=evidence["snapshot"].state_version,
                    issued_from_sim_time=evidence["snapshot"].simulation_time_s,
                    error=self._failure))

    def _process_dispatch(self, dispatch):
        worker_started_ns = perf_counter_ns()
        revision, command, history = dispatch["revision"], dispatch["command"], dispatch["history"]
        requested_ns, history_ready_ns = dispatch["requested_ns"], dispatch["history_ready_ns"]
        snapshot_work_ns, dispatch_wait_ns = history_ready_ns-requested_ns, worker_started_ns-history_ready_ns
        self.provider.request_evidence = None
        self.numeric_writer.check()
        try:
            if dispatch["capture_error"] is not None:
                raise dispatch["capture_error"]
            if history is None or history.integrated_intent_revision != revision:
                self.request()
                return
            self._last_request_ns = requested_ns
            update = self.provider.propose(history, command=command,
                intent_revision=revision, requested_at_ns=requested_ns, current_sim_time=self.simulation_time)
        except Exception as request_error:
            latest_revision, _ = self.intents.latest()
            if latest_revision == revision and not self._stop.is_set():
                raise
            evidence = self.provider.request_evidence
            record = dict(kind="obsolete_mb_request_failure", intent_revision=revision,
                latest_intent_revision=latest_revision, command=command,
                state_version=history.state_version if history is not None else None,
                issued_from_sim_time=history.simulation_time_s if history is not None else None,
                error=f"{type(request_error).__name__}:{request_error}",
                planner_process_failure=self.provider.failure_evidence,
                stopped=self._stop.is_set(), requested_at_ns=requested_ns)
            if evidence is not None:
                record.update(raw_source=evidence["proposal"].source,
                              raw_blocks=evidence["proposal"].raw_blocks,
                              conditioning_history=evidence["conditioning_history"])
            self.numeric_writer.submit(f"obsolete_failure:{requested_ns}", record)
            self.recorder.write(dict(kind="mb_result_discarded", reason="obsolete_request_failed",
                intent_revision=revision, latest_intent_revision=latest_revision, error=record["error"]))
            if not self._stop.is_set():
                self.request()
            return
        clocks = dict(update.clocks_ns, snapshot_started=requested_ns, snapshot_ready=history_ready_ns,
            snapshot_captured=history.captured_at_monotonic_ns, worker_started=worker_started_ns,
            input_received=command["server_received_ns"])
        update = replace(update,
            timing_ns=dict(update.timing_ns, history_snapshot=snapshot_work_ns,
                history_snapshot_stages=history.timing_ns, worker_dispatch_wait=dispatch_wait_ns,
                provider_total=update.timing_ns["provider_total"]+snapshot_work_ns+dispatch_wait_ns),
            clocks_ns=clocks, packet=replace(update.packet, clocks_ns=clocks))
        latest_revision, _ = self.intents.latest()
        published_ns, replaced_plan_id = None, None
        if not self._stop.is_set():
            with self._lock:
                if self._ready is not None:
                    replaced_plan_id = self._ready.packet.plan_id
                self._ready = update
                self._published_ns = perf_counter_ns()
                published_ns = self._published_ns
        # A newer input schedules the next solve; it does not invalidate this fresh solve.
        # Publication precedes all large recording copies.
        logged_clocks = dict(update.clocks_ns, reference_published=published_ns)
        self.numeric_writer.submit(update.packet.plan_id, dict(
            kind="mb_raw_and_adapted_reference", plan_id=update.packet.plan_id,
            intent_revision=revision, command=command,
            packet=replace(update.packet, clocks_ns=logged_clocks),
            conditioning_history=update.conditioning_history,
            estimated_actual_history_diagnostic=update.estimated_actual_history,
            raw_source=update.proposal.source, raw_blocks=update.proposal.raw_blocks,
            adapted_reference=update.adapted_reference,
            snapshot_time_s=history.history_time_s, snapshot_state_version=history.state_version,
            snapshot_monotonic_ns=history.captured_at_monotonic_ns,
            control_tick=dispatch["control_tick"], initial_reference_wait=dispatch["initial_wait"],
            clocks_ns=logged_clocks, timing_ns=update.timing_ns, proposal_timing_ns=update.proposal.timing_ns,
            consumer_horizon_s=float(update.packet.future_timestamps[-1]-update.packet.future_timestamps[0]),
            replaced_ready_plan_id=replaced_plan_id,
            publication_status="published" if published_ns is not None else "not_published"))
        if latest_revision != revision:
            self.recorder.write(dict(kind="mb_newer_intent_pending",
                published_plan_id=update.packet.plan_id, intent_revision=revision,
                latest_intent_revision=latest_revision))
            self.request()


@dataclass(frozen=True)
class TrackerObservation:
    time_s: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    base_angular_velocity: np.ndarray

    def __post_init__(self):
        if not np.isfinite(self.time_s):
            raise ValueError("Tracker observation time must be finite")
        for name, shape in (("joint_positions", (29,)), ("joint_velocities", (29,)),
                            ("base_angular_velocity", (3,))):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Tracker observation must contain actual finite joint/pelvis measurements")
            value.flags.writeable = False
            object.__setattr__(self, name, value)


class CurrentTrackerObservation:
    def __init__(self):
        self.entries = deque(maxlen=1)

    def append(self, observation):
        if self.entries and not np.isclose(observation.time_s-self.entries[-1].time_s,
                                           POLICY_DT, atol=1e-8, rtol=0):
            raise ValueError("Tracker proprioception must advance at exactly50Hz")
        self.entries.append(observation)


def _load_tracker_policy(settings):
    backend = settings.get("tracker_backend", "sonic")
    if backend == "protomotions":
        from .protomotions_policy import FrozenProtoMotionsPolicy

        config = settings["protomotions_policy"]
        metadata = Path(config["metadata_path"])
        digest = sha256(metadata)
        if digest != config["metadata_sha256"]:
            raise ValueError("ProtoMotions deployment metadata differs from its explicit pin")
        policy = FrozenProtoMotionsPolicy(config["onnx_path"], metadata,
            onnx_sha256=config["onnx_sha256"], metadata_sha256=config["metadata_sha256"],
            threads=settings["policy_threads"])
        return policy, metadata, digest
    if backend != "sonic":
        raise ValueError("Unknown explicit tracker backend")
    directory = Path(settings["policy_dir"])
    metadata = directory/"observation_config.yaml"
    digest = sha256(metadata)
    if digest != settings["observation_config_sha256"]:
        raise ValueError("Lab observation_config.yaml does not match the pinned policy pair")
    policy = FrozenSonicPolicy(directory/"model_encoder.onnx", directory/"model_decoder.onnx", metadata,
        encoder_sha256=settings["encoder_sha256"], decoder_sha256=settings["decoder_sha256"],
        threads=settings["policy_threads"])
    return policy, metadata, digest


class SonicPhysicsRuntime:
    """Use the very same compiled model as scene queries and the renderer.

    scene is the canonical SceneManifest binding, not a separate scene spec.
    The caller owns task-specific failure criteria (including crawling support).
    """

    def __init__(self, scene, model, policy, *, physics_dt=0.005, robot_model_sha256=None,
                 tracker_backend="sonic",
                 physics_execution="native",
                 warmup_control_ticks=9,
                 contact_layout="tuple",
                 failure_evaluator: Callable[[PhysicsState], str | None] | None = None):
        if contact_layout not in ("tuple", "batch"):
            raise ValueError("Unknown explicit contact storage layout")
        self.contact_layout = contact_layout
        if physics_execution not in ("native", "cached_forward_v1"):
            raise ValueError("Unknown explicit physics execution path")
        self.physics_execution = physics_execution
        if tracker_backend not in ("sonic", "protomotions"):
            raise ValueError("Unknown explicit tracker backend")
        self.tracker_backend = tracker_backend
        self.is_protomotions = tracker_backend == "protomotions"
        if (type(warmup_control_ticks) is not int or not 0 <= warmup_control_ticks <= 9
                or (not self.is_protomotions and warmup_control_ticks != 9)):
            raise ValueError("SONIC requires nine warmup ticks; ProtoMotions can start from its measured initial state")
        self.warmup_control_ticks = warmup_control_ticks
        if self.is_protomotions and (policy.physics_dt != .001 or physics_dt != .001
                or policy.control_mode != "implicit_pd"
                or not np.array_equal(policy.reference_offsets_s, [0., .02, .04, .08, .16])):
            raise ValueError("ProtoMotions requires pinned1ms physics,50Hz control and exact future offsets")
        self.history_frames = policy.minimum_history_frames if self.is_protomotions else HISTORY_FRAMES
        if self.is_protomotions and self.history_frames != 1:
            raise ValueError("ProtoMotions uses one current observation, not a SONIC history stack")
        self.reference_lookahead_s = .16 if self.is_protomotions else float(SONIC_FUTURE_OFFSETS[-1])/SONIC_FPS
        substeps = round(POLICY_DT / physics_dt)
        if physics_dt <= 0 or substeps < 1 or not np.isclose(substeps * physics_dt, POLICY_DT):
            raise ValueError("Physics dt must divide the native 0.02s policy period")
        self.scene, self.model, self.policy = scene, model, policy
        self.robot_model_sha256 = robot_model_sha256
        self.scene_id = scene.manifest.scene_id
        self.manifest_sha256 = sha256(scene.manifest_path)
        self.substeps, self.physics_dt = substeps, physics_dt
        self.root_joint = model.joint("floating_base_joint").id
        if (model.jnt_type[self.root_joint] != mujoco.mjtJoint.mjJNT_FREE
                or model.jnt_qposadr[self.root_joint] != 0 or model.jnt_dofadr[self.root_joint] != 0):
            raise ValueError("Expected the official unactuated G1 free root")
        if model.nmocap or model.neq:
            raise ValueError("G1 runtime does not allow mocap or equality constraints")
        self.joint_ids = np.array([model.joint(name).id for name in G1_JOINT_NAMES])
        self.qpos_indices = model.jnt_qposadr[self.joint_ids].copy()
        self.qvel_indices = model.jnt_dofadr[self.joint_ids].copy()
        if np.any(model.jnt_type[self.joint_ids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("SONIC body controls require 29 hinge joints")
        self.actuator_ids = []
        for joint_id in self.joint_ids:
            ids = np.flatnonzero((model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                                 & (model.actuator_trnid[:, 0] == joint_id))
            if len(ids) != 1:
                raise ValueError("Each G1 body joint needs exactly one torque motor")
            self.actuator_ids.append(int(ids[0]))
        self.actuator_ids = np.asarray(self.actuator_ids)
        root_dofs = np.arange(6)
        for actuator in range(model.nu):
            joint_id = model.actuator_trnid[actuator, 0]
            if (model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT
                    or model.jnt_dofadr[joint_id] in root_dofs
                    or not np.array_equal(model.actuator_gear[actuator], [1, 0, 0, 0, 0, 0])
                    or model.actuator_dyntype[actuator] != mujoco.mjtDyn.mjDYN_NONE
                    or model.actuator_gaintype[actuator] != mujoco.mjtGain.mjGAIN_FIXED
                    or model.actuator_gainprm[actuator, 0] != 1
                    or model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_NONE):
                raise ValueError("Only unfiltered unit-gear joint torque motors are permitted")
        self.torque_lower = model.actuator_ctrlrange[self.actuator_ids, 0].copy()
        self.torque_upper = model.actuator_ctrlrange[self.actuator_ids, 1].copy()
        limited = model.jnt_actfrclimited[self.joint_ids].astype(bool)
        ranges = model.jnt_actfrcrange[self.joint_ids]
        self.torque_lower[limited] = np.maximum(self.torque_lower[limited], ranges[limited, 0])
        self.torque_upper[limited] = np.minimum(self.torque_upper[limited], ranges[limited, 1])
        if not np.all(model.actuator_ctrllimited[self.actuator_ids]):
            raise ValueError("Official G1 body motor torque limits are required")
        self.joint_limits = model.jnt_range[self.joint_ids].copy()
        self.kp, self.kd = KP, KD
        if self.is_protomotions:
            self.kp, self.kd = (np.array(value, dtype=float, copy=True) for value in (policy.kp, policy.kd))
            if (self.kp.shape != (29,) or self.kd.shape != (29,) or not np.isfinite(self.kp).all()
                    or not np.isfinite(self.kd).all() or not (self.kp > 0).all() or not (self.kd > 0).all()):
                raise ValueError("ProtoMotions requires its own finite positive named29joint PD gains")
            force_limited = model.actuator_forcelimited[self.actuator_ids].astype(bool)
            force_ranges = model.actuator_forcerange[self.actuator_ids]
            self.torque_lower[force_limited] = np.maximum(self.torque_lower[force_limited], force_ranges[force_limited, 0])
            self.torque_upper[force_limited] = np.minimum(self.torque_upper[force_limited], force_ranges[force_limited, 1])
            if not (self.torque_lower < self.torque_upper).all():
                raise ValueError("Original controlled-motor force bounds must be nonempty")
            # Keep the cached original XML model unchanged across explicit episodes.
            model = copy.copy(model)
            self.model = model
            ids = self.actuator_ids
            model.actuator_gainprm[ids] = 0
            model.actuator_gainprm[ids, 0] = self.kp
            model.actuator_biastype[ids] = mujoco.mjtBias.mjBIAS_AFFINE
            model.actuator_biasprm[ids] = 0
            model.actuator_biasprm[ids, 1] = -self.kp
            model.actuator_biasprm[ids, 2] = -self.kd
            model.actuator_ctrllimited[ids] = False
            model.actuator_forcelimited[ids] = True
            model.actuator_forcerange[ids] = np.column_stack((self.torque_lower, self.torque_upper))
            model.jnt_stiffness[self.joint_ids] = 0
            model.dof_damping[self.qvel_indices] = 0
            model.dof_frictionloss[self.qvel_indices] = 0
            self.anchor_body_id = model.body(policy.anchor_body_name).id
        model.opt.timestep = physics_dt
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        self.data = mujoco.MjData(model)
        self.warning_names = tuple(mujoco.mjtWarning(index).name for index in range(len(self.data.warning)))
        self.history = CurrentTrackerObservation() if self.is_protomotions else ObservationHistory()
        self.failure_evaluator = failure_evaluator
        self.terminal_reason = None
        self.initialized = False
        self.heading_initialized = False
        if not self.is_protomotions:
            self.last_action_isaac = np.zeros(29)
        self.last_motor_control_ns = None
        self.root_velocity_task = None

    @property
    def needs_warmup(self):
        return (self.time_s < self._warmup_until_s-1e-9 if self.is_protomotions
                else len(self.history.entries) < self.history_frames)

    @property
    def control_metadata(self):
        return dict(tracker_backend=self.tracker_backend,
            mode="implicit_pd" if self.is_protomotions else "explicit_torque_pd",
            joint_names=list(G1_JOINT_NAMES), kp=self.kp.tolist(), kd=self.kd.tolist(),
            motor_force_lower=self.torque_lower.tolist(), motor_force_upper=self.torque_upper.tolist(),
            physics_dt_s=self.physics_dt, control_dt_s=POLICY_DT,
            timestamp_source='integer_native_physics_step;MuJoCo_data_time_unmodified',
            root_velocity_task=self.root_velocity_task.identity if self.root_velocity_task is not None else None,
            reference_lookahead_s=self.reference_lookahead_s, observation_frames=self.history_frames,
            warmup_duration_s=self.warmup_control_ticks*POLICY_DT,
            warmup_target_source="actual_time_zero_reset_joints" if self.is_protomotions else "official_sonic_default",
            warmup_joint_target=self.warmup_joint_target.tolist())

    @property
    def time_s(self):
        return float(round(self.data.time/self.physics_dt)*self.physics_dt)

    def reset(self, qpos, qvel):
        """The only state assignment path; call explicitly before an episode."""
        qpos, qvel = np.asarray(qpos, dtype=float), np.asarray(qvel, dtype=float)
        if qpos.shape != (self.model.nq,) or qvel.shape != (self.model.nv,):
            raise ValueError("Reset requires full model qpos and qvel")
        if (not np.isfinite(qpos).all() or not np.isfinite(qvel).all()
                or not np.isclose(np.linalg.norm(qpos[3:7]), 1, atol=1e-5)):
            raise ValueError("Reset state must be finite with a unit free-root quaternion")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.terminal_reason = None
        self.initialized = True
        self.heading_initialized = False
        self.history = CurrentTrackerObservation() if self.is_protomotions else ObservationHistory()
        if not self.is_protomotions:
            self.last_action_isaac = np.zeros(29)
        mujoco.mj_forward(self.model, self.data)
        self.warmup_joint_target = (self.data.qpos[self.qpos_indices].copy() if self.is_protomotions
                                    else DEFAULT_ANGLES.copy())
        self._warmup_until_s = self.time_s+self.warmup_control_ticks*POLICY_DT
        self._append_proprioception()
        return self.snapshot()

    def _append_proprioception(self):
        if self.is_protomotions:
            self.history.append(TrackerObservation(self.time_s, self.data.qpos[self.qpos_indices],
                self.data.qvel[self.qvel_indices], self.data.qvel[3:6]))
            return
        self.history.append(Proprioception(
            self.time_s, self.data.qpos[self.qpos_indices], self.data.qvel[self.qvel_indices],
            self.data.qpos[3:7], self.data.qvel[3:6], self.last_action_isaac))

    def snapshot(self):
        if self.contact_layout == "batch":
            contacts = capture_contacts(self.model, self.data)
        else:
            contact_rows = []
            for index in range(self.data.ncon):
                contact = self.data.contact[index]
                wrench = np.empty(6)
                mujoco.mj_contactForce(self.model, self.data, index, wrench)
                geoms = tuple(int(i) for i in contact.geom)
                contact_rows.append(ContactForce(
                    geoms, tuple(int(self.model.geom_bodyid[i]) for i in geoms), contact.pos.copy(),
                    contact.frame.reshape(3, 3).copy(), wrench, float(contact.dist)))
            contacts = tuple(contact_rows)
        ground_height, ground_known = self.scene.query_height_point(self.data.qpos[0], self.data.qpos[1])
        return PhysicsState(
            self.time_s, perf_counter_ns(), round(self.time_s/self.physics_dt),
            self.data.qpos.copy(), self.data.qvel.copy(),
            self.data.qpos[self.qpos_indices].copy(), self.data.qvel[self.qvel_indices].copy(),
            self.data.xpos.copy(), self.data.xquat.copy(), self.data.ctrl.copy(),
            self.data.actuator_force.copy(), self.data.qfrc_actuator.copy(),
            self.data.qfrc_actuator[:6].copy(),
            self.data.qfrc_applied.copy(), self.data.xfrc_applied.copy(),
            {name: dict(number=int(number), lastinfo=int(info))
             for name, number, info in zip(self.warning_names, self.data.warning.number, self.data.warning.lastinfo)},
            ground_known, ground_height, contacts,
            self.terminal_reason)

    def terminate(self, reason):
        if not reason:
            raise ValueError("Terminal failure must have a reason")
        self.terminal_reason = reason
        return self.snapshot()

    def _advance_pd(self, joint_target, *, substeps=None):
        if not self.initialized or self.terminal_reason:
            raise RuntimeError("Explicit reset is required before physics can advance")
        frames = []
        for _ in range(self.substeps if substeps is None else substeps):
            # Holding a joint target is not state driving: only motor torque enters physics.
            if self.is_protomotions:
                target = np.asarray(joint_target)
                if target.shape != (29,) or not np.isfinite(target).all():
                    raise ValueError("ProtoMotions requires finite29joint targets, without clipping")
                self.data.ctrl[self.actuator_ids] = target
            else:
                torque = (self.kp * (joint_target - self.data.qpos[self.qpos_indices])
                          - self.kd * self.data.qvel[self.qvel_indices])
                self.data.ctrl[self.actuator_ids] = np.clip(torque, self.torque_lower, self.torque_upper)
            self.last_motor_control_ns = perf_counter_ns()
            before = float(self.data.time)
            if self.physics_execution == "cached_forward_v1":
                cached_forward_step(self.model, self.data)
            else:
                mujoco.mj_step(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                self.terminal_reason = "nonfinite_physics_state"
            elif (not np.isfinite(self.data.qacc).all()
                    or not np.isfinite(self.data.qfrc_constraint).all()):
                self.terminal_reason = "nonfinite_physics_dynamics"
                raise FloatingPointError("MuJoCo returned nonfinite acceleration or constraint force")
            elif not np.isclose(float(self.data.time) - before, self.physics_dt, atol=1e-9, rtol=0):
                self.terminal_reason = "unexpected_physics_clock_change"
            else:
                warnings = [self.warning_names[index] for index in np.flatnonzero(self.data.warning.number)]
                if warnings:
                    self.terminal_reason = "mujoco_warning:" + ",".join(warnings)
            state = self.snapshot()
            if not self.terminal_reason and self.failure_evaluator is not None:
                reason = self.failure_evaluator(state)
                if reason:
                    state = self.terminate(reason)
            frames.append(state)
            if self.terminal_reason:
                break
        return tuple(frames)

    def warmup_history(self):
        """Explicit tracker-specific PD warmup; return all real substep frames."""
        frames = []
        while self.needs_warmup:
            frames.extend(self._advance_pd(self.warmup_joint_target))
            if self.terminal_reason:
                break
            self._append_proprioception()
        return tuple(frames)

    def step(self, reference, *, raw_reference, issued_at_monotonic_ns, phase):
        start = perf_counter_ns()
        if not self.initialized or self.terminal_reason:
            raise RuntimeError("Cannot run policy before explicit reset or after terminal failure")
        reference, command = self.begin_control(reference)
        inferred = perf_counter_ns()
        frames = self._advance_pd(command.joint_target)
        stepped = perf_counter_ns()
        if not self.terminal_reason:
            self._append_proprioception()
        finished = perf_counter_ns()
        return RuntimeTick(
            self.scene_id, reference.command_id, phase, float(reference.timestamps_s[0]),
            issued_at_monotonic_ns, raw_reference, reference, command, frames,
            dict(**command.timing_ns, policy_total=inferred-start, physics=stepped-inferred,
                 history=finished-stepped, chain_total=finished-start))

    def begin_control(self, reference):
        """Compute a 50Hz command without advancing physics; paced consumers own substeps."""
        if not self.initialized or self.terminal_reason:
            raise RuntimeError("Cannot run policy before explicit reset or after terminal failure")
        if reference.scene_id != self.scene_id:
            raise ValueError("Reference and physics must use the same canonical SceneManifest")
        if reference.provenance.get("kind") == "calibrated_soma_reference":
            declared = reference.provenance["calibration"]["robot_xml_sha256"]
            if not self.robot_model_sha256 or declared != self.robot_model_sha256:
                raise ValueError("Reference calibration and actual physics robot XML hashes differ")
        if self.is_protomotions:
            reference = self.policy.reference_window(reference, self.time_s)
            anchor = reference.body_names.index(self.policy.anchor_body_name)
            actual_anchor = self.data.xquat[self.anchor_body_id].copy()
            if not self.heading_initialized:
                self.policy.reset_heading(actual_anchor, reference.body_quat_w[0, anchor])
                self.heading_initialized = True
            return reference, self.policy.infer(reference, self.history, anchor_quat_wxyz=actual_anchor)

        reference = reference.sonic_window(self.time_s)
        if not self.heading_initialized:
            self.policy.reset_heading(self.data.qpos[3:7], reference.qpos[0, 3:7])
            self.heading_initialized = True
        command = self.policy.infer(reference, self.history)
        self.last_action_isaac = command.action_isaac.copy()
        return reference, command

    def apply_velocity_task(self, command, intent, heading_rad):
        target, report = self.root_velocity_task.solve(self.data, command.joint_target,
            velocity_xy_mps=intent['velocity_xy_mps'], yaw_rate_radps=intent['yaw_rate_radps'],
            heading_rad=heading_rad)
        self.policy.previous_target = target.copy()
        return replace(command, joint_target=target, processed_action_mujoco=target.copy(),
            execution_task=report, timing_ns=dict(command.timing_ns, root_velocity_task=report['timing_ns']['total']))


class SonicLabRuntime:
    """Live, independently paced physics. Official-reference mode is read-only."""

    def __init__(self, settings, *, allocation_id):
        self._tracker_backend = settings.get("tracker_backend", "sonic")
        if self._tracker_backend not in ("sonic", "protomotions"):
            raise ValueError("Unknown explicit tracker backend")
        if self._tracker_backend == "protomotions" and settings["physics_dt_s"] != .001:
            raise ValueError("ProtoMotions requires exactly1ms physics")
        self.mode = settings["mode"]
        if self.mode not in ("official_reference", "fixed_reference", "mb"):
            raise ValueError("Expected explicit official_reference, fixed_reference or MB mode")
        self._mb_planner_backend = settings.get("mb_planner_backend", "thread")
        if (self._mb_planner_backend not in ("thread", "spawn")
                or (self._mb_planner_backend == "spawn" and self.mode != "mb")):
            raise ValueError("Spawned MB planning must be explicitly selected for real MB mode")
        self.settings = settings
        self.hardware_execution_profile = 'normal_dynamic_gpu_power'
        if settings.get("physics_publication", "substeps") not in ("substeps", "control_ticks"):
            raise ValueError("Unknown explicit physics publication clock")
        self.run_mode = settings["run_mode"] if "run_mode" in settings else "finite"
        if self.run_mode not in ("finite", "interactive"):
            raise ValueError("Expected explicit finite or interactive run mode")
        if self.run_mode == "interactive":
            if (self.mode != "mb" or not 0 < settings["max_run_wall_s"] <= 300
                    or not 0 < settings["duration_s"] <= 300 or settings["display_hz"] != 30
                    or settings["physics_dt_s"] != (.001 if self._tracker_backend == "protomotions" else .005)
                    or "scene_seeds" not in settings):
                raise ValueError("Interactive mode requires scoped MB seeds, tracker-native physics/50/30Hz and at most300s")
        if settings["clock"]["strategy"] != "windows_event_wait_1ms":
            raise ValueError("An explicit windows_event_wait_1ms clock strategy is required")
        WindowsTimerResolution(settings["clock"]["timer_resolution_ms"])
        configure_thread_schedule(settings["clock"])
        self._mb_mode_label = ("mb_actual_g1_error_state_B2" if settings.get("mb_conditioning_mode") == "B2"
                               else "mb_legacy_reference_history_B1")
        if settings.get('reference_stream_policy') is not None:
            self._mb_mode_label = 'mb_native_continuation_actual_scene_feedback'
        if settings.get("mb_conditioning_mode") == "B2" and settings.get("warmup_control_ticks", 9) < 5:
            raise ValueError("B2 requires at least100ms of factual G1 warmup for four30Hz frames")
        if settings.get("mb_conditioning_mode") == "B2" and settings.get("contact_layout") != "batch":
            raise ValueError("B2 requires explicit columnar actual contacts")
        if settings.get("heap_strategy") not in (None, "freeze_static"):
            raise ValueError("Unknown explicit heap collection strategy")
        self._static_heap = StaticHeap() if settings.get("heap_strategy") == "freeze_static" else None
        self.allocation_id = allocation_id
        self._lock, self._stop_event = Lock(), Event()
        self._snapshot_subscription_lock = Lock()
        self._snapshot_subscriber = None
        self._begin_event = Event()
        self._thread = None
        self._provider, self._mb_worker, self._numeric_writer, self._tick_writer = None, None, None, None
        self._compression_executor = None
        self._recording_budget = None
        if "logging" in settings:
            log = settings["logging"]
            if ((self.run_mode != "interactive" and self.mode != 'fixed_reference')
                    or log["tick_format"] not in ("numeric_npz", "columnar_npz_v1")):
                raise ValueError("Binary full-tick logging is explicit interactive-MB mode only")
            if (not 1 <= log["tick_pending_records"] <= 16 or log["max_numeric_record_bytes"] <= 0
                    or log["max_json_record_bytes"] <= 0):
                raise ValueError("Interactive numeric logging requires bounded queues and record sizes")
            minimum_reserve = ((settings["raw_log_pending_plans"]+log["tick_pending_records"]+4)
                               * (log["max_numeric_record_bytes"]+log["max_json_record_bytes"])
                               +4*log["max_json_record_bytes"])
            if log["required_log_reserve_bytes"] < minimum_reserve:
                raise ValueError("Storage reserve must cover both bounded queues, in-flight and fatal-overflow records")
            self._recording_budget = RecordingBudget(log["max_recording_bytes"],
                log["required_log_reserve_bytes"], log["max_json_record_bytes"])
        self._assets = {}
        self._scene_cache = OrderedDict()
        self._scene_cache_capacity = settings["scene_cache_capacity"] if "scene_cache_capacity" in settings else 1
        self._scene_asset_limit = settings["scene_asset_cache_limit_bytes"] if "scene_asset_cache_limit_bytes" in settings else 134217728
        if (type(self._scene_cache_capacity) is not int or not 1 <= self._scene_cache_capacity <= 2
                or type(self._scene_asset_limit) is not int or self._scene_asset_limit <= 0):
            raise ValueError("Scene cache requires one/two slots and an explicit positive asset byte bound")
        self._latest = None
        self._failure = None
        self._errors = []
        self._episode = 0
        self._frame_id = 0
        self._recording_root = Path(settings["recording_dir"]).resolve()
        self._recording_root.mkdir(parents=True, exist_ok=False)
        self._catalog = []
        self._manifest_paths = {}
        enabled = set(settings["enabled_scene_ids"])
        labels = {}
        if "scene_catalog" in settings:
            catalog_path = Path(settings["scene_catalog"]["path"]).resolve(strict=True)
            if sha256(catalog_path) != settings["scene_catalog"]["sha256"]:
                raise ValueError("Pinned PFNN scene catalog changed")
            labels = {row["id"]: row["label"] for row in json.loads(catalog_path.read_text())["entries"]}
            if not enabled <= labels.keys():
                raise ValueError("Enabled scene IDs must belong to the actual PFNN catalog")
        for path in sorted(Path(settings["scene_root"]).glob("*/scene_manifest.json")):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            scene_id = manifest["scene_id"]
            available = scene_id in enabled
            self._manifest_paths[scene_id] = path
            self._catalog.append(dict(id=scene_id, label=labels[scene_id] if scene_id in labels else scene_id, available=available,
                                      unavailable_reason=None if available else
                                      "No enabled task-specific reference/initialization for this scene"))
        self._policy, self._observation_config, self._observation_config_sha256 = _load_tracker_policy(settings)
        try:
            self._load_episode(settings["initial_scene_id"])
        except Exception:
            if self._mb_planner_backend == "spawn" and self._provider is not None:
                self._provider.close()
            raise
        self._notify_snapshots()

    def _packed_mesh(self, vertices, faces):
        vertices = np.ascontiguousarray(vertices, dtype="<f4")
        indices = np.ascontiguousarray(faces, dtype="<u4").reshape(-1)
        packed = vertices.tobytes() + indices.tobytes()
        asset_id = hashlib.sha256(packed).hexdigest()
        self._assets[asset_id] = packed
        return dict(asset_id=asset_id, vertex_count=len(vertices), index_count=len(indices))

    def _geometry_config(self):
        model, scene = self.physics.model, self.physics.scene
        cached = self._scene_cache[self.physics.manifest_sha256]
        if cached["geometry"] is not None:
            self._assets, self._robot_geom_ids = cached["assets"], cached["geom_ids"]
            return cached["geometry"]
        self._assets = {}
        scene_geometries = []
        for mesh in scene.render_meshes():
            obj = next(row for row in scene.manifest.objects if row["object_id"] == mesh.object_id)
            scene_geometries.append(dict(id=mesh.object_id, kind="mesh", rgba=obj["rgba"],
                                          mesh=self._packed_mesh(mesh.vertices_m, mesh.faces),
                                          position=[0, 0, 0], quaternion_xyzw=[0, 0, 0, 1]))
        # Official group-1 visual meshes, with transforms from the live compiled model.
        self._robot_geom_ids = np.flatnonzero((model.geom_bodyid > 0) & (model.geom_group == 1))
        robot_geometries = []
        mesh_cache = {}
        for geom_id in self._robot_geom_ids:
            if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                kinds = {mujoco.mjtGeom.mjGEOM_SPHERE: "sphere", mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder"}
                kind = kinds[int(model.geom_type[geom_id])]
                robot_geometries.append(dict(id=int(geom_id), kind=kind,
                    rgba=model.geom_rgba[geom_id].tolist(), size=model.geom_size[geom_id].tolist()))
                continue
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id not in mesh_cache:
                va, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
                fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
                mesh_cache[mesh_id] = self._packed_mesh(model.mesh_vert[va:va+vn], model.mesh_face[fa:fa+fn])
            robot_geometries.append(dict(id=int(geom_id), kind="mesh", rgba=model.geom_rgba[geom_id].tolist(),
                                          mesh=mesh_cache[mesh_id]))
        cached.update(geometry=(scene_geometries, robot_geometries), assets=self._assets,
                      geom_ids=self._robot_geom_ids)
        while sum(sum(len(value) for value in row["assets"].values())
                  for row in self._scene_cache.values() if row["assets"] is not None) > self._scene_asset_limit:
            oldest = next(iter(self._scene_cache))
            if oldest == self.physics.manifest_sha256:
                self._assets.clear()
                cached.update(geometry=None, assets=None, geom_ids=None)
                raise MemoryError("Current full-resolution scene exceeds the explicit packed-asset cache byte bound")
            self._scene_cache.pop(oldest)
        return cached["geometry"]

    def _load_episode(self, scene_id):
        entry = next(row for row in self._catalog if row["id"] == scene_id)
        if not entry["available"]:
            raise ValueError(entry["unavailable_reason"])
        scene_key = sha256(self._manifest_paths[scene_id])
        if scene_key in self._scene_cache:
            cached = self._scene_cache.pop(scene_key)
        else:
            while len(self._scene_cache) >= self._scene_cache_capacity:
                self._scene_cache.popitem(last=False)
            binding = self.settings.get("robot_contact_pairs")
            flat = self.settings.get("flat_ground_primitive")
            if binding is None and flat is None:
                scene, model, robot_metadata = load_physics_scene(self._manifest_paths[scene_id], self.settings["robot_model"])
            else:
                scene, model, robot_metadata = load_physics_scene(self._manifest_paths[scene_id],
                    self.settings["robot_model"], contact_pair_binding=binding, flat_ground_primitive=flat)
            cached = dict(scene=scene, model=model, robot=robot_metadata, geometry=None, assets=None, geom_ids=None)
        self._scene_cache[scene_key] = cached
        scene, model, robot_metadata = cached["scene"], cached["model"], cached["robot"]
        if self.mode in ('mb', 'fixed_reference'):
            if robot_metadata["sha256"] != self.settings["robot_model_sha256"]:
                raise ValueError("Physics robot XML differs from the explicit MB runtime pin")
            if sha256(scene.manifest_path) != self.settings["scene_manifest_sha256"][scene_id]:
                raise ValueError("Canonical scene differs from the explicit MB runtime pin")
        self.source = (load_official_reference(Path(self.settings["reference_dir"]), scene_id=scene_id)
                       if self.mode == "official_reference" else None)
        if self.mode == 'fixed_reference':
            self.source = load_g1_reference(self.settings['fixed_reference']['path'],
                expected_sha256=self.settings['fixed_reference']['sha256'], scene_id=scene_id)
        if (self.settings["duration_s"] <= 0 or (self.source is not None and
                self.settings["duration_s"] + (.16 if self._tracker_backend == "protomotions" else .9)
                > self.source.trusted_until_s + 1e-9)):
            raise ValueError("Lab duration plus selected tracker lookahead exceeds actual official reference coverage")
        scene_geoms = {model.geom(row["geom_name"]).id for row in scene.manifest.objects}
        fall_bodies = {model.body(name).id for name in ("pelvis", "torso_link")}
        scene_geom_mask, fall_body_mask = np.zeros(model.ngeom, bool), np.zeros(model.nbody, bool)
        scene_geom_mask[list(scene_geoms)], fall_body_mask[list(fall_bodies)] = True, True

        def walking_failure(state):
            if self.settings.get("enforce_actual_joint_limits", False):
                margin = np.minimum(state.joint_positions-self.physics.joint_limits[:, 0],
                                    self.physics.joint_limits[:, 1]-state.joint_positions)
                if margin.min() < -1e-6:
                    return "actual_joint_limit:"+G1_JOINT_NAMES[int(margin.argmin())]
            if isinstance(state.contacts, ContactBatch):
                contacts = state.contacts
                a, b = contacts.geom_ids.T
                ba, bb = contacts.body_ids.T
                touching = ((scene_geom_mask[a] & fall_body_mask[bb])
                    | (scene_geom_mask[b] & fall_body_mask[ba]))
                if np.any(touching & (contacts.wrench_contact[:, 0] > 1)):
                    return "walking_task:pelvis_or_torso_scene_contact"
                return None
            for contact in state.contacts:
                if (contact.wrench_contact[0] > 1 and
                        ((contact.geom_ids[0] in scene_geoms and contact.body_ids[1] in fall_bodies) or
                         (contact.geom_ids[1] in scene_geoms and contact.body_ids[0] in fall_bodies))):
                    return "walking_task:pelvis_or_torso_scene_contact"
            return None

        self.physics = SonicPhysicsRuntime(scene, model, self._policy,
                                            physics_dt=self.settings["physics_dt_s"],
                                            tracker_backend=self._tracker_backend,
                                            physics_execution=self.settings.get("physics_execution", "native"),
                                            warmup_control_ticks=self.settings.get("warmup_control_ticks", 9),
                                            contact_layout=self.settings.get("contact_layout", "tuple"),
                                            robot_model_sha256=robot_metadata["sha256"],
                                            failure_evaluator=walking_failure if (self.mode != "mb"
                                                or self.settings.get("task") == "walking") else None)
        self._physics_numerics = configure_physics_numerics(self.physics.model, self.physics.physics_dt, self.settings)
        if 'execution_root_velocity_task' in self.settings:
            from .root_velocity_task import RootVelocityTask

            task_settings = dict(self.settings['execution_root_velocity_task'])
            if (self.mode != 'mb' or task_settings.pop('schema') != 'motionbricks.mb_sonic.root_velocity_task.v1'):
                raise ValueError('Root velocity task requires its explicit schema and live MB mode')
            self.physics.root_velocity_task = RootVelocityTask(self.physics, **task_settings)
        qpos, qvel = model.qpos0.copy(), np.zeros(model.nv)
        if self.mode != "mb":
            qpos[:7] = self.source.qpos[0, :7]
            qpos[self.physics.qpos_indices] = self.source.joint_pos[0]
            if self.mode == 'official_reference':
                qvel[:6], qvel[self.physics.qvel_indices] = self.source.qvel[0, :6], self.source.joint_vel[0]
            initialization_source = self.source.provenance
        else:
            from .mb_reference import decode_soma_reference

            if self._provider is None:
                if self._mb_planner_backend == "spawn":
                    from .planner_process import ProcessMBProvider

                    self._provider = ProcessMBProvider.load(self.settings, canonical_scene=scene)
                else:
                    self._provider = FrozenMBProvider.load(self.settings, canonical_scene=scene)
            else:
                self._provider.bind_scene(scene)
            self._seed_history = load_mb_seed_history(self.settings, scene)
            features = self._seed_history.features.copy()
            self._seed_source = decode_soma_reference(features, self._seed_history.timestamps_s,
                self._provider.retargeter.rep, scene_id=scene_id, command_id="explicit_initial_source_seed",
                trusted_until_s=0., provenance=dict(self._seed_history.provenance))
            initial_reference = self._provider.retargeter.retarget(self._seed_source, start_time_s=-.1)
            qpos[:7], qvel[:6] = initial_reference.qpos[-1, :7], initial_reference.qvel[-1, :6]
            qpos[self.physics.qpos_indices] = initial_reference.joint_pos[-1]
            qvel[self.physics.qvel_indices] = initial_reference.joint_vel[-1]
            initialization_source = self._seed_history.provenance
        initial = self.physics.reset(qpos, qvel)
        initial_heading = _root_heading(initial.qpos)
        if self.mode == "mb" and (initial_heading is None or not np.isfinite(initial_heading)):
            raise ValueError("The actual explicit MB seed reset requires an observable finite root heading")
        preparation_started_ns = perf_counter_ns()
        preparation = (self._provider.prepare(self._seed_history, actual_root_xyz_m=initial.qpos[:3],
                                              desired_heading_rad=_root_heading(initial.qpos), initial_qpos=initial.qpos)
                       if self.mode == "mb" else None)
        preparation_ns = perf_counter_ns()-preparation_started_ns
        self._episode += 1
        self._status = ("awaiting_explicit_start_"+self.mode+"_read_only" if self.mode != "mb"
                        else "awaiting_explicit_start_"+self._mb_mode_label)
        self._begin_event.clear()
        self._controller_steps, self._physics_steps = 0, 0
        self._source_clock_origin_s = None
        self._last_chain_ms = None
        self._wall_start_ns = None
        self._run_deadline_ns, self._budget_finished = None, False
        self._failure = None
        self._active_update, self._consumed_command = None, None
        self._consumed_intent_revision = None
        self._last_used_plan_id = None
        self._reference_cache, self._seed_clock_origin_s = None, None
        self._intents, self._integrated_intent_revision = IntentMailbox(), 0
        self._controller_session_id, self._last_intent_seq = None, 0
        self._yaw = YawIntentIntegrator(_root_heading(initial.qpos))
        self._executed = ExecutedHistory(scene_id, body_names=tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) for index in range(model.nbody)))
        self._append_executed(initial)
        self._onset = (BodyOnsetObserver(**self.settings["body_onset"], sample_hz=1/self.physics.physics_dt)
                       if self.mode == "mb" else None)
        scene_geometries, robot_geometries = self._geometry_config()
        body_names = self.source.body_names if self.source is not None else OFFICIAL_BODY_NAMES
        parents = _reference_body_parents(model, body_names)
        if self.mode == 'fixed_reference':
            self._reference_version = 'fixed_g1_'+self.settings['fixed_reference']['sha256']
            reference_description = dict(version=self._reference_version, kind='fixed_continuous_g1',
                source=self.settings['fixed_reference']['path'],
                coverage_start_s=float(self.source.timestamps_s[0]), coverage_end_s=float(self.source.trusted_until_s),
                parents=parents, body_names=body_names)
        elif self.source is not None:
            version_hash = hashlib.sha256(json.dumps(self.source.provenance["source_files_sha256"],
                                                     sort_keys=True).encode()).hexdigest()
            self._reference_version = "official_csv_" + version_hash
            reference_description = dict(version=self._reference_version, source=str(self.settings["reference_dir"]),
                coverage_start_s=float(self.source.timestamps_s[0]), coverage_end_s=float(self.source.trusted_until_s),
                parents=parents)
        else:
            reference_identity = dict(weights=self._provider.generator.identity["checkpoint_sha256"],
                scene=self.physics.manifest_sha256, seed=self._seed_history.provenance["sha256"],
                calibration=self._provider.retargeter.calibration["identity"],
                conditioning_mode=self._provider.conditioning_mode)
            if self.settings["mb_conditioning_mode"] == "B2":
                reference_identity['state_observer'] = preparation['state_observer']
                reference_identity['state_observer_limb_source'] = preparation['state_observer_limb_source']
                reference_identity["yaw_goal_lead_s"] = self.settings.get("yaw_goal_lead_s", 0.)
                reference_identity['replan_interval_s'] = self.settings['mb_replan_interval_s']
                reference_identity['adoption_interval_s'] = self.settings.get('mb_adopt_interval_s', 0.)
                reference_identity['adoption_phase_anchor'] = 'generation_simulation_time'
                reference_identity['replan_on_integrated_heading'] = self.settings.get('replan_on_integrated_heading', True)
            if "reference_constraints" in self.settings:
                reference_identity["reference_constraints"] = dict(self.settings["reference_constraints"])
            if self.settings.get('reference_stream_policy') is not None:
                reference_identity['reference_stream'] = preparation['reference_stream']
                reference_identity['consumer_horizon_s'] = self.settings['consumer_horizon_s']
                reference_identity['generation_state_age_s'] = self.settings['max_generation_state_age_s']
                reference_identity['native_blocks'] = self.settings['mb_native_blocks']
            self._reference_version = "frozen_mb_"+hashlib.sha256(json.dumps(reference_identity, sort_keys=True).encode()).hexdigest()
            reference_description = dict(version=self._reference_version, source=self.settings["mb_artifact"],
                coverage_start_s=None, coverage_end_s=None, kind="adapted_g1",
                body_names=body_names, parents=parents,
                conditioning_mode=self._provider.conditioning_mode,
                state_observer=preparation.get('state_observer'),
                reference_stream=preparation.get('reference_stream'),
                state_observer_limb_source=preparation.get('state_observer_limb_source'),
                robot_pose_prior='fixed_initial_pair' if self.settings['mb_conditioning_mode'] == 'B2' else None,
                yaw_goal_lead_s=self.settings.get("yaw_goal_lead_s", 0.),
                actual_state_conditioning=("actual_Root_scene_queries;immutable_native_forecast_context"
                    if self.settings.get('reference_stream_policy') is not None else
                    "fixed_initial_pair_G1_observer" if self.settings["mb_conditioning_mode"] == "B2"
                    else "legacy_reference_history_B1"),
                calibration=self._provider.retargeter.calibration)
            if "reference_constraints" in self.settings:
                reference_description["adaptation"] = dict(self.settings["reference_constraints"])
        position = initial.qpos[:3]
        self._configuration = dict(
            schema="motionbricks.mb_sonic.lab.v1", optimizer_updates=0, world_up="z",
            tracker_backend=self._tracker_backend, tracker=self._policy.asset_metadata,
            actuator_control=self.physics.control_metadata, physics_numerics=self._physics_numerics,
            physics_dt_s=self.physics.physics_dt, controller_hz=50,
            display_hz=30, episode_id=self._episode, run_mode=self.run_mode,
            initial_root_heading_rad=initial_heading,
            run_budget_s=self.settings["max_run_wall_s"] if self.run_mode == "interactive" else self.settings["duration_s"],
            scene_cache=dict(capacity=self._scene_cache_capacity,
                resident_scene_ids=[row["scene"].manifest.scene_id for row in self._scene_cache.values()],
                packed_asset_bytes=sum(sum(len(value) for value in row["assets"].values())
                    for row in self._scene_cache.values() if row["assets"] is not None),
                packed_asset_limit_bytes=self._scene_asset_limit, lod="none_full_resolution"),
            recording=dict(self.settings["logging"]) if self._recording_budget is not None else dict(tick_format="legacy_jsonl"),
            clock=dict(self.settings["clock"], activation="explicit_episode_start", busy_spin=False),
            control_mode=self.mode+"_read_only" if self.mode != "mb" else self._mb_mode_label,
            intent_enabled=self.mode == "mb", input_frame="world_z_up", input_hz=100,
            admission_limits=_admission_limits(self.mode, self.settings),
            replan_clock_owner="native_50Hz_control_tick_dispatch" if self.mode == "mb" else None,
            start_required=True, start_action="begin_episode",
            intent_unavailable_reason="Fixed reference comparison" if self.mode != "mb" else None,
            scene_catalog=self._catalog, scene=dict(manifest_id=scene_id,
                manifest_sha256=self.physics.manifest_sha256, geometries=scene_geometries),
            robot=dict(name=f"Unitree G1 {self.physics.model.nv-6}DOF", source=robot_metadata["path"], geometries=robot_geometries),
            reference=reference_description,
            initial_camera=dict(position=(position+[2.5, -3., 1.4]).tolist(), target=position.tolist()))
        self._recorder = PhysicsRecorder(self._recording_root / f"episode_{self._episode:04d}_{scene_id}.jsonl",
            dict(schema="motionbricks.mb_sonic.physics_record.v1", allocation_id=self.allocation_id,
                 mode=self._configuration["control_mode"], optimizer_updates=0,
                 physics_dt_s=self.physics.physics_dt, controller_hz=50,
                 clock=self._configuration["clock"],
                 admission_limits=self._configuration["admission_limits"],
                 manifest_sha256=self.physics.manifest_sha256, scene_manifest=asdict(scene.manifest),
                 robot=robot_metadata, policy=self._policy.asset_metadata,
                 tracker_backend=self._tracker_backend, actuator_control=self.physics.control_metadata,
                 source=initialization_source, initialization_source=initialization_source,
                 mb_provider=None if self.mode != "mb" else dict(
                     model_identity=self._provider.generator.identity,
                     conditioning_mode=self.settings["mb_conditioning_mode"],
                     history_kind=("actual_g1_error_state" if self.settings["mb_conditioning_mode"] == "B2" else "legacy_reference_history"),
                     replan_clock_owner=self._configuration["replan_clock_owner"],
                     execution=self._provider.execution,
                     cpu_history_observer=preparation["cpu_history_observer"],
                     scene_query=copy.deepcopy(self._provider.generator.scene_query_receipt),
                     scene_adapter_cache=self._provider.cache_description(),
                     calibration_path=self.settings["calibration_path"],
                     calibration_sha256=self.settings["calibration_sha256"],
                     calibration_identity=self._provider.retargeter.calibration["identity"],
                     morphology_root_z_offset_m=self._provider.retargeter.root_z_offset_m,
                     query_origin_frame="soma_world_y_up",
                     query_origin_rule="inverse_C(actual_G1_XYZ); nativeY -= calibrated_root_z_offset",
                     reference_root_features=("measured_Root_and_body_error_state" if self.settings["mb_conditioning_mode"] == "B2" else "unchanged"),
                     consumer_horizon_s=self._provider.consumer_horizon_s,
                     consumer_horizon_contract=consumer_horizon_contract(self.settings),
                     body_onset=dict(self.settings["body_onset"], sample_hz=self._onset.sample_hz),
                     raw_log_pending_plans=self.settings["raw_log_pending_plans"]),
                 sonic_reference_inputs=(dict(joint_positions=True, joint_velocities=True,
                     relative_root_orientation=True, world_root_translation=False, root_height=False)
                     if self._tracker_backend == "sonic" else None),
                 chain_timing_scope=("control,physics,history,publish,scheduler and synchronous full-tick logging; "
                     "excludes explicit waits, async persistence and final timing-telemetry write"
                     if self._recording_budget is not None else
                     "all necessary control,physics,history,publish,scheduler work; excludes explicit waits and optional recording writes"),
                 observation_config=dict(path=str(self._observation_config.resolve()),
                                         sha256=self._observation_config_sha256)), storage_budget=self._recording_budget)
        self._tick_writer = None
        if self.mode == 'fixed_reference' and self._recording_budget is not None:
            self._tick_writer = NumericArtifactWriter(
                self._recording_root/f'episode_{self._episode:04d}_ticks', self._recorder,
                pending_plans=self.settings['logging']['tick_pending_records'], storage_budget=self._recording_budget,
                max_record_bytes=self.settings['logging']['max_numeric_record_bytes'])
            self._recorder.tick_writer = self._tick_writer
            self._recorder.tick_format = self.settings['logging']['tick_format']
        if self.mode == "mb":
            numeric_options = (dict(storage_budget=self._recording_budget,
                max_record_bytes=self.settings["logging"]["max_numeric_record_bytes"])
                if self._recording_budget is not None else {})
            compression_execution = self.settings.get("logging", {}).get("compression_execution", "thread")
            if compression_execution not in ("thread", "process"):
                raise ValueError("Unknown explicit numeric compression execution")
            if compression_execution == "process":
                self._compression_executor = ProcessPoolExecutor(1, mp_context=get_context("spawn"))
                self._compression_executor.submit(encode_npz, {}).result(timeout=10)
                numeric_options["compression_executor"] = self._compression_executor
            self._numeric_writer = NumericArtifactWriter(
                self._recording_root / f"episode_{self._episode:04d}_numeric", self._recorder,
                pending_plans=self.settings["raw_log_pending_plans"], **numeric_options)
            if self._recording_budget is not None:
                self._tick_writer = NumericArtifactWriter(
                    self._recording_root / f"episode_{self._episode:04d}_ticks", self._recorder,
                    pending_plans=self.settings["logging"]["tick_pending_records"], **numeric_options)
                self._recorder.tick_writer = self._tick_writer
                self._recorder.tick_format = self.settings["logging"]["tick_format"]
            self._mb_worker = OnlineMBWorker(self._provider, self._intents, self._mb_generation_snapshot,
                self._recorder, numeric_writer=self._numeric_writer,
                simulation_time=lambda: self.physics.time_s,
                replan_interval_s=self.settings["mb_replan_interval_s"],
                max_state_age_s=self.settings["max_generation_state_age_s"],
                max_plan_wall_age_s=self.settings["max_plan_wall_age_s"],
                required_future_s=self.physics.reference_lookahead_s)
        self._recorder.write(dict(kind="explicit_initial_reset", actual=asdict(initial)))
        if self.mode == "mb":
            self._recorder.write(dict(kind="mb_deployment_preparation", execution=self._provider.execution,
                report=preparation, wall_time_ns=preparation_ns, command_seq=None,
                executed_physics_steps=0, included_in_online_latency=False,
                scope="cold_model_initialization_and_parity_not_consumed_input_or_physical_evidence"))
        self._publish(initial)
        if self._static_heap is not None:
            self._configuration["static_heap"] = self._static_heap.prepare()
            self._recorder.write(dict(kind="static_heap_preparation", scope="parent_CPU_runtime",
                included_in_online_latency=False, **self._configuration["static_heap"]))

    def config(self):
        with self._lock:
            return dict(copy.deepcopy(self._configuration),hardware_execution_profile=self.hardware_execution_profile)

    def asset(self, asset_id):
        with self._lock:
            return self._assets[asset_id]

    def submit_intent(self, command):
        if self.mode != "mb":
            if command.get("source") == "disconnect_stop":
                return dict(consumed=False, mode="official_reference_read_only")
            raise ValueError("Official-reference mode is read-only; no joystick command was consumed")
        if self._status.startswith(("completed", "terminal", "runtime_failure")):
            raise ValueError("The episode ended; select a scene explicitly for a new episode")
        if self._recording_budget is not None:
            self._recording_budget.check()
        with self._lock:
            if (command["scene_id"] != self.physics.scene_id
                    or command["scene_manifest_sha256"] != self.physics.manifest_sha256
                    or type(command["episode_id"]) is not int or command["episode_id"] != self._episode):
                raise ValueError("Intent belongs to a stale scene, manifest or episode")
            session = command["session_id"]
            if not isinstance(session, str) or not session:
                raise ValueError("Intent requires an explicit controller session")
            if self._controller_session_id is not None and session != self._controller_session_id:
                raise ValueError("Controller session changed; an explicit manual Reset is required")
            if (type(command["command_seq"]) is not int or type(command["command_id"]) is not int
                    or command["command_seq"] != command["command_id"]
                    or command["command_seq"] <= self._last_intent_seq):
                raise ValueError("Expected a strictly increasing raw command sequence")
            velocity = np.asarray(command["velocity_xy_mps"], dtype=float)
            if velocity.shape != (2,) or not np.isfinite(velocity).all() or not np.isfinite(command["yaw_rate_radps"]):
                raise ValueError("Intent requires finite independent world XY velocity and yaw rate")
            revision = self._intents.replace(command)
            self._controller_session_id, self._last_intent_seq = session, command["command_seq"]
        self._recorder.write(dict(kind="intent_received", intent_revision=revision, raw_intent=command,
                                 received_at_monotonic_ns=perf_counter_ns(), consumed_by_sonic=False))
        self._mb_worker.request()
        return dict(received=True, intent_revision=revision, consumed=False)

    def snapshot(self):
        with self._lock:
            if self._failure is not None:
                raise RuntimeError(self._failure)
            return copy.deepcopy(self._latest)

    def subscribe_snapshots(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Register one nonblocking UI wakeup, including an immediate initial wakeup.

        The subscription survives episode resets. Callbacks must only schedule work
        on their event loop; notifications carry no snapshot or browser acknowledgement.
        Unsubscribe is idempotent and does not wait for an already in-flight callback.
        """
        if not callable(callback):
            raise TypeError("Snapshot notification requires a callable")
        subscription = (object(), callback)
        with self._snapshot_subscription_lock:
            if self._snapshot_subscriber is not None:
                raise RuntimeError("A snapshot subscriber already owns the UI publication stream")
            self._snapshot_subscriber = subscription

        def unsubscribe():
            with self._snapshot_subscription_lock:
                if self._snapshot_subscriber is subscription:
                    self._snapshot_subscriber = None

        self._notify_snapshots()
        return unsubscribe

    def _notify_snapshots(self):
        with self._snapshot_subscription_lock:
            subscription = self._snapshot_subscriber
        if subscription is not None:
            try:
                subscription[1]()
            except Exception:
                # A closed browser/event loop cannot terminate the physical executor.
                pass

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Lab episode already started; select a scene explicitly for a new episode")
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="sonic-physics", daemon=False)
        self._thread.start()

    def begin_episode(self):
        """Explicit parent/user gate after assets are ready; never restarts a finite clip."""
        if self._thread is None or not self._thread.is_alive() or self._stop_event.is_set():
            raise RuntimeError("The lab must be armed before an explicit episode start")
        if self._begin_event.is_set():
            raise ValueError("This episode already started; no automatic reference replay")
        self._begin_event.set()
        return dict(started=True, control_mode=self._configuration["control_mode"], command_consumed=False)

    def stop(self):
        self._stop_event.set()
        self._begin_event.set()
        if self._thread is not None:
            self._thread.join()
        if self._mb_worker is not None:
            self._mb_worker.stop()
        self._close_numeric_logs()
        if not self._recorder.stream.closed:
            self._recorder.write(dict(kind="explicit_stop", actual=asdict(self.physics.snapshot()),
                                      status=self._status))
            self._recorder.close()
        if self.settings.get("heap_strategy") == "freeze_static":
            self._static_heap.release()

    def shutdown(self):
        """Final service cleanup; manual stop/reset keeps the cached planner alive."""
        try:
            self.stop()
        finally:
            if self._mb_planner_backend == "spawn" and self._provider is not None:
                self._provider.close()

    def select_scene(self, scene_id):
        if self._recording_budget is not None:
            self._recording_budget.check()
        entry = next(row for row in self._catalog if row["id"] == scene_id)
        if not entry["available"]:
            raise ValueError(entry["unavailable_reason"])
        self.stop()
        self._thread = None
        with self._lock:
            self._errors = []
            self._load_episode(scene_id)
        self._notify_snapshots()
        self.start()
        return self.config()

    def _publish(self, state):
        self._latest_actual_state = state
        data = self.physics.data
        ids = self._robot_geom_ids
        quats = Rotation.from_matrix(data.geom_xmat[ids].reshape(-1, 3, 3)).as_quat()
        geom_poses = [dict(id=int(geom_id), position=data.geom_xpos[geom_id].tolist(),
                           quaternion_xyzw=quat.tolist()) for geom_id, quat in zip(ids, quats)]
        reference = None
        tracking_error, heading_error = None, None
        if self.mode != "mb" and self._source_clock_origin_s is not None:
            index = int(np.floor((state.time_s-self._source_clock_origin_s)/POLICY_DT+1e-8))
            reference = dict(version=self._reference_version, time_s=float(self.source.timestamps_s[index]),
                             joint_positions=self.source.body_pos_w[index].tolist(),
                             command_id=self.source.command_id)
            tracking_error = float(np.linalg.norm(state.qpos[:3]-self.source.qpos[index, :3]))
            actual_forward = Rotation.from_quat(state.qpos[[4, 5, 6, 3]]).apply([1, 0, 0])
            ref_forward = Rotation.from_quat(self.source.qpos[index, [4, 5, 6, 3]]).apply([1, 0, 0])
            angle = np.arctan2(actual_forward[1], actual_forward[0])-np.arctan2(ref_forward[1], ref_forward[0])
            heading_error = float(np.arctan2(np.sin(angle), np.cos(angle)))
        if self.mode == "mb" and self._active_update is not None:
            adapted = self._active_update.adapted_reference
            qindex = int(np.searchsorted(adapted.timestamps_s, state.time_s+1e-9, side="right")-1)
            if qindex < 0 or state.time_s > adapted.trusted_until_s+1e-8:
                raise ReferenceCoverageError("Display reference exceeds the actual trusted adapted G1 reference")
            reference = dict(version=self._reference_version, kind="adapted_g1",
                time_s=float(adapted.timestamps_s[qindex]),
                joint_positions=adapted.body_pos_w[qindex].tolist(),
                command_id=self._active_update.command["command_id"],
                command_seq=self._active_update.packet.generated_from_command_seq,
                session_id=self._active_update.command["session_id"], plan_id=self._active_update.packet.plan_id)
            tracking_error = float(np.linalg.norm(state.qpos[:3]-adapted.qpos[qindex, :3]))
            actual_yaw, reference_yaw = _root_heading(state.qpos), _root_heading(adapted.qpos[qindex])
            heading_error = _angle_delta(actual_yaw, reference_yaw) if actual_yaw is not None and reference_yaw is not None else None
        wall_s = ((perf_counter_ns()-self._wall_start_ns)/1e9) if self._wall_start_ns is not None else None
        self._frame_id += 1
        onset = self._onset.onset if self._onset is not None else None
        consumed = copy.deepcopy(self._consumed_command)
        if consumed is not None:
            consumed["effective_intent"] = dict(velocity_xy_mps=consumed["velocity_xy_mps"],
                                                 yaw_rate_radps=consumed["yaw_rate_radps"])
        self._latest = dict(
            frame_id=self._frame_id, sim_time_s=state.time_s, server_monotonic_ns=state.captured_at_monotonic_ns,
            state_version=state.state_version, episode_id=self._episode,
            scene_manifest_id=self.physics.scene_id, scene_manifest_sha256=self.physics.manifest_sha256,
            status=self._status, actual=dict(root_position=state.qpos[:3].tolist(), geom_poses=geom_poses,
                root_heading_rad=_root_heading(state.qpos),
                root_velocity_world_mps=state.qvel[:3].tolist(), root_quaternion_wxyz=state.qpos[3:7].tolist(),
                body_onset_command_id=onset["command_id"] if onset else None,
                body_onset_session_id=onset["session_id"] if onset else None,
                body_onset_sim_time_s=onset["sim_time_s"] if onset else None,
                body_onset_server_ns=onset["sample_monotonic_ns"] if onset else None),
            reference=reference, command=consumed,
            derived_heading_rad=self._active_update.derived_heading_rad if self._active_update is not None else None,
            contacts=[dict(geom1=contact.geom_ids[0], geom2=contact.geom_ids[1],
                           position=contact.position_world.tolist(), normal_force_n=float(contact.wrench_contact[0]))
                      for contact in state.contacts],
            timing=dict(chain_ms=self._last_chain_ms, physics_steps=self._physics_steps,
                        controller_steps=self._controller_steps,
                        realtime_factor=state.time_s/wall_s if wall_s else None,
                        tracking_error_m=tracking_error, heading_error_rad=heading_error),
            root_error_semantics="diagnostic_reference_root_deviation; original_G1_encoder_ignores_root_translation_height",
            errors=list(self._errors))

    def _integrate_mb_intent(self):
        revision, command = self._intents.latest()
        if command is None:
            return revision, command
        with self._lock:
            old_revision, old_heading = self._integrated_intent_revision, self._yaw.desired_heading_rad
            self._yaw.for_tick(command, self.physics.time_s)
            self._integrated_intent_revision = revision
        if revision != old_revision or (self.settings.get('replan_on_integrated_heading', True)
                and abs(_angle_delta(self._yaw.desired_heading_rad, old_heading)) > 1e-10):
            self._mb_worker.request()
        return revision, command

    def _append_executed(self, state):
        feedback = self.mode == "mb" and self.settings.get("mb_conditioning_mode") == "B2"
        if feedback and state.state_version % self.physics.substeps:
            return
        if self._executed.samples and state.time_s <= self._executed.samples[-1]["timestamp_s"]:
            raise ValueError("Executed simulation time must advance; automatic reset is unsupported")
        if self.mode == "mb" and not feedback and self.settings.get("estimate_actual_diagnostics") is False:
            # B1 does not consume executed pose history; full states remain in the required recorder.
            sample = dict(timestamp_s=state.time_s, qpos=state.qpos,
                qvel=state.qvel,
                captured_at_monotonic_ns=state.captured_at_monotonic_ns, state_version=state.state_version)
            self._executed.samples[:] = [_numeric_snapshot(sample)]
            return
        # Reuse this substep's measured state; never query contacts a second time.
        sample = dict(timestamp_s=state.time_s, qpos=state.qpos, qvel=state.qvel,
            body_pos_w=state.body_positions_world, body_quat_w=state.body_quat_wxyz,
            captured_at_monotonic_ns=state.captured_at_monotonic_ns, state_version=state.state_version)
        if feedback:
            sample["contact_columns"] = dict(body_ids=state.contacts.body_ids,
                position_world=state.contacts.position_world, wrench_contact=state.contacts.wrench_contact)
        else:
            sample["contacts"] = tuple(dict(body_ids=contact.body_ids, position_world=contact.position_world,
                wrench_contact=contact.wrench_contact) for contact in state.contacts)
        sample = _numeric_snapshot(sample)
        if feedback:
            active = self._active_update
            reference_pose = None
            if active is not None:
                reference = active.adapted_reference
                index = int(np.searchsorted(reference.timestamps_s, state.time_s-1e-9))
                if (index >= len(reference.timestamps_s)
                        or abs(reference.timestamps_s[index]-state.time_s) > 1e-8):
                    raise ReferenceCoverageError('Executed control boundary is outside its consumed reference grid')
                reference_pose = reference.qpos[index].copy()
                reference_pose.flags.writeable = False
            sample.update(reference_qpos=reference_pose,
                reference_plan_id=None if active is None else active.packet.plan_id,
                bootstrap_until_s=self.physics.warmup_control_ticks*POLICY_DT)
        self._executed.samples.append(sample)
        keep = int(np.ceil(.25/(POLICY_DT if feedback else self.physics.physics_dt)))+2
        del self._executed.samples[:-keep]

    def _mb_generation_snapshot(self):
        started_ns = perf_counter_ns()
        with self._lock:
            locked_ns = perf_counter_ns()
            if self._seed_clock_origin_s is None:
                return None
            executed = ExecutedHistory(self.physics.scene_id, samples=list(self._executed.samples),
                                        body_names=self._executed.body_names)
            actual = self._executed.samples[-1]
            captured_at_ns = actual["captured_at_monotonic_ns"]
            history_time = self.physics.history.entries[-1].time_s
            source, seed_clock = self._reference_cache, self._seed_clock_origin_s
            previous_packet = None if self._active_update is None else self._active_update.packet
            revision, heading = self._integrated_intent_revision, self._yaw.desired_heading_rad
        snapshot_ready_ns = perf_counter_ns()
        if self.settings.get("mb_conditioning_mode") == "B2":
            history = self._seed_history
            source = None
        elif source is None:
            history = replace(self._seed_history, timestamps_s=self._seed_history.timestamps_s+seed_clock)
        elif self._mb_planner_backend == "spawn":
            history = None
        else:
            history = self._provider.reference_history_builder(source, end_time_s=history_time,
                representation=self._provider.retargeter.rep)
        history_ready_ns = perf_counter_ns()
        return MBGenerationSnapshot(actual["timestamp_s"],
            (float(history_time) if self.settings.get("mb_conditioning_mode") == "B2" else
             float(history.timestamps_s[-1]) if history is not None else float(history_time)),
            actual["state_version"], revision, actual["qpos"][:3].copy(), heading,
            history, executed, captured_at_ns,
            dict(lock_wait=locked_ns-started_ns, state_snapshot=snapshot_ready_ns-locked_ns,
                 reference_history=history_ready_ns-snapshot_ready_ns, total=history_ready_ns-started_ns),
            reference_source=(source if self._mb_planner_backend == "spawn"
                and self.settings.get('reference_stream_policy') is None else None),
            previous_reference_packet=previous_packet if "reference_constraints" in self.settings else None)

    def _close_numeric_logs(self):
        self._close_numeric_writer(self._numeric_writer)
        self._close_numeric_writer(self._tick_writer)
        if self._compression_executor is not None:
            self._compression_executor.shutdown(wait=True)
            self._compression_executor = None

    def _close_numeric_writer(self, writer):
        if writer is None:
            return
        try:
            writer.close()
        except Exception as error:
            failure = f"required_numeric_log_failure:{type(error).__name__}:{error}"
            with self._lock:
                if self._failure == failure:
                    return
                self._status, self._failure = "runtime_failure", failure
                self._errors.append(failure)
                self._latest = dict(self._latest, status=self._status, errors=list(self._errors))
            self._notify_snapshots()
            self.physics.terminate(failure)
            self._recorder.write(dict(kind="required_numeric_log_failure", error=failure,
                                     sim_time_s=self.physics.time_s, monotonic_ns=perf_counter_ns()))

    def _adopt_mb_update(self, update):
        packet = update.packet
        if packet.robot_model_sha256 != self.physics.robot_model_sha256:
            raise ValueError("ReferencePacket belongs to a different actual robot model")
        if packet.scene_manifest_sha256 != self.physics.manifest_sha256 or packet.joint_names != BODY_JOINT_NAMES:
            raise ValueError("ReferencePacket geometry/joint identity differs from the actual runtime")
        if update.adapted_reference.body_names != OFFICIAL_BODY_NAMES:
            raise ValueError("Adapted G1 reference body order differs from the released 14-body hierarchy")
        if packet.generated_from_command_seq != update.command["command_seq"]:
            raise ValueError("ReferencePacket raw command sequence differs from its generation input")
        if (update.command["episode_id"] != self._episode
                or packet.episode_id != self._episode or packet.session_id != self._controller_session_id
                or update.command["session_id"] != self._controller_session_id
                or update.command["scene_id"] != self.physics.scene_id
                or update.command["scene_manifest_sha256"] != self.physics.manifest_sha256):
            raise ValueError("ReferencePacket belongs to a stale controller session or episode")
        with self._lock:
            self._active_update = update
            self._reference_cache = update.proposal.source
        return dict(kind="reference_packet_adopted", plan_id=packet.plan_id,
            generated_from_command_seq=packet.generated_from_command_seq,
            state_version=packet.state_version, sim_time_s=self.physics.time_s,
            reference_adopted_monotonic_ns=perf_counter_ns(), consumed_by_sonic=False,
            snapshot_age_sim_s=self.physics.time_s-packet.issued_from_sim_time)

    def _check_mb_input_deadline(self, now_ns):
        if self._consumed_intent_revision is None:
            return
        pending_since_ns = self._intents.pending_since_ns(self._consumed_intent_revision)
        if (pending_since_ns is not None and
                now_ns-pending_since_ns > round(self.settings["max_pending_intent_age_s"]*1e9)):
            raise RuntimeError("Newest intent was not applied before the declared pending-intent age bound")

    def _run_mb(self, deadline_ns, interval_ns):
        self._seed_clock_origin_s = self.physics.time_s
        self._recorder.write(dict(kind="mb_seed_clock_assignment",
            original_times_s=self._seed_history.timestamps_s,
            assigned_times_s=self._seed_history.timestamps_s+self._seed_clock_origin_s,
            spatial_transform="none", source_seed_sha256=self._seed_history.provenance["sha256"],
            seed_role=("initial_pose_prior_only; conditioning_uses_measured_error_state" if self.settings["mb_conditioning_mode"] == "B2"
                       else "source_seed_history")))
        self._numeric_writer.start()
        if self._tick_writer is not None:
            self._tick_writer.start()
        self._mb_worker.start()
        self._integrate_mb_intent()
        self._mb_worker.request()
        initialization_started_ns = perf_counter_ns()
        while self._active_update is None and not self._stop_event.is_set():
            if self._run_budget_exhausted():
                return
            if self._recording_budget is not None:
                self._recording_budget.check()
            boundary_intent = self._integrate_mb_intent()
            update = self._mb_worker.take_ready(self.physics.time_s)
            if update is not None:
                self._recorder.write(self._adopt_mb_update(update))
                break
            self._mb_worker.control_tick(self.physics.time_s, intent=boundary_intent, initial_wait=True)
            if perf_counter_ns()-initialization_started_ns > self.settings["mb_initial_reference_timeout_s"]*1e9:
                raise RuntimeError("No valid initial MB reference before the explicit startup timeout")
            self._stop_event.wait(self.physics.physics_dt)
        if self._stop_event.is_set():
            return
        self._recorder.write(dict(kind="initial_mb_reference_wait",
            wall_time_ns=perf_counter_ns()-initialization_started_ns, sim_time_s=self.physics.time_s,
            physics_advanced_during_wait=False, included_in_episode_RTF=True))
        deadline_ns = perf_counter_ns()
        count = int(np.floor(self.settings["duration_s"]/POLICY_DT+1e-9))
        for control_index in range(count):
            if self._stop_event.is_set() or self._run_budget_exhausted():
                return
            if self._recording_budget is not None:
                self._recording_budget.check()
                self._tick_writer.check()
            control_started_ns = perf_counter_ns()
            optional_recording_ns = 0
            boundary_intent = self._integrate_mb_intent()
            update = self._mb_worker.take_ready(self.physics.time_s)
            if update is not None:
                if reference_adoption_due(update, self._active_update, self.physics.time_s,
                        self.settings.get('mb_adopt_interval_s', 0.)):
                    adoption_record = self._adopt_mb_update(update)
                else:
                    adoption_record = dict(kind='reference_candidate_not_adopted', plan_id=update.packet.plan_id,
                        reason='steady_intent_execution_phase', sim_time_s=self.physics.time_s,
                        generated_from_command_seq=update.command['command_seq'], consumed_by_tracker=False)
                recording_started_ns = perf_counter_ns()
                self._recorder.write(adoption_record)
                optional_recording_ns += perf_counter_ns()-recording_started_ns
            active_now = self._active_update
            stream_due = (self.settings.get('reference_stream_policy') is None
                or continuous_reference_replan_due(active_now,boundary_intent[0],self.physics.time_s,
                    self.physics.reference_lookahead_s,self.settings['max_plan_wall_age_s']))
            if stream_due:
                self._mb_worker.control_tick(self.physics.time_s, intent=boundary_intent)
            active = self._active_update
            self._check_mb_input_deadline(perf_counter_ns())
            latest_start = min(active.adapted_reference.trusted_until_s,
                               active.adapted_reference.timestamps_s[-1])-self.physics.reference_lookahead_s
            if self.physics.time_s > latest_start+1e-8:
                raise ReferenceCoverageError("Trusted active reference exhausted; no tail hold or fallback")
            policy_started_ns = perf_counter_ns()
            window, command = self.physics.begin_control(active.adapted_reference)
            execution_intent = active.command
            execution_heading = active.derived_heading_rad
            if self.physics.root_velocity_task is not None:
                execution_intent = boundary_intent[1]
                execution_heading = self._yaw.desired_heading_rad
                command = self.physics.apply_velocity_task(command, execution_intent, execution_heading)
            action_ready_ns = perf_counter_ns()
            self._check_mb_input_deadline(action_ready_ns)
            policy_ns = action_ready_ns-policy_started_ns
            with self._lock:
                old_identity = ((self._consumed_command["session_id"], self._consumed_command["command_id"])
                                 if self._consumed_command is not None else None)
                first_command_use = self._consumed_intent_revision != active.intent_revision
                first_execution_use = old_identity != (execution_intent['session_id'], execution_intent['command_id'])
                self._consumed_command = copy.deepcopy(execution_intent)
                self._consumed_intent_revision = active.intent_revision
                self._onset.consume(execution_intent, self._latest_actual_state,
                    desired_heading_rad=execution_heading, consumed_at_monotonic_ns=action_ready_ns)
            states, physics_ns, publishing_ns, waiting_ns = [], 0, 0, 0
            wait_requested_ns, wait_elapsed_ns, wait_overshoot_ns = 0, 0, 0
            first_motor_ns = None
            onset_before = self._onset.onset
            for _ in range(self.physics.substeps):
                if self._run_budget_exhausted():
                    break
                deadline_ns += interval_ns
                wait = wait_until(self._stop_event, deadline_ns)
                waiting_ns += wait.explicit_wait_ns
                wait_requested_ns += wait.requested_ns
                wait_elapsed_ns += wait.elapsed_ns
                wait_overshoot_ns += wait.overshoot_ns
                if wait.interrupted:
                    break
                physical_started_ns = perf_counter_ns()
                state = self.physics._advance_pd(command.joint_target, substeps=1)[0]
                physics_ns += perf_counter_ns()-physical_started_ns
                first_motor_ns = self.physics.last_motor_control_ns if first_motor_ns is None else first_motor_ns
                if not state.root_ground_known:
                    state = self.physics.terminate("unknown_canonical_ground_domain")
                states.append(state)
                self._physics_steps += 1
                publishing_started_ns = perf_counter_ns()
                publish_substep = (self.settings.get("physics_publication", "substeps") == "substeps"
                    and len(states) < self.physics.substeps and not self.physics.terminal_reason)
                with self._lock:
                    self._latest_actual_state = state
                    self._append_executed(state)
                    self._onset.observe(state)
                    self._status = "terminal_failure" if self.physics.terminal_reason else self._mb_mode_label
                    if self.physics.terminal_reason:
                        self._errors.append(self.physics.terminal_reason)
                    if publish_substep:
                        self._publish(state)
                if publish_substep:
                    self._notify_snapshots()
                publishing_ns += perf_counter_ns()-publishing_started_ns
                if self.physics.terminal_reason:
                    break
            if not states:
                return
            history_started_ns = perf_counter_ns()
            if not self.physics.terminal_reason and len(states) == self.physics.substeps:
                self.physics._append_proprioception()
            history_ns = perf_counter_ns()-history_started_ns
            self._controller_steps += 1
            publishing_started_ns = perf_counter_ns()
            with self._lock:
                if control_index == count-1 and not self.physics.terminal_reason:
                    self._status = "completed_"+self._mb_mode_label
                if len(states) == self.physics.substeps or self.physics.terminal_reason:
                    self._publish(states[-1])
            if len(states) == self.physics.substeps or self.physics.terminal_reason:
                self._notify_snapshots()
            publishing_ns += perf_counter_ns()-publishing_started_ns
            work_finished_ns = perf_counter_ns()
            controller_work_ns = work_finished_ns-control_started_ns-waiting_ns-optional_recording_ns
            plan_chain_ns = active.timing_ns["provider_total"]+controller_work_ns
            self._last_chain_ms = plan_chain_ns/1e6
            with self._lock:
                self._latest = dict(self._latest, timing=dict(self._latest["timing"],
                    chain_ms=self._last_chain_ms, controller_chain_ms=controller_work_ns/1e6,
                    controller_steps=self._controller_steps,
                    mb_provider_ms=active.timing_ns["provider_total"]/1e6,
                    generation_snapshot_age_s=states[-1].time_s-active.packet.issued_from_sim_time))
            self._notify_snapshots()
            receipt = active.packet.use_receipt(float(window.timestamps_s[0]),
                policy_action_ready_ns=action_ready_ns, first_motor_control_ns=first_motor_ns,
                reference_offsets_s=self._policy.reference_offsets_s if self._tracker_backend == "protomotions" else None)
            first_plan_use = self._last_used_plan_id != active.packet.plan_id
            self._last_used_plan_id = active.packet.plan_id
            self._recorder.write(dict(kind="reference_packet_used", **receipt,
                actual_raw_intent=active.command, derived_heading_rad=active.derived_heading_rad,
                executor_intent=execution_intent, executor_heading_rad=execution_heading,
                first_executor_command_use=first_execution_use,
                executor_input_to_action_ns=action_ready_ns-execution_intent['server_received_ns'] if first_execution_use else None,
                first_command_use=first_command_use, first_plan_use=first_plan_use,
                used_state_version=states[0].state_version-1,
                physics_done_state_version=states[-1].state_version,
                input_to_action_ns=action_ready_ns-active.command["server_received_ns"] if first_command_use else None,
                input_to_motor_ns=first_motor_ns-active.command["server_received_ns"] if first_command_use else None,
                current_intent_age_ns=action_ready_ns-active.command["server_received_ns"],
                controller_work_ns=controller_work_ns, plan_necessary_compute_chain_ns=plan_chain_ns,
                clock_semantics="monotonic_ns; reused_plan_compute_is_not_new_generation"))
            if self._onset.onset is not None and self._onset.onset is not onset_before:
                self._recorder.write(dict(kind="actual_body_onset", **self._onset.onset))
            tick = RuntimeTick(self.physics.scene_id, execution_intent["command_id"],
                self._mb_mode_label, float(window.timestamps_s[0]),
                execution_intent["server_received_ns"], dict(plan_id=active.packet.plan_id,
                    generated_from_command_seq=active.packet.generated_from_command_seq,
                    raw_source_record="mb_raw_and_adapted_reference"), window, command, tuple(states),
                dict(**command.timing_ns, policy_total=policy_ns, physics=physics_ns,
                    history=history_ns, publishing=publishing_ns,
                    scheduler=controller_work_ns-policy_ns-physics_ns-history_ns-publishing_ns,
                    explicit_wait=waiting_ns, optional_recording=optional_recording_ns,
                    wait_requested=wait_requested_ns, wait_elapsed=wait_elapsed_ns, wait_overshoot=wait_overshoot_ns,
                    controller_chain_total=controller_work_ns, provider_total=active.timing_ns["provider_total"],
                    chain_total=plan_chain_ns, control_wall=work_finished_ns-control_started_ns))
            self._recorder.tick(tick)
            if self._tick_writer is not None:
                synchronous_log_tail_ns = perf_counter_ns()-work_finished_ns
                full_chain_ns = plan_chain_ns+synchronous_log_tail_ns
                self._last_chain_ms = full_chain_ns/1e6
                with self._lock:
                    self._latest = dict(self._latest, timing=dict(self._latest["timing"],
                        chain_ms=self._last_chain_ms, synchronous_log_tail_ms=synchronous_log_tail_ns/1e6,
                        recording_reserved_payload_bytes=self._recording_budget.used_bytes))
                self._notify_snapshots()
                self._recorder.write(dict(kind="policy_tick_recording_timing",
                    state_version=states[-1].state_version, plan_id=active.packet.plan_id,
                    synchronous_log_tail_ns=synchronous_log_tail_ns,
                    controller_work_ns=controller_work_ns, provider_work_ns=active.timing_ns["provider_total"],
                    full_necessary_chain_ns=full_chain_ns,
                    scope="includes required synchronous snapshots/receipts; excludes async persistence and this final telemetry write"))
            if self.physics.terminal_reason or self._stop_event.is_set() or self._run_budget_exhausted():
                return

    def _run(self):
        self._begin_event.wait()
        if self._stop_event.is_set():
            return
        timer = WindowsTimerResolution(self.settings["clock"]["timer_resolution_ms"])
        try:
            with timer:
                self._recorder.write(dict(kind="timer_resolution_acquired", **timer.metadata))
                self._run_paced()
        except Exception as error:
            self._record_runtime_failure(error)
        finally:
            self._recorder.write(dict(kind="timer_resolution_lifecycle", **timer.metadata))

    def _record_runtime_failure(self, error):
        with self._lock:
            self._status = "runtime_failure"
            self._failure = f"{type(error).__name__}: {error}"
            self._errors.append(self._failure)
            self._latest = dict(self._latest, status=self._status, errors=list(self._errors))
        self._notify_snapshots()
        self.physics.terminate(self._failure)
        self._recorder.write(dict(kind="terminal_runtime_failure", error=self._failure,
                                 actual=asdict(self.physics.snapshot()), monotonic_ns=perf_counter_ns()))

    def _run_budget_exhausted(self):
        if self.run_mode != "interactive":
            return False
        if self._budget_finished:
            return True
        now_ns = perf_counter_ns()
        if now_ns < self._run_deadline_ns:
            return False
        with self._lock:
            self._budget_finished = True
            self._status = "completed_interactive_run_budget"
            self._latest = dict(self._latest, status=self._status)
        self._notify_snapshots()
        self._recorder.write(dict(kind="explicit_run_budget_exhausted", monotonic_ns=now_ns,
            wall_start_ns=self._wall_start_ns, max_run_wall_s=self.settings["max_run_wall_s"],
            actual_sim_time_s=self.physics.time_s, episode_id=self._episode, automatic_reset=False))
        return True

    def _run_paced(self):
        self._wall_start_ns = perf_counter_ns()
        self._run_deadline_ns = (self._wall_start_ns+round(self.settings["max_run_wall_s"]*1e9)
                                 if self.run_mode == "interactive" else None)
        deadline_ns = self._wall_start_ns
        interval_ns = round(self.physics.physics_dt*1e9)
        warmup_substeps = 0
        try:
            if self.mode == 'fixed_reference' and self._tick_writer is not None:
                self._tick_writer.start()
            if self.mode == "mb" and self.settings.get("warmup_control_ticks", 9) == 0:
                state = self.physics.snapshot()
                with self._lock:
                    self._executed.samples.clear()
                    self._append_executed(state)
                    self._publish(state)
                self._recorder.write(dict(kind="initial_measured_state_refresh", actual=asdict(state),
                    physics_advanced=False, state_assignment=False))
                self._notify_snapshots()
            while self.physics.needs_warmup and not self._stop_event.is_set():
                if self._run_budget_exhausted():
                    return
                if self._recording_budget is not None:
                    self._recording_budget.check()
                deadline_ns += interval_ns
                if wait_until(self._stop_event, deadline_ns).interrupted:
                    return
                state = self.physics._advance_pd(self.physics.warmup_joint_target, substeps=1)[0]
                self._physics_steps += 1
                warmup_substeps += 1
                self._recorder.write(dict(kind="actual_history_warmup", actual=asdict(state)))
                if self.physics.terminal_reason:
                    self._status = "terminal_failure"
                    self._errors.append(self.physics.terminal_reason)
                elif warmup_substeps % self.physics.substeps == 0:
                    self.physics._append_proprioception()
                with self._lock:
                    if self.mode == "mb":
                        self._append_executed(state)
                    self._status = "terminal_failure" if self.physics.terminal_reason else "actual_history_warmup"
                    self._publish(state)
                self._notify_snapshots()
                if self.physics.terminal_reason:
                    return
            if self._stop_event.is_set():
                return
            if self.mode == "mb":
                self._run_mb(deadline_ns, interval_ns)
                return
            self._source_clock_origin_s = 0. if self.mode == 'fixed_reference' else self.physics.time_s
            reference = replace(self.source, timestamps_s=self.source.timestamps_s+self._source_clock_origin_s,
                                trusted_until_s=self.source.trusted_until_s+self._source_clock_origin_s)
            self._recorder.write(dict(kind="reference_clock", source_start_s=0.,
                                      simulation_start_s=self._source_clock_origin_s))
            issued_ns = perf_counter_ns()
            remaining = self.settings['duration_s']-(self.physics.time_s if self.mode == 'fixed_reference' else 0.)
            count = int(np.floor(remaining/POLICY_DT+1e-9))
            for control_index in range(count):
                if self._stop_event.is_set():
                    return
                began_ns = perf_counter_ns()
                window, command = self.physics.begin_control(reference)
                policy_ns = perf_counter_ns()-began_ns
                states, physics_ns, publish_ns, waiting_ns = [], 0, 0, 0
                wait_requested_ns, wait_elapsed_ns, wait_overshoot_ns = 0, 0, 0
                for _ in range(self.physics.substeps):
                    deadline_ns += interval_ns
                    wait = wait_until(self._stop_event, deadline_ns)
                    waiting_ns += wait.explicit_wait_ns
                    wait_requested_ns += wait.requested_ns
                    wait_elapsed_ns += wait.elapsed_ns
                    wait_overshoot_ns += wait.overshoot_ns
                    if wait.interrupted:
                        break
                    began_step_ns = perf_counter_ns()
                    state = self.physics._advance_pd(command.joint_target, substeps=1)[0]
                    physics_ns += perf_counter_ns()-began_step_ns
                    states.append(state)
                    self._physics_steps += 1
                    publish_started_ns = perf_counter_ns()
                    with self._lock:
                        self._status = "terminal_failure" if self.physics.terminal_reason else self.mode+"_read_only"
                        if self.physics.terminal_reason:
                            self._errors.append(self.physics.terminal_reason)
                        if len(states) < self.physics.substeps and not self.physics.terminal_reason:
                            self._publish(state)
                    if len(states) < self.physics.substeps and not self.physics.terminal_reason:
                        self._notify_snapshots()
                    publish_ns += perf_counter_ns()-publish_started_ns
                    if self.physics.terminal_reason:
                        break
                if not states:
                    return
                history_start_ns = perf_counter_ns()
                if not self.physics.terminal_reason and len(states) == self.physics.substeps:
                    self.physics._append_proprioception()
                history_ns = perf_counter_ns()-history_start_ns
                self._controller_steps += 1
                publish_started_ns = perf_counter_ns()
                with self._lock:
                    if control_index == count-1 and not self.physics.terminal_reason:
                        self._status = "completed_"+self.mode+"_read_only"
                    if len(states) == self.physics.substeps or self.physics.terminal_reason:
                        self._publish(states[-1])
                if len(states) == self.physics.substeps or self.physics.terminal_reason:
                    self._notify_snapshots()
                publish_ns += perf_counter_ns()-publish_started_ns
                work_finished_ns = perf_counter_ns()
                total_ns = work_finished_ns-began_ns-waiting_ns
                self._last_chain_ms = total_ns/1e6
                with self._lock:
                    self._latest = dict(self._latest, timing=dict(self._latest["timing"],
                        chain_ms=self._last_chain_ms, controller_steps=self._controller_steps))
                self._notify_snapshots()
                timing = dict(**command.timing_ns, policy_total=policy_ns, physics=physics_ns,
                              history=history_ns, publishing=publish_ns,
                              scheduler=total_ns-policy_ns-physics_ns-history_ns-publish_ns,
                              explicit_wait=waiting_ns, chain_total=total_ns,
                              wait_requested=wait_requested_ns, wait_elapsed=wait_elapsed_ns, wait_overshoot=wait_overshoot_ns,
                              control_wall=work_finished_ns-began_ns)
                tick = RuntimeTick(self.physics.scene_id, self.source.command_id, "official_walking_reference",
                    float(window.timestamps_s[0]), issued_ns,
                    dict(kind="official_g1_csv_identity_adaptation", qpos=window.qpos, qvel=window.qvel,
                         source_times_s=window.timestamps_s-self._source_clock_origin_s),
                    window, command, tuple(states), timing)
                self._recorder.tick(tick)
                if self.physics.terminal_reason or self._stop_event.is_set():
                    return
            with self._lock:
                self._status = "completed_"+self.mode+"_read_only"
                self._latest = dict(self._latest, status=self._status)
            self._notify_snapshots()
        except Exception as error:
            self._record_runtime_failure(error)
        finally:
            if self._mb_worker is not None:
                self._mb_worker.stop()
            self._close_numeric_logs()


def create_lab_runtime(config_path, *, allocation_id):
    """Load real assets and explicit settings; never starts a service or physics thread."""
    settings = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if settings["schema"] != "motionbricks.mb_sonic.runtime_config.v1" or not allocation_id:
        raise ValueError("Expected an explicit lab runtime configuration and parent resource allocation")
    return SonicLabRuntime(settings, allocation_id=allocation_id)
