"""One spawned frozen MB provider, behind a bounded synchronous worker-thread RPC."""

import copy
from dataclasses import dataclass, fields, is_dataclass, replace
import multiprocessing
import os
from pathlib import Path
import pickle
from threading import Event, Lock, Thread
from time import perf_counter, perf_counter_ns
import traceback

import numpy as np

from .scene import load_scene, sha256


def restore_ipc_value(value, memo=None):
    """Restore owned readonly data, preserving repeated immutable graph nodes."""
    if memo is None:
        memo = {}
    compound = isinstance(value, (np.ndarray, dict, tuple, list)) or (is_dataclass(value) and not isinstance(value, type))
    if compound and id(value) in memo:
        return memo[id(value)]
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or not np.isfinite(value).all():
            raise ValueError("Planner IPC requires finite non-object numeric arrays")
        result = np.array(value, copy=True, order="C")
        result.flags.writeable = False
        memo[id(value)] = result
        return result
    if is_dataclass(value) and not isinstance(value, type):
        result = type(value)(**{field.name: restore_ipc_value(getattr(value, field.name), memo)
                               for field in fields(value) if field.init})
        memo[id(value)] = result
        return result
    if isinstance(value, dict):
        result = {}
        memo[id(value)] = result
        result.update((key, restore_ipc_value(item, memo)) for key, item in value.items())
        return result
    if isinstance(value, tuple):
        result = tuple(restore_ipc_value(item, memo) for item in value)
        memo[id(value)] = result
        return result
    if isinstance(value, list):
        result = []
        memo[id(value)] = result
        result.extend(restore_ipc_value(item, memo) for item in value)
        return result
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float, Exception)):
        return value
    raise TypeError(f"Unsupported planner IPC value: {type(value).__name__}")


def _send(connection, message, maximum_bytes):
    payload = pickle.dumps(message, protocol=5)
    if len(payload) > maximum_bytes:
        raise ValueError("Planner IPC message exceeds its explicit byte bound")
    connection.send_bytes(payload)


def _receive(connection, maximum_bytes, timeout_s):
    if not connection.poll(max(0., timeout_s)):
        raise TimeoutError("Planner process RPC exceeded its explicit deadline")
    return restore_ipc_value(pickle.loads(connection.recv_bytes(maxlength=maximum_bytes)))


def _receive_initial(connection, maximum_bytes, timeout_s):
    completed, outcome = Event(), {}

    def receive():
        try:
            outcome["response"] = _receive(connection, maximum_bytes, timeout_s)
        except Exception as error:
            outcome["error"] = error
        finally:
            completed.set()

    Thread(target=receive, name="mb-planner-initialization", daemon=True).start()
    if not completed.wait(timeout_s):
        connection.close()
        raise TimeoutError("Planner initialization exceeded its explicit deadline")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["response"]


def source_identity():
    root = Path(__file__).resolve().parent
    return {name: sha256(root/name) for name in
            ("planner_process.py", "runtime.py", "retarget.py", "mb_reference.py", "reference.py", "constrained_reference.py", "trajectory_reference.py", "state_feedback.py", "reference_stream.py", "root_plan.py", "static_heap.py")}


def _provider_metadata(provider):
    return dict(identity=copy.deepcopy(provider.generator.identity),
        scene_query_receipt=copy.deepcopy(provider.generator.scene_query_receipt),
        scene_manifest_sha256=provider.scene_manifest_sha256,
        robot_model_sha256=provider.robot_model_sha256, rig_id=provider.rig_id,
        execution=provider.execution, conditioning_mode=provider.conditioning_mode,
        consumer_horizon_s=provider.consumer_horizon_s, cache=provider.cache_description(),
        process=dict(pid=os.getpid(), parent_pid=os.getppid(), start_method="spawn",
                     source_sha256=source_identity(), gpu_provider_loads=1))


def serve_provider(connection, provider, *, maximum_bytes, callback_timeout_s):
    """Serve a real provider; the caller owns construction and the sole CUDA model."""
    try:
        while True:
            request = restore_ipc_value(pickle.loads(connection.recv_bytes(maxlength=maximum_bytes)))
            request_id, operation, scope = request["request_id"], request["operation"], request["scope"]
            received_ns = perf_counter_ns()

            def simulation_time():
                _send(connection, dict(kind="simulation_time", request_id=request_id, scope=scope), maximum_bytes)
                response = _receive(connection, maximum_bytes, callback_timeout_s)
                if (response["kind"] != "simulation_time_result" or response["request_id"] != request_id
                        or response["scope"] != scope or not np.isfinite(response["time_s"])):
                    raise ValueError("Planner simulation-clock callback identity differs")
                return float(response["time_s"])

            try:
                arguments = request["arguments"]
                if operation == "propose":
                    provider.request_evidence = None
                    command = arguments["command"]
                    if (command["scene_manifest_sha256"] != provider.scene_manifest_sha256
                            or scope["scene_manifest_sha256"] != provider.scene_manifest_sha256
                            or scope["episode_id"] != command["episode_id"]
                            or scope["session_id"] != command["session_id"]
                            or scope["command_seq"] != command["command_seq"]):
                        raise ValueError("Planner request belongs to a different scene/session/command")
                    history_started_ns = perf_counter_ns()
                    snapshot = arguments["snapshot"]
                    if snapshot.reference_source is not None and provider.reference_stream_policy is None:
                        history = provider.reference_history_builder(snapshot.reference_source,
                            end_time_s=snapshot.history_time_s, representation=provider.retargeter.rep)
                        snapshot = replace(snapshot, reference_history=history,
                            history_time_s=float(history.timestamps_s[-1]), reference_source=None)
                        arguments = dict(arguments, snapshot=snapshot)
                    history_ready_ns = perf_counter_ns()
                    value = provider.propose(current_sim_time=simulation_time, **arguments)
                    clocks = dict(value.clocks_ns, child_history_started=history_started_ns,
                                  child_history_ready=history_ready_ns)
                    history_ns = history_ready_ns-history_started_ns
                    value = replace(value, clocks_ns=clocks, packet=replace(value.packet, clocks_ns=clocks),
                        timing_ns=dict(value.timing_ns, child_reference_history=history_ns,
                            provider_total=value.timing_ns["provider_total"]+history_ns))
                elif operation == "prepare":
                    value = provider.prepare(**arguments)
                elif operation == "bind_scene":
                    if sha256(arguments["manifest_path"]) != arguments["manifest_sha256"]:
                        raise ValueError("Planner canonical manifest changed before scene binding")
                    provider.bind_scene(load_scene(arguments["manifest_path"]))
                    value = None
                elif operation == "close":
                    _send(connection, dict(kind="result", request_id=request_id, scope=scope,
                        value=None, evidence=None, received_ns=received_ns, reply_started_ns=perf_counter_ns()), maximum_bytes)
                    return
                else:
                    raise ValueError("Unknown planner process operation")
                result = dict(kind="result", request_id=request_id, scope=scope,
                    value=value, evidence=None,
                    received_ns=received_ns, reply_started_ns=perf_counter_ns())
                if operation in ("prepare", "bind_scene"):
                    result["metadata"] = _provider_metadata(provider)
                _send(connection, result, maximum_bytes)
            except Exception as error:
                _send(connection, dict(kind="error", request_id=request_id, scope=scope,
                    error=error, traceback=traceback.format_exc(), evidence=provider.request_evidence,
                    received_ns=received_ns, reply_started_ns=perf_counter_ns()), maximum_bytes)
    except EOFError:
        return
    finally:
        connection.close()


def _planner_main(connection, settings, manifest_path, expected_sources, maximum_bytes, callback_timeout_s):
    try:
        if source_identity() != expected_sources:
            raise ValueError("Spawned planner source files differ from the parent's frozen sources")
        from .runtime import FrozenMBProvider

        scene = load_scene(manifest_path)
        if sha256(scene.manifest_path) != settings["scene_manifest_sha256"][scene.manifest.scene_id]:
            raise ValueError("Spawned planner canonical manifest differs from its explicit pin")
        provider = FrozenMBProvider.load(settings, canonical_scene=scene)
        _send(connection, dict(kind="initialized", metadata=_provider_metadata(provider)), maximum_bytes)
    except Exception as error:
        _send(connection, dict(kind="initialization_error", error=error, traceback=traceback.format_exc()), maximum_bytes)
        connection.close()
        return
    try:
        serve_provider(connection, provider, maximum_bytes=maximum_bytes, callback_timeout_s=callback_timeout_s)
    finally:
        if provider._static_heap is not None:
            provider._static_heap.release()


class PlannerRPC:
    """One outstanding call, used only by initialization or the existing MB worker."""

    def __init__(self, connection, *, maximum_bytes, timeout_s):
        self.connection, self.maximum_bytes, self.timeout_s = connection, maximum_bytes, timeout_s
        self._lock, self._request_id = Lock(), 0
        self.failure = None

    def call(self, operation, arguments, *, scope, current_sim_time=None, timeout_s=None):
        with self._lock:
            if self.failure is not None:
                raise RuntimeError(self.failure)
            self._request_id += 1
            request_id = self._request_id
            started_ns = perf_counter_ns()
            duration = self.timeout_s if timeout_s is None else timeout_s
            deadline, completed, outcome = perf_counter()+duration, Event(), {}

            def exchange():
                try:
                    _send(self.connection, dict(kind="request", request_id=request_id,
                        operation=operation, arguments=arguments, scope=scope), self.maximum_bytes)
                    while True:
                        response = _receive(self.connection, self.maximum_bytes, deadline-perf_counter())
                        if response["request_id"] != request_id or response["scope"] != scope:
                            raise ValueError("Planner process response identity differs from its outstanding request")
                        if response["kind"] == "simulation_time":
                            if current_sim_time is None:
                                raise ValueError("Unexpected planner simulation-clock callback")
                            _send(self.connection, dict(kind="simulation_time_result", request_id=request_id,
                                scope=scope, time_s=current_sim_time()), self.maximum_bytes)
                            continue
                        if response["kind"] not in ("result", "error"):
                            raise ValueError("Unknown planner process response")
                        response["ipc_parent_started_ns"] = started_ns
                        response["ipc_parent_received_ns"] = perf_counter_ns()
                        outcome["response"] = response
                        return
                except Exception as error:
                    outcome["error"] = error
                finally:
                    completed.set()

            # A stalled pipe write/header read must not trap the worker's stop/join.
            thread = Thread(target=exchange, name="mb-planner-rpc", daemon=True)
            thread.start()
            if not completed.wait(max(0., deadline-perf_counter())):
                outcome = dict(error=TimeoutError("Planner process RPC exceeded its explicit deadline"))
            if "error" in outcome:
                error = outcome["error"]
                self.failure = f"planner_process_transport:{type(error).__name__}:{error}"
                self.connection.close()
                raise error
            return outcome["response"]


@dataclass(frozen=True)
class GeneratorMetadata:
    identity: dict
    scene_query_receipt: dict


class ProcessMBProvider:
    """Parent CPU seed/history adapter; all live MB proposals execute in one child."""

    def __init__(self, settings, scene, retargeter, process, rpc, metadata):
        self.settings, self.retargeter, self._process, self._rpc = settings, retargeter, process, rpc
        self._metadata = metadata
        self.execution, self.conditioning_mode = metadata["execution"], metadata["conditioning_mode"]
        self.robot_model_sha256, self.rig_id = metadata["robot_model_sha256"], metadata["rig_id"]
        self.scene_manifest_sha256 = metadata["scene_manifest_sha256"]
        self.consumer_horizon_s = metadata["consumer_horizon_s"]
        self.estimate_actual_diagnostics = settings["estimate_actual_diagnostics"]
        self.preparation_report, self.request_evidence = None, None
        self.failure_evidence = None
        self._closed = False

    @classmethod
    def load(cls, settings, *, canonical_scene):
        import torch
        from motionbricks.data.native_mb_data import build_native_representation
        from .retarget import CalibratedSomaG1Adapter

        options = settings["planner_process"]
        maximum = options["maximum_message_bytes"]
        if (type(maximum) is not int or maximum <= 0
                or any(not np.isfinite(options[key]) or options[key] <= 0
                       for key in ("startup_timeout_s", "rpc_timeout_s", "shutdown_timeout_s"))):
            raise ValueError("Spawned planner requires explicit bounded message sizes and positive lifecycle deadlines")
        if sha256(settings["calibration_path"]) != settings["calibration_sha256"]:
            raise ValueError("Parent CPU calibration differs from the explicit pin")
        # Preserve the original loader's CPU history execution contract in the parent.
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        retargeter = CalibratedSomaG1Adapter.load(settings["calibration_path"],
            robot_model_path=settings["robot_model"], representation=build_native_representation(),
            limb_rotation_source=settings.get('retarget_limb_rotation_source', 'markers'))
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        sources = source_identity()
        process = context.Process(target=_planner_main, name="mb-frozen-planner", daemon=False,
            args=(child, settings, str(canonical_scene.manifest_path), sources, maximum, options["rpc_timeout_s"]))
        try:
            process.start()
        except Exception:
            parent.close()
            child.close()
            raise
        child.close()
        try:
            response = _receive_initial(parent, maximum, options["startup_timeout_s"])
            if response["kind"] == "initialization_error":
                response["error"].remote_traceback = response["traceback"]
                response["error"].args += (response["traceback"],)
                raise response["error"]
            if response["kind"] != "initialized" or response["metadata"]["process"]["source_sha256"] != sources:
                raise ValueError("Spawned planner initialization identity differs")
            metadata = response["metadata"]
            if (metadata["scene_manifest_sha256"] != sha256(canonical_scene.manifest_path)
                    or metadata["robot_model_sha256"] != sha256(settings["robot_model"])):
                raise ValueError("Spawned planner geometry/robot identity differs from the parent")
            return cls(settings, canonical_scene, retargeter, process,
                PlannerRPC(parent, maximum_bytes=maximum, timeout_s=options["rpc_timeout_s"]), metadata)
        except Exception:
            parent.close()
            process.join(options["shutdown_timeout_s"])
            if process.is_alive():
                process.terminate()
                process.join(options["shutdown_timeout_s"])
            raise

    @property
    def generator(self):
        return GeneratorMetadata(copy.deepcopy(self._metadata["identity"]),
                                 copy.deepcopy(self._metadata["scene_query_receipt"]))

    def cache_description(self):
        return dict(copy.deepcopy(self._metadata["cache"]), planner_process=copy.deepcopy(self._metadata["process"]))

    def _call(self, operation, arguments, *, scope=None, current_sim_time=None):
        if self._closed:
            raise RuntimeError("The explicitly closed planner process cannot be restarted")
        if scope is None:
            scope = dict(scene_manifest_sha256=self.scene_manifest_sha256)
        options = self.settings["planner_process"]
        timeout = (options["startup_timeout_s"] if operation in ("prepare", "bind_scene") else
                   options["shutdown_timeout_s"] if operation == "close" else None)
        try:
            response = self._rpc.call(operation, arguments, scope=scope, current_sim_time=current_sim_time,
                                      timeout_s=timeout)
        except Exception as error:
            self.failure_evidence = dict(operation=operation, request_id=self._rpc._request_id,
                scope=copy.deepcopy(scope), error_type=type(error).__name__, error=str(error),
                child_traceback=None, parent_traceback=traceback.format_exc(),
                parent_failure_ns=perf_counter_ns(), child_pid=self._process.pid)
            raise
        self.request_evidence = response["evidence"]
        if response["kind"] == "error":
            self.failure_evidence = dict(operation=operation, request_id=response["request_id"],
                scope=copy.deepcopy(scope), error_type=type(response["error"]).__name__,
                error=str(response["error"]), child_traceback=response["traceback"],
                child_received_ns=response["received_ns"], child_reply_started_ns=response["reply_started_ns"],
                parent_received_ns=response["ipc_parent_received_ns"], child_pid=self._process.pid)
            response["error"].remote_traceback = response["traceback"]
            raise response["error"]
        if "metadata" in response:
            self._metadata = response["metadata"]
        return response

    def bind_scene(self, scene):
        expected = sha256(scene.manifest_path)
        response = self._call("bind_scene", dict(manifest_path=str(scene.manifest_path), manifest_sha256=expected))
        if response["metadata"]["scene_manifest_sha256"] != expected:
            raise ValueError("Planner bound a different canonical scene")
        self.scene_manifest_sha256, self.preparation_report = expected, None

    def prepare(self, history, *, actual_root_xyz_m, desired_heading_rad, initial_qpos=None):
        response = self._call("prepare", dict(history=history, actual_root_xyz_m=actual_root_xyz_m,
            desired_heading_rad=desired_heading_rad, initial_qpos=initial_qpos))
        self.preparation_report = dict(response["value"],
            cpu_history_observer_location="planner_child",
            planner_process=copy.deepcopy(self._metadata["process"]))
        return copy.deepcopy(self.preparation_report)

    def propose(self, snapshot, *, command, intent_revision, requested_at_ns, current_sim_time):
        started_ns = perf_counter_ns()
        self.request_evidence, self.failure_evidence = None, None
        if not self.estimate_actual_diagnostics and self.conditioning_mode != "B2":
            initial_boundary = (snapshot.previous_reference_packet is None
                and self.settings.get('reference_constraints', {}).get('schema')
                    == 'motionbricks.mb_sonic.trajectory_reference.v1')
            if initial_boundary or self.settings.get('reference_stream_policy') is not None:
                from .reference import ExecutedHistory

                executed = snapshot.executed_history
                snapshot = replace(snapshot, executed_history=ExecutedHistory(executed.scene_id,
                    samples=[executed.samples[-1]], body_names=executed.body_names))
            else:
                snapshot = replace(snapshot, executed_history=None)
        scope = dict(scene_manifest_sha256=self.scene_manifest_sha256, episode_id=command["episode_id"],
            session_id=command["session_id"], command_seq=command["command_seq"],
            intent_revision=intent_revision, state_version=snapshot.state_version)
        response = self._call("propose", dict(snapshot=snapshot, command=command,
            intent_revision=intent_revision, requested_at_ns=requested_at_ns),
            scope=scope, current_sim_time=current_sim_time)
        update = response["value"]
        if (update.packet.episode_id != command["episode_id"] or update.packet.session_id != command["session_id"]
                or update.packet.generated_from_command_seq != command["command_seq"]
                or update.packet.state_version != snapshot.state_version
                or update.packet.scene_manifest_sha256 != self.scene_manifest_sha256
                or update.packet.robot_model_sha256 != self.robot_model_sha256
                or update.clocks_ns["requested"] != requested_at_ns):
            raise ValueError("Planner ReferencePacket identity differs across the process boundary")
        packet, reference = update.packet, update.adapted_reference
        if (not np.array_equal(packet.future_timestamps, reference.timestamps_s)
                or not np.array_equal(packet.root_pose, reference.qpos[:, :7])
                or not np.array_equal(packet.body_joint_positions, reference.joint_pos)
                or not np.array_equal(packet.body_joint_velocities, reference.joint_vel)
                or packet.valid_until != reference.trusted_until_s
                or update.proposal.source.scene_id != command["scene_id"]):
            raise ValueError("Planner packet/reference numeric payload differs across IPC")
        finished_ns = perf_counter_ns()
        clocks = dict(update.clocks_ns, ipc_parent_started=started_ns,
            ipc_child_received=response["received_ns"], ipc_child_reply_started=response["reply_started_ns"],
            ipc_parent_received=response["ipc_parent_received_ns"], ipc_validated=finished_ns)
        child_total = update.timing_ns["provider_total"]
        return replace(update, clocks_ns=clocks, packet=replace(update.packet, clocks_ns=clocks),
            timing_ns=dict(update.timing_ns, child_provider_total=child_total,
                ipc_and_parent_work=finished_ns-started_ns-child_total, provider_total=finished_ns-started_ns))

    def close(self):
        if self._closed:
            return
        error = None
        try:
            if self._process.is_alive():
                self._call("close", {})
        except Exception as caught:
            error = caught
        finally:
            self._closed = True
            self._rpc.connection.close()
            timeout = self.settings["planner_process"]["shutdown_timeout_s"]
            self._process.join(timeout)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout)
                error = RuntimeError("Planner process exceeded shutdown deadline and was explicitly terminated")
        if self._process.is_alive():
            raise RuntimeError("Planner process remained alive after explicit termination")
        if error is not None:
            raise error
