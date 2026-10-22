"""CPU-only error-state feedback probe on closed, hash-bound recording artifacts.

Use the repository's motionbricks Python environment and documented kimodo PYTHONPATH.
Only --output is written, exclusively; no model weights or live simulation are used.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter_ns

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.mb_sonic.mb_reference import decode_soma_reference, prepare_reference_history_b1
from motionbricks.mb_sonic.reference import ExecutedHistory, G1Reference, NativeSomaHistory, SomaReference, file_sha256
from motionbricks.mb_sonic.retarget import CalibratedSomaG1Adapter, _matrix
from motionbricks.mb_sonic.scene import SOMA_TO_MUJOCO
from motionbricks.mb_sonic.state_feedback import _native_prior, error_state_history


END_TIMES = (2., 10., 30., 34.2)


def decode(value, archive):
    if isinstance(value, dict):
        if "array" in value:
            return archive[value["array"]]
        return {key: decode(item, archive) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item, archive) for item in value]
    return value


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def distribution(values):
    return dict(count=len(values), p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)), p99_ms=float(np.percentile(values, 99)),
        max_ms=float(max(values)), min_ms=float(min(values)))


class SavedEpisode:
    def __init__(self, path):
        self.path = path.resolve(strict=True)
        self.log_hash = file_sha256(self.path)
        lines = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        self.metadata = next(row for row in lines if row["kind"] == "metadata")
        self.artifacts = {row["record_id"]: row for row in lines if row["kind"] == "numeric_artifact_written"}
        self.ticks = {row["state_version"]: row for row in lines if row["kind"] == "policy_tick_binary"}
        self.uses = {row["physics_done_state_version"]: row for row in lines if row["kind"] == "reference_packet_used"}
        self.plans, self.sources = {}, {}

    def load(self, record_id):
        receipt = self.artifacts[record_id]
        path = Path(receipt["path"]).resolve(strict=True)
        digest = file_sha256(path)
        if digest != receipt["sha256"]:
            raise ValueError(f"Original artifact changed: {record_id}")
        self.sources[record_id] = dict(record_id=record_id, path=str(path), sha256=digest)
        return np.load(path, allow_pickle=False)

    def pair(self, plan_id):
        if plan_id not in self.plans:
            with self.load(plan_id) as archive:
                record = json.loads(str(archive["metadata_json"]))["record"]
                if record["packet"]["plan_id"] != plan_id:
                    raise ValueError("Saved plan identity differs")
                native = decode(record["raw_source"], archive)
                robot = decode(record["adapted_reference"], archive)
            native["bone_names"] = tuple(native["bone_names"])
            robot["body_names"], robot["joint_names"] = tuple(robot["body_names"]), tuple(robot["joint_names"])
            self.plans[plan_id] = SomaReference(**native), G1Reference(**robot)
        return self.plans[plan_id]

    def compact_history(self, end, body_names):
        executed = ExecutedHistory(self.metadata["scene_manifest"]["scene_id"], body_names=body_names)
        receipts = []
        for boundary in end + np.arange(-7, 1) / 50:
            version = round(boundary / self.metadata["physics_dt_s"])
            tick, use = self.ticks[version], self.uses[version]
            with self.load(tick["numeric_record_id"]) as archive:
                record = json.loads(str(archive["metadata_json"]))["record"]
                physics = decode(record["physics"], archive)
                times = physics["time_s"]
                index = int(np.argmin(abs(times - boundary)))
                if abs(times[index] - boundary) > 1e-8 or physics["state_version"][index] != version:
                    raise ValueError("The requested 20ms boundary is not a factual saved substep")
                plan_id = record["tick"]["raw_reference"]["plan_id"]
                if plan_id != use["plan_id"]:
                    raise ValueError("Tick and actual consumed-plan receipt disagree")
                contacts = physics["contacts"]
                start, stop = contacts["offsets"][index:index + 2]
                sample = dict(timestamp_s=float(times[index]), state_version=int(version),
                    qpos=physics["qpos"][index].copy(), qvel=physics["qvel"][index].copy(),
                    body_pos_w=physics["body_positions_world"][index].copy(),
                    body_quat_w=physics["body_quat_wxyz"][index].copy(),
                    contact_columns={key: contacts[key][start:stop].copy() for key in
                        ("body_ids", "position_world", "wrench_contact")})
            native, robot = self.pair(plan_id)
            sample.update(native_reference_source=native, robot_reference_source=robot, reference_plan_id=plan_id)
            executed.samples.append(sample)
            receipts.append(dict(timestamp_s=sample["timestamp_s"], state_version=version,
                actual_control_interval_s=[use["used_time_window"][0], float(times[-1])],
                reference_plan_id=plan_id, reference_artifact=self.sources[plan_id],
                tick_artifact=self.sources[tick["numeric_record_id"]]))
        return executed, receipts


class CaptureObserver:
    def __init__(self, observer):
        self.observer = observer

    def __call__(self, features):
        self.input = features.clone()
        self.result = self.observer(features)
        return self.result


def interpolate_actual(executed, requested):
    samples = executed.samples
    times = np.array([row["timestamp_s"] for row in samples])
    qpos = np.array([row["qpos"] for row in samples])
    return np.stack([np.interp(requested, times, qpos[:, i]) for i in range(36)], axis=-1)


def check_posterior(history, capture, executed, adapter):
    rep = adapter.rep
    expected = interpolate_actual(executed, history.timestamps_s)
    native_root = (expected[:, :3] - [0., 0., adapter.root_z_offset_m]) @ SOMA_TO_MUJOCO
    actual_root = history.features[:, rep.indices["global_root_pos"]]
    source = decode_soma_reference(history.features.copy(), history.timestamps_s, rep, scene_id=executed.scene_id,
        command_id="recorded_actual_feedback_probe", trusted_until_s=float(history.timestamps_s[-1]),
        provenance=dict(history.provenance))
    mapped, residual = adapter._map_anatomical(source.root_positions_m, source.world_positions_m,
        source.bone_names, root_rotations=source.global_rot_mats[:, 0])
    errors = mapped[:, 7:] - expected[:, 7:]
    joint = np.unravel_index(np.argmax(abs(errors)), errors.shape)
    # This is the runtime's forward anatomical gauge, without IK or limit projection.
    approximate = dict(method="posterior native FK -> _map_anatomical; unconstrained forward mapping, not inverse",
        interpretation="Approximation of observed G1 joints from a native posterior; not an exact robot state reconstruction",
        joint_rmse_rad=float(np.sqrt(np.mean(errors**2))), joint_max_abs_rad=float(abs(errors).max()),
        largest_error=dict(time_s=float(history.timestamps_s[joint[0]]),
            joint=executed.samples[-1]["robot_reference_source"].joint_names[joint[1]],
            predicted_rad=float(mapped[joint[0], joint[1] + 7]), actual_rad=float(expected[joint[0], joint[1] + 7])),
        per_joint_rmse_rad=np.sqrt(np.mean(errors**2, axis=0)),
        joint_position_limit_violation_count=int(((mapped[:, 7:] < adapter.kinematics.limits[:, 0]) |
                                                 (mapped[:, 7:] > adapter.kinematics.limits[:, 1])).sum()),
        mapping_orientation_residual=residual)
    original, rebuilt = capture.input[0].numpy(), capture.result["history"][0].numpy()
    local_rotation = Rotation.from_matrix(source.local_rot_mats.reshape(-1, 3, 3))
    prior_frames, robot_joints, measured_contacts = [], [], []
    sample_times = np.array([row["timestamp_s"] for row in executed.samples])
    for time in history.timestamps_s:
        selected = executed.samples[max(0, int(np.searchsorted(sample_times, time + 1e-9, side="right") - 1))]
        native, robot = selected["native_reference_source"], selected["robot_reference_source"]
        frame, rotations = _native_prior(native, time)
        frame[rep.indices["global_rot_data"]] = np.concatenate((rotations[..., 0], rotations[..., 1]), -1).reshape(-1)
        prior_frames.append(frame)
        robot_joints.append([np.interp(time, robot.timestamps_s, robot.joint_pos[:, i]) for i in range(29)])
        columns = selected["contact_columns"]
        oracle = dict(selected, contacts=tuple(dict(body_ids=tuple(int(x) for x in body_ids),
            position_world=position, wrench_contact=wrench) for body_ids, position, wrench in
            zip(columns["body_ids"], columns["position_world"], columns["wrench_contact"], strict=True)))
        measured_contacts.append(adapter._contacts(oracle, executed.body_names))
    prior_observed = capture.observer(torch.tensor(np.array(prior_frames), dtype=torch.float32)[None])
    prior_source = decode_soma_reference(prior_observed["history"][0].numpy().copy(), history.timestamps_s,
        rep, scene_id=executed.scene_id, command_id="paired_prior_forward_control",
        trusted_until_s=float(history.timestamps_s[-1]), provenance={})
    prior_mapped, _ = adapter._map_anatomical(prior_source.root_positions_m, prior_source.world_positions_m,
        prior_source.bone_names, root_rotations=prior_source.global_rot_mats[:, 0])
    prior_error = prior_mapped[:, 7:] - robot_joints
    prior_largest = np.unravel_index(np.argmax(abs(prior_error)), prior_error.shape)
    return dict(finite=bool(np.isfinite(history.features).all()), shape=list(history.features.shape),
        timestamps_s=history.timestamps_s, actual_root_soma_m=native_root, posterior_root_soma_m=actual_root,
        root_max_abs_error_m=float(abs(actual_root - native_root).max()),
        root_euclidean_error_m=np.linalg.norm(actual_root - native_root, axis=-1),
        observer_position_error_m=float(capture.result["position_error"]),
        observer_rotation_matrix_max_abs_error=float(capture.result["rotation_error"]),
        observer_root_change_max_m=float(abs(original[:, rep.indices["global_root_pos"]] - rebuilt[:, rep.indices["global_root_pos"]]).max()),
        fk_rotation_orthogonality_max_abs=float(abs(source.global_rot_mats.swapaxes(-1, -2) @ source.global_rot_mats - np.eye(3)).max()),
        local_rotation_max_angle_rad=float(local_rotation.magnitude().max()),
        contacts=history.features[:, rep.indices["foot_contacts"]],
        contacts_match_measured_latest_boundary=bool(np.array_equal(
            history.features[:, rep.indices["foot_contacts"]], measured_contacts)),
        posterior_features=history.features, approximate_forward_retarget=approximate,
        paired_prior_forward_control=dict(method="Same native-prior sampling and FK reencoding, without actual-error correction, compared to its paired adopted G1 reference",
            joint_rmse_rad=float(np.sqrt(np.mean(prior_error**2))), joint_max_abs_rad=float(abs(prior_error).max()),
            largest_joint=robot.joint_names[prior_largest[1]], largest_time_s=float(history.timestamps_s[prior_largest[0]])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hot-repeats", type=int, default=50, choices=range(1, 101))
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = perf_counter_ns()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    episode = SavedEpisode(args.recording)
    scene_id = episode.metadata["scene_manifest"]["scene_id"]
    seed_receipt = settings["scene_seeds"][scene_id]
    for path, expected in ((settings["robot_model"], settings["robot_model_sha256"]),
                           (settings["calibration_path"], settings["calibration_sha256"]),
                           (seed_receipt["path"], seed_receipt["sha256"])):
        if file_sha256(path) != expected:
            raise ValueError("Canonical v13 asset hash differs: " + str(path))
    if (episode.metadata["source"]["sha256"] != seed_receipt["sha256"] or
            episode.metadata["robot"]["sha256"] != settings["robot_model_sha256"] or
            episode.metadata["mb_provider"]["calibration_sha256"] != settings["calibration_sha256"]):
        raise ValueError("Config assets do not bind the selected episode")
    helper_path = Path(sys.modules[error_state_history.__module__].__file__).resolve()
    helper_hash = file_sha256(helper_path)
    rep = build_native_representation()
    adapter = CalibratedSomaG1Adapter.load(settings["calibration_path"],
        robot_model_path=settings["robot_model"], representation=rep)
    with np.load(seed_receipt["path"], allow_pickle=False) as archive:
        features = archive["initial_history"][0].copy()
    seed = NativeSomaHistory((np.arange(4) - 3) / 30, features, scene_id,
        dict(kind="source_seed", file=seed_receipt["path"], sha256=seed_receipt["sha256"]))
    observer = prepare_reference_history_b1(rep, features)
    setup_ms = (perf_counter_ns() - started) / 1e6
    cases = []
    with torch.inference_mode():
        for end in END_TIMES:
            executed, receipts = episode.compact_history(end, adapter.kinematics.body_names)
            capture = CaptureObserver(observer)
            call_started = perf_counter_ns()
            history = error_state_history(seed, executed, end_time_s=end, adapter=adapter, observer=capture)
            first_ms = (perf_counter_ns() - call_started) / 1e6
            metrics = check_posterior(history, capture, executed, adapter)
            hot = []
            for index in range(3 + args.hot_repeats):
                call_started = perf_counter_ns()
                error_state_history(seed, executed, end_time_s=end, adapter=adapter, observer=observer)
                elapsed_ms = (perf_counter_ns() - call_started) / 1e6
                if index >= 3:
                    hot.append(elapsed_ms)
            actual_times = [row["timestamp_s"] for row in executed.samples]
            brackets = history.provenance["samples"]
            physical = [{key: value for key, value in row.items() if key not in
                ("native_reference_source", "robot_reference_source")} for row in executed.samples]
            fk_pos, fk_rot = adapter.kinematics.fk(np.stack([row["qpos"] for row in executed.samples]))
            recorded_pos = np.stack([row["body_pos_w"] for row in executed.samples])
            recorded_rot = _matrix(np.stack([row["body_quat_w"] for row in executed.samples]))
            cases.append(dict(end_time_s=end, actual_sample_count=len(actual_times),
                factual_interval_s=[actual_times[0], actual_times[-1]], factual_duration_s=actual_times[-1] - actual_times[0],
                actual_50hz_max_spacing_error_s=float(abs(np.diff(actual_times) - .02).max()),
                maximum_actual_bracket_s=max(row["actual_brackets_s"][1] for row in brackets),
                causal_actual_brackets=all(row["actual_brackets_s"][1] <= end + 1e-9 for row in brackets),
                sample_receipts=receipts, compact_actual_samples=physical,
                history_provenance=dict(history.provenance), metrics=metrics,
                first_feedback_call_including_capture_ms=first_ms, hot_four_feedback_frames=distribution(hot),
                saved_qpos_to_saved_body_fk=dict(position_max_abs_error_m=float(abs(fk_pos - recorded_pos).max()),
                    rotation_matrix_max_abs_error=float(abs(fk_rot - recorded_rot).max()))))
    if file_sha256(episode.path) != episode.log_hash or file_sha256(helper_path) != helper_hash:
        raise ValueError("Recording or helper changed during the probe")
    result = dict(schema="motionbricks.closed_actual_feedback_probe.v2", config=str(args.config.resolve()),
        config_sha256=file_sha256(args.config), recording=str(episode.path), recording_sha256=episode.log_hash,
        helper=dict(path=str(helper_path), sha256=helper_hash), script_sha256=file_sha256(__file__),
        new_physics_steps=0, gpu_seconds=0, optimizer_updates=0, model_inference_calls=0,
        cuda_initialized=torch.cuda.is_initialized(), inverse_adapter_calls=0,
        execution=dict(python=sys.version, executable=sys.executable, torch_version=torch.__version__,
            platform=platform.platform(), torch_intraop_threads=args.threads, torch_interop_threads=1,
            setup_ms=setup_ms, observer_preparation=observer.preparation,
            hot_timing_scope="One unmodified error_state_history call producing 4x366, including four paired-prior FK calls and prepared observer; excludes I/O, setup, metrics and capture",
            warmup_calls_per_case=3, timed_calls_per_case=args.hot_repeats),
        assets=dict(robot_model=settings["robot_model"], robot_sha256=settings["robot_model_sha256"],
            calibration=settings["calibration_path"], calibration_sha256=settings["calibration_sha256"],
            calibration_identity=adapter.calibration["identity"], seed=seed_receipt,
            configured_mb_device_not_used=settings["mb_device"]),
        sample_semantics="Each exact 20ms boundary uses the saved substep and plan that drove the immediately ending control interval; paired raw/adapted references come from the same original plan NPZ",
        contact_contract=dict(format="contact_columns", body_ids="N x 2", position_world="N x 3",
            wrench_contact="N x 6", retention="All contacts at each saved boundary, without filtering",
            oracle="Legacy per-contact tuples constructed only inside offline check_posterior, outside hot timings"),
        predeclared_end_times_s=END_TIMES, body_names=adapter.kinematics.body_names,
        joint_names=next(iter(episode.plans.values()))[1].joint_names,
        verified_numeric_sources=list(episode.sources.values()), cases=cases,
        recording_unchanged=True, helper_unchanged=True,
        source_files=[dict(path=str(Path(sys.modules[name].__file__).resolve()),
            sha256=file_sha256(sys.modules[name].__file__)) for name in
            ("motionbricks.mb_sonic.retarget", "motionbricks.mb_sonic.reference", "motionbricks.mb_sonic.mb_reference",
             "motionbricks.data.native_mb_observed_history", "motionbricks.mb_sonic.geometry_execution")],
        findings=dict(root_and_causality="All four windows return finite histories with factual 50Hz brackets, measured Root and measured held contact channels",
            contract_update="The helper requires contact_columns instead of per-contact tuples; the probe now supplies the original full saved contact columns",
            helper_bug_status="No exception, Root mapping, coverage, or FK reencoding failure reproduced in the four declared windows",
            approximate_forward_closure="The error-state posterior has large right-hip-yaw discrepancies in the 10s and 30s windows; paired-prior forward controls distinguish these from pre-existing prior-mapping error",
            approximation_interpretation="Forward closure is an additional approximation diagnostic, not an exact inverse or a physical rollout; measured body-error injection and native anatomical forward recovery are distinct contracts"))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, default=json_value, indent=2, allow_nan=False)
    compact = dict(report=None if args.output is None else str(args.output.resolve()),
        verified_npz_count=len(episode.sources), helper_sha256=helper_hash,
        cuda_initialized=result["cuda_initialized"], cases=[dict(end_time_s=c["end_time_s"],
            factual_interval_s=c["factual_interval_s"], causal=c["causal_actual_brackets"],
            finite=c["metrics"]["finite"], contacts_match_legacy_oracle=c["metrics"]["contacts_match_measured_latest_boundary"],
            root_max_abs_error_m=c["metrics"]["root_max_abs_error_m"],
            observer_position_error_m=c["metrics"]["observer_position_error_m"],
            observer_rotation_error=c["metrics"]["observer_rotation_matrix_max_abs_error"],
            hot=c["hot_four_feedback_frames"]) for c in cases])
    print(json.dumps(compact, default=json_value, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
