"""Frozen trained MotionBricks proposals, with raw and trusted coverage separate."""

from dataclasses import dataclass, replace
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from .reference import NativeSomaHistory, RawMBBlock, ReferenceCoverageError, SomaReference, file_sha256


TRAINED_WEIGHTS_SHA256 = "0f5d8e2b05b8ec518a104a4979cca70f23efa64080abfc843f874634de5a9646"
TRAINED_SOURCE_SHA256 = "b960fbe9f792cf46333c23762a8f047259132acd5f6b595bbc726dffa8b40c19"
TRAINED_BUNDLE_SHA256 = "0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d"
TRAINED_MODEL_STATE_SHA256 = "a6366a7e8522371be01d632efa6a952839a690a1749e14d6fb65819ff584b50c"
TRAINED_CODEC_STATE_SHA256 = "ee4a08d79a5c5268799352507c9da8d323e5e81a40c36828fc68fe35f042f109"


def native_precision_contract():
    import torch

    return dict(matmul=torch.get_float32_matmul_precision(),
                cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32))


def native_host_contract():
    import torch

    return dict(torch_intraop_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads())


def _check_plan_blocks(plan_blocks):
    if type(plan_blocks) is not int or plan_blocks not in (1, 2):
        raise ValueError("Full-plan execution requires an explicit one or two AR8 calls")


def world_xy_command_to_soma(velocity_xy_mps, heading_world_rad):
    """MuJoCo XY/yaw -> native XZ/Y-yaw (zero native yaw faces +Z)."""
    from .scene import mujoco_to_soma

    velocity = np.asarray(velocity_xy_mps, dtype=np.float64)
    if velocity.shape != (2,) or not np.isfinite(velocity).all() or not np.isfinite(heading_world_rad):
        raise ValueError("Finite world XY velocity and heading are required")
    native_velocity = mujoco_to_soma([velocity[0], velocity[1], 0.])
    native_forward = mujoco_to_soma([np.cos(heading_world_rad), np.sin(heading_world_rad), 0.])
    return native_velocity[[0, 2]], float(np.arctan2(native_forward[0], native_forward[2]))


def reference_history_b1_oracle(source: SomaReference, *, end_time_s, representation):
    """Unchanged eager/full-sequence oracle, for tests and diagnostics only."""
    from scipy.spatial.transform import Rotation, Slerp
    import torch
    from motionbricks.data.native_mb_observed_history import reencode_played_history

    times = float(end_time_s) + (np.arange(4) - 3) / 30
    source_times = source.timestamps_s
    if (not np.isfinite(times).all() or times[0] < source_times[0] - 1e-9
            or times[-1] > min(source.trusted_until_s, source_times[-1]) + 1e-9):
        raise ReferenceCoverageError("B1 history requires four covered reference timestamps; no extrapolation")
    upper = np.minimum(np.searchsorted(source_times, times), len(source_times) - 1)
    lower = np.maximum(upper - 1, 0)
    nearest = np.where(np.abs(times - source_times[lower]) < np.abs(times - source_times[upper]), lower, upper)
    exact = np.isclose(times, source_times[nearest], atol=1e-9, rtol=0)
    times[exact] = source_times[nearest[exact]]
    left = np.maximum(np.searchsorted(source_times, times, side="right") - 1, 0)
    right = np.minimum(left + 1, len(source_times) - 1)
    span = source_times[right] - source_times[left]
    weight = np.divide(times - source_times[left], span, out=np.zeros(4), where=span > 0)
    features = ((1 - weight[:, None]) * source.raw_features[left]
                + weight[:, None] * source.raw_features[right]).astype(source.raw_features.dtype)
    interpolated = np.flatnonzero(~exact)
    if len(interpolated):
        matrices = np.stack([Slerp(source_times, Rotation.from_matrix(source.global_rot_mats[:, bone]))(
            times[interpolated]).as_matrix() for bone in range(30)], axis=1)
        rotations6d = np.concatenate((matrices[..., 0], matrices[..., 1]), axis=-1).reshape(-1, 180)
        features[np.ix_(interpolated, representation.indices["global_rot_data"])] = rotations6d
        heading = np.interp(times[interpolated], source_times,
                            np.unwrap(np.arctan2(source.raw_features[:, 4], source.raw_features[:, 3])))
        features[interpolated, 3], features[interpolated, 4] = np.cos(heading), np.sin(heading)
    features[exact] = source.raw_features[nearest[exact]]
    checks = {}
    observed = reencode_played_history(torch.tensor(features)[None], representation, tensor_checks=checks)
    return NativeSomaHistory(times, observed[0].detach().cpu().numpy(), source.scene_id, dict(
        kind="legacy_reference_history", conditioning_mode="B1", source_command_id=source.command_id,
        source_brackets=np.stack((left, right), -1).tolist(), interpolation_weights=weight.tolist(),
        sampling="exact_native_frames_or_linear_scalars_with_SO3_global_rotations",
        history_policy="native_played_fk_reencode",
        reencode_position_error_m=float(checks["position_error"]),
        reencode_rotation_error=float(checks["rotation_error"]),
        contacts="reference_predictions_not_measured_forces", reference_root_reanchored=False,
    ))


class PreparedNativeHistoryObserver:
    """One fixed-shape CPU trace of the existing native played-FK observer."""

    def __init__(self, representation, example_features):
        from functools import partial
        import torch
        from motionbricks.data.soma_static_execution import trace_fixed_call
        from .geometry_execution import native_observed_history

        example = example_features.detach() if isinstance(example_features, torch.Tensor) else torch.tensor(example_features)
        if example.ndim == 2:
            example = example[None]
        if example.device.type != "cpu" or example.dtype != torch.float32 or example.shape != (1, 4, 366):
            raise ValueError("CPU history preparation requires actual FP32 [1,4,366] native features")
        example = example.clone()
        started, rng = perf_counter_ns(), torch.random.get_rng_state().clone()
        native = partial(native_observed_history, representation)
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            expected = native(example)
            function = trace_fixed_call(native, (example,), shared_cuda_constants=True)
            for _ in range(3):
                actual = function(example)
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
        if not torch.equal(rng, torch.random.get_rng_state()):
            raise ValueError("CPU observer preparation changed RNG state")
        self.representation, self.function = representation, function
        self.preparation = dict(execution="prepared_native_cpu_trace", preparation_wall_ns=perf_counter_ns() - started,
                                initial_all_channels_equal=True, cpu_rng_state_equal=True,
                                native_position_error_m=float(actual["position_error"]),
                                native_rotation_error=float(actual["rotation_error"]),
                                numerical_optimization=False, warmup_calls=3, optimizer_steps=0)

    def __call__(self, features):
        import torch

        if features.device.type != "cpu" or features.dtype != torch.float32 or features.shape != (1, 4, 366):
            raise ValueError("Prepared CPU history observer requires FP32 [1,4,366]")
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            result = self.function(features)
        if (not torch.isfinite(features).all() or not torch.isfinite(result["history"]).all()
                or not torch.isfinite(result["position_error"]) or not torch.isfinite(result["rotation_error"])
                or result["position_error"] > 2e-5 or result["rotation_error"] > 2e-5):
            raise ValueError("Prepared CPU observer changed the displayed FK state")
        return result


def prepare_reference_history_b1(representation, example_features):
    """Explicit startup preparation; no compilation or mode selection during snapshots."""
    return PreparedNativeHistoryObserver(representation, example_features)


def reference_history_b1(source: SomaReference, *, end_time_s, representation, observer):
    """Bracket-only SO(3) sampling followed by the prepared native CPU observer."""
    from scipy.spatial.transform import Rotation, Slerp
    import torch

    if observer.representation is not representation:
        raise ValueError("B1 history must use its explicitly prepared native representation")
    times = float(end_time_s) + (np.arange(4) - 3) / 30
    source_times = source.timestamps_s
    if (not np.isfinite(times).all() or times[0] < source_times[0] - 1e-9
            or times[-1] > min(source.trusted_until_s, source_times[-1]) + 1e-9):
        raise ReferenceCoverageError("B1 history requires four covered reference timestamps; no extrapolation")
    upper = np.minimum(np.searchsorted(source_times, times), len(source_times) - 1)
    lower = np.maximum(upper - 1, 0)
    nearest = np.where(np.abs(times - source_times[lower]) < np.abs(times - source_times[upper]), lower, upper)
    exact = np.isclose(times, source_times[nearest], atol=1e-9, rtol=0)
    times[exact] = source_times[nearest[exact]]
    left = np.maximum(np.searchsorted(source_times, times, side="right") - 1, 0)
    right = np.minimum(left + 1, len(source_times) - 1)
    span = source_times[right] - source_times[left]
    weight = np.divide(times - source_times[left], span, out=np.zeros(4), where=span > 0)
    features = ((1 - weight[:, None]) * source.raw_features[left]
                + weight[:, None] * source.raw_features[right]).astype(source.raw_features.dtype)
    interpolated = np.flatnonzero(~exact)
    keyframe_count = 0
    if len(interpolated):
        keyframe_count = len(np.unique(np.r_[left[interpolated],right[interpolated]]))
        a = Rotation.from_matrix(source.global_rot_mats[left[interpolated]].reshape(-1,3,3))
        b = Rotation.from_matrix(source.global_rot_mats[right[interpolated]].reshape(-1,3,3))
        delta = (a.inv()*b).as_rotvec()*np.repeat(weight[interpolated],30)[:,None]
        matrices = (a*Rotation.from_rotvec(delta)).as_matrix().reshape(-1,30,3,3)
        rotations6d = np.concatenate((matrices[..., 0], matrices[..., 1]), axis=-1).reshape(-1, 180)
        features[np.ix_(interpolated, representation.indices["global_rot_data"])] = rotations6d
        heading = np.interp(times[interpolated], source_times,
                            np.unwrap(np.arctan2(source.raw_features[:, 4], source.raw_features[:, 3])))
        features[interpolated, 3], features[interpolated, 4] = np.cos(heading), np.sin(heading)
    features[exact] = source.raw_features[nearest[exact]]
    result = observer(torch.from_numpy(features)[None])
    return NativeSomaHistory(times, result["history"][0].numpy(), source.scene_id, dict(
        kind="legacy_reference_history", conditioning_mode="B1", source_command_id=source.command_id,
        source_brackets=np.stack((left, right), -1).tolist(), interpolation_weights=weight.tolist(),
        sampling="exact_native_frames_or_linear_scalars_with_SO3_global_rotations",
        history_policy="native_played_fk_reencode", execution="bracket_SO3_prepared_native_cpu_trace",
        rotation_keyframe_count=keyframe_count,
        reencode_position_error_m=float(result["position_error"]),
        reencode_rotation_error=float(result["rotation_error"]),
        contacts="reference_predictions_not_measured_forces", reference_root_reanchored=False,
    ))


def validate_conditioning(history, conditioning_mode, query_origin_soma_xyz):
    kind = history.provenance["kind"]
    if kind == "open_loop_generated_context":
        raise ValueError("Internal autoregressive context cannot start a new replan")
    if conditioning_mode == "native":
        if kind == "legacy_reference_history" or query_origin_soma_xyz is not None:
            raise ValueError("Legacy reference history and query overrides require explicit B1 conditioning")
        return None
    if conditioning_mode == "B2" and kind not in ("source_seed", "actual_g1_error_state"):
        raise ValueError("B2 requires measured error-state feedback or the explicit preparation seed")
    if conditioning_mode not in ("B1", "B2") or (conditioning_mode == "B1" and kind not in ("source_seed", "legacy_reference_history")):
        raise ValueError("B1 requires a source seed or explicitly labelled legacy reference history")
    origin = np.asarray(query_origin_soma_xyz, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("B1 requires an explicit finite calibrated SOMA query origin")
    return origin.copy()


def reframe_scene_inputs(history_root, query_root, values, sampler):
    """Keep query-local geometry local; register world points into the model chart."""
    values = list(values)
    delta = query_root[:, -1, :3] - history_root[:, -1, :3]
    for index in (2,6):
        values[index] = values[index]-delta[:,[0,2]][:,None,None,None]
    values[8] = values[8]-delta[:,None,None]
    return tuple(values)


def source_in_scene_frame(source, translation):
    """Register a complete model reference into the physical scene for retargeting."""
    shift=np.asarray(translation,dtype=float)
    if shift.shape!=(3,) or not np.isfinite(shift).all():
        raise ValueError('Reference registration requires a finite XYZ translation')
    features=source.raw_features.copy()
    features[:,:3]+=shift
    return replace(source,raw_features=features,root_positions_m=source.root_positions_m+shift,
        world_positions_m=source.world_positions_m+shift,
        provenance=dict(source.provenance,coordinate_frame='canonical_soma_world_y_up',
            model_to_scene_translation_soma_xyz_m=shift.tolist()))


def proposal_in_frame(proposal, frame, query_origin):
    if frame not in ('scene','reference_chart'):
        raise ValueError('Explicit scene or reference_chart output frame required')
    translation=(np.zeros(3) if query_origin is None else
        np.asarray(query_origin)-proposal.seed_history.features[-1,:3])
    source=(source_in_scene_frame(proposal.source,translation) if frame=='scene' else
        replace(proposal.source,provenance=dict(proposal.source.provenance,
            coordinate_frame='native_reference_chart_y_up',
            model_to_scene_translation_soma_xyz_m=translation.tolist())))
    return replace(proposal,source=source)


def rolling_query_origin(initial_history, current_history, actual_anchor):
    """Current actual anchor plus reference-predicted displacement, never future truth."""
    return current_history[:, -1, :3] + (actual_anchor - initial_history[:, -1, :3])


def inspect_trained_artifacts(artifact: Path, bundle: Path):
    """Read/hash files only; no torch import, model construction or GPU work."""
    artifact, bundle = Path(artifact), Path(bundle)
    receipt = json.loads((artifact / "acceptance.json").read_text(encoding="utf-8"))
    expected = dict(checkpoint_sha256=TRAINED_WEIGHTS_SHA256, bundle_sha256=TRAINED_BUNDLE_SHA256,
                    source_checkpoint_sha256=TRAINED_SOURCE_SHA256,
                    model_state_sha256=TRAINED_MODEL_STATE_SHA256,
                    codec_state_sha256=TRAINED_CODEC_STATE_SHA256,
                    mechanics_only=False, actual_updates=4000, actual_exposures=12800000)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Expected the completed trained full-corpus compact artifact, not the untrained v3 probe")
    weights = artifact / receipt["checkpoint_file"]
    if file_sha256(weights) != TRAINED_WEIGHTS_SHA256 or file_sha256(bundle) != TRAINED_BUNDLE_SHA256:
        raise ValueError("Trained MB checkpoint/bundle bytes differ from the preserved parent artifacts")
    return dict(expected, weights=str(weights.resolve()), bundle=str(bundle.resolve()),
                acceptance_sha256=file_sha256(artifact / "acceptance.json"),
                training_stages=receipt["training_stages"], runtime=receipt["runtime"],
                model_kind="trained_legacy_full_corpus", current_optimizer_steps=0)


def decode_soma_reference(features, timestamps_s, representation, *, scene_id, command_id,
                          trusted_until_s, provenance, timings=None):
    """Decode unnormalized native GLOBAL366 without altering any model channel."""
    import torch

    values = torch.as_tensor(features)
    if values.ndim == 2:
        values = values[None]
    if values.shape != (1, len(timestamps_s), 366):
        raise ValueError("One unnormalized native GLOBAL366 sequence is required")
    transfer_started = perf_counter_ns()
    original = values[0].detach().cpu().numpy().copy()
    raw_transfer_ns = perf_counter_ns() - transfer_started
    fk_started = perf_counter_ns()
    with torch.inference_mode():
        decoded = representation.inverse(values, False, joint_positions_from="global_rot_data", return_all=False)
    fk_dispatch_ns = perf_counter_ns() - fk_started
    transfer_started = perf_counter_ns()
    arrays = {name: decoded[name][0].detach().cpu().numpy() for name in (
        "root_pos", "local_joint_rots", "global_joint_rots", "posed_joints")}
    pose_transfer_ns = perf_counter_ns() - transfer_started
    snapshot_started = perf_counter_ns()
    reference = SomaReference(
        timestamps_s=np.asarray(timestamps_s), raw_features=original,
        bone_names=tuple(representation.skeleton.bone_order_names),
        root_positions_m=arrays["root_pos"], local_rot_mats=arrays["local_joint_rots"],
        global_rot_mats=arrays["global_joint_rots"], world_positions_m=arrays["posed_joints"],
        scene_id=scene_id, command_id=command_id, trusted_until_s=trusted_until_s, provenance=provenance,
    )
    if timings is not None:
        timings.update(raw_cpu_transfer_including_cuda_wait_wall_ns=raw_transfer_ns,
                       native_fk_dispatch_wall_ns=fk_dispatch_ns,
                       pose_cpu_transfer_including_cuda_wait_wall_ns=pose_transfer_ns,
                       immutable_snapshot_wall_ns=perf_counter_ns() - snapshot_started)
    return reference


@dataclass(frozen=True)
class MBProposal:
    source: SomaReference
    raw_blocks: tuple[SomaReference | RawMBBlock, ...]
    seed_history: NativeSomaHistory
    timing_ns: dict


def payload_layout(values):
    """A byte layout preserves every native tensor dtype and bit pattern."""
    import torch

    dtype_names = {torch.float32: "<f4", torch.float64: "<f8", torch.int64: "<i8",
                   torch.int32: "<i4", torch.bool: "|b1"}
    layout, offset = [], 0
    for name, value in sorted(values.items(), key=lambda item: item[1].element_size(), reverse=True):
        layout.append(dict(name=name, shape=list(value.shape), dtype=dtype_names[value.dtype],
                           offset_bytes=offset, count=value.numel(), nbytes=value.numel() * value.element_size()))
        offset += value.numel() * value.element_size()
    return layout


def pack_payload(values, layout):
    import torch

    return torch.cat([values[row["name"]].contiguous().reshape(-1).view(torch.uint8) for row in layout])


def unpack_payload_numpy(payload, layout):
    result = {}
    for row in layout:
        result[row["name"]] = np.frombuffer(payload, dtype=np.dtype(row["dtype"]), count=row["count"],
                                             offset=row["offset_bytes"]).reshape(row["shape"]).copy()
    return result


def unpack_payload_torch(payload, layout):
    import torch

    dtypes = {"<f4": torch.float32, "<f8": torch.float64, "<i8": torch.int64,
              "<i4": torch.int32, "|b1": torch.bool}
    return {row["name"]: payload[row["offset_bytes"]:row["offset_bytes"] + row["nbytes"]]
            .view(dtypes[row["dtype"]]).reshape(row["shape"]) for row in layout}


def audit_legacy_observation(canonical_scene, history, local_observation, volume, *,
                             model_to_scene_translation=(0.,0.,0.)):
    """Measure actual legacy inputs against canonical geometry; no SDF substitution."""
    from .scene import soma_to_mujoco

    root = history[0, -1, :5].detach().cpu().numpy()
    local = local_observation["points"][0].detach().cpu().numpy()
    native_valid = local_observation["valid"][0].detach().cpu().numpy()
    yaw = np.arctan2(root[4], root[3])
    world = np.empty_like(local)
    world[..., 0] = root[0] + np.cos(yaw) * local[..., 0] + np.sin(yaw) * local[..., 2]
    world[..., 2] = root[2] - np.sin(yaw) * local[..., 0] + np.cos(yaw) * local[..., 2]
    world[..., 1] = root[1] + local[..., 1]
    translation=np.asarray(model_to_scene_translation)
    world += translation
    canonical = canonical_scene.query_height(soma_to_mujoco(world)[..., :2])
    shared = canonical.known & native_valid

    def errors(delta, valid):
        selected = delta[valid]
        return dict(count=int(valid.sum()), max_abs_m=float(np.abs(selected).max()) if len(selected) else None,
                    rms_m=float(np.sqrt(np.mean(selected ** 2))) if len(selected) else None)

    points = volume["points"].detach().cpu().numpy()+translation
    distance = volume["distance"].detach().cpu().numpy()
    valid = volume["valid"].detach().cpu().numpy()
    volume_ground = canonical_scene.query_height(soma_to_mujoco(points)[..., :2])
    return dict(
        mode="legacy_nearest_node_and_prism_sdf_audited_against_canonical_triangles",
        same_query_as_E2_claimed=False, canonical_scene_id=canonical_scene.manifest.scene_id,
        local_height_quantization=errors(world[..., 1] - canonical.height_m, shared),
        local_valid_mask_disagreement_count=int((native_valid != canonical.known).sum()),
        legacy_sdf_minus_canonical_vertical_clearance=errors(
            distance - (points[..., 1] - volume_ground.height_m), valid & volume_ground.known),
        volume_valid_mask_disagreement_count=int((valid != volume_ground.known).sum()),
        comparison_semantics="vertical_clearance_is_not_global_euclidean_triangle_sdf",
        model_input_modified=False,
        model_to_scene_translation_soma_xyz_m=translation.tolist(),
        observed_near_patch_center_soma_xz_m=world[0][..., [0, 2]].reshape(-1, 2).mean(0).tolist(),
        observed_volume_body_center_soma_xyz_m=points[0, 0].mean(0).tolist(),
    )


def execution_parity_metrics(actual, expected, representation):
    import torch

    reference_pose = representation.inverse(expected["world"], False, return_all=False)
    actual_pose = representation.inverse(actual["world"], False, return_all=False)
    left = actual_pose["global_joint_rots"].double()
    right = reference_pose["global_joint_rots"].double()
    relative = left @ right.transpose(-1, -2)
    skew = torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
                        relative[..., 0, 2] - relative[..., 2, 0],
                        relative[..., 1, 0] - relative[..., 0, 1]), -1)
    sine = skew.norm(dim=-1) / 2
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return dict(
        active_tokens_equal=bool(torch.equal(actual["tokens"][:, :8], expected["tokens"][:, :8])),
        fk_max_error_m=float((actual_pose["posed_joints"] - reference_pose["posed_joints"]).norm(dim=-1).max()),
        root_xyz_max_error_m=float((actual_pose["root_pos"] - reference_pose["root_pos"]).norm(dim=-1).max()),
        global_rotation_max_error_rad=float(torch.atan2(sine, cosine).abs().max()),
        normalized_root_feature_max_abs=float((actual["root_features"] - expected["root_features"]).abs().max()),
        decoder_local_feature_max_abs=float((actual["decoded_local"] - expected["decoded_local"]).abs().max()),
        world_max_feature_error=float((actual["world"] - expected["world"]).abs().max()),
    )


class CompiledMBNeuralExecution:
    """Reuse native scene-input construction and fixed-call capture, without skinning."""

    def __init__(self, controller, scene):
        from motionbricks.data.soma_reusable_terrain_execution import ReusableTerrainExecution

        self.controller, self.scene = controller, scene
        self.inference = controller.inference
        self.field = controller.scene_field(scene)
        self._scene_input_builder = ReusableTerrainExecution._scene_inputs
        self.call = None
        self.execution = "native_prepared"
        self.validation = []

    def scene_inputs(self, history, goal, query_origin_soma_xyz=None):
        import torch

        if query_origin_soma_xyz is None:
            return self._scene_input_builder(self, history, goal)
        query_history = history.clone()
        origin = torch.as_tensor(query_origin_soma_xyz, device=history.device, dtype=history.dtype).reshape(1, 1, 3)
        query_history[..., :3] = history[..., :3] - history[:, -1:, :3] + origin
        values = self._scene_input_builder(self, query_history, goal)
        return reframe_scene_inputs(history[..., :5], query_history[..., :5], values, self.controller.sampler)

    def _goal(self, history, controls):
        import torch
        from motionbricks.data.native_mb_online_conditions import VELOCITY_PROTOCOL
        from motionbricks.data.native_mb_online_live import online_command_goal

        return online_command_goal(history[..., :5], controls[:, :2],
                                   torch.atan2(controls[:, 3], controls[:, 2]), 8, VELOCITY_PROTOCOL["version"])

    def _prediction(self, history, goal, *values):
        from motionbricks.data.local_terrain_observation import LocalTerrainObservation
        from motionbricks.data.native_mb_online_live import predict_planar_terrain
        from motionbricks.data.soma_terrain_execution import SETTINGS

        volume = dict(zip(("points", "distance", "normal", "valid", "features"), values[8:]))
        prepared = (LocalTerrainObservation(*values[:4]), LocalTerrainObservation(*values[4:8]))
        with self.controller.context(history, self.scene, volume_observation=volume):
            world, count = predict_planar_terrain(
                self.inference, history, goal, self.scene, num_tokens=8, config=SETTINGS,
                query_controls=history.new_tensor([[-1., .6, 1., .4]]), prepared_observations=prepared)
        return dict(world=world[:, :32], count=count, tokens=self._captured_tokens,
                    root_features=self.controller.last_root["pred_global_root_values"],
                    decoded_local=self.controller.last_local[:, :32],
                    points=values[0], valid=values[1], world_query_xz=values[2],
                    sdf_points=values[8], sdf_distance=values[9], sdf_normal=values[10],
                    sdf_valid=values[11], sdf_features=values[12])

    def prepare(self, seed, controls, *, execution, query_origin_soma_xyz=None, plan_blocks=2):
        import torch
        from motionbricks.data.full_corpus_inference import tensor_digest
        from motionbricks.data.native_mb_observed_history import reencode_played_history
        from motionbricks.data.native_mb_evaluation import full_precision_evaluation
        from motionbricks.data.soma_cuda_graph import StaticCudaCall
        from motionbricks.data.soma_static_execution import (
            trace_fixed_call, use_dense_transformers, restore_native_transformers,
        )

        _check_plan_blocks(plan_blocks)
        if execution not in ("trace", "compiled"):
            raise ValueError("Compiled execution must be trace or compiled")
        if seed.shape != (1, 4, 366) or not seed.is_cuda:
            raise ValueError("Preparing native neural execution requires an allocated CUDA four-frame history")
        original_decode = self.inference._vqvae_pose_model.forward_decoder
        initial_digest = tensor_digest(self.controller.state_dict())
        precision_before = native_precision_contract()
        cpu_rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state(seed.device).clone()

        def capture_tokens(ids, *args, **kwargs):
            self._captured_tokens = ids.clone()
            return original_decode(ids, *args, **kwargs)

        self.inference._vqvae_pose_model.forward_decoder = capture_tokens
        try:
            with torch.inference_mode(), torch.jit.optimized_execution(False), full_precision_evaluation():
                precision = native_precision_contract()
                restore_native_transformers(self.inference)
                probes = [controls] + [seed.new_tensor([[vx, vz, np.cos(yaw), np.sin(yaw)]])
                                      for vx, vz, yaw in ((0., 0., 0.), (0., 1.6, 0.),
                                                         (1.6, 0., np.pi / 2), (0., -1.6, np.pi),
                                                         (-1.6, 0., -np.pi / 2))]
                cases, reports = [], []
                for index, command in enumerate(probes):
                    current = seed
                    for block in range(plan_blocks if index == 0 else 1):
                        goal = self._goal(current, command)
                        if query_origin_soma_xyz is None:
                            native = self.controller(current, command, self.scene)
                            expected = dict(world=native["world"].clone(), tokens=self._captured_tokens.clone(),
                                            root_features=native["root_output"]["pred_global_root_values"].clone(),
                                            decoded_local=native["local_prediction"].clone())
                            values = self.scene_inputs(current, goal)
                        else:
                            anchor = seed.new_tensor(query_origin_soma_xyz).reshape(1, 3)
                            block_origin = rolling_query_origin(seed, current, anchor)
                            values = self.scene_inputs(current, goal, block_origin)
                            native = self._prediction(current, goal, *values)
                            expected = {key: native[key].clone() for key in (
                                "world", "tokens", "root_features", "decoded_local")}
                        expected_world = expected["world"]
                        inputs = (current, goal, *values)
                        prepared = self._prediction(*inputs)
                        reports.append(dict(stage="prepared_eager", probe=index, proposal_block=block,
                                            **execution_parity_metrics(prepared, expected, self.inference.global_motion_rep),
                                            fk_tolerance_m=3e-5, precision=precision))
                        for name in ("world", "tokens", "root_features", "decoded_local"):
                            torch.testing.assert_close(prepared[name], expected[name], atol=0, rtol=0)
                        cases.append((inputs, expected, index, block))
                        if block == 0 and index == 0:
                            current = reencode_played_history(expected_world[:, 24:28].contiguous(),
                                                               self.inference.global_motion_rep)
                use_dense_transformers(self.inference)

                def check_stage(stage, call):
                    for inputs, expected, index, block in cases:
                        actual = call(*inputs)
                        metrics = execution_parity_metrics(actual, expected, self.inference.global_motion_rep)
                        report = dict(stage=stage, probe=index, proposal_block=block, **metrics,
                                      fk_tolerance_m=3e-5, precision=precision)
                        reports.append(report)
                        if (not metrics["active_tokens_equal"] or not np.isfinite(metrics["fk_max_error_m"])
                                or metrics["fk_max_error_m"] > 3e-5):
                            raise ValueError("Compiled/native stage parity failed: " + json.dumps(report, allow_nan=False))
                        for name, offset in (("points", 2), ("valid", 3), ("world_query_xz", 4),
                                             ("sdf_points", 10), ("sdf_distance", 11), ("sdf_valid", 13)):
                            torch.testing.assert_close(actual[name], inputs[offset], atol=0, rtol=0)

                check_stage("dense_eager", self._prediction)
                function = trace_fixed_call(self._prediction, cases[0][0], cuda_constants=True)
                check_stage("traced_cuda_constants", function)
                call = StaticCudaCall(function) if execution == "compiled" else function
                if execution == "compiled":
                    check_stage("captured_cuda_graph", call)
                if tensor_digest(self.controller.state_dict()) != initial_digest:
                    raise ValueError("Compiling native execution changed frozen model tensors")
                self.call, self.execution, self.validation = call, execution, reports
        finally:
            self.inference._vqvae_pose_model.forward_decoder = original_decode
            restore_native_transformers(self.inference)
        return dict(execution=execution, plan_blocks=plan_blocks, model_tensors_unchanged=True,
                    rolling_history_policy="native_played_fk_reencode",
                    precision=precision, trace_strategy="cuda_constants_serialize_reload_optimize_numerics_true",
                    precision_before_prepare=precision_before,
                    prepared_eager_native_equal=True,
                    cpu_rng_state_equal=bool(torch.equal(cpu_rng_before, torch.random.get_rng_state())),
                    cuda_rng_state_equal=bool(torch.equal(cuda_rng_before, torch.cuda.get_rng_state(seed.device))),
                    rng_compared_cuda_device=str(seed.device),
                    native_comparison_scope="original_controller" if query_origin_soma_xyz is None
                    else "native_prepared_identical_actual_query_inputs",
                    query_origin_soma_xyz=None if query_origin_soma_xyz is None else np.asarray(query_origin_soma_xyz).tolist(),
                    scene_queries="native_current_scene_explicit_inputs", validation=self.validation)

    def predict(self, history, controls, *, query_origin_soma_xyz=None):
        import torch
        from motionbricks.data.native_mb_evaluation import full_precision_evaluation

        if self.call is None:
            raise ValueError("Compiled execution must pass native parity checks before replay")
        with torch.inference_mode(), torch.jit.optimized_execution(False), full_precision_evaluation():
            goal = self._goal(history, controls)
            query_begin, query_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            neural_begin, neural_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            query_started = perf_counter_ns()
            query_begin.record()
            values = self.scene_inputs(history, goal, query_origin_soma_xyz)
            query_end.record()
            query_dispatch_ns = perf_counter_ns() - query_started
            neural_started = perf_counter_ns()
            neural_begin.record()
            result = self.call(history, goal, *values)
            neural_end.record()
            neural_dispatch_ns = perf_counter_ns() - neural_started
        observation = dict(points=result["points"], valid=result["valid"])
        volume = {key: result["sdf_" + key] for key in ("points", "distance", "normal", "valid", "features")}
        return result, observation, volume, dict(scene_query_dispatch_wall_ns=query_dispatch_ns,
                                                neural_dispatch_wall_ns=neural_dispatch_ns,
                                                cuda_events=(query_begin, query_end, neural_begin, neural_end))

    def predict_native_prepared(self, history, controls, *, query_origin_soma_xyz):
        import torch
        from motionbricks.data.native_mb_evaluation import full_precision_evaluation

        original_decode = self.inference._vqvae_pose_model.forward_decoder

        def capture_tokens(ids, *args, **kwargs):
            self._captured_tokens = ids.clone()
            return original_decode(ids, *args, **kwargs)

        self.inference._vqvae_pose_model.forward_decoder = capture_tokens
        try:
            with torch.inference_mode(), full_precision_evaluation():
                goal = self._goal(history, controls)
                values = self.scene_inputs(history, goal, query_origin_soma_xyz)
                result = self._prediction(history, goal, *values)
        finally:
            self.inference._vqvae_pose_model.forward_decoder = original_decode
        observation = dict(points=result["points"], valid=result["valid"])
        volume = {key: result["sdf_" + key] for key in ("points", "distance", "normal", "valid", "features")}
        return result, observation, volume


class FullPlanQueryParityError(AssertionError):
    """Same-input preparation failure with context preserved across planner IPC."""

    def __init__(self, case, block, name, detail):
        self.case, self.block, self.name, self.detail = case, block, name, detail
        super().__init__(case, block, name, detail)

    def __str__(self):
        return (f"Full-plan same-input query parity failed: case={self.case} "
                f"block={self.block} name={self.name}: {self.detail}")


class FullPlanExecution:
    """One fixed-scene graph with one or two AR8 calls and native observe/FK."""

    def __init__(self, adapter, conditioning_mode, block_count=2):
        _check_plan_blocks(block_count)
        self.block_count = block_count
        self.adapter, self.conditioning_mode = adapter, conditioning_mode
        self.engine = CompiledMBNeuralExecution(adapter.controller, adapter.scene)
        self.representation = adapter.controller.inference.global_motion_rep

    def _query_native(self, history, goal, origin):
        return self.engine.scene_inputs(history, goal, origin if self.conditioning_mode in ("B1", "B2") else None)

    def _plan(self, history, controls, origin, query, neural, goal_function, observe, fk):
        import torch

        result, blocks, current = {"query_origin": origin}, [], history
        for index in range(self.block_count):
            result[f"history_{index}"] = current
            goal = goal_function(current, controls)
            block_origin = rolling_query_origin(history, current, origin)
            result[f"query_origin_{index}"] = block_origin
            values = query(current, goal, block_origin)
            output = neural(current, goal, *values)
            for name, value in output.items():
                result[f"block_{index}_{name}"] = value
            blocks.append(output["world"])
            if index == 0:
                observed = observe(output["world"][:, 24:28].contiguous())
                current = observed["history"]
                result["observe_position_error"] = observed["position_error"]
                result["observe_rotation_error"] = observed["rotation_error"]
        world = torch.cat([blocks[0][:, :28]] + [block[:, 4:28] for block in blocks[1:]], dim=1)
        result["world"] = world
        result.update({"fk_" + name: value for name, value in fk(world).items()})
        return result

    def _captured_plan(self, history, controls, origin):
        values = self._plan(history, controls, origin, self.query.function, self.engine.call, self.goal_function,
                            self.observe.function, self.fk.function)
        return pack_payload(values, self.layout)

    def _validate_block_queries(self, actual, expected, controls, *, case, block):
        """Preparation only: hold captured query inputs fixed, not the native AR path."""
        import torch
        from .geometry_execution import QUERY_ATOL, SCENE_INPUT_NAMES

        prefix = f"block_{block}_"
        history, origin = actual[f"history_{block}"], actual[f"query_origin_{block}"]
        goal = self.goal_function(history, controls)
        native = dict(zip(SCENE_INPUT_NAMES, self._query_native(history, goal, origin)))
        query_errors, trajectory_errors = {}, {}
        floats = ("points", "world_query_xz", "sdf_points", "sdf_distance", "sdf_normal", "sdf_features")
        for name in (*floats, "valid", "sdf_valid"):
            left, right = actual[prefix + name], native[name]
            try:
                torch.testing.assert_close(left, right, atol=QUERY_ATOL if name in floats else 0, rtol=0)
            except AssertionError as error:
                raise FullPlanQueryParityError(case, block, name, str(error)) from error
            baseline = expected[prefix + name]
            if name in floats:
                query_errors[name + "_max_abs"] = float((left - right).abs().max())
                trajectory_errors[name + "_max_abs"] = float((left - baseline).abs().max())
            else:
                query_errors[name + "_equal"] = True
                trajectory_errors[name + "_different_count"] = int((left != baseline).sum())
        return dict(same_input_query_errors=query_errors, cross_trajectory_query_errors=trajectory_errors)

    def prepare(self, seed, controls, origin):
        import functools
        import torch
        from motionbricks.data.full_corpus_inference import tensor_digest
        from motionbricks.data.native_mb_evaluation import full_precision_evaluation
        from motionbricks.data.soma_cuda_graph import StaticCudaCall
        from motionbricks.data.soma_static_execution import trace_fixed_call, restore_native_transformers
        from .geometry_execution import (SceneQueryExecution, SomaFKExecution, ObservedHistoryExecution,
                                         native_soma_fk, native_observed_history, admit_scene_queries,
                                         admit_soma_fk, admit_observed_history, SOMA_FK_NAMES)

        if not seed.is_cuda or seed.shape != (1, 4, 366):
            raise ValueError("Full-plan preparation requires allocated CUDA GLOBAL366 history")
        inference = self.engine.inference
        native_decode = inference._vqvae_pose_model.forward_decoder
        before = tensor_digest(self.adapter.controller.state_dict())
        cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state(seed.device).clone()
        native_fk = functools.partial(native_soma_fk, self.representation)
        native_observe = functools.partial(native_observed_history, self.representation)
        neural_origin = origin[0].detach().cpu().numpy() if self.conditioning_mode in ("B1", "B2") else None
        neural_proof = self.engine.prepare(seed, controls, execution="trace", query_origin_soma_xyz=neural_origin,
                                           plan_blocks=self.block_count)

        def capture_tokens(ids, *args, **kwargs):
            self.engine._captured_tokens = ids.clone()
            return native_decode(ids, *args, **kwargs)

        inference._vqvae_pose_model.forward_decoder = capture_tokens
        try:
            with torch.inference_mode(), full_precision_evaluation(), torch.jit.optimized_execution(False):
                restore_native_transformers(inference)
                commands = [controls] + [seed.new_tensor([[vx, vz, np.cos(yaw), np.sin(yaw)]])
                                         for vx, vz, yaw in ((0., 0., 0.), (0., 1.6, 0.),
                                                            (1.6, 0., np.pi / 2), (0., -1.6, np.pi),
                                                            (-1.6, 0., -np.pi / 2))]
                inputs = [(seed, command, origin) for command in commands]
                if self.conditioning_mode in ("B1", "B2"):
                    inputs += [(seed, controls, origin + seed.new_tensor([offset]))
                               for offset in ((.05, .02, -.04), (-.06, -.015, .03))]
                cases = []
                for case_inputs in inputs:
                    expected = self._plan(*case_inputs, self._query_native, self.engine._prediction,
                                          self.engine._goal, native_observe, native_fk)
                    cases.append((case_inputs, {key: value.clone() for key, value in expected.items()}))
                sample = cases[0][1]
                self.bindings = self.adapter.scene_graph_bindings()
                asset = self.adapter.canonical_scene.manifest.source["scene"]
                self.source_sha256 = asset["sha256"]
                query_cases = [(seed, self.engine._goal(seed, command), origin) for command in commands]
                if self.block_count > 1:
                    query_cases += [(case[1]["history_1"], self.engine._goal(case[1]["history_1"], case[0][1]),
                                     case[1]["query_origin_1"]) for case in cases]
                query_cases += [(seed, query_cases[0][1], origin + seed.new_tensor([[.05, .02, -.04]]))]
                self.query = SceneQueryExecution(self._query_native, query_cases[0], scene_id=self.adapter.scene_id,
                                                  source_sha256=self.source_sha256, bindings=self.bindings)
                self.observe = ObservedHistoryExecution(self.representation, sample["block_0_world"][:, 24:28].contiguous())
                self.fk = SomaFKExecution(self.representation, sample["world"])
                geometry_proof = dict(
                    query=admit_scene_queries(self.query, query_cases),
                    observe=admit_observed_history(self.observe, [case[1]["block_0_world"][:, 24:28].contiguous() for case in cases]),
                    fk=admit_soma_fk(self.fk, [case[1]["world"] for case in cases]),
                )
                self.layout = payload_layout(sample)
                self.goal_function = trace_fixed_call(self.engine._goal, (seed, controls), shared_cuda_constants=True)
                self.call = StaticCudaCall(self._captured_plan)
                proof = []
                for index, (inputs, expected) in enumerate(cases):
                    actual = unpack_payload_torch(self.call(*inputs), self.layout)
                    torch.testing.assert_close(actual["history_0"], inputs[0], atol=0, rtol=0)
                    torch.testing.assert_close(actual["query_origin"], inputs[2], atol=0, rtol=0)
                    errors = {}
                    for block in range(self.block_count):
                        prefix = f"block_{block}_"
                        metrics = execution_parity_metrics(
                            {key: actual[prefix + key] for key in ("world", "tokens", "root_features", "decoded_local")},
                            {key: expected[prefix + key] for key in ("world", "tokens", "root_features", "decoded_local")},
                            self.representation)
                        if (not metrics["active_tokens_equal"] or not np.isfinite(metrics["fk_max_error_m"])
                                or metrics["fk_max_error_m"] > 3e-5):
                            raise ValueError("Full-plan native block parity failed: " + json.dumps(dict(case=index, block=block, **metrics)))
                        metrics.update(self._validate_block_queries(actual, expected, inputs[1], case=index, block=block))
                        torch.testing.assert_close(actual[f"query_origin_{block}"], expected[f"query_origin_{block}"],
                                                   atol=3e-5, rtol=0)
                        errors[str(block)] = metrics
                    fk_error = float((actual["fk_posed_joints"] - expected["fk_posed_joints"]).norm(dim=-1).max())
                    if not np.isfinite(fk_error) or fk_error > 3e-5:
                        raise ValueError(f"Full-plan native FK{4 + 24 * self.block_count} error: {fk_error}")
                    for name in SOMA_FK_NAMES:
                        torch.testing.assert_close(actual["fk_" + name], expected["fk_" + name], atol=3e-5, rtol=0)
                    for name in ("observe_position_error", "observe_rotation_error"):
                        if not torch.isfinite(actual[name]) or float(actual[name]) > 2e-5:
                            raise ValueError("Full-plan native observed-history reconstruction failed")
                    proof.append(dict(case=index, blocks=errors,
                                      **{f"fk{4 + 24 * self.block_count}_max_error_m": fk_error},
                                      fk_frames=4 + 24 * self.block_count,
                                      query_origin_soma_xyz=inputs[2][0].detach().cpu().tolist(),
                                      history_root_unchanged=True))
                if tensor_digest(self.adapter.controller.state_dict()) != before:
                    raise ValueError("Full-plan preparation changed frozen model tensors")
        finally:
            inference._vqvae_pose_model.forward_decoder = native_decode
            restore_native_transformers(inference)
        return dict(execution="full_plan", conditioning_mode=self.conditioning_mode,
                    execution_contract=("static_scene_two_AR8_native_observe_FK52_single_payload_v1"
                                        if self.block_count == 2 else
                                        "static_scene_one_AR8_native_observe_FK28_single_payload_v1"),
                    blocks=self.block_count, fk_frames=4 + 24 * self.block_count,
                    future_coverage_s=.8 * self.block_count,
                    rolling_history_policy="native_played_fk_reencode", geometry_proof=geometry_proof,
                    neural_proof=neural_proof, outer_torch_trace=False,
                    validation=proof, model_tensors_unchanged=True, precision=native_precision_contract(),
                    query_comparison_scope="native_query_at_captured_block_history_compiled_goal_and_origin",
                    cross_trajectory_query_comparison_scope="diagnostic_only_independent_AR_trajectories",
                    query_atol=1e-5, query_valid_masks_exact=True,
                    origin_input_varied=self.conditioning_mode in ("B1", "B2"),
                    cpu_rng_state_equal=bool(torch.equal(cpu_rng, torch.random.get_rng_state())),
                    cuda_rng_state_equal=bool(torch.equal(cuda_rng, torch.cuda.get_rng_state(seed.device))),
                    payload_layout=self.layout, payload_transport="lossless_native_dtype_byte_payload_v1",
                    geometry_implementation_sha256=file_sha256(Path(__file__).with_name("geometry_execution.py")),
                    static_scene_id=self.adapter.scene_id, source_sha256=self.source_sha256)

    def generate(self, history, features, controls, origin, command_id):
        import torch
        from motionbricks.data.native_mb_evaluation import full_precision_evaluation

        started = perf_counter_ns()
        self.query.assert_binding(scene_id=self.adapter.scene_id, source_sha256=self.source_sha256,
                                   bindings=self.adapter.scene_graph_bindings(), replay=True)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.inference_mode(), full_precision_evaluation():
            begin.record()
            payload = self.call(features, controls, origin)
            end.record()
        dispatched = perf_counter_ns()
        cpu_payload = payload.cpu().numpy()
        transferred = perf_counter_ns()
        values = unpack_payload_numpy(cpu_payload, self.layout)
        observe_errors = [float(values[name]) for name in ("observe_position_error", "observe_rotation_error")]
        if not np.isfinite(observe_errors).all() or max(observe_errors) > 2e-5:
            raise ValueError("Full-plan observed-history reconstruction exceeds native tolerance")
        times = float(history.timestamps_s[-1]) + (np.arange(4 + 24 * self.block_count) - 3) / 30
        audits, blocks = [], []
        for index in range(self.block_count):
            prefix = f"block_{index}_"
            if int(values[prefix + "count"].reshape(-1)[0]) != 8:
                raise ValueError("Full-plan prediction changed its fixed eight-token horizon")
            if not values[prefix + "sdf_valid"].any():
                raise ValueError("Full-plan scene observation is entirely unknown")
            local = {key: torch.from_numpy(values[prefix + key]) for key in ("points", "valid")}
            volume = {key: torch.from_numpy(values[prefix + "sdf_" + key]) for key in ("points", "distance", "valid")}
            input_history = values[f"history_{index}"]
            translation=(values[f'query_origin_{index}'][0]-input_history[0,-1,:3]
                if self.conditioning_mode in ('B1','B2') else np.zeros(3))
            audit = audit_legacy_observation(self.adapter.canonical_scene, torch.from_numpy(input_history), local, volume,
                model_to_scene_translation=translation)
            audits.append(audit)
            block_times = float(history.timestamps_s[-1]) + index * .8 + (np.arange(32) - 3) / 30
            blocks.append(RawMBBlock(block_times, values[prefix + "world"][0], input_history[0],
                                      self.adapter.scene_id, command_id, float(block_times[27]), dict(
                                          execution="full_plan", pose_expansion="offline_only",
                                          model_identity=self.adapter.identity, native_tokens=8, native_pose_passes=4,
                                          native_pose_token_ids=values[prefix + "tokens"][0].tolist(),
                                          history_kind=history.provenance["kind"] if index == 0 else "open_loop_generated_context",
                                          rolling_history_policy="native_played_fk_reencode", geometry_audit=audit,
                                          conditioning_mode=self.conditioning_mode,
                                          actual_anchor_soma_xyz_m=values["query_origin"][0].tolist()
                                          if self.conditioning_mode in ("B1", "B2") else None,
                                          query_origin_soma_xyz_m=values[f"query_origin_{index}"][0].tolist(),
                                          query_origin_policy="actual_anchor_plus_reference_displacement"
                                          if self.conditioning_mode in ("B1", "B2") else "native_reference_root",
                                          predicted_reference_offset_soma_xyz_m=(input_history[0, -1, :3] - values["history_0"][0, -1, :3]).tolist(),
                                          query_origin_is_future_truth=False)))
        source = SomaReference(times, values["world"][0], tuple(self.representation.skeleton.bone_order_names),
                               values["fk_root_pos"][0], values["fk_local_joint_rots"][0],
                               values["fk_global_joint_rots"][0], values["fk_posed_joints"][0], self.adapter.scene_id,
                               command_id, float(times[-1]), dict(
                                   kind="trained_mb_autoregressive_proposal", execution="full_plan",
                                   conditioning_mode=self.conditioning_mode, seed_kind=history.provenance["kind"],
                                   model_identity=self.adapter.identity, rolling_history_policy="native_played_fk_reencode",
                                   query_origin_soma_xyz=values["query_origin"][0].tolist()
                                   if self.conditioning_mode in ("B1", "B2") else None,
                                   query_origin_consumed=self.conditioning_mode in ("B1", "B2"), blocks=self.block_count,
                                   native_tokens_per_call=8, pose_passes_per_call=4,
                                   future_coverage_s=.8 * self.block_count,
                                   optimizer_steps=0, physical_state_writes=0, geometry_audits=audits))
        end.synchronize()
        return MBProposal(source, tuple(blocks), history, dict(
            total=perf_counter_ns() - started, graph_dispatch_wall_ns=dispatched - started,
            single_payload_cpu_transfer_including_cuda_wait_wall_ns=transferred - dispatched,
            cpu_unpack_audit_snapshot_wall_ns=perf_counter_ns() - transferred,
            full_plan_cuda_elapsed_ms=begin.elapsed_time(end), full_plan_single_graph=True,
            raw_block_fk="offline", final_native_fk_calls=1, cuda_elapsed_times_overlap_wall_times=True))


class TrainedMBReferenceAdapter:
    """Each replan starts from explicit history; generated context stays inside it."""

    @classmethod
    def load(cls, artifact, bundle_path, *, canonical_scene, device="cpu",
             scene_query_backend="native", scene_query_proof=None,root_reference_acceleration_mps2=None):
        import torch

        torch.set_num_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        from motionbricks.data.full_corpus_acceptance import load_acceptance_candidate
        from motionbricks.data.unified_terrain import UnifiedTerrainController
        from scripts.portable_parc_online_soma import load_terrain_bundle

        identity = inspect_trained_artifacts(artifact, bundle_path)
        identity["precision_contract"] = native_precision_contract()
        identity["host_contract"] = native_host_contract()
        receipt = json.loads((Path(artifact) / "acceptance.json").read_text(encoding="utf-8"))
        inference, bundle = load_terrain_bundle(bundle_path, device)
        controller = UnifiedTerrainController(inference, bundle)
        saved = torch.load(identity["weights"], map_location="cpu", mmap=True, weights_only=False)
        load_acceptance_candidate(controller, saved, receipt, TRAINED_BUNDLE_SHA256, TRAINED_WEIGHTS_SHA256)
        if root_reference_acceleration_mps2 is not None:
            from .root_plan import install_root_reference_plan
            install_root_reference_plan(inference,acceleration_mps2=root_reference_acceleration_mps2)
            identity['root_reference_plan']=dict(kind='bounded_planar_velocity_before_Pose',
                acceleration_mps2=root_reference_acceleration_mps2,
                input='accepted_native_history_and_latest_intent',
                height_heading='original_Root_prediction',weights_changed=False)
        return cls(controller, canonical_scene=canonical_scene, identity=identity,
                   scene_query_backend=scene_query_backend, scene_query_proof=scene_query_proof)

    def __init__(self, controller, *, canonical_scene, identity,
                 scene_query_backend="native", scene_query_proof=None):
        import torch
        from .geometry_execution import install_scene_field

        if identity.get("checkpoint_sha256") != TRAINED_WEIGHTS_SHA256:
            raise ValueError("This adapter requires the preserved trained compact checkpoint")
        self.controller = controller
        self.identity = dict(identity,scene_frame_contract='query_local_geometry_registered_reference_targets_v1')
        self.canonical_scene = canonical_scene
        self.scene_id = canonical_scene.manifest.scene_id
        if any(obj["kind"] != "hfield" for obj in canonical_scene.manifest.objects):
            raise ValueError("This frozen MB adapter requires a ground-only canonical scene; "
                             "independent rigid objects need a shared full-solid query adapter")
        source = canonical_scene.root / canonical_scene.manifest.source["scene"]["file"]
        if file_sha256(source) != canonical_scene.manifest.source["scene"]["sha256"]:
            raise ValueError("Canonical scene source changed")
        with np.load(source, allow_pickle=False) as archive:
            self.scene = {key: torch.as_tensor(archive[key].copy(), device=controller.inference.device)
                          for key in ("heightfield_y_m", "min_point_world_xyz_m", "dx_m")}
            if "heightfield_valid" in archive:
                self.scene["heightfield_valid"] = torch.as_tensor(archive["heightfield_valid"].copy(),
                                                                device=controller.inference.device)
        self.scene_field, self._scene_query_receipt = install_scene_field(
            controller, self.scene, backend=scene_query_backend,
            source_sha256=canonical_scene.manifest.source["scene"]["sha256"], proof_path=scene_query_proof,
            scene_id=self.scene_id)
        self.controller.eval().requires_grad_(False)
        self._compiled = None
        self._native_prepared = None
        self._full_plan = None
        self.execution = "native"
        self.compilation = None

    def activate_scene(self, *, validate_binding=True):
        """Reinstall this exact owner after a serialized cache hit, without rebuilding."""
        if self.scene_field is None:
            raise RuntimeError("Released MB scene cannot be activated or replayed")
        if validate_binding and self._full_plan is not None:
            self._full_plan.query.assert_binding(scene_id=self.scene_id, source_sha256=self._full_plan.source_sha256,
                                                bindings=self.scene_graph_bindings())
        self.controller._fields[id(self.scene["heightfield_y_m"])] = (self.scene, self.scene_field)

    def scene_graph_bindings(self):
        """Strong field/tree owners and live identities for fixed-scene graph admission."""
        if self.scene_field is None:
            raise RuntimeError("Released MB scene has no graph binding")
        field = self.scene_field
        bindings = dict(scene=self.scene, field_owner=field, field=vars(field),
                        sampler=self.controller.sampler, sampler_state=vars(self.controller.sampler),
                        sampling_config=vars(self.controller.sampler.config),
                        backend=self._scene_query_receipt["backend"],
                        source_sha256=self._scene_query_receipt["source_sha256"])
        if self._scene_query_receipt["field_kind"] in ("BVHHeightfieldPrismSDF", "SphereBVHHeightfieldPrismSDF"):
            bindings.update(field_id=field.field_id, solid_tree=field.solid_tree, air_tree=field.air_tree,
                            solid_tree_id=field.solid_tree.id, air_tree_id=field.air_tree.id)
        return bindings

    @property
    def accounted_bytes(self):
        from .geometry_execution import unique_tensor_storage_bytes

        bindings = self.scene_graph_bindings()
        return unique_tensor_storage_bytes((bindings["scene"], bindings["field"]))

    @property
    def scene_query_receipt(self):
        bindings = self.scene_graph_bindings()
        handles = {name: bindings[name] for name in ("field_id", "solid_tree_id", "air_tree_id") if name in bindings}
        return dict(self._scene_query_receipt, scene_id=self.scene_id, **handles,
                    accounted_bytes=self.accounted_bytes,
                    accounted_bytes_scope="unique_cuda_torch_scene_and_field_storage",
                    warp_bvh_bytes=None if handles else 0, graph_arenas_and_jit_constants_included=False)

    def release_scene(self):
        """Owner must quiesce pending work/replays before eviction; shared model stays live."""
        key = id(self.scene["heightfield_y_m"])
        if key in self.controller._fields and self.controller._fields[key][1] is self.scene_field:
            del self.controller._fields[key]
        self._compiled = self._native_prepared = self._full_plan = None
        self.scene_field = None
        self.execution, self.compilation = "released", None

    def prepare_compiled(self, history: NativeSomaHistory, *, velocity_world_xz_mps,
                         heading_world_rad, execution="compiled", conditioning_mode="native",
                         query_origin_soma_xyz=None, plan_blocks=2):
        import torch

        _check_plan_blocks(plan_blocks)
        self.activate_scene()
        if history.scene_id != self.scene_id:
            raise ValueError("Compilation history must belong to the same canonical scene")
        origin = validate_conditioning(history, conditioning_mode, query_origin_soma_xyz)
        device = self.controller.inference.device
        seed = torch.tensor(history.features, dtype=torch.float32, device=device)[None]
        controls = seed.new_tensor([[*velocity_world_xz_mps, np.cos(heading_world_rad), np.sin(heading_world_rad)]])
        started = perf_counter_ns()
        if execution == "full_plan":
            compiled = FullPlanExecution(self, conditioning_mode, block_count=plan_blocks)
            anchor = seed[:, -1, :3].clone() if origin is None else seed.new_tensor(origin).reshape(1, 3)
            report = compiled.prepare(seed, controls, anchor)
            self._full_plan, self._compiled = compiled, None
        else:
            compiled = CompiledMBNeuralExecution(self.controller, self.scene)
            report = compiled.prepare(seed, controls, execution=execution, query_origin_soma_xyz=origin,
                                      plan_blocks=plan_blocks)
            self._compiled, self._full_plan = compiled, None
        self.compilation = dict(report, preparation_wall_ns=perf_counter_ns() - started,
                                conditioning_mode=conditioning_mode, scene_query=self.scene_query_receipt)
        self.execution = execution
        return self.compilation

    def generate(self, history: NativeSomaHistory, *, velocity_world_xz_mps, heading_world_rad,
                 command_id: str, blocks=2, conditioning_mode="native", query_origin_soma_xyz=None,
                 output_frame='scene') -> MBProposal:
        """8 tokens/call. Default two calls provide 1.6s genuine future coverage.

        Later blocks condition on earlier generated frames, not claimed execution.
        A subsequent replan must be called again with new estimated actual history.
        """
        import torch
        from motionbricks.data.native_mb_observed_history import reencode_played_history

        self.activate_scene(validate_binding=False)
        started = perf_counter_ns()
        if history.scene_id != self.scene_id:
            raise ValueError("History, scene queries and reference must share one canonical scene")
        if type(blocks) is not int or blocks < 1:
            raise ValueError("An explicit positive number of real MB block calls is required")
        origin = validate_conditioning(history, conditioning_mode, query_origin_soma_xyz)
        if self.execution != "native" and self.compilation["conditioning_mode"] != conditioning_mode:
            raise ValueError("Conditioning mode differs from the explicitly prepared execution contract")
        if origin is not None and self.execution == "native" and self._native_prepared is None:
            self._native_prepared = CompiledMBNeuralExecution(self.controller, self.scene)
        device = self.controller.inference.device
        values = torch.tensor(history.features, dtype=torch.float32, device=device)[None]
        velocity = np.asarray(velocity_world_xz_mps, dtype=np.float64)
        if velocity.shape != (2,) or not np.isfinite(velocity).all() or not np.isfinite(heading_world_rad):
            raise ValueError("Finite world SOMA XZ velocity and heading are required")
        controls = values.new_tensor([[*velocity, np.cos(heading_world_rad), np.sin(heading_world_rad)]])
        history_preparation_ns = perf_counter_ns() - started
        if self.execution == "full_plan":
            if blocks != self._full_plan.block_count:
                raise ValueError("Requested blocks differ from the prepared full-plan AR8 call count")
            anchor = values[:, -1, :3].clone() if origin is None else values.new_tensor(origin).reshape(1, 3)
            proposal = self._full_plan.generate(history, values, controls, anchor, command_id)
            proposal.timing_ns["history_preparation_wall_ns"] = history_preparation_ns
            proposal.timing_ns["total"] = perf_counter_ns() - started
            return proposal_in_frame(proposal,output_frame,origin)
        raw, stitched, timings, stage_profiles = [], [], [], []
        initial_values = values
        anchor = None if origin is None else values.new_tensor(origin).reshape(1, 3)
        initial_time = float(history.timestamps_s[-1])
        boundary = initial_time
        for index in range(blocks):
            block_started = perf_counter_ns()
            block_origin = None if anchor is None else rolling_query_origin(initial_values, values, anchor)
            model_started = perf_counter_ns()
            if self.execution == "native":
                stage = {}
                if values.is_cuda:
                    model_begin, model_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    model_begin.record()
                with torch.inference_mode():
                    if origin is None:
                        output = self.controller(values, controls, self.scene)
                        observation, volume = self.controller.observation, self.controller.volume_observation
                    else:
                        output, observation, volume = self._native_prepared.predict_native_prepared(
                            values, controls, query_origin_soma_xyz=block_origin)
                if values.is_cuda:
                    model_end.record()
            else:
                output, observation, volume, stage = self._compiled.predict(
                    values, controls, query_origin_soma_xyz=block_origin)
            stage["model_dispatch_including_queries_wall_ns"] = perf_counter_ns() - model_started
            check_started = perf_counter_ns()
            world = output["world"]
            if world.shape != (1, 32, 366) or not torch.isfinite(world).all():
                raise ValueError("The trained fixed 8-token call must return 32 real finite native frames")
            stage["output_contract_check_including_cuda_wait_wall_ns"] = perf_counter_ns() - check_started
            audit_started = perf_counter_ns()
            geometry_audit = audit_legacy_observation(
                self.canonical_scene, values, observation, volume,
                model_to_scene_translation=(np.zeros(3) if block_origin is None else
                    (block_origin[0]-values[0,-1,:3]).detach().cpu().numpy()))
            query_receipt = dict(
                frame="soma_world_y_up", conditioning_mode=conditioning_mode,
                actual_anchor_soma_xyz_m=None if origin is None else origin.tolist(),
                requested_origin_soma_xyz_m=None if block_origin is None else block_origin[0].detach().cpu().tolist(),
                predicted_reference_offset_soma_xyz_m=(values[0, -1, :3] - initial_values[0, -1, :3]).detach().cpu().tolist(),
                query_origin_is_future_truth=False,
                observed_body_query_center_soma_xyz_m=geometry_audit["observed_volume_body_center_soma_xyz_m"],
                observed_near_query_center_soma_xz_m=geometry_audit["observed_near_patch_center_soma_xz_m"],
                model_reference_root_soma_xyz_m=values[0, -1, :3].detach().cpu().tolist(),
                morphology_conversion="performed_by_caller_before_explicit_native_origin" if origin is not None else None,
                model_reference_root_modified=False,
            )
            stage["geometry_audit_including_cuda_wait_wall_ns"] = perf_counter_ns() - audit_started
            times = boundary + (np.arange(32, dtype=np.float64) - 3) / 30
            diagnostics_started = perf_counter_ns()
            root_error = float((world[:, :4, :3] - values[:, :, :3]).norm(dim=-1).max().item())
            root_step = float((world[:, 4, :3] - values[:, -1, :3]).norm().item())
            stage["source_diagnostics_wall_ns"] = perf_counter_ns() - diagnostics_started
            decode_started = perf_counter_ns()
            reference = decode_soma_reference(
                world, times, self.controller.inference.global_motion_rep,
                scene_id=self.scene_id, command_id=command_id,
                trusted_until_s=float(times[27]), provenance=dict(
                    kind="trained_mb_raw_generation", model_identity=self.identity,
                    proposal_block=index, history_kind=history.provenance["kind"] if index == 0
                    else "open_loop_generated_context", native_tokens=8, native_pose_passes=4,
                    history_root_reconstruction_max_m=root_error, next_root_step_m=root_step,
                    trusted_frame_indexes=[0, 27], decoder_edge_frame_indexes=[28, 31],
                    optimizer_steps=0, physical_state_writes=0,
                    execution=self.execution,
                    geometry_audit=geometry_audit, scene_query_origin=query_receipt,
                ), timings=stage)
            stage["block_decode_wall_ns"] = perf_counter_ns() - decode_started
            if self.execution != "native":
                query_begin, query_end, neural_begin, neural_end = stage.pop("cuda_events")
                neural_end.synchronize()
                stage.update(scene_query_cuda_elapsed_ms=query_begin.elapsed_time(query_end),
                             neural_cuda_elapsed_ms=neural_begin.elapsed_time(neural_end))
            elif values.is_cuda:
                model_end.synchronize()
                stage["native_model_including_scene_query_cuda_elapsed_ms"] = model_begin.elapsed_time(model_end)
            raw.append(reference)
            context_started = perf_counter_ns()
            # The decoder's last four frames are never played or used as context.
            stitched.append(world[:, :28] if index == 0 else world[:, 4:28])
            if index + 1 < blocks:
                values = reencode_played_history(world[:, 24:28].contiguous(),
                                                 self.controller.inference.global_motion_rep)
            boundary = float(times[27])
            stage["next_context_preparation_dispatch_wall_ns"] = perf_counter_ns() - context_started
            timings.append(perf_counter_ns() - block_started)
            stage_profiles.append(stage)
        assembly_started = perf_counter_ns()
        proposal_values = torch.cat(stitched, dim=1)
        times = initial_time + (np.arange(proposal_values.shape[1], dtype=np.float64) - 3) / 30
        assembly_ns = perf_counter_ns() - assembly_started
        proposal_decode, decode_started = {}, perf_counter_ns()
        source = decode_soma_reference(
            proposal_values, times, self.controller.inference.global_motion_rep,
            scene_id=self.scene_id, command_id=command_id, trusted_until_s=float(times[-1]),
            provenance=dict(kind="trained_mb_autoregressive_proposal", model_identity=self.identity,
                            seed_kind=history.provenance["kind"], seed_provenance=dict(history.provenance),
                            blocks=blocks, native_tokens_per_call=8, pose_passes_per_call=4,
                            internal_context="generated_not_executed", future_coverage_s=24 * blocks / 30,
                            rolling_history_policy="native_played_fk_reencode",
                            conditioning_mode=conditioning_mode,
                            query_origin_soma_xyz=None if origin is None else origin.tolist(),
                            query_origin_frame="soma_world_y_up", model_reference_root_modified=False,
                            geometry_mode="legacy_quantized_audited_not_E2_query_parity",
                            cross_block_smoothing=None, optimizer_steps=0, physical_state_writes=0),
            timings=proposal_decode,
        )
        proposal_decode["total_wall_ns"] = perf_counter_ns() - decode_started
        proposal = MBProposal(source=source, raw_blocks=tuple(raw), seed_history=history,
                          timing_ns=dict(blocks=timings, total=perf_counter_ns() - started,
                                         history_preparation_wall_ns=history_preparation_ns,
                                         block_stages=stage_profiles, proposal_assembly_wall_ns=assembly_ns,
                                         proposal_decode=proposal_decode,
                                         cuda_elapsed_times_overlap_wall_times=bool(proposal_values.is_cuda)))
        return proposal_in_frame(proposal,output_frame,origin)
