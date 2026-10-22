"""Read-only real PARC engineering windows, final 8x10 labels and native Pose."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .training import (
    BATCH_SCHEMA, CONFIG_SCHEMA, NativePoseAdapter, SceneControlTrainer, TrainingBudget,
    make_training_label_lineage, validate_training_batch,
)
from .quantizer import FrozenCodebookIdentity, QuantizerConfig, tensor_state_sha256


PROVIDER_SCHEMA = "motionbricks.scene_control_v2.real_engineering_provider.v1"
POLICY_ID = "motionbricks.scene_control_v2.legacy_heightfield_source_root.engineering.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(path):
    value = Path(path)
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve(strict=True)


def _state(state, prefix):
    selected = {name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)}
    if not selected:
        raise ValueError(f"Final checkpoint is missing {prefix} tensors")
    return selected


def _paths(spec):
    if spec["schema"] != PROVIDER_SCHEMA or spec["cpu_threads"] != 2:
        raise ValueError("Real engineering provider requires its explicit schema and cpu_threads=2")
    paths = {key: resolve_project_path(spec[key]) for key in
             ("accepted_path", "motion_path", "scene_path", "upstream_path")}
    paths.update({key: resolve_project_path(spec["artifacts"][key]) for key in
                  ("architecture_bundle", "checkpoint")})
    return paths


def _matrix_rotations(values, name):
    identity = torch.eye(3, dtype=values.dtype)
    if not torch.isfinite(values).all() or not torch.allclose(
            values.transpose(-1, -2) @ values, identity.expand_as(values), atol=2e-4, rtol=0):
        raise ValueError(f"{name} must contain finite orthogonal rotation matrices")
    if not torch.allclose(torch.linalg.det(values), torch.ones(values.shape[:-2]), atol=2e-4, rtol=0):
        raise ValueError(f"{name} contains a reflection")


def _native_foot_features(joints, skeleton, scene):
    from motionbricks.motionlib.core.motion_reps.tools.feet import foot_detect_from_pos_and_vel
    from motionbricks.motionlib.core.motion_reps.tools.motion_features import compute_vel_xyz

    feet = list(skeleton.left_foot_joint_idx) + list(skeleton.right_foot_joint_idx)
    cells = torch.floor((joints[:, feet][..., [0, 2]] - scene["min_point_world_xyz_m"][[0, 2]]) /
                        scene["dx_m"] + .5).long()
    x, z = cells.unbind(-1)
    heights = scene["heightfield_y_m"]
    inside = (x >= 0) & (z >= 0) & (x < heights.shape[1]) & (z < heights.shape[0])
    if not inside.all():
        raise ValueError("Native derived foot features require known bound terrain under every source foot")
    ground = heights[z, x]
    if not torch.isfinite(ground).all():
        raise ValueError("Native derived foot features have unknown source terrain")
    relative = joints[None].clone()
    relative[0, :, feet, 1] -= ground
    velocity = compute_vel_xyz(joints[None], 30, lengths=torch.tensor([len(joints)]))
    left, right = foot_detect_from_pos_and_vel(relative, velocity, skeleton, .15, .10)
    return torch.cat((left, right), -1)[0]


@torch.no_grad()
def load_bound_world(spec, representation):
    """Read raw target-SOMA arrays; preserve world frame, original split and FPS."""
    paths = _paths(spec)
    rows = json.loads(paths["accepted_path"].read_text(encoding="utf-8"))
    selected = [row for row in rows if row["source_id"] == spec["source_id"]]
    if len(selected) != 1:
        raise ValueError("accepted_path must bind exactly one requested source_id")
    row = selected[0]
    if row["split"] != "train" or row["origin"] != "parc_generated":
        raise ValueError("This engineering provider preserves the selected original PARC-generated train split")
    expected_frame = "same scaled SOMA Y-up world coordinates as retarget root_positions"
    if row["terrain_geometry_coordinates"] != expected_frame:
        raise ValueError("Source/scene coordinate binding is not the declared retarget-world frame")
    hashes = {key: _file_hash(paths[key]) for key in
              ("accepted_path", "motion_path", "scene_path", "upstream_path")}
    for key, evidence in (("motion_path", "source_shard_sha256"),
                          ("scene_path", "terrain_geometry_sha256"),
                          ("upstream_path", "source_manifest_snapshot_sha256")):
        if hashes[key] != row[evidence]:
            raise ValueError(f"{key} differs from accepted.{evidence}")
    upstream = json.loads(paths["upstream_path"].read_text(encoding="utf-8"))
    recordings = [item for item in upstream["recordings"]
                  if item["candidate_key"] == row["physics_candidate_key"] and
                  item["converted_sha256"] == row["source_original_sha256"]]
    if len(recordings) != 1:
        raise ValueError("Upstream export must bind one original recording and candidate")
    recording = recordings[0]
    with np.load(paths["scene_path"], allow_pickle=False) as archive:
        scene = {key: torch.from_numpy(np.array(archive[key], copy=True)).float() for key in
                 ("heightfield_y_m", "min_point_world_xyz_m")}
        scene["dx_m"] = float(archive["dx_m"])
        scene["scene_scale"] = float(archive["scene_scale"])
    with np.load(paths["motion_path"], allow_pickle=False) as archive:
        names = archive["target_names"].tolist()
        parents = np.asarray(archive["target_parents"], dtype=np.int64)
        fps = float(archive["source_fps"])
        arrays = {key: torch.from_numpy(np.array(archive[key], copy=True)).float() for key in
                  ("root_positions", "local_rot_mats", "target_positions_yup", "target_global_rot_mats")}
    if fps != 30 or row["source_shard_fps"] != fps or len(arrays["root_positions"]) != row["frames_30hz"]:
        raise ValueError("Raw source must retain its recorded 30 Hz frame count")
    if len(set(names)) != 30 or len(names) != 30 or parents.shape != (30,):
        raise ValueError("Raw target_names/target_parents must define exactly SOMA30")
    skeleton = representation.skeleton
    native_names = list(skeleton.bone_order_names)
    if set(names) != set(native_names):
        raise ValueError("Raw target_names do not map exactly to the native SOMA30 skeleton")
    order = [names.index(name) for name in native_names]
    native_parents = torch.as_tensor(skeleton.joint_parents).tolist()
    for i, old_i in enumerate(order):
        actual_parent = None if parents[old_i] == -1 else names[int(parents[old_i])]
        expected_parent = None if native_parents[i] == -1 else native_names[native_parents[i]]
        if actual_parent != expected_parent:
            raise ValueError(f"Raw target_parents differs at {native_names[i]}")
    frames = len(arrays["root_positions"])
    for key, tail in (("root_positions", (3,)), ("target_positions_yup", (30, 3)),
                      ("local_rot_mats", (30, 3, 3)), ("target_global_rot_mats", (30, 3, 3))):
        if arrays[key].shape != (frames, *tail) or not torch.isfinite(arrays[key]).all():
            raise ValueError(f"Raw {key} has missing, nonfinite or misaligned frames")
    local, global_rot = arrays["local_rot_mats"][:, order], arrays["target_global_rot_mats"][:, order]
    joints = arrays["target_positions_yup"][:, order]
    _matrix_rotations(local, "local_rot_mats")
    _matrix_rotations(global_rot, "target_global_rot_mats")
    if not torch.allclose(joints[:, skeleton.root_idx], arrays["root_positions"], atol=1e-5, rtol=0):
        raise ValueError("Target root positions do not agree with raw root_positions")
    contacts = _native_foot_features(joints, skeleton, scene)
    canonical, initial = representation({"posed_joints": joints, "global_joint_rots": global_rot,
                                        "local_joint_rots": local, "translation": arrays["root_positions"],
                                        "foot_contacts": contacts},
                                       to_normalize=False, return_init_heading_info=True)
    angle = 2 * torch.atan2(initial["init_heading_quat"][2], initial["init_heading_quat"][0])
    world = representation.change_first_heading(canonical[None], angle.reshape(1), False, False)[0]
    world[:, [0, 2]] += initial["root_pos_init_xz"]
    if world.shape != (frames, 366) or not torch.isfinite(world).all():
        raise ValueError("Native retarget-world feature construction did not produce finite global366")
    root_error = (world[:, :3] - arrays["root_positions"]).abs().max().item()
    recovered = representation.inverse(world, False, return_all=False,
                                        joint_positions_from="global_rot_data")["posed_joints"]
    fk_error = (recovered - joints).norm(dim=-1).max().item()
    if root_error > 1e-5 or fk_error > 3e-5:
        raise ValueError(f"Native world restoration mismatch: root={root_error}, FK={fk_error}")
    manifest = {"schema": PROVIDER_SCHEMA, "source_id": row["source_id"], "split": row["split"],
                "source_group": row["source_group"], "terrain_family": row["terrain_family"],
                "source_manifest_identity": row["source_manifest_identity"], "source_frames": frames,
                "fps": fps, "timestamps": "derived_source_frame_index_divided_by_30",
                "coordinate_frame": expected_frame, "source_file_sha256": hashes,
                "world_tensor_sha256": tensor_state_sha256({"world": world}),
                "source_joint_tensor_sha256": tensor_state_sha256({"joints": joints, "local_rotations": local}),
                "root_roundtrip_max_abs_m": root_error, "fk_roundtrip_max_m": fk_error,
                "native_foot_provenance": "derived_nearest_bound_heightfield_height_lt0.10_velocity_lt0.15_mps",
                "v2_full_body_contacts": "unknown_all_validity_false_no_anchors",
                "license_evidence": "record_motion_asset_license_chain_not_recorded",
                "upstream_training_approved": recording["training_approved"],
                "upstream_intent_status": recording["intent_status"],
                "data_kind": "engineering", "quality_accepted": False}
    return world, joints, scene, row, manifest


def center_source_window(world, joints, scene, representation, *, observation_config):
    """One heading-retained translation applied to source, RIC heights and scene."""
    from motionbricks.data.local_terrain_observation import extract_local_terrain_observation
    from motionbricks.data.native_mb_vertical_frame import observed_terrain_origin, shift_global_motion
    from .contracts import NativeWindowFrame

    raw = world[None]
    coarse = extract_local_terrain_observation(scene["heightfield_y_m"], scene["min_point_world_xyz_m"],
        scene["dx_m"], raw[:, 3, :3], torch.atan2(raw[:, 3, 4], raw[:, 3, 3]), raw.new_zeros(1, 2),
        config=observation_config)
    origin_y = observed_terrain_origin(raw[:, 3, 1], coarse.as_model_inputs())
    frame = NativeWindowFrame.from_global_history(raw[:, :4], vertical_origin_m=origin_y)
    centered = shift_global_motion(raw, representation, -origin_y)
    centered[..., [0, 2]] -= frame.translation_world_m[:, None, [0, 2]]
    centered_scene = {"heightfield_y_m": scene["heightfield_y_m"] - origin_y[0],
                      "min_point_world_xyz_m": scene["min_point_world_xyz_m"] - frame.translation_world_m[0],
                      "dx_m": scene["dx_m"], "scene_scale": scene["scene_scale"]}
    return centered, frame.points_to_window(joints[None]), centered_scene, frame


def source_root_corridor(path_xz, count, history):
    """Native predicted-Root observer's pure curve rule, without installing hooks."""
    centers = F.interpolate(path_xz.transpose(1, 2), size=count,
                            mode="linear", align_corners=True).transpose(1, 2)
    delta = torch.cat((centers[:, 1:2] - centers[:, :1], centers[:, 2:] - centers[:, :-2],
                       centers[:, -1:] - centers[:, -2:-1]), 1)
    length = delta.norm(dim=-1, keepdim=True)
    heading = history[:, -1, 3:5]
    forward = torch.stack((heading[:, 1], heading[:, 0]), -1)[:, None]
    tangent = torch.where(length > 1e-5, delta / length.clamp_min(1e-5), forward)
    return centers, torch.stack((tangent[..., 1], -tangent[..., 0]), -1)


def legacy_source_root_observations(world, scene, *, observation_config):
    """Offline source-Root corridor with the actual native patch/prism semantics."""
    from motionbricks.data.local_terrain_observation import extract_local_terrain_observation
    from motionbricks.data.unified_scene_field import VolumetricEgoSampler, field_from_heightfield

    history, frames = world[:, :4], world.shape[1] - 1
    path = torch.cat((history[:, -1:, [0, 2]], world[:, 4:frames-4, [0, 2]]), 1)
    velocity = (world[:, -1, [0, 2]] - world[:, -2, [0, 2]]) * 30
    yaw = torch.atan2(history[:, -1, 4], history[:, -1, 3])
    displacement = path[:, -1] - history[:, -1, [0, 2]]
    direction = torch.stack((yaw.cos() * displacement[:, 0] - yaw.sin() * displacement[:, 1],
                             yaw.sin() * displacement[:, 0] + yaw.cos() * displacement[:, 1]), -1)
    height = extract_local_terrain_observation(scene["heightfield_y_m"], scene["min_point_world_xyz_m"],
        scene["dx_m"], history[:, -1, :3], yaw, direction,
        config=observation_config, query_path_world_xz=path[:, 1:],
        query_path_valid=torch.ones(path[:, 1:].shape[:2], dtype=torch.bool)).as_model_inputs()
    sampler = VolumetricEgoSampler()
    points = sampler.points(history[..., :5], velocity, (frames - 4) / 30)
    side = sampler.config.grid_size
    centers, right = source_root_corridor(path, side, history)
    ids = torch.arange(side ** 3)
    lateral = torch.linspace(-sampler.config.corridor_halfwidth_m, sampler.config.corridor_halfwidth_m, side)
    xz = centers[:, ids % side] + right[:, ids % side] * lateral[ids // (side * side)][None, :, None]
    points[:, 1, :, 0], points[:, 1, :, 2] = xz[..., 0], xz[..., 1]
    volume = sampler.observe_points(history[..., :5], points, field_from_heightfield(scene))
    height["distance_semantics"] = "legacy_heightfield_sampled_surface_height"
    volume["distance_semantics"] = "native_heightfield_prism_sdf_m"
    return height, volume


class RealEngineeringProvider:
    def __init__(self, spec, codec, representation, pose_args, encoder_hash, normalization_hash, observation_config):
        self.spec, self.codec, self.representation, self.pose_args = deepcopy(spec), codec, representation, pose_args
        self.observation_config = observation_config
        self.world, self.joints, self.scene, self.row, source_manifest = load_bound_world(spec, representation)
        if codec.training or any(p.requires_grad for p in codec.parameters()):
            raise ValueError("Real label generation requires a frozen eval-mode final codec")
        q = codec.quantizer
        if q.num_heads != 8 or q.nb_code_per_head != 10 or codec.code_dim != 256:
            raise ValueError("This provider only accepts the real final eight-head ten-entry codec")
        codebook_state = {"codebook": q.vq.codebook}
        codebook_state.update({"project_in." + k: v for k, v in q.vq.project_in.state_dict().items()})
        codebook_state.update({"project_out." + k: v for k, v in q.vq.project_out.state_dict().items()})
        self.identity = FrozenCodebookIdentity(QuantizerConfig(10, num_heads=8, code_dim=256, decay=q.mu),
            tensor_state_sha256(codebook_state), tensor_state_sha256(codec.encoder.state_dict()),
            representation.name, normalization_sha256=normalization_hash)
        self.windows, self.cursors, windows_manifest = {}, {8: 0, 12: 0, 16: 0}, []
        before = tensor_state_sha256(codec.state_dict())
        with torch.no_grad():
            for horizon in (8, 12, 16):
                starts = spec["window_starts"][str(horizon)]
                if not starts or len(set(starts)) != len(starts):
                    raise ValueError("window_starts requires explicit nonempty unique starts for 8/12/16")
                for start in starts:
                    if type(start) is not int or start < 0 or start + horizon * 4 >= len(self.world):
                        raise ValueError("Source window must retain its real continuous T+1 endpoint")
                    stop = start + horizon * 4 + 1
                    world, joints, scene, frame = center_source_window(
                        self.world[start:stop], self.joints[start:stop], self.scene, representation,
                        observation_config=observation_config)
                    local = representation.dual_rep.global_to_local(world, False, True,
                        lengths=torch.tensor([world.shape[1]]))[:, :-1].detach()
                    features = codec.extract_feature(local, codec.encoder_input_feature_mode)
                    latent = codec.encoder(features.transpose(1, 2).contiguous())
                    _, tokens, _ = q.vq(latent.transpose(1, 2).contiguous(), freeze_codebook=True)
                    tokens = tokens.long()
                    if tokens.shape != (1, horizon, 8) or ((tokens < 0) | (tokens >= 10)).any():
                        raise ValueError("Frozen native codec did not produce valid per-head labels")
                    record = {"start_frame": start, "endpoint_frame": stop - 1, "horizon_tokens": horizon,
                              "world_tensor_sha256": tensor_state_sha256({"world": world}),
                              "normalized_local_sha256": tensor_state_sha256({"local": local}),
                              "tokens_sha256": tensor_state_sha256({"tokens": tokens}),
                              "translation_world_m": frame.translation_world_m.tolist(),
                              "heading_rotation": "retained_no_extra_window_yaw", "ce_token_zero": True}
                    self.windows[horizon, start] = dict(world=world, joints=joints, scene=scene,
                        frame=frame, local=local, tokens=tokens.detach().clone(), manifest=record)
                    windows_manifest.append(record)
        if tensor_state_sha256(codec.state_dict()) != before:
            raise ValueError("Frozen codec/quantizer buffers changed during real label construction")
        self.manifest = dict(source_manifest, windows=windows_manifest, sampling_policy=POLICY_ID,
                             codec_tensor_sha256=before, normalization_sha256=normalization_hash,
                             scene_encoder_sha256=encoder_hash,
                             final_checkpoint_sha256=spec["artifacts"]["checkpoint_sha256"],
                             assembly="native_final_pose_with_height_and_projected_frozen_scene_memory",
                             observation_scope="offline_source_root_teacher_not_new_body_time_actor",
                             height_observation_config=asdict(observation_config),
                             volume_distance_semantics="native_heightfield_prism_sdf_m",
                             corridor_lateral_axis="path_tangent_native_curve_rule",
                             ce_policy="all_valid_codes_including_latent0")
        self.lineage = make_training_label_lineage(self.identity, data_kind="engineering", codec_sha256=before,
            scene_encoder_sha256=encoder_hash, normalization_sha256=normalization_hash,
            label_manifest_sha256=_json_hash(self.manifest))

    @torch.no_grad()
    def __call__(self, request):
        from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

        if request.batch_size != 1 or request.policy_id != POLICY_ID:
            raise ValueError("Real engineering source provider requires batch1 and its explicit legacy policy")
        if request.root_source != ("source",) or request.history_source != ("source",):
            raise ValueError("Real provider has no predicted Root or accepted generated-history recovery target")
        horizon = request.horizon_tokens
        starts = self.spec["window_starts"][str(horizon)]
        start = starts[self.cursors[horizon] % len(starts)]
        self.cursors[horizon] += 1
        saved = self.windows[horizon, start]
        world, local = saved["world"].clone(), saved["local"]
        frames = horizon * 4
        history_id = tensor_state_sha256({"history": world[:, :4]})
        target_id = saved["manifest"]["world_tensor_sha256"]
        root_rep = self.representation.dual_rep.local_motion_rep if self.pose_args[
            "cond_root_feature_is_from_motion_rep"] == "local" else self.representation
        root_motion = local if root_rep is not self.representation else self.representation.normalize(world[:, :-1])
        root_values = extract_feature_from_motion_rep(root_motion, root_rep, self.pose_args["cond_root_feature"])
        root_height = world[:, :-1, 1:2] - world[:, 3:4, 1:2]
        root_id = tensor_state_sha256({"values": root_values, "height": root_height})
        observation, volume = legacy_source_root_observations(world, saved["scene"],
                                                             observation_config=self.observation_config)
        binding = dict(policy_id=POLICY_ID, horizon_tokens=horizon, root_condition_id=[root_id], history_id=[history_id])
        observation.update(binding)
        volume.update(binding)
        batch = {"schema": BATCH_SCHEMA, "data_kind": "engineering", "world": world,
                 "history": world[:, :4].clone(), "root_source": ["source"], "history_source": ["source"],
                 "root_condition": dict(values=root_values, height=root_height, source=["source"],
                     target_id=[target_id], history_id=[history_id], condition_id=[root_id]),
                 "observation": observation, "volume": volume,
                 "masks": dict(frames=torch.ones(1, frames, dtype=torch.bool),
                     motion_loss_frames=torch.arange(frames)[None] >= 4,
                     tokens=torch.ones(1, horizon, 8, dtype=torch.bool)),
                 "contact_targets": torch.full((1, frames, 15), torch.nan),
                 "contact_valid": torch.zeros(1, frames, 15, dtype=torch.bool),
                 "contacts": {key: torch.zeros(1, frames, 15, dtype=torch.bool)
                              for key in ("valid", "actual_valid", "anchor_valid")},
                 "recovery": None, "pose_tokens": saved["tokens"].clone(), "label_lineage": dict(self.lineage),
                 "source_id": [self.row["source_id"]], "family_id": [self.row["source_group"]],
                 "scene_id": [self.manifest["source_file_sha256"]["scene_path"]], "policy_id": [POLICY_ID],
                 "target_id": [target_id], "history_id": [history_id], "split": [self.row["split"]],
                 "source_frame_indices": torch.arange(start, start + frames + 1)[None],
                 "timestamps_s": torch.arange(start, start + frames + 1, dtype=torch.float64)[None] / 30,
                 "window_translation_world_m": saved["frame"].translation_world_m.clone(),
                 "source_manifest": self.manifest}
        validate_training_batch(batch, lineage=self.lineage, policy_id=POLICY_ID)
        return batch


def _load_components(spec, *, include_pose):
    from omegaconf import OmegaConf
    from motionbricks.data.native_mb_models import _resolved_backbone_args, _vqvae_config
    from motionbricks.data.native_mb_terrain_protocol import terrain_observation_config
    from motionbricks.data.unified_terrain import SharedTerrainEncoder
    from motionbricks.motion_backbone.neural_modules.pose_backbone import pose_backbone_network
    from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import GlobalRootGlobalJoints
    from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
    from motionbricks.motionlib.core.utils.stats import Stats

    paths = _paths(spec)
    torch.set_num_threads(2)
    for key, expected in (("architecture_bundle", "bundle_sha256"), ("checkpoint", "checkpoint_sha256")):
        if _file_hash(paths[key]) != spec["artifacts"][expected]:
            raise ValueError(f"{key} content differs from configured {expected}")
    bundle = torch.load(paths["architecture_bundle"], map_location="cpu", mmap=True, weights_only=False)
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
    if (bundle["schema"] != "motionbricks.mixed_online_soma_bundle.v1" or
            checkpoint["schema"] != "motionbricks.unified.full_corpus.v1" or
            checkpoint["contract"]["bundle_sha256"] != spec["artifacts"]["bundle_sha256"]):
        raise ValueError("Final checkpoint/architecture bundle identity mismatch")
    stats = Stats()
    stats.register_from_tensors(bundle["stats_mean"], bundle["stats_std"])
    normalization_hash = tensor_state_sha256({"mean": bundle["stats_mean"], "std": bundle["stats_std"]})
    rep = GlobalRootGlobalJoints(30, SOMASkeleton30(), "somaskel30_dual_root_global_joints", stats=stats)
    if list(rep.skeleton.bone_order_names) != bundle["bone_names"]:
        raise ValueError("Bundle and native SOMA skeleton identities differ")
    torch.testing.assert_close(rep.skeleton.neutral_joints, bundle["neutral_joints"], atol=1e-6, rtol=0)
    codec = _vqvae_config(OmegaConf.create(bundle["tokenizer_config"]), rep.dual_rep.local_motion_rep)
    state = checkpoint["model_state"]
    codec.load_state_dict(_state(state, "codec."), strict=True)
    codec.eval().requires_grad_(False)
    encoder = SharedTerrainEncoder(grid_size=8)
    encoder.load_state_dict(_state(state, "adaptation.encoder."), strict=True)
    encoder.eval().requires_grad_(False)
    encoder_hash = tensor_state_sha256(encoder.state_dict())
    pose_args = _resolved_backbone_args("pose", OmegaConf.create(bundle["pose_config"]), codec)
    root_range_policy = bundle["root_config"]["model"]["args"]["observation_range_policy"]
    provider = RealEngineeringProvider(spec, codec, rep, pose_args, encoder_hash, normalization_hash,
                                       terrain_observation_config(root_range_policy))
    if not include_pose:
        return None, provider
    backbone = pose_backbone_network(rep, pose_args)
    backbone.load_state_dict(_state(state, "pose."), strict=True)
    adapter = NativePoseAdapter(backbone, encoder, volume_width=128)
    projection = adapter.backbone._terrain_encoder.volume_projection
    with torch.no_grad():
        projection.weight.copy_(state["adaptation.pose.output.weight"])
        projection.bias.zero_()
    adapter.eval().requires_grad_(False)
    trainer = SceneControlTrainer(adapter, lineage=provider.lineage, policy_id=POLICY_ID).eval()
    trainer.requires_grad_(False)
    if any(value.device.type != "cpu" for value in trainer.state_dict().values()):
        raise ValueError("Real engineering factory must remain CPU-only")
    return trainer, provider


def prepare_real_engineering_config(provider_spec, *, budget, curriculum):
    """Resolve actual frozen-codec lineage in memory; no Pose forward or optimizer."""
    checked = TrainingBudget(**budget).validate("forward")
    if checked.kind != "diagnostic" or checked.batch_size != 1:
        raise ValueError("Real provider execution is limited to zero-Adam batch1 engineering diagnostics")
    _, provider = _load_components(provider_spec, include_pose=False)
    return {"schema": CONFIG_SCHEMA, "policy_id": POLICY_ID, "label_lineage": dict(provider.lineage),
            "provider": deepcopy(provider_spec), "budget": asdict(checked), "curriculum": deepcopy(curriculum)}


def make_real_engineering_factory(config):
    """CLI factory: strict real final parameters and bound real source provider."""
    TrainingBudget(**config["budget"]).validate("forward")
    if config["schema"] != CONFIG_SCHEMA or config["policy_id"] != POLICY_ID:
        raise ValueError("Real provider requires the resolved independent legacy engineering configuration")
    trainer, provider = _load_components(config["provider"], include_pose=True)
    if config["label_lineage"] != provider.lineage:
        raise ValueError("Resolved label lineage differs from the actual final codec/source/window manifest")
    return trainer, provider
