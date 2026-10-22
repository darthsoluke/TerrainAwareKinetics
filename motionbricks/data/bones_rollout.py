"""A single raw-SOMA control path for offline rollouts and engine integration."""

import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from motionbricks.data.bones_contract import file_sha256, require_same_snapshot
from motionbricks.data.bones_control import SomaMotionInference
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep


UE_PUBLISH_FRAMES = 8


def observed_history_features(history, representation, policy="predicted_features"):
    """Encode observed FK transforms, without modifying any published pose.

    MB full_navigation_agent._process_input_to_joint_transforms observes the
    executed skeleton. The experimental FK policy follows that route using the
    checkpoint's own representation (including its hips_pos heading convention).
    Predicted contacts remain predictions; no contact or force is inferred here.
    """
    if policy == "predicted_features":
        return history
    if policy != "executed_fk":
        raise ValueError("Unknown history geometry policy")
    if history.ndim != 3 or history.shape[1] != 4 or representation.root_mode != "global":
        raise ValueError("Observed geometry requires four raw global history frames")
    transforms = representation.inverse(history, False, return_all=False,
                                          joint_positions_from="global_rot_data")
    features, origin = representation({key: transforms[key] for key in (
        "posed_joints", "local_joint_rots", "global_joint_rots", "foot_contacts")},
        to_normalize=False, lengths=torch.full((len(history),), 4, device=history.device),
        return_init_heading_info=True)
    # Official feature extraction removes the initial heading even when the
    # representation retains per-frame heading. Restore it before world goals.
    heading = 2 * torch.atan2(origin["init_heading_quat"][..., 2], origin["init_heading_quat"][..., 0])
    features = representation.change_first_heading(features, heading, False, False)
    features[..., representation.indices["global_root_pos"]] = history[..., representation.indices["global_root_pos"]]
    return features


def load_control_models(root_run, pose_run, representation, device="cpu"):
    from scripts.train_bones_control import build_control_model

    contracts, models = {}, {}
    for stage, run in (("root", Path(root_run)), ("pose", Path(pose_run))):
        contract = json.loads((run / "run_contract.json").read_text())
        if contract["stage"] != stage or contract["post_processing"] is not None:
            raise ValueError("Expected an uncorrected SOMA control run")
        config_path = run / "upstream_config_overrides.yaml"
        if file_sha256(config_path) != contract["config_sha256"]:
            raise ValueError("Control configuration differs from the training run")
        config = OmegaConf.load(config_path)
        model = build_control_model(stage, config, representation, Path(contract["tokenizer_run"]),
                                    contract["tokenizer_sha256"])
        model.load_state_dict(torch.load(run / "checkpoints/final.ckpt", map_location="cpu",
                                         weights_only=False)["state_dict"], strict=True)
        model.requires_grad_(False).eval().to(device)
        models[stage], contracts[stage] = model, contract
    require_same_snapshot(contracts["root"]["dataset_snapshot"], contracts["pose"]["dataset_snapshot"])
    if contracts["root"]["tokenizer_sha256"] != contracts["pose"]["tokenizer_sha256"]:
        raise ValueError("Root and Pose tokenizer identities differ")
    args = {"min_tokens": models["pose"].args["min_tokens"],
            "max_tokens": models["pose"].args["max_tokens"],
            "cond_root_feature_is_from_motion_rep": models["pose"].args["cond_root_feature_is_from_motion_rep"]}
    return SomaMotionInference(models, args, device=device), contracts


def control_inputs(history, endpoint_root, endpoint_velocity, representation, target_pose=None, target_mask=None):
    """Inputs are raw global features and explicitly requested endpoint goals only.

    Y is up; velocity is world XZ in m/s; headings use the representation's cos/sin.
    Endpoint Y is always replaced by the last observed height, not future truth.
    """
    if history.ndim != 3 or history.shape[1] != 4 or endpoint_root.shape != (len(history), 4, 5):
        raise ValueError("Control requires four history frames and four endpoint root goals")
    if endpoint_velocity.shape not in ((len(history), 2), (len(history), 4, 2)):
        raise ValueError("Endpoint velocity must be world XZ, per sequence or per goal frame")
    endpoint_root = endpoint_root.clone()
    endpoint_root[..., 1] = history[:, -1:, 1]
    if (target_pose is None) != (target_mask is None):
        raise ValueError("A body goal requires both features and a mask")
    if target_pose is not None:
        if (target_pose.shape != history.shape or target_mask.shape != history.shape[:2]
                or target_mask.dtype != torch.bool or target_mask[:, :-1].any()
                or not torch.isfinite(target_pose).all()):
            raise ValueError("Only one finite endpoint reference pose is supported")
        endpoint_root[..., 1] = torch.where(target_mask, target_pose[..., 1], endpoint_root[..., 1])
    global_root = torch.cat((history[..., representation.indices["root"]], endpoint_root), 1)
    local_history = representation.dual_rep.global_to_local(
        history, False, False, lengths=torch.full((len(history),), 4, device=history.device))
    local_history = local_history[..., :4]
    local_goal = local_history.new_zeros((len(history), 4, 4))
    local_goal[..., 1:3] = endpoint_velocity[:, None] if endpoint_velocity.ndim == 2 else endpoint_velocity
    local_goal[..., 3] = history[:, -1:, 1]
    if target_pose is not None:
        local_goal[..., 3] = endpoint_root[..., 1]
    local_root = torch.cat((local_history, local_goal), 1)
    body = extract_feature_from_motion_rep(history, representation, "joint_positions_and_rotations")
    body = torch.cat((body, torch.zeros_like(body)), 1)
    root_mask = torch.ones(global_root.shape[:2], device=history.device, dtype=torch.bool)
    body_mask = root_mask.clone()
    body_mask[:, 4:] = False
    if target_pose is not None:
        target_body = extract_feature_from_motion_rep(target_pose, representation, "joint_positions_and_rotations")
        body[:, 4:] = torch.where(target_mask[..., None], target_body, 0)
        body_mask[:, 4:] = target_mask
    return global_root, root_mask, local_root, root_mask.clone(), body, body_mask


def intent_endpoint(history, velocity_xz, heading_rad, horizon_frames, fps=60, goal_acceleration_mps2=3.0):
    if horizon_frames < 8 or horizon_frames % 4:
        raise ValueError("Horizon must contain whole tokens and future frames")
    velocity = torch.as_tensor(velocity_xz, device=history.device, dtype=history.dtype).reshape(-1, 2)
    angle = torch.as_tensor(heading_rad, device=history.device, dtype=history.dtype).reshape(-1)
    if len(velocity) != len(history) or len(angle) != len(history):
        raise ValueError("Commands must match the sequence batch")
    goal = history[:, -1:, :5].expand(-1, 4, -1).clone()
    dt = (torch.arange(horizon_frames - 4, horizon_frames, device=history.device) - 3) / fps
    if goal_acceleration_mps2 <= 0:
        raise ValueError("Goal acceleration must be positive")
    observed_velocity = history[..., [0, 2]].diff(dim=1).mean(1) * fps
    delta = velocity - observed_velocity
    transition_time = delta.norm(dim=-1, keepdim=True) / goal_acceleration_mps2
    acceleration = delta / transition_time.clamp(min=1e-6)
    accelerating_time = torch.minimum(dt[None], transition_time)
    # This shapes requested endpoints only; no generated or published root is corrected.
    displacement = (observed_velocity[:, None] * dt[None, :, None]
                    + acceleration[:, None] * (accelerating_time * dt[None]
                                                - 0.5 * accelerating_time.square())[..., None])
    goal[..., [0, 2]] += displacement
    goal_velocity = observed_velocity[:, None] + acceleration[:, None] * accelerating_time[..., None]
    goal[..., 3] = angle.cos()[:, None]
    goal[..., 4] = angle.sin()[:, None]
    return goal, goal_velocity


class SomaLocomotionBackend:
    """Owns executed history; a request never revises previously published frames."""

    def __init__(self, inference, initial_history, horizon_frames=64, commit_frames=4,
                 argmax=False, goal_library=None, history_geometry=None):
        if initial_history.shape[1] != 4 or commit_frames not in (4, 8):
            raise ValueError("Four history frames and four/eight committed frames are required")
        self.inference = inference
        self.representation = inference.motion_rep
        self.horizon_frames, self.commit_frames = horizon_frames, commit_frames
        self.history = initial_history.detach().clone()
        self.history_geometry = history_geometry or getattr(inference, "history_geometry", "predicted_features")
        if self.history_geometry not in ("predicted_features", "executed_fk"):
            raise ValueError("Unknown history geometry policy")
        self.argmax = argmax
        self.goal_library = goal_library if goal_library is not None else getattr(inference, "goal_library", None)
        self.published_frames = 0

    @torch.inference_mode()
    def predict_plan(self, velocity_xz, heading_rad):
        """Predict from observed history without declaring future frames executed."""
        history = observed_history_features(self.history, self.representation, self.history_geometry)
        goal, velocity = intent_endpoint(history, velocity_xz, heading_rad,
                                         self.horizon_frames, self.representation.fps)
        target, mask, ids = None, None, None
        if self.goal_library is not None:
            target, mask, ids = self.goal_library.select(velocity_xz, heading_rad, goal)
        inputs = control_inputs(history, goal, velocity, self.representation, target, mask)
        if self.history.is_cuda:
            torch.cuda.synchronize(self.history.device)
        started = time.perf_counter()
        prediction, lengths = self.inference.predict(
            *inputs, self.horizon_frames // 4,
            config={"pose_token_sampling_use_argmax": self.argmax,
                    "final_root_pred_mode": "from_pose_module"})
        if self.history.is_cuda:
            torch.cuda.synchronize(self.history.device)
        elapsed = time.perf_counter() - started
        if not torch.isfinite(prediction).all() or (lengths.flatten() * 4 < 8).any():
            raise FloatingPointError("Invalid control prediction")
        return {"planned_motion": prediction[:, 4:self.horizon_frames].detach().clone(),
                "generation_seconds": elapsed, "goal_root": goal.detach().clone(),
                "reference_pose_ids": None if ids is None else ids.detach().clone(),
                "intent_velocity_xz_mps": torch.as_tensor(velocity_xz, device=self.history.device,
                                                           dtype=self.history.dtype).detach().clone(),
                "goal_velocity_xz_mps": velocity.detach().clone()}

    @torch.inference_mode()
    def step(self, velocity_xz, heading_rad):
        result = self.predict_plan(velocity_xz, heading_rad)
        committed = result.pop("planned_motion")[:, :self.commit_frames].clone()
        self.history = committed[:, -4:].clone()
        start = self.published_frames
        self.published_frames += self.commit_frames
        return {"motion": committed, "start_frame": start,
                "timestamps_s": torch.arange(start, start + self.commit_frames, device=committed.device) / self.representation.fps,
                **result}
