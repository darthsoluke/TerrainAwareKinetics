"""Planar smart-primitive goals with no fabricated future height or pose.

Offline end XZ/heading is a hindsight goal label, not a recorded WASD command.
Only four observed frames and that explicit goal enter the condition builder.
The original MB Root regression and Pose masked-token objectives are retained.
"""

from copy import deepcopy

import torch


ONLINE_PROTOCOL = {
    "version": "native-mb-planar-online-v1",
    "history_frames": 4,
    "future_goal": "one planar XZ and cos/sin heading goal in slot 7",
    "future_root_y": "unknown; predicted by Root",
    "future_body_pose": "unknown; predicted by Pose",
    "future_local_root_velocity": "unknown",
    "history_velocity": "first three forward intervals only; fourth crosses into future",
    "training_goal_source": "PARC endpoint XZ/heading hindsight goal, not recorded WASD",
    "terrain_direction": "current-root-local planar goal displacement",
    "terrain_time": "last observed history frame only",
    "pose_training_root": "frozen online Root prediction; no ground-truth trajectory substitution",
    "codec_context": "last observed pose frame only; one keyframe covered by native codec curriculum",
    "runtime_root": "native codec refinement; no projection or replacement",
    "runtime_postprocessing": "none",
}

VELOCITY_PROTOCOL = {**ONLINE_PROTOCOL,
    'version': 'native-mb-velocity-online-v2',
    'future_goal': 'terminal world-XZ velocity and desired cos/sin heading in slot 7',
    'future_root_xz': 'unknown; predicted by Root, including braking distance',
    'future_local_root_velocity': 'desired terminal planar velocity only; angular velocity and height unknown',
    'training_goal_source': 'source terminal velocity/heading as hindsight command goals, not recorded WASD',
    'terrain_direction': 'current-root-local desired velocity times the explicit horizon',
    'horizon': 'provided; duration is not inferred from a velocity-only goal',
    'codec_context': 'four observed poses; matched played_context4 training',
}

SOFT_INTENT_PROTOCOL = {**ONLINE_PROTOCOL,
    'version':'native-mb-soft-intent-online-v3',
    'future_goal':'separate learned soft XZ goal, VX/VZ intent, heading and horizon',
    'future_root_xz':'unknown; predicted without an endpoint overwrite',
    'future_local_root_velocity':'unknown; requested velocity is a soft input',
    'training_goal_source':'source-derived soft task goals; authored weak continuation/stop pairs declared',
    'codec_context':'four observed poses; matched joint decoder training',
    'runtime_postprocessing':'none'}


def online_protocol(version):
    for protocol in (ONLINE_PROTOCOL,VELOCITY_PROTOCOL,SOFT_INTENT_PROTOCOL):
        if version == protocol['version']:
            return dict(protocol)
    raise ValueError('Unknown online conditioning version')


def enable_velocity_conditioning(config):
    result = deepcopy(config)
    result['model']['args']['online_conditioning'] = VELOCITY_PROTOCOL['version']
    result['model']['backbone_network']['args']['online_conditioning'] = VELOCITY_PROTOCOL['version']
    if 'global_root_feature' in result['model']['args']:
        result['model']['backbone_network']['args']['local_component_conditioning'] = True
        result['model']['args']['prob_provide_num_tokens'] = 1.0
    return result


def sparse_velocity_conditions(history_root, history_velocity, history_pose, goal):
    """Root XYZ remains unknown; only desired terminal VX/VZ and heading are given.

Native LocalRootGlobalJoints is [angular velocity, world VX, world VZ, height].
This builder accepts raw or individually normalized feature values, as v1 does.
"""
    result = sparse_planar_conditions(history_root, history_velocity, history_pose, goal)
    result['global_root_values'][:, 7, :3] = 0
    result['global_root_component_mask'][:, 7, :3] = False
    result['has_global_root_values'] = result['global_root_component_mask'].any(-1)
    local_mask = torch.zeros_like(result['local_root_values'], dtype=torch.bool)
    local_mask[:, :3] = True
    local_mask[:, 7, 1:3] = True
    result['local_root_values'][:, 7, 1:3] = goal[:, :2].detach()
    result['local_root_component_mask'] = local_mask
    result['has_local_root_values'] = local_mask.any(-1)
    return result


def sparse_soft_intent_conditions(history_root,history_velocity,history_pose):
    result=sparse_velocity_conditions(history_root,history_velocity,history_pose,
                                     history_root.new_zeros((len(history_root),4)))
    result['global_root_values'][:,4:]=0
    result['global_root_component_mask'][:,4:]=False
    result['has_global_root_values']=result['global_root_component_mask'].any(-1)
    result['local_root_values'][:,3:]=0
    result['local_root_component_mask'][:,3:]=False
    result['has_local_root_values']=result['local_root_component_mask'].any(-1)
    return result


def velocity_goal(local_root, global_root):
    if local_root.shape != (global_root.shape[0], 4) or global_root.shape[-1] != 5:
        raise ValueError('Expected native terminal Root features')
    return torch.cat((local_root[:, 1:3], global_root[:, 3:5]), -1).detach()


def velocity_direction_local(history_root, goal, seconds):
    guide = goal.clone()
    duration = torch.as_tensor(seconds, device=goal.device, dtype=goal.dtype).reshape(-1, 1)
    guide[:, :2] = history_root[:, -1, [0, 2]] + goal[:, :2] * duration
    return goal_direction_local(history_root, guide)


def apply_terminal_velocity(trajectory, sparse_values, component_mask, lengths):
    """The native endpoint-velocity condition, restricted to explicitly known fields."""
    index = (lengths.reshape(-1, 1, 1).long() - 1).expand(-1, 1, 4)
    terminal = trajectory.gather(1, index)
    terminal = torch.where(component_mask[:, -1:], sparse_values[:, -1:], terminal)
    return trajectory.scatter(1, index, terminal)


def enable_online_conditioning(config):
    result = deepcopy(config)
    if "backbone_network" not in result["model"]:
        raise ValueError("Online conditioning applies to Root/Pose, not the tokenizer")
    if not result["model"]["backbone_network"]["args"].get("terrain_conditioning", False):
        raise ValueError("Online terrain conditioning requires terrain_conditioning")
    result["model"]["args"]["online_conditioning"] = ONLINE_PROTOCOL["version"]
    result["model"]["backbone_network"]["args"]["online_conditioning"] = ONLINE_PROTOCOL["version"]
    if "global_root_feature" in result["model"]["args"]:
        result["model"]["backbone_network"]["args"]["root_component_conditioning"] = True
        # Online replanning uses native 24..64-frame horizons, not off-range clips.
        result["model"]["args"]["prob_off_range_target_token"] = 0.0
    return result


def planar_goal(root):
    """Select only XZ and heading; root Y cannot become a goal input."""
    if root.ndim != 2 or root.shape[-1] != 5:
        raise ValueError("Root goal source must have shape [B,5]")
    return root[:, [0, 2, 3, 4]].detach()


def sparse_planar_conditions(history_root, history_velocity, history_pose, goal):
    """Shared training/runtime builder; values may be raw or normalized.

All inputs have exactly four history frames, so this interface cannot read a
future pose or height. Unknown numeric slots are storage sentinels, never known
zero measurements. The component mask travels through both Root decoder paths.
"""
    batch = history_root.shape[0]
    if history_root.shape != (batch, 4, 5) or history_velocity.shape != (batch, 4, 4):
        raise ValueError("Expected four history frames of Root [5] and velocity [4]")
    if history_pose.ndim != 3 or history_pose.shape[:2] != (batch, 4):
        raise ValueError("Expected four observed body poses")
    if goal.shape != (batch, 4):
        raise ValueError("Planar goal must have shape [B,4]: X,Z,cos(yaw),sin(yaw)")
    for value in (history_root, history_velocity[:, :3], history_pose, goal):
        if value.device != history_root.device or not torch.isfinite(value).all():
            raise ValueError("Observed history and planar goal must be finite on the same device")
    root = history_root.new_zeros(batch, 8, 5)
    root[:, :4] = history_root.detach()
    root[:, 7, [0, 2, 3, 4]] = goal.detach()
    component = torch.zeros_like(root, dtype=torch.bool)
    component[:, :4] = True
    component[:, 7, [0, 2, 3, 4]] = True
    velocity = history_velocity.new_zeros(batch, 8, 4)
    velocity[:, :3] = history_velocity[:, :3].detach()
    has_velocity = torch.zeros(batch, 8, dtype=torch.bool, device=root.device)
    has_velocity[:, :3] = True
    pose = history_pose.new_zeros(batch, 8, history_pose.shape[-1])
    pose[:, :4] = history_pose.detach()
    has_pose = torch.zeros_like(has_velocity)
    has_pose[:, :4] = True
    return {"global_root_values": root, "has_global_root_values": component.any(-1),
            "global_root_component_mask": component, "local_root_values": velocity,
            "has_local_root_values": has_velocity, "local_poses": pose,
            "has_local_poses": has_pose}


def goal_direction_local(history_root, goal):
    """Unnormalized XZ displacement expressed in the current body yaw frame."""
    current = history_root[:, -1]
    delta = goal[:, :2] - current[:, [0, 2]]
    yaw = torch.atan2(current[:, 4], current[:, 3])
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * delta[:, 0] - s * delta[:, 1],
                        s * delta[:, 0] + c * delta[:, 1]), -1).detach()


def dense_history_pose(local_pose):
    if local_pose.ndim != 3 or local_pose.shape[1] < 4:
        raise ValueError("Pose sequence must contain four history frames")
    mask = torch.zeros(local_pose.shape[:2], dtype=torch.bool, device=local_pose.device)
    mask[:, :4] = True
    return torch.where(mask[..., None], local_pose.detach(), torch.zeros_like(local_pose)), mask


def validate_online_inputs(root, root_mask, component_mask, local_root, local_mask, poses, pose_mask, config,
                           *, version=ONLINE_PROTOCOL['version'], local_component_mask=None):
    """Reject incompatible goals and output/debug overrides before inference."""
    batch = root.shape[0]
    expected = torch.zeros(batch, 8, 5, dtype=torch.bool, device=root.device)
    expected[:, :4] = True
    expected[:, 7, [0, 2, 3, 4]] = True
    velocity_mode = online_protocol(version)['version'] == VELOCITY_PROTOCOL['version']
    soft_mode=version==SOFT_INTENT_PROTOCOL['version']
    if velocity_mode:
        expected[:, 7, :3] = False
    if soft_mode:
        expected[:,4:]=False
    if component_mask is None or component_mask.dtype != torch.bool or component_mask.shape != expected.shape:
        raise ValueError("Online inference requires a boolean [B,8,5] component mask")
    if not torch.equal(component_mask, expected) or not torch.equal(root_mask, expected.any(-1)):
        raise ValueError("Online inputs require observed history and one planar goal, with future Y unknown")
    expected_velocity = torch.zeros(batch, 8, dtype=torch.bool, device=root.device)
    expected_velocity[:, :3] = True
    if velocity_mode or soft_mode:
        if velocity_mode:expected_velocity[:,7]=True
        expected_local = torch.zeros(batch, 8, 4, dtype=torch.bool, device=root.device)
        expected_local[:, :3] = True
        if velocity_mode:expected_local[:,7,1:3]=True
        if (local_component_mask is None or local_component_mask.dtype != torch.bool
                or not torch.equal(local_component_mask, expected_local)):
            raise ValueError('Velocity goals cannot supply future Root height or angular velocity')
    elif local_component_mask is not None:
        raise ValueError('Local component masks require the velocity-goal checkpoint')
    expected_pose = torch.zeros_like(expected_velocity)
    expected_pose[:, :4] = True
    if not torch.equal(local_mask, expected_velocity) or not torch.equal(pose_mask, expected_pose):
        raise ValueError("Online inputs cannot supply future velocity or body pose")
    if root.shape != expected.shape or local_root.shape != (batch, 8, 4) or poses.shape[:2] != (batch, 8):
        raise ValueError("Online sparse condition shapes are invalid")
    known_local=local_component_mask if velocity_mode or soft_mode else local_mask[...,None].expand_as(local_root)
    for values, mask in ((root, component_mask), (local_root, known_local),
                         (poses, pose_mask[..., None].expand_as(poses))):
        if not torch.isfinite(values[mask]).all():
            raise ValueError("Known online inputs must be finite")
    if (config.get("debug_ground_truth_root_trajectories") is not None
            or config.get("debug_ground_truth_pose_tokens") is not None
            or config.get("final_root_pred_mode", "from_pose_module") != "from_pose_module"):
        raise ValueError("Online inference forbids ground-truth substitution and output root replacement")
