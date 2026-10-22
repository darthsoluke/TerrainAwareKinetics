"""Frozen native Root predictions as Pose conditioning, without label leakage."""

import torch

from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from .native_mb_online_conditions import (sparse_planar_conditions, planar_goal, goal_direction_local,
    VELOCITY_PROTOCOL, sparse_velocity_conditions, velocity_goal, velocity_direction_local, apply_terminal_velocity)
from .native_mb_online_terrain import gather_online_terrain_observation


@torch.no_grad()
def predict_training_root(pose_model, batch, sample_info):
    root_model = getattr(pose_model, 'online_root_model', None)
    if root_model is None or not root_model.args.get('online_conditioning'):
        raise ValueError('Online Pose training requires a verified frozen online Root model')
    if any(parameter.requires_grad for parameter in root_model.parameters()):
        raise ValueError('Online training Root must be frozen')
    if root_model.args.get('vertical_frame') != pose_model.args.get('vertical_frame'):
        raise ValueError('Training Root and Pose vertical frames differ')
    if root_model.args.get('observation_range_policy') != pose_model.args.get('observation_range_policy'):
        raise ValueError('Training Root and Pose observation range policies differ')
    root_model.eval().to(batch['global_motions'].device)
    global_rep, local_rep = pose_model.global_motion_rep, pose_model.local_motion_rep
    global_root = extract_feature_from_motion_rep(batch['global_motions'], global_rep, 'root')
    local_root = extract_feature_from_motion_rep(batch['local_motions'], local_rep, 'root')
    local_pose = extract_feature_from_motion_rep(batch['local_motions'], local_rep,
                                                root_model.args['local_pose_feature'])
    raw_root = global_rep.unnormalize(global_root)
    velocity_mode = pose_model.args.get('online_conditioning') == VELOCITY_PROTOCOL['version']
    if root_model.args.get('online_conditioning') != pose_model.args.get('online_conditioning'):
        raise ValueError('Training Root and Pose online protocols differ')
    if velocity_mode:
        conditions = sparse_velocity_conditions(global_root[:, :4], local_root[:, :4], local_pose[:, :4],
                                                velocity_goal(local_root[:, -1], global_root[:, -1]))
        raw_local = local_rep.unnormalize(local_root)
        direction = velocity_direction_local(raw_root[:, :4], velocity_goal(raw_local[:, -1], raw_root[:, -1]),
                                               (global_root.shape[1]-4)/30)
    else:
        conditions = sparse_planar_conditions(global_root[:, :4], local_root[:, :4], local_pose[:, :4],
                                             planar_goal(global_root[:, -1]))
        direction = goal_direction_local(raw_root[:, :4], planar_goal(raw_root[:, -1]))
    scene = gather_online_terrain_observation(batch, sample_info, direction,
        range_policy=pose_model.args.get('observation_range_policy'))
    # Preserve the frozen teacher's condition when only the student's precision changes.
    with torch.autocast(device_type=global_root.device.type, enabled=False):
        prediction = root_model.backbone_net(
            conditions['global_root_values'], conditions['has_global_root_values'],
            conditions['local_root_values'], conditions['has_local_root_values'],
            conditions['local_poses'], conditions['has_local_poses'], batch['num_tokens'],
            global_root_component_mask=conditions['global_root_component_mask'], terrain_observation=scene,
            **({'local_root_component_mask': conditions['local_root_component_mask']} if velocity_mode else {}),
            config={'allow_pred_out_of_reach_num_tokens': False})
    if not torch.equal(prediction['pred_num_tokens'].reshape(-1), batch['num_tokens'].reshape(-1)):
        raise ValueError('Frozen Root changed the known training horizon')
    predicted_root = prediction['pred_global_root_values'][:, :global_root.shape[1]].detach()
    predicted_local = pose_model.motion_rep.dual_rep.global_to_local(
        predicted_root, is_normalized=True, to_normalize=True,
        lengths=batch['num_tokens'].reshape(-1) * 4).detach()
    if velocity_mode:
        predicted_local = apply_terminal_velocity(predicted_local, conditions['local_root_values'],
            conditions['local_root_component_mask'], batch['num_tokens'] * 4)
    return predicted_root, predicted_local, scene
