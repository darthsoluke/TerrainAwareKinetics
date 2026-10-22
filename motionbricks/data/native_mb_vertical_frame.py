"""One observed vertical reference for all native MB stages and runtime."""

import numpy as np
import torch


VERTICAL_FRAME = 'observed-pelvis-y-v1'
TERRAIN_VERTICAL_FRAME = 'observed-terrain-y-v1'


def validate_vertical_frame(value):
    if value not in (None, VERTICAL_FRAME, TERRAIN_VERTICAL_FRAME):
        raise ValueError(f'Unknown vertical frame: {value!r}')
    return value


def shift_global_motion(motion, rep, offset):
    """Shift all world-Y position channels, leaving rotations/velocities intact."""
    if motion.ndim != 3 or motion.shape[-1] != len(rep.indices['all']):
        raise ValueError('Expected full global motion [B,T,D]')
    offset = torch.as_tensor(offset, device=motion.device, dtype=motion.dtype).reshape(-1)
    if offset.numel() not in (1, motion.shape[0]) or not torch.isfinite(offset).all():
        raise ValueError('Vertical offset must be one finite constant per window')
    indices = np.concatenate((rep.indices['global_root_pos'][1:2], rep.indices['ric_data'][1::3]))
    result = motion.clone()
    result[..., indices] = result[..., indices] + offset[:, None, None]
    return result


def center_motion_window(motion, rep, *, normalized):
    """Use only observed frame 3; keep future relative height changes unchanged.

    Retain the immutable source training normalization for this paired
    warmstart experiment. Its Root-Y mean is a fixed coordinate gauge, not a
    measured terrain height or a tunable target for the generated body.
    """
    if motion.ndim != 3 or motion.shape[1] < 4:
        raise ValueError('Four observed frames are required')
    raw = rep.unnormalize(motion) if normalized else motion
    y_index = int(rep.indices['global_root_pos'][1])
    reference_y = rep.stats.mean[y_index].to(device=raw.device, dtype=raw.dtype)
    origin = (raw[:, 3, y_index] - reference_y).detach()
    centered = shift_global_motion(raw, rep, -origin)
    return (rep.normalize(centered) if normalized else centered), origin


def observed_terrain_origin(root_y, observation):
    """A coordinate origin from the current near-patch center, never a body target."""
    points, valid = observation['points'], observation['valid']
    if points.shape[1:] != (2, 9, 9, 3) or valid.shape != points.shape[:-1]:
        raise ValueError('Expected the actual current two-patch terrain observation')
    gap = points[:, 0, 4, 4, 1]
    if not valid[:, 0, 4, 4].all() or not torch.isfinite(gap).all():
        raise ValueError('Current terrain origin is unobserved; do not invent a ground height')
    origin = root_y.reshape(-1).detach() + gap.detach().to(root_y)
    if not torch.isfinite(origin).all():
        raise ValueError('Nonfinite observed terrain origin')
    return origin


def training_vertical_frame(motion, rep, args, *, batch=None, sample_info=None):
    version = validate_vertical_frame(args.get('vertical_frame'))
    if version == TERRAIN_VERTICAL_FRAME:
        if batch is None or sample_info is None:
            raise ValueError('Terrain-relative training needs the sampled source scene and history')
        from .native_mb_online_terrain import gather_online_terrain_observation
        observation = gather_online_terrain_observation(batch, sample_info,
            motion.new_zeros((len(motion), 2)))
        ids = sample_info['chosen_ids'].to(batch['terrain_root_world'].device)
        frames = sample_info['frame_indices'][:, 3].to(ids.device)
        origin = observed_terrain_origin(batch['terrain_root_world'][ids, frames, 1], observation)
        raw = shift_global_motion(rep.unnormalize(motion), rep, -origin)
        return rep.normalize(raw)
    return center_motion_window(motion, rep, normalized=True)[0] if version else motion


def center_sparse_conditions(root, poses, rep, pose_rep, *, pose_feature_mode,
                             vertical_frame=VERTICAL_FRAME, terrain_observation=None):
    """Center native unnormalized root and pose conditions with the same origin."""
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
    y_index = int(rep.indices['global_root_pos'][1])
    if vertical_frame == TERRAIN_VERTICAL_FRAME:
        if terrain_observation is None:
            raise ValueError('Terrain-relative inference requires a current observation')
        origin = observed_terrain_origin(root[:, 3, y_index], terrain_observation)
    else:
        origin = (root[:, 3, y_index] - rep.stats.mean[y_index].to(root)).detach()
    result_root, result_pose = root.clone(), poses.clone()
    result_root[..., y_index] -= origin[:, None]
    feature_indices = extract_feature_from_motion_rep(None, pose_rep, pose_feature_mode, fetch_feat_idx=True)
    world_y = set(int(i) for i in pose_rep.indices['ric_data'][1::3])
    channels = [i for i, index in enumerate(feature_indices) if int(index) in world_y]
    result_pose[..., channels] -= origin[:, None, None]
    return result_root, result_pose, origin
