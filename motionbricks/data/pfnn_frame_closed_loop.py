"""P2 source-relative targets for generated-history training.

Only factual source poses form labels. Both sensing passes use the selected
current history; the fine pass uses the model's own predicted Root corridor.
Each row chooses teacher or generated history for the entire rollout.
"""

import torch
import torch.nn.functional as F

from .native_mb_observed_history import reencode_played_history
from .pfnn_3d_scene import path_observation
from .pfnn_frame_actor import FPS, LOSS_WEIGHTS, advance
from ..motionlib.core.utils.rotations import cont6d_to_matrix


def transition_loss(rep, history, emitted, pred, source_previous, source_next):
    """Compare real frame displacements, without an absolute Root XZ target."""
    joints = rep.inverse(emitted[:, -1:], False, return_all=False,
                         joint_positions_from='global_rot_data')['posed_joints'][:, 0]
    previous = rep.inverse(history[:, -1:], False, return_all=False,
                           joint_positions_from='global_rot_data')['posed_joints'][:, 0]
    with torch.no_grad():
        truth = torch.stack((source_previous.detach(), source_next.detach()), dim=1)
        target = rep.inverse(truth, False, return_all=False,
                             joint_positions_from='global_rot_data')
        target_joints = target['posed_joints']
        root_delta = truth[:, 1, :3] - truth[:, 0, :3]
        joint_velocity = (target_joints[:, 1] - target_joints[:, 0]) * FPS
    losses = dict(
        root_delta=F.smooth_l1_loss(pred['root_delta'] / .05, root_delta / .05),
        body_fk=F.smooth_l1_loss((joints - pred['root_next'][:, None]) / .05,
                                (target_joints[:, 1] - truth[:, 1, None, :3]) / .05),
        rotation=F.mse_loss(cont6d_to_matrix(pred['local_rot6d']), target['local_joint_rots'][:, 1]),
        boundary_joint_velocity=F.smooth_l1_loss((joints - previous) * FPS / .5,
                                                 joint_velocity / .5),
        contact=F.binary_cross_entropy_with_logits(pred['contact_logits'],
                                                   truth[:, 1, rep.indices['foot_contacts']]),
    )
    losses['total'] = sum(LOSS_WEIGHTS[key] * value for key, value in losses.items())
    return losses


def relative_path_loss(path_world_xz, history, worlds, offset):
    """Future source displacements are labels, never sensor coordinates."""
    indices = offset + 3 + torch.arange(1, 9, device=worlds.device) * 3
    known = indices < worlds.shape[1]
    proposed = path_world_xz[:, known] - history[:, -1, [0, 2]][:, None]
    target = (worlds[:, indices[known]][..., [0, 2]]
              - worlds[:, offset + 3, [0, 2]][:, None]).detach()
    return F.smooth_l1_loss(proposed / .3, target / .3)


def closed_loop_loss(actor, rep, sampler, worlds, commands, field, start, steps,
                     generated_rows, *, path_weight=.25):
    if (generated_rows.shape != (len(worlds),) or generated_rows.dtype != torch.bool
            or generated_rows.device != worlds.device):
        raise ValueError('One explicit generated-history boolean per batch row is required')
    if (start < 0 or steps not in (4, 8, 16) or start + steps > commands.shape[1]
            or start + steps + 4 > worlds.shape[1]):
        raise ValueError('Rollout must fit the actual source window and command clock')
    times = torch.arange(start, start + steps, device=worlds.device)
    windows = worlds[:, times[:, None] + torch.arange(4, device=worlds.device)]
    teachers = reencode_played_history(windows.reshape(-1, 4, 366), rep).reshape(len(worlds), steps, 4, 366)
    history = teachers[:, 0]
    state = actor.initial_state(len(worlds))
    losses = {}
    for index, offset in enumerate(range(start, start + steps)):
        history = torch.where(generated_rows[:, None, None], history, teachers[:, index])
        command = commands[:, offset]
        coarse = sampler.observe(history[..., :5], command[:, :2], field, .8)
        root = actor.root_step(history, command, coarse, state)
        fine = path_observation(sampler, history[..., :5], root['path_world_xz'], coarse, field)
        pred = actor.pose_step(history, command, fine, root)
        emitted = advance(rep, history, pred)
        row = transition_loss(rep, history, emitted, pred, worlds[:, offset + 3], worlds[:, offset + 4])
        row['path'] = relative_path_loss(root['path_world_xz'], history, worlds, offset)
        row['total'] = row['total'] + path_weight * row['path']
        for key, value in row.items():
            losses[key] = losses.get(key, 0.) + value / steps
        history, state = emitted, root['state']
    return losses
