"""Four-frame rollout supervision in the source's fixed world coordinates."""

import torch
import torch.nn.functional as F

from .root_path_terrain import PredictedRootObservation


FPS = 30
HISTORY_FRAMES = 4
PREFIX_FRAMES = 4
POSITION_SCALE_M = .05
VELOCITY_SCALE_MPS = .5
LOSS_WEIGHTS = dict(root_xyz=1., root_relative=1., global_rotation=.1, joint_velocity=.25)
TRAINABLE_PREFIXES = (
    'root.', 'pose.', 'codec.decoder.', 'adaptation.root_stage1.',
    'adaptation.root_stage2.', 'adaptation.pose.', 'adaptation.decoder.',
    'adaptation.height_encoder.',
)


def _source_span(source, offset, stride):
    if type(stride) is not int or stride != PREFIX_FRAMES:
        raise ValueError('Prefix supervision requires stride=4')
    if type(offset) is not int or offset < 0:
        raise ValueError('Source offset must be a nonnegative integer')
    if source.ndim != 3 or source.shape[0] != 1 or source.shape[2] != 366:
        raise ValueError('Source must have shape [1, T, 366]')
    if source.shape[1] < offset + HISTORY_FRAMES + stride:
        raise ValueError('Source needs at least offset+8 genuine frames')


def prefix_command(source, offset, stride=4):
    """Average next-prefix world VX/VZ and its final cos/sin heading.

    The caller supplies genuine source frames, without padded or held tails.
    """
    _source_span(source, offset, stride)
    with torch.no_grad():
        first = source[:, offset + HISTORY_FRAMES - 1]
        last = source[:, offset + HISTORY_FRAMES + stride - 1]
        velocity = (last[:, [0, 2]] - first[:, [0, 2]]) / (stride / FPS)
        return torch.cat((velocity, last[:, 3:5]), -1).detach()


def prefix_loss(rep, predicted, history, source, offset, stride=4):
    """Supervise only emitted frames 4:8, including the actual-history join.

    All motion is unnormalized global motion. Source targets stay in their
    original world frame even when the supplied played history has drifted.
    """
    _source_span(source, offset, stride)
    if predicted.shape != (1, 32, 366) or history.shape != (1, 4, 366) or source.shape != (1, 33, 366):
        raise ValueError('Need predicted [1,32,366], history [1,4,366], source [1,33,366]')
    options = dict(is_normalized=False, return_all=False, joint_positions_from='global_rot_data')
    future = predicted[:, HISTORY_FRAMES:HISTORY_FRAMES + stride]
    pred_pose = rep.inverse(future, **options)
    with torch.no_grad():
        target = source[:, offset + HISTORY_FRAMES - 1:offset + HISTORY_FRAMES + stride].detach()
        truth = rep.inverse(target, **options)
        boundary = rep.inverse(history[:, -1:].detach(), **options)['posed_joints']
    pred_joints, truth_joints = pred_pose['posed_joints'], truth['posed_joints']
    pred_root, truth_root = future[..., :3], target[:, 1:, :3]
    pred_relative = pred_joints - pred_root[:, :, None]
    truth_relative = truth_joints[:, 1:] - truth_root[:, :, None]
    # The first interval starts at the actual played frame, not decoded past.
    pred_velocity = torch.diff(torch.cat((boundary, pred_joints), 1), dim=1) * FPS
    truth_velocity = torch.diff(truth_joints, dim=1) * FPS
    losses = dict(
        root_xyz=F.smooth_l1_loss(pred_root / POSITION_SCALE_M, truth_root / POSITION_SCALE_M),
        root_relative=F.smooth_l1_loss(pred_relative / POSITION_SCALE_M, truth_relative / POSITION_SCALE_M),
        global_rotation=F.mse_loss(pred_pose['global_joint_rots'], truth['global_joint_rots'][:, 1:]),
        joint_velocity=F.smooth_l1_loss(pred_velocity / VELOCITY_SCALE_MPS,
                                      truth_velocity / VELOCITY_SCALE_MPS),
    )
    losses['total'] = sum(LOSS_WEIGHTS[name] * value for name, value in losses.items())
    return losses


class TrainingRootPathObservation(PredictedRootObservation):
    """Use the deployed post-Root observation in training as well as evaluation."""

    def _after_root(self, _module, _args, output):
        controller = self.controller
        context = controller._context
        if self.deferred or context is None or not controller._adapting:
            return
        with torch.no_grad():
            volume, self.height, self.path = self.sample(
                context['history'].detach(), context['scene'], output['pred_global_root_values'].detach(),
                controller.volume_observation, controller.observation,
            )
        with torch.set_grad_enabled(context['training']):
            controller.memory, controller.valid = controller.adaptation['encoder'](volume)
        controller.volume_observation = volume
        self.context = context


def configure_scope(controller):
    """Train Root/Pose and decoder adapters with deployed eval-mode behavior."""
    controller.eval().requires_grad_(False)
    controller.zero_grad(set_to_none=True)
    names = []
    for name, parameter in controller.named_parameters():
        if name.startswith(TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
            names.append(name)
    return names
