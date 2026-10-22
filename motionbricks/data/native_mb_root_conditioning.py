"""Diagnostic implementation of MB section 7.3's decoder root interpolation.

The paper describes fixed pose tokens with interpolated decoder root inputs;
the public preview does not implement that utility. This implementation uses
piecewise-linear endpoint residual interpolation on XZ only. It is NOT upstream
code, a trained capability, a pose correction, or an approved runtime policy.
"""

from contextlib import contextmanager

import numpy as np
import torch


def interpolate_planar_residual(predicted, anchors, indices):
    """Preserve the predicted path's detail, interpolating only anchor errors."""
    predicted = np.asarray(predicted)
    anchors = np.asarray(anchors)
    indices = np.asarray(indices)
    if (predicted.ndim != 2 or predicted.shape[1] != 2 or
            anchors.shape != (len(indices), 2) or indices.ndim != 1 or
            not np.issubdtype(indices.dtype, np.integer) or len(indices) < 2 or
            np.any(np.diff(indices) <= 0) or indices[0] < 0 or indices[-1] >= len(predicted) or
            not np.isfinite(predicted).all() or not np.isfinite(anchors).all()):
        raise ValueError('Need finite XZ path and at least two ordered in-range anchors')
    residual = anchors - predicted[indices]
    clock = np.arange(len(predicted))
    correction = np.stack([np.interp(clock, indices, residual[:, axis]) for axis in range(2)], -1)
    return (predicted + correction).astype(predicted.dtype)


@contextmanager
def decoder_root_interpolation(infer):
    """Leave Root and Pose inference unchanged; alter only decoder conditioning.

    Explicit sparse input positions supply all anchors. No source trajectory,
    terrain query, output root rewrite, or foot constraint is available here.
    The original decoder produces ALL final motion channels.
    """
    name = '_decode_motions_from_predicted_root_and_pose_tokens'
    original, on_instance = getattr(infer, name), name in vars(infer)
    observed = []

    def decode(batch, config, info):
        if len(batch['pred_num_tokens']) != 1:
            raise ValueError('This diagnostic supports one character')
        count = int(batch['pred_num_tokens'].item()) * 4
        rep = infer.global_motion_rep
        xz = rep.indices['global_root_pos_2d']
        predicted = rep.unnormalize(batch['pred_global_root_values']).clone()
        inputs = rep.unnormalize(batch['global_root_values'])
        available = batch['has_global_root_values'][0].nonzero().flatten()
        indices = torch.where(available < 4, available, available + count - 8)
        anchors = inputs[0, available][:, xz]
        # MB reconstructs local XZ velocities from an implicit zero origin.
        # That origin must survive differencing/integration even when pose 0 is masked.
        if not bool((indices == 0).any()):
            indices = torch.cat([indices.new_zeros(1), indices])
            anchors = torch.cat([anchors.new_zeros(1, 2), anchors])
        refined = interpolate_planar_residual(
            predicted[0, :count, xz].detach().cpu().numpy(),
            anchors.detach().cpu().numpy(),
            indices.detach().cpu().numpy())
        predicted[0, :count, xz] = torch.as_tensor(refined, device=predicted.device)
        normalized = rep.normalize(predicted)
        local = infer.motion_rep.dual_rep.global_to_local(
            normalized, is_normalized=True, to_normalize=True, lengths=batch['pred_num_tokens'] * 4)
        # The final forward difference has no future sample. Keep the original
        # runtime's explicit terminal-velocity condition/padding contract.
        local[:, count - 1] = batch['pred_local_root_values'][:, count - 1]
        decoder_batch = dict(batch, pred_local_root_values=local, pred_global_root_values=normalized)
        observed.append(dict(frames=count, input_anchor_frames=indices.tolist(),
                             root_policy='piecewise_linear_XZ_residual_before_decoder_v2_origin',
                             pose_tokens_unchanged=True, output_rewritten=False))
        return original(decoder_batch, config, info)

    setattr(infer, name, decode)
    try:
        yield observed
    finally:
        if on_instance:
            setattr(infer, name, original)
        else:
            delattr(infer, name)
