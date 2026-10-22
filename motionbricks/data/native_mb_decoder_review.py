"""Offline decoder factorization; source futures never enter live inference."""

import torch

from motionbricks.helper.data_training_util import extract_feature_from_motion_rep


@torch.inference_mode()
def decoder_factorization(inference, source_global, source_local, call, online_prediction,
                          score_start=4, score_stop=63):
    """Replay the actual decoder with source/model tokens and root conditions.

    These crossed conditions are diagnostic, not necessarily physically valid
    motions or mathematical performance bounds. Observed keyframes, masks and
    the codebook remain fixed. Source tokens and root are offline oracles only.
    """
    codec = inference._vqvae_pose_model
    rep, local_rep = inference.global_motion_rep, inference.local_motion_rep
    if source_global.shape[0] != 1 or source_local.shape[:2] != source_global.shape[:2]:
        raise ValueError('Decoder review requires one aligned global/local window')
    frame_count = source_global.shape[1]
    if not 4 <= score_start < score_stop <= frame_count:
        raise ValueError('Scoring must exclude observed context and stay inside the window')
    if len(call['args']) != 1:
        raise ValueError('Expected the captured native decoder token argument')
    kwargs = call['kwargs']
    expected = {'target_cond', 'has_target_cond', 'external_cond',
                'use_overall_indices', 'token_mask'}
    if set(kwargs) != expected or kwargs['use_overall_indices']:
        raise ValueError('Decoder review requires the native per-head token layout')
    known = kwargs['has_target_cond']
    if known[:, 4:].any():
        raise ValueError('Offline replay must retain observed keyframes only')
    vertical_origin = None
    if inference._pose_model.args.get('vertical_frame'):
        from .native_mb_vertical_frame import center_motion_window
        _, vertical_origin = center_motion_window(source_global, rep, normalized=False)
        source_local = source_local.clone()
        source_local[..., local_rep.indices['global_root_y']] -= vertical_origin[:, None, None]
        source_local[..., local_rep.indices['ric_data'][1::3]] -= vertical_origin[:, None, None]
    normalized_local = local_rep.normalize(source_local)
    source_tokens = codec.encode_into_idx(normalized_local, fetch_overall_indices=False)
    model_tokens = call['args'][0]
    if source_tokens.shape != model_tokens.shape:
        raise ValueError('Source and model token horizons must match')
    source_external = extract_feature_from_motion_rep(
        normalized_local, local_rep, codec.decoder_external_cond_feature_mode)
    if source_external.shape != kwargs['external_cond'].shape:
        raise ValueError('Source and model external conditions must have identical layouts')

    roots = extract_feature_from_motion_rep(source_global, rep, 'root')
    offsets, heading, _ = inference._extract_initial_root_info(roots)
    source_fk = rep.inverse(source_global, False, return_all=True)['posed_joints']
    score = slice(score_start, score_stop)
    source_y = roots[:, score, 1]
    source_first_y_step = roots[:, 4, 1] - roots[:, 3, 1]
    rows = {}
    for token_source, tokens in (('model', model_tokens), ('source', source_tokens)):
        for root_source, external in (('model', kwargs['external_cond']),
                                      ('source', source_external)):
            conditions = {key: value.clone() if torch.is_tensor(value) else value
                          for key, value in kwargs.items()}
            conditions['external_cond'] = external.clone()
            reconstruction = codec.forward_decoder(tokens.clone(), **conditions)['recon_state']
            global_motion = inference.motion_rep.dual_rep.local_to_global(
                reconstruction, is_normalized=True, to_normalize=False,
                lengths=torch.tensor([frame_count], device=reconstruction.device))
            prediction = inference._reapply_initial_root_info({
                'pred_global_poses': global_motion,
                'reference_start_root_global_offsets': offsets,
                'reference_start_root_global_heading': heading})
            if vertical_origin is not None:
                from .native_mb_vertical_frame import shift_global_motion
                prediction = shift_global_motion(prediction, rep, vertical_origin)
            if not torch.isfinite(prediction).all():
                raise ValueError('Nonfinite decoder factorization output')
            parity_error = None
            if token_source == root_source == 'model':
                parity_error = float((prediction - online_prediction).abs().max())
                if parity_error > 2e-5:
                    raise ValueError('Decoder replay does not reproduce the actual online result')
            pred_root = extract_feature_from_motion_rep(prediction, rep, 'root')
            fk = rep.inverse(prediction, False, return_all=True)['posed_joints']
            relative_source = source_fk - source_fk[..., rep.skeleton.root_idx:rep.skeleton.root_idx + 1, :]
            relative_pred = fk - fk[..., rep.skeleton.root_idx:rep.skeleton.root_idx + 1, :]
            rows[f'{token_source}_tokens_{root_source}_root'] = {
                'tokens': token_source, 'external_root': root_source,
                'future_reference_used': token_source == 'source' or root_source == 'source',
                'root_y_mae_m': float((pred_root[:, score, 1] - source_y).abs().mean()),
                'root_y_signed_bias_m': float((pred_root[:, score, 1] - source_y).mean()),
                'current_frame_root_y_error_m': float((pred_root[:, 3, 1] - roots[:, 3, 1]).abs().mean()),
                'first_future_root_y_step_m': float((pred_root[:, 4, 1] - roots[:, 3, 1]).mean()),
                'source_first_future_root_y_step_m': float(source_first_y_step.mean()),
                'first_future_root_y_step_error_m': float(
                    (pred_root[:, 4, 1] - roots[:, 3, 1] - source_first_y_step).abs().mean()),
                'fk_mpjpe_m': float((fk[:, score] - source_fk[:, score]).norm(dim=-1).mean()),
                'root_relative_fk_mpjpe_m': float((relative_pred[:, score] - relative_source[:, score]).norm(dim=-1).mean()),
                'online_replay_max_abs_error': parity_error,
            }
    return {'scope': 'offline decoder-only oracle factorization, not a controller',
            'runtime_inputs_changed': False, 'known_keyframes_unchanged': True,
            'known_keyframe_indices': known.nonzero()[:, 1].unique().cpu().tolist(),
            'source_pose_token_match_fraction': float((source_tokens == model_tokens).float().mean()),
            'codec_external_feature_mode': codec.decoder_external_cond_feature_mode,
            'crossed_conditions_physical_validity': 'not guaranteed', 'cases': rows}
