import hashlib
import json

import numpy as np
import pytest
import torch

from motionbricks.data.bones_contract import dataset_snapshot, require_same_snapshot
from motionbricks.data.bones_evaluation import reconstruction_metrics, summarize_windows
from motionbricks.data.bones_models import (apply_root_velocity_condition, contact_geometry_losses,
                                          contact_masked_reconstruction, stance_anchor_distances)
from motionbricks.data.bones_seed import (BonesMotionDataset, build_representation, collate_bones_batch,
                                        deterministic_window_starts, official_sole_offsets, sole_positions,
                                        validation_window_records)
from motionbricks.data.soma_replay import skeleton_manifest
from motionbricks.helper.data_training_util import sample_motion_segments_from_motion_clips
from motionbricks.motionlib.core.utils.stats import Stats


def test_saved_confidence_survives_crop_and_padding(tmp_path):
    stats = Stats()
    stats.register_from_tensors(torch.zeros(366), torch.ones(366))
    masks = []
    for index, frames in enumerate((140, 180)):
        confidence = ((np.arange(frames)[:, None] + np.arange(4)[None]) % 3 == 0).astype(np.float32)
        features = np.zeros((frames, 366), dtype=np.float32)
        features[:, 0] = np.arange(frames)
        np.savez(tmp_path / f'{index}.npz', global_features=features, contact_confidence=confidence)
        masks.append(confidence)
    dataset = BonesMotionDataset(tmp_path, [{'shard': f'{index}.npz'} for index in range(2)], stats,
                                 crop_frames=160, training=True)
    items = [dataset[index] for index in range(2)]
    for index, item in enumerate(items):
        start = round(float(item['motion'][0, 0]))
        torch.testing.assert_close(item['contact_confidence'],
                                   torch.from_numpy(masks[index][start:start + len(item['motion'])]).bool())
    batch = collate_bones_batch(items)
    assert batch['contact_confidence'].shape == (2, 160, 4)
    assert not batch['contact_confidence'][0, 140:].any()


def test_validation_covers_same_beginning_middle_end_as_offline_review(tmp_path):
    stats = Stats()
    stats.register_from_tensors(torch.zeros(366), torch.ones(366))
    frames = 240
    features = np.zeros((frames, 366), dtype=np.float32)
    features[:, 0] = np.arange(frames)
    confidence = ((np.arange(frames)[:, None] + np.arange(4)[None]) % 2 == 0).astype(np.float32)
    np.savez(tmp_path / 'clip.npz', global_features=features, contact_confidence=confidence)
    records = [{'shard': 'clip.npz', 'frames_60hz': frames}]
    windows = validation_window_records(records, 128)
    dataset = BonesMotionDataset(tmp_path, windows, stats, crop_frames=129, training=False)
    starts = deterministic_window_starts(frames, 128)
    assert starts == [0, 55, 111]
    assert len(dataset) == 3
    for index, start in enumerate(starts):
        item = dataset[index]
        raw = dataset.stats.unnormalize(item['motion'])
        assert float(raw[0, 0]) == pytest.approx(start)
        assert float(raw[-1, 0]) == pytest.approx(start + 128)
        torch.testing.assert_close(item['contact_confidence'], torch.from_numpy(confidence[start:start + 129]).bool())
    assert deterministic_window_starts(129, 128) == [0]
    with pytest.raises(ValueError, match='additional root-velocity frame'):
        deterministic_window_starts(128, 128)
    with pytest.raises(ValueError, match='exceeds the saved shard'):
        BonesMotionDataset(tmp_path, [dict(records[0], window_start_frame=120)], stats,
                           crop_frames=129, training=False)[0]


def test_convergence_permission_is_bounded_and_not_full_training_approval():
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / 'scripts/train_bones_locomotion.py'
    spec = spec_from_file_location('bones_training_entry', path)
    entry = module_from_spec(spec)
    spec.loader.exec_module(entry)
    assert entry.canonical_cuda_uuid("abc-123") == "GPU-abc-123"
    assert entry.canonical_cuda_uuid("GPU-abc-123") == "GPU-abc-123"
    args = SimpleNamespace(stage='tokenizer', max_steps=10000, pipeline_smoke=False, convergence_pilot=True,
                           initialize_from='audited.ckpt', resume=None)
    report = {'full_training_gate_passed': False}
    protocol = {'experiment_kind': 'bounded_tokenizer_convergence'}
    preflight = {'automatic_preconditions_passed': True}
    assert entry.validate_training_scope(args, report, protocol, preflight) == 'bounded_tokenizer_convergence'
    assert not report['full_training_gate_passed']
    for attribute, value in (('stage', 'root'), ('max_steps', 10001), ('max_steps', 1000), ('initialize_from', None)):
        changed = SimpleNamespace(**vars(args))
        setattr(changed, attribute, value)
        with pytest.raises(RuntimeError, match='Convergence needs'):
            entry.validate_training_scope(changed, report, protocol, preflight)
    with pytest.raises(RuntimeError, match='Convergence needs'):
        entry.validate_training_scope(args, report, protocol, {'automatic_preconditions_passed': False})
    args.pipeline_smoke = True
    with pytest.raises(ValueError, match='not both'):
        entry.validate_training_scope(args, report, protocol, preflight)


def test_official_sampler_exposes_exact_auxiliary_alignment():
    motions = torch.zeros(3, 160, 2)
    motions[:, :, 0] = torch.arange(3)[:, None]
    motions[:, :, 1] = torch.arange(160)[None]
    lengths = torch.tensor([160, 140, 30])
    np.random.seed(44)
    torch.manual_seed(44)
    old_path = sample_motion_segments_from_motion_clips(motions, lengths, 128, 2)
    np.random.seed(44)
    torch.manual_seed(44)
    info = {}
    sampled = sample_motion_segments_from_motion_clips(motions, lengths, 128, 2, info=info)
    torch.testing.assert_close(old_path, sampled)
    torch.testing.assert_close(sampled[:, :, 0], info['chosen_ids'][:, None].expand(-1, 129).float())
    torch.testing.assert_close(sampled[:, :, 1], info['frame_indices'].float())


def test_uncertain_contact_has_no_reconstruction_gradient_but_body_does():
    prediction = torch.ones(1, 8, 6, requires_grad=True)
    target = torch.zeros_like(prediction)
    confidence = torch.ones(1, 8, 4, dtype=torch.bool)
    confidence[:, 3, 2] = False
    loss = contact_masked_reconstruction(prediction, target, confidence, [2, 3, 4, 5])
    loss.backward()
    assert prediction.grad[0, 3, 4] == 0
    assert prediction.grad[0, 3, 0] > 0
    assert prediction.grad[0, 3, 3] > 0
    with pytest.raises(ValueError, match='sampled motion window'):
        contact_masked_reconstruction(prediction, target, confidence[:, :7], [2, 3, 4, 5])


def test_geometry_losses_zero_for_stationary_grounded_feet_and_finite_gradients():
    soles = torch.zeros(2, 12, 4, 3, requires_grad=True)
    contacts = torch.ones(2, 12, 4, requires_grad=True)
    confidence = torch.ones_like(contacts, dtype=torch.bool)
    losses = contact_geometry_losses(soles, contacts, confidence, 60)
    assert all(value == 0 for value in losses.values())
    sum(losses.values()).backward()
    assert torch.isfinite(soles.grad).all() and not soles.grad.any()
    assert contacts.grad is None


def test_geometry_losses_reject_uncertain_stance_and_push_penetrating_feet_up():
    soles = torch.zeros(1, 4, 4, 3)
    soles[..., 0] = torch.arange(4)[None, :, None] / 60
    soles[..., 1] = -0.025
    soles.requires_grad_()
    contacts = torch.ones(1, 4, 4)
    confidence = torch.zeros_like(contacts, dtype=torch.bool)
    losses = contact_geometry_losses(soles, contacts, confidence, 60)
    assert losses['l_trusted_skate_mps'] == 0
    assert float(losses['l_sole_penetration_m'].detach()) == pytest.approx(0.020)
    sum(losses.values()).backward()
    assert (soles.grad[..., 1] < 0).all()
    assert not soles.grad[..., [0, 2]].any()


def test_source_stance_loss_scales_in_seconds_and_is_heading_translation_invariant():
    soles = torch.zeros(1, 8, 4, 3)
    soles[..., 0] = torch.arange(8)[None, :, None] / 60
    contacts = torch.ones(1, 8, 4)
    confidence = torch.ones_like(contacts, dtype=torch.bool)
    moving = soles.clone().requires_grad_()
    original = contact_geometry_losses(moving, contacts, confidence, 60)
    assert float(original['l_trusted_skate_mps'].detach()) == pytest.approx(1.0)
    original['l_trusted_skate_mps'].backward()
    assert (moving.grad[:, 0, :, 0] < 0).all() and (moving.grad[:, -1, :, 0] > 0).all()
    assert torch.isfinite(moving.grad).all()
    yaw = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    transformed = soles @ yaw.T + torch.tensor([12., 0., -5.])
    invariant = contact_geometry_losses(transformed, contacts, confidence, 60)
    torch.testing.assert_close(original['l_trusted_skate_mps'], invariant['l_trusted_skate_mps'])
    torch.testing.assert_close(original['l_sole_penetration_m'], invariant['l_sole_penetration_m'])
    half_rate = contact_geometry_losses(soles, contacts, confidence, 30)
    assert float(half_rate['l_trusted_skate_mps']) == pytest.approx(0.5)
    contacts[:, 4:] = 0
    confidence[:, :4] = False
    assert contact_geometry_losses(soles, contacts, confidence, 60)['l_trusted_skate_mps'] == 0


def test_geometry_loss_uses_deployment_fk_and_backpropagates_to_root_height():
    from kimodo.exports.motion_io import complete_motion_dict
    representation = build_representation()
    local = representation.dual_rep.local_motion_rep
    stats = Stats()
    stats.register_from_tensors(torch.zeros(365), torch.ones(365))
    local.stats = stats
    offsets, _ = official_sole_offsets(representation.skeleton)
    frames = 32
    root = torch.zeros(frames, 3)
    root[:, 1] = -(representation.skeleton.neutral_joints[representation.skeleton.foot_joint_idx] + offsets)[:, 1].min()
    motion = complete_motion_dict(torch.eye(3).expand(frames, 30, 3, 3).clone(), root, representation.skeleton, 60)
    target = local({'posed_joints': motion['posed_joints'], 'local_joint_rots': motion['local_rot_mats'],
                    'global_joint_rots': motion['global_rot_mats'], 'translation': root,
                    'foot_contacts': torch.ones(frames, 4)}, to_normalize=True)[None]
    lowered = target.clone()
    lowered[..., local.indices['global_root_y']] -= 0.08
    lowered.requires_grad_()
    pred = local.inverse(lowered, True, return_all=True)
    soles = sole_positions(pred['posed_joints'], pred['global_joint_rots'], representation.skeleton, offsets)
    confidence = torch.ones(1, frames, 4, dtype=torch.bool)
    losses = contact_geometry_losses(soles, pred['foot_contacts'], confidence, 60)
    losses['l_sole_penetration_m'].backward()
    assert (lowered.grad[..., local.indices['global_root_y']] < 0).all()
    assert torch.isfinite(lowered.grad).all()
    fake_positions = lowered.detach().clone()
    fake_positions[..., local.indices['ric_data']] += 10
    fake = local.inverse(fake_positions, True, return_all=True)
    fake_soles = sole_positions(fake['posed_joints'], fake['global_joint_rots'], representation.skeleton, offsets)
    torch.testing.assert_close(soles.detach(), fake_soles)


def test_geometry_losses_validate_shapes_and_units():
    soles = torch.zeros(1, 4, 4, 3)
    contacts = torch.ones(1, 4, 4)
    with pytest.raises(ValueError, match='match the sole frames'):
        contact_geometry_losses(soles, contacts[:, :3], contacts.bool(), 60)
    with pytest.raises(ValueError, match='Positive FPS'):
        contact_geometry_losses(soles, contacts, contacts.bool(), 0)
    with pytest.raises(ValueError, match='Soles must'):
        contact_geometry_losses(soles[:, :1], contacts[:, :1], contacts[:, :1].bool(), 60)


def test_stance_anchors_do_not_join_distinct_steps_or_channels():
    soles = torch.zeros(2, 8, 4, 3)
    soles[:, 4:, :, 0] = 10
    soles[1, :, :, 2] = 20
    soles[:, :, 2:, 2] += 5
    support = torch.ones(2, 8, 4, dtype=torch.bool)
    support[:, 3:5] = False
    distances = stance_anchor_distances(soles, support)
    assert not distances[support].any()
    shifted = soles + torch.tensor([12., 0., -8.])
    torch.testing.assert_close(distances[support], stance_anchor_distances(shifted, support)[support])


def test_support_objective_penalizes_slow_accumulated_drift_and_floating():
    soles = torch.zeros(1, 128, 4, 3)
    soles[..., 0] = torch.arange(128)[None, :, None] * 0.001
    soles[..., 1] = 0.025
    soles.requires_grad_()
    contacts = torch.ones(1, 128, 4)
    losses = contact_geometry_losses(soles, contacts, contacts.bool(), 60)
    assert float(losses['l_trusted_skate_mps'].detach()) == pytest.approx(0.06)
    assert float(losses['l_stance_anchor_m'].detach()) == pytest.approx(0.032)
    assert float(losses['l_support_height_m'].detach()) == pytest.approx(0.020)
    (losses['l_stance_anchor_m'] + losses['l_support_height_m']).backward()
    assert (soles.grad[:, 0, :, 0] < 0).all() and (soles.grad[:, -1, :, 0] > 0).all()
    assert (soles.grad[..., 1] > 0).all() and torch.isfinite(soles.grad).all()


def test_root_authority_never_copies_future_height_body_or_contact():
    rep = build_representation().dual_rep.local_motion_rep
    prediction = torch.randn(2, 16, 365, requires_grad=True)
    external = torch.randn(2, 16, 3)
    controlled = apply_root_velocity_condition(prediction, external, rep)
    indices = list(rep.indices['local_root_rot_vel']) + list(rep.indices['local_root_vel'])
    torch.testing.assert_close(controlled[..., indices], external)
    other = [index for index in range(365) if index not in indices]
    torch.testing.assert_close(controlled[..., other], prediction[..., other])
    controlled.sum().backward()
    assert not prediction.grad[..., indices].any()
    assert (prediction.grad[..., other] == 1).all()
    with pytest.raises(ValueError, match='every decoded frame'):
        apply_root_velocity_condition(prediction, external[:, :8], rep)


def test_contact_requires_leg_compensation_for_a_moving_root():
    frames = 32
    root = torch.zeros(1, frames, 1, 3)
    root[..., 0] = torch.arange(frames)[None, :, None] / 60
    relative = torch.zeros(1, frames, 4, 3)
    contacts = torch.ones(1, frames, 4)
    slipping = contact_geometry_losses(root + relative, contacts, contacts.bool(), 60)
    assert float(slipping['l_trusted_skate_mps']) == pytest.approx(1)
    compensated = contact_geometry_losses(root + relative - root, contacts, contacts.bool(), 60)
    assert compensated['l_trusted_skate_mps'] == 0
    airborne = contact_geometry_losses(root + relative, torch.zeros_like(contacts), contacts.bool(), 60)
    assert airborne['l_trusted_skate_mps'] == 0
    assert float(root[0, -1, 0, 0]) > 0


@pytest.mark.parametrize('overall_indices', [False, True])
def test_conditioned_adapter_preserves_official_layers_and_token_parity(overall_indices):
    from motionbricks.data.bones_models import ConditionedBonesVQVAE
    from motionbricks.vqvae.neural_modules.vqvae import VQVAE
    rep = build_representation().dual_rep.local_motion_rep
    kwargs = dict(pose_root_mode='pose', motion_rep=rep, encoder_state_dim=0, decoder_state_dim=0,
                  decoder_target_cond_dim=0, decoder_external_cond_dim=0,
                  feature_mode=['joint_positions_and_rotations', 'pose',
                                'joint_positions_and_rotations', 'root_without_hip_height'],
                  code_dim=16, output_emb_width=16, width=16, depth=1, down_t=2, stride_t=2,
                  nb_code=16, num_heads=2, kmeans_init=False)
    base = VQVAE(**kwargs).eval()
    adapter = ConditionedBonesVQVAE(**kwargs).eval()
    adapter.load_state_dict(base.state_dict(), strict=True)
    motion, external = torch.randn(1, 32, 365), torch.randn(1, 32, 3)
    mask = torch.zeros(1, 32, dtype=torch.bool)
    mask[:, :4] = True
    with torch.inference_mode():
        original = base(motion, motion, mask, external)['recon_state']
        output = adapter(motion, motion, mask, external)
        indices = adapter.encode_into_idx(motion, fetch_overall_indices=overall_indices)
        decoded = adapter.forward_decoder(indices, motion, mask, external, use_overall_indices=overall_indices)
    torch.testing.assert_close(output['raw_recon_state'], original)
    torch.testing.assert_close(output['recon_state'], apply_root_velocity_condition(original, external, rep))
    torch.testing.assert_close(decoded['recon_state'], output['recon_state'], rtol=1e-6, atol=1e-6)


def test_support_repair_permission_cannot_silently_change_convergence():
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / 'scripts/train_bones_locomotion.py'
    spec = spec_from_file_location('bones_support_training_entry', path)
    entry = module_from_spec(spec)
    spec.loader.exec_module(entry)
    args = SimpleNamespace(stage='tokenizer', max_steps=10000, pipeline_smoke=False, convergence_pilot=False,
                           support_repair_pilot=True, initialize_from='reviewed.ckpt', resume=None)
    preflight = {'automatic_preconditions_passed': True}
    report = {'full_training_gate_passed': False}
    protocol = {'experiment_kind': 'bounded_tokenizer_support_repair'}
    assert entry.validate_training_scope(args, report, protocol, preflight) == 'bounded_tokenizer_support_repair'
    for attribute, value in (('convergence_pilot', True), ('stage', 'root'), ('max_steps', 10001), ('resume', 'old.ckpt')):
        changed = SimpleNamespace(**vars(args))
        setattr(changed, attribute, value)
        with pytest.raises(RuntimeError, match='Support repair needs'):
            entry.validate_training_scope(changed, report, protocol, preflight)


def test_aggregate_skating_keeps_frame_and_second_units_consistent():
    base = {'contact_tp_auto': 1, 'contact_fp_auto': 0, 'contact_fn_auto': 0,
            'trusted_support_samples': 0, 'trusted_support_clearance_abs_m': None,
            'trusted_support_clearance_abs_p95_m': None, 'sole_penetration_max_m': 0,
            'max_joint_error_peak_m': 0}
    windows = [{'metrics': dict(base, trusted_stance_samples=count, trusted_skate_mps=speed,
                               trusted_skate_m_per_frame=speed / 60, source_trusted_skate_mps=0.01)}
               for count, speed in ((1, 1.0), (9, 3.0))]
    summary = summarize_windows(windows)
    assert summary['trusted_skate_mps'] == pytest.approx(2.8)
    assert summary['trusted_skate_m_per_frame'] * 60 == pytest.approx(summary['trusted_skate_mps'])


def test_tokenizer_keeps_upstream_variable_keyframe_curriculum():
    from types import SimpleNamespace
    from motionbricks.data.bones_models import BonesTokenizer
    from motionbricks.vqvae.models.motion_vqvae import MotionVQVAEModel
    assert BonesTokenizer._construct_keyframe_prob is MotionVQVAEModel._construct_keyframe_prob
    dummy = SimpleNamespace(trainer=SimpleNamespace(global_step=100), args={
        'keyframe_num_warmup_steps': 100, 'pose_vqvae_max_num_keyframes': 10,
        'root_vqvae_max_num_keyframes': 5, 'pose_vqvae_no_keyframe_prob': 0.25,
        'root_vqvae_no_keyframe_prob': 0.25})
    pose_prob, _ = MotionVQVAEModel._construct_keyframe_prob(dummy)
    assert len(pose_prob) == 11 and (pose_prob > 0).all()
    assert pose_prob[0] == pytest.approx(0.25)
    assert pose_prob.sum() == pytest.approx(1.0)


def test_official_decoder_mask_blocks_unavailable_future_pose_values():
    from motionbricks.vqvae.neural_modules.vqvae import VQVAE
    rep = build_representation().dual_rep.local_motion_rep
    network = VQVAE(pose_root_mode='pose', motion_rep=rep, encoder_state_dim=0, decoder_state_dim=0,
                    decoder_target_cond_dim=0, decoder_external_cond_dim=0,
                    feature_mode=['joint_positions_and_rotations', 'pose',
                                  'joint_positions_and_rotations', 'root_without_hip_height'],
                    code_dim=16, output_emb_width=16, width=16, depth=1, down_t=2, stride_t=2,
                    nb_code=16, num_heads=2, kmeans_init=False).eval()
    motion = torch.randn(1, 32, 365)
    mask = torch.zeros(1, 32, dtype=torch.bool)
    mask[:, :4] = True
    unavailable = motion.clone()
    unavailable[:, 4:] += 100
    external = torch.randn(1, 32, 3)
    with torch.inference_mode():
        original = network(motion, target_cond=motion, has_target_cond=mask, external_cond=external)
        changed = network(motion, target_cond=unavailable, has_target_cond=mask, external_cond=external)
    torch.testing.assert_close(original['recon_state'], changed['recon_state'], rtol=0, atol=0)


@pytest.mark.parametrize('overall_indices', [False, True])
def test_official_token_inference_matches_reconstruction(overall_indices):
    from motionbricks.vqvae.neural_modules.vqvae import VQVAE
    rep = build_representation().dual_rep.local_motion_rep
    network = VQVAE(pose_root_mode='pose', motion_rep=rep, encoder_state_dim=0, decoder_state_dim=0,
                    decoder_target_cond_dim=0, decoder_external_cond_dim=0,
                    feature_mode=['joint_positions_and_rotations', 'pose',
                                  'joint_positions_and_rotations', 'root_without_hip_height'],
                    code_dim=16, output_emb_width=16, width=16, depth=1, down_t=2, stride_t=2,
                    nb_code=16, num_heads=2, kmeans_init=False).eval()
    motion = torch.randn(2, 32, 365)
    mask = torch.zeros(2, 32, dtype=torch.bool)
    mask[:, :4] = True
    target = motion * mask[..., None]
    external = torch.randn(2, 32, 3)
    with torch.inference_mode():
        reference = network(motion, target_cond=target, has_target_cond=mask, external_cond=external)['recon_state']
        indices = network.encode_into_idx(motion, fetch_overall_indices=overall_indices)
        restored = network.forward_decoder(indices, target_cond=target, has_target_cond=mask,
                                           external_cond=external, use_overall_indices=overall_indices)['recon_state']
    torch.testing.assert_close(restored, reference, rtol=1e-6, atol=1e-6)


def test_a_contract_without_a_declared_revision_is_read_as_the_shipped_geometry(tmp_path):
    """Contracts written before the revisions existed were all built with position_only_v0."""
    from motionbricks.data.bones_contract import dataset_snapshot
    from motionbricks.data.bones_seed import SOLE_PROXY_GEOMETRY_OF_SHIPPED_DATASET
    make_snapshot_fixture(tmp_path)
    contract = json.loads((tmp_path / 'skeleton_contract.json').read_text())
    assert contract['sole_proxy_geometry'] != SOLE_PROXY_GEOMETRY_OF_SHIPPED_DATASET
    del contract['sole_proxy_geometry']
    legacy = dict(contract, sole_offsets_m=skeleton_manifest()['sole_offsets_by_revision'][
        SOLE_PROXY_GEOMETRY_OF_SHIPPED_DATASET])
    (tmp_path / 'skeleton_contract.json').write_text(json.dumps(legacy))
    snapshot = dataset_snapshot(tmp_path)
    assert snapshot['skeleton_contract_sha256']
    contract['sole_proxy_geometry'] = 'not_a_real_revision'
    (tmp_path / 'skeleton_contract.json').write_text(json.dumps(contract))
    with pytest.raises(ValueError, match='unknown sole proxy geometry revision'):
        dataset_snapshot(tmp_path)


def make_snapshot_fixture(tmp_path):
    official = skeleton_manifest()
    skeleton = {
        'joint_names': [joint['name'] for joint in official['joints']],
        'parents': [joint['parent_index'] for joint in official['joints']],
        'neutral_joints_m': [joint['neutral_position'] for joint in official['joints']],
        'kimodo_revision': official['provenance']['commit'], 'foot_channels': official['foot_channels'],
        'sole_offsets_m': official['sole_offsets_m'], 'sole_mesh_sha256': official['sole_mesh_sha256'],
        # A dataset is bound to the proxy geometry revision it was converted with.
        'sole_proxy_geometry': official['sole_proxy_geometry'],
    }
    (tmp_path / 'skeleton_contract.json').write_text(json.dumps(skeleton))
    record = dict(shard='clip.npz', split='train', source_sha256='a' * 64, recording_group='walk')
    manifest = tmp_path / 'accepted.json'
    manifest.write_text(json.dumps([record]))
    (tmp_path / 'quality_report.json').write_text(json.dumps({'stats_folder': 'stats'}))
    stats = tmp_path / 'stats'
    stats.mkdir()
    np.save(stats / 'mean.npy', np.zeros(370, dtype=np.float32))
    np.save(stats / 'std.npy', np.ones(370, dtype=np.float32))
    (stats / 'contract.json').write_text(json.dumps({
        'source_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(), 'split': 'train'}))
    np.savez(tmp_path / 'clip.npz', global_features=np.zeros((129, 366), dtype=np.float32))
    return record


def test_snapshot_rejects_changed_content_even_with_identical_stats(tmp_path):
    make_snapshot_fixture(tmp_path)
    before = dataset_snapshot(tmp_path)
    require_same_snapshot(before, dataset_snapshot(tmp_path))
    np.savez(tmp_path / 'clip.npz', global_features=np.ones((129, 366), dtype=np.float32))
    after = dataset_snapshot(tmp_path)
    assert before['stats_sha256'] == after['stats_sha256']
    with pytest.raises(ValueError, match='Dataset content changed'):
        require_same_snapshot(before, after)
    with pytest.raises(ValueError, match='immutable'):
        require_same_snapshot(None, after)


def test_snapshot_rejects_stale_train_stats_and_wrong_reference(tmp_path):
    record = make_snapshot_fixture(tmp_path)
    record['split'] = 'validation'
    (tmp_path / 'accepted.json').write_text(json.dumps([record]))
    with pytest.raises(ValueError, match='train-only manifest'):
        dataset_snapshot(tmp_path)
    contract_path = tmp_path / 'skeleton_contract.json'
    contract = json.loads(contract_path.read_text())
    contract['parents'][10] = 0
    contract_path.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match='parents'):
        dataset_snapshot(tmp_path)


def test_floating_support_cannot_hide_behind_zero_penetration():
    from kimodo.exports.motion_io import complete_motion_dict
    rep = build_representation()
    local = rep.dual_rep.local_motion_rep
    stats = Stats()
    stats.register_from_tensors(torch.zeros(365), torch.ones(365))
    local.stats = stats
    frames = 128
    rotations = torch.eye(3).expand(frames, 30, 3, 3).clone()
    offsets, _ = official_sole_offsets(rep.skeleton)
    root = torch.zeros(frames, 3)
    root[:, 1] = -(rep.skeleton.neutral_joints[rep.skeleton.foot_joint_idx] + offsets)[:, 1].median()
    motion = complete_motion_dict(rotations, root, rep.skeleton, 60)
    target = local({'posed_joints': motion['posed_joints'], 'local_joint_rots': motion['local_rot_mats'],
                    'global_joint_rots': motion['global_rot_mats'], 'translation': root,
                    'foot_contacts': torch.ones(frames, 4)}, to_normalize=True)[None]
    prediction = target.clone()
    prediction[..., local.indices['global_root_y']] += 0.2
    prediction[..., local.indices['local_root_rot_vel']] += 1.0
    metrics, _, _ = reconstruction_metrics(prediction, target, local, offsets)
    assert metrics['sole_penetration_max_m'] == 0
    assert metrics['trusted_support_clearance_abs_p95_m'] > 0.15
    assert metrics['root_heading_error_deg'] > 30
    summary = summarize_windows([{'metrics': metrics}])
    assert summary['trusted_support_samples'] == 4 * frames
    assert summary['trusted_support_clearance_abs_p95_m'] == metrics['trusted_support_clearance_abs_p95_m']
