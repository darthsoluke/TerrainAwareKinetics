import hashlib
import json

import numpy as np
import torch
import pytest

from test_native_mb_stage_training import _motion_and_representation
from test_native_mb_terrain_integration import small_config
from motionbricks.data.native_mb_models import build_native_model
from motionbricks.data.native_mb_online_conditions import enable_online_conditioning
from scripts import evaluate_parc_online_control as review


@pytest.mark.parametrize('relative_height', [False, True])
def test_evaluation_generates_without_future_pose_height_and_writes_world_motion(tmp_path, monkeypatch, relative_height):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        rep, batch = _motion_and_representation(frame_count=65)
        dataset = tmp_path / 'dataset'
        (dataset / 'stats').mkdir(parents=True)
        np.save(dataset / 'stats/mean.npy', rep.dual_rep.stats.mean.numpy())
        np.save(dataset / 'stats/std.npy', rep.dual_rep.stats.std.numpy())
        geometry = dataset / 'geometry.npz'
        np.savez(geometry, heightfield_y_m=np.zeros((81, 81), dtype=np.float32),
            min_point_world_xyz_m=np.array([-4., 0., -4.]), dx_m=np.array(.1),
            canonical_init_heading_quat=np.array([1., 0., 0., 0.]), canonical_root_pos_init_xz=np.zeros(2))
        digest = hashlib.sha256(geometry.read_bytes()).hexdigest()
        raw = rep.unnormalize(batch['motion'])[0].numpy()
        if relative_height:
            raw[:, 1] += 2.
            raw[:, rep.indices['ric_data'][1::3]] += 2.
        np.savez(dataset / 'motion.npz', global_features=raw,
            terrain_points=np.zeros((65, 2, 9, 9, 3), dtype=np.float32),
            terrain_valid=np.ones((65, 2, 9, 9), dtype=bool),
            canonical_init_heading_quat=np.array([1., 0., 0., 0.]), canonical_root_pos_init_xz=np.zeros(2))
        record = {'split': 'validation', 'shard': 'motion.npz', 'frames_30hz': 65,
                  'terrain_geometry_snapshot': 'geometry.npz', 'terrain_geometry_sha256': digest}
        (dataset / 'accepted.json').write_text(json.dumps([record]))
        (dataset / 'native_manifest.json').write_text(json.dumps({'artifact_sha256': {'geometry.npz': digest}}))
        codec = build_native_model('tokenizer', rep, small_config('tokenizer'))
        root = build_native_model('root', rep, enable_online_conditioning(small_config('root'))).eval()
        pose = build_native_model('pose', rep, enable_online_conditioning(small_config('pose')), tokenizer=codec).eval()
        if relative_height:
            from motionbricks.data.native_mb_vertical_frame import VERTICAL_FRAME
            root.args['vertical_frame'] = pose.args['vertical_frame'] = VERTICAL_FRAME
        pose._vqvae_model_loaded = True
        contract = {'dataset_manifest_sha256': 'test', 'online_contract': {
            'root_dependency': {'checkpoint_sha256': 'root-test'}}}
        monkeypatch.setattr(review, 'load_control_checkpoint', lambda run, device:
            (root if run == 'root' else pose, contract,
             {'final_checkpoint_sha256': 'root-test' if run == 'root' else 'pose-test'}))
        window = {'shard': 'motion.npz', 'start': 0, 'behavior': 'test'}
        monkeypatch.setattr(review, '_dataset_windows', lambda directory, maximum:
            (directory, [window], [window], [], {'dataset_manifest_sha256': 'test'}, {}, {}, [record]))
        seen = {}
        original = root.backbone_net.forward

        def root_forward(*args, **kwargs):
            seen.update(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(root.backbone_net, 'forward', root_forward)
        output = tmp_path / 'review'
        result = review.evaluate(dataset, 'root', 'pose', output, device='cpu', max_windows=1,
                                 decoder_review=True)
        assert not result['quality_approved'] and not result['root_overwrite']
        assert not seen['global_root_component_mask'][:, 4:, 1].any()
        assert not result['windows'][0]['future_pose_provided']
        assert result['frame_scoring'] == {'start': 4, 'stop_exclusive': 63}
        assert result['interval_scoring'] == {'start': 4, 'stop_exclusive': 62}
        assert result['stage_summary']['root_stage_xyz_error_m'] >= 0
        factorization = result['windows'][0]['offline_decoder_factorization']
        assert factorization['known_keyframe_indices'] == [3]
        assert not factorization['runtime_inputs_changed']
        assert len(result['offline_decoder_summary']) == 4
        for name, case in factorization['cases'].items():
            assert np.isfinite(case['root_y_mae_m'])
            assert np.isfinite(case['root_relative_fk_mpjpe_m'])
            assert np.isfinite(case['first_future_root_y_step_error_m'])
            if name == 'model_tokens_model_root':
                assert not case['future_reference_used']
                assert case['online_replay_max_abs_error'] <= 2e-5
            else:
                assert case['future_reference_used']
        assert (output / 'metrics.json').is_file()
        with np.load(output / 'online-000.npz') as data:
            assert data['roots_prediction'].shape == (64, 3)
            assert data['local_prediction'].shape == (64, 30, 3, 3)
            assert np.isfinite(data['roots_prediction']).all()
            assert data['root_stage_canonical'].shape == (64, 5)
        second = review.evaluate(dataset, 'root', 'pose', tmp_path / 'review-invalid',
                                 device='cpu', max_windows=1, inference_steps=4, terrain_mode='invalid')
        assert second['windows'][0]['valid_points'] == 0
        assert second['windows'][0]['observed_points'] == 162
        assert second['inference_steps'] == 4
    finally:
        torch.set_num_threads(previous)
