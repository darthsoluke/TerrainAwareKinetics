import hashlib
import json

import pytest
import torch
import yaml

from motionbricks.data.native_mb_warmstart import validate_dataset_continuation


def setup_run(tmp_path, *, complete_state=True):
    run, source, target = (tmp_path / name for name in ('run', 'source', 'target'))
    (run / 'checkpoints').mkdir(parents=True)
    artifacts = {}
    for data in (source, target):
        (data / 'stats').mkdir(parents=True)
        for name in ('mean.npy', 'std.npy'):
            (data / 'stats' / name).write_bytes(b'fixed-normalization')
            artifacts[f'stats/{name}'] = hashlib.sha256(b'fixed-normalization').hexdigest()
        (data / 'native_manifest.json').write_text(json.dumps({'artifact_sha256': artifacts}))
    config = {
        'model': {
            'args': {'pose_vqvae_max_num_keyframes': 10, 'joint_vel_loss_coeff': 2.0},
            'pose_vqvae_network': {'width': 512},
            'optimizer': {'lr': 5e-5},
            'scheduler': {'num_training_steps': 2000000},
        },
        'trainer': {'max_steps': 2000000, 'precision': 32, 'gradient_clip_val': 0.5},
        'data': {'folder': 'datasets/source'},
        'motion_rep': {'stats': {'folder': 'source/stats'}},
        'loggers': {'max_steps': 2000000, 'out_dir': 'source/logs', 'wandb_run': 'source-run'},
        'callbacks': {'callback_dict': {
            'checkpoint_epoch_cb': {'dirpath': 'source/checkpoints'},
            'eval_visualize': {'out_dir': 'source/visualizations'},
        }},
        'out_dir': 'source/output', 'run_dir': 'source/run', 'id': 'source-id',
        'wandb_run': 'source-run', 'data_root': 'datasets', 'matmul_precision': 'high',
    }
    (run / 'native_config.yaml').write_text(yaml.safe_dump(config))
    saved = dict(global_step=50000, state_dict={'weight': torch.ones(1)})
    if complete_state:
        saved.update(optimizer_states=[{'state': {0: {'step': torch.tensor(50000)}},
                      'param_groups': [{'lr': 4.98e-5}]}],
                     lr_schedulers=[{'last_epoch': 50000}], loops={'fit_loop': {'state': {}}})
    torch.save(saved, run / 'checkpoints/final.ckpt')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (run / 'run_contract.json').write_text(json.dumps(dict(stage='native_tokenizer', status='completed',
        seed=2333, batch_size=256, dataset=str(source),
        config_sha256=digest(run / 'native_config.yaml'),
        dataset_manifest_sha256=digest(source / 'native_manifest.json'))))
    (run / 'training_result.json').write_text(json.dumps(dict(optimizer_updates=50000,
        final_checkpoint_sha256=digest(run / 'checkpoints/final.ckpt'))))
    return run, target, config


def test_continuation_preserves_real_step_and_full_state_contract(tmp_path):
    run, target, config = setup_run(tmp_path)
    result = validate_dataset_continuation(run, target, config, seed=2333, batch_size=256, stop_steps=55000)
    assert result['saved_step'] == 50000
    assert result['expected_initial_lr'] == 4.98e-5
    assert result['expected_scheduler_last_epoch'] == 50000
    assert result['expected_optimizer_state_entries'] == 1
    assert result['normalization_byte_identical']
    assert 'trainer.precision' not in result['allowed_config_differences']
    assert 'Dataset changed' in result['sampling_note']
    assert '_checkpoint_state' not in result
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('field,value', [('seed', 7), ('batch_size', 128), ('stop_steps', 50000)])
def test_continuation_rejects_silent_contract_changes(tmp_path, field, value):
    run, target, config = setup_run(tmp_path)
    args = dict(seed=2333, batch_size=256, stop_steps=55000)
    args[field] = value
    with pytest.raises(ValueError):
        validate_dataset_continuation(run, target, config, **args)


def test_weights_only_checkpoint_cannot_be_called_full_continuation(tmp_path):
    run, target, config = setup_run(tmp_path, complete_state=False)
    with pytest.raises(ValueError, match='optimizer, scheduler and loop'):
        validate_dataset_continuation(run, target, config, seed=2333, batch_size=256, stop_steps=55000)


def test_continuation_rejects_normalization_and_optimizer_changes(tmp_path):
    run, target, config = setup_run(tmp_path)
    config['model']['optimizer']['lr'] = 1e-3
    with pytest.raises(ValueError, match='model.optimizer.lr'):
        validate_dataset_continuation(run, target, config, seed=2333, batch_size=256, stop_steps=55000)
    config['model']['optimizer']['lr'] = 5e-5
    (target / 'stats/mean.npy').write_bytes(b'changed')
    with pytest.raises(ValueError, match='normalization mismatch'):
        validate_dataset_continuation(run, target, config, seed=2333, batch_size=256, stop_steps=55000)


def test_continuation_allows_dataset_stop_and_output_path_differences(tmp_path):
    run, target, config = setup_run(tmp_path)
    config['data']['folder'] = 'datasets/stairs'
    config['motion_rep']['stats']['folder'] = 'stairs/stats'
    config['trainer']['max_steps'] = 55000
    config['loggers'].update(max_steps=55000, out_dir='stairs/logs', wandb_run='stairs-run')
    config['callbacks']['callback_dict']['checkpoint_epoch_cb']['dirpath'] = 'stairs/checkpoints'
    config['callbacks']['callback_dict']['eval_visualize']['out_dir'] = 'stairs/visualizations'
    config.update(out_dir='stairs/output', run_dir='stairs/run', id='stairs-id',
                  wandb_run='stairs-run', data_root='stairs-data')
    result = validate_dataset_continuation(run, target, config, seed=2333, batch_size=256, stop_steps=55000)
    assert result['saved_step'] == 50000


@pytest.mark.parametrize(('section', 'key', 'value', 'expected_path'), [
    ('model.args', 'joint_vel_loss_coeff', 3.0, 'Warmstart model architecture'),
    ('model.optimizer', 'lr', 1e-3, 'model.optimizer.lr'),
    ('model.scheduler', 'num_training_steps', 1000000, 'model.scheduler.num_training_steps'),
    ('trainer', 'precision', 'bf16-mixed', 'trainer.precision'),
    ('trainer', 'gradient_clip_val', 1.0, 'trainer.gradient_clip_val'),
    ('', 'matmul_precision', 'highest', 'matmul_precision'),
])
def test_continuation_rejects_training_config_changes(tmp_path, section, key, value, expected_path):
    run, target, config = setup_run(tmp_path)
    node = config
    for part in section.split('.') if section else ():
        node = node[part]
    node[key] = value
    match = expected_path if section == 'model.args' else expected_path.replace('.', r'\.')
    with pytest.raises(ValueError, match=match):
        validate_dataset_continuation(run, target, config, seed=2333, batch_size=256, stop_steps=55000)
