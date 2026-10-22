import copy
import json

import pytest

from scripts.run_parc_only_training import stage_command, validate_result
from scripts.train_native_mb import sha256
from motionbricks.data.native_mb_paper_protocol import PAPER_BATCH_PER_GPU, PAPER_STEPS


@pytest.mark.parametrize('stage', ['tokenizer', 'root', 'pose'])
def test_native_stage_commands_are_fresh_parc_only_with_original_schedule(stage, tmp_path):
    command = stage_command(stage, tmp_path / 'dataset', tmp_path / 'run', 1, 200,
        tokenizer_run=tmp_path / 'codec' if stage == 'pose' else None)
    assert '--parc-only' in command
    assert command[command.index('--batch-size') + 1] == str(PAPER_BATCH_PER_GPU)
    assert command[command.index('--schedule-steps') + 1] == str(PAPER_STEPS)
    assert command[command.index('--training-profile') + 1] == 'paper'
    assert command[command.index('--steps') + 1] == '200'
    assert '--warmstart-run' not in command
    assert '--resume-run' not in command
    assert ('--terrain-conditioning' in command) == (stage != 'tokenizer')
    assert ('--tokenizer-run' in command) == (stage == 'pose')


@pytest.mark.parametrize('stage,tokenizer', [('invalid', None), ('tokenizer', 'codec'),
                                          ('root', 'codec'), ('pose', None)])
def test_reject_wrong_stage_or_dependency(stage, tokenizer, tmp_path):
    with pytest.raises(ValueError):
        stage_command(stage, tmp_path, tmp_path, 1, 200, tokenizer_run=tokenizer)


def completed_run(tmp_path, stage):
    (tmp_path / 'checkpoints').mkdir()
    (tmp_path / 'checkpoints/final.ckpt').write_bytes(b'test checkpoint identity')
    (tmp_path / 'native_config.yaml').write_text('model: original\n')
    contract = dict(stage='native_' + stage, status='completed', seed=7, stop_steps=200,
        dataset_manifest_sha256='dataset-hash', batch_size=PAPER_BATCH_PER_GPU,
        original_schedule_steps=PAPER_STEPS, parc_only_sources={'train_origin_counts':
        {'parc_generated': 3, 'parc_official': 6}}, warmstart=None, resume=None,
        terrain_conditioning={'protocol': 'local-intent-observation'} if stage != 'tokenizer' else None,
        tokenizer={'checkpoint_sha256': 'codec-hash'} if stage == 'pose' else None,
        tokenizer_state_sha256='frozen-state' if stage == 'pose' else None,
        config_sha256=sha256(tmp_path / 'native_config.yaml'))
    result = dict(optimizer_updates=200, updates_this_run=200, skipped_batches=0,
        observed_input_batch_sizes=[PAPER_BATCH_PER_GPU], optimizer_class='torch.optim.adam.Adam',
        final_checkpoint_sha256=sha256(tmp_path / 'checkpoints/final.ckpt'),
        tokenizer_state_sha256=contract['tokenizer_state_sha256'], quality_approved=False)
    if stage != 'tokenizer':
        result['terrain_training_evidence'] = {'nonzero_gradient_updates':
            {'encoder': 199, 'attention': 200, 'height': 200 if stage == 'pose' else 0}}
    return contract, result


def save_run(path, contract, result):
    (path / 'run_contract.json').write_text(json.dumps(contract))
    (path / 'training_result.json').write_text(json.dumps(result))


@pytest.mark.parametrize('stage', ['tokenizer', 'root', 'pose'])
def test_actual_completed_stage_accounting(stage, tmp_path):
    contract, result = completed_run(tmp_path, stage)
    save_run(tmp_path, contract, result)
    assert validate_result(tmp_path, stage, 'dataset-hash', 200, 'codec-hash') == result


@pytest.mark.parametrize('location,key,value', [
    ('contract', 'seed', 1), ('contract', 'batch_size', 4),
    ('contract', 'stop_steps', 100), ('contract', 'original_schedule_steps', 1000),
    ('contract', 'parc_only_sources', None), ('contract', 'warmstart', {'run': 'flat'}),
    ('contract', 'resume', {'run': 'other'}), ('contract', 'config_sha256', 'wrong'),
    ('contract', 'terrain_conditioning', None), ('contract', 'status', 'running'),
    ('result', 'optimizer_updates', 199), ('result', 'updates_this_run', 0),
    ('result', 'skipped_batches', 1), ('result', 'observed_input_batch_sizes', [4]),
    ('result', 'optimizer_class', 'AdamW'), ('result', 'final_checkpoint_sha256', 'wrong'),
    ('result', 'tokenizer_state_sha256', 'modified-state'),
    ('result', 'terrain_training_evidence', None),
])
def test_reject_false_completion_or_altered_training(location, key, value, tmp_path):
    contract, result = completed_run(tmp_path, 'pose')
    {'contract': contract, 'result': result}[location][key] = copy.deepcopy(value)
    save_run(tmp_path, contract, result)
    with pytest.raises(RuntimeError):
        validate_result(tmp_path, 'pose', 'dataset-hash', 200, 'codec-hash')


def test_reject_pose_other_tokenizer(tmp_path):
    contract, result = completed_run(tmp_path, 'pose')
    save_run(tmp_path, contract, result)
    with pytest.raises(RuntimeError, match='different tokenizer'):
        validate_result(tmp_path, 'pose', 'dataset-hash', 200, 'other-codec')
