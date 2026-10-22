import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_parc_online_adaptation import (
    commands, completed_checkpoint, validate_step_budget,
)
from motionbricks.data.native_mb_paper_protocol import PAPER_STEPS, validate_paper_arguments
from scripts.train_native_mb import sha256


def test_online_workflow_reuses_verified_weights_not_new_dataset_or_codec_training(tmp_path):
    args = SimpleNamespace(dataset=tmp_path / 'dataset', output=tmp_path / 'run',
        root_baseline=tmp_path / 'root', pose_baseline=tmp_path / 'pose',
        tokenizer_run=tmp_path / 'codec', steps=1000, gpu=1, workers=2)
    result = commands(args)
    assert list(result) == ['root', 'pose', 'review']
    for stage in ('root', 'pose'):
        command = result[stage]
        assert '--online-conditioning' in command and '--parc-only' in command
        assert command[command.index('--steps') + 1] == '1000'
        assert command[command.index('--batch-size') + 1] == '256'
        schedule = int(command[command.index('--schedule-steps') + 1])
        assert schedule == PAPER_STEPS
        validate_paper_arguments(schedule, int(command[command.index('--batch-size') + 1]))
        assert '--resume-run' not in command and '--warmstart-run' not in command
    assert result['pose'][result['pose'].index('--online-root-run') + 1] == str(tmp_path / 'run/root-online')
    assert 'evaluate_parc_online_control.py' in result['review'][2]


def test_workflow_never_imports_running_baseline_or_partial_checkpoint(tmp_path):
    (tmp_path / 'checkpoints').mkdir()
    (tmp_path / 'checkpoints/final.ckpt').write_bytes(b'test')
    (tmp_path / 'training_result.json').write_text(json.dumps({
        'final_checkpoint_sha256': sha256(tmp_path / 'checkpoints/final.ckpt')}))
    contract = tmp_path / 'run_contract.json'
    contract.write_text(json.dumps({'status': 'running'}))
    assert not completed_checkpoint(tmp_path)
    contract.write_text(json.dumps({'status': 'trained_not_reviewed'}))
    assert completed_checkpoint(tmp_path)


def test_online_continuation_commands_resume_root_and_weights_only_warmstart_pose(tmp_path):
    args = SimpleNamespace(
        dataset=tmp_path / 'dataset', output=tmp_path / 'continued',
        tokenizer_run=tmp_path / 'codec', online_root_run=tmp_path / 'prior-root',
        online_pose_run=tmp_path / 'prior-pose', steps=8000, gpu=1, workers=2,
        continue_online=True,
    )
    result = commands(args)
    root, pose = result['root'], result['pose']
    assert root[root.index('--online-resume-run') + 1] == str(args.online_root_run)
    assert '--resume-run' not in root and '--online-warmstart-run' not in root
    assert pose[pose.index('--online-warmstart-run') + 1] == str(args.online_pose_run)
    assert pose[pose.index('--online-root-run') + 1] == str(args.output / 'root-online')
    assert '--online-resume-run' not in pose and '--resume-run' not in pose
    for command in (root, pose):
        assert command[command.index('--steps') + 1] == '8000'
        assert int(command[command.index('--schedule-steps') + 1]) == PAPER_STEPS
    assert list(result) == ['root', 'pose', 'review', 'review_invalid']
    for key in ('review', 'review_invalid'):
        assert result[key][result[key].index('--inference-steps') + 1] == '4'
    assert result['review_invalid'][result['review_invalid'].index('--terrain-mode') + 1] == 'invalid'
    assert result['review_invalid'][result['review_invalid'].index('--output') + 1] == str(args.output / 'review-invalid')


def test_preflight_and_continuation_have_separate_bounded_budgets():
    assert validate_step_budget(1000, continue_online=False) == 1000
    assert validate_step_budget(10000, continue_online=True) == 10000
    with pytest.raises(ValueError, match='--continue-online'):
        validate_step_budget(1001, continue_online=False)
    with pytest.raises(ValueError, match='10000'):
        validate_step_budget(10001, continue_online=True)
