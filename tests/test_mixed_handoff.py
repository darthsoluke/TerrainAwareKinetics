import json

import pytest

from scripts.run_diverse_mixed_handoff import verify_completed_arm
from scripts.train_native_mb import sha256


@pytest.fixture
def completed_arm(tmp_path):
    state = dict(source_sha256='a'*64, dataset_manifest_sha256='b'*64,
                 steps_per_stage_per_arm=10000, sampling_policy='parc-windows-v1')
    frame = 'observed-pelvis-y-v1'
    (tmp_path/'pipeline.pt').write_bytes(b'finished export')
    identity = sha256(tmp_path/'pipeline.pt')
    (tmp_path/'pipeline.json').write_text(json.dumps(dict(
        sha256=identity, dataset_manifest_sha256=state['dataset_manifest_sha256'])))
    for name in ('tokenizer', 'root', 'pose'):
        directory = tmp_path/name
        directory.mkdir()
        checkpoint = directory/'selected.ckpt'
        checkpoint.write_bytes(name.encode())
        contract = dict(stage=name, source_bundle_sha256=state['source_sha256'],
                        dataset_manifest_sha256=state['dataset_manifest_sha256'],
                        steps=10000, vertical_frame=frame, sampling_policy=state['sampling_policy'],
                        world_size=2 if name == 'pose' else 1,
                        precision='bf16-mixed' if name == 'pose' else '32',
                        batch={'global':2048 if name == 'pose' else 256})
        (directory/'run_contract.json').write_text(json.dumps(contract))
        (directory/'training_result.json').write_text(json.dumps(dict(
            optimizer_updates=10000, checkpoint_relative=checkpoint.name,
            checkpoint_sha256=sha256(checkpoint))))
    (tmp_path/'closed_loop_review').mkdir()
    (tmp_path/'closed_loop_review/metrics.json').write_text(json.dumps(dict(
        bundle_sha256=identity, dataset_manifest_sha256=state['dataset_manifest_sha256'],
        flat_stop_gate_passed=False)))
    return tmp_path, state, frame


def test_failed_motion_quality_does_not_discard_completed_training(completed_arm):
    verify_completed_arm(*completed_arm)


@pytest.mark.parametrize('change', ['checkpoint', 'source', 'budget', 'frame'])
def test_completed_arm_cannot_be_reused_with_changed_inputs(completed_arm, change):
    run, state, frame = completed_arm
    if change == 'checkpoint':
        (run/'root/selected.ckpt').write_bytes(b'changed')
    elif change == 'source':
        state['source_sha256'] = 'c'*64
    elif change == 'budget':
        state['steps_per_stage_per_arm'] = 20000
    else:
        frame = 'observed-terrain-y-v1'
    with pytest.raises(ValueError, match='changed'):
        verify_completed_arm(run, state, frame)
