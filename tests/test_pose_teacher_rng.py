from types import SimpleNamespace

import pytest
import torch

from motionbricks.data.pose_teacher_rng import isolated_pose_seed
from motionbricks.data.unified_full_training import NativeTrainingStage
from motionbricks.data.unified_pose_training import pose_teacher


def test_cpu_cannot_accidentally_enter_parallel_sampling():
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match='explicit CUDA'):
        with isolated_pose_seed(torch.zeros(2), 41):
            pytest.fail('CPU sampling entered a CUDA-only scope')
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('stage', ['root', 'codec'])
def test_non_pose_stage_rejects_rng_policy_before_touching_controller(stage):
    with pytest.raises(ValueError, match='only available for Pose'):
        NativeTrainingStage(None, stage, isolate_pose_teacher_rng=True)


def test_sparse_teacher_cannot_use_an_unvalidated_rng_policy():
    controller = SimpleNamespace(inference=SimpleNamespace(_pose_model=None))
    example = dict(tokens=torch.zeros(2, 8, 3, dtype=torch.long),
                   target=dict(local_motions=torch.zeros(2, 32, 10)))
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match='full factual history'):
        pose_teacher(controller, example, 41, isolate_cuda_rng=True,
                     history_policy='native-sparse-start-v1')
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Two CUDA devices required')
def test_target_seed_matches_native_and_never_changes_other_device():
    tokens = torch.zeros(64, 8, 3, device='cuda:0')
    original_cpu = torch.get_rng_state()
    original_cuda = torch.cuda.get_rng_state_all()
    try:
        with torch.random.fork_rng(devices=[0, 1]):
            torch.manual_seed(41)
            expected = torch.rand_like(tokens)
        for device in (0, 1):
            with isolated_pose_seed(tokens.to(device), 41):
                actual = torch.rand_like(tokens.to(device)).cpu()
            assert torch.equal(expected.cpu(), actual)
            assert torch.equal(original_cpu, torch.get_rng_state())
            assert all(torch.equal(x, y) for x, y in zip(original_cuda, torch.cuda.get_rng_state_all()))
        with pytest.raises(RuntimeError, match='injected'):
            with isolated_pose_seed(tokens, 43):
                torch.rand_like(tokens)
                raise RuntimeError('injected')
        assert all(torch.equal(x, y) for x, y in zip(original_cuda, torch.cuda.get_rng_state_all()))
    finally:
        torch.set_rng_state(original_cpu)
        torch.cuda.set_rng_state_all(original_cuda)
