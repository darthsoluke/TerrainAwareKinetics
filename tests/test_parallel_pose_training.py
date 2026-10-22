from types import SimpleNamespace

import pytest

from motionbricks.data.parallel_pose_training import ParallelPoseMicrobatches, pose_micro_weights
from motionbricks.data.reference_exposure_budget import ExposureBudget, TARGET_EXPOSURES


def test_final_partial_batch_keeps_all_exposures_without_padding():
    # Two verified startup reference batches followed by global6144 leave1536.
    budget = ExposureBudget(2, 512, 6144)
    full_steps, remainder = divmod(TARGET_EXPOSURES - 512, 6144)
    for _ in range(full_steps):
        budget.advance(6144)
    assert budget.next_batch == remainder == 1536
    assert pose_micro_weights([remainder]) == [1.0]
    budget.advance(remainder)
    assert budget.equivalent_steps == 50000
    assert budget.actual_exposures == TARGET_EXPOSURES
    assert budget.actual_updates == 2 + full_steps + 1


def test_partial_second_micro_uses_its_actual_exposure_weight():
    assert pose_micro_weights([3072, 1024]) == [.75, .25]


@pytest.mark.parametrize('sizes', [[], [0], [64], [4096], [3072, 0], [1024, 3072], [3072]*3])
def test_invalid_reference_partition_is_rejected_before_gpu_work(sizes):
    with pytest.raises(ValueError):
        pose_micro_weights(sizes)


@pytest.mark.parametrize('override', [dict(stage='root'), dict(cache_pose_tokens=False),
    dict(coalesce_dilated_bias=False), dict(isolate_pose_teacher_rng=False)])
def test_unverified_teacher_or_stage_cannot_enter_parallel_executor(override):
    options = dict(stage='pose', cache_pose_tokens=True, coalesce_dilated_bias=True,
                   isolate_pose_teacher_rng=True)
    options.update(override)
    with pytest.raises(ValueError, match='isolated teacher RNG'):
        ParallelPoseMicrobatches([SimpleNamespace(**options)] * 2)
