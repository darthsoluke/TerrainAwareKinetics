import math

import pytest

from motionbricks.data.reference_exposure_budget import ExposureBudget, TARGET_EXPOSURES, validate_completion_counts


def test_continuation_keeps_old_adam_counts_and_total_samples():
    budget = ExposureBudget(12000, 12000 * 256, 1024)
    while budget.next_batch:
        budget.advance(budget.next_batch)
    assert budget.actual_updates == 21500
    assert budget.equivalent_steps == 50000
    assert budget.actual_exposures == TARGET_EXPOSURES
    assert budget.learning_rate() == 5e-7
    validate_completion_counts(dict(steps=50000, batch=1024, exposure_budget=budget.contract()),
                               budget.actual_updates, budget.actual_exposures)


def test_partial_final_batch_does_not_overshoot():
    budget = ExposureBudget(49995, 49995 * 256, 1024)
    budget.advance(1024)
    assert budget.next_batch == 256
    budget.advance(256)
    assert budget.actual_updates == 49997
    with pytest.raises(ValueError):
        budget.advance(0)


@pytest.mark.parametrize('updates,exposures', [(12001,12000*256), (12000,12004*256),
                                             (12001,12001*256), (12001,12004*256+1)])
def test_rejects_ambiguous_or_inflated_progress(updates, exposures):
    with pytest.raises(ValueError):
        ExposureBudget(12000,12000*256,1024,updates,exposures)


def test_learning_rate_tracks_exposures():
    budget = ExposureBudget(12000,12000*256,1024)
    expected = 5e-7 + (5e-5-5e-7)*(1+math.cos(math.pi*12000/50000))/2
    assert budget.learning_rate() == expected
    budget.advance(1024)
    assert budget.equivalent_steps == 12004


def test_legacy_accounting_stays_strict():
    validate_completion_counts(dict(steps=50000,batch=256),50000,12800000)
    with pytest.raises(ValueError):
        validate_completion_counts(dict(steps=50000,batch=256),12500,12800000)


def test_no_claim_of_identical_optimizer_trajectory():
    budget = ExposureBudget(0,0,1024)
    policy = budget.contract()
    policy['trajectory_equivalent'] = True
    with pytest.raises(ValueError):
        validate_completion_counts(dict(steps=50000,batch=1024,exposure_budget=policy),12500,12800000)


def test_grouped_sampler_preserves_reference_batches_and_each_ema_quota():
    from collections import Counter
    from types import SimpleNamespace
    import numpy as np
    from motionbricks.data.full_terrain_cache import BalancedPrimitiveBatches
    from motionbricks.data.reference_exposure_budget import ReferenceExposureBatches
    domains=('pfnn','parc','bones')
    dataset=SimpleNamespace(records=[{'domain':domains[(i//2)%3]} for i in range(120)],
                            ids=np.arange(0,120,2))
    original=BalancedPrimitiveBatches(dataset,256,33,seed=2333,start_step=28)
    old=list(original)
    grouped=list(ReferenceExposureBatches(original,dataset,1024,28,33))
    assert [len(batch) for batch in grouped]==[1024,256]
    flat=[index for batch in grouped for index in batch]
    for i,batch in enumerate(old):
        assert sorted(flat[i*256:(i+1)*256])==sorted(batch)
    for first in range(0,len(flat),64):
        assert Counter(dataset.records[dataset.ids[i]]['domain'] for i in flat[first:first+64])=={
            'pfnn':32,'parc':16,'bones':16}
