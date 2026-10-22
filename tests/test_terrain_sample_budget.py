import pytest

from motionbricks.data.terrain_sample_budget import TerrainSampleBudget


def test_large_batch_reduces_updates_without_multiplying_sample_budget():
    budget=TerrainSampleBudget(32,50000,9000*32,896)
    assert budget.remaining_updates==1465
    assert budget.equivalent_updates_after(0)==9000
    assert budget.examples_after(1464)<budget.target_examples<=budget.examples_after(1465)
    assert budget.report()['rounding_extra_examples']==640
    assert budget.report()['final_reference_batch_equivalent_updates']==50020


def test_partitioning_completed_updates_preserves_example_clock():
    budget=TerrainSampleBudget(32,50000,9000*32,896)
    resumed=TerrainSampleBudget(32,50000,budget.examples_after(100),896)
    assert resumed.remaining_updates==budget.remaining_updates-100
    assert resumed.examples_after(resumed.remaining_updates)==budget.examples_after(budget.remaining_updates)
    assert resumed.equivalent_updates_after(7)==budget.equivalent_updates_after(107)


def test_original_batch_preserves_original_remaining_update_count():
    assert TerrainSampleBudget(32,50000,9000*32,32).remaining_updates==41000
    assert TerrainSampleBudget(32,50000,50000*32,32).remaining_updates==0
    with pytest.raises(ValueError,match='exceeds'):TerrainSampleBudget(32,50000,1600001,896)


def test_learning_rate_and_feedback_clock_follow_samples_across_resume():
    import math
    budget=TerrainSampleBudget(32,50000,9000*32,896)
    initial=2e-6+(5e-5-2e-6)*(.5+.5*math.cos(math.pi*(4000-1000)/(45000-1000)))
    assert budget.learning_rate_after(0,5000,1000)==pytest.approx(initial)
    assert budget.learning_clock_after(36,5000)==5008
    assert budget.learning_rate_after(budget.remaining_updates,5000,1000)==pytest.approx(2e-6)
    resumed=TerrainSampleBudget(32,50000,budget.examples_after(100),896)
    assert resumed.learning_rate_after(20,5000,1000)==budget.learning_rate_after(120,5000,1000)
    assert budget.learning_rate_after(700,5000,1000)<initial
