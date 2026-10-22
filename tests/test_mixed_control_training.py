import pytest
import torch

from motionbricks.data.mixed_control_training import sampling_weights, validate_source_policy


def test_native_evaluation_none_return_is_read_from_the_official_loss_dictionary():
    from motionbricks.data.mixed_control_training import native_validation_loss
    class Model:
        def training_step(self, batch, index, use_outside_training=False, meta_info=None):
            if use_outside_training:
                meta_info['losses'] = {'loss': torch.tensor(2.5)}
                return None
            return torch.tensor(3.)
    assert native_validation_loss(Model(), 'root', {}, 0) == 2.5
    assert native_validation_loss(Model(), 'pose', {}, 0) == 2.5
    assert native_validation_loss(Model(), 'tokenizer', {}, 0) == 3.


def test_mixture_retains_half_terrain_and_balances_real_control_categories():
    rows = [{'origin': 'parc_official', 'split': 'train'}] * 3
    rows += [{'origin': 'bones_seed', 'split': 'train', 'control_category': name}
             for name in ['idle', 'idle', 'start', 'stop']]
    weights = sampling_weights(rows)
    torch.testing.assert_close(weights.sum(), torch.tensor(1., dtype=torch.float64))
    assert weights[:3].sum() == .5
    assert weights[3:5].sum() == weights[5] == weights[6]
    with pytest.raises(ValueError, match='held-out'):
        sampling_weights([dict(row, split='validation') for row in rows])


def test_unknown_source_and_unbound_mixed_policy_are_rejected():
    with pytest.raises(ValueError, match='source policy'):
        validate_source_policy({'version': 'pretend-parc-only'})
    with pytest.raises(ValueError, match='identity'):
        validate_source_policy({'version': 'parc-bones-soma-v1',
            'allowed_origins': ['parc_official', 'parc_generated', 'bones_seed']})


def test_long_parc_clips_receive_equal_probability_per_valid_window():
    rows = [dict(origin='parc_official', split='train', frames_30hz=65),
            dict(origin='parc_generated', split='train', frames_30hz=164)]
    rows += [dict(origin='bones_seed', split='train', control_category=name)
             for name in ('idle', 'idle', 'start', 'stop')]
    weights = sampling_weights(rows, policy='parc-windows-v1')
    assert weights[1] / weights[0] == pytest.approx(100.)
    assert weights[:2].sum() == pytest.approx(.5)
    assert weights[2:].sum() == pytest.approx(.5)
    assert weights[2:4].sum() == weights[4] == weights[5]
    with pytest.raises(ValueError, match='frame count'):
        sampling_weights([dict(rows[0], frames_30hz=64), *rows[1:]], policy='parc-windows-v1')
    with pytest.raises(ValueError, match='held-out'):
        sampling_weights([dict(rows[0], split='validation'), *rows[1:]], policy='parc-windows-v1')


def test_validation_partial_batch_is_not_overweighted_and_categories_remain_visible():
    from motionbricks.data.mixed_control_training import summarize_validation
    rows = [dict(domain='parc', group='parc/stairs', samples=16, loss=1.),
            dict(domain='parc', group='parc/stairs', samples=1, loss=18.),
            dict(domain='bones', group='bones/idle', samples=20, loss=4.),
            dict(domain='bones', group='bones/stop', samples=2, loss=8.)]
    result = summarize_validation(rows)
    assert result['parc_mean_loss'] == 2.
    assert result['bones_category_mean_loss'] == 6.
    assert result['selection_loss'] == 4.
    assert result['samples'] == 39
    assert result['groups']['bones/stop']['mean_loss'] == 8.


def test_validation_cannot_select_a_checkpoint_without_both_domains():
    from motionbricks.data.mixed_control_training import summarize_validation
    with pytest.raises(ValueError, match='both PARC and BONES'):
        summarize_validation([dict(domain='bones', group='bones/idle', samples=1, loss=1.)])
