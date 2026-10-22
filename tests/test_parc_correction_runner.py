"""Pipeline regressions; these do not certify generated motion quality."""

from pathlib import Path
import sys

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
try:
    import run_parc_official_correction as runner
finally:
    sys.path.pop(0)


def row(rank, score, eligible=True):
    return {'rank': rank, 'candidate_id': rank + 5, 'eligible': eligible,
            'scores': {'total_loss': score}}


def test_retry_preserves_prior_candidate_and_its_provenance():
    pool = []
    runner.accumulate_candidates(pool, ['first'], ['terrain1'], [row(0, 3)], 11, 2)
    runner.accumulate_candidates(pool, ['second'], ['terrain2'], [row(0, 2)], 12, 2)
    assert [x['frames'] for x in pool] == ['second', 'first']
    assert [x['terrain'] for x in pool] == ['terrain2', 'terrain1']
    assert [x['row']['generation_seed'] for x in pool] == [12, 11]


def test_pool_uses_sorted_rank_and_excludes_failed_candidates():
    pool = runner.accumulate_candidates([], ['a', 'b', 'c'], ['x', 'y', 'z'],
        [row(0, .01, False), row(1, 4), row(2, 2)], 15, 1)
    assert len(pool) == 1
    assert pool[0]['frames'] == 'c' and pool[0]['row']['candidate_id'] == 7


def test_incomplete_or_rejected_case_cannot_report_finished():
    def complete(name):
        return {'case': name, 'status': 'kinematic_correction_complete_physics_pending',
                'stages': dict.fromkeys(('raw', 'optimized', 'final'), {})}

    cases = [complete('stairs_up')]
    assert runner.completed_run_status(cases, ['stairs_up'], 2).startswith('partial_')
    cases.append(complete('stairs_up_variant_1'))
    assert runner.completed_run_status(cases, ['stairs_up'], 2).startswith('kinematic_stage_finished')
    cases[1]['stages'].pop('final')
    assert runner.completed_run_status(cases, ['stairs_up'], 2).startswith('partial_')
    assert runner.completed_run_status([], [], 2).startswith('partial_')


def test_candidate_completion_cannot_be_overridden_by_low_scores():
    info = {'losses': torch.tensor([0., 1.]), 'contact_losses': torch.zeros(2),
            'pen_losses': torch.zeros(2), 'candidate_ids': torch.tensor([5, 3]),
            'completed': torch.tensor([False, True]), 'frame_counts': torch.tensor([300, 100])}
    config = {'max_contact_loss': 3., 'max_pen_loss': 8., 'max_total_loss': 30.}
    rows = runner.candidate_rows(info, config)
    assert not rows[0]['eligible']
    assert rows[1]['eligible'] and rows[1]['candidate_id'] == 3
