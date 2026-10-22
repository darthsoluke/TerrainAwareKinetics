import json

import numpy as np
import pytest

from scripts.review_native_mb_stair_live_contact import (
    endpoint_report, published_phase_states, scene_clearance, scene_sole_report, summarize_ue_probe,
)


def surface(height):
    return dict(top_polygon_ue_cm=[[-10, -10, height], [10, -10, height],
                                   [10, 10, height], [-10, 10, height]])


def test_scene_clearance_uses_solid_top_not_nearest_surface_or_foot_height():
    points = np.array([[0, 0, 25], [0, 0, 15], [20, 0, 30]])
    gap = scene_clearance(points, [surface(0), surface(20)])
    np.testing.assert_array_equal(gap[:2], [5, -5])
    assert np.isnan(gap[2])


def test_scene_clearance_rejects_nonhorizontal_and_nonfinite_geometry():
    value = surface(0)
    value['top_polygon_ue_cm'][0][2] = 1
    with pytest.raises(ValueError):
        scene_clearance(np.zeros((1, 3)), [value])
    with pytest.raises(ValueError):
        scene_clearance(np.zeros((1, 3)), [surface(float('nan'))])


def test_audit_summary_uses_same_surface_and_preserves_unknown(tmp_path):
    inputs = dict(source_id='test', frames=[dict(kind='generated', frame_index=i,
        sole_left_m=[[x, .1, 0]], sole_right_m=[[x, .1, 0]]) for i, x in enumerate((0, 1))])
    audit = dict(source_id='test', course='up', scene_top_surfaces=[surface(5)],
        frames=[dict(kind='generated', frame_index=i, target_sole_left_cm=[[0, y, 11]],
                     target_sole_right_cm=[[0, y, 11]]) for i, y in enumerate((0, -100))])
    input_path, audit_path, output = [tmp_path / name for name in ('input.json', 'audit.json', 'summary.json')]
    input_path.write_text(json.dumps(inputs))
    audit_path.write_text(json.dumps(audit))
    summary = summarize_ue_probe(input_path, audit_path, output)['generated']
    assert summary['source']['bilateral_clearance_cm']['mean'] == 5
    assert summary['manny']['bilateral_clearance_cm']['mean'] == 6
    assert summary['source']['both_feet_above_5cm_samples'] == 0
    assert summary['manny']['both_feet_above_5cm_samples'] == 1
    assert summary['source']['unknown_samples'] == summary['manny']['unknown_samples'] == 1
    report = json.loads(output.read_text())
    assert report['quality_approved'] is False
    assert report['frames'][1]['source_bilateral_clearance_cm'] is None


def test_audit_rejects_misaligned_source_frames(tmp_path):
    source = tmp_path / 'input.json'
    audit = tmp_path / 'audit.json'
    source.write_text(json.dumps(dict(source_id='test', frames=[dict(kind='authored', frame_index=3)])))
    audit.write_text(json.dumps(dict(source_id='test', frames=[dict(kind='generated', frame_index=3)])))
    with pytest.raises(ValueError, match='alignment mismatch'):
        summarize_ue_probe(source, audit, tmp_path / 'output.json')


def test_native_scene_counts_do_not_turn_unknown_into_grounded():
    feet = np.array([[[0., .1, 0.]], [[1., .1, 0.]], [[0., -.04, 0.]]])
    report = scene_sole_report(dict(left=feet, right=feet), [surface(0)])
    assert report['frames'] == 3
    assert report['both_feet_known_frames'] == 2
    assert report['bilateral_clearance_above_5cm_frames'] == 1
    assert report['any_foot_below_top_by_3cm_frames'] == 1


def test_endpoint_ledger_scores_end_of_generated_phase_not_request_end():
    features = np.zeros((24, 366))
    joints = np.zeros((24, 30, 3))
    authored = np.zeros((5, 30, 3))
    authored[4, :, 0] = 10.
    features[15, :3] = joints[15] = [2., 1., 0.]
    plans = [dict(plans=[dict(state='advance', goal_index=0, goal_root_m=[2., 1., 0.])],
                  remaining_frames=8), dict(plans=[], remaining_frames=0),
             dict(plans=[dict(state='advance', goal_index=0, goal_root_m=[2., 1., 0.])],
                  remaining_frames=8)]
    report = endpoint_report(features, joints, authored, plans, [24, 25, 28, 29])
    assert len(report['rows']) == 1  # Last phase is not yet fully published.
    assert report['rows'][0]['endpoint_frame'] == 15
    assert report['rows'][0]['root_error_m'] == 0.
    assert report['rows'][0]['max_foot_joint_error_m'] == 0.


def test_endpoint_ledger_rejects_ambiguous_phase_boundaries():
    with pytest.raises(ValueError, match='at most one'):
        endpoint_report([], [], [], [dict(plans=[{}, {}])], [])


def test_phase_boundary_inside_request_keeps_old_phase_for_its_four_frames():
    plans = [dict(plans=[dict(frames=16, state='advance')], remaining_frames=4),
             dict(plans=[dict(frames=16, state='hold')], remaining_frames=8)]
    states = published_phase_states(plans, 16)
    assert states.tolist() == ['advance'] * 12 + ['hold'] * 4


def test_phase_ledger_rejects_gaps():
    with pytest.raises(ValueError, match='gaps or rewrites'):
        published_phase_states([dict(plans=[dict(frames=16, state='advance')], remaining_frames=0)], 8)


def test_phase_ledger_permits_discarding_unplayed_hold_tail_only():
    plans = [dict(plans=[dict(frames=20, state='hold')], remaining_frames=8),
             dict(plans=[dict(frames=12, state='advance')], remaining_frames=0)]
    assert published_phase_states(plans, 16).tolist() == ['hold'] * 8 + ['advance'] * 8


def test_hold_stage_scores_exclude_context_and_measure_world_error():
    import torch
    from scripts.review_native_mb_stair_hold import scores
    source = torch.zeros(1, 8, 30, 3)
    output = source.clone()
    output[:, :4] = 1000
    output[:, 4:, :, 1] = .07
    result = scores(dict(posed_joints=source), dict(posed_joints=output), [24, 25, 28, 29])
    assert result['root_height_bias_m'] == pytest.approx(.07)
    assert result['foot_joint_error_m'] == pytest.approx(.07)
