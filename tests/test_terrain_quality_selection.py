from copy import deepcopy

import pytest

from motionbricks.data.terrain_quality_selection import pilot_decision


def report(scale=1.,progress=.7):
    flat=[];walls=[]
    for i in range(4):
        metrics={'drift_max_m':.2*scale,'settled_speed_p95_mps':.2*scale,
            'foot_penetration_p99_m':.04*scale,'median_higher_foot_gap_m':.08*scale,
            'unknown_body_fraction':0,'audited_vertices_per_frame':18056}
        if i%2:metrics['start_progress_ratio']=progress
        flat.append({'metrics':metrics})
        walls.append({'metrics':{'body_penetration_max_m':.2*scale,'final_speed_mps':.3*scale,
            'unknown_body_fraction':0,'audited_vertices_per_frame':18056}})
    return {'flat':flat,'walls':walls,'short':[{'fk_mpjpe_m':.15}]*8}


def test_frozen_character_cannot_win_by_perfect_collision_numbers():
    result=pilot_decision(report(),report(.8),report(.01,progress=0))
    assert not result['eligible_for_long_training']
    assert not result['criteria']['learned_start_instead_of_freezing']


def test_improvement_cannot_erase_a_large_reconstruction_regression():
    candidate=report(.4);candidate['short']=[{'fk_mpjpe_m':.4}]*8
    assert not pilot_decision(report(),report(.8),candidate)['eligible_for_long_training']


def test_sampled_surface_or_unassessed_space_cannot_pass_as_full_audit():
    candidate=report(.4)
    candidate['walls'][0]['metrics']['audited_vertices_per_frame']=200
    assert not pilot_decision(report(),report(.8),candidate)['eligible_for_long_training']
    candidate=report(.4);candidate['flat'][0]['metrics']['unknown_body_fraction']=.01
    assert not pilot_decision(report(),report(.8),candidate)['eligible_for_long_training']


def test_pilot_success_never_means_deployment_approval():
    result=pilot_decision(report(),report(.8),report(.4))
    assert result['eligible_for_long_training'] and not result['quality_approved']
    incomplete=deepcopy(report());incomplete['flat'].pop()
    with pytest.raises(ValueError):pilot_decision(report(),report(),incomplete)
