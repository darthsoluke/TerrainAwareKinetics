from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motionbricks.data.terrain_learning_data import learning_sampling_weights,sampling_manifest
from motionbricks.data.terrain_traversal_metrics import traversal_window,traversal_metrics,traversal_checks,traversal_pilot_checks


def staircase():
    z=np.linspace(0,6,300)
    return np.stack((np.zeros_like(z),1+np.clip(z-1,0,3)*.4,z),-1)


def assess(prediction,reference):
    return {**traversal_metrics(prediction,reference),'audited_vertices_per_frame':18056,
            'unknown_body_fraction':0.,'body_penetration_p99_m':0.}


def test_real_stair_traversal_passes_without_a_stopping_requirement():
    reference=staircase()
    metrics=assess(reference,reference)
    assert all(traversal_checks(metrics).values())
    assert metrics['predicted_path_length_m']>6


@pytest.mark.parametrize('failure',['freeze','bypass','wrong_height','teleport'])
def test_collision_free_stop_detour_and_teleport_cannot_pass_as_traversal(failure):
    reference=staircase();prediction=reference.copy()
    if failure=='freeze':prediction[:]=reference[0]
    elif failure=='bypass':prediction[20:-20,0]+=2
    elif failure=='wrong_height':prediction[:,1]=1
    else:prediction[:-1]=reference[0]
    checks=traversal_checks(assess(prediction,reference))
    assert not checks['completes_ordered_route']


def test_complete_window_keeps_every_real_command_endpoint_and_late_terrain():
    root=np.zeros((700,3));root[:,2]=np.arange(700)/30
    root[500:,1]=np.linspace(0,2,200)
    start,blocks,_=traversal_window(root,10)
    assert start>0 and blocks==37
    for block in range(blocks):assert len(root[start+block*8:start+block*8+33])==33
    with pytest.raises(ValueError):traversal_window(root[:30])


def test_leaving_observed_scene_is_a_failed_partial_rollout_without_padding():
    reference=staircase();prefix=reference[:80].copy()
    metrics=assess(prefix,reference)
    assert metrics['root_path_rmse_m']==0  # Only emitted frames can be measured.
    assert metrics['generated_frames']==80 and metrics['requested_frames']==300
    assert metrics['root_endpoint_error_m']>4
    assert metrics['ordered_route_fraction']<.5
    assert not traversal_checks(metrics)['complete_rollout']


def test_default_sampling_uses_real_data_even_when_artificial_wall_files_exist():
    records=[{'origin':'parc_official','terrain_family':'stairs','frames_30hz':100},
             {'origin':'parc_generated','terrain_family':'mixed','frames_30hz':600},
             {'origin':'bones_seed','control_category':'idle','terrain_family':'flat'},
             {'origin':'bones_seed','control_category':'walk','terrain_family':'flat'},
             {'origin':'weak','terrain_family':'wall'}]
    data=SimpleNamespace(records=records,base=[None]*4)
    weights=learning_sampling_weights(data)
    assert weights[-1]==0
    torch.testing.assert_close(weights[:2].sum(),torch.tensor(.5,dtype=torch.double))
    torch.testing.assert_close(weights[2:4].sum(),torch.tensor(.5,dtype=torch.double))
    assert sampling_manifest(data,weights)['weak:wall']['probability']==0
    assert sampling_manifest(data,weights)['bones:idle']['probability']==.25


def report(error=1.,fraction=.25):
    rows=[{'split':split,'source_id':split+'-stairs','start_frame':10,'emitted_frames':296,
        'metrics':{'root_path_rmse_m':error,'ordered_route_fraction':fraction}} for split in ('train','validation')]
    return {'traversal':rows,'short_mean_mpjpe_m':.2,
        'flat':[{'metrics':{'drift_max_m':.1,'start_progress_ratio':.6}}]*4,
        'traversal_coverage':{split:{'missing_families':[]} for split in ('train','validation')}}


def test_artificial_wall_scores_cannot_drive_long_training_decision():
    initial=report();candidate=report()
    candidate['walls']=[{'passed':True}]*20
    assert not all(traversal_pilot_checks(initial,candidate).values())
    candidate=report(.8,.4);candidate['walls']=[{'passed':False}]*20
    assert all(traversal_pilot_checks(initial,candidate).values())


def test_fitting_train_examples_does_not_hide_validation_failure_or_dropped_cases():
    initial=report();candidate=report(.8,.4)
    candidate['traversal'][1]['metrics']['root_path_rmse_m']=2
    assert not all(traversal_pilot_checks(initial,candidate).values())
    candidate=report(.8,.4);candidate['traversal'].pop()
    assert not all(traversal_pilot_checks(initial,candidate).values())
    candidate=deepcopy(report(.8,.4));candidate['traversal_coverage']['train']['missing_families']=['climbing']
    assert not all(traversal_pilot_checks(initial,candidate).values())
