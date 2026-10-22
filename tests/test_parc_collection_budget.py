"""Admission hours cannot be inflated by repeats, base data or split leakage."""

import numpy as np
import pytest

from scripts.parc_collection_budget import (
    POLICY, FAMILIES, candidate_descriptor, diverse_selection, motion_identity, summarize, verify_training_gate,
)


def row(name, seconds=100, family='stairs', split='train', geometry=None):
    return dict(source_id=name, disposition='admitted', origin='parc_generated', whole_motion=True,
                motion_content_sha256=name, terrain_content_sha256=geometry or name,
                admitted_seconds=seconds, terrain_family=family, split=split)


def complete_rows():
    rows = []
    for family in FAMILIES:
        for split,count in [('train',12),('validation',2),('test',2)]:
            rows += [row(f'{family}_{split}_{i}',family=family,split=split) for i in range(count)]
    return rows


def test_duplicate_motion_cannot_add_time_even_in_another_scene():
    a,b = row('a'),row('b')
    b['motion_content_sha256']='a'
    result=summarize([a,b])
    assert result['credited_train_seconds']==100
    assert result['unique_admitted_clips']==1
    assert result['duplicate_source_ids']==['b']


def test_huge_single_scene_is_not_diverse_hours():
    result=summarize([row(str(i),600,geometry='same') for i in range(20)])
    assert result['admitted_seconds']==12000
    assert result['credited_train_seconds']==120
    assert not result['minimum_met']


def test_many_holdouts_do_not_count_as_training_hours():
    result=summarize([row(str(i),120,split='validation') for i in range(100)])
    assert result['credited_train_seconds']==0
    assert not result['minimum_met']


def test_exact_geometry_may_not_cross_splits():
    with pytest.raises(ValueError,match='crossed'):
        summarize([row('a',geometry='same'),row('b',geometry='same',split='test')])


@pytest.mark.parametrize('field,value',[('whole_motion',False),('origin','parc_official')])
def test_partial_or_existing_official_data_cannot_count(field,value):
    a=row('a'); a[field]=value
    with pytest.raises(ValueError,match='new generated'):
        summarize([a])


def test_rejected_rows_never_count():
    a=row('a'); a['disposition']='excluded'
    assert summarize([a])['admitted_seconds']==0


def test_gate_demands_target_and_all_families_in_all_splits():
    coverage=summarize(complete_rows())
    assert coverage['minimum_met'] and coverage['target_met']
    assert coverage['credited_train_seconds']==7200
    verify_training_gate({'campaign_policy':POLICY,'campaign_coverage':coverage})
    with pytest.raises(ValueError,match='versioned'):
        verify_training_gate({'new_admitted':10000})
    with pytest.raises(ValueError,match='not been met'):
        verify_training_gate({'campaign_policy':POLICY,'campaign_coverage':summarize(complete_rows()[:-1])})


def test_farthest_point_selection_uses_pose_diversity_after_quality_filter():
    candidates=[dict(name=str(i),sha256=str(i),total_loss=i,vector=np.array([v,v]))
                for i,v in enumerate([0.,.001,.4,.2])]
    chosen=diverse_selection(candidates,3,descriptor=lambda r:r['vector'])
    assert [c['name'] for c in chosen]==['0','2','3']
    assert len(diverse_selection(candidates,16,descriptor=lambda r:r['vector']))==3


def test_duplicate_candidate_hash_cannot_fill_quota():
    a=dict(name='a',sha256='same',total_loss=1)
    b=dict(name='b',sha256='same',total_loss=2)
    assert diverse_selection([a,b],16,descriptor=lambda r:np.zeros(3))==[a]


def test_admitted_duration_comes_from_frames_and_same_motion_ignores_translation(tmp_path):
    root=np.zeros((91,3)); root[:,1]=1
    local=np.tile(np.eye(3),(91,30,1,1))
    first=tmp_path/'first.npz'; second=tmp_path/'second.npz'
    np.savez(first,root_positions=root,local_rot_mats=local,source_fps=30)
    root += [7,0,-3]
    np.savez(second,root_positions=root,local_rot_mats=local,source_fps=30)
    identity,seconds=motion_identity(first)
    assert seconds==3
    assert motion_identity(second)==(identity,seconds)
    np.savez(second,root_positions=root,local_rot_mats=local,source_fps=60)
    with pytest.raises(ValueError,match='timing'):
        motion_identity(second)


def test_next_wave_is_balanced_preflighted_and_uses_fresh_immutable_groups(tmp_path):
    from scripts.run_parc_duration_campaign import build_wave
    indices={family:0 for family in FAMILIES}
    plan=build_wave(tmp_path/'wave',indices,summarize([]))
    assert len(plan['jobs'])==24
    assert plan['retained_per_layout']==16
    assert len({job['terrain_sha256'] for job in plan['jobs']})==24
    for family in FAMILIES:
        jobs=[job for job in plan['jobs'] if job['family']==family]
        assert len(jobs)==4
        assert {job['split'] for job in jobs}=={'train','validation','test'}
        assert all(job['route_length_m']>0 for job in jobs)
        assert all(15<=job['max_seconds']<=90 for job in jobs)
    assert plan['physics_required'] and not plan['training_approved']


def test_runtime_failure_cannot_masquerade_as_rejected_motion():
    from scripts.finish_parc_diverse_dataset import validate_retarget_execution
    with pytest.raises(RuntimeError,match='not a motion quality rejection'):
        validate_retarget_execution({'items':[{'behavior':'a','execution_status':'failed',
            'error':'missing Git revision'}]},['a'])
    with pytest.raises(RuntimeError):
        validate_retarget_execution({'items':[]},['a'])
    validate_retarget_execution({'items':[{'behavior':'a','execution_status':'processed',
        'screen_status':'fail'}]},['a'])


def test_diversity_descriptor_preserves_speed_changes_not_only_phase_pose(tmp_path):
    descriptors=[]
    for name,speed in [('slow',.5),('fast',2.)]:
        folder=tmp_path/name; folder.mkdir()
        root=np.zeros((91,3)); root[:,0]=np.arange(91)/30*speed
        body=np.repeat(root[:,None,:],15,axis=1)
        body[:,:,2]=np.linspace(.1,1.8,15)
        np.savez(folder/'fk.npz',root_pos=root,body_world_pos=body)
        descriptors.append(candidate_descriptor({'motion':str(folder/'generated.pkl')}))
    assert np.sqrt(np.mean((descriptors[0]-descriptors[1])**2))>.02
