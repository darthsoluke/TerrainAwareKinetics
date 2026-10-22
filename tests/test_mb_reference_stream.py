"""Future-only append, fixed registration and coverage of continuous motion memory."""

from dataclasses import replace
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.mb_sonic.mb_reference import prepare_reference_history_b1, reference_history_b1
from motionbricks.mb_sonic.mb_reference import MBProposal,decode_soma_reference
from motionbricks.mb_sonic.reference import ExecutedHistory, NativeSomaHistory
from motionbricks.mb_sonic.reference_stream import (ContinuousReferenceStream,continuation_context,
    source_join,anchor_integrated_forecast,canonical_native_clock)
from motionbricks.mb_sonic.retarget import CalibratedSomaG1Adapter
from motionbricks.mb_sonic.runtime import FrozenMBProvider, MBGenerationSnapshot
from motionbricks.mb_sonic.trajectory_reference import TrajectoryReferenceSolver
from motionbricks.mb_sonic.protomotions_policy import reference_window
from scripts.fixed_mb_source_provider import FixedNativeForecast
from scripts.isolate_mb_continuous_reference import load_fixed_reference


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT/'out/mb_sonic_e0_e3_20261018/continuous_reference_isolation_v1'


@pytest.fixture(scope='module')
def assets():
    import torch

    torch.set_num_threads(2)
    config = json.loads((ROOT/'configs/mb_protomotions_wasd_v32_reference_only.json').read_text())
    adapter = CalibratedSomaG1Adapter.load(config['calibration_path'],robot_model_path=config['robot_model'],
        representation=build_native_representation())
    features = np.load(CASE/'scene/source_scene.npz')['initial_history'][0]
    history = NativeSomaHistory((np.arange(4)-3)/30+.1,features,'continuous_flat64',dict(kind='source_seed'))
    initial = load_fixed_reference(CASE)[0].qpos[0]
    return adapter,history,initial


@pytest.mark.parametrize('prefetch', [False,True])
def test_append_preserves_existing_motion_and_uses_forecast_context(assets,prefetch):
    adapter,seed,initial = assets
    p = FrozenMBProvider(FixedNativeForecast(CASE/'generation/proposal.npz'),adapter,
        conditioning_mode='B1',estimate_actual_diagnostics=False,robot_model_sha256='robot',
        scene_manifest_sha256='scene',rig_id='g1',reference_history_builder=None,
        consumer_horizon_s=.76,required_future_s=.16,maximum_adoption_age_s=.1,planning_lead_s=.02,
        constraint_solver=TrajectoryReferenceSolver(adapter.kinematics,iterations=1,pose_task_frame='root'),
        reference_stream_policy='continuous_native_AR_v3',native_blocks=1)
    observer = prepare_reference_history_b1(adapter.rep,seed.features)
    p._history_observer = observer
    p.reference_history_builder = partial(reference_history_b1,observer=observer)
    stream = ContinuousReferenceStream(p)

    def snapshot(time,pose,parent=None):
        actual = ExecutedHistory(seed.scene_id,samples=[dict(timestamp_s=time,qpos=pose,qvel=np.zeros(35))])
        return MBGenerationSnapshot(time,time,round(time*1000),1,pose[:3].copy(),-np.pi/2,
            seed,actual,1,{},previous_reference_packet=parent)

    command = dict(session_id='unit',episode_id=1,command_id=1,command_seq=1,
        velocity_xy_mps=[0.,-1.2],yaw_rate_radps=0.)
    first = stream.propose(snapshot(.1,initial),command=command,intent_revision=1,requested_at_ns=1,current_sim_time=lambda:.1)
    old_state = stream.states[first.packet.plan_id]
    time = .46
    pose = old_state.reference.qpos[np.searchsorted(old_state.reference.timestamps_s,time-1e-9)].copy()
    before = pose.copy()
    second = stream.propose(snapshot(time,pose,first.packet),command=command,intent_revision=1,
        requested_at_ns=2,current_sim_time=lambda:time)
    assert second.packet.valid_until > first.packet.valid_until
    assert second.conditioning_history.timestamps_s[-1] > time
    assert second.conditioning_history.provenance['context'].startswith('continuous_native_forecast')
    a = reference_window(first.adapted_reference,.54,np.array([0,.02,.04,.08,.16]))
    b = reference_window(second.adapted_reference,.54,np.array([0,.02,.04,.08,.16]))
    np.testing.assert_array_equal(a.qpos,b.qpos)
    np.testing.assert_allclose(a.qvel,b.qvel,atol=1e-10,rtol=0)
    np.testing.assert_array_equal(pose,before)
    assert second.adapted_reference.adaptation['continuous_buffer_audit']['status']=='PASS'
    if prefetch:
        for request,time in enumerate((.50,.54),start=10):
            state=stream.states[second.packet.plan_id]
            pose=state.reference.qpos[np.searchsorted(state.reference.timestamps_s,time-1e-9)].copy()
            second=stream.propose(snapshot(time,pose,second.packet),command=command,intent_revision=1,
                requested_at_ns=request,current_sim_time=lambda:time)
        assert second.packet.valid_until-.6>1.2
    accepted = stream.states[second.packet.plan_id]
    prior_features = accepted.source.raw_features.copy()
    prior_poses = accepted.reference.qpos.copy()
    time = .6
    pose = accepted.reference.qpos[np.searchsorted(accepted.reference.timestamps_s,time-1e-9)].copy()
    changed = dict(command,command_id=2,command_seq=2,yaw_rate_radps=.2)
    third = stream.propose(snapshot(time,pose,second.packet),command=changed,intent_revision=2,
        requested_at_ns=3,current_sim_time=lambda:time)
    prefix = third.adapted_reference.timestamps_s<=time+.28+1e-8
    requested = third.adapted_reference.timestamps_s[prefix]
    index = np.searchsorted(accepted.reference.timestamps_s,requested-1e-9)
    np.testing.assert_allclose(third.adapted_reference.qpos[prefix],prior_poses[index],atol=1e-8,rtol=0)
    np.testing.assert_array_equal(accepted.source.raw_features,prior_features)
    np.testing.assert_array_equal(accepted.reference.qpos,prior_poses)
    assert third.conditioning_history.timestamps_s[-1]>=time+.28-1e-8
    assert not third.adapted_reference.adaptation['stream_append']
    for age in np.arange(.02,.101,.02):
        times=time+age+np.array([0,.02,.04,.08,.16])
        old_indices=np.searchsorted(accepted.reference.timestamps_s,times-1e-9)
        new_indices=np.searchsorted(third.adapted_reference.timestamps_s,times-1e-9)
        np.testing.assert_allclose(third.adapted_reference.qpos[new_indices],
            accepted.reference.qpos[old_indices],atol=1e-8,rtol=0)
        np.testing.assert_allclose(third.adapted_reference.body_quat_w[new_indices],
            accepted.reference.body_quat_w[old_indices],atol=1e-8,rtol=0)
        np.testing.assert_allclose(third.adapted_reference.qvel[new_indices],
            accepted.reference.qvel[old_indices],atol=1e-8,rtol=0)


def test_measured_tracking_offset_moves_queries_not_native_history(assets):
    from motionbricks.mb_sonic.scene import SOMA_TO_MUJOCO

    adapter,seed,initial=assets
    snapshot=SimpleNamespace(simulation_time_s=.1,actual_root_xyz_m=initial[:3].copy())
    first,origin=continuation_context(seed,None,snapshot,adapter)
    snapshot.actual_root_xyz_m += [1.5,-.8,.2]
    second,shifted=continuation_context(seed,None,snapshot,adapter)
    np.testing.assert_array_equal(first.features,seed.features)
    np.testing.assert_array_equal(second.features,seed.features)
    np.testing.assert_allclose(shifted-origin,np.array([1.5,-.8,.2])@SOMA_TO_MUJOCO,atol=1e-12)


def test_repeated_analog_updates_continue_from_history_on_one_grid(assets):
    adapter,seed,initial=assets
    class HistorySensitiveForecast:
        def generate(self,history,*,velocity_world_xz_mps,command_id,**kwargs):
            times=history.timestamps_s[-1]+(np.arange(32)-3)/30
            features=np.repeat(history.features[-1:],32,axis=0)
            features[:4]=history.features
            for axis,velocity in zip((0,2),velocity_world_xz_mps):
                features[4:,axis]=history.features[-1,axis]+velocity*(times[4:]-times[3])
            # A decoder's reconstructed past can disagree with its input context.
            features[:,:3] += np.array([.57,0.,-.47],dtype=features.dtype)
            source=decode_soma_reference(features[:28],times[:28],adapter.rep,
                scene_id=seed.scene_id,command_id=command_id,trusted_until_s=float(times[27]),
                provenance=dict(test='uses_input_history;not_a_saved_motion_replay'))
            return MBProposal(source,(),history,dict(total=0))
    p=FrozenMBProvider(HistorySensitiveForecast(),adapter,conditioning_mode='B1',
        estimate_actual_diagnostics=False,robot_model_sha256='robot',scene_manifest_sha256='scene',
        rig_id='g1',reference_history_builder=None,consumer_horizon_s=.6,required_future_s=.16,
        maximum_adoption_age_s=.1,planning_lead_s=.02,
        constraint_solver=TrajectoryReferenceSolver(adapter.kinematics,iterations=1,pose_task_frame='root'),
        reference_stream_policy='continuous_native_AR_v3',native_blocks=1)
    observer=prepare_reference_history_b1(adapter.rep,seed.features)
    p.reference_history_builder=partial(reference_history_b1,observer=observer)
    stream=ContinuousReferenceStream(p); parent=None; prior=None
    for step in range(20):
        time=.1+step*.08
        actual=ExecutedHistory(seed.scene_id,samples=[dict(timestamp_s=time,qpos=initial.copy(),qvel=np.zeros(35))])
        snapshot=MBGenerationSnapshot(time,time,round(time*1000),step+1,initial[:3]+[2.,-.5,0.],
            -np.pi/2,seed,actual,1,{},previous_reference_packet=parent)
        command=dict(session_id='analog',episode_id=1,command_id=step+1,command_seq=step+1,
            velocity_xy_mps=[0.,-.01*(step+1)],yaw_rate_radps=0.)
        result=stream.propose(snapshot,command=command,intent_revision=step+1,requested_at_ns=step+1,
            current_sim_time=lambda:time)
        state=stream.states[result.packet.plan_id]
        np.testing.assert_allclose(np.diff(state.source.timestamps_s),1/30,atol=1e-12,rtol=0)
        speed=np.linalg.norm(np.diff(state.source.root_positions_m,axis=0),axis=-1)*30
        assert speed.max()<.201
        assert state.source.root_positions_m[:,0].max()<.01
        if prior is not None:
            handoff=result.conditioning_history.timestamps_s[-1]
            assert handoff>=time+.28-1e-8
            index=np.searchsorted(prior.timestamps_s,handoff-1e-8)
            np.testing.assert_allclose(result.conditioning_history.features[-1,:3],
                prior.root_positions_m[index],atol=2e-6,rtol=0)
        prior=state.source;parent=result.packet


def test_integration_anchor_preserves_future_motion_and_absolute_height(assets):
    adapter,seed,_=assets
    features=np.repeat(seed.features[-1:],28,axis=0)
    features[:4]=seed.features
    times=seed.timestamps_s[-1]+(np.arange(28)-3)/30
    features[:,0]+=.57+np.maximum(np.arange(28)-3,0)*.02
    features[:,2]-=.47
    features[:,1]+=.015
    source=decode_soma_reference(features,times,adapter.rep,scene_id=seed.scene_id,
        command_id='biased_past',trusted_until_s=float(times[-1]),provenance={})
    before=source.raw_features.copy()
    corrected=anchor_integrated_forecast(source,seed)
    np.testing.assert_allclose(corrected.root_positions_m[3,[0,2]],seed.features[-1,[0,2]],atol=1e-7)
    np.testing.assert_allclose(np.diff(corrected.root_positions_m,axis=0),
        np.diff(source.root_positions_m,axis=0),atol=1e-7)
    np.testing.assert_array_equal(corrected.root_positions_m[:,1],source.root_positions_m[:,1])
    np.testing.assert_array_equal(corrected.global_rot_mats,source.global_rot_mats)
    np.testing.assert_allclose(corrected.world_positions_m-corrected.root_positions_m[:,None],
        source.world_positions_m-source.root_positions_m[:,None],atol=2e-7)
    np.testing.assert_array_equal(source.raw_features,before)


def test_shared_clock_repairs_only_roundoff_at_a_real_native_endpoint(assets):
    adapter,seed,_=assets
    times=14.999999999999996+(np.arange(55)-54)/30
    features=np.repeat(seed.features[-1:],len(times),axis=0)
    source=decode_soma_reference(features,times,adapter.rep,scene_id=seed.scene_id,
        command_id='recorded_clock_boundary',trusted_until_s=float(times[-1]),provenance={})
    with pytest.raises(ValueError,match='Trusted coverage'):
        replace(source,trusted_until_s=15.)
    corrected=canonical_native_clock(source)
    assert corrected.timestamps_s[-1]==corrected.trusted_until_s==15.
    np.testing.assert_array_equal(corrected.raw_features,source.raw_features)
    np.testing.assert_array_equal(corrected.root_positions_m,source.root_positions_m)
    replace(corrected,trusted_until_s=15.)
    with pytest.raises(ValueError,match='Trusted coverage'):
        replace(corrected,trusted_until_s=15.02)
    with pytest.raises(ValueError,match='shared simulation clock'):
        canonical_native_clock(replace(source,timestamps_s=times+.001,trusted_until_s=float(times[-1]+.001)))


def test_recorded_user_window_retargets_through_15_seconds_without_extrapolation():
    from motionbricks.mb_sonic.reference import SomaReference,file_sha256
    from scripts.audit_mb_tracker_binary import read_index
    from scripts.isolate_mb_continuous_reference import load_record
    case=ROOT/'out/mb_sonic_e0_e3_20261018/interactive_session_ready_v47'
    path=case/'recording/episode_0003_000.jsonl'
    events=[json.loads(line) for line in path.read_text().splitlines()]
    _,_,artifacts,_,_=read_index(path)
    item=next(e for e in events if e['kind']=='numeric_artifact_written' and e['record_id'].startswith('failed:'))
    assert file_sha256(item['path'])==item['sha256']
    failed=load_record(item['path'])
    original=SomaReference(**failed['raw_source'])
    assert failed['consumer_end_s']==15.
    assert 0<15.-original.timestamps_s[-1]<1e-13
    source=canonical_native_clock(original)
    before=[e for e in events if e['kind']=='reference_packet_adopted' and e['sim_time_s']<=failed['issued_from_sim_time']]
    previous=load_record(artifacts[before[-1]['plan_id']]['path'])['adapted_reference']
    config=json.loads((case/'config.json').read_text())
    adapter=CalibratedSomaG1Adapter.load(config['calibration_path'],robot_model_path=config['robot_model'],
        representation=build_native_representation(),limb_rotation_source='native_material_axes')
    solver=TrajectoryReferenceSolver(adapter.kinematics,iterations=1,pose_task_frame='root')
    boundary=dict(timestamps_s=previous['timestamps_s'],root_pose=previous['qpos'][:,:7],
        joint_positions=previous['qpos'][:,7:],plan_id=before[-1]['plan_id'],
        adoption_until_s=previous['trusted_until_s']-.02)
    reference=adapter.retarget(replace(source,trusted_until_s=15.),start_time_s=failed['consumer_start_s'],
        constraints=solver,boundary=boundary)
    assert len(reference.timestamps_s)==31
    assert reference.timestamps_s[-1]==reference.trusted_until_s==source.timestamps_s[-1]==15.
    np.testing.assert_array_equal(reference.timestamps_s,np.arange(720,751)/50)
