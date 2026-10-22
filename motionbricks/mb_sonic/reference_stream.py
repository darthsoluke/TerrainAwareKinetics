"""Continue a native motion forecast and append genuine G1 reference coverage."""

from dataclasses import dataclass, replace
from time import perf_counter_ns

import numpy as np
from scipy.spatial.transform import Rotation

from .reference import G1Reference,NativeSomaHistory,SomaReference
from .retarget import _quat
from .mb_reference import MBProposal, world_xy_command_to_soma, source_in_scene_frame


SOURCE_FIELDS = ('timestamps_s','raw_features','root_positions_m','local_rot_mats',
                 'global_rot_mats','world_positions_m')
ROBOT_FIELDS = ('timestamps_s','qpos','qvel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w')


def canonical_native_clock(source):
    """Use integer150Hz ticks shared by the native30Hz and controller50Hz clocks."""
    times=np.rint(source.timestamps_s*150).astype(np.int64)/150
    tolerance=16*np.spacing(np.maximum(np.abs(source.timestamps_s),1.))
    if np.any(np.abs(times-source.timestamps_s)>tolerance):
        raise ValueError('Native reference timestamps do not match the shared simulation clock')
    trusted=float(round(source.trusted_until_s*150)/150)
    return replace(source,timestamps_s=times,trusted_until_s=trusted,
        provenance=dict(source.provenance,timebase='integer150Hz_ticks'))


def anchor_integrated_forecast(source, history):
    """Integrate future planar displacement from the accepted handoff position."""
    index = int(np.searchsorted(source.timestamps_s,history.timestamps_s[-1]-1e-8))
    if index>=len(source.timestamps_s) or abs(source.timestamps_s[index]-history.timestamps_s[-1])>1e-8:
        raise ValueError('Decoded motion must contain its conditioned handoff sample')
    # XZ are integrated velocities; height and body rotations are absolute predictions.
    shift = np.zeros(3,dtype=source.root_positions_m.dtype)
    shift[[0,2]] = history.features[-1,[0,2]]-source.root_positions_m[index,[0,2]]
    features = source.raw_features.copy()
    features[:,:3] += shift
    return replace(source,raw_features=features,root_positions_m=source.root_positions_m+shift,
        world_positions_m=source.world_positions_m+shift,
        provenance=dict(source.provenance,planar_integration_origin='accepted_context_endpoint',
            integration_origin_correction_m=shift.tolist(),integration_origin_time_s=float(history.timestamps_s[-1])))


def source_join(previous, incoming, cut, earliest):
    old = (previous.timestamps_s <= cut+1e-8)&(previous.timestamps_s >= earliest-1/30)
    new = incoming.timestamps_s > cut+1e-8
    values = {key:np.concatenate((getattr(previous,key)[old],getattr(incoming,key)[new]))
              for key in SOURCE_FIELDS}
    if not np.allclose(np.diff(values['timestamps_s']),1/30,atol=1e-8,rtol=0):
        raise ValueError('Native continuation must retain its original 30Hz time grid')
    return replace(incoming, **values,
        provenance=dict(incoming.provenance, reference_stream='continuous_native_AR_v3',
            retained_prefix_until_s=float(cut), physical_state_writes=0))


def from_robot_poses(adapter, times, poses, template):
    positions, rotations = adapter.kinematics.fk(poses)
    ids = adapter.kinematics.output_ids
    positions, rotations = positions[:,ids], rotations[:,ids]
    dt = np.diff(times)
    linear = np.diff(positions,axis=0)/dt[:,None,None]
    angular = Rotation.from_matrix((rotations[1:]@rotations[:-1].swapaxes(-1,-2)).reshape(-1,3,3))
    angular = angular.as_rotvec().reshape(-1,len(ids),3)/dt[:,None,None]
    linear,angular = np.r_[linear,linear[-1:]],np.r_[angular,angular[-1:]]
    joint = np.diff(poses[:,7:],axis=0)/dt[:,None]
    velocity = np.c_[linear[:,0],np.einsum('nij,nj->ni',rotations[:,0].swapaxes(-1,-2),angular[:,0]),
                     np.r_[joint,joint[-1:]]]
    return replace(template,timestamps_s=times,qpos=poses,qvel=velocity,
        body_pos_w=positions,body_quat_w=_quat(rotations),body_lin_vel_w=linear,body_ang_vel_w=angular,
        trusted_until_s=float(times[-1]))


def continuation_context(history, memory, snapshot, adapter):
    """Preserve native motion; place scene queries using measured tracking offset."""
    from .runtime import calibrated_native_query_origin

    root_ids = adapter.rep.indices['global_root_pos']
    current = snapshot.simulation_time_s
    if memory is None:
        planned = history.features[-1,root_ids]
    else:
        if not memory.timestamps_s[0]-1e-8<=current<=memory.trusted_until_s+1e-8:
            raise ValueError('Current measured state must overlap the native reference')
        planned = np.array([np.interp(current,memory.timestamps_s,memory.root_positions_m[:,i]) for i in range(3)])
    actual = calibrated_native_query_origin(snapshot.actual_root_xyz_m,adapter.root_z_offset_m)
    query_origin = actual+(history.features[-1,root_ids]-planned)
    context = NativeSomaHistory(history.timestamps_s,history.features,history.scene_id,
        dict(kind='legacy_reference_history', reference_stream='continuous_native_AR_v3',
            context='continuous_native_forecast;not_claimed_factual_body_history',
            measured_root_time_s=current, measured_root_xyz_m=snapshot.actual_root_xyz_m.tolist(),
            forecast_context_end_s=float(history.timestamps_s[-1]),
            history_transform='identity_fixed_world_frame',
            scene_query_origin_soma_xyz_m=query_origin.tolist(),
            query_displacement_source='accepted_native_forecast',
            optimizer_updates=0,physical_state_writes=0))
    return context,query_origin


@dataclass
class StreamState:
    source: SomaReference
    reference: G1Reference
    next_window_start: float
    intent: tuple


class ContinuousReferenceStream:
    schema = 'motionbricks.mb_sonic.continuous_native_AR.v3'

    def __init__(self, provider):
        self.provider = provider
        self.states = {}

    def propose(self,snapshot,*,command,intent_revision,requested_at_ns,current_sim_time):
        from .runtime import (MBReferenceUpdate,ReferencePacket,BODY_JOINT_NAMES,
                              next_controller_grid,_angle_delta)

        p = self.provider
        started = perf_counter_ns()
        p.request_evidence,p.failure_evidence = None,None
        parent = snapshot.previous_reference_packet
        old = None if parent is None else self.states[parent.plan_id]
        intent = (command['session_id'],command['command_seq'])
        append = old is not None and old.intent == intent
        start = (old.next_window_start if append else next_controller_grid(snapshot.simulation_time_s+
                 (0. if old is None else p.planning_lead_s)))
        end = round((start+p.consumer_horizon_s)*50)/50
        protected_until = snapshot.simulation_time_s+p.maximum_adoption_age_s+p.required_future_s
        source = None if old is None else old.source
        need_generation = not append or end>source.trusted_until_s+1e-8
        raw_blocks,neural_timing = (),dict(model_calls=0)
        history_ready = started
        if need_generation:
            if old is None:
                context = snapshot.reference_history
                cut = float(context.timestamps_s[-1])
            else:
                handoff = source.trusted_until_s if append else protected_until+.02
                index = int(np.searchsorted(source.timestamps_s,handoff-1e-8))
                if index>=len(source.timestamps_s) or source.timestamps_s[index]>source.trusted_until_s+1e-8:
                    raise ValueError('The accepted native forecast must cover the handoff context')
                cut = float(source.timestamps_s[index])
                context = p.reference_history_builder(source,end_time_s=cut,representation=p.retargeter.rep)
            history,query_origin = continuation_context(context,source,snapshot,p.retargeter)
            history_ready = perf_counter_ns()
            lead = max(0.,history.timestamps_s[-1]-snapshot.simulation_time_s)+p.yaw_goal_lead_s
            heading = _angle_delta(snapshot.derived_heading_rad+float(command['yaw_rate_radps'])*lead,0.)
            velocity,heading = world_xy_command_to_soma(command['velocity_xy_mps'],heading)
            generated = p.generator.generate(history,velocity_world_xz_mps=velocity,
                heading_world_rad=heading,command_id=str(command['command_id']),blocks=p.native_blocks,
                conditioning_mode='B1',query_origin_soma_xyz=query_origin,output_frame='reference_chart')
            raw_blocks,neural_timing = generated.raw_blocks,dict(generated.timing_ns,model_calls=p.native_blocks)
            continuation = anchor_integrated_forecast(canonical_native_clock(generated.source),history)
            if old is None:
                source = continuation
            else:
                source = source_join(old.source,continuation,cut,snapshot.simulation_time_s-.3)
        else:
            context = p.reference_history_builder(source,end_time_s=snapshot.history_time_s,
                representation=p.retargeter.rep)
            history,query_origin = continuation_context(context,source,snapshot,p.retargeter)
        generated_at = perf_counter_ns()
        if end>source.trusted_until_s+1e-8:
            raise ValueError('Continuous reference requires genuine generated future coverage')
        if old is None:
            initial = snapshot.executed_history.samples[-1]
            boundary = dict(initial_qpos=initial['qpos'],initial_qvel=initial['qvel'],
                initial_time_s=initial['timestamp_s'],plan_id='measured_stream_start')
        else:
            reference = old.reference
            boundary = dict(timestamps_s=reference.timestamps_s,root_pose=reference.qpos[:,:7],
                joint_positions=reference.joint_pos,plan_id=parent.plan_id,
                adoption_until_s=(reference.trusted_until_s-.02 if append else
                                  protected_until))
        proposal = MBProposal(source,raw_blocks,history,neural_timing)
        p.request_evidence = dict(proposal=proposal,conditioning_history=history,command=command,
            snapshot=snapshot,consumer_start_s=start,consumer_end_s=end,
            consumer_horizon_s=p.consumer_horizon_s,requested_at_ns=requested_at_ns)
        registered=source_in_scene_frame(source,np.asarray(query_origin)-history.features[-1,:3])
        reference = p.retargeter.retarget(replace(registered,trusted_until_s=end),start_time_s=start,
            constraints=p.constraint_solver,boundary=boundary)
        if append:
            keep = old.reference.timestamps_s<=old.reference.trusted_until_s+1e-8
            added = reference.timestamps_s>old.reference.trusted_until_s+1e-8
            times = np.r_[old.reference.timestamps_s[keep],reference.timestamps_s[added]]
            poses = np.r_[old.reference.qpos[keep],reference.qpos[added]]
            reference = from_robot_poses(p.retargeter,times,poses,reference)
        if old is not None:
            index = np.searchsorted(old.reference.timestamps_s,reference.timestamps_s[0]-1e-9)
            if index>=len(old.reference.timestamps_s) or abs(old.reference.timestamps_s[index]-reference.timestamps_s[0])>1e-8:
                raise ValueError('Quaternion gauge requires an overlapping accepted reference sample')
            signs = np.where(np.sum(reference.body_quat_w[0]*old.reference.body_quat_w[index],axis=-1)<0,-1.,1.)
            reference = replace(reference,body_quat_w=reference.body_quat_w*signs[None,:,None])
        audit = p.retargeter._constraint_audit(reference.qpos,reference.timestamps_s,scope='continuous_G1_buffer')
        if audit['status']!='PASS':
            raise ValueError(f'Continuous G1 reference violates original limits: {audit}')
        earliest = snapshot.simulation_time_s-.2
        keep = reference.timestamps_s>=earliest-1e-8
        reference = replace(reference,**{field:getattr(reference,field)[keep] for field in ROBOT_FIELDS},
            adaptation=dict(reference.adaptation,reference_stream=self.schema,stream_append=append,
                protected_tracker_future_until_s=None if old is None or append else protected_until+.02,
                continuous_buffer_audit=audit))
        plan_id = f'{intent_revision}:{requested_at_ns}'
        self.states = {} if parent is None else {parent.plan_id:old}
        stride = p.consumer_horizon_s-p.required_future_s-p.maximum_adoption_age_s-.02
        self.states[plan_id] = StreamState(source,reference,next_controller_grid(start+stride),intent)
        consumer_start = next_controller_grid(snapshot.simulation_time_s+(0. if old is None else p.planning_lead_s))
        view = reference.timestamps_s>=consumer_start-1e-8
        selected = replace(reference,**{field:getattr(reference,field)[view] for field in ROBOT_FIELDS})
        ready = perf_counter_ns()
        clocks = dict(requested=requested_at_ns,provider_started=started,history_ready=history_ready,
            mb_ready=generated_at,reference_ready=ready)
        packet = ReferencePacket(2,p.rig_id,p.robot_model_sha256,p.scene_manifest_sha256,BODY_JOINT_NAMES,
            dict(length='m',time='s',angle='rad'),'mujoco_world_z_up',snapshot.simulation_time_s,
            command['command_seq'],plan_id,snapshot.state_version,selected.timestamps_s,selected.qpos[:,:7],
            selected.joint_pos,selected.joint_vel,selected.trusted_until_s,'B1',history.timestamps_s,
            snapshot.actual_root_xyz_m.copy(),np.array(query_origin,copy=True),'soma_world_y_up',
            p.retargeter.root_z_offset_m,clocks,command['episode_id'],command['session_id'])
        return MBReferenceUpdate(intent_revision,command,snapshot.history_time_s,snapshot.derived_heading_rad,
            history,None,proposal,selected,dict(history_estimation=history_ready-started,
                mb_generation=generated_at-history_ready,retarget=ready-generated_at,provider_total=ready-started),
            clocks,packet)
