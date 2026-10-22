"""Explicit experimental controls for the PFNN map viewer, with immutable packets."""

import copy
import time

import torch

from .anticipatory_speed import DEFAULTS as GOVERNOR_DEFAULTS, GovernorState, validate_governor

from .native_mb_online_live import NativeTerrainOnlineBackend
from .terrain_runtime_controls import validate_runtime_controls
from .unified_committed_playback import CommittedUnifiedBackend
from .unified_terrain import UnifiedTerrainBackend, PROFILE


TUNING_SCHEMA='motionbricks.pfnn_runtime_tuning.v1'
REFRESH_CHOICES=(1,2,4,8,16,24)


def validate_tuning(values):
    if not isinstance(values,dict):raise ValueError('Expected runtime settings')
    refresh=values.get('refresh_frames')
    if type(refresh) is not int or refresh not in REFRESH_CHOICES:
        raise ValueError('Replan interval must be 1, 2, 4, 8, 16 or24 frames')
    checked=validate_runtime_controls(dict(values,refresh_frames=max(8,refresh)))
    if checked['path_refinements']!=0:
        raise ValueError('Route refinement remains disabled')
    if refresh>4*checked['num_tokens']-8:
        raise ValueError('Replan interval exceeds the available generated motion')
    checked['refresh_frames']=refresh
    return checked


class TunablePFNNBackend(CommittedUnifiedBackend):
    """Replan inside an eight-frame packet using its immutable generated prefix.

    The playback-acknowledged live history never advances while constructing a
    packet. Every output frame remains unmodified native model output.
    """

    validate_runtime_controls=staticmethod(validate_tuning)

    def __init__(self,controller,initial_history,scene,*,speed_map=None):
        super().__init__(controller,initial_history,scene)
        self.speed_map=speed_map
        self.speed_governor=dict(GOVERNOR_DEFAULTS)
        self.governor_state=GovernorState()
        self._published_governor_state=None

    def configure_governor(self,values):
        values=validate_governor(values)
        if values['enabled'] and self.speed_map is None:
            raise ValueError('Terrain speed controls require the selected static heightfield')
        if values!=self.speed_governor:
            self.speed_governor=values
            self.governor_state=GovernorState(frame=self.governor_state.frame)
            self._published_governor_state=None
            self._future_buffer=self._last_published_history=self._last_command=self._debug_future=None

    def commit_governor_state(self,state):
        if not isinstance(state,GovernorState) or state.frame!=self.governor_state.frame+8:
            raise ValueError('Governor state does not match the acknowledged eight frames')
        self.governor_state=state

    def _future_backend(self,expected):
        future=super()._future_backend(expected)
        if self._published_governor_state is None:
            raise ValueError('Published governor state is missing')
        future.governor_state=self._published_governor_state
        return future

    def commit_prepared_plan(self,candidate):
        runtime=validate_tuning(candidate['generation']['runtime_controls'])
        inference=dict(candidate['generation']['inference_config'])
        if inference['num_inference_step']!=runtime['inference_steps']:
            raise ValueError('Prepared inference settings differ from the runtime profile')
        settings=validate_governor(candidate['generation']['speed_governor'])
        before=candidate['governor_state_before']
        after=candidate['governor_state_after']
        if (not isinstance(before,GovernorState) or not isinstance(after,GovernorState)
                or before.frame!=self.governor_state.frame or after.frame!=before.frame+8):
            raise ValueError('Prepared governor state does not follow the acknowledged packet')
        super().commit_prepared_plan(candidate)
        self.num_tokens=runtime['num_tokens']
        self.runtime_controls=runtime
        self.inference_config=inference
        self.speed_governor=settings
        self.governor_state=before
        self._published_governor_state=after

    def can_prefetch_runtime(self,values):
        from .root_path_split_execution import HORIZONS
        old=self.runtime_controls
        return (self.execution is not None and self.execution.supports(values) and self.execution.supports(old)
            and old['num_tokens'] in HORIZONS and values['num_tokens'] in HORIZONS
            and old['refresh_frames']==values['refresh_frames']==4
            and all(old[key]==values[key] for key in old if key!='num_tokens'))

    def _governed_step(self,velocity_xz,heading):
        started=time.perf_counter()
        before=self.governor_state
        requested=torch.as_tensor(velocity_xz,dtype=torch.float64).reshape(2).tolist()
        if self.speed_map is None:
            from dataclasses import replace
            after=replace(before,frame=before.frame+self.PUBLISH_FRAMES)
            info=dict(enabled=False,mode='off',requested_speed_mps=float(torch.tensor(requested).norm()),
                      target_speed_mps=float(torch.tensor(requested).norm()),risk_point_world_xyz_m=None)
            target=requested
        else:
            roots=self.history[0,:,:5].detach().cpu().numpy()
            target,after,info=self.speed_map.govern(roots,requested,before,self.speed_governor,self.PUBLISH_FRAMES)
        info['overhead_ms']=(time.perf_counter()-started)*1000
        governed=velocity_xz if not self.speed_governor['enabled'] else [target]
        result=UnifiedTerrainBackend.predict_plan(self,governed,heading)
        debug=dict(result['debug'])
        path=debug.get('query_proposal_world_xz_m')
        if path and self.last_generation['neural_generation']:
            after=after.bind_plan(path,before.frame)
        info.update(requested_velocity_xz=requested,target_velocity_xz=target)
        debug['speed_governor']=info
        result.update(debug=debug,governor_state_after=after)
        return result

    def configure_runtime(self,values):
        values=validate_tuning(values)
        # The shared validator only accepts complete eight-frame refresh units.
        if values!=self.runtime_controls:
            NativeTerrainOnlineBackend.configure_runtime(self,dict(values,refresh_frames=max(8,values['refresh_frames'])))
            self.runtime_controls=values

    def _publication(self,future):
        if getattr(self,'_subpacket',False):
            history=torch.cat((self.history,future),1)[:,-4:].contiguous()
            return {'published_history':self.observe_history(history)}
        return super()._publication(future)

    @torch.inference_mode()
    def predict_plan(self,velocity_xz,heading):
        execution=getattr(self,'execution',None)
        native=execution is not None and not execution.supports(self.runtime_controls)
        if native:
            from .soma_static_execution import restore_native_transformers,use_dense_transformers
            restore_native_transformers(self.inference)
        try:
            before=self.governor_state
            result=self._predict_tuned(velocity_xz,heading)
            result['governor_state_before']=before
            self.last_generation['neural_execution']=('native' if native else
                getattr(execution,'neural_execution','verified-cuda'))
            self.last_generation['speed_governor']=dict(self.speed_governor)
            self._published_governor_state=result['governor_state_after']
            return result
        finally:
            if native:use_dense_transformers(self.inference)

    def _predict_tuned(self,velocity_xz,heading):
        refresh=self.runtime_controls['refresh_frames']
        if refresh>=8:
            result=self._governed_step(velocity_xz,heading)
            self.last_generation=dict(self.last_generation,tuning_schema=TUNING_SCHEMA,
                internal_replans=int(self.last_generation['neural_generation']))
            return result
        started=time.perf_counter()
        worker=copy.copy(self)
        worker.PUBLISH_FRAMES=refresh
        worker._subpacket=True
        worker.history=self.history.clone()
        worker._future_buffer=None
        worker._last_published_history=None
        worker._last_command=None
        parts,observations=[],[]
        for _ in range(8//refresh):
            result=worker._governed_step(velocity_xz,heading)
            parts.append(result['planned_motion'])
            worker.history=result['published_history'].clone()
            worker.governor_state=result['governor_state_after']
            observations.append(result['debug'])
        future=torch.cat(parts,1)
        self._future_buffer=future[:,:0].clone()
        self._last_command=worker._last_command
        debug=dict(observations[0])
        debug['internal_observations']=observations
        debug['internal_replan_frames']=refresh
        debug['decoded_world_xyz_m']=future[0,:,:3].cpu().tolist()
        self._debug_future=debug
        self.last_generation=dict(neural_generation=True,predicted_frames=self.num_tokens*4,remaining_frames=0,
            future_height_source='model',future_pose_source='model',runtime_controls=dict(self.runtime_controls),
            inference_config=dict(self.inference_config),internal_replans=len(parts),tuning_schema=TUNING_SCHEMA,
            internal_history_source='immutable_generated_packet_prefix')
        elapsed=time.perf_counter()-started
        publication=super()._publication(future)
        self._last_published_history=publication['published_history'].clone()
        return dict(planned_motion=future,generation_seconds=elapsed,debug=debug,
                    governor_state_after=worker.governor_state,**publication)


def tuning_contract():
    return dict(schema=TUNING_SCHEMA,defaults=dict(PROFILE),trained_defaults=dict(PROFILE),locked=False,
        max_speed_mps=12.,apply='next_fully_played_boundary',experimental=True,
        limits=dict(num_tokens=[6,16],inference_steps=[1,8],refresh_frames=list(REFRESH_CHOICES),
            lookahead_m=[.8,20.],lateral_halfwidth_m=[.3,6.],lookahead_gain=[0.,4.],lookahead_base_m=[0.,10.]),
        fixed=dict(history_frames=4,packet_frames=8,fps=30,path_refinements=0,postprocessing=None))
