"""Governor-only changes are prepared on a private next-packet backend."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from motionbricks.data.anticipatory_speed import DEFAULTS,GovernorState
from motionbricks.data.pfnn_tuning import TunablePFNNBackend
from motionbricks.data.soma_lab import SomaLabSession
from motionbricks.data.unified_terrain import PROFILE


def session(monkeypatch):
    backend=object.__new__(TunablePFNNBackend)
    backend.history=torch.zeros(1,4,6)
    backend.speed_map=object()
    backend.speed_governor=dict(DEFAULTS)
    backend.governor_state=GovernorState()
    backend._published_governor_state=None
    backend._future_buffer=backend._last_published_history=backend._last_command=backend._debug_future=None
    backend.execution=SimpleNamespace(supports=lambda _:True,validation=[dict(tokens_equal=True)])
    backend.runtime_controls=dict(PROFILE,refresh_frames=4)
    backend.num_tokens=8
    backend.inference_config=dict(num_inference_step=4)
    backend.observe_history=lambda frames:frames.clone()

    def predict(self,velocity,heading):
        scale=1. if not self.speed_governor['enabled'] else .5
        motion=self.history[:,-1:].repeat(1,8,1)+torch.arange(1,9).view(1,8,1)*scale
        after=replace(self.governor_state,frame=self.governor_state.frame+8,target_speed=scale)
        self._future_buffer=motion[:,:0].clone()
        self._last_published_history=motion[:,-4:].clone()
        self._last_command=(scale,)
        self.last_generation=dict(runtime_controls=dict(self.runtime_controls),inference_config=dict(self.inference_config))
        return dict(planned_motion=motion,published_history=self._last_published_history,
                    generation_seconds=0.,governor_state_after=after)
    monkeypatch.setattr(TunablePFNNBackend,'_predict_tuned',predict)
    runtime=SimpleNamespace(fps=30,create_backend=lambda _:backend,
        skin_features=lambda motion:(motion[0,:,:3].numpy(),motion[0,:,3:].numpy().reshape(8,1,3)))
    return SomaLabSession(runtime,'terrain')


@pytest.mark.parametrize('prefix',[0,2,4])
def test_toggle_candidate_is_private_until_ack_and_matches_serial(monkeypatch,prefix):
    live=session(monkeypatch)
    command=dict(mode='terrain',velocity_xz=[0.,3.],heading=0.)
    live.plan(dict(command,type='plan',request_id=1))
    for source,enabled in ((1,True),(2,False),(3,True)):
        backend=live.backend
        pending=live.pending_playback
        old_config=dict(backend.speed_governor)
        old_state=backend.governor_state
        old_motion=pending['motion'].clone()
        old_after=pending['governor_state']
        old_buffer=backend._future_buffer
        desired=dict(DEFAULTS,enabled=enabled)
        serial=copy.copy(backend)
        serial.history=pending['history'].clone()
        serial.commit_governor_state(old_after)
        serial.configure_governor(desired)
        expected=serial.predict_plan([[0.,3.]],[0.])
        metadata,payload=live.prefetch(dict(command,type='prefetch',request_id=source+1,
            source_request_id=source,played_frame_count=prefix,speed_governor=desired))
        candidate=pending['prepared']['candidate']
        assert metadata['type']=='plan' and metadata['prefetched'] and payload
        assert backend.speed_governor==old_config and backend.governor_state is old_state
        assert backend._future_buffer is old_buffer and torch.equal(pending['motion'],old_motion)
        assert pending['governor_state'] is old_after
        assert torch.equal(candidate['planned_motion'],expected['planned_motion'])
        assert torch.equal(candidate['published_history'],expected['published_history'])
        assert candidate['governor_state_before']==serial.governor_state
        assert candidate['governor_state_after']==expected['governor_state_after']
        live.acknowledge_played(dict(request_id=source,frame_count=8))
        assert backend.speed_governor==desired
        assert backend.governor_state==candidate['governor_state_before']
        assert backend.governor_state.frame==source*8
        assert torch.equal(backend.history,pending['history'])
    live.acknowledge_played(dict(request_id=4,frame_count=8))
    assert live.backend.governor_state.frame==32


def test_model_controls_still_defer_without_applying_governor(monkeypatch):
    live=session(monkeypatch)
    command=dict(mode='terrain',velocity_xz=[0.,3.],heading=0.)
    live.plan(dict(command,type='plan',request_id=1))
    before=live.backend.governor_state
    metadata,payload=live.prefetch(dict(command,type='prefetch',request_id=2,source_request_id=1,
        played_frame_count=4,speed_governor=dict(DEFAULTS,enabled=True),
        runtime_controls=dict(live.backend.runtime_controls,refresh_frames=8)))
    assert metadata['reason']=='controls_apply_at_played_boundary' and payload is None
    assert live.backend.governor_state is before and not live.backend.speed_governor['enabled']
    assert 'prepared' not in live.pending_playback


def test_horizon_candidate_copies_and_commits_all_runtime_settings(monkeypatch):
    live=session(monkeypatch)
    command=dict(mode='terrain',velocity_xz=[0.,3.],heading=0.)
    live.plan(dict(command,type='plan',request_id=1))
    for source,tokens in enumerate((12,16,8),1):
        backend=live.backend
        old_runtime=backend.runtime_controls
        old_inference=backend.inference_config
        desired=dict(old_runtime,num_tokens=tokens)
        meta,_=live.prefetch(dict(command,type='prefetch',request_id=source+1,source_request_id=source,
            played_frame_count=4,runtime_controls=desired,speed_governor=dict(DEFAULTS,enabled=source%2==1)))
        assert meta['type']=='plan' and meta['generation']['runtime_controls']==desired
        assert backend.runtime_controls is old_runtime and backend.inference_config is old_inference
        assert backend.num_tokens==old_runtime['num_tokens']
        live.acknowledge_played(dict(request_id=source,frame_count=8))
        assert backend.num_tokens==tokens and backend.runtime_controls==desired
        assert backend.inference_config==old_inference and backend.inference_config is not old_inference
        assert backend.runtime_controls is not old_runtime
    live.acknowledge_played(dict(request_id=4,frame_count=8))
    assert live.backend.governor_state.frame==32


def test_shared_trace_preserves_frozen_weight_storage():
    from motionbricks.data.soma_static_execution import trace_fixed_call
    weight=torch.randn(4,4)
    sample=torch.randn(1,4)
    graphs=[trace_fixed_call(lambda x:x@weight,(sample,),shared_cuda_constants=True) for _ in range(3)]
    pointers=[]
    for graph in graphs:
        for node in graph.graph.nodes():
            if node.kind()=='prim::Constant' and 'value' in node.attributeNames() and node.kindOf('value')=='t':
                value=node.t('value')
                if value.shape==weight.shape:pointers.append(value.untyped_storage().data_ptr())
        torch.testing.assert_close(graph(sample),sample@weight,atol=0,rtol=0)
    assert pointers==[weight.untyped_storage().data_ptr()]*3
