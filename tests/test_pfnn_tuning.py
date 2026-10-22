from types import SimpleNamespace

import pytest
import torch

from motionbricks.data.native_mb_online_live import NativeTerrainOnlineBackend
from motionbricks.data.pfnn_tuning import TunablePFNNBackend,validate_tuning,tuning_contract
from motionbricks.data.unified_terrain import UnifiedTerrainBackend,PROFILE


@pytest.mark.parametrize('refresh',[1,2,4,8,16,24])
def test_tuning_explicit_refresh_choices_keep_original_default(refresh):
    value=validate_tuning(dict(PROFILE,refresh_frames=refresh))
    assert value['refresh_frames']==refresh
    assert tuning_contract()['defaults']==PROFILE
    assert tuning_contract()['fixed']['packet_frames']==8


@pytest.mark.parametrize('change',[{'refresh_frames':3},{'refresh_frames':True},{'num_tokens':5},
    {'inference_steps':0},{'path_refinements':1},{'lookahead_m':float('nan')},
    {'num_tokens':6,'refresh_frames':24},{'unknown':1}])
def test_rejects_unbounded_or_undeclared_tuning(change):
    with pytest.raises(ValueError):validate_tuning(dict(PROFILE,**change))


@pytest.mark.parametrize('refresh',[1,2,4])
def test_internal_replanning_keeps_all_eight_frames_and_does_not_advance_live_history(monkeypatch,refresh):
    backend=TunablePFNNBackend.__new__(TunablePFNNBackend)
    backend.history=torch.zeros(1,4,6)
    backend.runtime_controls=dict(PROFILE,refresh_frames=refresh)
    backend.num_tokens=8
    backend.inference_config={}
    calls=[]
    def predict(worker,velocity,heading):
        calls.append(worker.history.clone())
        first=float(worker.history[0,-1,0])+1
        motion=torch.zeros(1,worker.PUBLISH_FRAMES,6)
        motion[0,:,0]=torch.arange(first,first+worker.PUBLISH_FRAMES)
        history=torch.cat((worker.history,motion),1)[:,-4:]
        worker._last_command=(tuple(velocity),heading)
        return dict(planned_motion=motion,published_history=history,debug={})
    monkeypatch.setattr(UnifiedTerrainBackend,'predict_plan',predict)
    monkeypatch.setattr(NativeTerrainOnlineBackend,'_publication',lambda self,motion:
        dict(published_history=motion[:,-4:].clone()))
    before=backend.history.clone()
    result=backend.predict_plan([0.,4.],0.)
    torch.testing.assert_close(result['planned_motion'][0,:,0],torch.arange(1,9).float(),atol=0,rtol=0)
    torch.testing.assert_close(backend.history,before,atol=0,rtol=0)
    torch.testing.assert_close(result['published_history'][0,:,0],torch.arange(5,9).float(),atol=0,rtol=0)
    assert len(calls)==8//refresh
    assert backend.last_generation['internal_replans']==8//refresh
    assert backend._future_buffer.shape==(1,0,6)


def test_failed_internal_replan_preserves_live_history_and_existing_buffer(monkeypatch):
    backend=TunablePFNNBackend.__new__(TunablePFNNBackend)
    backend.history=torch.zeros(1,4,6)
    backend.runtime_controls=dict(PROFILE,refresh_frames=4)
    backend._future_buffer=torch.ones(1,16,6)
    before=backend._future_buffer.clone()
    def predict(worker,*_):raise ValueError('Unknown terrain')
    monkeypatch.setattr(UnifiedTerrainBackend,'predict_plan',predict)
    with pytest.raises(ValueError,match='Unknown terrain'):backend.predict_plan([0.,4.],0.)
    assert torch.count_nonzero(backend.history)==0
    torch.testing.assert_close(backend._future_buffer,before,atol=0,rtol=0)


def test_tuning_assets_and_contract_are_served_only_by_capable_runtime(tmp_path):
    import asyncio
    from aiohttp.test_utils import TestClient,TestServer
    from scripts.soma_lab_server import create_app
    (tmp_path/'vendor').mkdir()
    for name in ('tuning.html','tuning.js','tuning.css','tuning-contract.js'):
        (tmp_path/name).write_text(name)
    async def run():
        runtime=SimpleNamespace(clips={},tuning_config=lambda:{'runtime':tuning_contract()})
        async with TestClient(TestServer(create_app(runtime,tmp_path))) as client:
            assert (await (await client.get('/api/tuning')).json())['runtime']['fixed']['packet_frames']==8
            for url in ('/settings','/tuning.js','/tuning.css','/tuning-contract.js'):
                assert (await client.get(url)).status==200
        async with TestClient(TestServer(create_app(SimpleNamespace(clips={}),tmp_path))) as client:
            assert (await client.get('/settings')).status==404
            assert (await client.get('/api/tuning')).status==404
    asyncio.run(run())
