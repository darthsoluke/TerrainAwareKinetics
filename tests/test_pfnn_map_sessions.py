from types import SimpleNamespace

import pytest
import torch

from motionbricks.data.soma_lab import SomaLabSession


@pytest.mark.parametrize('method', ['plan','prefetch','acknowledge_played'])
def test_changed_map_rejects_old_session_before_released_backend_is_used(method):
    runtime=SimpleNamespace(scene_revision=0,create_backend=lambda mode:object())
    session=SomaLabSession(runtime,'terrain')
    runtime.scene_revision=1
    session.backend=None
    with pytest.raises(ValueError,match='Map changed'):
        getattr(session,method)({})


def test_map_change_restores_the_original_eager_transformer_reference():
    from motionbricks.data.soma_static_execution import use_dense_transformers,restore_native_transformers
    module=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(8,2,16,dropout=0.,batch_first=True),1).eval()
    inputs=torch.randn(2,4,8)
    mask=torch.tensor([[False,False,False,True],[False,False,True,True]])
    with torch.inference_mode():
        expected=module(inputs,src_key_padding_mask=mask)
        use_dense_transformers(module)
        installed=module.forward
        use_dense_transformers(module)
        assert module.forward is installed
        restore_native_transformers(module)
        torch.testing.assert_close(module(inputs,src_key_padding_mask=mask),expected,atol=0,rtol=0)


def test_map_selection_endpoint_requires_an_explicit_known_map(tmp_path):
    import asyncio
    from aiohttp.test_utils import TestClient,TestServer
    from scripts.soma_lab_server import create_app
    (tmp_path/'vendor').mkdir()
    selected=['urban']
    def select(map_id):
        if map_id not in ('urban','000'):
            raise ValueError('Unknown PFNN map')
        selected[0]=map_id
    runtime=SimpleNamespace(select_map=select,config=lambda:{'map':selected[0]},clips={})
    async def run():
        async with TestClient(TestServer(create_app(runtime,tmp_path))) as client:
            assert (await (await client.get('/api/config')).json())['map']=='urban'
            response=await client.post('/api/maps/select',json={'map_id':'000'})
            assert response.status==200 and (await response.json())['map']=='000'
            for value in ({},{'map_id':'missing'},{'map_id':'urban','extra':True}):
                assert (await client.post('/api/maps/select',json=value)).status==400
                assert selected[0]=='000'
    asyncio.run(run())


def test_cached_map_switch_rebinds_field_seed_and_validation_without_recompiling(monkeypatch,tmp_path):
    from scripts.soma_full_corpus_acceptance_server import FullCorpusAcceptanceRuntime
    scene={'heightfield_y_m':object()}
    field=object()
    runner=SimpleNamespace(scene=None,field=None,validation=None)
    runtime=FullCorpusAcceptanceRuntime.__new__(FullCorpusAcceptanceRuntime)
    seed=object()
    runtime._scene_cache={str(tmp_path.resolve()):dict(terrain_scene=scene,terrain_initial_history=seed,
        terrain_quality={'execution_parity':['verified']})}
    runtime.inference=SimpleNamespace(device='cuda:0',_terrain_execution=runner)
    runtime.unified_controller=SimpleNamespace(scene_field=lambda value:field if value is scene else None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda _:None)
    runtime.switch_artifact(tmp_path)
    assert runner.scene is scene and runner.field is field
    assert runner.validation==['verified'] and runtime.terrain_initial_history is seed
    assert runtime.inference._terrain_execution is runner


def test_scene_selection_can_omit_browser_cached_mesh_but_always_switches():
    from scripts.soma_pfnn_maps_server import PFNNMapRuntime
    runtime=SimpleNamespace(current_map='012',load_reports={'012':{'seconds':.001}},
        config=lambda:dict(soma_skin='skin',faces='faces',clips=[],terrain_scene='mesh',
                           map_catalog={'current':'012'},initial_root=[1.,2.,3.]))
    first=PFNNMapRuntime.selection_config(runtime)
    cached=PFNNMapRuntime.selection_config(runtime,cached_scene=True)
    assert first['terrain_scene']=='mesh' and 'terrain_scene' not in cached
    assert not {'soma_skin','faces','clips'}.intersection(first)
    assert first['map_catalog']==cached['map_catalog']=={'current':'012'}


def test_prepared_volume_is_disallowed_in_training_and_context_is_restored():
    from motionbricks.data.unified_terrain import UnifiedTerrainController
    runtime=SimpleNamespace(_context=None,_adapting=False,_straight_through=False)
    with pytest.raises(ValueError,match='inference-only'):
        with UnifiedTerrainController.context(runtime,None,None,volume_observation={}):pass
    with torch.inference_mode():
        with pytest.raises(ValueError,match='inference-only'):
            with UnifiedTerrainController.context(runtime,None,None,training=True,volume_observation={}):pass
        with pytest.raises(RuntimeError,match='probe'):
            with UnifiedTerrainController.context(runtime,None,None,volume_observation={'valid':True}):
                assert runtime._context['volume_observation']=={'valid':True}
                raise RuntimeError('probe')
    assert runtime._context is None
