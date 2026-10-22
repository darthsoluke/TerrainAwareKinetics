import torch

from motionbricks.data.terrain_detail_adapter import near_relief
from motionbricks.data.terrain_model_v3 import FarSceneResidual


def test_near_adaptation_is_zero_on_planes_even_with_nonzero_learned_gate():
    torch.set_num_threads(2)
    torch.manual_seed(12)
    points=torch.rand(2,2,9,9,3);points[...,1]=-.9
    valid=torch.ones(2,2,9,9,dtype=torch.bool)
    observation={'points':points,'valid':valid,'speed_mps':torch.tensor([0.,1.6]),'horizon_seconds':torch.ones(2)}
    module=FarSceneResidual(16);module.gate.data.fill_(.7)
    queries=torch.randn(2,8,16)
    torch.testing.assert_close(module(queries,near_relief(observation)),queries,atol=0,rtol=0)
    observation['points'][0,1,4:,:,1]+=.4
    result=module(queries,near_relief(observation))
    assert not torch.equal(result[0],queries[0])
    torch.testing.assert_close(result[1],queries[1],atol=0,rtol=0)
    result[0].square().mean().backward()
    assert module.encoder[0].weight.grad.abs().sum()>0


def test_detail_initialization_preserves_actual_native_prediction():
    from omegaconf import OmegaConf
    from test_native_mb_online_conditions import online_infer
    from test_native_mb_stage_training import _motion_and_representation
    from test_native_mb_terrain_integration import small_config
    from motionbricks.data.terrain_model_v3 import JointTerrainController
    from motionbricks.data.terrain_detail_adapter import TerrainDetailController
    from motionbricks.data.terrain_control import SoftCommandState
    torch.set_num_threads(2)
    original_inference,_=online_infer();detail_inference,_=online_infer()
    bundle={'tokenizer_config':OmegaConf.to_container(small_config('tokenizer'))}
    original=JointTerrainController(original_inference,bundle).eval()
    detail=TerrainDetailController(detail_inference,bundle).eval()
    loaded=detail.load_state_dict(original.state_dict(),strict=False)
    assert not loaded.unexpected_keys
    detail.initialize_detail_from_far()
    _,batch=_motion_and_representation(33)
    history=original_inference.global_motion_rep.unnormalize(batch['motion'][:1,:4])
    hf=torch.zeros(161,161);hf[85:]=.3
    scene={'heightfield_y_m':hf,'min_point_world_xyz_m':[-8.,0.,-8.],'dx_m':.1}
    with torch.inference_mode():
        task=SoftCommandState().intent(history,[0.,1.6],0.,8)
        expected=original(history,task,[scene])['world']
        actual=detail(history,task,[scene])['world']
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
