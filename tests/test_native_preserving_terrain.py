import pytest
import torch

from motionbricks.data.native_preserving_terrain import SceneResidual, terrain_contrast


def fixture():
    observation = dict(points=torch.zeros(1,2,9,9,3), valid=torch.ones(1,2,9,9,dtype=torch.bool))
    actual = torch.randn(1,8,15,8)
    valid = torch.ones(1,8,15,dtype=torch.bool)
    return observation, actual, actual.clone(), valid


def test_flat_geometry_and_roundoff_do_not_activate_terrain():
    observation, actual, plane, valid = fixture()
    assert not terrain_contrast(observation, actual, plane, valid).any()
    observation['points'][...,1] = -1.24
    actual[...,3] += 1e-7
    assert not terrain_contrast(observation, actual, plane, valid).any()


def test_real_step_or_limb_sdf_difference_activates():
    observation, actual, plane, valid = fixture()
    observation['points'][:,1,3,3,1] = .12
    assert terrain_contrast(observation, actual, plane, valid).all()
    observation['points'].zero_()
    actual[:,3,4,3] += .03
    assert terrain_contrast(observation, actual, plane, valid).all()


def test_unknown_samples_never_become_observed_terrain():
    observation, actual, plane, valid = fixture()
    actual[:,3,4] += 30
    valid[:,3,4] = False
    assert not terrain_contrast(observation, actual, plane, valid).any()
    observation['valid'][:,0,4,4] = False
    with pytest.raises(ValueError, match='unknown'):
        terrain_contrast(observation, actual, plane, valid)


def test_trained_residual_is_identity_for_identical_flat_features():
    torch.manual_seed(7)
    module = SceneResidual(32).eval()
    torch.nn.init.normal_(module.output.weight)
    value = torch.randn(1,8,32)
    memory = torch.randn(1,20,128)
    valid = torch.ones(1,20,dtype=torch.bool)
    output = module(value,memory,memory.clone(),valid)
    torch.testing.assert_close(output,value,atol=0,rtol=0)
    output.square().sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in module.parameters())


def test_zero_initialization_has_zero_forward_effect_and_nonzero_learning_signal():
    torch.manual_seed(8)
    module = SceneResidual(32).eval()
    value = torch.randn(1,8,32)
    actual, plane = torch.randn(1,20,128), torch.randn(1,20,128)
    output = module(value,actual,plane,torch.ones(1,20,dtype=torch.bool))
    torch.testing.assert_close(output,value,atol=0,rtol=0)
    output.square().sum().backward()
    assert module.output.weight.grad.norm()>0


def test_no_valid_landmarks_yields_no_residual():
    module = SceneResidual(16)
    value = torch.randn(1,8,16)
    memory = torch.randn(1,20,128)
    assert module(value,memory,memory,torch.zeros(1,20,dtype=torch.bool)) is value
