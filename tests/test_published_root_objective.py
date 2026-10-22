import torch

from motionbricks.data.published_root_objective import published_root_reconstruction


def test_late_height_is_constrained_without_editing_actor_or_training_unknown_tail():
    source=torch.zeros(1,32,3,requires_grad=True)
    prediction=torch.zeros(1,32,5);prediction[:,20:24,1]=.12;prediction.requires_grad_()
    before=prediction.detach().clone()
    loss=published_root_reconstruction(prediction,source,[2.,2.],.5)
    loss.backward()
    assert prediction.grad[:,20:24,1].abs().sum()>0
    assert prediction.grad[:,28:].abs().sum()==0 and prediction.grad[...,3:].abs().sum()==0
    assert source.grad is None
    torch.testing.assert_close(prediction.detach(),before,atol=0,rtol=0)


def test_matches_real_source_and_shared_world_translation_is_invariant():
    source=torch.randn(2,40,3)
    assert published_root_reconstruction(source,source,[2.,3.],.6)==0
    prediction=source.clone();prediction[:,10:15,0]+=.04
    first=published_root_reconstruction(prediction,source,[2.,3.],.6)
    shift=source.new_tensor([20.,3.,-15.])
    second=published_root_reconstruction(prediction+shift,source+shift,[2.,3.],.6)
    torch.testing.assert_close(first,second,atol=1e-5,rtol=1e-5)
