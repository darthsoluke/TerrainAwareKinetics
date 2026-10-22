import torch

from motionbricks.data.replan_boundary_objective import all_replan_boundaries


def test_internal_replan_is_supervised_and_both_published_sides_receive_gradients():
    source=torch.zeros(1,32,2,3)
    prediction=source.clone();prediction[:,12:,0,0]=.1
    prediction.requires_grad_()
    loss=all_replan_boundaries(prediction,source)
    assert loss>0
    loss.backward()
    assert prediction.grad[:,11].abs().sum()>0 and prediction.grad[:,12].abs().sum()>0
    assert prediction.grad[:,28:].abs().sum()==0


def test_identical_source_and_first_primitive_match_existing_boundary_objective():
    source=torch.randn(2,32,5,3)
    assert all_replan_boundaries(source,source)==0
    prediction=source[:,:16].clone();prediction[:,4]+= .02
    expected=((prediction[:,4]-prediction[:,3])-(source[:,4]-source[:,3])).norm(dim=-1).mean()*30
    torch.testing.assert_close(all_replan_boundaries(prediction,source,publish_frames=16),expected)
