import torch
from scripts.fit_joint_sdf_render_head import convolution_design, ridge_delta


def test_final_convolution_design_replays_every_output():
    torch.manual_seed(4)
    head=torch.nn.Conv1d(5,7,3,padding=1).double()
    x=torch.randn(2,5,11,dtype=torch.double)
    design=convolution_design(x,head)
    weights=torch.cat((head.weight.flatten(1),head.bias[:,None]),-1)
    torch.testing.assert_close(design@weights.T,head(x).transpose(1,2),atol=1e-12,rtol=1e-12)


def test_ridge_reduces_held_out_affine_error_without_changing_inputs():
    torch.manual_seed(7)
    x=torch.cat((torch.randn(120,8),torch.ones(120,1)),1)
    truth=torch.randn(3,9)
    y=x@truth.T
    prior=truth+torch.randn_like(truth)*.1
    validation=torch.arange(120)%3==1
    before=x.clone()
    weights,report=ridge_delta(x,y,prior,validation)
    assert report['validation_mse'] < report['prior_validation_mse']*.01
    torch.testing.assert_close(x,before,atol=0,rtol=0)
