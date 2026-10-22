import pytest
import torch
from torch import nn

from motionbricks.data.short_sequence_convolution import short_sequence_convolutions


@pytest.mark.parametrize('length,dilation',[(1,1),(8,9),(8,27),(16,27),(27,27),(32,27)])
def test_zero_padding_elimination_preserves_values_and_all_gradients(length,dilation):
    torch.manual_seed(123)
    layer=nn.Conv1d(3,5,3,padding=dilation,dilation=dilation).double()
    original=torch.randn(2,3,length,dtype=torch.float64,requires_grad=True)
    reference=layer(original)
    weights=torch.randn_like(reference)
    (reference*weights).sum().backward()
    expected=[original.grad.clone(),layer.weight.grad.clone(),layer.bias.grad.clone()]
    layer.zero_grad(set_to_none=True)
    copied=original.detach().clone().requires_grad_(True)
    with short_sequence_convolutions(layer):
        actual=layer(copied)
        (actual*weights).sum().backward()
    assert 'forward' not in layer.__dict__
    torch.testing.assert_close(actual,reference,atol=1e-13,rtol=1e-13)
    for first,second in zip(expected,[copied.grad,layer.weight.grad,layer.bias.grad]):
        torch.testing.assert_close(first,second,atol=1e-13,rtol=1e-13)
    if dilation>=length:
        assert torch.count_nonzero(layer.weight.grad[:,:,[0,2]])==0


def test_convolution_methods_restore_after_an_exception():
    layer=nn.Conv1d(2,2,3,padding=9,dilation=9)
    with pytest.raises(ValueError):
        with short_sequence_convolutions(layer):
            raise ValueError('probe')
    assert 'forward' not in layer.__dict__
