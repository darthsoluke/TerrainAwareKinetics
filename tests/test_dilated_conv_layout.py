import pytest
import torch

from scripts.check_dilated_conv_layout import vertical_convolutions


@pytest.mark.parametrize('length,dilation', [(8, 3), (8, 27), (16, 9), (32, 27)])
@pytest.mark.parametrize('channels_last', [False, True])
def test_all_taps_and_gradients_survive_axis_exchange(length, dilation, channels_last):
    torch.manual_seed(17)
    layer = torch.nn.Conv1d(4, 5, 3, padding=dilation, dilation=dilation).double()
    value = torch.randn(2, length, 4, dtype=torch.float64).transpose(1, 2).requires_grad_()
    cotangent = torch.randn(2, 5, length, dtype=torch.float64)
    native = layer(value)
    native_grad = torch.autograd.grad(native, (value, layer.weight, layer.bias), cotangent)
    with vertical_convolutions(layer, channels_last=channels_last) as names:
        assert names == ['']
        vertical = layer(value)
        vertical_grad = torch.autograd.grad(vertical, (value, layer.weight, layer.bias), cotangent)
    torch.testing.assert_close(vertical, native, atol=1e-12, rtol=1e-12)
    assert vertical.stride() == native.stride()
    for actual, expected in zip(vertical_grad, native_grad):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    assert 'forward' not in layer.__dict__


def test_guards_and_exception_restore_existing_forward():
    model = torch.nn.ModuleList([
        torch.nn.Conv1d(2, 2, 3, padding=3, dilation=3),
        torch.nn.Conv1d(2, 2, 3, padding=1),
        torch.nn.Conv1d(2, 2, 3, padding=3, dilation=3, groups=2),
        torch.nn.Conv1d(2, 2, 3, padding=3, dilation=3, padding_mode='reflect'),
    ])
    existing = model[0].forward
    model[0].forward = existing
    with pytest.raises(RuntimeError, match='probe failed'):
        with vertical_convolutions(model) as names:
            assert names == ['0']
            raise RuntimeError('probe failed')
    assert model[0].forward is existing
    assert all('forward' not in layer.__dict__ for layer in model[1:])
