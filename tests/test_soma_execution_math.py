import pytest
import torch

from motionbricks.motionlib.core.utils.rotations import (
    _sqrt_positive_part, matrix_to_quaternion, quaternion_to_matrix,
)
from motionbricks.motionlib.core.utils.torch_utils import batch_rigid_transform


def test_positive_sqrt_preserves_zero_subgradient_and_values():
    x = torch.tensor([-2., 0., .04, 1., 4.], dtype=torch.float64, requires_grad=True)
    reference = torch.zeros_like(x)
    mask = x > 0
    reference[mask] = x[mask].sqrt()
    actual = _sqrt_positive_part(x)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    expected_grad, = torch.autograd.grad(reference.sum(), x)
    actual_grad, = torch.autograd.grad(actual.sum(), x)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert torch.isfinite(actual_grad).all()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_quaternion_and_fk_batch_preserve_native_transforms(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    generator = torch.Generator().manual_seed(17)
    q = torch.randn(3, 8, 4, generator=generator, dtype=torch.float64).to(device)
    q /= q.norm(dim=-1, keepdim=True)
    rotations = quaternion_to_matrix(q)
    torch.testing.assert_close(quaternion_to_matrix(matrix_to_quaternion(rotations)), rotations,
                               atol=1e-12, rtol=1e-12)
    joints = torch.randn(3, 8, 3, generator=generator, dtype=torch.float64).to(device)
    parents = torch.tensor([-1, 0, 0, 1, 1, 2, 4, 5], device=device)
    actual_position, actual_rotation = batch_rigid_transform(rotations, joints, parents, 0)
    positions, global_rotations = [joints[:, 0]], [rotations[:, 0]]
    for index in range(1, 8):
        parent = int(parents[index])
        positions.append(positions[parent] + (global_rotations[parent] @
                          (joints[:, index]-joints[:, parent])[..., None]).squeeze(-1))
        global_rotations.append(global_rotations[parent] @ rotations[:, index])
    torch.testing.assert_close(actual_position, torch.stack(positions, 1), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual_rotation, torch.stack(global_rotations, 1), atol=1e-12, rtol=1e-12)
