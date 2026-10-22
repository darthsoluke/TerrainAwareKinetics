import torch
from types import SimpleNamespace

from scripts.pilot_joint_sdf_leg_head import leg_rotation_rows, restrict_head_gradients


def test_leg_rows_exclude_root_and_upper_body():
    names = ['Hips']+[s+p for s in ('Left','Right') for p in ('Leg','Shin','Foot','ToeBase')]+['Other'+str(i) for i in range(21)]
    rep = SimpleNamespace(indices={'global_rot_data':list(range(100,280))}, skeleton=SimpleNamespace(bone_order_names=names))
    rows, joints = leg_rotation_rows(rep)
    assert rows == list(range(106,154))
    assert len(joints) == 8 and len(rows) == 48


def test_adam_cannot_change_unselected_rows_or_input():
    torch.manual_seed(1)
    head = torch.nn.Conv1d(8, 16, 3, padding=1)
    rows = [3,4,9]
    frozen = sorted(set(range(16))-set(rows))
    before_weight, before_bias = head.weight.detach().clone(), head.bias.detach().clone()
    handles = restrict_head_gradients(head, rows)
    x = torch.randn(2,8,32)
    original = x.clone()
    optimizer = torch.optim.Adam(head.parameters(), lr=5e-5, weight_decay=0.)
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        head(x).square().mean().backward()
        assert torch.count_nonzero(head.weight.grad[frozen]) == 0
        optimizer.step()
    assert torch.equal(before_weight[frozen], head.weight[frozen])
    assert torch.equal(before_bias[frozen], head.bias[frozen])
    assert not torch.equal(before_weight[rows], head.weight[rows])
    assert torch.equal(x, original)
    for handle in handles:
        handle.remove()
