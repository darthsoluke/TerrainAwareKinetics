"""A destination directory stops the old runner only after torch.save finishes."""

import torch
import pytest


def test_rename_barrier_preserves_complete_optimizer_checkpoint(tmp_path):
    latest = tmp_path/'latest.pt'
    torch.save({'actual_updates': 1}, latest)
    latest.replace(tmp_path/'previous.pt')
    latest.mkdir()
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.Adam([parameter], lr=.01)
    stopped_at = None
    for step in range(1, 5):
        optimizer.zero_grad()
        parameter.square().sum().backward()
        optimizer.step()
        if step == 2:
            value = dict(model=parameter.detach().clone(), optimizer=optimizer.state_dict(),
                actual_updates=step, rng=torch.get_rng_state())
            torch.save(value, tmp_path/'latest.tmp.pt')
            with pytest.raises(OSError):
                (tmp_path/'latest.tmp.pt').replace(latest)
            stopped_at = step
            break
    checkpoint = torch.load(tmp_path/'latest.tmp.pt', weights_only=False)
    assert stopped_at == checkpoint['actual_updates'] == 2
    assert checkpoint['optimizer']['state'][0]['step'].item() == 2
    assert torch.equal(checkpoint['model'], parameter.detach())
    assert torch.equal(checkpoint['rng'], torch.get_rng_state())
    assert torch.load(tmp_path/'previous.pt', weights_only=False)['actual_updates'] == 1
