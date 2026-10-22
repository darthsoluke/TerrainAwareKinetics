import torch

from motionbricks.data.source_episode_control import source_episode_task


def test_real_source_task_preserves_release_anchor_then_restarts():
    source = torch.zeros(1, 28, 5)
    source[..., 3] = 1
    source[0, :, 0] = torch.arange(28.) / 10
    controls = torch.tensor([[[1., 0., 1., 0.], [0., 0., 1., 0.],
                              [0., 0., 1., 0.], [2., 0., 1., 0.]]])
    torch.testing.assert_close(source_episode_task(source, controls, 2)[0, 0], source[0, 11, 0])
    expected = source[0, 27, 0] + 2*28/30
    torch.testing.assert_close(source_episode_task(source, controls, 3)[0, 0], expected)


def test_source_task_does_not_read_future_body_or_height():
    source = torch.zeros(1, 60, 9)
    source[..., 3] = 1
    controls = torch.tensor([[[0., 0., 1., 0.], [1., 0., 1., 0.]]])
    before = source_episode_task(source, controls, 1)
    source[:, 12:] = 100
    source[..., 1] = 30
    source[..., 5:] = 40
    torch.testing.assert_close(source_episode_task(source, controls, 1), before, atol=0, rtol=0)
