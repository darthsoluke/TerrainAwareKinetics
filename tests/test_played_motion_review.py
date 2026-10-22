import torch
from motionbricks.data.played_motion_review import played_window


def test_review_preserves_every_future_value_and_does_not_repair_root():
    history = torch.randn(1, 4, 6)
    prediction = torch.randn(1, 32, 6, requires_grad=True)
    before = prediction.detach().clone()
    actual = played_window(history, prediction)
    torch.testing.assert_close(actual[:, :4], history, atol=0, rtol=0)
    torch.testing.assert_close(actual[:, 4:], prediction[:, 4:28], atol=0, rtol=0)
    torch.testing.assert_close(prediction, before, atol=0, rtol=0)
    actual[:, 4:].sum().backward()
    assert prediction.grad[:, :4].abs().sum() == 0
    assert prediction.grad[:, 4:28].eq(1).all()


def test_past_decoder_velocity_errors_still_affect_published_future():
    velocity = torch.ones(1, 31, 1, requires_grad=True)
    position = torch.cat((torch.zeros(1, 1, 1), velocity.cumsum(1) / 30), 1)
    prediction = torch.cat((position, torch.zeros(1, 32, 5)), -1)
    played_window(torch.zeros(1, 4, 6), prediction)[:, 4:, 0].sum().backward()
    assert velocity.grad[:, :4].gt(0).all()
