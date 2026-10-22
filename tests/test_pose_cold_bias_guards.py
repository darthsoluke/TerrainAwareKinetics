import pytest
import torch

from motionbricks.data.unified_full_training import NativeTrainingStage


class Controller(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.root=torch.nn.Linear(2,2)
        self.pose=torch.nn.Linear(2,2)
        self.codec=torch.nn.Linear(2,2)
        self.adaptation=torch.nn.ModuleDict({'pose':torch.nn.Linear(2,2)})


def test_pose_coalescing_requires_cache_contract():
    with pytest.raises(ValueError,match='memoization'):
        NativeTrainingStage(Controller(),'pose',coalesce_dilated_bias=True)


@pytest.mark.parametrize('mutation',['training','gradient'])
def test_pose_coalescing_refuses_mutable_codec_before_preparation(mutation):
    controller=Controller()
    stage=NativeTrainingStage(controller,'pose',cache_pose_tokens=True,coalesce_dilated_bias=True).train(True)
    assert not controller.codec.training
    assert not any(p.requires_grad for p in controller.codec.parameters())
    if mutation=='training':controller.codec.train(True)
    else:controller.codec.requires_grad_(True)
    with pytest.raises(ValueError,match='eval-mode frozen codec'):
        stage({})
