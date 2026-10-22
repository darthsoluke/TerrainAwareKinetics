from types import SimpleNamespace

import pytest
import torch

from motionbricks.data.unified_full_training import NativeTrainingStage


class FrozenCodec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.calls = 0
        self.eval()

    def encode_into_idx(self, local, fetch_overall_indices):
        assert fetch_overall_indices is False
        self.calls += 1
        return local.long().permute(2, 0, 1).contiguous().permute(1, 2, 0)


def stage_fixture():
    stage = object.__new__(NativeTrainingStage)
    torch.nn.Module.__init__(stage)
    stage.controller = SimpleNamespace(codec=FrozenCodec())
    stage.cache_pose_tokens = True
    stage._pose_tokens = {}
    stage._pose_token_layouts = {}
    stage.token_cache_hits = stage.token_cache_misses = 0
    return stage


def test_shuffled_cache_preserves_native_tensor_layout_and_values():
    stage = stage_fixture()
    local = torch.arange(4*8*3).reshape(4, 8, 3).float()
    ids = torch.tensor([10, 20, 30, 40])
    first = stage.pose_tokens(dict(sample_ids=ids), local)
    order = torch.tensor([3, 1, 0, 2])
    cached = stage.pose_tokens(dict(sample_ids=ids[order]), local[order])
    assert stage.controller.codec.calls == 1
    assert cached.stride() == first.stride() == (8, 1, 32)
    torch.testing.assert_close(cached, local[order].long(), atol=0, rtol=0)
    # An unseen microbatch size must learn its own native layout on a miss.
    short = stage.pose_tokens(dict(sample_ids=ids[:2]), local[:2])
    assert stage.controller.codec.calls == 2
    assert short.stride() == (8, 1, 16)
    repeated = stage.pose_tokens(dict(sample_ids=ids[:2]), local[:2])
    assert stage.controller.codec.calls == 2
    assert repeated.stride() == short.stride()


def test_trainable_or_training_codec_cannot_reuse_labels():
    stage = stage_fixture()
    batch, local = dict(sample_ids=torch.tensor([0])), torch.zeros(1, 8, 3)
    stage.controller.codec.train()
    with pytest.raises(ValueError, match='eval-mode frozen'):
        stage.pose_tokens(batch, local)
    stage.controller.codec.eval().requires_grad_(True)
    with pytest.raises(ValueError, match='eval-mode frozen'):
        stage.pose_tokens(batch, local)


def test_cache_requires_exact_integer_primitive_ids():
    stage = stage_fixture()
    for ids in (None, torch.tensor([.5]), torch.tensor([0, 1])):
        with pytest.raises(ValueError, match='immutable primitive IDs'):
            stage.pose_tokens(dict(sample_ids=ids), torch.zeros(1, 8, 3))
