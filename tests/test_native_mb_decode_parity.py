"""Numeric parity of the shipped quantizer, not a modified quantizer."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from motionbricks.data.native_mb_evaluation import verify_tokenizer_decode_parity
from motionbricks.vqvae.neural_modules.quantize_cnn_multihead import QuantizeEMAResetMultiHead


class Decoder(nn.Module):
    def forward(self, latent, external, target, mask):
        result = latent * 20000
        if target is not None:
            result = result + (target * mask[..., None]).permute(0, 2, 1)
        return result


class Network(nn.Module):
    encoder_input_feature_mode = "pose"
    decoder_target_cond_feature_mode = "pose"

    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.decoder = Decoder()
        self.quantizer = QuantizeEMAResetMultiHead(
            16, 4, SimpleNamespace(mu=0.99, num_heads=2, kmeans_init=False))
        with torch.no_grad():
            self.quantizer.vq._codebook.embed.fill_(0.01)
        self.eval()

    def extract_feature(self, value, mode):
        return value


def example(context=False):
    network = Network()
    local = torch.full((1, 4, 4), 1.7)
    encoded = local.permute(0, 2, 1)
    indices = network.quantizer.forward_into_idx(encoded, fetch_overall_indices=False)
    quantized = network.quantizer(encoded)[0]
    exact = network.quantizer.dequantize(indices, use_overall_indices=False).permute(0, 2, 1).contiguous()
    target = local if context else None
    mask = torch.tensor([[True, False, False, False]]) if context else None
    prediction = network.decoder(quantized, None, target, mask).permute(0, 2, 1)
    decoded = network.decoder(exact, None, target, mask).permute(0, 2, 1)
    return network, local, indices, prediction, decoded, target, mask


@pytest.mark.parametrize("context", [False, True])
def test_official_float32_straight_through_roundoff_is_proven_not_hidden(context):
    net, local, indices, prediction, decoded, target, mask = example(context)
    assert (prediction - decoded).abs().max() > 1e-5
    result = verify_tokenizer_decode_parity(net, local, indices, prediction, decoded, target, mask)
    assert result["roundoff_replay_required"]
    assert result["token_identity_equal"]
    assert result["straight_through_decoder_replay_error"] == 0
    assert result["token_decoder_replay_error"] == 0


def test_wrong_token_identity_is_rejected_even_when_codes_are_duplicates():
    net, local, indices, pred, decoded, _, _ = example()
    with pytest.raises(ValueError, match="token identities differ"):
        verify_tokenizer_decode_parity(net, local, (indices + 1) % 4, pred, decoded)


def test_decoder_output_damage_is_not_excused_as_roundoff():
    net, local, indices, pred, decoded, _, _ = example()
    with pytest.raises(ValueError, match="decoder replay errors"):
        verify_tokenizer_decode_parity(net, local, indices, pred, decoded + 0.1)


def test_nonfinite_output_is_rejected():
    net, local, indices, pred, decoded, _, _ = example()
    pred[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        verify_tokenizer_decode_parity(net, local, indices, pred, decoded)


def test_training_mode_is_not_allowed_to_update_the_codebook():
    net, local, indices, pred, decoded, _, _ = example()
    net.train()
    with pytest.raises(ValueError, match="eval mode"):
        verify_tokenizer_decode_parity(net, local, indices, pred, decoded)
