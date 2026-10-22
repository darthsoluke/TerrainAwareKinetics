from types import SimpleNamespace

import pytest
import torch
from torch import nn

from motionbricks.scene_control_v2.codec import SceneControlCodec, SomaGeometryAdapter, native_window_transform
from motionbricks.scene_control_v2.quantizer import (
    ExplicitHeadQuantizer, QuantizerConfig, make_token_cache, read_token_cache,
    tensor_state_sha256,
)


@pytest.mark.parametrize("entries", [10, 128, 256])
def test_explicit_heads_forward_decode_and_gradient(entries):
    quantizer = ExplicitHeadQuantizer(QuantizerConfig(entries, code_dim=32)).eval()
    latents = torch.randn(2, 32, 3, requires_grad=True)
    result = quantizer(latents)
    assert result.indices.shape == (2, 3, 8)
    assert result.indices.dtype == torch.int64
    assert result.perplexity.shape == (8,)
    assert result.indices.min() >= 0 and result.indices.max() < entries
    torch.testing.assert_close(quantizer.dequantize(result.indices), result.latents.transpose(1, 2))
    (result.latents.square().mean() + result.commitment).backward()
    assert torch.isfinite(latents.grad).all() and latents.grad.abs().sum() > 0


def test_256_eight_heads_never_pack_a_total_integer():
    quantizer = ExplicitHeadQuantizer(QuantizerConfig(256, code_dim=32)).eval()
    indices = torch.full((2, 3, 8), 255, dtype=torch.int64)
    decoded = quantizer.dequantize(indices)
    assert decoded.shape == (2, 3, 32)
    assert torch.isfinite(decoded).all()
    with pytest.raises(ValueError, match="packed total indices"):
        quantizer.dequantize(torch.zeros(2, 3, dtype=torch.int64))
    with pytest.raises(ValueError, match="outside"):
        quantizer.dequantize(indices + 1)
    with pytest.raises(ValueError, match="outside"):
        quantizer.dequantize(-indices)
    with pytest.raises(TypeError, match="int64"):
        quantizer.dequantize(indices.to(torch.int32))


def test_encoding_does_not_update_ema_and_frozen_codebook_stays_frozen_in_train():
    quantizer = ExplicitHeadQuantizer(QuantizerConfig(10, code_dim=32))
    before = {key: value.clone() for key, value in quantizer.vq.state_dict().items()}
    quantizer.encode(torch.randn(2, 32, 4))
    for key, value in before.items():
        torch.testing.assert_close(value, quantizer.vq.state_dict()[key], rtol=0, atol=0)
    frozen = quantizer.freeze()
    quantizer.train()
    quantizer(torch.randn(2, 32, 4))
    assert frozen == quantizer.codebook_sha256()
    for key, value in before.items():
        torch.testing.assert_close(value, quantizer.vq.state_dict()[key], rtol=0, atol=0)


def test_uninitialized_kmeans_vocabulary_cannot_be_frozen_or_used_for_labels():
    quantizer = ExplicitHeadQuantizer(QuantizerConfig(10, code_dim=32, kmeans_init=True))
    with pytest.raises(RuntimeError, match="uninitialized"):
        quantizer.freeze()
    with pytest.raises(RuntimeError, match="Initialize"):
        quantizer.encode(torch.randn(2, 32, 4))


def test_native_ten_head_vq_inheritance_matches_real_cpu_forward_and_decode():
    from motionbricks.vqvae.neural_modules.quantize_cnn_multihead import QuantizeEMAResetMultiHead
    args = SimpleNamespace(mu=0.99, num_heads=8, kmeans_init=False, calculate_per_head_perplexity=True)
    native = QuantizeEMAResetMultiHead(100000000, 32, args).eval()
    explicit = ExplicitHeadQuantizer(QuantizerConfig(10, code_dim=32)).eval()
    explicit.vq.load_state_dict(native.vq.state_dict(), strict=True)
    latent = torch.randn(2, 32, 4)
    with torch.no_grad():
        native_value, native_commit, native_perplexity = native(latent)
        expected_indices = native.forward_into_idx(latent, fetch_overall_indices=False)
        actual = explicit(latent)
        native_decoded = native.dequantize(expected_indices, use_overall_indices=False)
    torch.testing.assert_close(actual.latents, native_value, rtol=0, atol=0)
    torch.testing.assert_close(actual.commitment, native_commit, rtol=0, atol=0)
    torch.testing.assert_close(actual.perplexity.mean(), native_perplexity, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.indices, expected_indices)
    torch.testing.assert_close(explicit.dequantize(expected_indices), native_decoded, rtol=0, atol=0)


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.ConvTranspose1d(32, 6, 4, stride=4)

    def forward(self, latents, external_cond, target_cond, has_target_cond):
        return self.up(latents)


def make_codec():
    native = SimpleNamespace(
        encoder=nn.Conv1d(6, 32, 4, stride=4), decoder=TinyDecoder(),
        motion_rep=SimpleNamespace(indices={"all": list(range(6))}),
        encoder_input_feature_mode="pose", decoder_target_cond_feature_mode="pose", _down_t=2,
    )
    return SceneControlCodec(native, ExplicitHeadQuantizer(QuantizerConfig(256, code_dim=32)), 5)


def test_codec_forward_decode_cache_and_frozen_parent_chain():
    codec = make_codec().eval()
    motion = torch.randn(2, 8, 6)
    kwargs = dict(target_cond=None, has_target_cond=None, external_cond=None)
    result = codec(motion, **kwargs)
    assert result["recon_state"].shape == motion.shape
    assert result["contact_logits"].shape == (2, 8, 5)
    assert result["indices"].shape == (2, 2, 8)
    with pytest.raises(RuntimeError, match="Freeze"):
        codec.encode(motion)
    normalization = tensor_state_sha256({"mean": torch.zeros(6), "std": torch.ones(6)})
    first = codec.freeze_vocabulary("soma30/local365/stats-test-v1", normalization_sha256=normalization)
    codec.train()
    assert not codec.encoder.training and not codec.quantizer.training
    assert codec.decoder.training
    indices = codec.encode(motion)
    cache = make_token_cache(indices, first)
    torch.testing.assert_close(read_token_cache(cache, first), indices.cpu())
    decoded = codec.decode(indices, **kwargs)
    torch.testing.assert_close(decoded["recon_state"], result["recon_state"])
    codec.decoder.up.bias.data.add_(1)
    unchanged = codec.freeze_vocabulary(first.representation_id, normalization_sha256=normalization, parent=first)
    assert unchanged == first
    codec.encoder.bias.data.add_(0.25)
    changed = codec.freeze_vocabulary(first.representation_id, normalization_sha256=normalization, parent=first)
    assert changed.parent_id == first.identity_id and changed.identity_id != first.identity_id
    with pytest.raises(ValueError, match="regenerate"):
        read_token_cache(cache, changed)
    representation_change = codec.freeze_vocabulary("soma30/new-layout", normalization_sha256=normalization,
                                                    parent=changed)
    assert representation_change.parent_id == changed.identity_id
    assert representation_change.identity_id != changed.identity_id


def test_same_encoder_codebook_and_repr_with_changed_stats_rejects_old_cache():
    codec = make_codec().eval()
    stats = {"mean": torch.zeros(6), "std": torch.ones(6)}
    original_norm = tensor_state_sha256(stats)
    original = codec.freeze_vocabulary("soma30/local-v2", normalization_sha256=original_norm)
    indices = codec.encode(torch.randn(1, 8, 6))
    cache = make_token_cache(indices, original)
    assert cache["normalization_sha256"] == original_norm
    assert cache["vocabulary"]["normalization_sha256"] == original_norm
    stats["std"][0] = 2.0
    changed_norm = tensor_state_sha256(stats)
    changed = codec.freeze_vocabulary("soma30/local-v2", normalization_sha256=changed_norm, parent=original)
    assert original.encoder_sha256 == changed.encoder_sha256
    assert original.codebook_sha256 == changed.codebook_sha256
    assert original.representation_id == changed.representation_id
    assert original.normalization_sha256 != changed.normalization_sha256
    assert original.identity_id != changed.identity_id
    assert changed.parent_id == original.identity_id
    with pytest.raises(ValueError, match="different frozen normalization"):
        read_token_cache(cache, changed)
    new_cache = make_token_cache(indices, changed)
    torch.testing.assert_close(read_token_cache(new_cache, changed), indices.cpu())


@pytest.mark.parametrize("bad_hash", [None, "", "a" * 63, "g" * 64, "A" * 64, 123])
def test_frozen_normalization_hash_requires_64_lowercase_hex(bad_hash):
    codec = make_codec()
    with pytest.raises(ValueError, match="normalization_sha256"):
        codec.freeze_vocabulary("soma30/local-v2", normalization_sha256=bad_hash)
    assert not bool(codec.vocabulary_frozen)
    assert not bool(codec.quantizer.vocabulary_frozen)
    assert all(parameter.requires_grad for parameter in codec.encoder.parameters())


def test_freeze_vocabulary_requires_explicit_normalization_hash():
    with pytest.raises(TypeError, match="normalization_sha256"):
        make_codec().freeze_vocabulary("soma30/local-v2")


@pytest.mark.parametrize("location", ["receipt", "vocabulary"])
def test_token_cache_rejects_tampered_normalization_binding(location):
    codec = make_codec().eval()
    normalization = tensor_state_sha256({"mean": torch.zeros(6), "std": torch.ones(6)})
    identity = codec.freeze_vocabulary("soma30/local-v2", normalization_sha256=normalization)
    cache = make_token_cache(codec.encode(torch.randn(1, 8, 6)), identity)
    if location == "receipt":
        cache["normalization_sha256"] = "f" * 64
    else:
        cache["vocabulary"]["normalization_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="regenerate"):
        read_token_cache(cache, identity)


def test_codec_gradients_reach_encoder_decoder_and_contact_classifier():
    codec = make_codec().eval()
    result = codec(torch.randn(2, 8, 6), target_cond=None, has_target_cond=None, external_cond=None)
    (result["recon_state"].square().mean() + result["contact_logits"].square().mean()
     + result["l_commit"]).backward()
    for module in (codec.encoder, codec.decoder, codec.contact_head):
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                   for parameter in module.parameters())


@pytest.fixture(scope="module")
def soma_seed():
    from motionbricks.data.bones_seed import build_representation
    from motionbricks.data.soma_skin_reference import load_official_skinner
    from motionbricks.motionlib.core.utils.stats import Stats
    representation = build_representation()
    local_rep = representation.dual_rep.local_motion_rep
    local_rep.stats = Stats()
    local_rep.stats.register_from_tensors(torch.zeros(local_rep.motion_rep_dim), torch.ones(local_rep.motion_rep_dim))
    rotations = torch.eye(3).expand(1, 5, 30, 3, 3).clone()
    angles = torch.arange(5) * 0.1 + 0.6
    rotations[0, :, 0, 0, 0] = angles.cos()
    rotations[0, :, 0, 0, 2] = angles.sin()
    rotations[0, :, 0, 2, 0] = -angles.sin()
    rotations[0, :, 0, 2, 2] = angles.cos()
    translation = torch.zeros(1, 5, 3)
    translation[..., 1] = 0.95
    translation[..., 0] = 2.0 + torch.arange(5) * 0.02
    translation[..., 2] = -1.5
    _, source_joints, _ = representation.skeleton.fk(rotations[0], translation[0])
    shifted = source_joints.clone()
    shifted[..., 1] -= 0.4
    unnormalized, initial = local_rep({"local_joint_rots": rotations, "posed_joints": shifted[None]},
                                      to_normalize=False, lengths=torch.tensor([5]), return_init_heading_info=True)
    features = local_rep.normalize(unnormalized).detach()
    skinner = load_official_skinner(representation.skeleton)
    return local_rep, skinner, features, initial, source_joints, rotations


def test_soma_selected_skin_fk_frame_restore_and_gradients(soma_seed):
    local_rep, skinner, seed, initial, source_joints, source_rotations = soma_seed
    indices = torch.tensor([0, 500, 2000, 6000, 12000, 18055])
    adapter = SomaGeometryAdapter(local_rep, skinner, indices)
    features = seed.clone().requires_grad_(True)
    identity = torch.eye(4)[None]
    window = adapter(features, world_from_window=identity)
    native = local_rep.inverse(features, True, return_quat=False, joint_positions_from="global_rot_data")
    torch.testing.assert_close(window["joint_positions_m"], native["posed_joints"])
    full = skinner.skin(native["local_joint_rots"].reshape(-1, 30, 3, 3),
                        native["posed_joints"].reshape(-1, 30, 3))
    torch.testing.assert_close(window["body_samples_m"], full[:, indices].reshape(1, -1, len(indices), 3),
                               atol=2e-6, rtol=1e-5)
    source_restore = native_window_transform(initial, torch.tensor([0.4]))
    source_world = adapter(features, world_from_window=source_restore)
    torch.testing.assert_close(source_world["joint_positions_m"][0], source_joints, atol=2e-6, rtol=1e-5)
    source_skin = skinner.skin(source_rotations[0], source_joints)[:, indices]
    torch.testing.assert_close(source_world["body_samples_m"][0], source_skin, atol=2e-6, rtol=1e-5)
    restore = torch.tensor([[[0., 0., 1., 3.], [0., 1., 0., 0.7],
                             [-1., 0., 0., -2.], [0., 0., 0., 1.]]])
    world = adapter(features, world_from_window=restore)
    for name in ("joint_positions_m", "body_samples_m"):
        expected = torch.einsum("bij,btkj->btki", restore[:, :3, :3], window[name]) + restore[:, None, None, :3, 3]
        torch.testing.assert_close(world[name], expected)
    (world["body_samples_m"].square().mean() + world["joint_positions_m"].square().mean()).backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad[..., local_rep.indices["global_rot_data"]].abs().sum() > 0
    assert features.grad[..., local_rep.indices["root"]].abs().sum() > 0
    assert features.grad[..., local_rep.indices["ric_data"]].abs().sum() == 0
    with pytest.raises(TypeError, match="world_from_window"):
        adapter(features)
