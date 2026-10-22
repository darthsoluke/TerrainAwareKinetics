from types import SimpleNamespace
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "out" / "vendor" / "kimodo"))

from scripts.review_native_mb_stair_tokenizer import reconstruct


class _RecordingRepresentation:
    def __init__(self):
        self.dual_rep = self
        self.local_motion_rep = self
        self.inverse_inputs = []

    def change_first_heading(self, raw, heading, is_normalized, to_normalize):
        assert heading == 0.0
        assert is_normalized is False
        assert to_normalize is False
        return raw + 1.0

    def normalize(self, features):
        return features * 2.0

    def global_to_local(self, features, is_normalized, to_normalize, lengths):
        assert is_normalized is True
        assert to_normalize is True
        assert lengths.tolist() == [65]
        return features[:, :64] + 3.0

    def inverse(self, features, is_normalized, return_all):
        assert is_normalized is True
        assert return_all is True
        self.inverse_inputs.append(features.detach().clone())
        return {"posed_joints": features.detach().clone()}


class _RecordingTokenizer:
    decoder_external_cond_feature_mode = "fixture"

    def __init__(self):
        self.encoded = None
        self.codes = torch.zeros((1, 16, 2), dtype=torch.long)
        self.decode_args = None

    def encode_into_idx(self, local):
        self.encoded = local.detach().clone()
        return self.codes

    def forward_decoder(self, codes, **kwargs):
        self.decode_args = (codes, kwargs)
        return {"recon_state": self.encoded + 0.25}


def test_reconstruct_uses_original_encoded_tokens_frame3_mask_and_raw_decode(
        monkeypatch):
    from motionbricks.helper import data_training_util

    representation = _RecordingRepresentation()
    network = _RecordingTokenizer()
    model = SimpleNamespace(
        motion_rep=representation,
        local_motion_rep=representation,
        pose_net=network,
    )
    raw = torch.arange(65 * 4, dtype=torch.float32).reshape(1, 65, 4)
    external_values = []

    def extract(values, motion_rep, mode):
        assert motion_rep is representation
        assert mode == "fixture"
        external_values.append(values.detach().clone())
        return values[..., :2]

    monkeypatch.setattr(data_training_util, "extract_feature_from_motion_rep", extract)

    source, decoded = reconstruct(model, raw)

    local = (raw + 1.0)[:, :64] * 2.0 + 3.0
    assert torch.equal(network.encoded, local)
    assert len(external_values) == 1
    assert torch.equal(external_values[0], local)
    codes, decode_args = network.decode_args
    assert codes is network.codes
    mask = decode_args["has_target_cond"]
    assert mask.shape == (1, 64)
    assert mask[0].nonzero().flatten().tolist() == [3]
    assert torch.equal(decode_args["target_cond"], local * mask[..., None])
    assert torch.equal(decode_args["external_cond"], local[..., :2])
    assert torch.equal(representation.inverse_inputs[0], local)
    assert torch.equal(representation.inverse_inputs[1], local + 0.25)
    assert torch.equal(source["posed_joints"], local)
    assert torch.equal(decoded["posed_joints"], local + 0.25)


def _native_representation_and_source():
    from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import (
        GlobalRootGlobalJoints,
    )
    from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
    from motionbricks.motionlib.core.utils.stats import Stats

    frames = 65
    skeleton = SOMASkeleton30()
    local_rots = torch.eye(3).expand(frames, skeleton.nbjoints, 3, 3).clone()
    root = torch.zeros(frames, 3)
    root[:, 1] = 0.85 + torch.floor(torch.arange(frames) / 16) * 0.08
    root[:, 0] = torch.arange(frames) * 0.006
    base = GlobalRootGlobalJoints(30.0, skeleton, "somaskel30_dual_root_global_joints")
    features = base({
        "local_joint_rots": local_rots,
        "translation": root,
        "foot_contacts": torch.zeros(frames, 4),
    }, to_normalize=False, lengths=torch.tensor(frames))
    local = base.dual_rep.global_to_local(
        features, is_normalized=False, to_normalize=False,
        lengths=torch.tensor([frames]),
    )
    dual = base.dual_rep.cat_global_and_local(features, local)
    stats = Stats()
    stats.register_from_tensors(
        dual.mean(0), dual.std(0, unbiased=False).clamp_min(0.1)
    )
    representation = GlobalRootGlobalJoints(
        30.0, SOMASkeleton30(), "somaskel30_dual_root_global_joints", stats=stats
    )
    return representation, features[None]


def test_reconstruct_smokes_real_native_global_local_and_inverse_coordinates():
    from motionbricks.data.bones_seed import official_sole_surfaces
    from motionbricks.data.soma_skin_reference import load_official_skinner

    representation, raw = _native_representation_and_source()

    class IdentityIndexTokenizer:
        decoder_external_cond_feature_mode = "root_without_hip_height"

        def encode_into_idx(self, local):
            self.encoded = local.detach().clone()
            return torch.zeros((1, 16, 1), dtype=torch.long, device=local.device)

        def forward_decoder(self, codes, **kwargs):
            self.mask = kwargs["has_target_cond"].detach().clone()
            self.target = kwargs["target_cond"].detach().clone()
            return {"recon_state": self.encoded.clone()}

    network = IdentityIndexTokenizer()
    model = SimpleNamespace(
        motion_rep=representation,
        local_motion_rep=representation.dual_rep.local_motion_rep,
        pose_net=network,
    )

    source, decoded = reconstruct(model, raw)

    assert network.mask[0].nonzero().flatten().tolist() == [3]
    assert torch.equal(network.target, network.encoded * network.mask[..., None])
    torch.testing.assert_close(decoded["posed_joints"], source["posed_joints"])
    canonical = representation.change_first_heading(raw, 0.0, False, False)
    expected = representation.inverse(canonical, False, return_all=True)
    torch.testing.assert_close(
        source["posed_joints"], expected["posed_joints"][:, :64], atol=2e-4, rtol=0
    )
    surfaces, _, _, _ = official_sole_surfaces(representation.skeleton)
    skinner = load_official_skinner(representation.skeleton)
    expected_vertices = skinner.skin(
        expected["local_joint_rots"][0, :64], expected["posed_joints"][0, :64],
        rot_is_global=False,
    )
    reconstructed_vertices = skinner.skin(
        source["local_joint_rots"][0], source["posed_joints"][0], rot_is_global=False
    )
    for surface in surfaces:
        selected = torch.as_tensor(surface["selected_indices"], dtype=torch.long)
        torch.testing.assert_close(
            reconstructed_vertices.index_select(1, selected),
            expected_vertices.index_select(1, selected), atol=3e-3, rtol=0,
        )


def test_reconstruct_matches_original_native_tokenizer_index_decode():
    from motionbricks.data.native_mb_evaluation import (
        full_precision_evaluation,
        verify_tokenizer_decode_parity,
    )
    from motionbricks.data.native_mb_models import build_native_model, native_config
    from motionbricks.helper.data_training_util import extract_feature_from_motion_rep

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(311)
    try:
        representation, raw = _native_representation_and_source()
        model = build_native_model(
            "tokenizer", representation, native_config("tokenizer", max_steps=10)
        ).eval()
        network = model.pose_net
        canonical = representation.change_first_heading(raw, 0.0, False, False)
        local = representation.dual_rep.global_to_local(
            representation.normalize(canonical), True, True,
            lengths=torch.tensor([65]),
        )[:, :64]
        mask = torch.zeros((1, 64), dtype=torch.bool)
        mask[:, 3] = True
        target = local * mask[..., None]
        external = extract_feature_from_motion_rep(
            local, model.local_motion_rep, network.decoder_external_cond_feature_mode
        )

        with full_precision_evaluation(), torch.inference_mode():
            original_forward = network(
                local, target_cond=target, has_target_cond=mask, external_cond=external
            )["recon_state"]
            codes = network.encode_into_idx(local)
            index_decode = network.forward_decoder(
                codes, target_cond=target, has_target_cond=mask, external_cond=external
            )["recon_state"]
            verify_tokenizer_decode_parity(
                network, local, codes, original_forward, index_decode, target, mask
            )
            expected_motion = model.local_motion_rep.inverse(
                index_decode, True, return_all=True
            )

        _, reconstructed = reconstruct(model, raw)
        torch.testing.assert_close(
            reconstructed["posed_joints"], expected_motion["posed_joints"],
            atol=1e-5, rtol=0,
        )
    finally:
        torch.set_num_threads(previous_threads)


def test_reconstruct_rejects_invalid_context_or_window_length():
    model = SimpleNamespace()
    with pytest.raises(ValueError, match="65 source frames"):
        reconstruct(model, torch.zeros((1, 64, 4)))
    with pytest.raises(ValueError, match="indices in 0..3"):
        reconstruct(model, torch.zeros((1, 65, 4)), context_frames=(4,))
