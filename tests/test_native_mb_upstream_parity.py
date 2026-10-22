"""Behavioral parity checks for optional extensions to the upstream tokenizer."""

from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT))

from motionbricks.helper.data_training_util import sample_motion_segments_from_motion_clips
from motionbricks.vqvae.models.motion_vqvae import MotionVQVAEModel


def _head_motion_vqvae_class():
    source = subprocess.run(
        ["git", "show", "HEAD:motionbricks/motionbricks/vqvae/models/motion_vqvae.py"],
        cwd=REPO_ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    namespace = {"__name__": "_native_mb_upstream_motion_vqvae"}
    exec(compile(source, "HEAD:motion_vqvae.py", "exec"), namespace)
    return namespace["MotionVQVAEModel"]


class _TinyMotionRep:
    def __init__(self, name="local"):
        self.name = name
        self.num_joints = 2
        self.indices = {
            "all": torch.arange(8),
            "local_vel": torch.arange(6),
            "foot_contacts": torch.tensor([6, 7]),
        }
        self.stats = SimpleNamespace(mean=torch.zeros(6), std=torch.ones(6))

    def inverse(self, state, **kwargs):
        positions = state[..., :6].reshape(*state.shape[:2], 2, 3)
        if kwargs.get("return_all", False):
            return {"posed_joints": positions, "foot_contacts": state[..., 6:8].sigmoid()}
        return {"posed_joints": positions}


def _make_model(model_class):
    local_rep = _TinyMotionRep()
    global_rep = _TinyMotionRep("global")
    skeleton = SimpleNamespace(foot_joint_idx=torch.tensor([0, 1]))
    motion_rep = SimpleNamespace(
        dual_rep=SimpleNamespace(local_motion_rep=local_rep, global_motion_rep=global_rep),
        num_joints=2,
        skeleton=skeleton,
        fps=30,
    )
    pose_net = SimpleNamespace(motion_rep=local_rep)
    return model_class(
        pose_vqvae_network=pose_net,
        root_vqvae_network=None,
        motion_rep=motion_rep,
        optimizer=None,
        scheduler=None,
        args={"commit_loss_coeff": 0.02, "joint_vel_loss_coeff": 2.0, "skate_contact_loss_coeff": 0.01},
    ), local_rep, global_rep


def _loss_inputs(local_rep, global_rep, recon):
    torch.manual_seed(413)
    target = torch.randn(2, 7, 8) * 0.15
    return {
        "local_motions": target,
        "global_motions": target.clone(),
    }, {
        "recon_state": recon,
        "perplexity": torch.tensor(3.25),
        "l_commit": torch.tensor(0.125),
    }


def test_base_tokenizer_loss_and_prediction_gradient_match_committed_head_without_custom_keys():
    baseline_class = _head_motion_vqvae_class()
    baseline, baseline_local, baseline_global = _make_model(baseline_class)
    current, current_local, current_global = _make_model(MotionVQVAEModel)

    torch.manual_seed(719)
    prediction = torch.randn(2, 7, 8) * 0.2
    baseline_prediction = prediction.clone().requires_grad_(True)
    current_prediction = prediction.clone().requires_grad_(True)
    baseline_batch, baseline_output = _loss_inputs(baseline_local, baseline_global, baseline_prediction)
    current_batch, current_output = _loss_inputs(current_local, current_global, current_prediction)

    baseline_losses = baseline.loss(baseline_batch, baseline_output, None)
    current_losses = current.loss(current_batch, current_output, None)
    assert current_losses.keys() == baseline_losses.keys()
    for name in baseline_losses:
        expected, actual = baseline_losses[name], current_losses[name]
        if torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7, msg=name)
        else:
            assert actual == expected, name

    baseline_gradient = torch.autograd.grad(baseline_losses["loss"], baseline_prediction)[0]
    current_gradient = torch.autograd.grad(current_losses["loss"], current_prediction)[0]
    torch.testing.assert_close(current_gradient, baseline_gradient, rtol=1e-6, atol=1e-7)


def test_segment_sampling_matches_committed_head_and_metadata_tracks_sampled_rows():
    namespace = {"__name__": "_native_mb_upstream_data_training_util"}
    source = subprocess.run(
        ["git", "show", "HEAD:motionbricks/motionbricks/helper/data_training_util.py"],
        cwd=REPO_ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    exec(compile(source, "HEAD:data_training_util.py", "exec"), namespace)
    original_sampler = namespace["sample_motion_segments_from_motion_clips"]

    motions = torch.arange(3 * 15 * 2, dtype=torch.float32).reshape(3, 15, 2)
    lengths = torch.tensor([15, 12, 10])
    np.random.seed(37)
    torch.manual_seed(89)
    original = original_sampler(motions, lengths, 6, 2)

    info = {}
    np.random.seed(37)
    torch.manual_seed(89)
    current = sample_motion_segments_from_motion_clips(motions, lengths, 6, 2, info=info)

    torch.testing.assert_close(current, original, rtol=0, atol=0)
    assert info["chosen_ids"].shape == (current.shape[0],)
    assert info["frame_indices"].shape == (current.shape[0], 7)
    expected = motions[info["chosen_ids"]].gather(
        1, info["frame_indices"][..., None].expand(-1, -1, motions.shape[-1])
    )
    torch.testing.assert_close(current, expected, rtol=0, atol=0)
