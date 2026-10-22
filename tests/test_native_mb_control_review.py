import hashlib
import json

import numpy as np
import pytest
import torch

from motionbricks.data.native_mb_control_review import evaluate_control


class FakeRep:
    def __init__(self, root_mode):
        self.root_mode = root_mode
        self.fps = 30.0
        self.indices = {"all": np.arange(7), "root": np.arange(5 if root_mode == "global" else 4)}
        self.global_motion_rep = self if root_mode == "global" else None
        self.local_motion_rep = self if root_mode == "local" else None
        self.skeleton = type("Skeleton", (), {"root_idx": 0})()
        self.name = root_mode

    def normalize(self, value):
        return value

    def unnormalize(self, value):
        return value

    def extract_root(self, value):
        return value[..., :len(self.indices["root"])]

    def compute_root_pos_and_rot(self, value, return_quat=True, return_angle=False):
        position = value[..., :3]
        yaw = torch.atan2(value[..., 4], value[..., 3])
        if return_angle and not return_quat:
            return position, yaw
        raise AssertionError("Test only expects positions and yaw")

    def change_first_heading(self, value, *args, **kwargs):
        return value

    def global_to_local(self, value, **kwargs):
        return value

    def inverse(self, value, is_normalized, return_all):
        joints = value[..., :3].unsqueeze(-2).expand(-1, -1, 30, -1).clone()
        return {"posed_joints": joints}


class FakeDual:
    def __init__(self):
        self.global_motion_rep = FakeRep("global")
        self.local_motion_rep = FakeRep("local")

    def global_to_local(self, value, **kwargs):
        return value


class FakeMotionRep:
    def __init__(self):
        self.fps = 30.0
        self.dual_rep = FakeDual()
        self.skeleton = type("Skeleton", (), {"root_idx": 0})()

    def change_first_heading(self, value, *args, **kwargs):
        return value

    def normalize(self, value):
        return value


class FakeRootBackbone:
    MASKED_NUM_TOKENS = 99
    _args = {"min_tokens": 6, "max_tokens": 16}

    def get_num_frames_per_token(self):
        return 4

    def __call__(self, global_root, has_global, local_root, has_local,
                 local_pose, has_pose, num_tokens, allowed_pred_num_tokens=None, config=None, **kwargs):
        assert num_tokens.tolist() == [[self.MASKED_NUM_TOKENS]]
        assert global_root.shape == (1, 8, 5)
        if not has_global.all():
            assert has_global[0].nonzero().flatten().tolist() == [3, 7]
            if has_pose.any():
                assert has_pose[0].nonzero().flatten().tolist() == [3, 7]
                assert torch.equal(has_local, has_pose)
            else:
                assert not has_local.any()
        else:
            assert not has_local.any()
            assert not has_pose.any()
        assert config["allow_pred_out_of_reach_num_tokens"] is False
        if allowed_pred_num_tokens is None:
            predicted_tokens = 12
        else:
            assert allowed_pred_num_tokens.shape == (1, 11)
            assert allowed_pred_num_tokens[0].nonzero().flatten().tolist() == [10]
            predicted_tokens = 16
        prediction = torch.rand((1, 64, 5), dtype=global_root.dtype, device=global_root.device)
        prediction[:, :4] = global_root[:, :4]
        prediction[:, -4:] = global_root[:, -4:]
        return {"pred_global_root_values": prediction,
                "pred_num_tokens": torch.tensor([predicted_tokens], dtype=torch.long,
                                                  device=global_root.device)}


class FakeRootModel:
    def __init__(self):
        self.motion_rep = FakeMotionRep()
        self.backbone_net = FakeRootBackbone()
        self.args = {"local_root_feature": "root", "local_pose_feature": "pose",
                     "prob_provide_num_tokens": 0.0}
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self


class FakePoseBackbone:
    POSE_MASK_ID = 8

    def get_num_frames_per_token(self):
        return 4

    def get_num_heads(self):
        return 2, 1

    def __call__(self, pose_tokens, root_cond, pose_cond, has_pose_cond, num_tokens,
                 text_embeddings=None, has_text_embeddings=None):
        assert pose_tokens.shape == (1, 16, 2)
        assert torch.all(pose_tokens == self.POSE_MASK_ID)
        assert has_pose_cond[0].nonzero().flatten().tolist() in (
            [0, 1, 2, 3], [0, 1, 2, 3, 60, 61, 62, 63], [3, 63]
        )
        assert num_tokens.tolist() == [[16]]
        logits = torch.zeros(1, 16, 2, 3)
        logits[..., 2] = 10
        return {"pose_logits": logits}


class FakePoseVQVAE(torch.nn.Module):
    motion_rep = FakeRep("local")
    decoder_external_cond_feature_mode = "root"

    def __init__(self):
        super().__init__()
        self.register_buffer("device_marker", torch.zeros(1))

    def encode_into_idx(self, *args, **kwargs):
        raise AssertionError("Held-out future tokens must not be encoded from source")

    def forward_decoder(self, tokens, target_cond, has_target_cond, external_cond,
                        use_overall_indices, token_mask):
        assert tokens.device == self.device_marker.device
        assert tokens.shape == (1, 16, 2)
        assert not torch.any(tokens == 8)
        assert torch.all(tokens == 2)
        assert has_target_cond[0].nonzero().flatten().tolist() in (
            [0, 1, 2, 3], [0, 1, 2, 3, 60, 61, 62, 63], [3, 63]
        )
        assert token_mask.all()
        state = torch.zeros((1, 64, 7), dtype=target_cond.dtype, device=target_cond.device)
        state[..., 0] = 2.0
        return {"recon_state": state}


class FakePoseModel:
    def __init__(self, tokenizer_loaded=True):
        self.motion_rep = FakeMotionRep()
        self.backbone_net = FakePoseBackbone()
        self.supporting_nets = {"pose_net": FakePoseVQVAE()}
        self.tokenizer_loaded = tokenizer_loaded
        self.support_device_moves = []
        self.args = {
            "cond_root_feature_is_from_motion_rep": "local",
            "cond_root_feature": "root",
            "local_pose_feature": "pose",
            "no_end_keyframe_prob": 0.0,
            "keyframe_num_warmup_steps": 200000,
        }

    def eval(self):
        return self

    def move_supporting_nets_to_device(self, device):
        self.support_device_moves.append(str(device))
        self.supporting_nets["pose_net"].to(device).eval()

    @property
    def vqvae_model_loaded(self):
        return self.tokenizer_loaded


def _dataset(path, validation_records=19, finite=True, behaviors=5):
    path.mkdir()
    rows = []
    for index in range(validation_records):
        shard = f"shard-{index:02}.npz"
        features = np.zeros((75, 7), dtype=np.float32)
        features[:, 0] = np.linspace(0, 1, 75) + index
        features[:, 1] = 1.0
        features[:, 3] = 1.0
        if not finite and index == 0:
            features[0, 0] = np.nan
        np.savez(path / shard, global_features=features)
        rows.append({"shard": shard, "split": "validation", "behavior": f"behavior-{index % behaviors}"})
    rows.append({"shard": "ignored-train.npz", "split": "train", "behavior": "train-only"})
    accepted_path = path / "accepted.json"
    accepted_path.write_text(json.dumps(rows), encoding="utf-8")
    artifacts = {"accepted.json": hashlib.sha256(accepted_path.read_bytes()).hexdigest()}
    for row in rows:
        shard_path = path / row["shard"]
        if shard_path.is_file():
            artifacts[row["shard"]] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    (path / "native_manifest.json").write_text(
        json.dumps({"artifact_sha256": artifacts}), encoding="utf-8"
    )
    return path


def test_root_evaluation_uses_exact_sparse_conditions_and_only_validation(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    report = evaluate_control(FakeRootModel(), "root", dataset, tmp_path / "review")

    assert len(report["windows"]) == 57
    assert report["conditions"] == {
        "global_root_frames": [0, 1, 2, 3, 60, 61, 62, 63],
        "local_root_frames": [], "pose_frames": [], "future_body_used": False,
    }
    assert all(row["shard"].startswith("shard-") for row in report["windows"])
    assert all(len(row["source_window_sha256"]) == 64 for row in report["windows"])
    assert report["summary"]["fixed_horizon_16"]["predicted_duration_seconds"] == pytest.approx(64 / 30)
    assert report["summary"]["implicit_duration_diagnostic"]["predicted_duration_seconds"] == pytest.approx(48 / 30)
    assert all(row["fixed_horizon_16"]["scored_frames_inclusive"] == [4, 63]
               for row in report["windows"])
    assert all(row["implicit_duration_diagnostic"]["scored_frames_inclusive"] == [4, 47]
               for row in report["windows"])
    assert len(report["evaluator_source_sha256"]) == 64
    assert report["additional_mode_conditions"]["native_keyframe1"]["future_body_used"]
    assert all("native_keyframe1" in row and "root_only_keyframe1" in row for row in report["windows"])
    saved = json.loads((tmp_path / "review" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["stage"] == "root"


def test_root_evaluation_is_deterministic_for_same_seed(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    first = evaluate_control(FakeRootModel(), "root", dataset, tmp_path / "first", seed=91)
    second = evaluate_control(FakeRootModel(), "root", dataset, tmp_path / "second", seed=91)
    assert first["summary"] == second["summary"]
    assert first["windows"] == second["windows"]


def test_pose_generates_future_tokens_and_saves_five_renderer_arrays(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    first = evaluate_control(FakePoseModel(), "pose", dataset, tmp_path / "pose-a", seed=19)
    second = evaluate_control(FakePoseModel(), "pose", dataset, tmp_path / "pose-b", seed=19)

    assert first["conditions"]["pose_modes"]["context4_stress_probe"] == [0, 1, 2, 3]
    assert first["conditions"]["generated_pose_tokens_per_mode"] == 16
    assert first["conditions"]["source_encoded_future_tokens"] is False
    assert len(first["examples"]) == 5
    arrays = sorted((tmp_path / "pose-a").glob("*_reconstruction.npz"))
    assert len(arrays) == 5
    for path in arrays:
        with np.load(path, allow_pickle=False) as data:
            assert data["source"].shape == data["prediction"].shape == (64, 30, 3)
            assert np.isfinite(data["source"]).all()
            assert np.isfinite(data["prediction"]).all()
    assert first["summary"] == second["summary"]
    assert "out of distribution" in first["interpretation"]
    assert len(first["windows"]) == first["windows_per_mode"] == 57
    assert all(set(row["modes"]) == {"context4_stress_probe", "inbetween_endpoint_pose", "native_keyframe1_inbetween"}
               for row in first["windows"])
    assert len(list((tmp_path / "pose-a" / "native_keyframe1_inbetween").glob("*_reconstruction.npz"))) == 5
    assert json.loads((tmp_path / "pose-a" / "native_keyframe1_inbetween" / "metrics.json").read_text())["render_mode"] == "native_keyframe1_inbetween"
    for row in first["windows"]:
        assert "joint_velocity_error_mps" in row["modes"]["context4_stress_probe"]
        assert row["modes"]["inbetween_endpoint_pose"]["unconditioned_interior"][
            "scored_frames_inclusive"
        ] == [4, 59]
    assert len(first["evaluator_source_sha256"]) == 64


def test_manifest_controls_window_count_not_the_old_pilot_size(tmp_path):
    dataset = _dataset(tmp_path / "stairs", validation_records=13)
    report = evaluate_control(FakeRootModel(), "root", dataset, tmp_path / "review")
    assert len(report['windows']) == 39


def test_pose_supports_four_manifest_behaviors_and_39_windows(tmp_path):
    dataset = _dataset(tmp_path / 'stairs', validation_records=13, behaviors=4)
    report = evaluate_control(FakePoseModel(), 'pose', dataset, tmp_path / 'review')
    assert len(report['windows']) == report['windows_per_mode'] == 39
    for mode in report['summary']:
        assert len(list((tmp_path / 'review' / mode).glob('*_reconstruction.npz'))) == 4


def test_rejects_empty_validation_and_nonfinite_sources_before_output(tmp_path):
    short_dataset = _dataset(tmp_path / "short", validation_records=0)
    with pytest.raises(ValueError, match="No eligible validation"):
        evaluate_control(FakeRootModel(), "root", short_dataset, tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()

    bad_dataset = _dataset(tmp_path / "bad", finite=False)
    with pytest.raises(ValueError, match="nonfinite"):
        evaluate_control(FakeRootModel(), "root", bad_dataset, tmp_path / "also-not-created")
    assert not (tmp_path / "also-not-created").exists()


def test_rejects_missing_validation_shard_instead_of_shrinking_count(tmp_path):
    dataset = _dataset(tmp_path / 'dataset', validation_records=13)
    (dataset / 'shard-00.npz').unlink()
    with pytest.raises(ValueError, match='Validation shard is missing'):
        evaluate_control(FakeRootModel(), 'root', dataset, tmp_path / 'review')
    assert not (tmp_path / 'review').exists()


def test_rejects_a_shard_assigned_to_multiple_splits(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    accepted_path = dataset / "accepted.json"
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    accepted.append({"shard": "shard-00.npz", "split": "train", "behavior": "leaked"})
    accepted_path.write_text(json.dumps(accepted), encoding="utf-8")
    manifest_path = dataset / "native_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"]["accepted.json"] = hashlib.sha256(accepted_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        evaluate_control(FakeRootModel(), "root", dataset, tmp_path / "review")
    assert not (tmp_path / "review").exists()


def test_pose_requires_verified_tokenizer_loaded_flag(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    with pytest.raises(ValueError, match="verified tokenizer binding"):
        evaluate_control(FakePoseModel(tokenizer_loaded=False), "pose", dataset,
                         tmp_path / "review")
    assert not (tmp_path / "review").exists()


@pytest.mark.parametrize("support_device", ["cpu", "cuda"])
def test_pose_review_resynchronizes_unregistered_support_after_teardown(tmp_path, support_device):
    if support_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required to reproduce Lightning's split-device teardown")
    model = FakePoseModel()
    model.supporting_nets["pose_net"].to(support_device).train()
    evaluate_control(model, "pose", _dataset(tmp_path / "dataset"), tmp_path / "review")
    assert model.support_device_moves == ["cpu"]
    support = model.supporting_nets["pose_net"]
    assert support.device_marker.device.type == "cpu"
    assert not support.training


def test_rejects_existing_output_unknown_stage_and_invalid_seed(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        evaluate_control(FakeRootModel(), "root", dataset, existing)
    with pytest.raises(ValueError, match="stage"):
        evaluate_control(FakeRootModel(), "tokenizer", dataset, tmp_path / "unknown")
    with pytest.raises(ValueError, match="seed"):
        evaluate_control(FakeRootModel(), "root", dataset, tmp_path / "bad-seed", seed=True)
