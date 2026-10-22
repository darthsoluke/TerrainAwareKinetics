"""Focused synthetic checks; no checkpoint execution, training or remote access."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "out" / "vendor" / "kimodo"))

from scripts import review_native_mb_lafan_system as review


def motion(offset=0.0):
    positions = torch.zeros(1, 64, 30, 3)
    positions[..., 0] = offset
    return {"posed_joints": positions,
            "local_joint_rots": torch.eye(3).expand(1, 64, 30, 3, 3).clone()}


def test_free_interior_excludes_context_endpoint_and_boundary_velocity():
    source, generated = motion(), motion()
    generated["posed_joints"][:, :4] += 100
    generated["posed_joints"][:, 63] += 200
    scores = review.score_motion(generated, source, 0, 30)
    assert all(value == 0 for value in scores["free_interior"].values())
    assert scores["endpoint_63"]["mpjpe_m"] == pytest.approx(200 * np.sqrt(3))
    assert "joint_velocity_error_mps" not in scores["endpoint_63"]


def test_actual_predicted_root_is_not_replaced_or_aligned():
    source, generated = motion(), motion(2)
    before = generated["posed_joints"].clone()
    scores = review.score_motion(generated, source, 0, 30)
    assert scores["free_interior"]["root_error_m"] == 2
    assert scores["free_interior"]["mpjpe_m"] == 2
    assert scores["free_interior"]["root_relative_fk_mpjpe_m"] == 0
    torch.testing.assert_close(generated["posed_joints"], before, rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["shape", "nan", "rotation"])
def test_scores_fail_closed_on_invalid_full_body(bad):
    generated = motion()
    if bad == "shape":
        generated["posed_joints"] = generated["posed_joints"][:, :, :29]
    elif bad == "nan":
        generated["posed_joints"][0, 20, 4, 2] = float("nan")
    else:
        generated["local_joint_rots"] = generated["local_joint_rots"][:, :63]
    with pytest.raises(ValueError, match="full-body"):
        review.score_motion(generated, motion(), 0, 30)


def test_raw_acceleration_units_stencils_and_actual_root_displacement():
    joints = np.zeros((64, 30, 3), dtype=np.float64)
    times = np.arange(64) / 30
    joints[..., 0] = times[:, None] ** 2
    result = review.raw_fk_diagnostics(joints, 0, 30)
    assert result["raw_fk_acceleration_mean_mps2"] == pytest.approx(2)
    assert result["raw_fk_acceleration_p95_mps2"] == pytest.approx(2)
    assert result["root_displacement_xyz_m"] == pytest.approx([(63 / 30) ** 2, 0, 0])
    assert result["acceleration_center_frames_inclusive"] == [5, 61]
    changed = joints.copy()
    changed[:4] += 1000
    changed[63] += 2000
    assert review.raw_fk_diagnostics(changed, 0, 30)["raw_fk_acceleration_mean_mps2"] == pytest.approx(2)


@pytest.mark.parametrize("bad", ["short", "nan", "fps", "root"])
def test_diagnostics_reject_invalid_input(bad):
    joints, fps, root = np.zeros((64, 30, 3)), 30, 0
    if bad == "short":
        joints = joints[:63]
    elif bad == "nan":
        joints[10] = np.nan
    elif bad == "fps":
        fps = 0
    else:
        root = 30
    with pytest.raises(ValueError, match="Diagnostics"):
        review.raw_fk_diagnostics(joints, root, fps)


def windows():
    return [{"shard": behavior + ".npz", "start": start, "behavior": behavior,
             "subject_id": "subject4", "split": "validation"}
            for behavior in ("walk", "run") for start in (0, 7, 15)]


@pytest.mark.parametrize("bad", ["short", "duplicate", "heldout", "subject", "behavior"])
def test_exact_six_window_contract(bad):
    selected = windows()
    if bad == "short":
        selected.pop()
    elif bad == "duplicate":
        selected[1] = selected[0].copy()
    elif bad == "heldout":
        selected[0]["split"] = "test"
    elif bad == "subject":
        selected[0]["subject_id"] = "subject5"
    else:
        selected[0]["behavior"] = "idle"
    with pytest.raises(ValueError, match="six unique subject4"):
        review.validate_windows(selected)


def test_fresh_output_outside_every_input(tmp_path):
    protected = tmp_path / "source"
    protected.mkdir()
    with pytest.raises(FileExistsError, match="fresh"):
        review.validate_output(protected, [])
    with pytest.raises(ValueError, match="separate"):
        review.validate_output(protected / "review", [protected])
    parent = tmp_path / "new_parent"
    with pytest.raises(ValueError, match="separate"):
        review.validate_output(parent, [parent / "dataset"])
    assert review.validate_output(tmp_path / "review", [protected]) == tmp_path / "review"


def fake_infer():
    return SimpleNamespace(_root_model=torch.nn.Linear(1, 1),
                           _pose_model=torch.nn.Linear(1, 1),
                           _vqvae_pose_model=torch.nn.Linear(1, 1))


def test_benchmark_full_predict_sync_warmup_count_and_quantiles(monkeypatch):
    infer = fake_infer()
    raws = [torch.full((1, 65, 5), float(index)) for index in range(6)]
    events, captured_flags = [], []
    times = iter(value for index in range(20) for value in (index * 2, index * 2 + .001 * (index + 1)))

    def predict(instance, raw):
        assert instance is infer
        assert not torch.is_grad_enabled()
        events.append(("predict", int(raw[0, 0, 0])))
        captured_flags.append(torch.backends.cuda.matmul.allow_tf32)
        return raw[:, :64].clone(), raw[:, :64].clone()

    original = torch.backends.cuda.matmul.allow_tf32
    result = review.benchmark_prediction(infer, raws, predictor=predict,
        synchronize=lambda: events.append(("sync",)), timer=lambda: next(times))
    assert len(events) == 23 * 3
    assert all(events[i][0] == kind for i in range(0, len(events), 3)
               for kind in ["sync"])
    assert [e[0] for e in events[:3]] == ["sync", "predict", "sync"]
    assert len(captured_flags) == 23 and not any(captured_flags)
    assert torch.backends.cuda.matmul.allow_tf32 == original
    assert result["latency_ms"]["p50"] == pytest.approx(10.5)
    assert result["latency_ms"]["p95"] == pytest.approx(19.05)
    assert result["latency_ms"]["max"] == pytest.approx(20)
    assert result["sample_window_indices"] == [i % 6 for i in range(3, 23)]
    assert result["output_horizon_frames"] == 64
    assert result["hardware"]["device"] == "cpu"
    assert result["precision"]["input_dtype"] == "torch.float32"
    assert result["precision"]["parameter_dtypes"]["pose"] == ["torch.float32"]
    assert not result["ue_measured"] and not result["rtx3080_target_measured"]


def test_benchmark_rejects_wrong_output_horizon():
    raws = [torch.zeros(1, 65, 5)] * 6
    with pytest.raises(ValueError, match="horizon64"):
        review.benchmark_prediction(fake_infer(), raws,
            predictor=lambda infer, raw: (raw[:, :64], raw[:, :63]))


def test_saved_three_panel_full_body_arrays_are_unmodified(tmp_path):
    motions = {"source": motion(), "tokenizer": motion(1), "generated": motion(2)}
    calls = []
    artifacts = review.save_window(tmp_path, 5, windows()[5], motions, 30,
                                  renderer=lambda *args, **kwargs: calls.append((args, kwargs)))
    assert set(artifacts) == {"npz", "png", "gif"}
    assert calls[0][1]["three_panel"] is True
    assert calls[0][1]["panel_labels"] == ("Source SOMA motion", "Tokenizer reconstruction", "Root/Pose generation")
    with np.load(tmp_path / artifacts["npz"], allow_pickle=False) as data:
        for name, expected in motions.items():
            np.testing.assert_array_equal(data[name], expected["posed_joints"][0].numpy())
            assert data[name].shape == (64, 30, 3)
            np.testing.assert_array_equal(data[name + "_local_rot_mats"], expected["local_joint_rots"][0].numpy())


def test_missing_floor_provenance_is_not_fake_zero(tmp_path):
    source, dataset = tmp_path / "source", tmp_path / "dataset"
    source.mkdir()
    dataset.mkdir()
    (source / "accepted.json").write_text("[]")
    (dataset / "native_manifest.json").write_text("{}")
    setup, status = review.prepare_sole_review(source, dataset, None)
    assert setup is None and status["status"] == "not_assessed"
    assert "penetration_mean_m" not in status


@pytest.mark.parametrize("airborne", [False, True])
def test_sole_slip_uses_source_fixed_vertex_mask_not_prediction_contacts(airborne):
    class Skin:
        def skin(self, rotations, roots, rot_is_global):
            vertices = torch.zeros(64, 2, 3)
            vertices[:, 1, 1] = 1
            assert roots.shape == (64, 30, 3)
            if roots[0, 0, 0] != 0:
                vertices[:, 0, 0] = torch.arange(64) * .1
                vertices[:, 0, 1] = 2 if airborne else -.02
                vertices[:, 1, 0] = torch.arange(64) * 10
                vertices[:, 1, 1] = -.05
            return vertices

    result = review.sole_diagnostics({"source": motion(), "tokenizer": motion(1),
                                    "generated": motion(2)}, 0, 30,
                                   (Skin(), {"Left": np.array([0, 1])}))
    foot = result["Left"]
    assert foot["source_measured_vertex_intervals"] == 58
    assert foot["metrics"]["source"]["slip_mean_mps"] == 0
    assert foot["metrics"]["generated"]["slip_mean_mps"] == pytest.approx(3)
    assert foot["metrics"]["generated"]["penetration_max_m"] == pytest.approx(.05)


def test_no_source_contact_means_slip_not_assessed():
    class Skin:
        def skin(self, *args, **kwargs):
            return torch.ones(64, 1, 3)

    result = review.sole_diagnostics({"source": motion(), "generated": motion(1)}, 0, 30,
                                   (Skin(), {"Left": np.array([0])}))
    assert result["Left"]["metrics"]["generated"]["slip_status"] == "not_assessed"
    assert result["Left"]["metrics"]["generated"]["slip_mean_mps"] is None


def test_real_representation_tokenizer_source_matches_canonical_global_fk():
    from test_native_mb_stage_training import _motion_and_representation
    rep, batch = _motion_and_representation(75)
    raw = rep.unnormalize(batch["motion"][:, :65])
    raw[:, :, 0] = 3 + torch.arange(65) * .01

    class Network:
        decoder_external_cond_feature_mode = "root_without_hip_height_without_heading"

        def __call__(self, local, target_cond, has_target_cond, external_cond):
            assert target_cond is None and has_target_cond is None
            assert external_cond.shape == (1, 64, 2)
            return {"recon_state": local.clone()}

    model = SimpleNamespace(motion_rep=rep, local_motion_rep=rep.dual_rep.local_motion_rep,
                            pose_net=Network())
    source, reconstructed = review.tokenizer_reconstruction(model, raw)
    canonical = rep.change_first_heading(raw, 0.0, False, False)
    expected = rep.inverse(canonical[:, :64], False, return_all=True)
    for name in source:
        if name in ("posed_joints", "local_joint_rots"):
            torch.testing.assert_close(source[name], expected[name], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(source[name], reconstructed[name], rtol=0, atol=0)


@pytest.mark.parametrize("guard", [None, "tokenizer_dataset", "bundle_dataset", "tokenizer_hash", "source_parity"])
def test_end_to_end_six_windows_bindings_all_artifacts_and_read_only_inputs(tmp_path, monkeypatch, guard):
    source, dataset = tmp_path / "source", tmp_path / "dataset"
    source.mkdir()
    dataset.mkdir()
    source_rows, native_rows = [], []
    for behavior in ("walk", "run"):
        shard = behavior + ".npz"
        (source / shard).write_bytes(behavior.encode())
        np.savez(dataset / shard, global_features=np.zeros((80, 5), dtype=np.float32))
        source_rows.append({"shard": shard, "filename": behavior, "subject_id": "subject4"})
        native_rows.append({"shard": shard, "source_shard": shard, "filename": behavior,
                            "behavior": behavior, "split": "validation"})
    (source / "accepted.json").write_text(json.dumps(source_rows))
    (dataset / "accepted.json").write_text(json.dumps(native_rows))
    (dataset / "native_manifest.json").write_text("{}")
    input_bytes = {path: path.read_bytes() for folder in (source, dataset) for path in folder.iterdir()}
    digest = review._sha256(dataset / "native_manifest.json")
    info = {"checkpoint_sha256": "tokenizer-hash", "dataset_manifest_sha256": digest,
            "run_status_at_review": "completed", "global_step": 50000}
    model = torch.nn.Module()
    model.pose_net = torch.nn.Linear(1, 1)
    infer = fake_infer()
    infer.eval = lambda: infer
    infer.global_motion_rep = SimpleNamespace(skeleton=SimpleNamespace(root_idx=0), fps=30,
                                               inverse=lambda features, *args, **kwargs: motion(float(features[0, 0, 0])))
    bundle = {"dataset_manifest_sha256": digest, "tokenizer": {"checkpoint_sha256": "tokenizer-hash"},
              "root_step": 50000, "pose_step": 50000}
    if guard == "tokenizer_dataset":
        info["dataset_manifest_sha256"] = "wrong-dataset"
    elif guard == "bundle_dataset":
        bundle["dataset_manifest_sha256"] = "wrong-dataset"
    elif guard == "tokenizer_hash":
        bundle["tokenizer"]["checkpoint_sha256"] = "wrong-tokenizer"
    monkeypatch.setattr(review, "load_native_checkpoint", lambda *args, **kwargs:
                        (model, info, {"dataset_dir": dataset}))
    monkeypatch.setattr(review, "state_digest", lambda model: "frozen-digest")
    monkeypatch.setattr(review, "_validate_frozen_tokenizer", lambda *args: None)
    monkeypatch.setattr(review, "tokenizer_reconstruction", lambda *args:
                        (motion(5 if guard == "source_parity" else 0), motion(1)))
    monkeypatch.setattr(review, "load_bundle", lambda *args: (infer, dataset, bundle))
    monkeypatch.setattr(review, "prepare_sole_review", lambda *args:
                        (None, {"status": "not_assessed", "reason": "fixture has no audited floor"}))
    calls = []

    def predict(instance, raw):
        calls.append(raw.clone())
        return raw[:, :64].clone(), raw[:, :64].clone() + 2

    monkeypatch.setattr(review, "predict_window", predict)
    original_save = review.save_window

    def render(path, motions, fps, folder, title, **kwargs):
        path.with_suffix(".png").write_bytes(b"synthetic-png")
        path.with_suffix(".gif").write_bytes(b"synthetic-gif")

    monkeypatch.setattr(review, "save_window", lambda *args: original_save(*args, renderer=render))
    if guard is not None:
        with pytest.raises(ValueError, match="Tokenizer must bind|identity differs|source FK disagree"):
            review.review_system(source, dataset, tmp_path / "tokenizer", tmp_path / "root",
                                 tmp_path / "pose", tmp_path / "review", device="cpu")
        assert not (tmp_path / "review" / "report.json").exists()
        assert all(path.read_bytes() == before for path, before in input_bytes.items())
        return
    report = review.review_system(source, dataset, tmp_path / "tokenizer", tmp_path / "root",
                                  tmp_path / "pose", tmp_path / "review", device="cpu")
    assert len(calls) == 6 + 3 + 20
    assert report["window_count"] == 6
    assert report["free_interior_frames_inclusive"] == [4, 62]
    assert report["endpoint_frame"] == 63
    assert report["slip_and_penetration"]["status"] == "not_assessed"
    assert not report["conditions"]["source_interior_body_conditioning_for_generation"]
    assert not report["root_overwrite"] and not report["smoothing"] and not report["ik"]
    assert len(list((tmp_path / "review").glob("*.npz"))) == 6
    assert len(list((tmp_path / "review").glob("*.png"))) == 6
    assert len(list((tmp_path / "review").glob("*.gif"))) == 6
    assert report["windows"][0]["scores"]["generated"]["free_interior"]["root_error_m"] == 2
    assert (tmp_path / "review" / "report.json").is_file()
    assert all(path.read_bytes() == before for path, before in input_bytes.items())


def test_official_skin_floor_provenance_and_full_position_api(tmp_path):
    from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30

    source, dataset = tmp_path / "source", tmp_path / "dataset"
    source.mkdir()
    dataset.mkdir()
    (source / "accepted.json").write_text(json.dumps([
        {"shard": "walk.npz", "fixed_plane_source_y_m": .04}]))
    manifest = {"source_manifest_sha256": review._sha256(source / "accepted.json"),
                "contract": {"ground": "source fixed ground plane retained; no per-frame fitting or vertical correction"}}
    (dataset / "native_manifest.json").write_text(json.dumps(manifest))
    skeleton = SOMASkeleton30()
    setup, status = review.prepare_sole_review(source, dataset, skeleton)
    assert setup is not None and status["status"] == "assessed"
    assert status["ground_height_m"] == 0
    assert status["source_fixed_plane_y_m"] == {"walk.npz": .04}
    from motionbricks.data.soma_skin_reference import skinned_vertices
    rotations = torch.eye(3).expand(4, 30, 3, 3).clone()
    _, positions, _ = skeleton.fk(rotations, torch.zeros(4, 3))
    vertices = skinned_vertices(setup[0], rotations, positions)
    assert vertices.shape[0] == 4 and vertices.shape[-1] == 3
    assert torch.isfinite(vertices).all()
    assert all(max(indices) < vertices.shape[1] for indices in setup[1].values())
    manifest["source_manifest_sha256"] = "changed"
    (dataset / "native_manifest.json").write_text(json.dumps(manifest))
    missing, status = review.prepare_sole_review(source, dataset, skeleton)
    assert missing is None and status["status"] == "not_assessed"


def test_existing_renderer_three_panel_png_gif_smoke(tmp_path):
    from scripts.render_native_mb_review import _render_motion_set
    from PIL import Image

    joints = np.zeros((3, 30, 3))
    joints[..., 1] = np.linspace(0, 1.7, 30)
    joints[..., 0] = np.arange(3)[:, None] * .1
    result = _render_motion_set(tmp_path / "smoke.npz", (joints, joints + .1, joints + .2),
                                30, tmp_path, "Three-panel smoke", three_panel=True,
                                panel_labels=("Source", "Tokenizer", "Generated"))
    assert result == "smoke.gif"
    assert (tmp_path / "smoke.png").is_file()
    with Image.open(tmp_path / result) as image:
        assert image.n_frames == 3
        assert image.width > 1000
