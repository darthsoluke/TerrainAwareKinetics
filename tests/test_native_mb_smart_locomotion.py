"""Source binding, raw playback, and bounded native locomotion comparison."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.native_mb_locomotion import NativeReferenceClips, NativeSmartLocomotion
from scripts.evaluate_native_mb_pipeline import rollout
from scripts.evaluate_native_mb_smart_locomotion import (
    BUFFER_CASES, CASES, CASE_CONDITIONS, capture_inference, evaluate_comparison, load_sources, main, rollout_case, shipped_buffer_step,
    sha256, validate_frozen_bundle, write_json,
)


def _fixture():
    torch.set_num_threads(2)
    rep = build_native_representation()
    rotations = torch.eye(3).expand(12, rep.skeleton.nbjoints, 3, 3).clone()
    translation = torch.zeros(12, 3)
    translation[:, 1] = 1
    raw = rep({"local_joint_rots": rotations, "translation": translation,
               "foot_contacts": torch.ones(12, 4)}, to_normalize=False,
              lengths=torch.tensor(12))[None]
    infer = SimpleNamespace(motion_rep=rep, global_motion_rep=rep,
        local_motion_rep=rep.dual_rep.local_motion_rep, device="cpu",
        EXTERNAL_POSE_FEATURE_MODE="joint_positions_and_rotations",
        _args={"min_tokens": 6, "max_tokens": 16, "down_t": 2},
        _root_model=SimpleNamespace(backbone_net=SimpleNamespace(MASKED_NUM_TOKENS=-1)),
        eval=lambda: None)
    return infer, raw


def _fake_predictions(infer, raw, *, nonfinite=False):
    calls = []

    def predict(*args, **kwargs):
        names = ("global_root_values", "has_global_root_values", "local_root_values", "has_local_root_values",
                 "local_poses", "has_local_poses", "num_tokens")
        kwargs = {**dict(zip(names, args)), **kwargs}
        calls.append(kwargs)
        allowed = kwargs["allowed_pred_num_tokens"][0]
        options = torch.nonzero(allowed).flatten() + infer._args["min_tokens"]
        count = int(options[0] if len(calls) % 2 else options[-1])
        prediction = raw[:, :1].repeat(1, 64, 1)
        # Unrequested jumps must survive in the played output and next history.
        prediction[:, :, 0] = 10 * len(calls) + torch.arange(64)[None] / 30
        if nonfinite:
            prediction[:, 4, 0] = float("nan")
        return prediction, torch.tensor([count])

    infer.predict = predict
    return calls


def _bundle(root_step=45000, pose_step=12000, tokenizer_step=100000):
    return {"root_step": root_step, "pose_step": pose_step,
            "root_checkpoint_sha256": "a" * 64, "pose_checkpoint_sha256": "b" * 64,
            "dataset_manifest_sha256": "c" * 64,
            "tokenizer": {"global_step": tokenizer_step, "checkpoint_sha256": "d" * 64,
                          "dataset_manifest_sha256": "c" * 64}}


@pytest.mark.parametrize("steps", [(45000, 12000, 100000), (100000, 100000, 100000),
                                  (220000, 220000, 100000), (100000, 220000, 100000), (1, 2, 3)])
def test_frozen_bundle_accepts_any_positive_training_steps(steps):
    bundle = _bundle(*steps)
    bundle["root_checkpoint_sha256"] = "ABCDEF0123456789" * 4
    validate_frozen_bundle(bundle)
    assert (bundle["root_step"], bundle["pose_step"], bundle["tokenizer"]["global_step"]) == steps


@pytest.mark.parametrize("field", ["root_step", "pose_step", "global_step"])
@pytest.mark.parametrize("step", [0, -1, True, False, 100000.0, "100000", None, float("nan"), float("inf")])
def test_frozen_bundle_rejects_invalid_steps(field, step):
    bundle = _bundle()
    target = bundle["tokenizer"] if field == "global_step" else bundle
    target[field] = step
    with pytest.raises(ValueError, match="positive integer"):
        validate_frozen_bundle(bundle)


@pytest.mark.parametrize("section,field", [
    (None, "root_checkpoint_sha256"), (None, "pose_checkpoint_sha256"),
    (None, "dataset_manifest_sha256"), ("tokenizer", "checkpoint_sha256"),
    ("tokenizer", "dataset_manifest_sha256"),
])
@pytest.mark.parametrize("digest", [None, 123, "", "a" * 63, "a" * 65, "g" * 64, "a" * 63 + " ",
                                   "0x" + "a" * 62, b"a" * 64])
def test_frozen_bundle_rejects_incomplete_or_nonhex_hashes(section, field, digest):
    bundle = _bundle()
    target = bundle[section] if section else bundle
    target[field] = digest
    with pytest.raises(ValueError, match="64-hex"):
        validate_frozen_bundle(bundle)


def test_frozen_bundle_requires_identical_tokenizer_dataset_binding():
    bundle = _bundle(220000, 220000)
    bundle["tokenizer"]["dataset_manifest_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="dataset binding differ"):
        validate_frozen_bundle(bundle)


def _dataset(tmp_path, raw):
    folder = tmp_path / "dataset"
    folder.mkdir()
    records, artifacts = [], {}
    rows = [("validation", "idle", "heldout_idle")]
    for style in ("idle", "walk", "run", "backward"):
        rows.extend([("train", style, style + "_stop_start"),
                     ("train", style, style + "_loop_first"),
                     ("train", style, style + "_loop_second")])
    rows.append(("train", "idle", "idle_neutral"))
    for index, (split, style, filename) in enumerate(rows):
        shard = filename + ".npz"
        value = raw[0].numpy().copy()
        value[:, 0] += index
        if split == "validation":
            value[4:] = float("nan")  # No validation future is needed or validated.
        np.savez_compressed(folder / shard, global_features=value, fps=np.float32(30))
        digest = sha256(folder / shard)
        artifacts[shard] = digest
        records.append({"split": split, "behavior": style, "filename": filename,
                        "shard": shard, "frames30": len(value), "converted_shard_sha256": digest})
    write_json(folder / "accepted.json", records)
    artifacts["accepted.json"] = sha256(folder / "accepted.json")
    write_json(folder / "native_manifest.json", {"artifact_sha256": artifacts})
    bundle = _bundle()
    digest = sha256(folder / "native_manifest.json")
    bundle["dataset_manifest_sha256"] = digest
    bundle["tokenizer"]["dataset_manifest_sha256"] = digest
    return folder, bundle


def test_source_selection_is_bound_train_only_neutral_then_first_loop(tmp_path):
    infer, raw = _fixture()
    folder, bundle = _dataset(tmp_path, raw)
    seed, references, identity = load_sources(folder, bundle, raw.shape[-1])
    assert seed.shape == (4, raw.shape[-1]) and np.isfinite(seed).all()
    assert list(references) == ["idle", "walk", "run", "backward"]
    assert identity["seed"]["loaded_source_frames_inclusive"] == [0, 3]
    for style, reference in identity["train_references"].items():
        assert reference["split"] == "train"
        assert reference["shard_sha256"] == sha256(folder / reference["shard"])
        assert reference["filename"] == ("idle_neutral" if style == "idle" else style + "_loop_first")
        assert not reference["source_encoded_tokens"]
    write_json(folder / "accepted.json", [])
    with pytest.raises(ValueError, match="accepted.json hash"):
        load_sources(folder, bundle, raw.shape[-1])


def test_reference_shard_tampering_and_cross_split_rejected(tmp_path):
    _, raw = _fixture()
    folder, bundle = _dataset(tmp_path, raw)
    (folder / "walk_loop_first.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="shard hash"):
        load_sources(folder, bundle, raw.shape[-1])
    import json
    records = json.loads((folder / "accepted.json").read_text())
    records.append({**records[1], "split": "validation"})
    write_json(folder / "accepted.json", records)
    manifest = json.loads((folder / "native_manifest.json").read_text())
    manifest["artifact_sha256"]["accepted.json"] = sha256(folder / "accepted.json")
    write_json(folder / "native_manifest.json", manifest)
    bundle["dataset_manifest_sha256"] = sha256(folder / "native_manifest.json")
    bundle["tokenizer"]["dataset_manifest_sha256"] = bundle["dataset_manifest_sha256"]
    with pytest.raises(ValueError, match="multiple splits"):
        load_sources(folder, bundle, raw.shape[-1])


@pytest.mark.parametrize("case", [*CASES, "native_smart4_fixed8"])
def test_raw_playback_preserves_history_and_reports_seam_vs_intra(case):
    infer, raw = _fixture()
    calls = _fake_predictions(infer, raw)
    seed = raw[:, :4]
    controller = None
    if case != "legacy_start_only":
        bank = NativeReferenceClips(infer.motion_rep, {name: raw[0] for name in ("idle", "walk", "run", "backward")})
        conditions = CASE_CONDITIONS[case]
        controller = NativeSmartLocomotion(infer, bank, keyframes_per_end=conditions["target_keyframes"],
            context_keyframes=conditions["context_keyframes"],
            fixed_tokens=8 if case == "native_smart4_fixed8" else None)
    before = seed.clone()
    arrays, report = rollout_case(infer, seed, case, controller, plans_per_command=1)
    assert report["summary"]["status"] == "completed"
    assert report["summary"]["real_generated_frames"] == 40
    np.testing.assert_array_equal(arrays["global_features"][:4], before[0].numpy())
    torch.testing.assert_close(seed, before, rtol=0, atol=0)
    for index, plan in enumerate(report["plans"]):
        played = arrays["global_features"][4 + 8 * index:12 + 8 * index]
        start = 0 if case in BUFFER_CASES else 4
        np.testing.assert_allclose(played[:, 0], 10 * (index + 1) + np.arange(start, start + 8) / 30)
        assert plan["played_prediction_frames_inclusive"] == [start, start + 7]
        if index:
            if case in BUFFER_CASES:
                np.testing.assert_allclose(arrays[f"plan_{index:03d}_input_context"][:, 0],
                                           10 * index + np.arange(8, 12) / 30)
                assert plan["input_context_origin"] == "previous_prediction_unplayed_lookahead4"
                assert plan["input_context_is_played"] is False
                assert plan["input_history_frames"] is None
            else:
                np.testing.assert_array_equal(arrays[f"plan_{index:03d}_input_context"],
                                             arrays["global_features"][8 * index:4 + 8 * index])
        assert plan["cross_replan_seam"]["root_planar_speed_mps"] > 200
        assert plan["intra_committed_clip"]["root_planar_speed_mean_mps"] == pytest.approx(1, abs=0.001)
        assert plan["intra_committed_clip"]["intervals"] == 7
        assert plan["constraints"]["source_encoded_tokens"] is False
        if case == "legacy_start_only":
            assert plan["constraints"]["future_pose_constraint_count"] == 0
        else:
            expected = [7] if CASE_CONDITIONS[case]["target_keyframes"] == 1 else [4, 5, 6, 7]
            assert plan["constraints"]["future_pose_constraint_slots"] == expected
            expected_context = [False, False, False, True] if CASE_CONDITIONS[case]["context_keyframes"] == 1 else [True] * 4
            assert plan["constraints"]["has_local_poses"][:4] == expected_context
    if case in ("legacy_start_only", "native_smart4_fixed8"):
        assert report["summary"]["predicted_token_counts"] == [8] * 5
    else:
        assert report["summary"]["predicted_token_counts"] == [6, 11, 6, 11, 6]
    assert len(calls) == 5


def test_buffer_indices_match_original_getters_exactly():
    from motionbricks.motion_backbone.demo.full_agent import full_navigation_agent

    prediction = torch.arange(64 * 5).reshape(1, 64, 5).float()
    state = SimpleNamespace(frames={"model_features": prediction, "mujoco_qpos": prediction},
        _device="cpu", NUM_FRAMES_PER_TOKEN=4, PRED_OFFSETS=full_navigation_agent.DEFAULT_PRED_OFFSETS,
        _current_frame_idx=4 - full_navigation_agent.DEFAULT_PRED_OFFSETS)
    played = np.stack([full_navigation_agent.get_next_frame(state) for _ in range(8)])
    expected_context = full_navigation_agent.get_context_motion_features(state)
    committed, context, start, indices = shipped_buffer_step(prediction, "cpu")
    assert state._current_frame_idx == 8 and start == 0 and indices == [8, 9, 10, 11]
    np.testing.assert_array_equal(committed[0].numpy(), played)
    torch.testing.assert_close(context, expected_context, rtol=0, atol=0)
    torch.testing.assert_close(context, prediction[:, 8:12], rtol=0, atol=0)


def test_legacy_matches_existing_rollout_and_nonfinite_counts_are_real():
    infer, raw = _fixture()
    _fake_predictions(infer, raw)
    expected, _, _, _ = rollout(infer, raw[:, :4], plans_per_command=1)
    _fake_predictions(infer, raw)
    arrays, _ = rollout_case(infer, raw[:, :4], "legacy_start_only", plans_per_command=1)
    np.testing.assert_array_equal(arrays["global_features"], expected[0].numpy())
    _fake_predictions(infer, raw, nonfinite=True)
    arrays, report = rollout_case(infer, raw[:, :4], "legacy_start_only", plans_per_command=1)
    assert report["summary"]["status"] == "failed"
    assert report["summary"]["real_generated_frames"] == 8
    assert report["summary"]["feature_finiteness"]["nonfinite_elements"] == 1
    assert np.isnan(arrays["global_features"][4, 0])
    import json
    json.dumps(report, allow_nan=False)


def test_capture_returns_original_tensors_and_restores_methods_on_exception():
    class Infer:
        def _predict_root_trajectories(self):
            return self.result

    infer = Infer()
    infer.result = (torch.tensor([8]), torch.zeros(1, 32, 5), torch.zeros(1, 32, 4))
    with pytest.raises(RuntimeError, match="intentional"):
        with capture_inference(infer) as (snapshots, available):
            assert available == ["_predict_root_trajectories"]
            assert infer._predict_root_trajectories() is infer.result
            np.testing.assert_array_equal(snapshots[available[0]][1], infer.result[1].numpy())
            raise RuntimeError("intentional")
    assert "_predict_root_trajectories" not in vars(infer)
    assert infer._predict_root_trajectories() is infer.result


def test_root_and_decoder_trace_arrays_keep_original_axes_and_outputs():
    from scripts.evaluate_native_mb_smart_locomotion import _trace_arrays

    infer, raw = _fixture()
    infer.global_motion_rep.stats = SimpleNamespace(unnormalize=lambda value, index: value + 2)
    infer.local_motion_rep.stats = SimpleNamespace(unnormalize=lambda value, index: value + 3)
    root = (torch.tensor([8]), torch.zeros(1, 64, 5), torch.zeros(1, 64, 4))
    local = torch.zeros(1, 64, len(infer.local_motion_rep.indices["all"]))
    decoded = (local, raw[:, :1].repeat(1, 64, 1))
    infer._predict_root_trajectories = lambda: root
    infer._decode_motions_from_predicted_root_and_pose_tokens = lambda: decoded
    originals = (infer._predict_root_trajectories, infer._decode_motions_from_predicted_root_and_pose_tokens)
    with capture_inference(infer) as (snapshots, _):
        assert infer._predict_root_trajectories() is root
        assert infer._decode_motions_from_predicted_root_and_pose_tokens() is decoded
        arrays = _trace_arrays(infer, snapshots, 32)
    assert infer._predict_root_trajectories is originals[0]
    assert infer._decode_motions_from_predicted_root_and_pose_tokens is originals[1]
    assert arrays["root_module_global_root"].shape == (32, 5)
    assert arrays["root_module_local_root"].shape == (32, 4)
    assert arrays["decoded_local_root"].shape == (32, 4)
    assert arrays["decoded_global_root_before_world_restore"].shape == (32, 5)
    assert np.all(arrays["root_module_global_root"] == 2)
    assert np.all(arrays["root_module_local_root"] == 3)
    np.testing.assert_array_equal(arrays["decoded_global_root_before_world_restore"],
                                 decoded[1][0, :32, :5].numpy())


def test_comparison_artifacts_repeatable_immutable_and_same_bundle(tmp_path):
    infer, raw = _fixture()
    folder, bundle = _dataset(tmp_path, raw)
    bundle.update(root_step=100000, pose_step=220000)
    expected_steps = {"root": 100000, "pose": 220000, "tokenizer": 100000}
    original = raw.clone()
    reports = []
    for name in ("first", "second"):
        _fake_predictions(infer, raw)
        reports.append(evaluate_comparison(infer, folder, tmp_path / name, bundle, include_fixed8=True))
    import json
    for case in (*CASES, "native_smart4_fixed8"):
        with np.load(tmp_path / "first" / case / "motion.npz") as first, np.load(tmp_path / "second" / case / "motion.npz") as second:
            np.testing.assert_array_equal(first["global_features"], second["global_features"])
            assert len(first["prediction"]) == 244
        report = json.loads((tmp_path / "first" / case / "report.json").read_text())
        assert report["bundle"] == bundle
        assert report["training_steps"] == expected_steps
        assert report["convergence_assessed"] is False
        assert report["summary"]["real_generated_frames"] == 240
        assert len(report["plans"]) == 30
        if case != "legacy_start_only":
            assert all(plan["train_reference_input"]["split"] == "train" for plan in report["plans"])
            assert report["plans"][0]["train_reference_input"]["source_frames"] == list(range(2333 % 8, 2333 % 8 + 4))
    torch.testing.assert_close(raw, original, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        evaluate_comparison(infer, folder, tmp_path / "first", bundle)
    with pytest.raises(ValueError, match="positive integer"):
        validate_frozen_bundle({**bundle, "root_step": 0})
    assert all(report["training_steps"] == expected_steps and not report["convergence_assessed"] for report in reports)
    assert reports[0]["sources"] == reports[1]["sources"]


def test_cli_checks_gpu_before_bundle_loading(tmp_path, monkeypatch):
    import scripts.evaluate_native_mb_pipeline as pipeline
    import scripts.evaluate_native_mb_smart_locomotion as evaluator
    order = []
    monkeypatch.setattr(evaluator, "select_gpu", lambda index: order.append(("gpu", index)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def load(*args):
        order.append(("load", args[-1]))
        raise RuntimeError("stop before loading weights")

    monkeypatch.setattr(pipeline, "load_bundle", load)
    with pytest.raises(RuntimeError, match="stop before loading"):
        main(["--root-run", "root", "--pose-run", "pose", "--output", str(tmp_path / "new"),
              "--device", "cuda", "--gpu", "3"])
    assert order == [("gpu", 3), ("load", "cuda")]
