import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_native_mb_stair_review import export
from motionbricks.data.soma_replay import UE_BASIS, motion_to_ue


DATASET = ROOT / "out/native_mb_stairs_20261007/plant-v4-whole-warm"


def test_native_stair_review_export_contract(tmp_path):
    manifest = export(DATASET, tmp_path)
    assert [clip["id"] for clip in manifest["clips"]] == [
        "up_source", "up_model", "down_source", "down_model"]
    assert manifest["model_inference"] is False
    assert manifest["runtime_state"] == "offline_review_sidecar"
    assert manifest["fps"] == 30

    for direction, artifact in (("up", "00_up_start000000"), ("down", "03_down_start000000")):
        source, model = (tmp_path / f"{direction}_{kind}.json" for kind in ("source", "model"))
        source_data, model_data = json.loads(source.read_text()), json.loads(model.read_text())
        assert source_data["schema"] == model_data["schema"] == "motionbricks.soma.replay.v2"
        assert source_data["runtime_state"] == "source_replay"
        assert model_data["runtime_state"] == "generated_replay"
        assert source_data["model_inference"] is model_data["model_inference"] is False
        assert model_data["offline_generated"] is True
        assert "no time-warp or root correction" in model_data["motion_postprocessing"]
        assert source_data["fps"] == model_data["fps"] == 30
        assert source_data["preprocess_version"] == model_data["preprocess_version"] == "native-mb-stair-review-v1"
        assert source_data["clip"]["artifact"] == model_data["clip"]["artifact"] == artifact
        assert source_data["clip"]["feature_array"] == "features_source"
        assert model_data["clip"]["feature_array"] == "features_official_landings_c1_t1_i1"
        assert len(source_data["frames"]) == len(model_data["frames"])
        for source_frame, model_frame in zip(source_data["frames"][:4], model_data["frames"][:4]):
            np.testing.assert_allclose(source_frame["root_m"], model_frame["root_m"], atol=0.0)
            np.testing.assert_allclose(source_frame["positions_m"], model_frame["positions_m"], atol=0.0)
        assert source.stat().st_size < 32 * 1024 * 1024
        assert model.stat().st_size < 32 * 1024 * 1024
        source_clip = next(c for c in manifest["clips"] if c["id"] == f"{direction}_source")
        model_clip = next(c for c in manifest["clips"] if c["id"] == f"{direction}_model")
        assert source_clip["support_patches"] == model_clip["support_patches"]
        assert "not measured full terrain" in source_clip["support_evidence"]
        for clip, path in ((source_clip, source), (model_clip, model)):
            assert clip["fixture"] == path.name
            assert clip["fixture_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            assert np.asarray(clip["bounds_min_ue_cm"]).shape == (3,)
            assert np.asarray(clip["bounds_max_ue_cm"]).shape == (3,)
            assert np.all(np.asarray(clip["bounds_min_ue_cm"]) <= np.asarray(clip["bounds_max_ue_cm"]))


def test_support_polygon_uses_documented_ue_basis():
    from scripts.export_native_mb_stair_review import _support_patches

    result = _support_patches([{
        "id": 3,
        "height_m": 0.4,
        "polygon_xz_m": [[0.2, -0.3], [0.4, -0.3], [0.4, 0.1]],
        "confidence": 0.8,
    }])[0]
    assert result["polygon_ue_cm"] == [[-30.0, -20.0, 40.0], [-30.0, -40.0, 40.0], [10.0, -40.0, 40.0]]


def test_v2_fixture_ue_fk_matches_reference_with_zero_order_sampling(tmp_path):
    export(DATASET, tmp_path)
    for fixture_path in sorted(tmp_path.glob("*_*.json")):
        fixture = json.loads(fixture_path.read_text())
        if fixture["schema"] != "motionbricks.soma.replay.v2":
            continue
        neutral_ue = np.asarray(fixture["neutral_positions_m"]) @ UE_BASIS.T * 100
        max_position_error = 0.0
        max_rotation_error = 0.0
        for frame in fixture["frames"]:
            local = Rotation.from_quat(np.asarray(frame["local_xyzw"])).as_matrix()
            local_ue = UE_BASIS @ local @ UE_BASIS.T
            calculated_positions = np.zeros((30, 3))
            calculated_rotations = np.zeros((30, 3, 3))
            calculated_positions[0] = np.asarray(frame["root_m"]) @ UE_BASIS.T * 100
            calculated_rotations[0] = local_ue[0]
            for joint, parent in enumerate(fixture["parents"]):
                if parent < 0:
                    continue
                calculated_rotations[joint] = calculated_rotations[parent] @ local_ue[joint]
                calculated_positions[joint] = calculated_positions[parent] + calculated_rotations[parent] @ (
                    neutral_ue[joint] - neutral_ue[parent])

            reference_positions, reference_rotations = motion_to_ue(
                np.asarray(frame["positions_m"]), Rotation.from_quat(np.asarray(frame["global_xyzw"])).as_matrix())
            max_position_error = max(max_position_error,
                                     float(np.linalg.norm(calculated_positions - reference_positions, axis=-1).max()))
            delta = calculated_rotations @ np.swapaxes(reference_rotations, -1, -2)
            max_rotation_error = max(max_rotation_error,
                                     float(Rotation.from_matrix(delta).magnitude().max()))
        assert max_position_error <= 0.02, (fixture_path.name, max_position_error)
        assert max_rotation_error <= 1e-4, (fixture_path.name, max_rotation_error)
