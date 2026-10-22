"""CPU-only contract tests for exporting completed PARC physics recordings."""

import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_parc_physics_recordings as exporter


class LegacyTerrain:
    def __init__(self):
        self.hf = np.asarray([[0.0, 0.2], [0.4, 0.6]], dtype=np.float32)
        self.hf_maxmin = np.stack((self.hf + 0.1, self.hf - 0.1), axis=-1)
        self.min_point = np.asarray([-0.2, -0.2], dtype=np.float32)
        self.dxdy = np.asarray([0.2, 0.2], dtype=np.float32)

    def update_old(self):
        return None

    def to_torch(self, device):
        if device != "cpu":
            raise AssertionError("export must stay on CPU")

    def numpy_copy(self):
        return self


class FakeMotionData:
    def __init__(self, data, device="cpu"):
        if device != "cpu":
            raise AssertionError("legacy data must stay on CPU")
        self.data = data
        self.data["terrain"].update_old()
        self.data["terrain"].to_torch(device)

    def get_frames(self):
        return self.data["frames"]

    def has_contacts(self):
        return "contacts" in self.data

    def get_contacts(self):
        return self.data["contacts"]

    def get_fps(self):
        return self.data["fps"]

    def get_loop_mode(self):
        return self.data["loop_mode"]

    def has_terrain(self):
        return "terrain" in self.data

    def get_terrain(self):
        return self.data["terrain"]


class FakeCharacter:
    _device = types.SimpleNamespace(type="cpu")

    def load_char_file(self, path):
        self.character_path = path

    def get_dof_size(self):
        return 28

    def get_num_non_root_joints(self):
        return 14

    def get_num_bodies(self):
        return 15

    def get_num_contact_bodies(self):
        return 15

    def dof_to_rot(self, dof):
        shaped = dof.reshape((len(dof), 14, 2))
        ones = np.ones((len(dof), 14, 1), dtype=np.float32)
        quat = np.concatenate((shaped, np.zeros_like(ones), ones), axis=-1)
        return quat / np.linalg.norm(quat, axis=-1, keepdims=True)


def _fake_expmap_to_quat(expmap):
    ones = np.ones((len(expmap), 1), dtype=np.float32)
    quat = np.concatenate((expmap, ones), axis=-1)
    return quat / np.linalg.norm(quat, axis=-1, keepdims=True)


def _fake_quat_pos(quat):
    result = np.asarray(quat).copy()
    negative = result[..., 3] < 0
    result[negative] *= -1
    return result


def _as_array(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _fake_motion_to_ms_file(raw, motion_data, character):
    frames = np.asarray(motion_data.get_frames(), dtype=np.float32)
    root_rot = _fake_quat_pos(_fake_expmap_to_quat(frames[:, 3:6]))
    joint_rot = _fake_quat_pos(character.dof_to_rot(frames[:, 6:]))
    contacts = np.asarray(motion_data.get_contacts(), dtype=np.float32)
    terrain = motion_data.get_terrain()
    misc = {key: value for key, value in raw.items()
            if key not in {"frames", "contacts", "terrain", "fps", "loop_mode"}}
    return types.SimpleNamespace(
        motion_data=types.SimpleNamespace(
            root_pos=frames[:, :3].copy(), root_rot=root_rot, joint_rot=joint_rot,
            body_contacts=contacts.copy(), fps=int(motion_data.get_fps()),
            loop_mode=motion_data.get_loop_mode()),
        terrain_data=types.SimpleNamespace(
            hf=terrain.hf.copy(), hf_maxmin=terrain.hf_maxmin.copy(),
            min_point=terrain.min_point.copy(), dx=float(terrain.dxdy[0])),
        misc_data=misc)


def _fake_save_ms_file(data, filepath):
    sections = {}
    for name in ("motion_data", "terrain_data", "misc_data"):
        value = getattr(data, name)
        payload = value if name == "misc_data" else (None if value is None else vars(value))
        sections[name] = None if payload is None else pickle.dumps(payload, protocol=4)
    with open(filepath, "wb") as stream:
        pickle.dump(sections, stream, protocol=4)


def _fake_load_ms_file(filepath):
    with open(filepath, "rb") as stream:
        sections = pickle.load(stream)
    result = {}
    for name in ("motion_data", "terrain_data", "misc_data"):
        value = sections[name]
        decoded = None if value is None else pickle.loads(value)
        result[name] = (decoded if name == "misc_data" or decoded is None else
                        types.SimpleNamespace(**decoded))
    return types.SimpleNamespace(**result)


class ExporterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self.run = self.root / "run"
        self.bundle.mkdir()
        self.run.mkdir()
        (self.bundle / "PARC" / "scripts").mkdir(parents=True)
        (self.bundle / "PARC" / "scripts" / "convert_legacy_motions.py").write_bytes(
            b"supplemental official converter fixture")
        (self.bundle / "motions").mkdir()
        (self.run / "recorded_motions").mkdir()
        self.source_path = self.bundle / "motions" / "candidate.pkl"
        self.source_path.write_bytes(b"immutable source candidate")
        self.source_hash = exporter.sha256_file(self.source_path)
        self.character = FakeCharacter()
        self.converter = types.SimpleNamespace(
            motion_edit_lib=types.SimpleNamespace(
                load_motion_file=self._load_legacy,
                MotionData=FakeMotionData),
            motion_to_ms_file=_fake_motion_to_ms_file,
            torch=types.SimpleNamespace(float32=np.float32, as_tensor=lambda value, **kwargs:
                np.asarray(value, dtype=kwargs.get("dtype", np.float32))),
            torch_util=types.SimpleNamespace(exp_map_to_quat=_fake_expmap_to_quat,
                                             quat_pos=_fake_quat_pos),
            file_io=types.SimpleNamespace(save_ms_file=_fake_save_ms_file,
                                          load_ms_file=_fake_load_ms_file),
            _convert_misc_value=lambda value: value)
        self.saved_rows = []
        self.input_rows = []
        for index, whole_motion in enumerate((True, False)):
            frames = np.zeros((3 + index, 34), dtype=np.float32)
            frames[:, 2] = 0.9
            frames[:, 3:6] = np.asarray([0.01, 0.02, 0.03], dtype=np.float32)
            frames[:, 6:] = np.arange(28, dtype=np.float32) * 0.001
            contacts = np.zeros((len(frames), 15), dtype=np.float32)
            contacts[:, index] = 1.0
            legacy = {"fps": 30, "loop_mode": "CLAMP", "frames": frames,
                      "contacts": contacts, "terrain": LegacyTerrain(),
                      "obs": np.arange(len(frames) * 2, dtype=np.float32).reshape(len(frames), 2),
                      "obs_shapes": {"char_obs": {"shape": (2,), "use_normalizer": True}}}
            raw_path = self.run / "recorded_motions" / ("raw_%02d.pkl" % index)
            with raw_path.open("wb") as stream:
                pickle.dump(legacy, stream, protocol=4)
            key = "course/candidate_%02d" % index
            raw_hash = exporter.sha256_file(raw_path)
            self.saved_rows.append({
                "candidate_key": key, "source_sha256": self.source_hash,
                "filename": raw_path.name, "sha256": raw_hash, "frames": len(frames),
                "fps": 30, "source_frames": len(frames), "start_fraction": 0.0 if whole_motion else 0.25,
                "whole_motion": whole_motion, "recorded_state": "simulated",
                "recorded_terrain_error_m": 0.0,
                "contacts": "simulated_contact_force_threshold",
                "training_approved": False,
                "path": "recorded_motions/" + raw_path.name})
            self.input_rows.append({"candidate_key": key, "sha256": self.source_hash,
                                    "path": "motions/candidate.pkl"})

        self.report = {"status": "official_recording_finished",
                       "physics_recording_completed": True,
                       "inputs_unchanged_after_run": True,
                       "packed_terrain_validation": {"grid_spacing_matches": True,
                           "all_tile_cells_checked": 100, "maximum_height_error_m": 0.0},
                       "inputs": self.input_rows, "saved": self.saved_rows}
        self._write_report()
        self.output = self.root / "export"

    def tearDown(self):
        self.temp.cleanup()

    def _load_legacy(self, path, convert_to_class=False):
        if convert_to_class:
            raise AssertionError("export should invoke the official converter in raw mode")
        with open(path, "rb") as stream:
            return pickle.load(stream)

    def _write_report(self):
        (self.run / "report.json").write_text(json.dumps(self.report), encoding="utf-8")

    def _export(self):
        with patch.object(exporter, "_load_official_converter",
                          return_value=(self.converter, self.character,
                                        self.bundle / "PARC/data/assets/humanoid.xml")):
            return exporter.export_run(self.bundle, self.run, self.output)

    def test_exports_every_saved_record_and_keeps_report_provenance(self):
        raw_hashes = [exporter.sha256_file(self.run / row["path"]) for row in self.saved_rows]
        manifest = self._export()
        self.assertEqual(len(manifest["recordings"]), 2)
        rows = manifest["recordings"]
        self.assertEqual([row["whole_motion"] for row in rows], [True, False])
        self.assertEqual([row["start_fraction"] for row in rows], [0.0, 0.25])
        for index, row in enumerate(rows):
            self.assertEqual(row["sha256"], raw_hashes[index])
            self.assertEqual(row["source_sha256"], self.source_hash)
            self.assertEqual(row["frames"], self.saved_rows[index]["frames"])
            self.assertEqual(row["fps"], 30)
            self.assertTrue(row["converted_path"].startswith("recordings/"))
            self.assertEqual(len(row["converted_sha256"]), 64)
            self.assertTrue(all(value for key, value in row["conversion_parity"].items()
                                if key.endswith("_exact") or key == "observations_preserved"))
            converted_path = self.output / row["converted_path"]
            with converted_path.open("rb") as stream:
                container = pickle.load(stream)
            self.assertEqual(set(container), {"motion_data", "terrain_data", "misc_data"})
            self.assertTrue(all(isinstance(container[name], bytes)
                                for name in container if name != "misc_data") or
                            container["misc_data"] is not None)
            nested = _fake_load_ms_file(converted_path)
            self.assertNotIn("path_nodes", nested.misc_data)
            provenance = nested.misc_data[exporter.PROVENANCE_KEY]
            self.assertEqual(provenance["candidate_key"], row["candidate_key"])
            self.assertEqual(provenance["source_sha256"], row["source_sha256"])
            self.assertEqual(provenance["recording_sha256"], row["sha256"])
            self.assertEqual(provenance["converter_sha256"], manifest["official_converter_sha256"])
            self.assertEqual(provenance["whole_motion"], row["whole_motion"])
            self.assertEqual(provenance["frames"], row["frames"])
            self.assertEqual(provenance["fps"], row["fps"])
        for index, row in enumerate(self.saved_rows):
            self.assertEqual(exporter.sha256_file(self.run / row["path"]), raw_hashes[index])

    def test_rejects_tampered_recording_before_creating_output(self):
        raw_path = self.run / self.saved_rows[0]["path"]
        raw_path.write_bytes(raw_path.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "recording hash mismatch"):
            self._export()
        self.assertFalse(self.output.exists())

    def test_rejects_source_hash_mismatch(self):
        self.saved_rows[0]["source_sha256"] = "0" * 64
        self._write_report()
        with self.assertRaisesRegex(ValueError, "does not match run report input hash"):
            self._export()
        self.assertFalse(self.output.exists())

    def test_rejects_incomplete_run_report(self):
        self.report["physics_recording_completed"] = False
        self._write_report()
        with self.assertRaisesRegex(ValueError, "does not confirm physics recording completion"):
            self._export()
        self.assertFalse(self.output.exists())

    def test_refuses_existing_output(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self._export()

    def test_report_path_traversal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            exporter._resolve_inside(self.run, "../outside.pkl", "saved recording path")

    def test_cpu_device_check_accepts_string_and_torch_device_like(self):
        self.assertEqual(exporter._device_type("cpu"), "cpu")
        self.assertEqual(exporter._device_type(types.SimpleNamespace(type="cpu")), "cpu")
        self.assertEqual(exporter._device_type(types.SimpleNamespace(type="cuda")), "cuda")


if __name__ == "__main__":
    unittest.main()
