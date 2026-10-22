"""Physical comparisons must not silently align different motion intervals."""

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("physics_comparison_under_test",
                                             SCRIPTS / "render_parc_physics_comparison.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def pair(tmp_path):
    source, physical = tmp_path / "raw.pkl", tmp_path / "physics.pkl"
    source.write_bytes(b"raw")
    physical.write_bytes(b"actual simulated recording")
    record = {"whole_motion": True, "start_fraction": 0., "frames": 4, "fps": 30.,
              "recorded_terrain_error_m": 0.,
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "converted_sha256": hashlib.sha256(physical.read_bytes()).hexdigest()}
    motion = {"root_pos": np.zeros((4, 3)), "fps": 30.}
    return record, source, physical, motion.copy(), motion.copy()


def test_matching_full_motion_is_accepted(pair):
    module.verify_pair(*pair)


def test_unvalidated_terrain_is_rejected(pair):
    del pair[0]["recorded_terrain_error_m"]
    with pytest.raises(ValueError, match="validated terrain pairing"):
        module.verify_pair(*pair)


@pytest.mark.parametrize("key,value", [("whole_motion", False), ("start_fraction", .1)])
def test_partial_motion_is_not_compared_as_full(pair, key, value):
    pair[0][key] = value
    with pytest.raises(ValueError, match="Partial"):
        module.verify_pair(*pair)


def test_changed_source_is_rejected(pair):
    pair[1].write_bytes(b"changed")
    with pytest.raises(ValueError, match="Raw source hash"):
        module.verify_pair(*pair)


def test_changed_export_is_rejected(pair):
    pair[2].write_bytes(b"changed")
    with pytest.raises(ValueError, match="Physical export hash"):
        module.verify_pair(*pair)


@pytest.mark.parametrize("key,value", [("fps", 60.), ("root_pos", np.zeros((3, 3)))])
def test_time_axis_mismatch_is_rejected(pair, key, value):
    pair[4][key] = value
    with pytest.raises(ValueError, match="time axes"):
        module.verify_pair(*pair)
