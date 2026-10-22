from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.expand_bones_native_dataset import (select_training_expansion,
                                                seed_reserved_hashes, validate_source_paths)


def row(name, group, split="train", behavior="walk"):
    return {"path": f"/source/{name}.bvh", "filename": name,
            "recording_group": group, "split": split, "behavior": behavior}


def test_preserves_all_reserved_families_not_only_selected_holdouts():
    pilot = [row("train0", "t"), row("val0", "v", "validation"), row("test0", "x", "test")]
    candidates = pilot + [row("new", "new"), row("val_variant", "v", "validation"),
                          row("unselected_holdout", "v2", "validation"), row("test_variant", "x", "test")]
    selected, plan = select_training_expansion(pilot, candidates, 3)
    assert [r["filename"] for r in selected] == ["new"]
    assert plan["reserved_families"] == ["v", "v2", "x"]
    assert len(pilot) == 3


def test_rejects_family_cross_split_and_pilot_identity_changes():
    pilot = [row("train0", "t"), row("val0", "v", "validation")]
    with pytest.raises(ValueError, match="crosses splits"):
        select_training_expansion(pilot, pilot + [row("variant", "v")], 3)
    changed = [dict(pilot[0]), dict(pilot[1], recording_group="renamed")]
    with pytest.raises(ValueError, match="Pilot identity"):
        select_training_expansion(pilot, changed, 3)


def test_balances_behavior_then_family_deterministically():
    pilot = [row("base", "base")]
    candidates = pilot + [row(f"a{i}", "a") for i in range(5)] + [row("b0", "b")]
    candidates += [row("run0", "run", behavior="run"), row("run1", "run", behavior="run")]
    first, _ = select_training_expansion(pilot, candidates, 5)
    second, _ = select_training_expansion(pilot, list(reversed(candidates)), 5)
    assert first == second
    assert [r["behavior"] for r in first] == ["run", "walk", "run", "walk"]
    assert {r["recording_group"] for r in first if r["behavior"] == "walk"} == {"a", "b"}


def test_rejects_duplicate_candidate_paths_and_nonexpanding_budget():
    pilot = [row("base", "base")]
    with pytest.raises(ValueError, match="Duplicate candidate"):
        select_training_expansion(pilot, pilot * 2, 3)
    with pytest.raises(ValueError, match="must exceed"):
        select_training_expansion(pilot, pilot, 1)


def test_source_paths_cannot_escape_and_must_match_filename(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "clip.bvh"
    path.write_text("fixture")
    record = {"path": str(path), "filename": "clip"}
    validate_source_paths([record], source)
    with pytest.raises(ValueError, match="Invalid or unavailable"):
        validate_source_paths([dict(record, filename="renamed")], source)
    outside = tmp_path / "outside.bvh"
    outside.write_text("fixture")
    with pytest.raises(ValueError, match="Invalid or unavailable"):
        validate_source_paths([{"path": str(outside), "filename": "outside"}], source)


def test_reserved_source_hashes_detect_existing_training_leakage():
    reserved = [{"source_sha256": "same-motion", "split": "validation"}]
    with pytest.raises(ValueError, match="Existing training content duplicates"):
        seed_reserved_hashes({"same-motion": "train"}, reserved)
    assert seed_reserved_hashes({}, reserved) == {"same-motion": "validation"}
    assert seed_reserved_hashes({"same-motion": "validation"}, reserved) == {"same-motion": "validation"}
