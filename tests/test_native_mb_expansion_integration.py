"""Exercise expansion orchestration with real files and synchronous fake conversion."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import expand_bones_native_dataset as expansion


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def corpus(tmp_path):
    source, pilot = tmp_path / "source", tmp_path / "pilot"
    source.mkdir()
    (pilot / "shards").mkdir(parents=True)
    candidates, records = [], []
    contract = {"fps": 60.0, "up_axis": "Y", "fixture": "frozen pilot"}

    def add_candidate(name, split="train", behavior="walk", content=None):
        path = source / f"{name}.bvh"
        path.write_bytes(content if content is not None else f"source:{name}\n".encode())
        row = {"path": str(path), "filename": name, "recording_group": name,
               "split": split, "behavior": behavior}
        candidates.append(row)
        return row

    for split, count in (("train", 152), ("validation", 19), ("test", 20)):
        for index in range(count):
            row = add_candidate(f"pilot_{split}_{index:03d}", split,
                                "run" if index % 2 else "walk")
            digest = _digest(row["path"])
            shard = f"shards/{digest[:20]}.npz"
            np.savez_compressed(pilot / shard,
                                global_features=np.full((150, 3), index, dtype=np.float32))
            records.append({**row, "source_sha256": digest, "shard": shard,
                            "frames_60hz": 150, "duration_seconds": 2.5,
                            "frozen_metadata": {"actor": index, "tags": [split, "retain"]}})

    def save():
        (pilot / "accepted.json").write_text(json.dumps(records), encoding="utf-8")
        (pilot / "candidate_manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in candidates), encoding="utf-8")
        (pilot / "skeleton_contract.json").write_text(json.dumps(contract), encoding="utf-8")

    save()
    return SimpleNamespace(source=source, pilot=pilot, output=tmp_path / "expanded",
                           records=records, candidates=candidates, contract=contract,
                           add_candidate=add_candidate, save=save)


@pytest.fixture
def conversion(monkeypatch, corpus):
    state = SimpleNamespace(initialized=[], converted=[], pools=[], parity_calls=[],
                            arrays={"global_features": np.arange(540, dtype=np.float32).reshape(180, 3)})
    state.parity = [{"filename": corpus.records[index]["filename"],
                     "max_array_error": {"global_features": 0.0}} for index in (0, 1)]
    pilot_names = {row["filename"] for row in corpus.records}

    def initialize(contract):
        state.initialized.append(contract)

    def parity(pilot_dir, records):
        state.parity_calls.append((pilot_dir, records))
        return state.parity

    def convert(record):
        assert record["split"] == "train"
        assert record["filename"] not in pilot_names
        state.converted.append(record)
        return {"arrays": state.arrays,
                "report": {"source_sha256": expansion.sha256(record["path"]),
                           "frames_60hz": 180, "duration_seconds": 3.0}}

    class SynchronousPool:
        def __init__(self, *, max_workers, mp_context, initializer, initargs):
            self.workers, self.batches = max_workers, []
            self.initializer, self.initargs = initializer, initargs
            state.pools.append(self)

        def __enter__(self):
            self.initializer(*self.initargs)
            return self

        def __exit__(self, *_):
            return False

        def map(self, function, records):
            batch = list(records)
            self.batches.append(batch)
            return map(function, batch)

    monkeypatch.setattr(expansion, "ProcessPoolExecutor", SynchronousPool)
    monkeypatch.setattr(expansion, "_initialize_conversion", initialize)
    monkeypatch.setattr(expansion, "verify_converter_parity", parity)
    monkeypatch.setattr(expansion, "_convert_record", convert)
    return state


def _assert_frozen_pilot(corpus, accepted, before):
    assert _snapshot(corpus.pilot) == before
    assert accepted[:191] == corpus.records
    assert [row for row in accepted if row["split"] != "train"] == [
        row for row in corpus.records if row["split"] != "train"]
    for row in corpus.records:
        assert (corpus.output / row["shard"]).read_bytes() == before[row["shard"]]


def test_expand_preserves_pilot_bytes_and_emits_train_only_manifest(corpus, conversion):
    additions = [corpus.add_candidate(f"new_{index}", behavior="run" if index % 2 else "walk")
                 for index in range(5)]
    corpus.add_candidate("unselected_validation", "validation")
    corpus.add_candidate("unselected_test", "test")
    corpus.save()
    before, source_before = _snapshot(corpus.pilot), _snapshot(corpus.source)

    report = expansion.expand(corpus.pilot, corpus.source, corpus.output, max_train=157, workers=2)

    accepted = _read_json(corpus.output / "accepted.json")
    _assert_frozen_pilot(corpus, accepted, before)
    assert _snapshot(corpus.source) == source_before
    added = accepted[191:]
    assert len(accepted) == 196
    assert {row["filename"] for row in added} == {row["filename"] for row in additions}
    assert all(row["split"] == "train" for row in added)
    assert _read_json(corpus.output / "accepted.in_progress.json") == accepted
    assert _read_json(corpus.output / "rejected.json") == []
    assert _read_json(corpus.output / "rejected.in_progress.json") == []
    assert _read_json(corpus.output / "expansion_report.json") == report
    assert report["status"] == "completed"
    assert report["split_counts"] == {"train": 157, "validation": 19, "test": 20}
    assert report["selection"]["existing_train_clips"] == 152
    assert report["selection"]["selected_additional_candidates"] == 5
    assert report["rejected"] == 0
    assert report["train_behaviors"] == {"walk": 79, "run": 78}
    assert report["train_recording_families"] == 157
    assert report["train_duration_hours"] == pytest.approx((152 * 2.5 + 5 * 3.0) / 3600)
    assert report["sole_proxy_geometry"] == "position_only_v0"
    assert report["quality_approved"] is False
    assert report["source_accepted_sha256"] == _digest(corpus.pilot / "accepted.json")
    assert report["candidate_manifest_sha256"] == _digest(corpus.pilot / "candidate_manifest.jsonl")
    assert report["reserved_content_hashes_sha256"] == _digest(corpus.output / "reserved_content_hashes.json")
    assert report["copied_pilot_shard_sha256"] == {
        row["shard"]: _digest(corpus.pilot / row["shard"]) for row in corpus.records}
    reserved = _read_json(corpus.output / "reserved_content_hashes.json")
    assert len(reserved) == 41
    assert Counter(row["split"] for row in reserved) == {"validation": 20, "test": 21}
    assert all(row["source_sha256"] == _digest(row["path"]) for row in reserved)
    assert conversion.initialized == [corpus.contract, corpus.contract]
    assert conversion.parity_calls == [(corpus.pilot.resolve(), corpus.records)]
    assert _read_json(corpus.output / "converter_parity.json") == report["converter_parity"] == conversion.parity
    assert len(conversion.pools) == 1
    assert conversion.pools[0].workers == 2
    assert [len(batch) for batch in conversion.pools[0].batches] == [4, 1]
    assert conversion.converted == _read_json(corpus.output / "selection_plan.json")["candidates"]
    for row in added:
        assert row["source_sha256"] == _digest(row["path"])
        with np.load(corpus.output / row["shard"], allow_pickle=False) as shard:
            assert set(shard.files) == set(conversion.arrays)
            for name, array in conversion.arrays.items():
                np.testing.assert_array_equal(shard[name], array)


def test_expand_rejects_new_train_duplicate_of_unselected_validation(corpus, conversion):
    reserved = corpus.add_candidate("unselected_validation", "validation", content=b"reserved content\n")
    duplicate = corpus.add_candidate("new_train_duplicate", content=Path(reserved["path"]).read_bytes())
    unique = corpus.add_candidate("new_train_unique")
    corpus.save()
    before, source_before = _snapshot(corpus.pilot), _snapshot(corpus.source)

    report = expansion.expand(corpus.pilot, corpus.source, corpus.output, max_train=154, workers=1)

    accepted = _read_json(corpus.output / "accepted.json")
    _assert_frozen_pilot(corpus, accepted, before)
    assert _snapshot(corpus.source) == source_before
    assert [row["filename"] for row in accepted[191:]] == [unique["filename"]]
    assert _read_json(corpus.output / "rejected.json") == [
        {**duplicate, "reason": "Duplicate source content", "duplicate_split": "validation"}]
    assert {row["filename"] for row in conversion.converted} == {unique["filename"], duplicate["filename"]}
    assert not (corpus.output / f"shards/{_digest(duplicate['path'])[:20]}.npz").exists()
    assert report["split_counts"] == {"train": 153, "validation": 19, "test": 20}
    assert report["rejected"] == 1
    assert report["selection"]["selected_additional_candidates"] == 2
    assert _read_json(corpus.output / "expansion_report.json") == report


def test_expand_raises_on_existing_train_duplicate_of_reserved_candidate(corpus, conversion):
    corpus.add_candidate("reserved_pilot_duplicate", "validation",
                         content=Path(corpus.records[0]["path"]).read_bytes())
    corpus.add_candidate("new_train")
    corpus.save()
    before, source_before = _snapshot(corpus.pilot), _snapshot(corpus.source)

    with pytest.raises(ValueError, match="Existing training content duplicates a reserved held-out candidate"):
        expansion.expand(corpus.pilot, corpus.source, corpus.output, max_train=153, workers=1)

    _assert_frozen_pilot(corpus, corpus.records, before)
    assert _snapshot(corpus.source) == source_before
    assert conversion.pools == []
    assert conversion.converted == []
    assert not (corpus.output / "accepted.json").exists()
    assert not (corpus.output / "expansion_report.json").exists()


@pytest.mark.parametrize("existing_kind", ["directory", "file"])
def test_expand_rejects_existing_output_without_overwriting(corpus, conversion, existing_kind):
    if existing_kind == "directory":
        corpus.output.mkdir()
        (corpus.output / "accepted.json").write_bytes(b"existing output must survive\n")
    else:
        corpus.output.write_bytes(b"existing output file must survive\n")
    before = _snapshot(corpus.output.parent)

    with pytest.raises(ValueError, match="Use a new output directory"):
        expansion.expand(corpus.pilot, corpus.source, corpus.output, max_train=153, workers=1)

    assert _snapshot(corpus.output.parent) == before
    assert conversion.initialized == conversion.parity_calls == conversion.pools == conversion.converted == []


def test_expand_does_not_overwrite_copied_pilot_shard_on_name_collision(corpus, conversion):
    candidate = corpus.add_candidate("new_train")
    collision = f"shards/{_digest(candidate['path'])[:20]}.npz"
    old = corpus.records[0]
    (corpus.pilot / old["shard"]).rename(corpus.pilot / collision)
    old["shard"] = collision
    corpus.save()
    before, source_before = _snapshot(corpus.pilot), _snapshot(corpus.source)

    with pytest.raises(ValueError, match="Shard filename collision"):
        expansion.expand(corpus.pilot, corpus.source, corpus.output, max_train=153, workers=1)

    _assert_frozen_pilot(corpus, corpus.records, before)
    assert _snapshot(corpus.source) == source_before
    assert conversion.converted == [candidate]
    assert not (corpus.output / "accepted.json").exists()
    assert not (corpus.output / "expansion_report.json").exists()
