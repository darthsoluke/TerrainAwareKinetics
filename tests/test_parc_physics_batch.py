"""Local-only packaging tests using tiny synthetic PARC input fixtures."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_parc_physics_batch as batch


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ParcPhysicsBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo"
        self.output = Path(self.temp.name) / "bundle"
        self._make_fixture()

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def _make_fixture(self):
        self.records = []
        showcase = []
        for index in range(9):
            key = f"fixture/run_{index // 3}/candidate_{index:03d}"
            payload = f"raw-pickle-placeholder-{index}\x00".encode("ascii")
            relative = f"{key}/generated.pkl"
            source_path = self._write(f"out/parc_complete_20261008/downloads/{relative}", payload)
            frames = 20 + index
            row = {
                "candidate_key": key,
                "source_path": relative,
                "reported_source_sha256": sha256(payload),
                "frames": frames,
                "rank": 8 - index,
                "report_entry": {"sha256": sha256(payload), "frames": frames},
            }
            self.records.append(row)
            if index >= 2:
                showcase.append({"name": f"showcase_{index}", "candidate_key": key})

        catalog = {"candidate_count": len(self.records), "candidates": self.records}
        summary = {"showcase": showcase}
        self._write("out/parc_complete_20261008/gallery/catalog.json",
                    json.dumps(catalog).encode("utf-8"))
        self._write("out/parc_complete_20261008/summary.json",
                    json.dumps(summary).encode("utf-8"))

        repo = "out/vendor/PARC"
        self._write(f"{repo}/PARC/__init__.py", b"# package\n")
        self._write(f"{repo}/PARC/nested/core.py", b"VALUE = 1\n")
        self._write(f"{repo}/scripts/run_tracker.py", b"def main(): pass\n")
        self._write(f"{repo}/scripts/__init__.py", b"\"\"\"scripts\"\"\"\n")
        self._write(
            f"{repo}${PRIVATE_STORAGE}/humanoid.xml",
            b'<mujoco><compiler meshdir="meshes"/><asset><mesh file="body.obj"/></asset></mujoco>',
        )
        self._write(f"{repo}${PRIVATE_STORAGE}/meshes/body.obj", b"tiny mesh fixture\n")
        tracker = "out/parc_legacy_models_20261008/extracted/april272025/iter_3/p3_tracker"
        self._write(f"{tracker}/model.pt", b"tiny model fixture\x00")
        self._write(f"{tracker}/dm_env.yaml", b"env: {}\n")
        self._write(f"{tracker}/agent_config.yaml", b"agent_name: fixture\n")

    def test_builds_hash_recorded_bundle_with_raw_motions_and_id_showcase(self):
        manifest = batch.build_bundle(self.root, self.output, expected_motion_count=9)
        archive_path = self.output / manifest["archive"]["path"]

        self.assertEqual(manifest["motion_count"], 9)
        self.assertEqual(len(manifest["representatives"]), 7)
        self.assertEqual(
            {entry["candidate_key"] for entry in manifest["representatives"]},
            {row["candidate_key"] for row in json.loads(
                (self.root / "out/parc_complete_20261008/summary.json").read_text())["showcase"]},
        )
        self.assertEqual(manifest["archive"]["sha256"], batch.digest(archive_path))
        self.assertEqual(manifest["archive"]["size_bytes"], archive_path.stat().st_size)
        self.assertEqual(len(manifest["files"]), 9 + 2 + 3 + 1 + 3)

        motion_entries = manifest["motions"]
        self.assertEqual(len(motion_entries), 9)
        with tarfile.open(archive_path, "r:") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(set(names), {entry["path"] for entry in manifest["files"]} | {"_bundle.json"})
            for member in members:
                parsed = PurePosixPath(member.name)
                self.assertFalse(parsed.is_absolute())
                self.assertNotIn("..", parsed.parts)
                self.assertNotIn("\\", member.name)
                self.assertTrue(member.isfile())

            for archive_name, details in motion_entries.items():
                self.assertRegex(archive_name, r"^motions/motion_\d{3}_[0-9a-f]{12}\.pkl$")
                original = self.root / "out/parc_complete_20261008/downloads" / next(
                    row["source_path"] for row in self.records
                    if row["candidate_key"] == details["candidate_key"]
                )
                archived = archive.extractfile(archive_name).read()
                self.assertEqual(archived, original.read_bytes())
                self.assertEqual(details["sha256"], sha256(archived))
                row = next(row for row in self.records if row["candidate_key"] == details["candidate_key"])
                self.assertEqual(details["source_frames"], row["frames"])

            self.assertIn("PARC/data/assets/meshes/body.obj", names)
            embedded = json.load(archive.extractfile("_bundle.json"))
            self.assertEqual(embedded["source_sha256"], manifest["source_sha256"])

    def test_rejects_existing_output_without_modifying_it(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "overwrite"):
            batch.build_bundle(self.root, self.output, expected_motion_count=9)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_catalog_hash_mismatch_fails_before_output_creation(self):
        self.records[0]["reported_source_sha256"] = "0" * 64
        self.records[0]["report_entry"]["sha256"] = "0" * 64
        catalog = {"candidate_count": len(self.records), "candidates": self.records}
        (self.root / "out/parc_complete_20261008/gallery/catalog.json").write_text(
            json.dumps(catalog), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "Catalog hash mismatch"):
            batch.build_bundle(self.root, self.output, expected_motion_count=9)
        self.assertFalse(self.output.exists())

    def test_rejects_duplicate_candidate_ids_and_traversal(self):
        duplicate = dict(self.records[1])
        duplicate["candidate_key"] = self.records[0]["candidate_key"]
        catalog = {"candidate_count": 9, "candidates": [self.records[0], duplicate, *self.records[2:]]}
        (self.root / "out/parc_complete_20261008/gallery/catalog.json").write_text(
            json.dumps(catalog), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "duplicate candidate ID"):
            batch.build_bundle(self.root, self.output, expected_motion_count=9)

        self._make_fixture()
        self.records[0]["source_path"] = "../../outside/generated.pkl"
        catalog = {"candidate_count": 9, "candidates": self.records}
        (self.root / "out/parc_complete_20261008/gallery/catalog.json").write_text(
            json.dumps(catalog), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "Non-canonical archive path"):
            batch.build_bundle(self.root, self.output, expected_motion_count=9)

    def test_bundle_limit_is_checked_before_creating_output(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            batch.build_bundle(self.root, self.output, expected_motion_count=9, max_bundle_bytes=1024)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
