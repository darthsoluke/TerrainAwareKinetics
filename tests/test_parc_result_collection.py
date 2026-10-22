"""Local-only tests for checked PARC result extraction and recovery selection."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import deploy_parc_remote_batch as deploy
import download_parc_batch_results as downloader
import finish_parc_remote_batch as recovery


def _make_archive(path: Path, entries):
    with tarfile.open(path, "w") as archive:
        for entry in entries:
            if isinstance(entry, tuple):
                name, content = entry
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            else:
                archive.addfile(entry)
    return path


@pytest.mark.parametrize(
    "name",
    ("../escape.txt", "nested/../../escape.txt", "/absolute.txt", "nested\\escape.txt",
     "a/./b", "a//b", "a/file:stream"),
)
def test_unpack_checked_rejects_traversal_and_platform_separators(tmp_path, name):
    archive = _make_archive(tmp_path / "unsafe.tar", [(name, b"blocked")])
    destination = tmp_path / "downloads"

    with pytest.raises(ValueError, match="Rejected archive member"):
        downloader.unpack_checked(archive, destination)

    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize("run_id", ("../outside", "/absolute", "a/b", ""))
def test_download_rejects_invalid_run_before_ssh(tmp_path, monkeypatch, run_id):
    monkeypatch.setattr(deploy, "checked_run", lambda *a, **kw: pytest.fail("SSH must not run"))
    with pytest.raises(ValueError, match="Invalid host or run ID"):
        downloader.download("g3", run_id, tmp_path)


def test_remote_library_checks_hash_before_loading_code(tmp_path, monkeypatch):
    original = tmp_path / "runs" / recovery.ORIGINAL
    original.mkdir(parents=True)
    script = tmp_path / "not-executable.py"
    script.write_text("raise AssertionError('must not execute')", encoding="utf-8")
    (original / "launch.json").write_text(json.dumps({
        "queue_script": str(script), "queue_script_sha256": "0" * 64,
    }), encoding="utf-8")
    monkeypatch.setattr(recovery, "BASE", tmp_path)
    monkeypatch.setattr(recovery.importlib.util, "spec_from_file_location",
                        lambda *a: pytest.fail("must not import before verification"))
    with pytest.raises(ValueError, match="hash mismatch"):
        recovery.remote_library()


def test_unpack_checked_rejects_archive_symlinks(tmp_path):
    link = tarfile.TarInfo("results/candidate_link")
    link.type = tarfile.SYMTYPE
    link.linkname = "../../outside"
    archive = _make_archive(tmp_path / "symlink.tar", [link])

    with pytest.raises(ValueError, match="Rejected archive member"):
        downloader.unpack_checked(archive, tmp_path / "downloads")

    assert not (tmp_path / "outside").exists()


def test_unpack_checked_rejects_duplicate_archive_members(tmp_path):
    archive = _make_archive(tmp_path / "duplicate.tar", [
        ("results/generation.json", b"same"),
        ("results/generation.json", b"same"),
    ])
    destination = tmp_path / "downloads"

    with pytest.raises(ValueError, match="Duplicate archive members"):
        downloader.unpack_checked(archive, destination)

    assert not (destination / "results" / "generation.json").exists()


def test_unpack_checked_rejects_different_preexisting_file_without_overwriting(tmp_path):
    archive = _make_archive(tmp_path / "changed.tar", [("results/candidate.pkl", b"new")])
    destination = tmp_path / "downloads"
    existing = destination / "results" / "candidate.pkl"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"local different bytes")

    with pytest.raises(ValueError, match="Refusing changed output"):
        downloader.unpack_checked(archive, destination)

    assert existing.read_bytes() == b"local different bytes"


def test_unpack_checked_accepts_identical_preexisting_file_and_extracts_new_files(tmp_path):
    archive = _make_archive(tmp_path / "identical.tar", [
        ("results/generation.json", b"same report"),
        ("results/candidate_000/generated.pkl", b"motion bytes"),
    ])
    destination = tmp_path / "downloads"
    existing = destination / "results" / "generation.json"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"same report")

    downloader.unpack_checked(archive, destination)

    assert existing.read_bytes() == b"same report"
    assert (destination / "results" / "candidate_000" / "generated.pkl").read_bytes() == b"motion bytes"


class _RecoveryIO:
    def __init__(self, base: Path):
        self.base = base
        self.snapshot = None
        self.launch_calls = []

    def owned_path(self, path):
        path = Path(path)
        path.relative_to(self.base)
        return path

    def verify_workspace(self, work):
        work = Path(work)
        assert (work / "_bundle.json").is_file()
        for entry in self.snapshot["files"]:
            path = work / entry["path"]
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        return self.snapshot

    @staticmethod
    def make_jobs(host):
        return deploy.make_jobs(host)

    @staticmethod
    def stamp():
        return "2026-10-08T00:00:00Z"

    @staticmethod
    def job_command(run, job):
        return ["mock-command", str(run), job["id"]]

    @staticmethod
    def write_json(path, value, exclusive=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
            json.dump(value, stream)

    def remote_launch(self, run):
        self.launch_calls.append(Path(run))
        return {"state": "prepared", "run_dir": str(run)}


def _prepare_recovery_fixture(tmp_path, monkeypatch):
    base = tmp_path / "remote batch"
    original = base / "runs" / recovery.ORIGINAL
    workspace = original / "workspace"
    source_file = workspace / "scripts" / "source.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"frozen source")
    bundle = {"source_sha256": "test-snapshot"}
    (workspace / "_bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    snapshot = {"files": [{
        "path": "scripts/source.py",
        "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
    }]}
    (original / "manifest.json").write_text(json.dumps({"jobs": ["original"]}), encoding="utf-8")
    (original / "checkpoint_probe.json").write_text(
        json.dumps({"ready": True, "device": "cpu"}), encoding="utf-8"
    )

    remote = _RecoveryIO(base)
    remote.snapshot = snapshot
    monkeypatch.setattr(recovery, "BASE", base)
    monkeypatch.setattr(recovery, "remote_library", lambda: remote)
    return base, original, remote


@pytest.mark.parametrize("job_ids", [[], ["not-a-job"], ["platform_v0_s20262008"] * 2])
def test_start_recovery_rejects_empty_unknown_or_duplicate_job_subsets(
        tmp_path, monkeypatch, job_ids):
    base, _, remote = _prepare_recovery_fixture(tmp_path, monkeypatch)
    run_id = "complete_20261008_invalid"

    with pytest.raises(ValueError, match="unique subset"):
        recovery.start_recovery(run_id, job_ids)

    assert not (base / "runs" / run_id).exists()
    assert remote.launch_calls == []


def test_start_recovery_uses_only_the_requested_original_jobs(tmp_path, monkeypatch):
    base, original, remote = _prepare_recovery_fixture(tmp_path, monkeypatch)
    jobs = deploy.make_jobs("g2") + deploy.make_jobs("g3")
    requested = [jobs[1]["id"], jobs[10]["id"]]
    run_id = "complete_20261008_subset"
    original_manifest = (original / "manifest.json").read_text(encoding="utf-8")

    result = recovery.start_recovery(run_id, requested)

    run = base / "runs" / run_id
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    status = json.loads((run / "queue_status.json").read_text(encoding="utf-8"))
    assert result == {"state": "prepared", "run_dir": str(run)}
    assert [job["id"] for job in manifest["jobs"]] == requested
    assert manifest["max_candidates"] == 16
    assert manifest["recovery_of"] == recovery.ORIGINAL
    assert all(job["command"][-1] == job_id for job, job_id in zip(manifest["jobs"], requested))
    assert status["state"] == "prepared" and status["generation_started"] is False
    assert len(remote.launch_calls) == 1 and remote.launch_calls[0] == run
    assert (run / "workspace" / "scripts" / "source.py").read_bytes() == b"frozen source"
    assert (original / "manifest.json").read_text(encoding="utf-8") == original_manifest
