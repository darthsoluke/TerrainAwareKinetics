"""Queue state remains truthful if local inference or snapshot checks fail."""

import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_parc_local_extensions as queue


@pytest.mark.parametrize("failure", ("timeout", "snapshot"))
def test_local_queue_records_failures(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(queue.deploy, "source_files", lambda root: [])
    monkeypatch.setattr(queue.deploy, "checked_run", lambda *a, **kw: "5000")
    def subprocess_run(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired("fixture", 7200)
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(queue.subprocess, "run", subprocess_run)
    def verify(work):
        if failure == "snapshot":
            raise RuntimeError("snapshot fixture mismatch")
    monkeypatch.setattr(queue.deploy, "verify_workspace", verify)
    output = tmp_path / "run"
    result = queue.run(output, original_jobs=["platform_v0_s20262008"])
    saved = json.loads((output / "queue_status.json").read_text())
    assert saved == result
    assert saved["state"] == "finished_with_errors"
    if failure == "timeout":
        assert saved["jobs"][0]["state"] == "failed"
        assert "TimeoutExpired" in saved["jobs"][0]["error"]
    else:
        assert saved["source_snapshot_verified"] is False
        assert "snapshot fixture mismatch" in saved["error"]
