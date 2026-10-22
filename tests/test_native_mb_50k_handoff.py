import hashlib
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_lafan_50k_handoff as handoff


@pytest.fixture
def mock_process(monkeypatch):
    identity = {"pid": 4321, "pgid": 4321, "proc_start_ticks": "123456",
                "command": "python -u scripts/train_native_mb.py --steps 2000000"}
    current = {**identity, "uid": 1001, "state": "T", "exists": True}
    signals = []

    class ProcFile:
        def __init__(self, name):
            self.name = name

        def read_text(self):
            assert self.name == "stat"
            fields = ["0"] * 50
            fields[0] = current["state"]
            fields[19] = str(current["proc_start_ticks"])
            return f"{identity['pid']} (worker (training) thread) " + " ".join(fields)

        def read_bytes(self):
            assert self.name == "cmdline"
            return current["command"].replace(" ", "\0").encode() + b"\0"

    class ProcDirectory:
        def exists(self):
            return current["exists"]

        def stat(self):
            return SimpleNamespace(st_uid=current["uid"])

        def __truediv__(self, name):
            return ProcFile(name)

    class ProcRoot:
        def __truediv__(self, pid):
            assert pid == str(identity["pid"])
            return ProcDirectory()

    def proc_path(path):
        assert path == "/proc"
        return ProcRoot()

    def getpgid(pid):
        assert pid == identity["pid"]
        return current["pgid"]

    monkeypatch.setattr(handoff, "Path", proc_path)
    monkeypatch.setattr(handoff, "os", SimpleNamespace(
        getuid=lambda: 1001, getpgid=getpgid,
        killpg=lambda pgid, sig: signals.append((pgid, sig)),
    ))
    return identity, current, signals


def test_verified_signal_requires_same_uid_start_ticks_group_and_command(mock_process):
    identity, _, signals = mock_process
    assert handoff.process_state(identity) == "T"

    handoff.signal_verified(identity, signal.SIGTERM)

    assert signals == [(identity["pgid"], signal.SIGTERM)]


@pytest.mark.parametrize(("field", "changed"), [
    ("proc_start_ticks", "999999"), ("uid", 2002), ("pgid", 9876),
    ("command", "python unrelated_job.py"),
], ids=["pid-reuse", "wrong-uid", "wrong-group", "wrong-command"])
def test_identity_mismatch_refuses_signal(mock_process, field, changed):
    identity, current, signals = mock_process
    current[field] = changed

    with pytest.raises(ValueError, match="identity changed; refusing signal"):
        handoff.signal_verified(identity, signal.SIGTERM)

    assert signals == []


@pytest.mark.parametrize("state", ["missing", "zombie"])
def test_exited_or_zombie_process_is_not_signaled(mock_process, state):
    identity, current, signals = mock_process
    if state == "missing":
        current["exists"] = False
    else:
        current["state"] = "Z"

    assert handoff.process_state(identity) is None
    handoff.signal_verified(identity, signal.SIGTERM)
    handoff.terminate_verified(identity)

    assert signals == []


def test_each_signal_rechecks_identity_after_pid_reuse(mock_process):
    identity, current, signals = mock_process
    handoff.signal_verified(identity, signal.SIGTERM)
    current["proc_start_ticks"] = "reused-pid-start"

    with pytest.raises(ValueError, match="refusing signal"):
        handoff.signal_verified(identity, signal.SIGTERM)

    assert signals == [(identity["pgid"], signal.SIGTERM)]


@pytest.fixture
def pose_run(tmp_path):
    run = tmp_path / "pose-50000"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints/final.ckpt"
    checkpoint.write_bytes(b"synthetic pose checkpoint; no training")
    tokenizer = {"checkpoint_sha256": "selected-tokenizer-hash",
                 "dataset_manifest_sha256": "native-manifest-hash"}
    contract = {
        "status": "trained_not_reviewed", "stage": "native_pose",
        "original_schedule_steps": 2_000_000,
        "tokenizer": {"checkpoint_sha256": tokenizer["checkpoint_sha256"]},
        "dataset_manifest_sha256": tokenizer["dataset_manifest_sha256"],
        "tokenizer_state_sha256": "frozen-tokenizer-state",
    }
    result = {
        "optimizer_updates": 50_000, "updates_this_run": 50_000, "skipped_batches": 0,
        "tokenizer_state_sha256": contract["tokenizer_state_sha256"],
        "final_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    return run, tokenizer, contract, result


def write_pose_metadata(run, contract, result):
    (run / "run_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (run / "training_result.json").write_text(json.dumps(result), encoding="utf-8")


@pytest.mark.parametrize("status", ["completed", "trained_not_reviewed"])
def test_pose_result_accepts_exact_updates_and_preserved_two_million_schedule(pose_run, status):
    run, tokenizer, contract, result = pose_run
    contract["status"] = status
    write_pose_metadata(run, contract, result)

    assert handoff.verify_pose_result(run, 50_000, tokenizer) == result


@pytest.mark.parametrize("changed", [
    "status", "stage", "updates", "new_updates", "skipped", "schedule",
    "tokenizer_checkpoint", "manifest", "missing_tokenizer_state", "tokenizer_state",
    "checkpoint_bytes",
])
def test_pose_result_rejects_changed_updates_schedule_or_frozen_dependency(pose_run, changed):
    run, tokenizer, contract, result = pose_run
    if changed == "status":
        contract["status"] = "running"
    elif changed == "stage":
        contract["stage"] = "native_root"
    elif changed == "updates":
        result["optimizer_updates"] -= 1
    elif changed == "new_updates":
        result["updates_this_run"] -= 1
    elif changed == "skipped":
        result["skipped_batches"] = 1
    elif changed == "schedule":
        contract["original_schedule_steps"] = 50_000
    elif changed == "tokenizer_checkpoint":
        contract["tokenizer"]["checkpoint_sha256"] = "another-tokenizer"
    elif changed == "manifest":
        contract["dataset_manifest_sha256"] = "another-dataset"
    elif changed == "missing_tokenizer_state":
        del contract["tokenizer_state_sha256"]
    elif changed == "tokenizer_state":
        result["tokenizer_state_sha256"] = "unfrozen-tokenizer-state"
    else:
        (run / "checkpoints/final.ckpt").write_bytes(b"changed pose checkpoint")
    write_pose_metadata(run, contract, result)

    with pytest.raises(ValueError, match="binding failed"):
        handoff.verify_pose_result(run, 50_000, tokenizer)
