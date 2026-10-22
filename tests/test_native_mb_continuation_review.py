"""Review runs only on paired, immutable native training milestones."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import review_native_mb_continuation as reviewer
from scripts.review_native_mb_continuation import evaluation_commands, paired_milestone_ready, training_failure
from scripts.train_native_mb import write_json


def test_incomplete_or_partially_written_training_never_starts_review(tmp_path):
    assert not paired_milestone_ready(tmp_path, 100000)
    for stage in ("root", "pose"):
        path = tmp_path / f"{stage}-100000"
        path.mkdir()
        write_json(path / "run_contract.json", {"stage": "native_" + stage, "stop_steps": 100000,
                                                "status": "completed" if stage == "root" else "running"})
    assert not paired_milestone_ready(tmp_path, 100000)
    (tmp_path / "pose-100000/run_contract.json").write_text('{"status":')
    assert not paired_milestone_ready(tmp_path, 100000)
    write_json(tmp_path / "pose-100000/run_contract.json", {"stage": "native_pose", "stop_steps": 100000,
                                                          "status": "completed"})
    assert paired_milestone_ready(tmp_path, 100000)


def test_wrong_pair_and_reported_training_failure_are_not_success(tmp_path):
    (tmp_path / "root-100000").mkdir()
    write_json(tmp_path / "root-100000/run_contract.json", {"stage": "native_pose", "stop_steps": 100000,
                                                          "status": "completed"})
    with pytest.raises(ValueError, match="identity"):
        paired_milestone_ready(tmp_path, 100000)
    assert training_failure(tmp_path) is None
    write_json(tmp_path / "pose_state.json", {"status": "failed", "error": "nonfinite gradient"})
    assert training_failure(tmp_path) == "nonfinite gradient"


def test_evaluators_receive_same_checkpoint_pair_and_truthful_renderer_input(tmp_path):
    commands = evaluation_commands(tmp_path, 220000, 4)
    for command in commands[:2]:
        assert command[command.index("--root-run") + 1] == str(tmp_path / "root-220000")
        assert command[command.index("--pose-run") + 1] == str(tmp_path / "pose-220000")
        assert command[command.index("--gpu") + 1] == "4"
    assert "--include-fixed8" in commands[1]
    assert commands[2][2] == "scripts/render_native_mb_review.py"
    assert commands[2][commands[2].index("--review") + 1] == str(tmp_path / "review-220000/inbetween")


def _completed_pair(base, step):
    for stage in ("root", "pose"):
        path = base / f"{stage}-{step}"
        path.mkdir()
        write_json(path / "run_contract.json", {"stage": "native_" + stage,
                                                "stop_steps": step, "status": "completed"})


def test_orchestration_waits_runs_both_reviews_renders_and_keeps_quality_unapproved(tmp_path, monkeypatch):
    commands = []
    gpu_checks = []
    sleeps = []

    def finish_training(seconds):
        sleeps.append(seconds)
        for step in reviewer.MILESTONES:
            _completed_pair(tmp_path, step)

    def idle_gpu(index):
        gpu_checks.append(index)
        return {"index": index, "utilization_percent": 0}

    def run(command, **kwargs):
        commands.append(command)
        step = 100000 if "100000" in " ".join(command) else 220000
        if "--output" in command:
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
            if output.name == "inbetween":
                write_json(output / "metrics.json", {"summary": {"world_fk_mpjpe_m": .1}})
            else:
                write_json(output / "report.json", {"bundle": {"root_step": step, "pose_step": step},
                                                    "summary": {"full_native": {"finite": True}}})
        else:
            review = Path(command[command.index("--review") + 1])
            assert (review / "metrics.json").is_file()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(reviewer.time, "sleep", finish_training)
    monkeypatch.setattr(reviewer, "select_gpu", idle_gpu)
    monkeypatch.setattr(reviewer.subprocess, "run", run)
    state = reviewer.review_milestones(tmp_path, 4, poll_seconds=.01)
    assert sleeps == [.01]
    assert len(commands) == 6
    assert gpu_checks == [4, 4, 4, 4]
    assert [row["step"] for row in state["completed"]] == [100000, 220000]
    assert state["status"] == "completed"
    assert not state["quality_approved"]
    for step in reviewer.MILESTONES:
        summary = json.loads((tmp_path / f"review-{step}/summary.json").read_text())
        assert summary["step"] == step
        assert not summary["quality_approved"]
        assert len(list((tmp_path / f"review-{step}").glob("command-*.log"))) == 3


def test_failed_evaluation_stops_without_claiming_a_reviewed_milestone(tmp_path, monkeypatch):
    _completed_pair(tmp_path, 100000)
    monkeypatch.setattr(reviewer, "select_gpu", lambda index: {"index": index})
    monkeypatch.setattr(reviewer.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="Evaluation exited 1"):
        reviewer.review_milestones(tmp_path, 4)
    state = json.loads((tmp_path / "review_state.json").read_text())
    assert state["status"] == "failed"
    assert state["completed"] == []
    assert not state["quality_approved"]
    assert not (tmp_path / "review-100000/summary.json").exists()
