"""The scheduler must preserve native dependencies, failure states and update targets."""

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_native_mb_workflow as workflow


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_dependency_dag_waits_for_final_tokenizer_not_old_pose(tmp_path):
    stages = {"tokenizer": {"status": "running"}, "pose": {"status": "completed"}}
    assert workflow.ready_to_launch("tokenizer", stages, tmp_path)
    assert not workflow.ready_to_launch("pose", stages, tmp_path)
    stages["tokenizer"]["status"] = "completed"
    assert workflow.ready_to_launch("pose", stages, tmp_path)


def test_root_waits_for_completed_220k_and_propagates_upstream_failure(tmp_path):
    assert not workflow.ready_to_launch("root", {}, tmp_path)
    write(tmp_path / "root-220000/run_contract.json",
          {"status": "running", "stage": "native_root", "stop_steps": 220000})
    assert not workflow.ready_to_launch("root", {}, tmp_path)
    write(tmp_path / "root-220000/run_contract.json",
          {"status": "completed", "stage": "native_root", "stop_steps": 220000})
    assert workflow.ready_to_launch("root", {}, tmp_path)
    write(tmp_path / "root_state.json", {"status": "failed", "error": "nonfinite"})
    with pytest.raises(RuntimeError, match="nonfinite"):
        workflow.ready_to_launch("root", {}, tmp_path)


def test_root_rejects_wrong_completed_checkpoint(tmp_path):
    write(tmp_path / "root-220000/run_contract.json",
          {"status": "completed", "stage": "native_pose", "stop_steps": 100000})
    with pytest.raises(ValueError, match="identity"):
        workflow.ready_to_launch("root", {}, tmp_path)


def test_atomic_state_and_partial_upstream_writes(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"status":')
    assert workflow.read_pending(path) is None
    workflow.save_state(path, {"status": "running"})
    assert workflow.read_pending(path) == {"status": "running"}
    assert not path.with_suffix(".json.tmp").exists()


def test_source_identity_is_checked(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("unchanged\n")
    expected = {"source.py": workflow.source_lf_sha256(path)}
    workflow.assert_sources(tmp_path, expected)
    path.write_text("changed\n")
    with pytest.raises(ValueError, match="source changed"):
        workflow.assert_sources(tmp_path, expected)


def mock_runner(tmp_path, monkeypatch, *, failing_stage=None):
    args = SimpleNamespace(output=tmp_path / "result", pilot=tmp_path / "pilot",
                           tokenizer_resume=tmp_path / "tokenizer-old", tokenizer_gpu=5,
                           root_gpu=0, pose_gpu=3, review_gpu=4, timeout_hours=1, poll_seconds=0)
    template = {"dataset": str(tmp_path / "dataset"), "dataset_manifest_sha256": "data",
                "original_schedule_steps": 2000001, "batch_size": 32, "seed": 7,
                "config_sha256": "configuration", "tokenizer_state_sha256": "old-code-space"}
    monkeypatch.setattr(workflow, "load_initial", lambda run, stage: (dict(template), {}))
    monkeypatch.setattr(workflow, "validate_common_identity", lambda value: None)
    monkeypatch.setattr(workflow, "verify_resume_training_sources", lambda *args: None)
    monkeypatch.setattr(workflow, "validate_native_resume", lambda *args: {"saved_step": 100000})
    monkeypatch.setattr(workflow, "validate_resume", lambda *args: {"saved_step": 220000})
    monkeypatch.setattr(workflow, "source_lf_sha256", lambda *args: "source")
    monkeypatch.setattr(workflow, "select_gpu", lambda index: {"index": index})
    monkeypatch.setattr(workflow, "frozen_tokenizer_identity", lambda run: {
        "checkpoint_sha256": "new-checkpoint", "tensor_state_sha256": "new-code-space", "run": str(run)})
    write(args.pilot / "root-220000/run_contract.json",
          {"status": "completed", "stage": "native_root", "stop_steps": 220000})
    events = []
    monkeypatch.setattr(workflow, "stop_owned", lambda processes: events.append(("cleanup", tuple(processes))))

    class Process:
        pid = 123

        def __init__(self, stage):
            self.stage = stage

        def poll(self):
            return 1 if self.stage == failing_stage else 0

    def launch(command, code_root, log):
        stage = command[command.index("--stage") + 1] if "--stage" in command else "tokenizer"
        events.append(("launch", stage, command))
        return Process(stage), SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(workflow, "launch", launch)

    def verify(run, stage, initial, previous_step, **kwargs):
        events.append(("verify", stage, previous_step))
        if stage == "pose":
            assert initial["tokenizer_state_sha256"] == "new-code-space"
            assert kwargs["tokenizer_checkpoint_sha256"] == "new-checkpoint"
            assert previous_step == 0
        return {}, {"final_checkpoint_sha256": stage + "-hash", "elapsed_seconds": 1}

    monkeypatch.setattr(workflow, "verify_workflow_run", verify)

    def review(*args):
        events.append(("review",))
        return "fixed-native-review"

    monkeypatch.setattr(workflow, "run_final_review", review)
    return args, events


def test_entire_dag_uses_original_entries_fresh_pose_and_then_review(tmp_path, monkeypatch):
    args, events = mock_runner(tmp_path, monkeypatch)
    state = workflow.run_workflow(args)
    assert state["status"] == "completed" and state["manual_gates"] is False
    assert state["quality_approved"] is False and state["ue_deployment"] is False
    launches = [event for event in events if event[0] == "launch"]
    assert [event[1] for event in launches] == ["tokenizer", "root", "pose"]
    assert events.index(("verify", "tokenizer", 100000)) < events.index(launches[2])
    pose = launches[2][2]
    assert "--resume-run" not in pose
    assert Path(pose[pose.index("--tokenizer-run") + 1]) == args.output / "tokenizer-2000001"
    for _, stage, command in launches:
        assert command[command.index("--steps") + 1] == "2000001"
        assert command[command.index("--batch-size") + 1] == "32"
        assert "--skip-review" not in command
    assert events.index(("verify", "pose", 0)) < events.index(("review",))


def test_native_failure_does_not_launch_pose_or_report_success(tmp_path, monkeypatch):
    args, events = mock_runner(tmp_path, monkeypatch, failing_stage="tokenizer")
    with pytest.raises(RuntimeError, match="tokenizer exited 1"):
        workflow.run_workflow(args)
    state = json.loads((args.output / "workflow_state.json").read_text())
    assert state["status"] == "failed"
    assert not any(event[:2] == ("launch", "pose") for event in events)
    assert ("review",) not in events
    assert events[-1] == ("cleanup", ("root",))


def test_busy_gpu_waits_without_changing_training_command(tmp_path, monkeypatch):
    args, events = mock_runner(tmp_path, monkeypatch)
    checked = []

    def select(index):
        checked.append(index)
        if len(checked) == 1:
            raise RuntimeError("occupied")
        return {"index": index}

    monkeypatch.setattr(workflow, "select_gpu", select)
    assert workflow.run_workflow(args)["status"] == "completed"
    assert checked.count(5) == 2
    assert len([event for event in events if event[0] == "launch"]) == 3


@pytest.mark.parametrize("tokenizer_step", [2000001, 100000])
def test_final_review_checks_all_three_native_steps_and_keeps_candidate_unapproved(
        tmp_path, monkeypatch, tokenizer_step):
    events = []
    monkeypatch.setattr(workflow, "assert_sources", lambda *args: None)
    monkeypatch.setattr(workflow, "select_gpu", lambda index: {"index": index})
    monkeypatch.setattr(workflow, "stop_owned", lambda processes: None)

    def launch(command, code_root, log):
        events.append(command[2])
        if "--output" in command:
            output = Path(command[command.index("--output") + 1])
            if "smart_locomotion" in command[2]:
                write(output / "report.json", {"summary": {}, "bundle": {
                    "root_step": 2000001, "pose_step": 2000001,
                    "tokenizer": {"global_step": tokenizer_step}}})
            else:
                write(output / "metrics.json", {"summary": {}})
        return SimpleNamespace(wait=lambda **kwargs: 0), SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(workflow, "launch", launch)
    state = {"dataset": "/dataset", "dataset_manifest_sha256": "dataset-hash",
             "frozen_tokenizer": {"checkpoint_sha256": "tokenizer-hash"},
             "stages": {stage: {"output": "/" + stage, "checkpoint_sha256": stage + "-hash"}
                        for stage in ("tokenizer", "root", "pose")}}
    if tokenizer_step != 2000001:
        with pytest.raises(ValueError, match="different native bundle"):
            workflow.run_final_review(tmp_path, 4, tmp_path, {}, workflow.time.monotonic() + 60, 0, state)
        assert not (tmp_path / "candidate_bundle.json").exists()
    else:
        workflow.run_final_review(tmp_path, 4, tmp_path, {}, workflow.time.monotonic() + 60, 0, state)
        bundle = json.loads((tmp_path / "candidate_bundle.json").read_text())
        assert bundle["quality_approved"] is False and bundle["ue_deployment"] is False
        assert bundle["status"] == "trained_evaluated_not_approved"
        assert events == ["scripts/evaluate_native_mb_inbetween.py",
                          "scripts/evaluate_native_mb_smart_locomotion.py",
                          "scripts/render_native_mb_review.py"]
