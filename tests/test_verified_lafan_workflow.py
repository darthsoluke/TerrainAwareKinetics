"""Receipt gating and mocked orchestration for the verified LAFAN workflow."""

import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import run_verified_lafan_workflow as workflow


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_lafan_paper_command_skips_generic_root_pose_reviews_only(monkeypatch):
    module = ModuleType("scripts.run_paper_mb_workflow")
    module.paper_command = lambda stage, *args, **kwargs: ["python", "--stage", stage]
    monkeypatch.setitem(sys.modules, "scripts.run_paper_mb_workflow", module)

    tokenizer = workflow.paper_command("tokenizer", "data", "run", 3)
    root = workflow.paper_command("root", "data", "run", 0)
    pose = workflow.paper_command("pose", "data", "run", 3, tokenizer_run="tokenizer-run")
    root_keyword = workflow.paper_command(stage="root", dataset="data", output="run", gpu=0)
    assert "--skip-review" not in tokenizer
    assert "--skip-review" in root and "--skip-review" in pose and "--skip-review" in root_keyword


def accepted_source(root):
    records = []
    for split, subject in (("train", "subject1"), ("validation", "subject4"), ("test", "subject5")):
        relative = f"shards/{subject}.npz"
        shard = root / relative
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_bytes(subject.encode())
        records.append({"split": split, "subject_id": subject, "shard": relative,
                        "retarget_assessment": {"status": "passed"}})
    write(root / "accepted.json", records)
    review = {"status": "passed", "training_eligible": True,
              "source_accepted_sha256": digest(root / "accepted.json"),
              "accepted_contains_failed_clips": False,
              "source_inventory_quality_status": "partial_rejection",
              "accepted_shard_sha256": {row["shard"]: digest(root / row["shard"]) for row in records},
              "heldout_subjects": {"validation": ["subject4"], "test": ["subject5"]},
              "assessment_limitations": ["retarget review is not motion-quality approval"]}
    write(root / "retarget_review.json", review)
    return review


def test_receipt_validates_manifest_all_shards_and_subject_disjointness(tmp_path):
    review = accepted_source(tmp_path)
    result = workflow._source_acceptance(tmp_path, tmp_path / "retarget_review.json")
    assert result["accepted_sha256"] == review["source_accepted_sha256"]
    assert result["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert result["source_inventory_quality_status"] == "partial_rejection"

    shard = tmp_path / "shards/subject1.npz"
    shard.write_bytes(b"modified")
    with pytest.raises(ValueError, match="shard hash"):
        workflow._source_acceptance(tmp_path, tmp_path / "retarget_review.json")


@pytest.mark.parametrize("change, message", [
    (lambda r: r.update(status="failed"), "not machine-approved"),
    (lambda r: r.update(training_eligible=False), "not machine-approved"),
    (lambda r: r.update(source_accepted_sha256="wrong"), "does not bind"),
    (lambda r: r.update(accepted_shard_sha256={}), "hash every accepted shard"),
    (lambda r: r.update(accepted_contains_failed_clips=True), "no failed clips"),
    (lambda r: r.pop("accepted_contains_failed_clips"), "no failed clips"),
    (lambda r: r.update(heldout_subjects={"validation": ["subject3"], "test": ["subject5"]}),
     "must reserve"),
])
def test_receipt_rejects_unapproved_or_incomplete_acceptance(tmp_path, change, message):
    accepted_source(tmp_path)
    path = tmp_path / "retarget_review.json"
    review = json.loads(path.read_text())
    change(review)
    write(path, review)
    with pytest.raises(ValueError, match=message):
        workflow._source_acceptance(tmp_path, path)


@pytest.mark.parametrize("assessment", [None, {"status": "failed"}])
def test_receipt_rejects_missing_or_failed_per_clip_assessment(tmp_path, assessment):
    accepted_source(tmp_path)
    manifest_path = tmp_path / "accepted.json"
    rows = json.loads(manifest_path.read_text())
    if assessment is None:
        rows[0].pop("retarget_assessment")
    else:
        rows[0]["retarget_assessment"] = assessment
    write(manifest_path, rows)
    review_path = tmp_path / "retarget_review.json"
    review = json.loads(review_path.read_text())
    review["source_accepted_sha256"] = digest(manifest_path)
    write(review_path, review)
    with pytest.raises(ValueError, match="passed retarget_assessment"):
        workflow._source_acceptance(tmp_path, review_path)


def test_nonempty_splits_with_overlapping_subjects_are_rejected(tmp_path):
    review = accepted_source(tmp_path)
    rows = json.loads((tmp_path / "accepted.json").read_text())
    rows[0]["subject_id"] = "subject4"
    write(tmp_path / "accepted.json", rows)
    review["source_accepted_sha256"] = digest(tmp_path / "accepted.json")
    write(tmp_path / "retarget_review.json", review)
    with pytest.raises(ValueError, match="subject IDs must be disjoint"):
        workflow._source_acceptance(tmp_path, tmp_path / "retarget_review.json")


def test_workflow_uses_fresh_paper_stages_and_pose_waits_for_tokenizer(tmp_path, monkeypatch):
    source = tmp_path / "source"
    accepted_source(source)
    args = SimpleNamespace(source=source, review=source / "retarget_review.json",
        dataset=tmp_path / "dataset30", output=tmp_path / "run", preparation_pid=None,
        tokenizer_gpu=3, root_gpu=0, pose_gpu=3, review_gpu=4, timeout_days=1, poll_seconds=0)
    monkeypatch.setattr(workflow, "_source_hashes", lambda root: {"workflow.py": "source"})
    source_checks = []
    monkeypatch.setattr(workflow, "assert_sources", lambda *args: source_checks.append(args))
    monkeypatch.setattr(workflow, "verify_dataset", lambda *args: None)
    monkeypatch.setattr(workflow, "select_gpu", lambda index: {"index": index})
    monkeypatch.setattr(workflow, "require_technical_checks", lambda path: True)
    monkeypatch.setattr(workflow, "paper_command", lambda stage, dataset, output, gpu, tokenizer_run=None:
        ["python", "train.py", *( ["--stage", stage] if stage != "tokenizer" else []),
         "--training-profile", "paper", "--dataset", str(dataset), "--output", str(output),
         "--steps", str(workflow.PAPER_STEPS), "--batch-size", "256", "--gpu", str(gpu),
         *( ["--tokenizer-run", str(tokenizer_run)] if tokenizer_run else []),
         *( ["--skip-review"] if stage in ("root", "pose") else [])])
    monkeypatch.setattr(workflow, "frozen_tokenizer_identity", lambda run, step: {
        "checkpoint_sha256": "tokenizer-checkpoint", "tensor_state_sha256": "tokenizer-state",
        "run": str(run)})
    monkeypatch.setattr(workflow, "capture_stage_identity", lambda run, stage, dataset, manifest, token=None: {
        "dataset_manifest_sha256": manifest, "config_sha256": stage + "-config", "seed": 7,
        "batch_size": 256, "original_schedule_steps": workflow.PAPER_STEPS,
        "tokenizer_state_sha256": "tokenizer-state" if stage == "pose" else None})
    events = []

    class Process:
        pid = 123
        returncode = 0

        def __init__(self, stage):
            self.stage = stage

        def poll(self):
            return 0

    class Stream:
        def close(self):
            pass

    def fake_launch(command, code_root, log):
        if any("build_native_mb_dataset.py" in item for item in command):
            args.dataset.mkdir()
            write(args.dataset / "native_manifest.json", {"contract": {}, "artifact_sha256": {}})
            return Process("data"), Stream()
        if any("check_paper_mb_training.py" in item for item in command):
            destination = Path(command[command.index("--output") + 1])
            destination.mkdir()
            write(destination / "checks.json", {"status": "completed", "production_checkpoint": False,
                "stages": {stage: {"run": str(destination / stage), "checkpoint_sha256": stage}
                           for stage in ("tokenizer", "root", "pose")}})
            events.append(("technical_check", command))
            return Process("technical-check"), Stream()
        stage = command[command.index("--stage") + 1] if "--stage" in command else "tokenizer"
        events.append(("launch", stage, command))
        return Process(stage), Stream()

    def fake_verify(run, stage, initial, previous, **kwargs):
        events.append(("verified", stage))
        return {}, {"final_checkpoint_sha256": stage + "-checkpoint", "elapsed_seconds": 2,
                    "optimizer_updates": workflow.PAPER_STEPS, "optimizer_class": "torch.optim.adam.Adam"}

    monkeypatch.setattr(workflow, "launch", fake_launch)
    monkeypatch.setattr(workflow, "verify_workflow_run", fake_verify)
    monkeypatch.setattr(workflow, "run_lafan_final_review", lambda *a, **k: {
        "report": "review-result", "quality_approved": False,
        "wasd_continuous_review": {"status": "not_assessed"}})
    monkeypatch.setattr(workflow, "stop_owned", lambda processes: None)

    result = workflow.run_workflow(args)
    assert result["status"] == "completed"
    assert len(source_checks) > 1
    assert result["technical_check"]["status"] == "passed"
    assert result["technical_check"]["production_checkpoint"] is False
    assert result["quality_approved"] is False and result["ue_deployment"] is False
    checks = [event for event in events if event[0] == "technical_check"]
    assert len(checks) == 1
    check_command = checks[0][1]
    assert check_command[check_command.index("--gpu") + 1] == "3"
    assert check_command[check_command.index("--dataset") + 1] == str(args.dataset)
    assert check_command[check_command.index("--output") + 1] == str(args.output / "technical-check")
    launches = [event for event in events if event[0] == "launch"]
    assert [event[1] for event in launches] == ["tokenizer", "root", "pose"]
    assert events.index(("verified", "tokenizer")) < events.index(launches[2])
    assert events.index(checks[0]) < events.index(launches[0])
    assert launches[-1][2][launches[-1][2].index("--tokenizer-run") + 1] == str(
        args.output / f"tokenizer-{workflow.PAPER_STEPS}")
    for _, _, command in launches:
        assert command[command.index("--training-profile") + 1] == "paper"
        assert command[command.index("--steps") + 1] == str(workflow.PAPER_STEPS)
        assert command[command.index("--batch-size") + 1] == "256"
        assert ("--skip-review" in command) == ("--stage" in command)


def test_unapproved_receipt_does_not_create_any_output_or_launch(tmp_path, monkeypatch):
    source = tmp_path / "source"
    accepted_source(source)
    review_path = source / "retarget_review.json"
    review = json.loads(review_path.read_text())
    review["status"] = "not_approved"
    write(review_path, review)
    args = SimpleNamespace(source=source, review=review_path, dataset=tmp_path / "dataset",
        output=tmp_path / "run", preparation_pid=None, tokenizer_gpu=3, root_gpu=0,
        pose_gpu=3, review_gpu=4, timeout_days=1, poll_seconds=0)
    launched = []
    monkeypatch.setattr(workflow, "launch", lambda *a: launched.append(a))
    with pytest.raises(ValueError, match="not machine-approved"):
        workflow.run_workflow(args)
    assert not args.dataset.exists() and not args.output.exists() and not launched


def test_technical_check_requires_verified_artifacts_before_pass(tmp_path, monkeypatch):
    source, dataset, output, code_root = (tmp_path / name for name in
                                          ("source", "dataset", "run", "code"))
    source.mkdir()
    dataset.mkdir()
    output.mkdir()
    code_root.mkdir()
    accepted_source(source)
    write(dataset / "native_manifest.json", {})
    state = {"acceptance": workflow._source_acceptance(source, source / "retarget_review.json"),
             "dataset_manifest_sha256": digest(dataset / "native_manifest.json"),
             "stages": {stage: {"output": str(output / f"{stage}-2000000")}
                        for stage in ("tokenizer", "root", "pose")}}
    monkeypatch.setattr(workflow, "assert_sources", lambda *args: None)
    monkeypatch.setattr(workflow, "select_gpu", lambda index: {"index": index})
    monkeypatch.setattr(workflow, "stop_owned", lambda owned: None)

    class Process:
        returncode = 0

        def poll(self):
            return 0

    class Stream:
        def close(self):
            pass

    launched = []

    def fake_launch(command, root, log):
        launched.append(command)
        target = Path(command[command.index("--output") + 1])
        target.mkdir()
        write(target / "checks.json", {"status": "completed", "production_checkpoint": False,
            "stages": {stage: {"run": str(target / stage)}
                       for stage in ("tokenizer", "root", "pose")}})
        return Process(), Stream()

    monkeypatch.setattr(workflow, "launch", fake_launch)
    monkeypatch.setattr(workflow, "require_technical_checks", lambda path: False)
    with pytest.raises(ValueError, match="did not pass artifact verification"):
        workflow.run_lafan_technical_check(source, dataset, output, 3, code_root, {},
            workflow.time.monotonic() + 30, 0, state)
    assert len(launched) == 1 and any("check_paper_mb_training.py" in item for item in launched[0])
    assert state["technical_check"]["status"] == "running"


def test_missing_review_waits_without_requiring_accepted_manifest(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    assert workflow._source_acceptance(source, source / "retarget_review.json") is None


def test_preparation_failure_stops_wait_before_manifest_is_published(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write(source / "preparation_failure.json", {"error": "all clips rejected"})
    with pytest.raises(RuntimeError, match="all clips rejected"):
        workflow._raise_preparation_failure(source)


def test_linux_zombie_preparation_worker_is_not_considered_alive(monkeypatch):
    pid = 3811060
    monkeypatch.setattr(workflow.sys, "platform", "linux")
    monkeypatch.setattr(workflow.os, "kill", lambda *_: None)
    read_text = Path.read_text

    def fake_read_text(path, *args, **kwargs):
        if str(path) == f"/proc/{pid}/stat":
            return f"{pid} (worker (retarget)) Z 1 2 3"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    assert workflow._preparation_alive(pid) is False


def test_lafan_final_review_uses_walk_run_evaluator_and_unapproved_bundle(tmp_path, monkeypatch):
    source, dataset, output, code_root = (tmp_path / name for name in
                                          ("source", "dataset", "run", "code"))
    source.mkdir()
    dataset.mkdir()
    output.mkdir()
    code_root.mkdir()
    (source / "accepted.json").write_text("[]")
    (dataset / "native_manifest.json").write_text("{}")
    state = {"stages": {stage: {"output": str(output / stage), "checkpoint_sha256": stage + "-hash"}
                        for stage in ("tokenizer", "root", "pose")},
             "acceptance": {"accepted_sha256": workflow.sha256(source / "accepted.json"),
                            "shard_sha256": {}},
             "dataset_manifest_sha256": workflow.sha256(dataset / "native_manifest.json"),
             "frozen_tokenizer": {"checkpoint_sha256": "tokenizer-frozen",
                                  "tensor_state_sha256": "tokenizer-state"}}
    review = output / f"lafan-review-{workflow.PAPER_STEPS}"
    monkeypatch.setattr(workflow, "assert_sources", lambda *a: None)
    monkeypatch.setattr(workflow, "select_gpu", lambda gpu: {"index": gpu})
    monkeypatch.setattr(workflow, "verify_dataset", lambda *a: None)
    monkeypatch.setattr(workflow, "stop_owned", lambda owned: None)

    class Process:
        returncode = 0

        def poll(self):
            return 0

    class Stream:
        def close(self):
            pass

    launched = []

    def fake_launch(command, root, log):
        launched.append(command)
        destination = Path(command[command.index("--output") + 1])
        destination.mkdir()
        write(destination / "report.json", {"status": "completed", "quality_approved": False,
            "source_accepted_sha256": state["acceptance"]["accepted_sha256"],
            "dataset_manifest_sha256": state["dataset_manifest_sha256"],
            "subject_id": "subject4", "behaviors": ["walk", "run"],
            "window_count": 6, "windows": [{"subject_id": "subject4", "split": "validation",
                "behavior": behavior} for behavior in ("walk", "run")],
            "wasd_continuous_review": {"status": "not_assessed"}})
        write(destination / "candidate_bundle.json", {
            "status": "trained_evaluated_not_approved", "quality_approved": False,
            "dataset_manifest_sha256": state["dataset_manifest_sha256"],
            "paper_steps": workflow.PAPER_STEPS,
            "frozen_tokenizer": state["frozen_tokenizer"],
            "stages": {stage: {"checkpoint_sha256": stage + "-hash",
                                "optimizer_updates": workflow.PAPER_STEPS}
                       for stage in ("tokenizer", "root", "pose")}})
        return Process(), Stream()

    monkeypatch.setattr(workflow, "launch", fake_launch)
    result = workflow.run_lafan_final_review(source, dataset, output, 4, code_root, {},
        workflow.time.monotonic() + 30, 0, state)
    assert "scripts/evaluate_native_mb_lafan.py" in launched[0]
    assert "scripts/evaluate_native_mb_smart_locomotion.py" not in launched[0]
    assert result["quality_approved"] is False and result["window_count"] == 6
