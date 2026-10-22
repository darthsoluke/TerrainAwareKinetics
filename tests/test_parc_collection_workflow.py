"""Mocked integration tests for finite PARC collection completion."""

import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import deploy_parc_remote_batch as deploy
import finish_parc_collection as collection


LOCAL_RUN_ID = "complete_20261008_extensions_r1"
REMOTE_RUN_ID = "complete_20261008_remote_r1"


def _remote_jobs(states=None):
    jobs = deploy.make_jobs("g2") + deploy.make_jobs("g3")
    states = states or {}
    return [{"id": job["id"], "state": states.get(job["id"], "completed")}
            for job in jobs]


def _configure_collection(tmp_path, monkeypatch, *, remote_state="completed",
                          remote_job_states=None, local_state="completed",
                          local_job_count=6, candidate_count=160,
                          render_results=None, catalog_completed_job_ids=(),
                          local_original_jobs=(), local_backup_state="running"):
    output = tmp_path / "parc collection"
    bootstrap = tmp_path / "out" / "parc_remote_deployment_20261008"
    bootstrap.mkdir(parents=True)
    (bootstrap / "finish_bootstrap.json").write_text(
        json.dumps({"script": "/mock/finish.py"}), encoding="utf-8"
    )
    monkeypatch.setattr(deploy, "ROOT", tmp_path)
    monkeypatch.setattr(collection.fetch, "download",
                        lambda *args, **kwargs: pytest.fail("unexpected remote download"))

    remote_jobs = _remote_jobs(remote_job_states)
    inventory = {"runs": [{
        "run_id": REMOTE_RUN_ID,
        "state": remote_state,
        "jobs": remote_jobs,
    }]}
    monkeypatch.setattr(deploy, "ssh", lambda *args, **kwargs: ["mock-ssh"])
    monkeypatch.setattr(deploy, "checked_run", lambda *args, **kwargs: json.dumps(inventory))

    remote_download = output / "downloads" / "g3" / REMOTE_RUN_ID
    remote_download.mkdir(parents=True)
    (remote_download / "download_manifest.json").write_text(json.dumps({
        "jobs": [job["id"] for job in remote_jobs if job["state"] == "completed"]
    }), encoding="utf-8")

    local_run = output / "downloads" / "local" / LOCAL_RUN_ID
    local_run.mkdir(parents=True)
    local_jobs = [{"id": f"extension_{index}", "state": local_state}
                  for index in range(local_job_count)]
    (local_run / "queue_status.json").write_text(json.dumps({
        "state": local_state,
        "jobs": local_jobs,
    }), encoding="utf-8")
    if local_original_jobs:
        backup_run = output / "downloads" / "local" / "original_backup"
        backup_run.mkdir(parents=True)
        (backup_run / "queue_status.json").write_text(json.dumps({
            "state": local_backup_state,
            "jobs": local_original_jobs,
        }), encoding="utf-8")

    candidates = [{
        "candidate_key": f"local/job/candidate_{index:03d}",
        "report_status": "kinematic_candidates_generated_physics_pending",
    } for index in range(candidate_count)]
    catalog_jobs = [{
        "job_id": job_id,
        "report_status": "kinematic_candidates_generated_physics_pending",
    } for job_id in catalog_completed_job_ids]
    monkeypatch.setattr(collection.gallery, "build_catalog",
                        lambda *_: {"candidates": candidates, "jobs": catalog_jobs})
    run_results = render_results or [{
        "candidate_count": candidate_count,
        "rendered": 0,
        "cached": candidate_count,
        "failed": 0,
        "gallery_path": str(output / "gallery" / "index.html"),
    }]
    run_calls = []

    def run_batch(*args, **kwargs):
        run_calls.append((args, kwargs))
        if len(run_results) > 1:
            return run_results.pop(0)
        return run_results[0]

    monkeypatch.setattr(collection.gallery, "run_batch", run_batch)
    return output, inventory, run_calls


def test_complete_fourteen_remote_and_six_local_jobs_requires_160_rendered_candidates(
        tmp_path, monkeypatch):
    output, _, run_calls = _configure_collection(tmp_path, monkeypatch, render_results=[
        {"candidate_count": 160, "rendered": 159, "cached": 1, "failed": 0,
         "gallery_path": str(tmp_path / "gallery" / "index.html")},
        {"candidate_count": 160, "rendered": 0, "cached": 160, "failed": 0,
         "gallery_path": str(tmp_path / "gallery" / "index.html")},
    ])

    result = collection.finish(output, timeout_minutes=1)

    progress = json.loads((output / "completion_progress.json").read_text(encoding="utf-8"))
    assert result["state"] == "generated_and_rendered"
    assert result["total_candidates"] == 160
    assert progress["original_completed_jobs"] == progress["original_expected_jobs"] == 14
    assert progress["local_completed_jobs"] == progress["local_expected_jobs"] == 6
    assert progress["generation_complete"] is True
    assert "remote_completed_jobs" not in progress
    assert len(run_calls) == 2
    assert all(call[1]["mode"] == "render-existing" for call in run_calls)


@pytest.mark.parametrize("final_render", [
    {"candidate_count": 159, "rendered": 0, "cached": 159, "failed": 0},
    {"candidate_count": 160, "rendered": 0, "cached": 160, "failed": 1},
])
def test_missing_or_failed_render_does_not_certify_complete_collection(
        tmp_path, monkeypatch, final_render):
    first_render = {
        "candidate_count": 160, "rendered": 0, "cached": 160, "failed": 0,
    }
    output, _, _ = _configure_collection(
        tmp_path, monkeypatch, render_results=[first_render, final_render]
    )

    with pytest.raises(RuntimeError, match="Incomplete render collection"):
        collection.finish(output, timeout_minutes=1)

    progress = json.loads((output / "completion_progress.json").read_text(encoding="utf-8"))
    assert progress["generation_complete"] is True
    assert progress.get("state") != "generated_and_rendered"


def test_queued_and_launched_awaiting_cuda_remote_jobs_remain_active(tmp_path, monkeypatch):
    planned = deploy.make_jobs("g2") + deploy.make_jobs("g3")
    states = {job["id"]: "queued" for job in planned}
    states[planned[1]["id"]] = "launched_awaiting_cuda"
    output, _, _ = _configure_collection(
        tmp_path, monkeypatch, remote_state="running", remote_job_states=states,
        local_state="running", candidate_count=0,
    )
    clock = [0.0]
    monkeypatch.setattr(collection.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(collection.time, "sleep", lambda _: clock.__setitem__(0, 1.0))

    with pytest.raises(TimeoutError, match="deadline reached"):
        collection.finish(output, timeout_minutes=0.001)

    progress = json.loads((output / "completion_progress.json").read_text(encoding="utf-8"))
    assert progress["original_completed_jobs"] == 0
    assert progress["generation_complete"] is False


@pytest.mark.parametrize("local_state", ["queued", "running"])
def test_queued_or_running_local_original_backup_job_remains_active(
        tmp_path, monkeypatch, local_state):
    original_id = (deploy.make_jobs("g2") + deploy.make_jobs("g3"))[0]["id"]
    states = {job["id"]: "queued" for job in _remote_jobs()}
    output, _, _ = _configure_collection(
        tmp_path, monkeypatch, remote_state="cancelled_for_reassignment",
        remote_job_states=states, local_state="running", candidate_count=0,
        local_original_jobs=[{"id": original_id, "state": local_state}],
    )
    clock = [0.0]
    monkeypatch.setattr(collection.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(collection.time, "sleep", lambda _: clock.__setitem__(0, 1.0))

    with pytest.raises(TimeoutError, match="deadline reached"):
        collection.finish(output, timeout_minutes=0.001)

    progress = json.loads((output / "completion_progress.json").read_text(encoding="utf-8"))
    assert progress["generation_complete"] is False


def test_successful_local_backup_report_completes_cancelled_original_job(
        tmp_path, monkeypatch):
    planned = deploy.make_jobs("g2") + deploy.make_jobs("g3")
    recovered_id = planned[0]["id"]
    remote_states = {job["id"]: "completed" for job in planned}
    remote_states[recovered_id] = "cancelled"
    output, _, _ = _configure_collection(
        tmp_path, monkeypatch, remote_state="cancelled_for_reassignment",
        remote_job_states=remote_states, candidate_count=160,
        catalog_completed_job_ids=[recovered_id],
        local_original_jobs=[{"id": recovered_id, "state": "completed"}],
        local_backup_state="completed",
    )

    result = collection.finish(output, timeout_minutes=1)

    progress = json.loads((output / "completion_progress.json").read_text(encoding="utf-8"))
    assert result["state"] == "generated_and_rendered"
    assert progress["original_completed_jobs"] == 14
    assert progress["original_expected_jobs"] == 14
    assert progress["generation_complete"] is True


def test_failed_local_generation_is_reported_without_certification(tmp_path, monkeypatch):
    queued = {job["id"]: "queued" for job in _remote_jobs()}
    output, _, _ = _configure_collection(
        tmp_path, monkeypatch, remote_state="running", remote_job_states=queued,
        local_state="finished_with_errors", candidate_count=0,
    )

    with pytest.raises(RuntimeError, match="Local generation has failed jobs"):
        collection.finish(output, timeout_minutes=1)


def test_terminated_remote_run_is_reported_without_certification(tmp_path, monkeypatch):
    states = {job["id"]: "queued" for job in _remote_jobs()}
    output, _, _ = _configure_collection(
        tmp_path, monkeypatch, remote_state="cancelled_for_reassignment",
        remote_job_states=states,
        local_state="running", candidate_count=0,
    )

    with pytest.raises(RuntimeError, match="Remote queue stopped before completion"):
        collection.finish(output, timeout_minutes=1)
