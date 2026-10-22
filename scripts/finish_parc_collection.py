"""Collect and render the finite PARC jobs while their generators finish."""

import argparse
import json
from pathlib import Path
import time

import deploy_parc_remote_batch as deploy
import download_parc_batch_results as fetch
import render_parc_batch_gallery as gallery


def finish(output, timeout_minutes=90):
    output = output.resolve()
    bootstrap = json.loads((deploy.ROOT / "out/parc_remote_deployment_20261008/finish_bootstrap.json").read_text())
    expected = {j["id"] for host in ("g2", "g3") for j in deploy.make_jobs(host)}
    local_run = output / "downloads/local/complete_20261008_extensions_r1"
    deadline = time.monotonic() + timeout_minutes * 60
    last_rendered_keys = set()
    while time.monotonic() < deadline:
        inventory = json.loads(deploy.checked_run(deploy.ssh("g3", [
            "python3", "-B", bootstrap["script"], "inventory"]), timeout=90))
        deploy.write_json(output / "remote_inventory.json", inventory)
        complete = set()
        for run in inventory["runs"]:
            completed = {j["id"] for j in run["jobs"] if j["state"] == "completed"}
            complete.update(completed)
            manifest_path = output / "downloads/g3" / run["run_id"] / "download_manifest.json"
            saved = set(json.loads(manifest_path.read_text())["jobs"]) if manifest_path.exists() else set()
            if completed - saved:
                fetch.download("g3", run["run_id"], output)
        local = json.loads((local_run / "queue_status.json").read_text())
        local_done = sum(j["state"] == "completed" for j in local["jobs"])
        catalog = gallery.build_catalog(output / "downloads")
        for job in catalog["jobs"]:
            if job["report_status"] == "kinematic_candidates_generated_physics_pending":
                complete.add(job["job_id"])
        ready = {c["candidate_key"] for c in catalog["candidates"]
                 if c["report_status"] == "kinematic_candidates_generated_physics_pending"}
        if ready - last_rendered_keys:
            result = gallery.run_batch(output / "downloads", output / "gallery", mode="render-existing")
            deploy.write_json(output / "last_render_result.json", result)
            last_rendered_keys = ready
            print(json.dumps({"render": result}), flush=True)
        progress = {"at": deploy.stamp(), "original_completed_jobs": len(complete & expected),
                    "original_expected_jobs": len(expected), "local_completed_jobs": local_done,
                    "local_expected_jobs": len(local["jobs"]), "downloaded_candidates": len(ready),
                    "generation_complete": expected <= complete and local["state"] == "completed",
                    "physics_verified": False, "training_approved": False}
        deploy.write_json(output / "completion_progress.json", progress)
        print(json.dumps(progress), flush=True)
        if progress["generation_complete"]:
            result = gallery.run_batch(output / "downloads", output / "gallery", mode="render-existing")
            wanted = 8 * (len(expected) + len(local["jobs"]))
            if result["candidate_count"] != wanted or result["failed"] or result["cached"] != wanted:
                raise RuntimeError(f"Incomplete render collection, expected {wanted}: {result}")
            progress.update(state="generated_and_rendered", total_candidates=wanted,
                            gallery=result["gallery_path"])
            deploy.write_json(output / "completion_progress.json", progress)
            return progress
        if local["state"] == "finished_with_errors":
            raise RuntimeError("Local generation has failed jobs; see queue_status.json")
        active = any(j["state"] in ("running", "queued", "launched_awaiting_cuda")
                     for r in inventory["runs"] if r["state"] in ("running", "queued")
                     for j in r["jobs"])
        for status_path in (output / "downloads/local").glob("*/queue_status.json"):
            local_status = json.loads(status_path.read_text())
            active = active or any(j["id"] in expected and j["state"] in ("queued", "running")
                                   for j in local_status["jobs"])
        if not expected <= complete and not active:
            raise RuntimeError(f"Remote queue stopped before completion: {sorted(expected - complete)}")
        time.sleep(60)
    raise TimeoutError("Collection deadline reached; outputs are retained for resumption")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=deploy.ROOT / "out/parc_complete_20261008")
    parser.add_argument("--timeout-minutes", type=int, default=90)
    args = parser.parse_args()
    print(json.dumps(finish(args.output, args.timeout_minutes), indent=2), flush=True)
