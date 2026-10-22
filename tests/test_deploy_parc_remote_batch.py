"""Queue safety and deployment boundaries; no SSH or CUDA is used by tests."""

import json
from pathlib import Path, PureWindowsPath
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import deploy_parc_remote_batch as deploy


class DeploymentTests(unittest.TestCase):
    def test_manifest_has_112_candidates_and_unique_seeds(self):
        jobs = deploy.make_jobs("g2") + deploy.make_jobs("g3")
        self.assertEqual(sum(j["batch_size"] for j in jobs), 112)
        self.assertEqual(len({j["seed"] for j in jobs}), 14)
        for host in ("g2", "g3"):
            rows = deploy.make_jobs(host)
            self.assertEqual([j["variant"] for j in rows[:6]], [0, 0, 1, 1, 2, 2])
            self.assertEqual(rows[-1]["kind"], "stairs")
            command = deploy.job_command(Path("/tmp/run"), rows[0])
            self.assertEqual(command[command.index("--max-seconds") + 1], "45")
            self.assertEqual(command[command.index("--score-frame-chunk") + 1], "4")
            self.assertIn("--checkpoint", command)

    def test_windows_controller_serializes_remote_paths_as_posix(self):
        path = PureWindowsPath("${PRIVATE_STORAGE}/parc-remote-batch-20261008/bootstrap/a.py")
        command = deploy.ssh("g3", ["python3", path])[-1]
        self.assertIn("${PRIVATE_STORAGE}/parc-remote-batch-20261008/bootstrap/a.py", command)
        self.assertNotIn("\\", command)
        for argument in deploy.job_command(Path("${PRIVATE_STORAGE}/run"), deploy.make_jobs("g3")[0]):
            self.assertNotIn("\\", argument)

    def test_busy_process_blocks_even_at_zero_utilization(self):
        rows = deploy.parse_gpu_samples("0, GPU-a, RTX 4090, 23000, 0\n", "GPU-a, 123\n")
        self.assertFalse(deploy.eligible_gpu(rows["GPU-a"], 20480))

    def test_two_samples_same_uuid_and_memory_required(self):
        free = deploy.parse_gpu_samples("0, GPU-a, RTX 4090, 23000, 0\n", "")
        busy = deploy.parse_gpu_samples("0, GPU-a, RTX 4090, 23000, 0\n", "GPU-a, 123\n")
        low = deploy.parse_gpu_samples("0, GPU-a, RTX 4090, 20000, 0\n", "")
        self.assertEqual(deploy.twice_idle({}, free, 20480), [])
        self.assertEqual(deploy.twice_idle(busy, free, 20480), [])
        self.assertEqual(deploy.twice_idle(free, low, 20480), [])
        self.assertEqual(deploy.twice_idle(free, free, 20480), ["GPU-a"])

    def test_unknown_telemetry_fails_closed(self):
        for gpu, processes in (("", ""), ("0,GPU-a,4090,N/A,0", ""),
                               ("0,GPU-a,4090,23000,0", "GPU-other,123"),
                               ("0,GPU-a,4090,23000,0", "Permission denied")):
            with self.assertRaises(ValueError):
                deploy.parse_gpu_samples(gpu, processes)

    def test_source_allowlist_rejects_outputs_credentials_and_traversal(self):
        self.assertTrue(deploy.allowed_file(deploy.CHECKPOINT))
        self.assertTrue(deploy.allowed_file(deploy.VENDOR + "/PARC/util/torch_util.py"))
        for name in ("out/other.pkl", "../secret.py", "/etc/passwd", ".ssh/id_rsa",
                     deploy.VENDOR + "/.venv/foo.py", deploy.VENDOR + "/PARC/.hidden/foo.py"):
            self.assertFalse(deploy.allowed_file(name), name)

    def test_ssh_retains_saved_host_verification(self):
        for host in ("g2", "g3"):
            command = deploy.ssh(host, ["hostname"])
            self.assertIn("StrictHostKeyChecking=yes", command)
            self.assertIn("UpdateHostKeys=no", command)
            self.assertIn("HostKeyAlias=" + deploy.HOSTS[host]["alias"], command)
        with self.assertRaises(KeyError):
            deploy.ssh("g1", ["hostname"])

    def test_transfer_budget_counts_failed_or_reserved_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "ledger.json"
            deploy.reserve_transfer(ledger, 1_500_000_000, "g2", "a")
            with self.assertRaises(ValueError):
                deploy.reserve_transfer(ledger, 500_000_000, "g3", "b")
            self.assertEqual(len(json.loads(ledger.read_text())["transfers"]), 1)

    def test_ema_is_required_without_importing_generator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / deploy.SCRIPTS[1]
            path.parent.mkdir()
            path.write_text("pilot.load_official_model(a,b,c,weight_key='ema_model')")
            deploy.validate_generator(Path(directory))
            path.write_text("pilot.load_official_model(a,b,c)")
            with self.assertRaises(ValueError):
                deploy.validate_generator(Path(directory))

    def test_queue_expires_without_cuda_on_busy_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            manifest = {"max_wait_seconds": 60, "poll_seconds": 30,
                        "min_free_mib": 20480, "job_timeout_seconds": 7200,
                        "source_sha256": "abc", "jobs": deploy.make_jobs("g2")}
            clock = [0.0]
            def sleep(seconds):
                clock[0] += seconds
            busy = deploy.parse_gpu_samples("0,GPU-a,4090,23000,0", "GPU-a,123")
            with patch.object(deploy.time, "monotonic", side_effect=lambda: clock[0]), \
                 patch.object(deploy.time, "sleep", side_effect=sleep), \
                 patch.object(deploy, "gpu_sample", return_value=busy), \
                 patch.object(deploy.subprocess, "Popen") as launch, \
                 patch.object(deploy.subprocess, "run") as smoke:
                deploy.queue_loop(run, manifest)
            launch.assert_not_called()
            smoke.assert_not_called()
            status = json.loads((run / "queue_status.json").read_text())
            self.assertEqual(status["state"], "deadline_expired")
            self.assertFalse(status["generation_started"])
            self.assertTrue(all(j["state"] == "deadline_expired" for j in status["jobs"]))

    def test_queue_refuses_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            job = deploy.make_jobs("g2")[0]
            (run / "results" / job["id"]).mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                deploy.queue_loop(run, {"max_wait_seconds": 60, "jobs": [job]})


if __name__ == "__main__":
    unittest.main()
