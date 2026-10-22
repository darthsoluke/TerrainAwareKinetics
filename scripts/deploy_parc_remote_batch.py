"""Prepare immutable PARC bundles and a finite, idle-only g2/g3 inference queue.

The generator remains generate_parc_long_stairs.py. Preparation/import checks do
not initialize CUDA. Only remote-queue may run a tiny CUDA test and generation,
after two idle samples and acquisition of a private-user-owned per-GPU advisory lock.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import hashlib
import importlib
import io
import json
import os
from pathlib import Path, PurePath, PurePosixPath
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
REMOTE_BASE = Path("${PRIVATE_STORAGE}/parc-remote-batch-20261008")
CHECKPOINT = "out/parc_legacy_models_20261008/demo_iter3_verified.ckpt"
VENDOR = "out/vendor/PARC"
SCRIPTS = (
    "scripts/deploy_parc_remote_batch.py",
    "scripts/generate_parc_long_stairs.py",
    "scripts/run_parc_local_terrain_pilot.py",
    "scripts/build_parc_stair_course.py",
    "scripts/build_parc_platform_extensions.py",
)
ASSETS = (f"{VENDOR}${PRIVATE_STORAGE}/humanoid.xml",
          f"{VENDOR}${PRIVATE_STORAGE}/parc_2_kin_gen_default.yaml")
HOSTS = {
    "g2": {"port": 12302, "alias": "[remote-host.example]:12302",
           "known_hosts": "${SSH_CONFIG_DIR}/known_hosts",
           "parent": "${PRIVATE_STORAGE}/venvs/kimodo", "python_minor": "3.10"},
    "g3": {"port": 12303, "alias": "[remote-host.example]:12303",
           "known_hosts": "${USER_HOME}/.ssh/known_hosts",
           "parent": "${PRIVATE_STORAGE}/motionbricks-preparation/envs/mb-bones",
           "python_minor": "3.11"},
}
TRANSFER_LIMIT = 2_000_000_000
FILE_LIMIT = 10_000_000_000
DEPS = ("numpy", "torch", "yaml", "matplotlib", "tqdm", "scipy", "wandb", "dill", "trimesh")
INSTALL_NAMES = {name: name for name in DEPS if name != "torch"}
INSTALL_NAMES.update({"yaml": "PyYAML", "git": "GitPython", "google.protobuf": "protobuf", "google": "protobuf",
                      "sentry_sdk": "sentry-sdk", "platformdirs": "platformdirs",
                      "pydantic": "pydantic", "requests": "requests", "psutil": "psutil",
                      "packaging": "packaging", "typing_extensions": "typing_extensions",
                      "setproctitle": "setproctitle", "dockerpycreds": "docker-pycreds"})


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def checked_run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, capture_output=True,
                          timeout=kwargs.pop("timeout", 120), **kwargs).stdout


def ssh(host, command):
    cfg = HOSTS[host]
    return ["ssh", "-i", "${SSH_KEY_FILE}", "-p", str(cfg["port"]),
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "StrictHostKeyChecking=yes", "-o", "UpdateHostKeys=no",
            "-o", "HostKeyAlias=" + cfg["alias"],
            "-o", "UserKnownHostsFile=" + cfg["known_hosts"],
            "private-user@remote-host.example", shlex.join([
                item.as_posix() if isinstance(item, PurePath) else str(item) for item in command])]


def owned_path(path):
    """Reject escaped/symlinked destinations and writes outside this workspace."""
    import pwd
    if pwd.getpwuid(os.getuid()).pw_name != "private-user":
        raise RuntimeError("Remote actions must run as private-user")
    path = Path(path)
    path.relative_to(REMOTE_BASE)
    for parent in [path, *path.parents]:
        if parent == Path("${PRIVATE_STORAGE}"):
            break
        if parent.is_symlink():
            raise ValueError(f"Symlink destination rejected: {parent}")
        if parent.exists() and parent.stat().st_uid != os.getuid():
            raise ValueError(f"Destination not owned by private-user: {parent}")
    return path


def runtime_env(cuda=""):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(CUDA_VISIBLE_DEVICES=cuda, PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="disabled", WANDB_SILENT="true",
               MPLBACKEND="Agg", MPLCONFIGDIR=str(REMOTE_BASE / "cache/matplotlib"),
               XDG_CACHE_HOME=str(REMOTE_BASE / "cache"),
               TORCH_HOME=str(REMOTE_BASE / "cache/torch"),
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    return env


def make_jobs(host):
    first_seed = 20262008 + (0 if host == "g2" else 7)
    jobs = []
    for variant in range(3):
        for repeat in range(2):
            seed = first_seed + variant * 2 + repeat
            jobs.append({"id": f"platform_v{variant}_s{seed}", "kind": "platform",
                         "variant": variant, "seed": seed, "batch_size": 8})
    jobs.append({"id": f"stairs_s{first_seed + 6}", "kind": "stairs", "variant": 0,
                 "seed": first_seed + 6, "batch_size": 8})
    return jobs


def job_command(run_dir, job):
    work = Path(run_dir) / "workspace"
    return [(REMOTE_BASE / "env/bin/python").as_posix(), "-B", "-u",
            (work / "scripts/generate_parc_long_stairs.py").as_posix(),
            "--repo", (work / VENDOR).as_posix(), "--checkpoint", (work / CHECKPOINT).as_posix(),
            "--output", (Path(run_dir) / "results" / job["id"]).as_posix(),
            "--device", "cuda:0", "--kind", job["kind"], "--variant", str(job["variant"]),
            "--seed", str(job["seed"]), "--batch-size", "8", "--max-seconds", "45",
            "--threads", "4", "--score-frame-chunk", "4"]


def allowed_file(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        return False
    return (name in SCRIPTS or name in ASSETS or name == CHECKPOINT
            or (name.startswith(VENDOR + "/PARC/") and name.endswith(".py")
                and not any(part.startswith(".") or part == "__pycache__" for part in path.parts)))


def source_files(root):
    files = [Path(root) / name for name in (*SCRIPTS, *ASSETS, CHECKPOINT)]
    files.extend(sorted((Path(root) / VENDOR / "PARC").rglob("*.py")))
    for path in files:
        name = path.relative_to(root).as_posix()
        if not allowed_file(name) or path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or disallowed source: {path}")
        if path.resolve() != Path(root).resolve().joinpath(name):
            raise ValueError(f"Symlinked source parent: {path}")
        if path.stat().st_size >= FILE_LIMIT:
            raise ValueError(f"File exceeds direct-transfer limit: {path}")
    return files


def validate_generator(root):
    tree = ast.parse((Path(root) / SCRIPTS[1]).read_text(encoding="utf-8"))
    selections = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "load_official_model"]
    if not selections or not all(any(kw.arg == "weight_key" and isinstance(kw.value, ast.Constant)
                                    and kw.value.value == "ema_model" for kw in node.keywords)
                                 for node in selections):
        raise ValueError("Generator must explicitly select ema_model")


def build_bundle(root, local_dir):
    validate_generator(root)
    archive_path = Path(local_dir) / "source.tar"
    entries = []
    with tarfile.open(archive_path, "x") as archive:
        for path in source_files(root):
            name = path.relative_to(root).as_posix()
            before = digest(path)
            archive.add(path, arcname=name, recursive=False)
            if before != digest(path):
                raise RuntimeError(f"Source changed during snapshot: {name}")
            entries.append({"path": name, "sha256": before, "bytes": path.stat().st_size})
        snapshot = {"created_at": stamp(), "files": entries,
                    "source_sha256": hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()}
        data = json.dumps(snapshot, indent=2).encode()
        member = tarfile.TarInfo("_bundle.json")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    if archive_path.stat().st_size * 2 >= TRANSFER_LIMIT:
        raise ValueError("Both server bundles would exceed the aggregate 2 GB transfer budget")
    write_json(Path(local_dir) / "snapshot.json", snapshot, exclusive=True)
    return archive_path, snapshot


def reserve_transfer(ledger_path, size, host, destination):
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_suffix(".lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"transfers": []}
        spent = sum(row["bytes"] for row in ledger["transfers"])
        if size >= FILE_LIMIT or spent + size >= TRANSFER_LIMIT:
            raise ValueError("Aggregate direct-transfer budget would exceed 2 GB")
        ledger["transfers"].append({"at": stamp(), "host": host,
                                     "destination": Path(destination).as_posix(), "bytes": size})
        write_json(ledger_path, ledger)
    finally:
        os.close(descriptor)
        lock_path.unlink()


RECEIVER = '''import hashlib, os, pathlib, sys, uuid
p = pathlib.Path(sys.argv[1]); expected = sys.argv[2]; size = int(sys.argv[3])
base = pathlib.Path('${PRIVATE_STORAGE}/parc-remote-batch-20261008')
p.relative_to(base)
for q in [p, *p.parents]:
    if q == pathlib.Path('${PRIVATE_STORAGE}'): break
    if q.is_symlink() or (q.exists() and q.stat().st_uid != os.getuid()): raise RuntimeError(str(q))
p.parent.mkdir(parents=True, exist_ok=True)
tmp = p.with_name(p.name + '.' + uuid.uuid4().hex + '.upload')
h = hashlib.sha256(); total = 0
try:
    with tmp.open('xb') as f:
        while True:
            block = sys.stdin.buffer.read(4 * 1024 * 1024)
            if not block: break
            total += len(block)
            if total > size: raise RuntimeError('Oversize upload')
            h.update(block); f.write(block)
    if total != size or h.hexdigest() != expected: raise RuntimeError('Upload hash mismatch')
    os.link(tmp, p)
finally:
    tmp.unlink(missing_ok=True)
print(str(p))
'''


def upload(host, path, destination, ledger):
    checksum = digest(path)
    probe = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); print('exists' if p.exists() else 'missing')"
    exists = checked_run(ssh(host, ["python3", "-B", "-c", probe, destination])).strip()
    if exists == "exists":
        actual = checked_run(ssh(host, ["sha256sum", destination])).split()[0]
        if actual != checksum:
            raise RuntimeError(f"Refusing to overwrite remote file: {destination}")
        return
    reserve_transfer(ledger, path.stat().st_size, host, destination)
    with path.open("rb") as stream:
        subprocess.run(ssh(host, ["python3", "-B", "-c", RECEIVER, destination,
                                  checksum, str(path.stat().st_size)]), stdin=stream,
                       check=True, timeout=1800)


def dependency_report():
    report = {"python": sys.version, "modules": {}, "errors": {}, "missing": []}
    for name in DEPS:
        try:
            module = importlib.import_module(name)
            report["modules"][name] = str(getattr(module, "__version__", "installed"))
            if name == "torch":
                report["torch_cuda_build"] = module.version.cuda
        except Exception as error:
            report["errors"][name] = f"{type(error).__name__}: {error}"
            if isinstance(error, ModuleNotFoundError):
                report["missing"].append(error.name)
    report["ready"] = not report["errors"] and bool(report.get("torch_cuda_build"))
    return report


def remote_prepare(host):
    if checked_run(["hostname"]).strip() != "taku-" + host:
        raise RuntimeError("Remote hostname does not match selected server")
    owned_path(REMOTE_BASE).mkdir(parents=True, exist_ok=True)
    env_dir = owned_path(REMOTE_BASE / "env")
    cfg = HOSTS[host]
    parent = Path(cfg["parent"])
    log_path = REMOTE_BASE / "environment_setup.log"
    with log_path.open("a", encoding="utf-8") as log:
        if not env_dir.exists():
            subprocess.run([str(parent / "bin/python"), "-B", "-m", "venv", "--without-pip", str(env_dir)],
                           check=True, env=runtime_env(), stdout=log, stderr=log, timeout=120)
            site = env_dir / "lib" / ("python" + cfg["python_minor"]) / "site-packages"
            with (site / "parc_readonly_parent.pth").open("x") as stream:
                stream.write(str(parent / "lib" / ("python" + cfg["python_minor"]) / "site-packages") + "\n")
        python = str(env_dir / "bin/python")
        for attempt in range(4):
            report = json.loads(checked_run([python, "-B", str(Path(__file__).resolve()), "dependency-probe"],
                                            env=runtime_env()))
            write_json(REMOTE_BASE / "environment.json", report)
            if report["ready"]:
                return report
            missing = set(report["missing"])
            packages = {INSTALL_NAMES[name] for name, error in report["errors"].items()
                        if name in INSTALL_NAMES and error.startswith("ModuleNotFoundError:")}
            packages.update(INSTALL_NAMES[name] for name in missing if name in INSTALL_NAMES)
            if attempt == 3 or not packages:
                return report
            pip_check = subprocess.run([python, "-B", "-m", "pip", "--version"], env=runtime_env(),
                                       stdout=log, stderr=log, timeout=30)
            if pip_check.returncode:
                subprocess.run([python, "-B", "-m", "ensurepip"], env=runtime_env(), check=True,
                               stdout=log, stderr=log, timeout=120)
            subprocess.run([python, "-B", "-m", "pip", "install", "--disable-pip-version-check",
                            "--no-cache-dir",
                            *sorted(packages)],
                           env=runtime_env(), check=True, stdout=log, stderr=log, timeout=300)
    return report


def verify_workspace(work):
    snapshot = json.loads((work / "_bundle.json").read_text())
    for entry in snapshot["files"]:
        path = work / entry["path"]
        if not allowed_file(entry["path"]) or path.is_symlink() or digest(path) != entry["sha256"]:
            raise RuntimeError(f"Snapshot verification failed: {entry['path']}")
    return snapshot


def import_probe(work):
    sys.path.insert(0, str(work / "scripts"))
    sys.path.insert(0, str(work / VENDOR))
    sys.modules.setdefault("parc", importlib.import_module("PARC"))
    generator = importlib.import_module("generate_parc_long_stairs")
    mdm_path = importlib.import_module("parc.motion_synthesis.procgen.mdm_path")
    gen_util = importlib.import_module("parc.motion_generator.gen_util")
    importlib.import_module("parc.util.file_io_helper")
    settings, gen, _ = generator.official_generation_settings(
        work / VENDOR, mdm_path.MDMPathSettings, gen_util.MDMGenSettings, 8, 45.)
    if gen.ddim_stride != 5 or settings.mdm_batch_size != 8:
        raise RuntimeError("Released generator settings were not preserved")
    validate_generator(work)
    report = dependency_report()
    report.update(source_imports=True, ddim_stride=gen.ddim_stride, weight_selection="ema_model")
    return report


def checkpoint_probe(work):
    """Strict CPU-only compatibility check in a short-lived subprocess."""
    import torch
    import_probe(work)
    pilot = importlib.import_module("run_parc_local_terrain_pilot")
    torch.set_num_threads(4)
    with contextlib.redirect_stdout(sys.stderr):
        model = pilot.load_official_model(work / VENDOR, work / CHECKPOINT,
                                         "cpu", weight_key="ema_model")
    statistics = {}
    for name in ("_mdm_features_mean", "_mdm_features_std"):
        tensor = getattr(model, name)
        if tuple(tensor.shape) != (15, 91) or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"Invalid checkpoint normalization tensor: {name}")
        statistics[name] = {"shape": list(tensor.shape), "all_finite": True}
    if not all(bool(torch.isfinite(value).all()) for value in model._denoise_model.state_dict().values()):
        raise RuntimeError("Non-finite EMA model parameters or buffers")
    return {"ready": True, "at": stamp(), "device": "cpu", "weight_selection": "ema_model",
            "strict_load": True, "all_weights_finite": True, "statistics": statistics,
            "parameters": sum(value.numel() for value in model._denoise_model.parameters()),
            "checkpoint_sha256": digest(work / CHECKPOINT),
            "source_sha256": json.loads((work / "_bundle.json").read_text())["source_sha256"],
            "torch": torch.__version__}


def remote_install(archive_path, batch_id, host):
    run_dir = owned_path(REMOTE_BASE / "runs" / batch_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    work = run_dir / "workspace"
    work.mkdir()
    with tarfile.open(archive_path, "r") as archive:
        snapshot = json.load(archive.extractfile("_bundle.json"))
        expected = {entry["path"]: entry for entry in snapshot["files"]}
        members = archive.getmembers()
        if len({m.name for m in members}) != len(members):
            raise ValueError("Duplicate archive members")
        if set(m.name for m in members) != set(expected) | {"_bundle.json"}:
            raise ValueError("Unexpected archive members")
        for member in members:
            if not member.isfile() or (member.name != "_bundle.json" and not allowed_file(member.name)):
                raise ValueError(f"Rejected archive member: {member.name}")
            target = work / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as destination:
                while block := source.read(4 * 1024 * 1024):
                    destination.write(block)
    verify_workspace(work)
    manifest = {"host": host, "batch_id": batch_id, "created_at": stamp(),
                "source_sha256": snapshot["source_sha256"], "files": snapshot["files"],
                "max_wait_seconds": 86400, "poll_seconds": 30, "min_free_mib": 20480,
                "job_timeout_seconds": 7200, "max_candidates": 56, "physics": "skipped",
                "jobs": make_jobs(host)}
    for job in manifest["jobs"]:
        job["command"] = job_command(run_dir, job)
    write_json(run_dir / "manifest.json", manifest, exclusive=True)
    result = subprocess.run([str(REMOTE_BASE / "env/bin/python"), "-B", str(Path(__file__).resolve()),
                             "source-probe", "--work", str(work)], env=runtime_env(), text=True,
                            capture_output=True, timeout=180)
    (run_dir / "import_probe.log").write_text(result.stdout + result.stderr)
    report = {"state": "prepared" if result.returncode == 0 else "blocked_dependencies",
              "at": stamp(), "run_dir": str(run_dir), "source_sha256": snapshot["source_sha256"],
              "returncode": result.returncode, "generation_started": False}
    write_json(run_dir / "queue_status.json", report)
    return report


def parse_gpu_samples(gpu_csv, processes_csv):
    rows = {}
    for row in csv.reader(io.StringIO(gpu_csv)):
        if len(row) != 5:
            raise ValueError("Unexpected GPU telemetry row")
        index, gpu_uuid, name, free, utilization = [value.strip() for value in row]
        rows[gpu_uuid] = {"index": int(index), "uuid": gpu_uuid, "model": name,
                          "free_mib": int(free), "utilization": int(utilization), "pids": []}
    if not rows:
        raise ValueError("No GPU telemetry; cannot establish idleness")
    for row in csv.reader(io.StringIO(processes_csv)):
        if not row or row == ["No running processes found"]:
            continue
        if len(row) != 2 or row[0].strip() not in rows:
            raise ValueError("Unexpected compute process telemetry")
        rows[row[0].strip()]["pids"].append(int(row[1].strip()))
    return rows


def gpu_sample():
    return parse_gpu_samples(
        checked_run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.free,utilization.gpu",
                     "--format=csv,noheader,nounits"], timeout=15),
        checked_run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                     "--format=csv,noheader,nounits"], timeout=15))


def eligible_gpu(row, minimum):
    return not row["pids"] and row["free_mib"] >= minimum and row["utilization"] <= 5


def twice_idle(previous, current, minimum):
    return [key for key, row in current.items() if key in previous
            and eligible_gpu(row, minimum) and eligible_gpu(previous[key], minimum)]


@contextlib.contextmanager
def file_lock(path, nonblocking=True):
    import fcntl
    owned_path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        except BlockingIOError:
            yield None
            return
        try:
            yield stream
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def stop_own_child(process):
    """Only terminate the new process group created for this queue's child."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def queue_loop(run_dir, manifest):
    started = time.monotonic()
    deadline = started + manifest["max_wait_seconds"]
    deadline_epoch = time.time() + manifest["max_wait_seconds"]
    jobs = [dict(job, state="queued") for job in manifest["jobs"]]
    active, previous = {}, {}
    for job in jobs:
        if (run_dir / "results" / job["id"]).exists() or (run_dir / "logs" / (job["id"] + ".log")).exists():
            raise RuntimeError("Existing job output/log; refusing to overwrite or rerun")
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)
    last_error = None
    def persist(state, sample=None):
        write_json(run_dir / "queue_status.json", {
            "state": state, "at": stamp(), "deadline_epoch": deadline_epoch,
            "generation_started": any(job.get("compute_observed_at") for job in jobs),
            "source_sha256": manifest["source_sha256"], "jobs": jobs,
            "gpu_sample": sample, "last_error": last_error})
    persist("queued")
    try:
        while time.monotonic() < deadline:
            try:
                current = gpu_sample()
                last_error = None
            except Exception as error:
                current, previous = {}, {}
                last_error = f"GPU query failed closed: {error}"
            for gpu_uuid, item in list(active.items()):
                process, job = item["process"], item["job"]
                row = current.get(gpu_uuid)
                if row and process.pid in row["pids"] and not job.get("compute_observed_at"):
                    job["compute_observed_at"] = stamp()
                    job["state"] = "running"
                foreign = set(row["pids"]) - {process.pid} if row else set()
                if foreign and process.poll() is None:
                    job["state"] = "yielded_to_foreign_process"
                    job["foreign_pids"] = sorted(foreign)
                    stop_own_child(process)
                elif time.monotonic() - item["start"] >= manifest["job_timeout_seconds"]:
                    job["state"] = "job_timeout"
                    stop_own_child(process)
                if process.poll() is not None:
                    job["returncode"] = process.returncode
                    job["finished_at"] = stamp()
                    if job["state"] in ("running", "launched_awaiting_cuda"):
                        job["state"] = "completed" if process.returncode == 0 else "failed"
                    item["log"].close()
                    item["lock"].__exit__(None, None, None)
                    del active[gpu_uuid]
            for gpu_uuid in twice_idle(previous, current, manifest["min_free_mib"]):
                pending = next((job for job in jobs if job["state"] == "queued"), None)
                if pending is None or gpu_uuid in active or time.monotonic() >= deadline:
                    continue
                lock = file_lock(REMOTE_BASE / "gpu-locks" / (gpu_uuid + ".lock"))
                stream = lock.__enter__()
                if stream is None:
                    lock.__exit__(None, None, None)
                    continue
                log = None
                handed_off = False
                try:
                    fresh = gpu_sample().get(gpu_uuid)
                    if fresh is None or not eligible_gpu(fresh, manifest["min_free_mib"]):
                        continue
                    smoke = "import torch; x=torch.ones((32,32),device='cuda:0'); y=x@x; torch.cuda.synchronize(); assert y[0,0].item()==32"
                    smoke_result = subprocess.run([str(REMOTE_BASE / "env/bin/python"), "-B", "-c", smoke],
                                                  env=runtime_env(gpu_uuid), text=True, capture_output=True,
                                                  timeout=60, pass_fds=(stream.fileno(),))
                    with (run_dir / "cuda_probe.log").open("a") as probe_log:
                        probe_log.write(json.dumps({"at": stamp(), "gpu": gpu_uuid,
                            "returncode": smoke_result.returncode, "stderr": smoke_result.stderr}) + "\n")
                    if smoke_result.returncode:
                        raise RuntimeError("Idle GPU CUDA probe failed; see cuda_probe.log")
                    fresh = gpu_sample().get(gpu_uuid)
                    if fresh is None or not eligible_gpu(fresh, manifest["min_free_mib"]):
                        continue
                    log = (run_dir / "logs" / (pending["id"] + ".log")).open("x")
                    process = subprocess.Popen(pending["command"], cwd=run_dir / "workspace",
                        env=runtime_env(gpu_uuid), stdout=log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, start_new_session=True, pass_fds=(stream.fileno(),))
                    pending.update(state="launched_awaiting_cuda", pid=process.pid, gpu_uuid=gpu_uuid,
                                   gpu_index=fresh["index"], launched_at=stamp())
                    active[gpu_uuid] = {"process": process, "job": pending, "lock": lock,
                                        "log": log, "start": time.monotonic()}
                    handed_off = True
                finally:
                    if not handed_off:
                        if log:
                            log.close()
                        lock.__exit__(None, None, None)
            if not active and not any(job["state"] == "queued" for job in jobs):
                persist("completed" if all(job["state"] == "completed" for job in jobs) else "finished_with_errors", current)
                return
            previous = current
            persist("running" if any(job["state"] == "running" for job in jobs) else "queued", current)
            time.sleep(min(manifest["poll_seconds"], max(0, deadline - time.monotonic())))
        for job in jobs:
            if job["state"] in ("queued", "running", "launched_awaiting_cuda"):
                job["state"] = "deadline_expired"
        persist("deadline_expired")
    except BaseException as error:
        last_error = f"{type(error).__name__}: {error}"
        persist("failed")
        raise
    finally:
        for item in active.values():
            stop_own_child(item["process"])
            item["log"].close()
            item["lock"].__exit__(None, None, None)


def remote_queue(run_dir):
    run_dir = owned_path(run_dir)
    with file_lock(run_dir / "queue.lock") as lock:
        if lock is None:
            raise RuntimeError("This manifest already has a queue process")
        verify_workspace(run_dir / "workspace")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        queue_loop(run_dir, manifest)


def remote_launch(run_dir):
    run_dir = owned_path(run_dir)
    with file_lock(run_dir / "launch.lock") as lock:
        if lock is None:
            raise RuntimeError("Launch already in progress")
        if (run_dir / "launch.json").exists():
            return json.loads((run_dir / "launch.json").read_text())
        status = json.loads((run_dir / "queue_status.json").read_text())
        if status["state"] not in ("prepared", "blocked_checkpoint"):
            raise RuntimeError(f"Not ready to queue: {status}")
        work = run_dir / "workspace"
        snapshot = verify_workspace(work)
        certificate = run_dir / "checkpoint_probe.json"
        if not certificate.exists():
            probe = subprocess.run([str(REMOTE_BASE / "env/bin/python"), "-B",
                                    str(Path(__file__).resolve()), "checkpoint-probe", "--work", str(work)],
                                   env=runtime_env(), capture_output=True, text=True, timeout=180)
            with (run_dir / "checkpoint_probe.log").open("a") as log:
                log.write(probe.stdout + probe.stderr)
            if probe.returncode:
                status.update(state="blocked_checkpoint", error=probe.stderr[-8000:], at=stamp())
                write_json(run_dir / "queue_status.json", status)
                raise RuntimeError("CPU checkpoint load failed; see checkpoint_probe.log")
            write_json(certificate, json.loads(probe.stdout), exclusive=True)
        cpu_check = json.loads(certificate.read_text())
        if not cpu_check.get("ready") or cpu_check["source_sha256"] != snapshot["source_sha256"]:
            raise RuntimeError("CPU checkpoint certificate does not match the source snapshot")
        # Bootstrap files are immutable and named by their content hash. Record
        # the final runner separately when a preflight was added after bundling.
        script = Path(__file__).resolve()
        with (run_dir / "queue.log").open("x") as log:
            process = subprocess.Popen(["/usr/bin/python3", "-B", str(script), "remote-queue",
                                        "--run-dir", str(run_dir)], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                       env=runtime_env())
        result = {"state": "queued", "queue_pid": process.pid, "at": stamp(),
                  "generation_started": False, "run_dir": str(run_dir),
                  "queue_script": str(script), "queue_script_sha256": digest(script),
                  "cpu_checkpoint_probe": str(certificate)}
        write_json(run_dir / "launch.json", result, exclusive=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "prepare", "deploy", "launch", "status",
        "dependency-probe", "source-probe", "checkpoint-probe", "remote-prepare", "remote-install", "remote-launch", "remote-queue"))
    parser.add_argument("--hosts", nargs="+", choices=tuple(HOSTS), default=["g3", "g2"])
    parser.add_argument("--host", choices=tuple(HOSTS))
    parser.add_argument("--batch-id", default="ema_stride5_20261008_v1")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--source-ready", action="store_true", help="Final course/generator snapshot is coordinated")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.batch_id):
        parser.error("Invalid batch ID")
    if args.action == "dependency-probe":
        print(json.dumps(dependency_report()))
        return
    if args.action == "source-probe":
        report = import_probe(args.work)
        print(json.dumps(report))
        if not report["ready"]:
            raise SystemExit(1)
        return
    if args.action == "checkpoint-probe":
        print(json.dumps(checkpoint_probe(args.work)))
        return
    if args.action.startswith("remote-"):
        if args.action == "remote-prepare":
            result = remote_prepare(args.host)
        elif args.action == "remote-install":
            result = remote_install(args.archive, args.batch_id, args.host)
        elif args.action == "remote-launch":
            result = remote_launch(args.run_dir)
        else:
            remote_queue(args.run_dir)
            return
        print(json.dumps(result, indent=2))
        return
    run_dir = REMOTE_BASE / "runs" / args.batch_id
    if args.action == "plan":
        print(json.dumps({"run_dir": run_dir.as_posix(), "checkpoint": CHECKPOINT,
            "checkpoint_ready": (args.root / CHECKPOINT).is_file(), "maximum_candidates": 112,
            "jobs": {host: [dict(job, command=job_command(run_dir, job)) for job in make_jobs(host)]
                     for host in args.hosts}}, indent=2))
        return
    local_base = args.root / "out/parc_remote_deployment_20261008"
    ledger = local_base / "transfer_ledger.json"
    if args.action == "status":
        for host in args.hosts:
            print(host, checked_run(ssh(host, ["cat", run_dir / "queue_status.json"])))
        return
    if args.action == "deploy" and not args.source_ready:
        parser.error("Coordinate the final course snapshot first, then pass --source-ready")
    # Freeze the controller too: an agent may edit the working file while a
    # large bundle is being uploaded to the first host.
    controller_bytes = Path(__file__).read_bytes()
    controller_hash = hashlib.sha256(controller_bytes).hexdigest()
    frozen_controller = local_base / "bootstrap" / (controller_hash + ".py")
    frozen_controller.parent.mkdir(parents=True, exist_ok=True)
    if not frozen_controller.exists():
        with frozen_controller.open("xb") as stream:
            stream.write(controller_bytes)
    elif digest(frozen_controller) != controller_hash:
        raise RuntimeError("Frozen controller hash mismatch")
    bootstrap = REMOTE_BASE / "bootstrap" / (controller_hash + ".py")
    deployment_path = local_base / (args.batch_id + ".deployment.json")
    deployment = (json.loads(deployment_path.read_text()) if deployment_path.exists()
                  else {"batch_id": args.batch_id, "hosts": {}})
    deployment.update(action=args.action, updated_at=stamp(), run_dir=run_dir.as_posix())
    write_json(deployment_path, deployment)
    if args.action == "deploy":
        local_dir = local_base / args.batch_id
        try:
            if local_dir.exists():
                archive = local_dir / "source.tar"
                snapshot = json.loads((local_dir / "snapshot.json").read_text())
                with tarfile.open(archive) as frozen:
                    if json.load(frozen.extractfile("_bundle.json")) != snapshot:
                        raise RuntimeError("Frozen bundle manifest mismatch")
            else:
                local_dir.mkdir(parents=True, exist_ok=False)
                archive, snapshot = build_bundle(args.root, local_dir)
        except Exception as error:
            deployment.update(state="blocked_bundle", error=f"{type(error).__name__}: {error}")
            write_json(deployment_path, deployment)
            raise
        remote_archive = REMOTE_BASE / "bundles" / (snapshot["source_sha256"] + ".tar")
        deployment["source_sha256"] = snapshot["source_sha256"]
    failed = False
    for host in args.hosts:
        deployment["hosts"][host] = {"state": args.action + "_in_progress", "at": stamp()}
        write_json(deployment_path, deployment)
        print(json.dumps({"host": host, "action": args.action, "state": "in_progress"}), flush=True)
        try:
            upload(host, frozen_controller, bootstrap, ledger)
            if args.action == "prepare":
                output = checked_run(ssh(host, ["python3", "-B", bootstrap, "remote-prepare", "--host", host]), timeout=1800)
            elif args.action == "deploy":
                upload(host, archive, remote_archive, ledger)
                output = checked_run(ssh(host, ["python3", "-B", bootstrap, "remote-install",
                    "--host", host, "--archive", remote_archive, "--batch-id", args.batch_id]), timeout=600)
            else:
                output = checked_run(ssh(host, ["python3", "-B", bootstrap, "remote-launch", "--run-dir", run_dir]))
            result = json.loads(output)
            deployment["hosts"][host] = {"state": args.action + "_finished", "at": stamp(), "result": result}
            if result.get("ready") is False or str(result.get("state", "")).startswith("blocked"):
                failed = True
                deployment["hosts"][host]["state"] = "blocked"
            print(host, json.dumps(result), flush=True)
        except Exception as error:
            failed = True
            deployment["hosts"][host] = {"state": "blocked", "at": stamp(),
                "error": f"{type(error).__name__}: {error}",
                "stdout": str(getattr(error, "stdout", ""))[-8000:],
                "stderr": str(getattr(error, "stderr", ""))[-8000:]}
            print(host, json.dumps(deployment["hosts"][host]), flush=True)
        finally:
            write_json(deployment_path, deployment)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
