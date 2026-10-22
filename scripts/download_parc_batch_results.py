"""Fetch completed PARC candidates with bounded transfers and checked extraction."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

import deploy_parc_remote_batch as deploy


def unpack_checked(archive, destination):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        if len({m.name for m in members}) != len(members):
            raise ValueError("Duplicate archive members")
        for member in members:
            relative = PurePosixPath(member.name)
            if (not member.isfile() or relative.is_absolute() or ".." in relative.parts
                    or "\\" in member.name or ":" in member.name
                    or relative.as_posix() != member.name):
                raise ValueError(f"Rejected archive member: {member.name}")
            target = destination.joinpath(*relative.parts)
            target.resolve().relative_to(destination)
            with stream.extractfile(member) as source:
                if target.exists():
                    if hashlib.sha256(source.read()).hexdigest() != deploy.digest(target):
                        raise ValueError(f"Refusing changed output: {target}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    while block := source.read(1024 * 1024):
                        output.write(block)


def download(host, run_id, output):
    if host not in deploy.HOSTS or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("Invalid host or run ID")
    bootstrap = json.loads((deploy.ROOT / "out/parc_remote_deployment_20261008/finish_bootstrap.json").read_text())
    result = json.loads(deploy.checked_run(deploy.ssh(host, ["python3", "-B", bootstrap["script"],
        "pack", "--run-id", run_id]), timeout=180))
    if not isinstance(result.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", result["sha256"]):
        raise ValueError("Invalid archive SHA256")
    remote_archive = PurePosixPath(result["archive"])
    expected_parent = PurePosixPath(deploy.REMOTE_BASE.as_posix()) / "runs" / run_id
    if remote_archive.parent != expected_parent or remote_archive.suffix != ".tar":
        raise ValueError("Archive is outside the requested remote run")
    if type(result.get("bytes")) is not int or not 0 < result["bytes"] < 1_000_000_000:
        raise ValueError("Unexpected download size: use NAS for large transfers")
    archive = output / "archives" / (host + "_" + result["sha256"] + ".tar")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        ledger = deploy.ROOT / "out/parc_remote_deployment_20261008/transfer_ledger.json"
        deploy.reserve_transfer(ledger, result["bytes"], host, archive)
        with archive.open("xb") as file:
            subprocess.run(deploy.ssh(host, ["cat", result["archive"]]), stdout=file,
                           check=True, timeout=600)
    if archive.stat().st_size != result["bytes"] or deploy.digest(archive) != result["sha256"]:
        raise ValueError("Archive byte count or SHA256 mismatch")
    destination = output / "downloads" / host / run_id
    unpack_checked(archive, destination)
    for job in result["jobs"]:
        directory = destination / "results" / job
        report = json.loads((directory / "generation.json").read_text())
        for row in report["candidates"]:
            path = directory / f"candidate_{row['rank']:03d}" / "generated.pkl"
            if deploy.digest(path) != row["sha256"]:
                raise ValueError(f"Local candidate hash mismatch: {path}")
    result.update(host=host, run_id=run_id, local_directory=str(destination.resolve()))
    deploy.write_json(destination / "download_manifest.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("g2", "g3"), required=True)
    parser.add_argument("--run-id", default="ema_stride5_20261008_v1")
    parser.add_argument("--output", type=Path, default=deploy.ROOT / "out/parc_complete_20261008")
    args = parser.parse_args()
    download(args.host, args.run_id, args.output)
