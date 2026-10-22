"""Create a private conda runtime and fetch public upstream companions on Linux."""

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path


KIMODO_REV = "58e781898b3d7e328a676a75d3e338c45dce3ad9"
MINIFORGE_RELEASE = "26.7.2-0"


def run(*command):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def download(url, path):
    if not path.exists():
        print("Downloading", url, flush=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    downloads = project / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    installer = downloads / "Miniforge3-Linux-x86_64.sh"
    url = ("https://github.com/conda-forge/miniforge/releases/download/"
           f"{MINIFORGE_RELEASE}/{installer.name}")
    download(url, installer)
    checksum = downloads / (installer.name + ".sha256")
    download(url.replace(installer.name,
                         f"Miniforge3-{MINIFORGE_RELEASE}-Linux-x86_64.sh.sha256"), checksum)
    expected = checksum.read_text().split()[0]
    actual = hashlib.sha256(installer.read_bytes()).hexdigest()
    if expected != actual:
        raise RuntimeError("Miniforge installer SHA256 mismatch")
    prefix = project / "miniforge3"
    if not (prefix / "bin/conda").exists():
        run("bash", installer, "-b", "-p", prefix)
    environment = project / "envs/mb-bones"
    if not (environment / "bin/python").exists():
        run(prefix / "bin/conda", "create", "-y", "-p", environment,
            "--override-channels", "-c", "conda-forge", "python=3.11", "pip")
    python = environment / "bin/python"
    # CUDA 12.6 matches the server's 560-series driver; do not change the driver.
    run(python, "-m", "pip", "install", "torch==2.11.0", "--index-url",
        "https://download.pytorch.org/whl/cu126")
    kimodo = project / "kimodo"
    if not (kimodo / ".git").exists():
        run("git", "-c", "filter.lfs.smudge=", "-c", "filter.lfs.required=false",
            "clone", "--depth", "1", "https://github.com/nv-tlabs/kimodo.git", kimodo)
    run("git", "-C", kimodo, "fetch", "--depth", "1", "origin", KIMODO_REV)
    run("git", "-C", kimodo, "checkout", "--detach", KIMODO_REV)
    run(python, "-m", "pip", "install", "-e", kimodo,
        "-e", project / "GR00T-WholeBodyControl/motionbricks", "pytest")
    run(python, "-m", "pip", "install", "vector-quantize-pytorch==1.28.2",
        "pytorch-lightning==2.6.1", "adam-atan2-pytorch==0.3.3")
    request = urllib.request.Request(
        "https://huggingface.co/api/datasets/nvidia/SEED-Timeline-Annotations",
        headers={"User-Agent": "motionbricks-data-preparation"})
    with urllib.request.urlopen(request) as response:
        revision = json.load(response)["sha"]
    timelines = downloads / "timelines.jsonl"
    download("https://huggingface.co/datasets/nvidia/SEED-Timeline-Annotations/"
             f"resolve/{revision}/timelines.jsonl", timelines)
    provenance = {
        "miniforge_release": MINIFORGE_RELEASE, "installer_sha256": actual,
        "kimodo_revision": KIMODO_REV, "timeline_revision": revision,
        "timeline_sha256": hashlib.sha256(timelines.read_bytes()).hexdigest(),
        "environment": str(environment),
    }
    (project / "bootstrap_provenance.json").write_text(json.dumps(provenance, indent=2))
    run(python, "-c", "import torch; print(torch.__version__, torch.cuda.is_available())")
    print("Bootstrap complete", flush=True)


if __name__ == "__main__":
    main()
