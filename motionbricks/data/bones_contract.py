"""Content-bound dataset identity for reproducible human MB experiments."""

import hashlib
import json
from pathlib import Path

import numpy as np

from motionbricks.data.bones_seed import SOLE_PROXY_GEOMETRIES, SOLE_PROXY_GEOMETRY_OF_SHIPPED_DATASET
from motionbricks.data.soma_replay import skeleton_manifest


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_snapshot(folder):
    folder = Path(folder).resolve()
    manifest = folder / "accepted.json"
    records = json.loads(manifest.read_text())
    report = json.loads((folder / "quality_report.json").read_text())
    contract = json.loads((folder / "skeleton_contract.json").read_text())
    official = skeleton_manifest()
    for key, expected in (("joint_names", [joint["name"] for joint in official["joints"]]),
                          ("parents", [joint["parent_index"] for joint in official["joints"]]),
                          ("kimodo_revision", official["provenance"]["commit"]),
                          ("foot_channels", official["foot_channels"]),
                          ("sole_mesh_sha256", official["sole_mesh_sha256"])):
        if contract[key] != expected:
            raise ValueError("Dataset skeleton contract mismatch: " + key)
    # A dataset is bound to the proxy geometry it was converted with, not to whatever
    # the code default has since become. Contracts written before the revisions existed
    # were all built with the position-only geometry.
    revision = contract.get("sole_proxy_geometry", SOLE_PROXY_GEOMETRY_OF_SHIPPED_DATASET)
    if revision not in SOLE_PROXY_GEOMETRIES:
        raise ValueError("Dataset contract names an unknown sole proxy geometry revision: " + str(revision))
    if revision not in official.get("sole_offsets_by_revision", {}):
        raise ValueError("Skeleton manifest does not carry revision " + str(revision))
    for key, expected in (("neutral_joints_m", [joint["neutral_position"] for joint in official["joints"]]),
                          ("sole_offsets_m", official["sole_offsets_by_revision"][revision])):
        np.testing.assert_allclose(contract[key], expected, rtol=0, atol=1e-7)
    manifest_hash = file_sha256(manifest)
    stats = (folder / report["stats_folder"]).resolve()
    if not stats.is_relative_to(folder):
        raise ValueError("Statistics path escaped the dataset")
    stats_contract = json.loads((stats / "contract.json").read_text())
    if stats_contract["source_manifest_sha256"] != manifest_hash or stats_contract["split"] != "train":
        raise ValueError("Statistics were not computed from this train-only manifest")
    groups, sources, shards = {}, {}, {}
    for record in records:
        if record["split"] not in ("train", "validation", "test"):
            raise ValueError("Unknown dataset split")
        for identifier, mapping in ((record["recording_group"], groups), (record["source_sha256"], sources)):
            if mapping.setdefault(identifier, record["split"]) != record["split"]:
                raise ValueError("Recording or source duplicate leaks across splits")
        shard = (folder / record["shard"]).resolve()
        if not shard.is_relative_to(folder):
            raise ValueError("Shard path escaped the dataset")
        name = shard.relative_to(folder).as_posix()
        if name not in shards:
            shards[name] = {"sha256": file_sha256(shard), "bytes": shard.stat().st_size,
                            "split": record["split"]}
        elif shards[name]["split"] != record["split"]:
            raise ValueError("Shared shard leaks across splits")
    snapshot = {
        "schema": "motionbricks.bones.dataset_snapshot.v1", "dataset": str(folder),
        "accepted_sha256": manifest_hash, "quality_report_sha256": file_sha256(folder / "quality_report.json"),
        "skeleton_contract_sha256": file_sha256(folder / "skeleton_contract.json"),
        "source_id": official["source_id"], "stats_folder": report["stats_folder"],
        "stats_contract_sha256": file_sha256(stats / "contract.json"),
        "stats_sha256": hashlib.sha256(b"".join((stats / name).read_bytes()
                                              for name in ("mean.npy", "std.npy"))).hexdigest(),
        "shards": shards,
    }
    snapshot["sha256"] = hashlib.sha256(json.dumps(snapshot, sort_keys=True,
                                                 separators=(",", ":")).encode()).hexdigest()
    return snapshot


def require_same_snapshot(previous, current):
    if not previous or previous.get("sha256") != current["sha256"]:
        raise ValueError("Dataset content changed, or checkpoint lacks an immutable dataset snapshot")
