"""Export four real V7 fixtures with the official Kimodo skeleton, never G1 weights."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.soma_replay import replay_fixture, skeleton_manifest


CLIPS = ("idle_loop_002__A065", "walk_forward_start_002__A023",
         "jog_forward_stop_003__A021", "turn_walk_270_003__A070")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    contract = json.loads((args.dataset / "skeleton_contract.json").read_text())
    if contract["preprocess_version"] != "soma60-planar-v7":
        raise ValueError("This replay review targets the frozen V7 preprocessing")
    manifest = skeleton_manifest()
    if contract["joint_names"] != [joint["name"] for joint in manifest["joints"]] or contract["parents"] != [joint["parent_index"] for joint in manifest["joints"]]:
        raise ValueError("Dataset hierarchy differs from the official skeleton")
    np.testing.assert_allclose(contract["neutral_joints_m"], [joint["neutral_position"] for joint in manifest["joints"]], atol=1e-7)
    np.testing.assert_allclose(contract["sole_offsets_m"], manifest["sole_offsets_m"], atol=1e-7)
    if contract["kimodo_revision"] != manifest["provenance"]["commit"] or contract["sole_mesh_sha256"] != manifest["sole_mesh_sha256"]:
        raise ValueError("Dataset skeleton/sole asset revision differs")
    records = {record["filename"]: record for record in json.loads((args.dataset / "accepted.json").read_text())}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "hierarchy").mkdir(exist_ok=True)
    (args.output / "hierarchy/soma30_kimodo_standard.json").write_text(json.dumps(manifest, indent=2))
    results = []
    for name in CLIPS:
        record = records[name]
        shard = (args.dataset / record["shard"]).resolve()
        if not shard.is_relative_to(args.dataset.resolve()):
            raise ValueError("Shard path escaped the dataset")
        with np.load(shard, allow_pickle=False) as arrays:
            fixture = replay_fixture(arrays, record, manifest, hashlib.sha256(shard.read_bytes()).hexdigest())
        output = args.output / (record["behavior"] + ".json")
        output.write_text(json.dumps(fixture, separators=(",", ":")))
        results.append({"filename": output.name, "source_clip": name, "frames": len(fixture["frames"]),
                        "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    report = {"runtime_state": "source_replay", "model_inference": False, "source_id": manifest["source_id"], "fixtures": results}
    (args.output / "export_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
