"""Bind the published G1 robot and its kinematically compatible MB calibration."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from motionbricks.mb_sonic.reference import OFFICIAL_BODY_NAMES
from motionbricks.mb_sonic.retarget import G1Kinematics
from motionbricks.mb_sonic.scene import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--source-robot", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--reference-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repository.resolve(strict=True)
    original = repo/"protomotions/data/assets/mjcf/g1_holo_compat.xml"
    urdf = repo/"protomotions/data/assets/urdf/for_retargeting/g1.urdf"
    args.output.mkdir(parents=True, exist_ok=False)
    root = ET.parse(original).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", str((original.parent/compiler.attrib["meshdir"]).resolve(strict=True)))
    contact, external_pairs = root.find("contact"), ET.Element("contact")
    for pair in list(contact):
        if pair.tag != "pair" or pair.attrib["geom2"] != "floor":
            raise ValueError("Unexpected robot contact template")
        contact.remove(pair)
        external_pairs.append(pair)
    robot = args.output/"g1.xml"
    ET.ElementTree(root).write(robot, encoding="utf-8", xml_declaration=True)
    pair_path = args.output/"scene_contact_pairs.xml"
    ET.ElementTree(external_pairs).write(pair_path, encoding="utf-8", xml_declaration=True)
    shutil.copyfile(urdf, robot.with_suffix(".urdf"))
    old, new = G1Kinematics(args.source_robot), G1Kinematics(robot)
    with np.load(args.reference_probe, allow_pickle=False) as archive:
        qpos = archive["solved_qpos"]
    old_pos, old_rot = old.fk(qpos)
    new_pos, new_rot = new.fk(qpos)
    by_body = {}
    for body in OFFICIAL_BODY_NAMES:
        a, b = old.body_names.index(body), new.body_names.index(body)
        by_body[body] = dict(position_max_m=float(np.linalg.norm(new_pos[:, b]-old_pos[:, a], axis=1).max()),
                             rotation_matrix_max=float(np.abs(new_rot[:, b]-old_rot[:, a]).max()))
    if any(row["rotation_matrix_max"] > 3e-5 for row in by_body.values()):
        raise ValueError("The control-reference orientation channels changed between robot assets")
    lower_bodies = OFFICIAL_BODY_NAMES[:7]
    if any(by_body[body]["position_max_m"] > 3e-5 for body in lower_bodies):
        raise ValueError("Root/leg calibration geometry differs")
    np.testing.assert_array_equal(old.limits, new.limits)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if calibration["robot_xml_sha256"] != sha256(args.source_robot):
        raise ValueError("Source calibration robot binding differs")
    proof = dict(schema="motionbricks.protomotions.robot_binding.v1", optimizer_updates=0,
        source_robot=dict(path=str(args.source_robot.resolve()), sha256=sha256(args.source_robot)),
        target_robot=dict(path=str(robot.resolve()), sha256=sha256(robot)),
        upstream_robot=dict(path=str(original), sha256=sha256(original)),
        urdf=dict(path=str(robot.with_suffix('.urdf').resolve()), sha256=sha256(urdf)),
        external_contact_pairs=dict(path=str(pair_path.resolve()), sha256=sha256(pair_path), count=len(external_pairs)),
        calibration_parent=dict(path=str(args.calibration.resolve()), sha256=sha256(args.calibration), identity=calibration["identity"]),
        reference_probe=dict(path=str(args.reference_probe.resolve()), sha256=sha256(args.reference_probe), frames=len(qpos)),
        named_body_comparison=by_body, joint_ranges_equal=True,
        preserved_control_reference_channels=["root_pose", "joint_positions", "joint_velocities", "body_orientations"],
        changed_physical_asset="published29DOF_G1_fixed_rubber_hands; different_inertia_collision_geometry",
        upper_body_origin_changes_retained=True,
        transfer_scope="unchanged rotation calibration and root/leg length offset; no trajectory or weight fitting")
    proof_path = args.output/"binding.json"
    proof_path.write_text(json.dumps(proof, indent=2)+"\n", encoding="utf-8")
    calibration["robot_xml_sha256"] = sha256(robot)
    calibration.pop("identity")
    calibration["kinematic_transfer"] = dict(parent_sha256=sha256(args.calibration), proof_path=str(proof_path.resolve()),
                                              proof_sha256=sha256(proof_path))
    payload = json.dumps(calibration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    calibration["identity"] = hashlib.sha256(payload).hexdigest()
    (args.output/"calibration.json").write_text(json.dumps(calibration, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(proof=str(proof_path.resolve()), robot_sha256=sha256(robot),
        calibration_sha256=sha256(args.output/"calibration.json"), contact_pairs_sha256=sha256(pair_path)), indent=2))


if __name__ == "__main__":
    main()
