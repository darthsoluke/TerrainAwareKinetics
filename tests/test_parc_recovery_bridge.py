from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from motionbricks.data.parc_soma_retarget import MAPPING, Y_TO_Z, official_parc_character, reference_calibration
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30
from scripts.audit_parc_recovery_bridge import inverse_calibrated_tasks


def test_inverse_task_calibration_recovers_native_parc_fk_without_projection():
    root_dir = Path(__file__).resolve().parents[1]
    character = official_parc_character(root_dir / 'out/vendor/PARC')
    skeleton = SOMASkeleton30()
    config, calibration = reference_calibration(character, skeleton)
    scale = calibration['uniform_scale']
    torch.manual_seed(83)
    count = 4
    root = torch.randn(count, 3)
    root_rotation = torch.from_numpy(Rotation.from_rotvec(np.array([[0., 0., yaw] for yaw in (0., .4, -.6, 1.2)])).as_quat()).float()
    joint_rotation = character.dof_to_rot(torch.randn(count, character.get_dof_size()) * .15)
    positions, rotations = character.forward_kinematics(root, root_rotation, joint_rotation)
    matrices = Rotation.from_quat(rotations.numpy().reshape(-1, 4)).as_matrix().reshape(count, 15, 3, 3)
    soma_names = list(skeleton.bone_order_names)
    parc_names = character.get_body_names()
    mapped_positions = np.zeros((count, len(soma_names), 3))
    mapped_rotations = np.broadcast_to(np.eye(3), (count, len(soma_names), 3, 3)).copy()
    for soma_name, parc_name in MAPPING.items():
        ti, si = soma_names.index(soma_name), parc_names.index(parc_name)
        entry = config['ik_match_table1'][soma_name]
        desired_z = matrices[:, si] @ Rotation.from_quat(entry[4], scalar_first=True).as_matrix()
        mapped_rotations[:, ti] = Y_TO_Z.T @ desired_z @ Y_TO_Z
        mapped_positions[:, ti] = (scale * positions[:, si].numpy()
                                   + (desired_z @ np.asarray(entry[3])[..., None])[..., 0]) @ Y_TO_Z
    recovered = inverse_calibrated_tasks(mapped_positions, mapped_rotations, soma_names, character, config, scale)
    np.testing.assert_allclose(recovered['root_pos_zup'], root.numpy(), atol=1e-6, rtol=0)
    np.testing.assert_allclose(recovered['positions_actual_parc_fk'], positions.numpy(), atol=1e-6, rtol=0)
    assert recovered['metrics']['task_position_residual_max_soma_m'] < 1e-6
    assert recovered['metrics']['rotation_inverse_fk_max_deg'] < .001
    assert recovered['metrics']['native_dof_incompatibility_max_deg'] < .001
