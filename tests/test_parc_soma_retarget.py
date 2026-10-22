from pathlib import Path
import inspect
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'out/vendor/kimodo'))
sys.path.insert(0, str(ROOT / 'scripts'))

from prepare_parc_soma import (
    AUDIT_ONLY_BOUNDS, BOUNDS, CRITERIA_REVISION, angular_steps,
    assess_geometric_contact_pairs, geometric_near_sole_contact,
    rotation_tracking_metrics,
)

from motionbricks.data.parc_soma_retarget import (
    MAPPING, SOMA_LOCAL_TO_PARC, Y_TO_Z, official_parc_character,
    reference_calibration, scale_scene, source_foot_geometry, terrain_gaps_zup, terrain_surface_distance_zup,
)
from motionbricks.motionlib.core.skeletons.soma import SOMASkeleton30


@pytest.fixture(scope='module')
def calibration():
    character = official_parc_character(ROOT / 'out/vendor/PARC')
    skeleton = SOMASkeleton30()
    config, metadata = reference_calibration(character, skeleton)
    return character, skeleton, config, metadata


def test_body_basis_maps_forward_left_up_without_reflection():
    np.testing.assert_allclose(SOMA_LOCAL_TO_PARC.T @ SOMA_LOCAL_TO_PARC, np.eye(3))
    assert np.linalg.det(SOMA_LOCAL_TO_PARC) == pytest.approx(1)
    np.testing.assert_allclose(SOMA_LOCAL_TO_PARC @ [0, 0, 1], [1, 0, 0])
    np.testing.assert_allclose(SOMA_LOCAL_TO_PARC @ [1, 0, 0], [0, 1, 0])
    np.testing.assert_allclose(SOMA_LOCAL_TO_PARC @ [0, 1, 0], [0, 0, 1])


def test_scene_scale_preserves_cell_indices_and_scaled_gaps():
    hf = np.array([[0, .2], [.4, .6]])
    origin = np.array([-.2, -.1])
    points = np.array([[-.2, -.1, .05], [0, .1, .7], [3, 0, 0]])
    before, known = terrain_gaps_zup(points, hf, origin, .2)
    shf, so, dx = scale_scene(hf, origin, .2, 1.12)
    after, sknown = terrain_gaps_zup(points * 1.12, shf, so, dx)
    np.testing.assert_array_equal(known, sknown)
    np.testing.assert_allclose(after, before * 1.12, equal_nan=True)
    assert np.isnan(after[-1])
    with pytest.raises(ValueError):
        scale_scene(hf, origin, .2, -1)


def test_all_soma_tasks_have_distinct_aliases_and_source_body_assignments(calibration):
    char, skeleton, config, metadata = calibration
    aliases = [entry[0] for entry in config['ik_match_table1'].values()]
    assert len(set(aliases)) == 30
    assert set(aliases) == set(skeleton.bone_order_names)
    assert len(set(metadata['joint_mapping'].values())) == 15
    assert set(metadata['joint_mapping'].values()) == set(char.get_body_names())
    assert metadata['joint_mapping']['LeftToeBase'] == 'left_foot'
    assert metadata['joint_mapping']['RightToeBase'] == 'right_foot'
    assert metadata['training_approved'] is False
    assert config['ground_height'] == 0
    assert metadata['per_frame_ground_fit'] is False


def test_foot_frames_use_surface_forward_not_sloping_toe_bone(calibration):
    _, skeleton, _, metadata = calibration
    rotations = np.asarray(metadata['reference_global_rot_mats'])
    for side in ('Left', 'Right'):
        foot = skeleton.bone_order_names.index(side + 'Foot')
        np.testing.assert_allclose(rotations[foot], Y_TO_Z.T @ SOMA_LOCAL_TO_PARC)
        np.testing.assert_allclose(rotations[foot] @ [0, 0, 1], [1, 0, 0])


def test_foot_task_exactly_maps_reference_surface_anchor_under_arbitrary_rotation(calibration):
    character, _, config, metadata = calibration
    scale = metadata['uniform_scale']
    for side in ('Left', 'Right'):
        _, source_anchor = source_foot_geometry(character, side)
        entry = config['ik_match_table1'][side + 'Foot']
        offset_rot = Rotation.from_quat(entry[4], scalar_first=True).as_matrix()
        target_anchor_z = Y_TO_Z @ np.asarray(metadata['sole_anchor_local_yup'][side + 'Foot'])
        for rotvec in ([0, 0, 0], [.2, -.4, .8]):
            source_rot = Rotation.from_rotvec(rotvec).as_matrix()
            source_pos = np.array([1.2, -.3, .8])
            desired_rot = source_rot @ offset_rot
            desired_pos = scale * source_pos + desired_rot @ entry[3]
            target_world_anchor = desired_pos + desired_rot @ target_anchor_z
            source_world_anchor = scale * (source_pos + source_rot @ source_anchor)
            np.testing.assert_allclose(target_world_anchor, source_world_anchor, atol=1e-10)


def test_unobserved_joints_inherit_reference_parent_orientation(calibration):
    _, skeleton, _, metadata = calibration
    refs = np.asarray(metadata['reference_global_rot_mats'])
    for index, name in enumerate(skeleton.bone_order_names):
        if name not in MAPPING:
            parent = int(skeleton.joint_parents[index])
            np.testing.assert_allclose(refs[index], refs[parent], atol=1e-12)


def test_official_sdf_includes_step_side_and_correct_interior_distance(calibration):
    hf, origin, dx = np.array([[0.], [1.]]), np.zeros(2), 1.
    points = np.array([[.45, 0, .2], [.55, 0, .2], [0, 0, .1], [3, 0, .1]])
    sdf, known = terrain_surface_distance_zup(points, hf, origin, dx, chunk_size=2)
    np.testing.assert_allclose(sdf[:3], [.05, -.05, .1], atol=1e-6)
    assert not known[-1] and np.isnan(sdf[-1])
    vertical, _ = terrain_gaps_zup(points, hf, origin, dx)
    np.testing.assert_allclose(vertical[:2], [.2, -.8])
    scaled, _ = terrain_surface_distance_zup(points * 2, hf * 2, origin * 2, dx * 2)
    np.testing.assert_allclose(scaled, sdf * 2, atol=2e-6, equal_nan=True)


def test_opposite_rotation_steps_cannot_hide_behind_equal_magnitudes():
    goal = Rotation.from_euler('z', [0, 10], degrees=True).as_matrix()[:, None]
    actual = Rotation.from_euler('z', [0, -10], degrees=True).as_matrix()[:, None]
    np.testing.assert_allclose(angular_steps(actual), angular_steps(goal))
    result = rotation_tracking_metrics(actual, goal)
    assert result['angular_step_error_max_deg'] == pytest.approx(20)
    assert result['task_orientation_p95_deg'] > 18
    world_yaw = Rotation.from_euler('y', 70, degrees=True).as_matrix()
    rotated = rotation_tracking_metrics(world_yaw @ actual, world_yaw @ goal)
    assert rotated == pytest.approx(result)


def test_sidewall_body_contact_with_sole_five_cm_above_top_is_not_paired_or_promoted():
    body_contact = np.array([False, True])
    source_gap = np.full((2, 4), .05)
    known = np.ones((2, 4), dtype=bool)

    candidate = geometric_near_sole_contact(body_contact, source_gap, known)

    assert not candidate[1]
    assert body_contact[1]
    assert CRITERIA_REVISION['candidate_mask_name'] == 'geometric_near_sole_contact'
    assert CRITERIA_REVISION['candidate_mask_is_loadbearing_truth'] is False
    assert CRITERIA_REVISION['source_contact_booleans_changed'] is False


def test_body_contact_with_known_sole_corner_near_top_is_included():
    body_contact = np.array([False, True])
    source_gap = np.array([[np.nan] * 4, [.04, .018, -.014, .03]])
    known = np.array([[False] * 4, [True] * 4])

    candidate = geometric_near_sole_contact(body_contact, source_gap, known)

    assert candidate.tolist() == [False, True]


def test_reset_frame_zero_is_unknown_and_never_eligible():
    body_contact = np.array([True, True])
    source_gap = np.array([[.001] * 4, [.002] * 4])
    known = np.array([[False] * 4, [True] * 4])

    candidate = geometric_near_sole_contact(body_contact, source_gap, known)

    assert candidate.tolist() == [False, True]
    assert CRITERIA_REVISION['candidate_mask_excludes_reset_frame0'] is True


def test_target_gap_values_cannot_control_source_pairing_mask():
    body_contact = np.array([False, True, True])
    source_gap = np.array([[np.nan] * 4, [.01, .03, .04, .05], [.05] * 4])
    known = np.array([[False] * 4, [True] * 4, [True] * 4])
    target_gaps_a = np.array([[.2] * 4, [.2] * 4, [.2] * 4])
    target_gaps_b = np.array([[0.] * 4, [0.] * 4, [0.] * 4])

    candidate_a = geometric_near_sole_contact(body_contact, source_gap, known)
    candidate_b = geometric_near_sole_contact(body_contact, source_gap, known)

    np.testing.assert_array_equal(candidate_a, candidate_b)
    assert not np.array_equal(target_gaps_a, target_gaps_b)
    assert 'target_gaps' not in inspect.signature(geometric_near_sole_contact).parameters
    assert BOUNDS['geometric_near_sole_contact_target_top_vertical_gap_p95_m'] == .03
    assert AUDIT_ONLY_BOUNDS['labelled_sole_surface_distance_p95_m'] == .03


def test_zero_geometric_pairs_are_not_assessed_and_fail_screening():
    result = assess_geometric_contact_pairs([])

    assert result == {
        'assessment_status': 'not_assessed',
        'target_gap_p95_m': None,
        'screen_failure': 'geometric_near_sole_contact_not_assessed_zero_eligible_frames',
    }
