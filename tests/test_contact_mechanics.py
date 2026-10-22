import math

import pytest
import torch

from motionbricks.data.contact_forces import ground_force_feasibility
from motionbricks.data.contact_mechanics import (
    body_contact_state,
    coulomb_contact_residuals,
    flat_ground_contact_kinematics,
    interval_average,
    masked_summary,
    support_frame_mask,
    support_interval_masks,
    sustained_support_segments,
)

MU_STATIC, MU_DYNAMIC = 0.6, 0.4
GRAVITY_MPS2 = 9.81
BLOCK_MASS_KG = 5.0
BLOCK_WEIGHT_N = BLOCK_MASS_KG * GRAVITY_MPS2
DTYPE = torch.float64


def residuals(gap=0.0, velocity=(0.0, 0.0, 0.0), force=(0.0, BLOCK_WEIGHT_N, 0.0), **kwargs):
    """One analytic point contact; the block weight is a fixture, not a foot load."""
    return coulomb_contact_residuals(
        torch.tensor(gap, dtype=DTYPE), torch.tensor(velocity, dtype=DTYPE),
        torch.tensor(force, dtype=DTYPE), static_friction=MU_STATIC, dynamic_friction=MU_DYNAMIC, **kwargs)


def zero_residuals(result):
    return {name: value for name, value in result.items() if value.dtype != torch.bool}


def test_flat_floor_decomposes_motion_with_explicit_interval_alignment():
    soles = torch.zeros(2, 3, 4, 3, dtype=DTYPE)
    soles[..., 0] = torch.arange(3, dtype=DTYPE)[None, :, None] / 60
    soles[..., 1] = 0.25 + 2 * torch.arange(3, dtype=DTYPE)[None, :, None] / 60
    soles[..., 2] = 3 * torch.arange(3, dtype=DTYPE)[None, :, None] / 60
    result = flat_ground_contact_kinematics(soles, 60, ground_height_m=0.25)
    assert result['gap_m'].shape == (2, 3, 4)
    assert result['penetration_m'].shape == (2, 3, 4)
    for name in ('interval_gap_m', 'interval_normal_velocity_mps', 'interval_tangential_velocity_mps',
                 'interval_tangential_speed_mps'):
        assert result[name].shape == (2, 2, 4) or result[name].shape == (2, 2, 4, 2)
    torch.testing.assert_close(result['interval_normal_velocity_mps'], torch.full((2, 2, 4), 2.0, dtype=DTYPE))
    torch.testing.assert_close(result['interval_tangential_velocity_mps'],
                               torch.tensor([1.0, 3.0], dtype=DTYPE).expand(2, 2, 4, 2))
    torch.testing.assert_close(result['interval_tangential_speed_mps'],
                               torch.full((2, 2, 4), math.sqrt(10), dtype=DTYPE))
    # Interval gaps are the two-endpoint mean, the documented discrete convention.
    torch.testing.assert_close(result['interval_gap_m'], interval_average(result['gap_m']))
    expected_interval_gap = torch.tensor([1.0, 3.0], dtype=DTYPE)[None, :, None].expand(2, 2, 4) / 60
    torch.testing.assert_close(result['interval_gap_m'], expected_interval_gap)
    assert not result['penetration_m'].any()


def test_ground_and_rigid_translation_leave_contact_kinematics_unchanged():
    soles = torch.zeros(1, 3, 4, 3, dtype=DTYPE)
    soles[..., 1] = -0.02
    soles[..., 0] = torch.arange(3, dtype=DTYPE)[None, :, None] * 0.01
    original = flat_ground_contact_kinematics(soles, 60)
    shifted = flat_ground_contact_kinematics(soles + torch.tensor([8.0, 1.0, -9.0], dtype=DTYPE), 60, 1.0)
    for name in original:
        torch.testing.assert_close(original[name], shifted[name])
    assert float(original['penetration_m'].mean()) == pytest.approx(0.02)


def test_rigid_yaw_rotation_preserves_contact_kinematics():
    generator = torch.Generator().manual_seed(7)
    soles = torch.rand(2, 5, 4, 3, dtype=DTYPE, generator=generator)
    cosine, sine = math.cos(0.7), math.sin(0.7)
    yaw = torch.tensor([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]], dtype=DTYPE)
    original = flat_ground_contact_kinematics(soles, 60, 0.3)
    rotated = flat_ground_contact_kinematics(soles @ yaw.T, 60, 0.3)
    for name in ('gap_m', 'penetration_m', 'interval_gap_m', 'interval_normal_velocity_mps',
                 'interval_tangential_speed_mps'):
        torch.testing.assert_close(original[name], rotated[name], atol=1e-12, rtol=1e-12)
    # The tangential velocity is a vector in the floor plane, so it rotates with the body.
    planar_yaw = yaw[[0, 2]][:, [0, 2]]
    torch.testing.assert_close(rotated['interval_tangential_velocity_mps'],
                               original['interval_tangential_velocity_mps'] @ planar_yaw.T,
                               atol=1e-12, rtol=1e-12)


def test_resampling_a_linear_path_preserves_physical_velocities():
    def linear_path(fps):
        times = torch.arange(int(round(0.2 * fps)) + 1, dtype=DTYPE) / fps
        soles = torch.zeros(1, len(times), 1, 3, dtype=DTYPE)
        soles[..., 0] = 0.4 * times[None, :, None]
        soles[..., 1] = 0.05 + 0.3 * times[None, :, None]
        soles[..., 2] = -0.2 * times[None, :, None]
        return flat_ground_contact_kinematics(soles, fps)

    slow, fast = linear_path(30), linear_path(120)
    for name in ('interval_normal_velocity_mps', 'interval_tangential_speed_mps'):
        torch.testing.assert_close(slow[name].mean(), fast[name].mean(), atol=1e-12, rtol=1e-9)
    assert float(slow['interval_normal_velocity_mps'].mean()) == pytest.approx(0.3)
    assert float(slow['interval_tangential_speed_mps'].mean()) == pytest.approx(math.hypot(0.4, 0.2))


def test_declared_frame_rate_scales_interval_velocities():
    soles = torch.zeros(1, 3, 4, 3, dtype=DTYPE)
    soles[..., 1] = torch.arange(3, dtype=DTYPE)[None, :, None] / 60
    full_rate = flat_ground_contact_kinematics(soles, 60)
    half_rate = flat_ground_contact_kinematics(soles, 30)
    torch.testing.assert_close(half_rate['interval_normal_velocity_mps'],
                               full_rate['interval_normal_velocity_mps'] / 2)
    # Gaps and interval gaps are frame quantities and do not depend on the declared rate.
    torch.testing.assert_close(half_rate['gap_m'], full_rate['gap_m'])


@pytest.mark.parametrize('fps', [0, -60, float('nan'), float('inf')])
def test_contact_kinematics_rejects_invalid_time_units(fps):
    with pytest.raises(ValueError, match='FPS'):
        flat_ground_contact_kinematics(torch.zeros(1, 3, 4, 3, dtype=DTYPE), fps)


def test_interval_average_rejects_a_single_frame():
    with pytest.raises(ValueError, match='two frames'):
        interval_average(torch.zeros(4, dtype=DTYPE))


def test_support_frame_mask_requires_both_label_and_confidence():
    contacts = torch.tensor([[True], [True], [False], [False]])
    confidence = torch.tensor([[True], [False], [True], [False]])
    assert torch.equal(support_frame_mask(contacts, confidence),
                       torch.tensor([[True], [False], [False], [False]]))


def test_unknown_and_transition_intervals_are_never_sustained_support():
    contacts = torch.tensor([[True], [True], [True], [False], [False], [True]])
    confidence = torch.tensor([[True], [True], [False], [True], [True], [True]])
    masks = support_interval_masks(contacts, confidence)
    for name in masks:
        assert masks[name].shape == (5, 1)
    assert torch.equal(masks['sustained_support'], torch.tensor([[True], [False], [False], [False], [False]]))
    assert torch.equal(masks['unknown'], torch.tensor([[False], [True], [True], [False], [False]]))
    assert torch.equal(masks['transition'], torch.tensor([[False], [False], [False], [False], [True]]))
    assert torch.equal(masks['swing'], torch.tensor([[False], [False], [False], [True], [False]]))
    # The four interval classes partition the interval axis exactly once.
    stacked = torch.stack([masks[name] for name in masks])
    assert int(stacked.sum(0).min()) == 1 and int(stacked.sum(0).max()) == 1


def test_contact_masks_reject_mismatched_or_non_boolean_inputs():
    with pytest.raises(ValueError, match='share a'):
        support_interval_masks(torch.zeros(4, 2, dtype=torch.bool), torch.zeros(4, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match='boolean'):
        support_interval_masks(torch.zeros(4, 2), torch.zeros(4, 2, dtype=torch.bool))


def test_masked_summary_reports_nulls_instead_of_zero_error():
    values = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    empty = masked_summary(values, torch.zeros(3, dtype=torch.bool))
    assert empty == {'samples': 0, 'mean': None, 'minimum': None, 'p95': None, 'peak': None}
    filled = masked_summary(values, torch.tensor([True, False, True]))
    assert filled['samples'] == 2 and filled['mean'] == pytest.approx(2.0) and filled['peak'] == pytest.approx(3.0)


def test_masked_summary_pools_samples_instead_of_averaging_window_quantiles():
    short = torch.arange(100, dtype=DTYPE)
    long = torch.arange(1000, dtype=DTYPE)
    pooled = masked_summary(torch.cat([short, long]))
    average_of_window_p95 = (masked_summary(short)['p95'] + masked_summary(long)['p95']) / 2
    # Pooled samples: 0..99 twice, then 100..999, so the 95th percentile is 944.05.
    assert pooled['p95'] == pytest.approx(944.05)
    assert pooled['p95'] > average_of_window_p95


def test_sustained_support_segments_report_path_net_and_normal_drift():
    frames, channels = 6, 2
    sole = torch.zeros(frames, channels, 3, dtype=DTYPE)
    sole[:, 0, 0] = 0.01 * torch.arange(frames, dtype=DTYPE)
    sole[:, 1, 0] = 0.01 * torch.arange(frames, dtype=DTYPE)
    gap = torch.zeros(frames, channels, dtype=DTYPE)
    gap[:, 0] = 0.005 * torch.arange(frames, dtype=DTYPE)
    support = torch.tensor([[True, True], [True, True], [True, False],
                            [True, True], [True, True]])
    segments = sustained_support_segments(sole, gap, support)
    assert len(segments[0]) == 1 and len(segments[1]) == 2
    whole = segments[0][0]
    assert (whole['start_frame'], whole['end_frame']) == (0, 5)
    assert whole['intervals'] == 5 and whole['frames'] == 6
    assert whole['path_length_m'] == pytest.approx(0.05)
    assert whole['net_drift_m'] == pytest.approx(0.05)
    assert whole['max_anchor_m'] == pytest.approx(0.05)
    assert whole['normal_range_m'] == pytest.approx(0.025)
    assert whole['normal_net_m'] == pytest.approx(0.025)
    first, second = segments[1]
    assert (first['start_frame'], first['end_frame']) == (0, 2)
    assert (second['start_frame'], second['end_frame']) == (3, 5)
    assert first['path_length_m'] == pytest.approx(0.02) and second['net_drift_m'] == pytest.approx(0.02)


def test_sustained_support_segments_separate_oscillation_from_steady_drift():
    frames = 5
    sole = torch.zeros(frames, 1, 3, dtype=DTYPE)
    sole[:, 0, 0] = torch.tensor([0.0, 0.01, 0.0, 0.01, 0.0], dtype=DTYPE)
    gap = torch.zeros(frames, 1, dtype=DTYPE)
    support = torch.ones(frames - 1, 1, dtype=torch.bool)
    segment = sustained_support_segments(sole, gap, support)[0][0]
    assert segment['path_length_m'] == pytest.approx(0.04)
    assert segment['net_drift_m'] == pytest.approx(0.0)
    assert segment['max_anchor_m'] == pytest.approx(0.01)


def test_static_block_carries_its_weight_through_the_floor():
    result = residuals()
    assert not result['kinetic_friction_branch']
    assert all(float(value) == pytest.approx(0.0) for value in zero_residuals(result).values())


def test_static_friction_is_set_valued_up_to_its_limit():
    limit_n = MU_STATIC * BLOCK_WEIGHT_N
    for force_x in (0.0, -0.5 * limit_n, limit_n):
        result = residuals(force=(force_x, BLOCK_WEIGHT_N, 0.0))
        assert not result['kinetic_friction_branch']
        assert float(result['friction_cone_excess_n']) == pytest.approx(0.0)
        assert float(result['kinetic_friction_residual_n']) == pytest.approx(0.0)
    beyond = residuals(force=(limit_n + 1.0, BLOCK_WEIGHT_N, 0.0))
    assert float(beyond['friction_cone_excess_n']) == pytest.approx(1.0)


def test_zero_friction_admits_no_tangential_force():
    at_rest = coulomb_contact_residuals(torch.tensor(0.0, dtype=DTYPE), torch.zeros(3, dtype=DTYPE),
                                        torch.tensor([2.0, BLOCK_WEIGHT_N, 0.0], dtype=DTYPE),
                                        static_friction=0.0, dynamic_friction=0.0)
    assert float(at_rest['friction_cone_excess_n']) == pytest.approx(2.0)
    assert float(at_rest['friction_energy_generation_w']) == pytest.approx(0.0)
    sliding = coulomb_contact_residuals(torch.tensor(0.0, dtype=DTYPE),
                                        torch.tensor([1.0, 0.0, 0.0], dtype=DTYPE),
                                        torch.tensor([2.0, BLOCK_WEIGHT_N, 0.0], dtype=DTYPE),
                                        static_friction=0.0, dynamic_friction=0.0)
    assert sliding['kinetic_friction_branch']
    assert float(sliding['kinetic_friction_residual_n']) == pytest.approx(2.0)


@pytest.mark.parametrize('direction', [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                                       (3.0, 4.0), (-3.0, -4.0)])
def test_kinetic_friction_opposes_motion_in_every_direction(direction):
    speed_mps = 2.0
    velocity = (speed_mps * direction[0], 0.0, speed_mps * direction[1])
    magnitude = MU_DYNAMIC * BLOCK_WEIGHT_N
    unit = math.hypot(*direction)
    force = (-magnitude * direction[0] / unit, BLOCK_WEIGHT_N, -magnitude * direction[1] / unit)
    result = residuals(velocity=velocity, force=force)
    assert result['kinetic_friction_branch']
    assert float(result['kinetic_friction_residual_n']) == pytest.approx(0.0, abs=1e-12)
    assert float(result['friction_energy_generation_w']) == pytest.approx(0.0)
    # Doubling the load and the tangential force together keeps the law satisfied.
    scaled = residuals(velocity=velocity, force=tuple(2 * value for value in force))
    assert float(scaled['kinetic_friction_residual_n']) == pytest.approx(0.0, abs=1e-12)
    # Doubling only the tangential force unbalances the cone by mu_k * Fn.
    overloaded = residuals(velocity=velocity, force=(2 * force[0], force[1], 2 * force[2]))
    assert float(overloaded['kinetic_friction_residual_n']) == pytest.approx(magnitude, abs=1e-9)


def test_kinetic_friction_branch_is_a_branch_not_a_contact_state():
    airborne = residuals(gap=0.5, velocity=(4.0, -2.0, 3.0), force=(0.0, 0.0, 0.0))
    assert airborne['kinetic_friction_branch']
    assert all(float(value) == pytest.approx(0.0) for value in zero_residuals(airborne).values())
    loaded_swing = residuals(gap=0.5, force=(2.0, 0.0, 0.0))
    assert float(loaded_swing['contact_complementarity_nm']) == pytest.approx(0.0)
    assert float(loaded_swing['friction_cone_excess_n']) == pytest.approx(2.0)


def test_kinetic_friction_never_does_positive_work_on_the_body():
    velocity = (3.0, 0.0, 4.0)
    magnitude = MU_DYNAMIC * BLOCK_WEIGHT_N
    opposing = residuals(velocity=velocity, force=(-0.6 * magnitude, BLOCK_WEIGHT_N, -0.8 * magnitude))
    assert float(opposing['friction_energy_generation_w']) == pytest.approx(0.0)
    aligned = residuals(velocity=velocity, force=(0.6 * magnitude, BLOCK_WEIGHT_N, 0.8 * magnitude))
    assert float(aligned['friction_energy_generation_w']) == pytest.approx(5 * magnitude)


def test_unilateral_contact_rejects_pulling_and_penetrating_loads():
    assert float(residuals(force=(0.0, -3.0, 0.0))['normal_pull_n']) == pytest.approx(3.0)
    separated = residuals(gap=0.02)
    assert float(separated['contact_complementarity_nm']) == pytest.approx(0.02 * BLOCK_WEIGHT_N)
    penetrating = residuals(gap=-0.01)
    assert float(penetrating['penetration_m']) == pytest.approx(0.01)
    assert float(penetrating['contact_complementarity_nm']) == pytest.approx(0.01 * BLOCK_WEIGHT_N)
    # A full weight applied to a point that is 0.5 m above the floor is not contact.
    floating = residuals(gap=0.5, force=(2.0, BLOCK_WEIGHT_N, 0.0))
    assert float(floating['contact_complementarity_nm']) == pytest.approx(0.5 * BLOCK_WEIGHT_N)
    assert float(floating['friction_cone_excess_n']) == pytest.approx(0.0)


def test_residuals_keep_separate_units_and_are_not_a_combined_loss():
    result = residuals(gap=-0.01, velocity=(1.0, 0.0, 0.0), force=(4.0, 5.0, 0.0))
    units = {'penetration_m': 'm', 'normal_pull_n': 'N', 'contact_complementarity_nm': 'N*m',
             'friction_cone_excess_n': 'N', 'kinetic_friction_residual_n': 'N',
             'friction_energy_generation_w': 'W'}
    assert set(result) == {*units, 'kinetic_friction_branch'}
    # Every residual is a separate quantity whose name carries its own unit suffix.
    assert {name.rsplit('_', 1)[-1] for name in units} == {'m', 'n', 'nm', 'w'}
    assert len({float(result[name]) for name in units}) == len(units)


def test_near_zero_speed_zero_load_and_threshold_boundary_stay_finite():
    speed = torch.tensor([0.0, 1e-12, 1e-4, 1.0000001e-4, 1e-3], dtype=DTYPE)
    velocity = torch.zeros(5, 3, dtype=DTYPE)
    velocity[:, 0] = speed
    force = torch.zeros(5, 3, dtype=DTYPE)
    force[:, 1] = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0], dtype=DTYPE)
    result = coulomb_contact_residuals(torch.zeros(5, dtype=DTYPE), velocity, force,
                                       static_friction=MU_STATIC, dynamic_friction=MU_DYNAMIC)
    for name, value in result.items():
        if value.dtype != torch.bool:
            assert torch.isfinite(value).all(), name
    torch.testing.assert_close(result['kinetic_friction_branch'],
                              torch.tensor([False, False, False, True, True]))


def test_zero_load_and_zero_speed_have_finite_gradients():
    gap = torch.zeros(2, dtype=DTYPE, requires_grad=True)
    velocity = torch.zeros(2, 3, dtype=DTYPE, requires_grad=True)
    force = torch.zeros(2, 3, dtype=DTYPE, requires_grad=True)
    result = coulomb_contact_residuals(gap, velocity, force, static_friction=MU_STATIC,
                                       dynamic_friction=MU_DYNAMIC)
    sum(value.sum() for name, value in result.items() if value.dtype != torch.bool).backward()
    assert all(torch.isfinite(value.grad).all() for value in (gap, velocity, force))


def test_coulomb_law_is_invariant_under_rigid_yaw():
    velocity = torch.tensor([[3.0, 0.0, 4.0], [0.0, 0.0, 0.0]], dtype=DTYPE)
    force = torch.tensor([[-2.4, BLOCK_WEIGHT_N, -3.2], [2.0, BLOCK_WEIGHT_N, 1.0]], dtype=DTYPE)
    yaw = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=DTYPE)
    kwargs = dict(static_friction=MU_STATIC, dynamic_friction=MU_DYNAMIC)
    original = coulomb_contact_residuals(torch.zeros(2, dtype=DTYPE), velocity, force, **kwargs)
    rotated = coulomb_contact_residuals(torch.zeros(2, dtype=DTYPE), velocity @ yaw.T, force @ yaw.T, **kwargs)
    for name in original:
        torch.testing.assert_close(original[name], rotated[name])


@pytest.mark.parametrize('mu_static,mu_dynamic', [(-0.1, 0), (0.2, 0.3), (float('nan'), 0.2), (0.6, float('inf'))])
def test_coulomb_law_rejects_invalid_material_parameters(mu_static, mu_dynamic):
    with pytest.raises(ValueError, match='Friction coefficients'):
        coulomb_contact_residuals(torch.tensor(0.0), torch.zeros(3), torch.zeros(3),
                                 static_friction=mu_static, dynamic_friction=mu_dynamic)


def test_contact_law_rejects_incompatible_shapes_and_nonfinite_values():
    kwargs = dict(static_friction=MU_STATIC, dynamic_friction=MU_DYNAMIC)
    with pytest.raises(ValueError, match='Matching'):
        coulomb_contact_residuals(torch.zeros(2), torch.zeros(3), torch.zeros(3), **kwargs)
    with pytest.raises(ValueError, match='finite floating-point'):
        coulomb_contact_residuals(torch.tensor(float('nan')), torch.zeros(3), torch.zeros(3), **kwargs)
    with pytest.raises(ValueError, match='dtype'):
        coulomb_contact_residuals(torch.tensor(0.0, dtype=DTYPE), torch.zeros(3), torch.zeros(3), **kwargs)
    with pytest.raises(ValueError, match='slip-speed'):
        coulomb_contact_residuals(torch.tensor(0.0), torch.zeros(3), torch.zeros(3), slip_speed_mps=0, **kwargs)


def test_airborne_motion_is_swing_never_support_slip():
    contacts = torch.tensor([[False], [False], [True], [True], [False], [False]])
    confidence = torch.ones_like(contacts)
    masks = support_interval_masks(contacts, confidence)
    assert int(masks['sustained_support'].sum()) == 1
    assert int(masks['transition'].sum()) == 2
    assert int(masks['swing'].sum()) == 2
    assert int(masks['unknown'].sum()) == 0
    # The foot flies at 5-7 m/s while airborne and barely moves while supported.
    speed = torch.tensor([[5.0], [4.0], [0.01], [0.02], [7.0]], dtype=DTYPE)
    supported = masked_summary(speed, masks['sustained_support'])
    assert supported['samples'] == 1 and supported['peak'] == pytest.approx(0.01)
    assert masked_summary(speed, masks['swing'])['peak'] == pytest.approx(7.0)
    assert masked_summary(speed, masks['transition'])['peak'] == pytest.approx(4.0)


def test_unknown_intervals_never_enter_support_statistics():
    contacts = torch.tensor([[True], [True], [True], [True]])
    confidence = torch.tensor([[True], [True], [False], [True]])
    masks = support_interval_masks(contacts, confidence)
    assert torch.equal(masks['sustained_support'], torch.tensor([[True], [False], [False]]))
    assert torch.equal(masks['unknown'], torch.tensor([[False], [True], [True]]))
    speed = torch.tensor([[0.01], [9.0], [9.0]], dtype=DTYPE)
    supported = masked_summary(speed, masks['sustained_support'])
    assert supported['samples'] == 1 and supported['peak'] == pytest.approx(0.01)
    assert masked_summary(speed, masks['unknown'])['peak'] == pytest.approx(9.0)


def body_state(contacts, confidence, gap):
    return body_contact_state(torch.tensor(contacts, dtype=torch.bool),
                              torch.tensor(confidence, dtype=torch.bool),
                              torch.tensor(gap, dtype=DTYPE))


def test_body_contact_state_needs_measured_clearance_before_it_calls_flight():
    contacts = [[True, False, False, False],
                [False, False, False, False],
                [False, False, False, False],
                [False, False, False, False]]
    confidence = [[True, True, True, True],
                  [True, True, True, True],
                  [False, True, True, True],
                  [True, True, True, True]]
    gap = [[0.0, 0.0, 0.0, 0.0],
           [0.20, 0.18, 0.22, 0.19],
           [0.20, 0.18, 0.22, 0.19],
           [0.02, 0.02, 0.02, 0.02]]
    state = body_state(contacts, confidence, gap)
    assert torch.equal(state['stable_support'], torch.tensor([True, False, False, False]))
    assert torch.equal(state['airborne'], torch.tensor([False, True, False, False]))
    assert torch.equal(state['near_ground_undecided'], torch.tensor([False, False, False, True]))
    assert torch.equal(state['unknown'], torch.tensor([False, False, True, False]))
    total = sum(state[name].long() for name in state)
    assert int(total.min()) == 1 and int(total.max()) == 1


def test_a_foot_on_the_floor_is_never_promoted_to_flight():
    """Regression: the old rule turned any confident non-support frame into flight."""
    frames = 6
    state = body_state([[False] * 4 for _ in range(frames)], [[True] * 4 for _ in range(frames)],
                       [[0.0] * 4 for _ in range(frames)])
    assert int(state['airborne'].sum()) == 0
    assert int(state['near_ground_undecided'].sum()) == frames


def test_being_near_the_floor_is_undecided_not_contact():
    """Regression: proximity alone must not become weight-bearing contact.

    Four unknown labels with every point 2 cm above the floor used to be reported as
    sliding contact and fed into the force check.
    """
    frames = 12
    state = body_state([[False] * 4 for _ in range(frames)], [[False] * 4 for _ in range(frames)],
                       [[0.02] * 4 for _ in range(frames)])
    assert int(state['near_ground_undecided'].sum()) == frames
    assert int(state['stable_support'].sum()) == 0
    assert int(state['airborne'].sum()) == 0
    assert int(state['unknown'].sum()) == 0
    force = torch.zeros(frames, 3, dtype=DTYPE)
    residual = ground_force_feasibility(force, state, static_friction=0.6)
    assert int(residual["evaluated_contact_frames"].sum()) == 0
    assert int(residual["evaluated_airborne_frames"].sum()) == 0


def test_the_clearance_band_between_contact_and_flight_is_unknown():
    state = body_state([[False] * 4], [[True] * 4], [[0.05] * 4])
    assert int(state['unknown'].sum()) == 1
    assert int(state['airborne'].sum()) == 0 and int(state['near_ground_undecided'].sum()) == 0


def test_contact_clearance_must_stay_below_the_airborne_clearance():
    with pytest.raises(ValueError, match='Clearances'):
        body_contact_state(torch.zeros(2, 4, dtype=torch.bool), torch.ones(2, 4, dtype=torch.bool),
                           torch.zeros(2, 4, dtype=DTYPE), contact_clearance_m=0.08)
    with pytest.raises(ValueError, match='one height per channel'):
        body_contact_state(torch.zeros(2, 4, dtype=torch.bool), torch.ones(2, 4, dtype=torch.bool),
                           torch.zeros(2, 3, dtype=DTYPE))
