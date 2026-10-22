import math

import pytest
import torch

from motionbricks.data.contact_forces import (AIRBORNE_FORCE_TOLERANCE_MPS2, CONTACT_STATES,
                                              center_of_mass_acceleration, ground_force_feasibility,
                                              phase_interior_mask, required_ground_force_per_mass,
                                              state_codes, violation_counts)

FPS = 60.0
FRAMES = 41
GRAVITY = 9.81
DTYPE = torch.float64


def trajectory(acceleration, *, velocity=(0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0),
               fps=FPS, frames=FRAMES):
    """Analytic centre-of-mass path with a constant acceleration, sampled in seconds."""
    time = torch.arange(frames, dtype=DTYPE)[:, None] / fps
    accel = torch.as_tensor(acceleration, dtype=DTYPE)
    speed = torch.as_tensor(velocity, dtype=DTYPE)
    start = torch.as_tensor(position, dtype=DTYPE)
    return start + speed * time + 0.5 * accel * time ** 2


def state(frames=FRAMES, kind="stable_support"):
    """A whole-body contact state that puts every frame in one class."""
    masks = {name: torch.zeros(frames, dtype=torch.bool) for name in CONTACT_STATES}
    masks[kind][:] = True
    return masks


def checks(acceleration, *, kind="stable_support", static_friction=0.6, velocity=(0.0, 0.0, 0.0), **kwargs):
    com = trajectory(acceleration, velocity=velocity, **kwargs)
    demand = required_ground_force_per_mass(com, kwargs.get("fps", FPS), window_frames=5, polyorder=2)
    residual = ground_force_feasibility(demand["required_force_per_mass_mps2"], state(com.shape[-2], kind),
                                        static_friction=static_friction)
    return demand, residual


def test_standing_body_needs_only_its_weight():
    demand, residual = checks((0.0, 0.0, 0.0))
    force = demand["required_force_per_mass_mps2"]
    assert float(force[..., 0].abs().max()) < 1e-9
    assert float(force[..., 2].abs().max()) < 1e-9
    torch.testing.assert_close(force[..., 1], torch.full_like(force[..., 1], GRAVITY), atol=1e-8, rtol=0)
    for name in ("ground_pull_mps2", "friction_capacity_excess_mps2", "airborne_force_magnitude_mps2"):
        assert float(residual[name].max()) < 1e-8


@pytest.mark.parametrize("acceleration_x", [2.0, -2.0, 0.0])
def test_acceleration_and_braking_demand_the_matching_inertial_force(acceleration_x):
    demand, _ = checks((acceleration_x, 0.0, 0.0))
    force = demand["required_force_per_mass_mps2"]
    torch.testing.assert_close(force[..., 0], torch.full_like(force[..., 0], acceleration_x),
                               atol=1e-8, rtol=0)


def test_friction_capacity_is_exceeded_only_above_mu_times_the_normal_force():
    limit = 0.6 * GRAVITY
    inside, inside_residual = checks((limit - 1e-6, 0.0, 0.0), static_friction=0.6)
    assert float(inside_residual["friction_capacity_excess_mps2"].max()) < 1e-6
    outside, outside_residual = checks((8.0, 0.0, 0.0), static_friction=0.6)
    torch.testing.assert_close(outside_residual["friction_capacity_excess_mps2"],
                               torch.full_like(outside["required_force_per_mass_mps2"][..., 0], 8.0 - limit),
                               atol=1e-6, rtol=0)
    frictionless, frictionless_residual = checks((0.5, 0.0, 0.0), static_friction=0.0)
    torch.testing.assert_close(frictionless_residual["friction_capacity_excess_mps2"],
                               torch.full_like(frictionless["required_force_per_mass_mps2"][..., 0], 0.5),
                               atol=1e-9, rtol=0)


def test_ballistic_flight_needs_no_ground_force_at_all():
    demand, residual = checks((0.0, -GRAVITY, 0.0), kind="airborne", velocity=(1.5, -2.0, 0.5))
    force = demand["required_force_per_mass_mps2"]
    assert float(force.norm(dim=-1).max()) < 1e-6
    assert float(residual["airborne_force_magnitude_mps2"].max()) < 1e-6


def test_hovering_in_air_is_not_explainable_by_ground_contact():
    demand, residual = checks((0.0, 0.0, 0.0), kind="airborne")
    torch.testing.assert_close(residual["airborne_force_magnitude_mps2"],
                               torch.full_like(demand["required_force_per_mass_mps2"][..., 0], GRAVITY),
                               atol=1e-9, rtol=0)


def test_an_undecided_near_ground_frame_carries_no_ground_reaction():
    force = torch.zeros(4, 3, dtype=DTYPE)
    force[:, 1] = 4.0
    masks = state(4, kind="near_ground_undecided")
    residual = ground_force_feasibility(force, masks, static_friction=0.6)
    assert int(residual["evaluated_contact_frames"].sum()) == 0
    assert float(residual["ground_pull_mps2"].max()) == 0.0
    assert float(residual["friction_capacity_excess_mps2"].max()) == 0.0


def test_ground_must_not_pull_the_body_down():
    demand, residual = checks((0.0, -20.0, 0.0))
    torch.testing.assert_close(residual["ground_pull_mps2"],
                               torch.full_like(demand["required_force_per_mass_mps2"][..., 0], 20.0 - GRAVITY),
                               atol=1e-8, rtol=0)


def test_unknown_frames_are_never_evaluated_and_never_called_airborne():
    demand, residual = checks((5.0, -30.0, 0.0), kind="unknown")
    assert int(residual["evaluated_contact_frames"].sum()) == 0
    assert int(residual["evaluated_airborne_frames"].sum()) == 0
    assert float(residual["ground_pull_mps2"].max()) == 0.0
    assert float(residual["airborne_force_magnitude_mps2"].max()) == 0.0


def test_contact_state_masks_must_partition_every_frame():
    demand, _ = checks((0.0, 0.0, 0.0))
    force = demand["required_force_per_mass_mps2"]
    overlapping = state(FRAMES)
    overlapping["airborne"][:] = True
    with pytest.raises(ValueError, match="partition"):
        ground_force_feasibility(force, overlapping, static_friction=0.6)
    with pytest.raises(ValueError, match="boolean"):
        ground_force_feasibility(force, {"stable_support": torch.ones(FRAMES),
                                         "near_ground_undecided": torch.zeros(FRAMES, dtype=torch.bool),
                                         "airborne": torch.zeros(FRAMES, dtype=torch.bool),
                                         "unknown": torch.zeros(FRAMES, dtype=torch.bool)},
                                 static_friction=0.6)
    with pytest.raises(ValueError, match="match the force frames"):
        ground_force_feasibility(force, state(FRAMES - 1), static_friction=0.6)


def test_the_filter_is_exact_for_polynomial_motion_and_flags_its_edges():
    acceleration = torch.tensor([2.0, -3.0, 1.5], dtype=DTYPE)
    com = trajectory(acceleration)
    for window, order in ((5, 2), (7, 3), (9, 2)):
        result = center_of_mass_acceleration(com, FPS, window_frames=window, polyorder=order)
        error = (result["acceleration_mps2"] - acceleration).abs().max()
        assert float(error) < 1e-8
        assert int(result["acceleration_edge"].sum()) == 2 * (window // 2)
        assert result["acceleration_mps2"].shape == com.shape


def test_acceleration_stays_aligned_with_the_frame_it_describes():
    com = trajectory((0.0, 0.0, 0.0), velocity=(1.0, 0.0, 0.0))
    # The horizontal speed drops at frame 20, so the acceleration belongs at frame 20
    # and nowhere else once the window has passed.
    com[20:] = com[20] + (torch.arange(FRAMES - 20, dtype=DTYPE)[:, None] / FPS) * torch.tensor([0.2, 0.0, 0.0], dtype=DTYPE)
    result = center_of_mass_acceleration(com, FPS, window_frames=5, polyorder=2)
    constant = result["acceleration_mps2"][:, 0].abs()
    assert float(constant[:18].max()) < 1e-9
    assert float(constant[23:].max()) < 1e-9


def test_resultant_force_check_is_invariant_under_rigid_yaw():
    demand, residual = checks((0.5, 3.0, -1.0), static_friction=0.3)
    force = demand["required_force_per_mass_mps2"]
    angle = torch.tensor(0.7, dtype=DTYPE)
    yaw = torch.tensor([[torch.cos(angle), 0.0, torch.sin(angle)],
                        [0.0, 1.0, 0.0],
                        [-torch.sin(angle), 0.0, torch.cos(angle)]], dtype=DTYPE)
    rotated = ground_force_feasibility(force @ yaw.T, state(FRAMES), static_friction=0.3)
    for name in ("ground_pull_mps2", "friction_capacity_excess_mps2", "airborne_force_magnitude_mps2"):
            torch.testing.assert_close(residual[name], rotated[name], atol=1e-9, rtol=0)


@pytest.mark.parametrize("window,polyorder", [(4, 2), (5, 5), (5, 1), (1, 2)])
def test_invalid_filter_settings_are_rejected(window, polyorder):
    with pytest.raises(ValueError):
        center_of_mass_acceleration(trajectory((0.0, 0.0, 0.0)), FPS, window_frames=window, polyorder=polyorder)


def test_invalid_gravity_and_window_length_are_rejected():
    com = trajectory((0.0, 0.0, 0.0))
    for gravity in (0.0, -9.81, float("nan")):
        with pytest.raises(ValueError, match="gravity"):
            required_ground_force_per_mass(com, FPS, gravity_mps2=gravity)
    with pytest.raises(ValueError, match="longer than the clip"):
        center_of_mass_acceleration(com[:3], FPS, window_frames=5, polyorder=2)
    with pytest.raises(ValueError, match="finite positive FPS"):
        center_of_mass_acceleration(com, 0.0)
    with pytest.raises(ValueError, match="xyz"):
        center_of_mass_acceleration(torch.zeros(5, 2, dtype=DTYPE), FPS)


def test_the_airborne_tolerance_covers_the_float32_noise_floor():
    """Regression: a bare zero threshold flagged an exactly ballistic float32 path."""
    gravity = torch.tensor([0.0, -GRAVITY, 0.0], dtype=torch.float32)
    time = torch.arange(FRAMES, dtype=torch.float32)[:, None] / FPS
    com = 0.5 * gravity * time ** 2 + torch.tensor([1.0, 2.0, 0.0], dtype=torch.float32)
    result = required_ground_force_per_mass(com, FPS, window_frames=5, polyorder=2)
    interior = ~result["acceleration_edge"]
    residual = result["required_force_per_mass_mps2"].norm(dim=-1)[interior]
    assert int(interior.sum()) > 0
    assert float(residual.max()) < AIRBORNE_FORCE_TOLERANCE_MPS2


def test_phase_interior_excludes_a_window_that_spans_a_state_change():
    codes = torch.tensor([0] * 6 + [1] * 6)
    assert torch.equal(phase_interior_mask(codes, 5),
                       torch.tensor([False, False, True, True, False, False,
                                     False, False, True, True, False, False]))


def test_state_codes_follow_the_declared_state_order():
    masks = state(6, kind="near_ground_undecided")
    codes = state_codes(masks)
    assert torch.equal(codes, torch.full((6,), CONTACT_STATES.index("near_ground_undecided")))
    with pytest.raises(ValueError, match="partition"):
        broken = state(6)
        broken["airborne"][:] = True
        state_codes(broken)


def test_violation_counts_use_the_stated_airborne_tolerance():
    force = torch.zeros(3, 3, dtype=DTYPE)
    force[1, 1] = 5e-4
    force[2, 1] = 2e-3
    masks = {"stable_support": torch.zeros(3, dtype=torch.bool),
             "near_ground_undecided": torch.zeros(3, dtype=torch.bool),
             "airborne": torch.ones(3, dtype=torch.bool),
             "unknown": torch.zeros(3, dtype=torch.bool)}
    residual = ground_force_feasibility(force, masks, static_friction=0.6)
    assert violation_counts(residual)["airborne_force_frames"] == 1
    assert violation_counts(residual, airborne_tolerance_mps2=1e-9)["airborne_force_frames"] == 2


def test_the_evaluation_mask_excludes_frames_from_the_residuals():
    force = torch.zeros(4, 3, dtype=DTYPE)
    force[:, 1] = -3.0
    masks = state(4)
    excluded = torch.tensor([True, True, False, False])
    residual = ground_force_feasibility(force, masks, static_friction=0.6, evaluated=excluded)
    assert int(residual["evaluated_contact_frames"].sum()) == 2
    assert float(residual["ground_pull_mps2"].sum()) == pytest.approx(6.0)
    with pytest.raises(ValueError, match="evaluation mask"):
        ground_force_feasibility(force, masks, static_friction=0.6, evaluated=torch.ones(4))
