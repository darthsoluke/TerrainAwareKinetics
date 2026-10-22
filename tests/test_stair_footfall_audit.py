from scripts.audit_joint_sdf_stair_footfalls import longest_regular_ascent, reciprocal_failures, true_runs


def event(frame, side, level):
    return dict(start_frame=frame, side=side, surface_height_m=level)


def test_reciprocal_ascent_allows_top_platform_catch_up():
    levels = [0., .15, .30, .45]
    rows = [event(0,'L',.15), event(10,'R',.30), event(20,'L',.45), event(30,'R',.45)]
    assert reciprocal_failures(rows, levels) == []


def test_same_tread_inside_stair_flight_is_rejected():
    rows = [event(0,'L',.15), event(10,'R',.15)]
    failures = reciprocal_failures(rows, [0., .15, .30, .45])
    assert len(failures) == 1
    assert failures[0]['current']['flight_level'] == 1


def test_small_contact_fragments_are_not_certified_landings():
    assert true_runs([False, True, True, False, True, True, True, False]) == [(4,7)]


def test_regular_stair_flight_excludes_different_height_platform_steps():
    levels = longest_regular_ascent([-.4, -.4, 0., 0., .15, .15, .30, .45, .45, .65])
    assert levels == [0., .15, .30, .45]


def test_vertical_support_does_not_choose_next_foothold_or_constrain_sliding():
    import torch
    from motionbricks.data.terrain_skin_tail_loss import teacher_support_gap

    points = torch.tensor([[[[.1, .05, .1]], [[.2, .05, .1]], [[.3, .05, .1]]]], requires_grad=True)
    mask = torch.ones(points.shape[:-1], dtype=torch.bool)
    scene = dict(heightfield_y_m=torch.zeros(3, 3), min_point_world_xyz_m=torch.zeros(3), dx_m=1.)
    loss, _ = teacher_support_gap(points, mask, [scene])
    gradient = torch.autograd.grad(loss, points)[0]
    assert torch.count_nonzero(gradient[..., [0, 2]]) == 0
    assert torch.count_nonzero(gradient[..., 1]) == 3
    shifted = points.detach().clone()
    shifted[..., 0] = .1
    stationary_loss, _ = teacher_support_gap(shifted, mask, [scene])
    torch.testing.assert_close(loss, stationary_loss, atol=0, rtol=0)
