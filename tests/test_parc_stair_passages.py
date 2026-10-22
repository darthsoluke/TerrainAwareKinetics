import pytest
import torch

from motionbricks.data.parc_stair_passages import regular_passages, validate_pure_cases


def scene_and_motion(entry):
    height = torch.arange(7).clamp_max(5) * .15
    if entry == 'down':
        height = .75 - height
    scene = dict(heightfield_y_m=height[None].expand(5, -1).clone(),
                 min_point_world_xyz_m=torch.tensor([0., 0., -1.]), dx_m=torch.tensor(.5))
    world = torch.zeros(101, 5)
    world[:, 0] = torch.linspace(.02, 3.22, 101)
    cell = (world[:, 0] / .5 + .5).floor().long().clamp_max(6)
    world[:, 1] = height[cell] + 1.
    world[:, 3] = 1.
    return scene, world


@pytest.mark.parametrize('entry', ['up', 'down'])
def test_factual_passage_has_complete_windows_and_goal(entry):
    scene, world = scene_and_motion(entry)
    # Move the first riser later to provide a real approach interval.
    prefix = world[:1].expand(20, -1)
    source = torch.cat((prefix, world, world[-1:].expand(40, -1)))
    result = regular_passages(source, scene, entry)
    assert result
    part = result[0]
    assert (part['end'] - part['start'] - 33) % 8 == 0
    assert part['end'] <= len(source)
    assert part['end'] - 22 >= part['goal']
    assert not regular_passages(source, scene, 'down' if entry == 'up' else 'up')


def test_holdout_identity_and_no_bones():
    cases = [dict(row=dict(origin='parc_generated', source_id=str(i)), entry=entry,
                  pilot_split=split, world=torch.zeros(89, 5), training_segments=8,
                  evaluation_goal_source_frame=60, start_frame=0)
             for i, (split, entry) in enumerate([('train', 'up'), ('train', 'up'),
                ('train', 'down'), ('train', 'down'), ('heldout', 'up'), ('heldout', 'down')])]
    validate_pure_cases(cases)
    cases[-1]['row']['source_id'] = '0'
    with pytest.raises(ValueError, match='leaks'):
        validate_pure_cases(cases)
    cases[-1]['row'] = dict(origin='bones_seed', source_id='5')
    with pytest.raises(ValueError, match='excludes BONES'):
        validate_pure_cases(cases)


def test_taller_stairs_are_selected_without_changing_source():
    scene, world = scene_and_motion('up')
    scene['heightfield_y_m'] *= 2
    world[:, 1] = (world[:, 1] - 1) * 2 + 1
    source = torch.cat((world[:1].expand(20, -1), world, world[-1:].expand(40, -1)))
    original = source.clone()
    assert regular_passages(source, scene, 'up')
    torch.testing.assert_close(source, original, atol=0, rtol=0)
