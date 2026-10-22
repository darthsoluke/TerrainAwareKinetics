"""Real, contiguous BONES episodes and declared weak planar command schedules."""

import torch

from .terrain_control import demonstration_controls

EPISODE_CONTROL_VERSION='local-start-intent-facing-v2'


def episode_controls(world, category, segments, reference=None):
    if world.ndim != 2 or len(world) < (segments-1)*8+33 or segments < 1:
        raise ValueError('Every primitive needs a complete real33-frame target window')
    if category not in ('idle', 'start', 'walk', 'run', 'stop', 'backward', 'lateral', 'turn'):
        raise ValueError('Unknown BONES locomotion category')
    heading = torch.atan2(world[3, 4], world[3, 3])
    # A long source can contain several unrelated turns/restarts. Its net
    # displacement is not the command for this contiguous episode.
    local_reference=world if category=='start' or reference is None else reference
    velocities = torch.cat([demonstration_controls(world[None, k*8:k*8+33], 32,
        category=category, reference=local_reference[None])[:, :2]
        for k in range(segments)])
    moving = torch.nonzero(velocities.norm(dim=-1) > .2).flatten()
    first_bearing = torch.atan2(velocities[moving[0], 0], velocities[moving[0], 1]) if len(moving) else heading
    offset = heading-first_bearing if category!='start' else heading.new_zeros(())
    controls = []
    for velocity in velocities:
        if velocity.norm() > .2:
            heading = torch.atan2(velocity[0], velocity[1])+offset
        controls.append(torch.cat((velocity, heading.cos()[None], heading.sin()[None])))
    return torch.stack(controls).detach()


def rotate_controls(controls, yaw):
    if controls.ndim != 2 or controls.shape[1] != 4:
        raise ValueError('Expected planar velocity and facing only')
    angle = controls.new_tensor(yaw)
    c, s = angle.cos(), angle.sin()
    result = controls.clone()
    result[:, 0] = c*controls[:, 0]+s*controls[:, 1]
    result[:, 1] = -s*controls[:, 0]+c*controls[:, 1]
    heading = torch.atan2(controls[:, 3], controls[:, 2])+angle
    result[:, 2], result[:, 3] = heading.cos(), heading.sin()
    return result
