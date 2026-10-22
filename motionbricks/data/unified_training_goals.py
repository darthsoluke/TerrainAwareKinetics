"""Source-derived terminal controls for native velocity-protocol training only."""

import torch

from .native_mb_online_conditions import velocity_goal


GOAL_SOURCE = 'source-window-terminal-vxz-heading-v1'


def source_terminal_goal(global_root, *, fps=30):
    """Use interval31->32 and heading31 for a32frame primitive, never height.

    These are hindsight trajectory labels, not recorded user commands. Runtime
    continues to receive actual requested velocity and heading from the user.
    """
    if (global_root.ndim != 3 or global_root.shape[1:] != (33, 5)
            or fps != 30 or not torch.isfinite(global_root).all()):
        raise ValueError('Terminal labels require33finite raw Root frames at30Hz')
    heading = global_root[:, 31, 3:5]
    if not torch.allclose(heading.square().sum(-1), torch.ones_like(heading[:, 0]), atol=1e-4, rtol=0):
        raise ValueError('Expected raw unit cos/sin heading, not normalized features')
    velocity = (global_root[:, 32, [0, 2]] - global_root[:, 31, [0, 2]]) * fps
    local_root = torch.cat((velocity.new_zeros(len(velocity), 1), velocity,
                            velocity.new_zeros(len(velocity), 1)), -1)
    return velocity_goal(local_root, global_root[:, 31]).detach().clone()


def validate_source_terminal_goal(global_root, controls):
    expected = source_terminal_goal(global_root)
    if (controls.shape != expected.shape or not torch.isfinite(controls).all()
            or not torch.allclose(controls, expected, atol=1e-5, rtol=1e-5)):
        raise ValueError('Training velocity/heading condition differs from this source window terminal goal')
    return expected
