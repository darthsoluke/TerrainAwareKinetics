"""Device-local teacher sampling for independently owned CUDA replicas."""

from contextlib import contextmanager

import torch


@contextmanager
def isolated_pose_seed(tokens, seed):
    if tokens.device.type != 'cuda' or tokens.device.index is None:
        raise ValueError('Isolated Pose sampling requires an explicit CUDA device')
    device = tokens.device
    previous = torch.cuda.get_rng_state(device)
    # A private generator seeds only this device. torch.manual_seed would also
    # reset the CPU and other replicas while their dropout is executing.
    generator = torch.Generator(device=device).manual_seed(seed)
    try:
        torch.cuda.set_rng_state(generator.get_state(), device)
        yield
    finally:
        torch.cuda.set_rng_state(previous, device)
