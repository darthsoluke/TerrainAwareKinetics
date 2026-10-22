"""Replay causal commands on genuine source histories for offline supervision."""

from .terrain_control import SoftCommandState


def source_episode_task(source, controls, block, tokens=8):
    if source.ndim != 3 or controls.shape != (source.shape[0], controls.shape[1], 4):
        raise ValueError('Expected batched source motion and command sequence')
    if not 0 <= block < controls.shape[1] or source.shape[1] < block*8+4:
        raise ValueError('Source history must contain the requested real block')
    state = SoftCommandState()
    # A stopped task retains its earlier release position. Reinitializing the
    # driver at the selected window would silently move this training target.
    for index in range(block+1):
        task = state.from_controls(source[:, index*8:index*8+4], controls[:, index], tokens)
    return task
