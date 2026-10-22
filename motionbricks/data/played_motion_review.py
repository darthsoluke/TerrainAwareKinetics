"""Assemble the existing playback contract for review, without changing future output."""
import torch


PUBLICATION_GEOMETRY = 'observed4-generated24-swept8-v1'


def played_window(history, prediction):
    if (history.ndim != 3 or history.shape[1] != 4 or prediction.ndim != 3
            or prediction.shape[1] < 28 or history.shape[0] != prediction.shape[0]
            or history.shape[2] != prediction.shape[2]):
        raise ValueError('Review requires actual history4and at least24generated future frames')
    return torch.cat((history, prediction[:, 4:28]), 1)
