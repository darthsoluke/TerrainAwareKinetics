"""Train every real primitive junction in a differentiable full passage."""

import torch


def all_replan_boundaries(predicted_joints, source_joints, *, publish_frames=8, fps=30):
    if (predicted_joints.ndim != 4 or predicted_joints.shape[-1] != 3
            or source_joints.shape[0] != predicted_joints.shape[0]
            or source_joints.shape[2:] != predicted_joints.shape[2:]
            or source_joints.shape[1] < predicted_joints.shape[1]
            or predicted_joints.shape[1] < 9 or publish_frames < 1 or fps != 30):
        raise ValueError('Expected complete30Hz joint sequences with4history and4forecast-tail frames')
    starts=torch.arange(4,predicted_joints.shape[1]-4,publish_frames,device=predicted_joints.device)
    pv=predicted_joints[:,starts]-predicted_joints[:,starts-1]
    sv=source_joints[:,starts]-source_joints[:,starts-1]
    return (pv-sv).norm(dim=-1).mean()*fps
