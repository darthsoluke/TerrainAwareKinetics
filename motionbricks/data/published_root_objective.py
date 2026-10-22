"""Native-scale Root targets on the actual published trajectory, loss only."""

import torch
from torch.nn import functional as F


def published_root_reconstruction(prediction,source_root,velocity_std,height_std,*,fps=30):
    if prediction.ndim!=3 or source_root.ndim!=3 or prediction.shape[0]!=source_root.shape[0]:
        raise ValueError('Expected batched complete Root trajectories')
    end=prediction.shape[1]-4
    if end<=4 or source_root.shape[1]<end or fps!=30:raise ValueError('Missing actual published frames')
    target=source_root.detach()
    scale=torch.as_tensor(velocity_std,device=prediction.device,dtype=prediction.dtype).reshape(1,1,2)
    height=torch.as_tensor(height_std,device=prediction.device,dtype=prediction.dtype)
    if not torch.isfinite(scale).all() or not torch.isfinite(height).all() or (scale<=0).any() or (height<=0).any():
        raise ValueError('Expected positive native Root scales')
    pv=(prediction[:,4:end,[0,2]]-prediction[:,3:end-1,[0,2]])*fps
    sv=(target[:,4:end,[0,2]]-target[:,3:end-1,[0,2]])*fps
    values=torch.cat(((pv-sv)/scale,((prediction[:,4:end,1]-target[:,4:end,1])/height)[...,None]),-1)
    return F.smooth_l1_loss(values,torch.zeros_like(values))
