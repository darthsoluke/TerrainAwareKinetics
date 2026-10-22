"""Check terrain-to-flat training credit without altering the flat actor output."""

import argparse
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_preserving_terrain import NativePreservingTerrain
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb_control import state_digest


def main(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inference,bundle=load_terrain_bundle(args.bundle,'cpu')
    keys=list(inference.state_dict());before=state_digest(inference,keys)
    controller=NativePreservingTerrain(inference,bundle)
    with np.load(args.seed_scene,allow_pickle=False) as archive:
        history=torch.from_numpy(archive['initial_history'].copy()).requires_grad_(True)
    scene=dict(heightfield_y_m=torch.zeros(201,201),min_point_world_xyz_m=torch.tensor([-50.,0.,-50.]),dx_m=torch.tensor(.5))
    control=history.new_tensor([[0.,1.6,1.,0.]])
    with torch.no_grad():expected=controller(history.detach(),control,scene)['world'].clone()
    result=controller(history,control,scene,training=True,straight_through=True)
    if result['terrain_active']:raise ValueError('Flat terrain incorrectly activates residuals')
    torch.testing.assert_close(result['world'].detach(),expected,atol=3e-5,rtol=1e-5)
    gradient=torch.autograd.grad(result['world'][:,4:28].square().mean(),history)[0]
    if not torch.isfinite(gradient).all() or not gradient.norm()>0:
        raise ValueError('Flat continuation lost factual-history gradients')
    if any(p.grad is not None for p in inference.parameters()) or state_digest(inference,keys)!=before:
        raise ValueError('Native model was modified')
    write_json(args.output,dict(forward_max_error=float((result['world'].detach()-expected).abs().max()),
        history_gradient_norm=float(gradient.norm()),terrain_active=False,frozen_hash_unchanged=True,
        optimizer_updates=0,device='cpu',quality_approved=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','seed-scene','output'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
