"""Gate the new add-on with actual frozen-model and full flat-rollout equality."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_preserving_terrain import NativePreservingTerrain, NativePreservingBackend, PROFILE
from motionbricks.data.native_mb_online_live import NativeTerrainOnlineBackend
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest


@torch.no_grad()
def rollouts(inference, controller, seed, scene):
    rep = inference.global_motion_rep
    result = {}
    for yaw in (0., np.pi/2, np.pi, -np.pi/2):
        for speed in (1.6, 5.):
            for idle in (0, 60):
                history = rep.change_first_heading(seed.clone(), yaw, False, False)
                backend = NativeTerrainOnlineBackend(inference, history, scene) if controller is None else NativePreservingBackend(controller, history, scene)
                backend.configure_runtime(dict(PROFILE))
                pieces = []
                for block in range(idle+30):
                    velocity = [np.sin(yaw)*speed, np.cos(yaw)*speed] if block>=idle else [0.,0.]
                    value = backend.predict_plan(velocity, yaw)
                    pieces.append(value['planned_motion'].clone())
                    backend.history = value['published_history'].clone()
                result[f'{yaw:.6f}-{speed:g}-{idle}'] = torch.cat(pieces, 1).cpu()
    # A continuous command-change case exercises the old buffer invalidation.
    backend = NativeTerrainOnlineBackend(inference, seed, scene) if controller is None else NativePreservingBackend(controller, seed, scene)
    backend.configure_runtime(dict(PROFILE))
    pieces = []
    for yaw,speed,blocks in [(0.,1.6,6),(0.,0.,6),(np.pi/2,1.6,6),(-np.pi/2,1.6,6),(np.pi,1.6,6),(0.,1.6,6)]:
        for _ in range(blocks):
            value = backend.predict_plan([np.sin(yaw)*speed,np.cos(yaw)*speed], yaw)
            pieces.append(value['planned_motion'].clone());backend.history = value['published_history'].clone()
    result['continuous-wasd-release'] = torch.cat(pieces, 1).cpu()
    return result


def assert_same(before, after):
    if set(before) != set(after):
        raise ValueError('Flat case coverage changed')
    rows = []
    for key in before:
        torch.testing.assert_close(after[key], before[key], atol=0, rtol=0)
        rows.append(dict(case=key,frames=before[key].shape[1],max_feature_error=float((before[key]-after[key]).abs().max())))
    return rows


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(6143)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha256(args.bundle)!='0672abe1edbe47dd2328b3bc6700598932e4228b7c98448ffb7d5a537b592f0d':
        raise ValueError('Approved baseline bundle differs')
    inference,bundle = load_terrain_bundle(args.bundle,args.device)
    keys = list(inference.state_dict())
    frozen = state_digest(inference,keys)
    root_args,pose_args = deepcopy(inference._root_model.args),deepcopy(inference._pose_model.args)
    original_codec = inference._vqvae_pose_model
    with np.load(args.seed_scene,allow_pickle=False) as data:
        seed = torch.from_numpy(data['initial_history'].copy()).to(args.device)
    seed = inference.global_motion_rep.change_first_heading(seed,0.,False,False)
    seed[..., [0,2]] -= seed[:,-1:,[0,2]].clone()
    scene = dict(heightfield_y_m=torch.zeros(201,201),min_point_world_xyz_m=torch.tensor([-50.,0.,-50.]),dx_m=torch.tensor(.5))
    write_json(args.output/'state.json',dict(status='unmodified_baseline'))
    before = rollouts(inference,None,seed,scene)
    torch.save(before,args.output/'flat_reference.pt')
    controller = NativePreservingTerrain(inference,bundle)
    if inference._vqvae_pose_model is not original_codec or inference._root_model.args != root_args or inference._pose_model.args != pose_args:
        raise ValueError('Native codec identity or condition configuration changed')
    if state_digest(inference,keys)!=frozen:
        raise ValueError('Installation changed frozen model tensors')
    initial = {k:v.detach().clone() for k,v in controller.adaptation.state_dict().items()}
    write_json(args.output/'state.json',dict(status='zero_adapter_flat_parity'))
    zero = assert_same(before,rollouts(inference,controller,seed,scene))
    with torch.no_grad():
        for parameter in controller.adaptation.parameters():
            parameter.uniform_(-.15,.15)
    write_json(args.output/'state.json',dict(status='nonzero_adapter_flat_parity'))
    random = assert_same(before,rollouts(inference,controller,seed,scene))
    controller.adaptation.load_state_dict(initial,strict=True)
    write_json(args.output/'flat_gate.json',dict(passed=True,bundle_sha256=sha256(args.bundle),profile=PROFILE,
        unchanged_codec_identity=True,unchanged_native_configuration=True,zero=zero,nonzero=random,
        optimizer_updates_before_gate=0,frozen_hash=frozen))
    cases = torch.load(args.cases,map_location='cpu',weights_only=False)
    optimizer = torch.optim.Adam(controller.adaptation.parameters(),lr=5e-5)
    gradients=[]
    for case in [c for c in cases if c['pilot_split']=='train'][:2]:
        source = case['world'][None,24:57].to(args.device)
        control = case['exact_constant_control'].to(args.device)
        with torch.no_grad():
            controller.enabled=False
            native_backend=NativeTerrainOnlineBackend(inference,source[:,:4],case['scene'])
            native_backend.configure_runtime(dict(PROFILE))
            base=native_backend.predict_plan(control[0,:2],torch.atan2(control[0,3],control[0,2]))['planned_motion'].clone()
            controller.enabled=True
            observed=controller(source[:,:4],control,case['scene'])
        if not observed['terrain_active']:
            raise ValueError('The real stair window did not activate terrain conditioning')
        # The initial terrain model should also reproduce the old controller.
        if not gradients:
            torch.testing.assert_close(base,observed['world'][:,4:12],atol=3e-5,rtol=1e-5)
        optimizer.zero_grad(set_to_none=True)
        result=controller(source[:,:4],control,case['scene'],training=True,straight_through=True)
        torch.testing.assert_close(result['world'].detach(),observed['world'],atol=3e-5,rtol=1e-5)
        loss=controller.native_losses(result,source,source[:,:4])['native_total']
        loss.backward()
        norms={name:sum(float(p.grad.square().sum()) for p in module.parameters() if p.grad is not None)**.5
               for name,module in controller.adaptation.items()}
        if any(p.grad is not None for p in inference.parameters()):
            raise ValueError('A preserved native parameter acquired gradients')
        for name in ('goal','root','pose','decoder'):
            if not np.isfinite(norms[name]) or norms[name]<=0:
                raise ValueError('Missing/invalid adaptation gradient: '+name)
        norm=torch.nn.utils.clip_grad_norm_(controller.adaptation.parameters(),.5)
        if not torch.isfinite(norm):raise ValueError('Nonfinite adaptation gradient')
        optimizer.step()
        gradients.append(dict(label=case['label'],loss=float(loss.detach()),groups=norms,
            training_forward_error=float((result['world'].detach()-observed['world']).abs().max())))
    if state_digest(inference,keys)!=frozen:
        raise ValueError('Native tensors changed after diagnostic optimization')
    write_json(args.output/'state.json',dict(status='post_update_flat_parity'))
    after_update=assert_same(before,rollouts(inference,controller,seed,scene))
    controller.adaptation.load_state_dict(initial,strict=True)
    write_json(args.output/'complete.json',dict(flat_gate_passed=True,zero=zero,nonzero=random,
        after_two_diagnostic_updates=after_update,gradients=gradients,native_hash_unchanged=True,
        diagnostic_updates=2,diagnostic_updates_retained=False,device=args.device,quality_approved=False,
        source_future_in_actor=False,postprocessing=None))
    write_json(args.output/'state.json',dict(status='complete',flat_gate_passed=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','cases','seed-scene','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    try:main(args)
    except Exception as error:
        if args.output.exists():write_json(args.output/'state.json',dict(status='failed',error=repr(error)))
        raise
