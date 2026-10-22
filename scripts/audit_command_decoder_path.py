"""Offline decoder-branch intervention with identical tokens and Root inputs."""

import argparse
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.locomotion_command_adapter import install_locomotion_command
from motionbricks.data.native_mb_observed_history import reencode_played_history
from motionbricks.data.native_mb_vertical_frame import shift_global_motion
from motionbricks.data.terrain_control import SoftCommandState
from motionbricks.data.terrain_decoder_pilot import vertical_origin
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.review_parc_decoder_height_response import capture_decoder_call
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb_control import state_digest


def restore(infer, local, history, observation):
    world=infer.motion_rep.dual_rep.local_to_global(local,True,False,
        lengths=torch.full((len(history),),32,device=history.device,dtype=torch.long))
    offsets,heading,_=infer._extract_initial_root_info(history[...,:5])
    world=infer._reapply_initial_root_info(dict(pred_global_poses=world,
        reference_start_root_global_offsets=offsets,reference_start_root_global_heading=heading))
    return shift_global_motion(world,infer.global_motion_rep,vertical_origin(infer,history,observation))[:,:32]


def motion_metrics(rep, world, history, row, axis):
    actual=world[row:row+1,4:12]
    root=torch.cat((history[row:row+1,-1:,:3],actual[:,:,:3]),1)
    velocity=torch.diff(root[0,:,[0,2]],dim=0)*30
    joints=rep.inverse(actual,False,return_all=False,joint_positions_from='global_rot_data')['posed_joints'][0]
    names=rep.skeleton.bone_order_names
    feet=joints[:,[names.index('LeftFoot'),names.index('RightFoot')]]-actual[0,:,None,:3]
    separation=(feet[:,0,[0,2]]-feet[:,1,[0,2]])@axis
    return dict(forward_speed_mean_mps=float((velocity@axis).mean()),
        forward_step_m=float((velocity@axis).sum()/30),
        feet_foreaft_rms_m=float(separation.square().mean().sqrt()),
        relative_feet_speed_mean_mps=float(torch.diff(feet,dim=0).norm(dim=-1).mean()*30))


@torch.no_grad()
def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(2333)
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    infer,bundle=load_terrain_bundle(Path(saved['contract']['source']),args.device)
    model=JointSDFController(infer,bundle);install_locomotion_command(model)
    model.load_state_dict(saved['model_state'],strict=True);model.requires_grad_(False);model.eval()
    frozen=state_digest(model);rep=infer.global_motion_rep;records=[]
    for i in range(8):
        with np.load(args.recordings/f'idle60-case{i}.npz',allow_pickle=False) as data:
            records.append(torch.from_numpy(data['world'].copy()))
            if i==0:scene={k:torch.from_numpy(data[k].copy()) for k in ('heightfield_y_m','min_point_world_xyz_m','dx_m')}
    recording=torch.stack(records).to(args.device)
    cases=[(y,s) for y in (0.,np.pi/2,np.pi,-np.pi/2) for s in (1.6,5.)]
    axes=recording.new_tensor([[np.sin(y),np.cos(y)] for y,s in cases])
    controls=torch.cat((axes*recording.new_tensor([s for y,s in cases])[:,None],
        recording.new_tensor([[np.cos(y),np.sin(y)] for y,s in cases])),1)
    rows=[]
    for block in (60,64,68,76,84,89):
        begin=block*8
        history=reencode_played_history(recording[:,begin-4:begin],rep)
        intent=SoftCommandState().from_controls(history,controls,8)
        with capture_decoder_call(model.codec) as calls:
            result=model(history,intent,[scene]*8)
        actual=result['world'].clone();call=calls[-1]
        baseline=restore(infer,call['recon_state'],history,result['observation'])
        torch.testing.assert_close(baseline,actual,atol=3e-5,rtol=1e-5)
        handle=model.locomotion_decoder_condition.register_forward_hook(lambda module,inputs,output:inputs[0])
        try: disabled=model.codec.forward_decoder(*call['args'],**call['kwargs'])['recon_state']
        finally:handle.remove()
        counter=restore(infer,disabled,history,result['observation'])
        restored=model.codec.forward_decoder(*call['args'],**call['kwargs'])['recon_state']
        torch.testing.assert_close(restored,call['recon_state'],atol=0,rtol=0)
        rows.append(dict(block=block,seconds_after_forward=(begin-480)/30,
            archived_gpu_replay_max_error=float((actual[:,4:12]-recording[:,begin:begin+8]).abs().max()),
            local_decoder_restore_max_error=float((baseline-actual).abs().max()),
            original=motion_metrics(rep,actual,history,4,axes[4]),
            same_tokens_without_decoder_command=motion_metrics(rep,counter,history,4,axes[4]),
            same_tokens=True,same_root_external_condition=True,same_history=True,
            intervention='Only disable the new decoder-latent command residual, no Root/Pose rerun or future oracle'))
        write_json(args.output/'progress.json',dict(rows=rows))
    if state_digest(model)!=frozen:raise ValueError('Diagnostic mutated model weights')
    write_json(args.output/'complete.json',dict(rows=rows,device=args.device,weights_unchanged=True,
        optimizer_updates=0,deployed=False,scope='Offline single-plan conditional effect, not a fixed rollout or proposed repair'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','recordings','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cpu');main(p.parse_args())
