"""Matched continuation testing supervision at every published primitive boundary."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from motionbricks.data.joint_sdf_controller import SDF_VERSION
from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.parc_stair_passages import validate_pure_cases
from motionbricks.data.replan_boundary_objective import all_replan_boundaries
from motionbricks.data.published_root_objective import published_root_reconstruction
from motionbricks.data.pose_support_gradient import add_pose_support_gradient
from motionbricks.data.source_sole_position_objective import source_sole_position_loss
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.pilot_joint_sdf_contact_preserve import load_model
from scripts.pilot_joint_sdf_reciprocal_source import configure_trainable, frozen_state
from scripts.pilot_pure_parc_stairs import episode_loss, review
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


class PassageGeometry(JointGeometryObjective):
    def __init__(self, rep, arm):
        super().__init__(rep, 'cuda:0')
        self.arm=arm

    def prepared_losses(self, prediction, history, prepared, *, offset, diagnostics=True):
        result=super().prepared_losses(prediction,history,prepared,offset=offset,diagnostics=diagnostics)
        if self.arm!='control':
            joints,_=self.geometry(prediction)
            result['boundary_mps']=all_replan_boundaries(joints,prepared['joints'][:,offset:])
        self.last_episode=prediction
        self.last_source_root=prepared['joints'][:,offset:,self.rep.skeleton.root_idx]
        self.last_target=prepared
        self.last_physical=result
        return result


def continuity(directory):
    rows=[]
    for path in sorted(directory.glob('*-speed*-*f.npz')):
        with np.load(path,allow_pickle=False) as recording:
            roots=recording['world'][:,:3]
        acceleration=np.linalg.norm(np.diff(roots,n=2,axis=0)*30,axis=-1)
        boundary=np.arange(2,len(roots))%8==0
        rows.append(dict(file=path.name,frames=len(roots),
            boundary_delta_v_mean=float(acceleration[boundary].mean()) if boundary.any() else None,
            interior_delta_v_mean=float(acceleration[~boundary].mean()),
            delta_v_max=float(acceleration.max()),delta_v_p95=float(np.quantile(acceleration,.95))))
    write_json(directory/'continuity.json',dict(rows=rows,motion_modified=False))
    return rows


def main(args):
    if not 1<=args.updates<=12:raise ValueError('Bounded12-episode review probe')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if sha256(args.checkpoint)!='ae558ee88542191b6cbe72659bd96f1482711ea2d8beec6c1641a0ae5b814934':
        raise ValueError('Expected newly approved pure PARC feedback checkpoint')
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    model,_=load_model(saved);configure_trainable(model,'coordinated')
    cases=torch.load(args.cases,map_location='cpu',weights_only=False);validate_pure_cases(cases)
    training=[c for c in cases if c['pilot_split']=='train']
    for prior in torch.load(args.regression_cases,map_location='cpu',weights_only=False):
        yaw=torch.tensor(prior['command_heading'])
        controls=torch.cat((torch.tensor(prior['command_velocity']),yaw.cos()[None],yaw.sin()[None]))[None]
        cases.append(dict(prior,label='approved-'+prior['entry'],pilot_split='prior_approved_regression',exact_constant_control=controls))
    geometry=PassageGeometry(model.inference.global_motion_rep,args.arm)
    with torch.no_grad():prepared=[tuple(v[0] for v in geometry.geometry(c['world'][None].cuda())) for c in training]
    fields=[HeightfieldPrismSDF(c['scene']) for c in training]
    frozen=frozen_state(model,'coordinated')
    contract=dict(arm=args.arm,parent_sha256=sha256(args.checkpoint),updates=args.updates,
        original_coefficients=saved['coefficients'],optimizer='fresh Adam5e-5,clip0.5',
        only_change='boundary_mps averaged over every8-frame replan' if args.arm=='all_boundaries' else 'archived episode-first boundary only',
        source_cases_sha256=sha256(args.cases),generated_history='complete differentiable actual rollout, no resets',
        future_truth_in_actor=False,postprocessing=None,quality_approved=False)
    write_json(args.output/'contract.json',contract)
    if args.arm in ('all_boundaries_root','pose_support'):
        contract['only_change']='every replan boundary plus native-scale decoded Root reconstruction on the complete published trajectory'
        contract['published_root_coefficient']=float(model.inference._root_model.args['local_root_loss_coeff'])
        write_json(args.output/'contract.json',contract)
    if args.arm=='pose_support':
        contract.update(only_change='relative to all_boundaries_root: support correction gradients only to Pose',
            support_scales=dict(position_m=.02,velocity_mps=.1),
            clipping_note='Existing global norm0.5 retained; stronger Pose gradients can indirectly change shared clipping scale',
            root_decoder_direct_gradient_correction=False)
        write_json(args.output/'contract.json',contract)
    write_json(args.output/'state.json',dict(status='baseline'))
    baseline=review(model,cases,geometry,args.output/'baseline');before_continuity=continuity(args.output/'baseline')
    torch.manual_seed(2333)
    parameters=[p for p in model.parameters() if p.requires_grad]
    pose_parameters=tuple(p for p in model.pose.parameters() if p.requires_grad)
    other_parameters=tuple(p for p in parameters if id(p) not in {id(v) for v in pose_parameters})
    if {id(p) for p in pose_parameters}.intersection(id(p) for p in model.root.parameters()):
        raise ValueError('Pose and Root share parameters')
    if {id(p) for p in pose_parameters}.intersection(id(p) for p in model.codec.decoder.parameters()):
        raise ValueError('Pose and decoder share parameters')
    optimizer=torch.optim.Adam(parameters,lr=5e-5)
    exposure=Counter();logs=[];began=time.perf_counter()
    write_json(args.output/'state.json',dict(status='training'))
    for step in range(args.updates):
        optimizer.zero_grad(set_to_none=True);details=[]
        for index,case in enumerate(training):
            loss,detail=episode_loss(model,case,geometry,prepared[index],fields[index],saved['coefficients'],step)
            if args.arm in ('all_boundaries_root','pose_support'):
                local=model.inference.local_motion_rep
                root_loss=published_root_reconstruction(geometry.last_episode,geometry.last_source_root,
                    local.stats.std[local.indices['local_root_vel']],local.stats.std[local.indices['global_root_y']])
                loss=loss+contract['published_root_coefficient']*root_loss
                detail.update(published_root_loss=float(root_loss.detach()),loss=float(loss.detach()))
            if not torch.isfinite(loss):raise ValueError('Nonfinite passage objective')
            correction=None
            if args.arm=='pose_support':
                target=geometry.last_target;physical=geometry.last_physical
                _,points=geometry.geometry(geometry.last_episode)
                end=geometry.last_episode.shape[1]-4
                xyz,active=source_sole_position_loss(points[:,4:end,:geometry.sole_count],
                    target['points'][:,4:end,:geometry.sole_count],target['heights'])
                coeff=saved['coefficients']
                original=coeff['fk_m']*xyz+coeff['support_gap_m']*physical['support_gap_m']+coeff['source_support_velocity_mps']*physical['source_support_velocity_mps']
                normalized=coeff['fk_m']*xyz/.02+coeff['support_gap_m']*physical['support_gap_m']/.02+coeff['source_support_velocity_mps']*physical['source_support_velocity_mps']/.1
                correction=(normalized-original)/len(training)
                detail.update(support_vertex_frames=int(active.sum()),original_support=float(original.detach()),normalized_support=float(normalized.detach()))
            (loss/len(training)).backward(retain_graph=correction is not None)
            if correction is not None:
                audit=step==0 and index==0
                before=[None if p.grad is None else p.grad.clone() for p in other_parameters] if audit else None
                detail['pose_support_extra_gradient_norm']=float(add_pose_support_gradient(correction,pose_parameters))
                if audit:
                    for p,prior in zip(other_parameters,before):
                        if prior is None:
                            if p.grad is not None:raise ValueError('Correction changed a non-Pose gradient')
                        else:torch.testing.assert_close(p.grad,prior,atol=0,rtol=0)
                    write_json(args.output/'gradient_isolation.json',dict(non_pose_gradient_error=0.,actual_model_checked=True,forward_output_modified=False))
            details.append(detail)
            exposure[case['label']]+=case['training_segments']
            del loss
            geometry.last_episode=None;geometry.last_source_root=None
            geometry.last_target=None;geometry.last_physical=None
        if any(p.grad is not None for name in frozen for p in model.get_submodule(name).parameters()):
            raise ValueError('Frozen module has gradients')
        norm=torch.nn.utils.clip_grad_norm_(parameters,.5)
        if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
        optimizer.step()
        if frozen_state(model,'coordinated')!=frozen:raise ValueError('Frozen tensor changed')
        row=dict(update=step+1,cases=details,gradient_norm=float(norm),elapsed_seconds=time.perf_counter()-began)
        logs.append(row);write_json(args.output/'progress.json',row);print(json.dumps(row),flush=True)
        stage=dict(schema='motionbricks.training_stage.v1',name='replan_boundary_'+args.arm,
            optimizer_updates=step+1,actor_primitives=sum(exposure.values()),source_autoencoder_examples=4*(step+1),
            inherited_optimizer_updates=saved['step'])
        torch.save(dict(schema=SDF_VERSION,model_state=model.state_dict(),optimizer=optimizer.state_dict(),step=step+1,
            contract=dict(saved['contract'],stair_training_stage=stage,replan_boundary_pilot=contract),
            coefficients=saved['coefficients'],exposure=dict(exposure),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state()),args.output/'weights.pt')
    model.requires_grad_(False)
    write_json(args.output/'state.json',dict(status='review'))
    final=review(model,cases,geometry,args.output/'final');after_continuity=continuity(args.output/'final')
    write_json(args.output/'complete.json',dict(baseline=baseline,final=final,logs=logs,
        before_continuity=before_continuity,after_continuity=after_continuity,
        actual_updates=args.updates,actor_primitives=sum(exposure.values()),source_ae_examples=4*args.updates,
        frozen_hashes_match=True,parent_sha256=sha256(args.checkpoint),quality_approved=False))
    write_json(args.output/'state.json',dict(status='complete',quality_approved=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','cases','regression-cases','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',choices=('control','all_boundaries','all_boundaries_root','pose_support'),required=True)
    parser.add_argument('--updates',type=int,default=8)
    args=parser.parse_args()
    try:main(args)
    except Exception as error:
        if args.output.exists():write_json(args.output/'state.json',dict(status='failed',error=str(error)))
        raise
