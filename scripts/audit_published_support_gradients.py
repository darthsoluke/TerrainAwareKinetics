"""Measure support supervision on the same factual-start actor rollout, no updates."""

import argparse
import json
from pathlib import Path

import torch

from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF
from motionbricks.data.published_root_objective import published_root_reconstruction
from motionbricks.data.replan_boundary_objective import all_replan_boundaries
from motionbricks.data.source_sole_position_objective import source_sole_position_loss
from motionbricks.data.terrain_joint_objective import JointGeometryObjective
from scripts.pilot_joint_sdf_contact_preserve import load_model
from scripts.pilot_joint_sdf_reciprocal_source import configure_trainable
from scripts.pilot_pure_parc_stairs import episode_loss
from scripts.train_joint_terrain_v3 import write_json
from scripts.train_native_mb import sha256


class CaptureGeometry(JointGeometryObjective):
    def prepared_losses(self,prediction,history,prepared,*,offset,diagnostics=True):
        values=super().prepared_losses(prediction,history,prepared,offset=offset,diagnostics=diagnostics)
        joints,_=self.geometry(prediction)
        values['boundary_mps']=all_replan_boundaries(joints,prepared['joints'][:,offset:])
        self.prediction,self.target,self.values=prediction,prepared,values
        return values


def norms_and_cosine(losses,parameters):
    gradients=[torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True) for loss in losses]
    squares=[sum((g.square().sum() for g in group if g is not None),losses[0].new_zeros(())) for group in gradients]
    dot=sum((a.mul(b).sum() for a,b in zip(gradients[0],gradients[1]) if a is not None and b is not None),losses[0].new_zeros(()))
    return dict(existing_support_norm=float(squares[0].sqrt()),other_objectives_norm=float(squares[1].sqrt()),
        normalized_support_norm=float(squares[2].sqrt()),existing_support_other_cosine=float(dot/(squares[0]*squares[1]).sqrt().clamp_min(1e-12)))


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    model,_=load_model(saved);configure_trainable(model,'coordinated')
    geometry=CaptureGeometry(model.inference.global_motion_rep,'cuda:0')
    cases=[c for c in torch.load(args.cases,map_location='cpu',weights_only=False) if c['pilot_split']=='train']
    coeff=saved['coefficients'];rows=[]
    for i,case in enumerate(cases):
        with torch.no_grad():prepared=tuple(v[0] for v in geometry.geometry(case['world'][None].cuda()))
        field=HeightfieldPrismSDF(case['scene'])
        loss,detail=episode_loss(model,case,geometry,prepared,field,coeff,0)
        prediction=geometry.prediction;physical=geometry.values;target=geometry.target
        _,points=geometry.geometry(prediction);end=prediction.shape[1]-4
        xyz,active=source_sole_position_loss(points[:,4:end,:geometry.sole_count],target['points'][:,4:end,:geometry.sole_count],target['heights'])
        original=coeff['fk_m']*xyz+coeff['support_gap_m']*physical['support_gap_m']+coeff['source_support_velocity_mps']*physical['source_support_velocity_mps']
        normalized=coeff['fk_m']*xyz/.02+coeff['support_gap_m']*physical['support_gap_m']/.02+coeff['source_support_velocity_mps']*physical['source_support_velocity_mps']/.1
        local=model.inference.local_motion_rep
        root=published_root_reconstruction(prediction,target['joints'][:,:,model.inference.global_motion_rep.skeleton.root_idx],
            local.stats.std[local.indices['local_root_vel']],local.stats.std[local.indices['global_root_y']])
        remainder=loss-original+float(model.inference._root_model.args['local_root_loss_coeff'])*root
        groups={name:norms_and_cosine((original,remainder,normalized),tuple(p for p in model.get_submodule(name).parameters() if p.requires_grad))
                for name in ('root','pose','codec.decoder')}
        row=dict(label=case['label'],source_support_vertex_frames=int(active.sum()),
            source_support_frames=int(active.any(-1).sum()),source_scored_frames=active.shape[1],
            existing_support_loss=float(original.detach()),normalized_support_loss=float(normalized.detach()),
            other_loss=float(remainder.detach()),groups=groups,detail=detail)
        rows.append(row);write_json(args.output/'progress.json',dict(rows=rows));print(json.dumps(row),flush=True)
        geometry.prediction=geometry.target=geometry.values=None
        del loss,prediction,physical,target,points,root,original,normalized,remainder
    write_json(args.output/'complete.json',dict(rows=rows,optimizer_updates=0,checkpoint_sha256=sha256(args.checkpoint),
        normalization_scales=dict(support_position_m=.02,support_velocity_mps=.1),
        note='Scales are existing support criteria. Gradient magnitude is evidence, not proof that reweighting fixes motion.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','cases','output'):parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
