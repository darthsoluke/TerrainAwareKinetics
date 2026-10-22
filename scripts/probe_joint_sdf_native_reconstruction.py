"""Bounded source-reconstruction capacity probe, explicitly not a live controller."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from motionbricks.data.joint_sdf_controller import JointSDFController
from motionbricks.data.terrain_control import SoftCommandState,demonstration_controls
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import sha256
from scripts.train_native_mb_control import state_digest
from scripts.train_joint_terrain_v3 import write_json


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(2333)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    source=Path(saved['contract']['source'])
    if sha256(source)!=saved['contract']['source_bundle_sha256']:raise ValueError('Source identity mismatch')
    inference,bundle=load_terrain_bundle(source,'cuda:0');model=JointSDFController(inference,bundle)
    model.load_state_dict(saved['model_state'],strict=True);model.requires_grad_(False);model.eval()
    model.codec.decoder.requires_grad_(True)
    frozen={name:state_digest(getattr(model,name)) for name in ('root','pose','control','root_far','pose_far',
        'decoder_near','decoder_attention','decoder_far','sdf_encoder','sdf_joint_embedding','sdf_root','sdf_pose','sdf_decoder')}
    frozen.update(codec_encoder=state_digest(model.codec.encoder),codec_quantizer=state_digest(model.codec.quantizer))
    cases=torch.load(args.cases,map_location='cpu',weights_only=False)
    optimizer=torch.optim.Adam(model.codec.decoder.parameters(),lr=5e-5);rng=np.random.default_rng(2333)
    contract={'initial_sha256':sha256(args.checkpoint),'steps':args.updates,'batch':3,
        'objective':'existing MotionVQVAEModel.loss on codec(source local motion, source Root, first4 source pose)',
        'source_reference':'motionbricks/vqvae/models/motion_vqvae.py training_step source encoder/quantizer/decoder path',
        'diagnostic_only':True,'future_source_root':'oracle reconstruction conditioning only; prohibited as live input',
        'pose_condition':'first4 frames only','trainable':'codec.decoder only','encoder_and_codebook':'frozen',
        'loss_coefficients':'unchanged native tokenizer configuration; no geometric penalty added',
        'optimizer':'fresh Adam5e-5, gradient clip0.5','postprocessing':None,'quality_approved':False}
    write_json(args.output/'contract.json',contract);started=time.perf_counter()
    for step in range(args.updates):
        selected=[cases[0],cases[0],cases[1+step%2]]
        starts=[int(rng.integers(len(case['world'])-32)) for case in selected]
        world=torch.stack([case['world'][start:start+33] for case,start in zip(selected,starts)]).cuda()
        with torch.no_grad():
            intent=SoftCommandState().from_controls(world[:,:4],demonstration_controls(world,32),8)
            result=model(world[:,:4],intent,[case['scene'] for case in selected],source=world)
            target=result['target'];local=target['local_motions']
            root=extract_feature_from_motion_rep(local,model.codec.motion_rep,model.codec.decoder_external_cond_feature_mode)
            mask=torch.zeros(local.shape[:2],dtype=torch.bool,device=local.device);mask[:,:4]=True
        output=model.codec(local,target_cond=local.detach(),has_target_cond=mask,external_cond=root.detach())
        losses=model.loss_model.loss(target,output,None);loss=losses['loss']
        if not torch.isfinite(loss):raise ValueError('Nonfinite native reconstruction loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.codec.decoder.parameters(),.5)
        if not torch.isfinite(norm):raise ValueError('Nonfinite reconstruction gradient')
        optimizer.step()
        if step%20==0 or step+1==args.updates:
            value={'step':step+1,'primitive_examples':(step+1)*3,'loss':float(loss.detach()),
                'gradient_norm':float(norm),'elapsed_seconds':time.perf_counter()-started,'diagnostic_only':True}
            write_json(args.output/'progress.json',value);print(json.dumps(value),flush=True)
    after={name:state_digest(getattr(model,name)) for name in frozen if not name.startswith('codec_')}
    after.update(codec_encoder=state_digest(model.codec.encoder),codec_quantizer=state_digest(model.codec.quantizer))
    if after!=frozen:raise ValueError('A supposedly frozen module changed')
    checkpoint=args.output/'diagnostic_weights.pt'
    torch.save({'schema':'motionbricks.native_codec_capacity_diagnostic.v1','model_state':model.state_dict(),
        'step':args.updates,'contract':dict(saved['contract'],native_codec_capacity_probe=contract),
        'coefficients':saved['coefficients']},checkpoint)
    del model,inference,bundle,saved,optimizer,output,result,loss,losses
    import gc
    gc.collect();torch.cuda.empty_cache()
    diagnostic_script=Path(__file__).with_name('diagnose_joint_sdf_codec_capacity.py')
    subprocess.run([sys.executable,'-u',str(diagnostic_script),'--checkpoint',str(checkpoint),
        '--output',str(args.output/'oracle_after')],check=True)
    write_json(args.output/'complete.json',{'actual_updates':args.updates,'primitive_examples':args.updates*3,
        'diagnostic_only':True,'quality_approved':False,'checkpoint_sha256':sha256(checkpoint)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--updates',type=int,default=200);main(p.parse_args())
