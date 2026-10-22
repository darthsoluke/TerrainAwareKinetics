"""Verify native checkpoint compaction and fresh offline inference construction."""

import argparse
import gc
import json
from pathlib import Path

import torch

from motionbricks.data.full_corpus_inference import compact_candidate,inference_receipt,load_candidate,validate_candidate
from motionbricks.data.unified_terrain import UnifiedTerrainController
from motionbricks.data.unified_training_goals import source_terminal_goal
from scripts.portable_parc_online_soma import load_terrain_bundle
from scripts.train_native_mb import sha256
from scripts.train_joint_terrain_v3 import write_json


@torch.no_grad()
def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False)
    source_receipt=json.loads((a.checkpoint.parent.parent/'complete.json').read_text())
    source_hash=sha256(a.checkpoint)
    if not source_receipt['passed'] or source_receipt['final_checkpoint_sha256']!=source_hash:
        raise ValueError('Actual mechanics final checkpoint identity differs')
    saved=torch.load(a.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if not saved['contract']['mechanics_only']:
        raise ValueError('This pre-release check accepts only the declared mechanics fixture')
    bundle_hash=sha256(a.bundle)
    try:
        validate_candidate(saved,bundle_hash)
    except ValueError:
        rejected_without_opt_in=True
    else:
        raise ValueError('Mechanics checkpoint incorrectly accepted as production')
    compact=compact_candidate(saved,bundle_hash,allow_mechanics=True)
    checkpoint=a.output/'diagnostic-weights.pt'
    torch.save(compact,checkpoint)
    compact_hash=sha256(checkpoint)
    receipt=inference_receipt(compact,source_hash,compact_hash,bundle_hash,allow_mechanics=True)
    write_json(a.output/'export.json',receipt)
    cases=torch.load(a.cases,map_location='cpu',weights_only=False)
    selected=[next(case for case in cases if case.get('entry')==entry or case.get('name')==entry)
        for entry in ('up','down','idle')]
    inference,bundle=load_terrain_bundle(a.bundle,'cuda:0')
    direct=UnifiedTerrainController(inference,bundle)
    direct.load_state_dict(saved['model_state'],strict=True)
    direct.codebook.copy_(direct.codec.get_codebook())
    direct.eval().requires_grad_(False)
    references=[]
    for case in selected:
        world=case['world'][None,:33].cuda()
        command=source_terminal_goal(world[...,:5])
        result=direct(world[:,:4],command,case['scene'])
        references.append({key:result[key].detach().cpu().clone() for key in ('world','local_prediction','pose_logits')})
    direct.close()
    del direct,inference,result
    gc.collect()
    torch.cuda.empty_cache()
    inference,bundle=load_terrain_bundle(a.bundle,'cuda:0')
    actual=UnifiedTerrainController(inference,bundle)
    initial_codec=actual.codec.get_codebook().detach().clone()
    if sha256(checkpoint)!=receipt['checkpoint_sha256']:
        raise ValueError('Compact file changed before load')
    reloaded=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    load_candidate(actual,reloaded,receipt,bundle_hash,allow_mechanics=True)
    codebook_changed_from_bundle=float((actual.codec.get_codebook()-initial_codec).abs().max())
    if codebook_changed_from_bundle==0:
        raise ValueError('Fixture did not exercise replacement of the old vocabulary')
    rows=[]
    for case,reference in zip(selected,references):
        world=case['world'][None,:33].cuda()
        command=source_terminal_goal(world[...,:5])
        result=actual(world[:,:4],command,case['scene'])
        differences={}
        for key in reference:
            value=result[key].detach().cpu()
            torch.testing.assert_close(value,reference[key],atol=0,rtol=0)
            differences[key]=float((value-reference[key]).abs().max())
        rows.append(dict(label=case['label'],max_differences=differences,
            active_token_changes=int((result['pose_logits'][:,:8].argmax(-1).cpu()!=reference['pose_logits'][:,:8].argmax(-1)).sum())))
    actual.close()
    write_json(a.output/'complete.json',dict(passed=True,rows=rows,source_checkpoint_sha256=source_hash,
        compact_checkpoint_sha256=compact_hash,model_state_sha256=receipt['model_state_sha256'],
        codec_state_sha256=receipt['codec_state_sha256'],codebook_changed_from_bundle_max=codebook_changed_from_bundle,
        mechanics_requires_explicit_opt_in=rejected_without_opt_in,actual_optimizer_updates=0,
        interactive_server_started=False,quality_approved=False,
        note='Mechanics fixture validates export/load behavior, not full-trained motion quality or WASD.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','checkpoint','cases','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if args.output.is_dir():
            write_json(args.output/'failure.json',dict(error=repr(error)))
        raise
