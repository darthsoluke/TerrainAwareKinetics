"""Read-only audit of completed training exposure and preserved PARC intents."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_learning_data import TerrainLearningDataset,learning_sampling_weights
from motionbricks.data.joint_sdf_training import approach_intent
from scripts.train_native_mb import sha256


def main(args):
    torch.set_num_threads(4)
    saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
    path=Path(saved['contract']['dataset'])
    data=TerrainLearningDataset(path,build_native_representation(path/'stats'),'train')
    weights=learning_sampling_weights(data,0.)
    exposure=Counter()
    for rank in saved['rank_state']:exposure.update(rank['episodes']['counts'])
    total=sum(value for key,value in exposure.items() if key.startswith('primitive:'))
    if total!=saved['primitive_examples']:raise ValueError('Rank exposure counters do not match retained sample budget')
    categories=Counter();examples=[];sidecars=0;missing=0
    for index,row in enumerate(data.records):
        kind=row.get('source_intent',{}).get('kind','absent')
        categories[(row['origin'],kind)]+=1
        if row.get('source_intent_sidecar'):sidecars+=1
        if row.get('source_intent_sidecar') and not (path/row['source_intent_sidecar']).is_file():missing+=1
        matching=sum(r['origin']==row['origin'] and (r.get('source_intent') or {}).get('kind','absent')==kind for r in examples)
        if row['source_id']=='official_df3f1541e287fcb3' or (row['origin'].startswith('parc_generated') and matching<2):
            entry={key:row.get(key) for key in ('source_id','origin','frames_30hz','terrain_family','source_intent',
                'source_intent_sidecar','source_intent_sidecar_sha256','scene_scale','intent_sidecar')}
            entry['episode_sampling_probability']=float(weights[index])
            sidecar=path/row['source_intent_sidecar'] if row.get('source_intent_sidecar') else None
            if sidecar and sidecar.is_file():
                entry['hash_verified']=sha256(sidecar)==row['source_intent_sidecar_sha256']
                with np.load(sidecar,allow_pickle=False) as values:
                    command=values['command_world_xy'];entry['sidecar_keys']=values.files
                    entry['command_norm_quantiles']=np.quantile(np.linalg.norm(command,axis=-1),[0,.5,.95,1]).tolist()
                    entry['command_coordinates']=str(values['command_coordinates']) if 'command_coordinates' in values else None
                    entry['fps']=float(values['fps']);entry['frames']=len(command)
            item=data.full_reference_item(index)
            if row['source_id']=='official_df3f1541e287fcb3':
                control=approach_intent(item);entry['full_clip_weak_approach_control']=None if control is None else control.tolist()
            examples.append(entry)
    report={'actual_updates':saved['step'],'primitive_examples':total,
        'all_rank_exposure':dict(exposure),'intent_inventory':[{'origin':k[0],'kind':k[1],'records':v} for k,v in categories.items()],
        'preserved_sidecars':sidecars,'missing_sidecars':missing,'examples':examples,
        'note':'Generator target displacement and measured root velocity are different units/semantics; no automatic substitution approved by this audit'}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'primitive_examples':total,'weak_approach_examples':exposure['weak_approach_intent_primitives'],
        'named_stair_examples':sum(v for k,v in exposure.items() if k.startswith('primitive:') and 'stairs' in k),
        'intent_inventory':report['intent_inventory'],'examples':examples},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
