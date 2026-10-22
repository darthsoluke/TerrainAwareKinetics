"""Category-balanced real BONES replay with causal facing controls."""

from collections import Counter,OrderedDict
import hashlib
import json
from pathlib import Path

import torch

from .terrain_control import demonstration_controls
from .terrain_learning_data import TerrainLearningDataset

CATEGORIES=('idle','walk','run','start','stop','backward','lateral','turn')


def bones_seed_controls(window,category,reference=None,observed_history=None):
    # Only the planar trajectory supplies weak intent labels, never future body facing.
    velocity=demonstration_controls(window,32,category=category,reference=reference)[:,:2]
    history=window[:,:4] if observed_history is None else observed_history
    yaw=torch.atan2(history[:,3,4],history[:,3,3])
    previous=(history[:,3,[0,2]]-history[:,0,[0,2]])*10
    turning=(previous.norm(dim=-1)>.2)&(velocity.norm(dim=-1)>.2)
    requested=torch.atan2(velocity[:,0],velocity[:,1])
    prior=torch.atan2(previous[:,0],previous[:,1])
    desired=torch.where(turning,yaw+requested-prior,yaw)
    return torch.cat((velocity,desired.cos()[:,None],desired.sin()[:,None]),-1).detach()


def category_slots(step,count=4):
    return [CATEGORIES[(step*count+j)%len(CATEGORIES)] for j in range(count)]


def frame_slot(frames,category,draw,identity):
    count=(frames-33)//8+1
    if count<1:raise ValueError('Need a complete real33-frame window')
    if category in ('start','stop'):
        return (0,(count-1)//2,count-1)[draw%3]*8
    seed=int(hashlib.sha256(identity.encode()).hexdigest()[:8],16)
    return ((seed+draw*7)%count)*8


def group(row):
    return row.get('source_group',row.get('recording_group',row['source_id']))


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


class BonesSeedPool:
    def __init__(self,directory,rep,expected_stats):
        self.root=Path(directory)
        manifest=json.loads((self.root/'native_manifest.json').read_text())
        if sha(self.root/'accepted.json')!=manifest['artifact_sha256']['accepted.json']:
            raise ValueError('BONES source list changed')
        for name in ('mean.npy','std.npy'):
            if sha(self.root/'stats'/name)!=sha(Path(expected_stats)/name):
                raise ValueError('BONES normalization differs from the accepted controller')
        self.datasets={s:TerrainLearningDataset(self.root,rep,s) for s in ('train','validation','test')}
        self.cache=OrderedDict();self.groups={};self.evaluation=[];reserved=set()
        eligible={s:{c:[] for c in CATEGORIES} for s in self.datasets}
        for split,data in self.datasets.items():
            for index,row in enumerate(data.records):
                category=row.get('control_category')
                if row.get('origin')=='bones_seed' and category in CATEGORIES:
                    eligible[split][category].append(index)
            for category,indices in eligible[split].items():
                indices.sort(key=lambda i:data.records[i]['source_id'])
        for category in CATEGORIES:
            split=next((s for s in ('validation','test','train') if eligible[s][category]),None)
            if split is None:raise ValueError('No BONES coverage: '+category)
            index=eligible[split][category][0];item=self._item(split,index)
            if split=='train':reserved.add(group(item['row']))
            world=item['world'];length=min(129,len(world))
            start=len(world)-length if category=='stop' else (0 if category in ('start','idle') else (len(world)-length)//2)
            self.evaluation.append(dict(item,world=world[start:start+length],reference=world,
                category=category,source_start_frame=start,evaluation_split=split,
                scope='original held split' if split!='train' else 'reserved recording family from this pilot; not unseen pretraining'))
        for category in CATEGORIES:
            self.groups[category]=[i for i in eligible['train'][category]
                if group(self.datasets['train'].records[i]) not in reserved]
            if not self.groups[category]:raise ValueError('Reservation removed a BONES category')
        self.contract=dict(source=str(self.root),manifest_sha256=sha(self.root/'native_manifest.json'),
            categories=list(CATEGORIES),train_counts={c:len(v) for c,v in self.groups.items()},
            reserved_recording_groups=sorted(reserved),
            evaluation=[dict(source_id=c['row']['source_id'],category=c['category'],split=c['evaluation_split'],scope=c['scope']) for c in self.evaluation],
            command='weak source planar velocity; causal facing from real/generated observed history',
            future_height_or_body_facing=False,plane='existing fixed zero plane; no pose/height fitting')

    def _item(self,split,index):
        key=(split,index)
        if key in self.cache:
            self.cache.move_to_end(key);return self.cache[key]
        data=self.datasets[split];row=data.records[index]
        if row.get('origin')!='bones_seed':raise ValueError('Non-BONES replay row')
        if sha(self.root/row['shard'])!=row['converted_shard_sha256']:
            raise ValueError('BONES motion shard changed')
        item=data.full_reference_item(index)
        heights=torch.as_tensor(item['scene']['heightfield_y_m'])
        if not torch.isfinite(heights).all() or torch.count_nonzero(heights):
            raise ValueError('BONES scene is not its original fixed zero plane')
        world=item['world'];low=world[:,[0,2]].amin(0)-50;high=world[:,[0,2]].amax(0)+50
        size=torch.ceil((high-low)/.5).long()+1
        scene=dict(heightfield_y_m=torch.zeros(int(size[1]),int(size[0])),
            min_point_world_xyz_m=torch.tensor([low[0],0.,low[1]]),dx_m=torch.tensor(.5))
        item=dict(item,scene=scene)
        self.cache[key]=item
        if len(self.cache)>64:self.cache.popitem(last=False)
        return item

    def sample(self,step):
        rows=[]
        for j,category in enumerate(category_slots(step)):
            draw=(step*4+j)//len(CATEGORIES)
            indices=self.groups[category];index=indices[draw%len(indices)]
            item=self._item('train',index);identity=item['row']['source_id']
            offset=frame_slot(len(item['world']),category,draw,identity)
            window=item['world'][offset:offset+33]
            value=bones_seed_controls(window[None],category,item['world'][None])
            rows.append((window,value,item['scene'],dict(source_id=identity,category=category,
                start_frame=offset,draw=draw,split='train',origin='bones_seed')))
        return rows
