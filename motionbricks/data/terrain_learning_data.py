"""Split-preserving source motion and explicit weak obstacle/stop pairings."""

import json
from pathlib import Path

import numpy as np
import torch

from motionbricks.data.native_mb_data import NativeMBMotionDataset
from motionbricks.data.terrain_control import SoftCommandState,demonstration_controls


def world_from_item(rep,item):
    raw=rep.unnormalize(item['motion'])[None]
    root=item['terrain_root_world'][None]
    world=rep.change_first_heading(raw,torch.atan2(root[:,0,4],root[:,0,3]),False,False)
    world[...,[0,2]]+=root[:,:1,[0,2]]
    torch.testing.assert_close(world[...,:5],root,atol=5e-5,rtol=1e-5)
    return world[0]


def source_task_intent(world,frames,*,category=None,override=None,history=None,state=None,reference=None):
    controls=demonstration_controls(world,frames,category=category,override=override,reference=reference)
    # A reference trajectory supplies an intended planar destination. Moving it
    # with prediction error would remove the information needed by fixed-world
    # supervision. No future Y or joint pose becomes a model condition.
    goal=world[:,frames-1,[0,2]] if override is None else None
    return (state or SoftCommandState()).from_controls(world[:,:4] if history is None else history,controls,frames//4,goal_xz=goal)


class TerrainLearningDataset(torch.utils.data.Dataset):
    def __init__(self,directory,representation,split,weak_pairs=None,*,max_frames=129,cache_bytes=2*1024**3):
        self.split=split
        self.base=NativeMBMotionDataset(directory,split,min_frames=65,max_frames=max_frames,
            require_terrain=True,require_online_terrain=True,online_only=True,cache_max_bytes=cache_bytes)
        self.rep=representation
        self.pairs=[];self.pair_root=None
        if weak_pairs is not None:
            self.pair_root=Path(weak_pairs)
            manifest=json.loads((self.pair_root/'manifest.json').read_text())
            if manifest.get('schema')!='motionbricks.weak_obstacle_stop_pairs.v2':
                raise ValueError('Rebuild weak pairs with independent heading and rolling-control labels')
            self.pairs=[r for r in manifest['items'] if r['split']==split]
        self.records=list(self.base.records)+self.pairs

    def __len__(self):return len(self.records)

    def full_reference_item(self,index):
        """Load an unchanged complete real clip for source-defined rollout review."""
        if not 0<=index<len(self.base):raise ValueError('A traversal reference must be a real paired source')
        row=self.base.records[index]
        with np.load(self.base.dataset_dir/row['shard'],allow_pickle=False) as values:
            motion=torch.from_numpy(values['global_features'].copy())
        scene,root=self.base._load_online_terrain(row,motion)
        item={'motion':self.base.stats.normalize(motion),'terrain_root_world':root}
        return {'world':world_from_item(self.rep,item),'scene':scene,'row':row,'override':None}

    def __getitem__(self,index):
        if isinstance(index,tuple):
            selected,draw=index
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(2333+int(draw))
                item=self[int(selected)]
                return dict(item,sampling_index=int(selected),sampling_draw=int(draw))
        if index<len(self.base):
            item=self.base[index];row=self.base.records[index]
            world=world_from_item(self.rep,item)
            scene=item['terrain_scene']
            if row['origin']=='bones_seed':
                if torch.count_nonzero(scene['heightfield_y_m']):raise ValueError('BONES plane assumption changed')
                center=world[3,:3]
                scene={'heightfield_y_m':torch.zeros(201,201),
                    'min_point_world_xyz_m':torch.tensor([center[0]-50,0,center[2]-50]),'dx_m':torch.tensor(.5)}
            return {'world':world,'scene':scene,'row':row,'override':None}
        row=self.pairs[index-len(self.base)]
        path=(self.pair_root/row['file']).resolve()
        if self.pair_root.resolve() not in path.parents:raise ValueError('Weak pair escapes its artifact root')
        with np.load(path,allow_pickle=False) as values:
            world=torch.from_numpy(values['world'].copy())
            scene={'heightfield_y_m':torch.from_numpy(values['heightfield_y_m'].copy()),
                'min_point_world_xyz_m':torch.from_numpy(values['min_point_world_xyz_m'].copy()),
                'dx_m':torch.tensor(float(values['dx_m']))}
            override=torch.from_numpy(values['control'].copy())
        return {'world':world,'scene':scene,'row':row,'override':override}


def learning_sampling_weights(dataset,weak_fraction=0.):
    if not 0<=weak_fraction<.5:raise ValueError('Weak fraction must be in [0, .5)')
    records=dataset.records
    domains={'parc':[],'bones':[],'weak':[]}
    categories={}
    for i,row in enumerate(records):
        if i>=len(dataset.base):domains['weak'].append(i)
        elif row['origin'].startswith('parc'):domains['parc'].append(i)
        else:
            domains['bones'].append(i);categories.setdefault(row['control_category'],[]).append(i)
    if not domains['weak']:weak_fraction=0.
    weights=torch.zeros(len(records),dtype=torch.double)
    for i in domains['parc']:
        weights[i]=max(1,records[i]['frames_30hz']-64)
    if domains['parc']:weights[domains['parc']]*=.5/weights[domains['parc']].sum()
    for group in categories.values():weights[group]=(.5-weak_fraction)/len(categories)/len(group)
    if domains['weak']:weights[domains['weak']]=weak_fraction/len(domains['weak'])
    if not (weights>0).any():raise ValueError('No eligible source motion')
    return weights


def sampling_manifest(dataset,weights):
    totals={};total=float(weights.sum())
    for index,(row,weight) in enumerate(zip(dataset.records,weights)):
        domain='weak' if index>=len(dataset.base) else 'parc' if row['origin'].startswith('parc') else 'bones'
        family=row.get('control_category','unspecified') if domain=='bones' else row.get('terrain_family','unspecified')
        key=domain+':'+str(family)
        entry=totals.setdefault(key,{'records':0,'probability':0.})
        entry['records']+=1;entry['probability']+=float(weight)/total
    return totals


def collate_learning_items(items):
    # Queueing a cached tensor promotes its storage to shared memory. The
    # worker cache then retains one file descriptor per storage until eviction.
    # Only batch-owned scene copies may cross the multiprocessing boundary.
    return [{**item,'scene':{key:value.clone() if torch.is_tensor(value) else value
                            for key,value in item['scene'].items()}} for item in items]


def sample_aligned_world_windows(items,frames,rng):
    """Sample complete source windows, including late braking in weak pairs.

    Source time advances within that same window during generated-history
    training. No source is moved with the prediction, padded, or re-timed.
    """
    windows=[];starts=[];late=0
    for item in items:
        source=item['world'];maximum=len(source)-frames
        if maximum<0:raise ValueError('Source cannot satisfy the complete primitive/feedback window')
        tail=item['override'] is not None and rng.random()<1/3
        start=maximum if tail else int(rng.integers(maximum+1))
        windows.append(source[start:start+frames]);starts.append(start);late+=int(tail)
    return torch.stack(windows),starts,late
