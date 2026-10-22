"""Cached terrain must not accumulate shared-memory file descriptors in workers."""

import os
import sys

import pytest
import torch
from torch.utils.data import DataLoader,Dataset

from motionbricks.data.terrain_learning_data import collate_learning_items


class CachedScenes(Dataset):
    def __init__(self):self.cache={}
    def __len__(self):return 1200
    def __getitem__(self,index):
        scene={'heightfield_y_m':torch.full((8,8),float(index)),
               'min_point_world_xyz_m':torch.tensor([index,0.,0.]),'dx_m':torch.tensor(.1)}
        self.cache[index]=scene
        return {'scene':scene,'world':torch.full((4,3),float(index)),
                'cached_shared':sum(value.is_shared() for entry in self.cache.values() for value in entry.values()),
                'open_files':len(os.listdir('/proc/self/fd')) if sys.platform=='linux' else 0,
                'row':{'source_id':str(index)},'override':None}


def low_file_limit(_worker):
    import resource
    _,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(512,hard),hard))
    torch.set_num_threads(1)


def test_scene_batch_values_do_not_alias_the_long_lived_cache():
    item=CachedScenes()[7]
    result=collate_learning_items([item])[0]
    assert torch.equal(result['scene']['heightfield_y_m'],item['scene']['heightfield_y_m'])
    result['scene']['heightfield_y_m'].fill_(-9)
    assert torch.all(item['scene']['heightfield_y_m']==7)
    assert result['world'] is item['world']
    assert result['row']==item['row']


@pytest.mark.skipif(sys.platform!='linux',reason='Linux shared-memory file-descriptor regression')
def test_spawn_workers_keep_scene_cache_private_under_low_file_limit():
    loader=DataLoader(CachedScenes(),batch_size=8,num_workers=2,multiprocessing_context='spawn',
        collate_fn=collate_learning_items,worker_init_fn=low_file_limit,prefetch_factor=1,timeout=45)
    counts=[];seen=0
    for batch in loader:
        for item in batch:
            assert not item['cached_shared']
            assert torch.all(item['scene']['heightfield_y_m']==int(item['row']['source_id']))
            counts.append(item['open_files']);seen+=1
    assert seen==1200 and max(counts)<256
