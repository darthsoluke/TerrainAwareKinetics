"""Print factual staircase approaches before selecting a pilot case."""
import sys
from pathlib import Path
import torch
from motionbricks.data.native_mb_data import build_native_representation
from motionbricks.data.terrain_learning_data import TerrainLearningDataset
from motionbricks.data.terrain_response_loss import heightfield_gap

saved=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
root=Path(saved['contract']['dataset'])
data=TerrainLearningDataset(root,build_native_representation(root/'stats'),'train')
for i,r in enumerate(data.records):
    if 'stairs' not in str(r.get('terrain_family')) or not str(r['source_id']).startswith('official'):continue
    item=data.full_reference_item(i);w=item['world'];gap,known=heightfield_gap(w[:,:3],item['scene']);g=w[:,1]-gap
    print(r['source_id'],len(w), 'edges',torch.nonzero((g[1:]-g[:-1]).abs()>.05).flatten().tolist(),flush=True)
    if len(w)==85:
        for t in range(0,len(w),5):print(t,w[t,:5].tolist(),float(g[t]),bool(known[t]),flush=True)
