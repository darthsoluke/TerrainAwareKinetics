import numpy as np
import pytest
import torch

from motionbricks.data.terrain_test_world import build_world, merged_surface, ZONE_TYPES
from motionbricks.data.local_terrain_observation import extract_local_terrain_observation
from motionbricks.data.native_mb_terrain_protocol import terrain_observation_config, INTENT_RANGE_POLICY


@pytest.fixture(autouse=True)
def cpu_threads():
    previous=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_large_world_has_varied_terrain_and_safe_entry_points():
    heights,origin,spawns=build_world()
    assert heights.shape==(1280,1280) and np.isfinite(heights).all()
    assert len(spawns)==37 and len({s['id'] for s in spawns})==37
    assert set(s['kind'] for s in spawns[1:])==set(row[0] for row in ZONE_TYPES)
    assert heights.min()<=-.79 and heights.max()>=3.5
    for spawn in spawns:
        point=torch.tensor([[spawn['spawn_xz'][0],1.,spawn['spawn_xz'][1]]])
        observation=extract_local_terrain_observation(torch.from_numpy(heights),origin,.2,point,0.,[0.,0.],
            config=terrain_observation_config(INTENT_RANGE_POLICY))
        assert observation.valid.all()
        torch.testing.assert_close(observation.points[:,0,:,:,1],torch.full((1,9,9),-1.))


def test_merged_tops_preserve_every_cell_height_and_area():
    heights=np.array([[0,0,.3,.3],[0,0,.3,.3],[-.5,.1,.1,.3]],dtype=np.float32)
    origin=np.array([-2.,0.,3.]);dx=.2
    mesh=merged_surface(heights,origin,dx)
    quads=np.asarray(mesh['vertices']).reshape(-1,4,3)
    top=quads[np.ptp(quads[:,:,1],axis=1)==0]
    area=sum(np.ptp(q[:,0])*np.ptp(q[:,2]) for q in top)
    assert area==pytest.approx(heights.size*dx*dx,abs=1e-6)
    for z in range(len(heights)):
        for x in range(heights.shape[1]):
            px,pz=origin[0]+x*dx,origin[2]+z*dx
            matches=[q for q in top if q[:,0].min()<px<q[:,0].max() and q[:,2].min()<pz<q[:,2].max()]
            assert len(matches)==1
            assert matches[0][0,1]==pytest.approx(float(heights[z,x]),abs=1e-7)
    flat=merged_surface(np.zeros((100,100),dtype=np.float32),origin,dx)
    assert len(flat['faces'])//3==2
