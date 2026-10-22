"""Eight disjoint regular stair courses for a64-candidate data pilot."""

import hashlib
import math

import torch

from build_parc_stair_course import _load_official_terrain_util


SPECS = (
    (.12,.30,8,2.4),(.14,.40,10,2.4),(.16,.40,8,2.8),(.18,.50,9,2.8),
    (.13,.40,11,2.4),(.15,.30,10,2.4),(.17,.50,9,2.8),(.14,.30,12,2.8),
)


def build_stair64_course(index, device='cpu'):
    if isinstance(index,bool) or index not in range(8):
        raise ValueError('Expected one of eight declared layouts')
    riser,tread,count,width = SPECS[index]
    dx=.1
    cells=round(tread/dx); run=count*cells
    start=10; bottom=40; top=bottom+run; far=top+20; end=far+run; finish=end+30
    nx=finish+11; ny=round((width+2.)/dx)+1; center=ny//2
    terrain=_load_official_terrain_util().SubTerrain('stair64_'+str(index),x_dim=nx,y_dim=ny,
        dx=dx,dy=dx,min_x=0.,min_y=0.,device='cpu')
    terrain.hf.zero_()
    half=round(width/dx)//2
    for x in range(bottom,end+1):
        level=min(count,(x-bottom)//cells) if x<=far else max(0,count-(x-far)//cells)
        terrain.hf[x,center-half:center+half+1]=level*riser
    lateral=(-1,0,1,0)[index%4]
    grid=torch.tensor([[x,center+lateral] for x in range(start,finish+1)],dtype=torch.long)
    reverse=index>=4
    if reverse:
        terrain.hf=terrain.hf.flip(0).contiguous(); grid[:,0]=nx-1-grid[:,0]
    orientation='y' if index%2 else 'x'
    if orientation=='y':
        transformed=_load_official_terrain_util().SubTerrain('stair64_'+str(index),x_dim=ny,y_dim=nx,
            dx=dx,dy=dx,min_x=0.,min_y=0.,device='cpu')
        transformed.hf.copy_(terrain.hf.T); terrain=transformed; grid=grid[:,[1,0]]
    terrain.hf_maxmin[...,0]=terrain.hf;terrain.hf_maxmin[...,1]=terrain.hf
    nodes=terrain.get_xyz_point(grid)
    segments=[]
    boundaries=[(start,bottom,'flat_approach'),(bottom,top,'ascent'),(top,far,'platform'),
                (far,end,'descent'),(end,finish,'flat_exit')]
    for a,b,stage in boundaries:
        segments.append(dict(name=stage,stage=stage,path_node_start=a-start,path_node_end=b-start,
            start_xyz_m=nodes[a-start].tolist(),end_xyz_m=nodes[b-start].tolist()))
    length=float(torch.diff(nodes[:,:2],dim=0).norm(dim=-1).sum())
    manifest=dict(schema_version=1,scenario_id='stair64_'+str(index),course_kind='regular_stair64_v1',
        layout_index=index,split='validation' if index==6 else 'test' if index==7 else 'train',
        terrain_sha256=hashlib.sha256(terrain.hf.numpy().tobytes()).hexdigest(),
        route_segments=segments,course=dict(planar_length_m=length,
            surface_length_m=float(torch.diff(nodes,dim=0).norm(dim=-1).sum()),flat_start_and_end=True),
        stair_flights=[dict(flight_id='flight',orientation=orientation,width_m=width,
            riser_count=count,riser_height_m=riser,tread_depth_m=tread,total_rise_m=count*riser,
            start_xyz_m=nodes[bottom-start].tolist(),ascent_end_xyz_m=nodes[top-start].tolist(),
            platform_end_xyz_m=nodes[far-start].tolist(),end_xyz_m=nodes[end-start].tolist())],
        actual_spacing_m=dx,reverse_axis=reverse,requested_motion='continuous ascent-platform-descent',
        requested_stops=['mid_ascent','mid_descent'],requested_speed_labels=False,
        history_reset_between_phases=False)
    if not math.isclose(float(nodes[0,2]),0.,abs_tol=1e-6) or not math.isclose(float(nodes[-1,2]),0.,abs_tol=1e-6):
        raise ValueError('Course must start and end on genuine flat ground')
    terrain.set_device(device)
    return terrain,nodes.to(device),manifest
