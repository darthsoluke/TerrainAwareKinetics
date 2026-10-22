"""One continuous route across independently authored PARC terrain sections."""

import hashlib
import random

import torch

from build_parc_mixed_course import build_mixed_course
from build_parc_stair_course import _load_official_terrain_util


def build_long_mixed_course(seed, tiles=3, difficulty=0, device='cpu'):
    if not 2 <= tiles <= 8 or difficulty not in (0,1,2):
        raise ValueError('Expected 2..8 independent terrain sections and difficulty 0..2')
    terrain_util=_load_official_terrain_util()
    rng=random.Random(seed)
    sections=[]
    for index in range(tiles):
        section_seed=seed+1009*index
        overrides={'stair_riser_m':rng.uniform(.10,.18),
                   'obstacle_height_m':rng.uniform(.5,1.0),
                   'rock_step_m':rng.uniform(.03,.09),
                   'trench_depth_m':rng.uniform(.35,.6)}
        sections.append((*build_mixed_course('cpu',difficulty,section_seed,profile_overrides=overrides),overrides))
    width,height=sections[0][0].hf.shape
    spacing=float(sections[0][0].dxdy[0])
    terrain=terrain_util.SubTerrain(f'continuous_mixed_{seed}',x_dim=width*tiles,y_dim=height,
        dx=spacing,dy=spacing,min_x=0.,min_y=0.,device='cpu')
    paths=[]; segments=[]; provenance=[]; count=0
    for index,(part,nodes,course,parameters) in enumerate(sections):
        terrain.hf[index*width:(index+1)*width]=part.hf
        translated=nodes.clone();translated[:,0]+=index*width*spacing
        if paths:
            previous=paths[-1][-1]
            if not torch.allclose(previous[1:],translated[0,1:],atol=1e-5,rtol=0):
                raise ValueError('Terrain sections require a level straight connecting walkway')
            steps=round(float((translated[0,0]-previous[0])/spacing))
            if steps<2:
                raise ValueError('Independent terrain sections must not overlap')
            connector=previous[None].repeat(steps,1)
            connector[:,0]+=torch.arange(1,steps+1)*spacing
            paths.append(connector)
            segments.append({'name':f'walkway_{index}','stage':'walk',
                             'path_node_start':count-1,'path_node_end':count+steps-1})
            count+=steps
            # The connector already contains this section's first node.
            offset=count-1
            paths.append(translated[1:]);count+=len(translated)-1
        else:
            offset=0;paths.append(translated);count=len(translated)
        segments.extend(dict(segment,name=f'section_{index}_'+segment['name'],
            path_node_start=offset+segment['path_node_start'],path_node_end=offset+segment['path_node_end'])
            for segment in course['route_segments'])
        provenance.append({'section':index,'seed':seed+1009*index,'parameters':parameters,
                           'terrain_sha256':hashlib.sha256(part.hf.numpy().tobytes()).hexdigest()})
    nodes=torch.cat(paths)
    sampled=terrain.get_hf_val_from_points(nodes[:,:2])
    if not torch.allclose(sampled,nodes[:,2],atol=1e-5,rtol=0):
        raise ValueError('Continuous route support nodes differ from terrain geometry')
    distance=float(torch.diff(nodes[:,:2],dim=0).norm(dim=-1).sum())
    terrain.hf_maxmin[...,0]=terrain.hf;terrain.hf_maxmin[...,1]=terrain.hf
    manifest={'schema_version':1,'course_kind':'continuous_mixed','scenario_id':f'continuous_mixed_{seed}',
        'stair_flights':[],'route_segments':segments,'course':{'planar_length_m':distance,
            'surface_length_m':float(torch.diff(nodes,dim=0).norm(dim=-1).sum())},
        'sections':provenance,'section_count':tiles,'continuous_generation_required':True,
        'motion_concatenation':False,'history_reset_between_sections':False,
        'terrain_sha256':hashlib.sha256(terrain.hf.numpy().tobytes()).hexdigest(),
        'requested_family':'mixed','generated_behavior_verified':False,
        'path_surface_max_error_m':float((sampled-nodes[:,2]).abs().max())}
    terrain.set_device(device)
    return terrain,nodes.to(device),manifest
