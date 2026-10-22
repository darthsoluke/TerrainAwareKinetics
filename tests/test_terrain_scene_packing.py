import torch

from motionbricks.data.terrain_model_v3 import pack_scene_chunks,multiscale_observation


def test_ragged_packing_preserves_observations_without_global_size_padding():
    torch.set_num_threads(2)
    data=[]
    for i in range(8):
        size=251 if i==0 else 41
        hf=torch.zeros(size,size);hf[size//2:,size//2:]=.2
        data.append({'heightfield_y_m':hf,'min_point_world_xyz_m':[-size*.05,0.,-size*.05],'dx_m':.1})
    packed=pack_scene_chunks(data,'cpu',cell_budget=130000)
    assert sum(c[2].numel() for c in packed)<8*251*251/2
    individual=[]
    for i,s in enumerate(data):
        _,_,h,o,d=pack_scene_chunks([s],'cpu')[0];individual.append((i,i+1,h,o,d))
    history=torch.zeros(8,4,8);history[...,1]=1;history[...,3]=1
    intent=torch.tensor([[0.,1.,0.,1.,1.,0.]]).expand(8,-1)
    a=multiscale_observation(history,intent,data,8,packed=packed)
    b=multiscale_observation(history,intent,data,8,packed=individual)
    for key in a:torch.testing.assert_close(a[key],b[key],atol=1e-6,rtol=1e-6)
