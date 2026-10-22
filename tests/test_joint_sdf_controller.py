import torch

from motionbricks.data.joint_sdf_observation import HeightfieldPrismSDF,JOINT_SDF_NAMES


def test_signed_step_distance_normals_and_gradient():
    scene={'heightfield_y_m':torch.tensor([[0.,1.,1.]]).expand(5,-1),
           'min_point_world_xyz_m':[0.,0.,-2.],'dx_m':1.}
    points=torch.tensor([[.6,.5,0.],[.4,.5,0.],[1.,1.1,0.],[1.,.9,0.]],requires_grad=True)
    field=HeightfieldPrismSDF(scene)
    assert len(field.rectangles)==2
    d,n,valid=field.query(points)
    assert valid.all()
    torch.testing.assert_close(d,torch.tensor([-.1,.1,.1,-.1]),atol=1e-6,rtol=0)
    torch.testing.assert_close(n,torch.tensor([[-1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,1.,0.]]),atol=1e-6,rtol=0)
    grad=torch.autograd.grad(d.sum(),points)[0]
    torch.testing.assert_close(grad,n,atol=1e-6,rtol=0)


def test_scene_translation_and_unknown_distance_certificate():
    hf=torch.zeros(9,9);hf[:,5:]=.4;hf[4,2]=torch.nan
    scene={'heightfield_y_m':hf,'min_point_world_xyz_m':[-4.,0.,-4.],'dx_m':1.}
    p=torch.tensor([[.35,.2,0.],[-2.,.1,0.],[10.,1.,0.],[-1.45,1.,0.]])
    a=HeightfieldPrismSDF(scene).query(p)
    shift=torch.tensor([10.,3.,-20.])
    moved=dict(scene,heightfield_y_m=hf+shift[1],min_point_world_xyz_m=torch.tensor(scene['min_point_world_xyz_m'])+shift)
    b=HeightfieldPrismSDF(moved).query(p+shift)
    torch.testing.assert_close(a[0],b[0],atol=2e-6,rtol=0)
    torch.testing.assert_close(a[1],b[1],atol=2e-6,rtol=0)
    assert torch.equal(a[2],b[2])
    assert a[2].tolist()==[True,False,False,False]


def test_equidistant_surfaces_have_nonunique_valid_normals():
    scene={'heightfield_y_m':torch.tensor([[0.,1.,1.]]).expand(5,-1),
           'min_point_world_xyz_m':[0.,0.,-2.],'dx_m':1.}
    distance,normal,known=HeightfieldPrismSDF(scene).query(torch.tensor([[.3,.2,0.]]))
    assert known.all()
    torch.testing.assert_close(distance,torch.tensor([.2]),atol=1e-6,rtol=0)
    options=torch.tensor([[-1.,0.,0.],[0.,1.,0.]])
    assert (normal-options).norm(dim=-1).min()<1e-6


def test_packed_prism_distances_normals_masks_and_gradients_match_scalar():
    from motionbricks.data.joint_sdf_observation import PackedPrismSDF
    torch.manual_seed(37);scenes=[];fields=[]
    for i in range(3):
        hf=torch.zeros(11,13);hf[:,5:8]=.7;hf[:,8:]=1.2
        if i==1:hf[3:5,4:6]=torch.nan
        scene={'heightfield_y_m':hf+i*.13,'min_point_world_xyz_m':[-2.+i*.2,0.,-2.],'dx_m':.4}
        scenes.append(scene);fields.append(HeightfieldPrismSDF(scene))
    points=(torch.rand(3,41,3)*torch.tensor([5.,2.,5.])+torch.tensor([-2.3,-.2,-2.3])).requires_grad_(True)
    expected=[field.query(p) for field,p in zip(fields,points)]
    actual=PackedPrismSDF(fields,scenes,'cpu').query(points)
    for i in range(3):torch.testing.assert_close(actual[i],torch.stack([value[i] for value in expected]),atol=1e-6,rtol=0)
    a=torch.autograd.grad(actual[0].sum(),points,retain_graph=True)[0]
    b=torch.autograd.grad(sum(value[0].sum() for value in expected),points)[0]
    torch.testing.assert_close(a,b,atol=1e-6,rtol=0)


def make_pair():
    from omegaconf import OmegaConf
    from test_native_mb_online_conditions import online_infer
    from test_native_mb_stage_training import _motion_and_representation
    from test_native_mb_terrain_integration import small_config
    from motionbricks.data.terrain_model_v3 import JointTerrainController
    from motionbricks.data.joint_sdf_controller import JointSDFController
    from motionbricks.data.terrain_control import SoftCommandState
    torch.set_num_threads(2);torch.manual_seed(19)
    a,_=online_infer();b,_=online_infer()
    config={'tokenizer_config':OmegaConf.to_container(small_config('tokenizer'))}
    base=JointTerrainController(a,config);model=JointSDFController(b,config)
    keys=model.load_state_dict(base.state_dict(),strict=False)
    assert not keys.unexpected_keys and all(k.startswith('sdf_') for k in keys.missing_keys)
    _,batch=_motion_and_representation(33)
    source=a.global_motion_rep.unnormalize(batch['motion'][:1,:33]);history=source[:,:4]
    task=SoftCommandState().intent(history,[0.,1.6],0.,8)
    hf=torch.zeros(81,81);hf[43:]=.3
    scene={'heightfield_y_m':hf,'min_point_world_xyz_m':[-4.,0.,-4.],'dx_m':.1}
    return base,model,source,history,task,scene


def test_zero_initialization_causal_targets_and_forward_modes():
    base,model,source,history,task,scene=make_pair()
    with torch.no_grad():
        expected=base(history,task,[scene])['world']
        actual=model(history,task,[scene],source=source)['world']
        changed=source.clone();changed[:,4:,:]+=3.
        counter=model(history,task,[scene],source=changed)['world']
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
    torch.testing.assert_close(actual,counter,atol=0,rtol=0)
    assert len(JOINT_SDF_NAMES)==15
    assert model.sdf_diagnostics['frames']==28
    assert model.sdf_diagnostics['future_source']=='model_proposal'
    with torch.inference_mode():production=model(history,task,[scene])['world']
    torch.testing.assert_close(actual,production,atol=1e-6,rtol=0)


def test_terrain_enters_pose_before_transformer_and_native_noise_loss_trains_it():
    _,model,source,history,task,scene=make_pair();model.freeze_base()
    with torch.no_grad():model.sdf_pose.output.weight.normal_(std=.02)
    recorded=[]
    hook=model.pose._transformer_model.register_forward_pre_hook(lambda m,args:recorded.append(args[0].detach().clone()))
    with torch.no_grad():
        model.sdf_enabled=False;model(history,task,[scene]);plain=recorded[4]
        recorded.clear();model.sdf_enabled=True;model(history,task,[scene]);conditioned=recorded[4]
    hook.remove()
    assert (plain-conditioned).abs().max()>1e-7
    result=model(history,task,[scene],source=source,straight_through=True)
    losses=model.native_losses(result,source)
    assert torch.isfinite(losses['native_total'])
    losses['native_total'].backward()
    assert model.sdf_pose.output.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.pose.parameters())


def test_independent_episode_lengths_and_sampler_resume():
    from motionbricks.data.joint_sdf_training import IndependentEpisodes,episode_segments
    _,model,source,_,_,scene=make_pair()
    class Data:
        def __len__(self):return 2
        def __getitem__(self,request):
            index,draw=request;length=(65,129)[index]
            motion=source[0,:1].repeat(length,1)
            motion[:,0]=torch.arange(length)*.005
            return {'world':motion,'scene':scene,'row':{'origin':'parc_test','control_category':None,'terrain_family':'stairs'}}
    data=Data();rep=model.inference.global_motion_rep
    first=IndependentEpisodes(data,[.5,.5],2,rep,'cpu',29)
    first.slots=[{'index':i,'draw':i,'start':0,'position':0,'segments':episode_segments((65,129)[i]),
        'yaw':.7,'item':data[(i,i)],'history':None} for i in range(2)]
    assert [s['segments'] for s in first.slots]==[5,13]
    for _ in range(5):
        values=first.window();first.advance(values[1][:,8:12])
    import copy
    state=copy.deepcopy(first.state_dict())
    second=IndependentEpisodes(data,[.5,.5],2,rep,'cpu',99);second.load_state_dict(state)
    a=first.window();b=second.window()
    assert first.slots[1]['position']==5 and first.slots[1]['draw']==1
    assert first.counts['completed_depth:5']==1
    for index in range(5):torch.testing.assert_close(a[index],b[index],atol=0,rtol=0)
    assert first.rng.bit_generator.state==second.rng.bit_generator.state


def test_sdf_features_are_invariant_to_training_coordinate_augmentation():
    from motionbricks.data.terrain_scene_frame import HorizontalSceneFrame
    _,model,_,history,task,scene=make_pair();rep=model.inference.global_motion_rep
    frame=HorizontalSceneFrame(torch.tensor([1.1]),history[:,0,[0,2]])
    seen=[]
    previous=model._set_sdf
    def capture(*args):
        previous(*args);seen.append(model._sdf_memory.detach().clone())
    model._set_sdf=capture
    with torch.no_grad():
        model(history,task,[scene]);physical=seen[0];seen.clear()
        model(frame.motion(rep,history),frame.task(task),[scene],scene_frame=frame)
    torch.testing.assert_close(physical,seen[0],atol=2e-6,rtol=1e-6)


def test_cruise_intent_uses_real_approach_and_rejects_turning_or_flat_clips():
    from motionbricks.data.joint_sdf_training import approach_intent
    hf=torch.zeros(61,61);hf[40:]=.3
    scene={'heightfield_y_m':hf,'min_point_world_xyz_m':[-2.,0.,-2.],'dx_m':.1}
    world=torch.zeros(85,5);world[:,2]=torch.arange(85)*.04;world[:,1]=.9;world[:,3]=1.
    item={'world':world,'scene':scene,'row':{'origin':'parc_official'}}
    torch.testing.assert_close(approach_intent(item),torch.tensor([0.,1.2,1.,0.]),atol=1e-6,rtol=0)
    bent=world.clone();bent[40:,0]=.5
    assert approach_intent(dict(item,world=bent)) is None
    flat=dict(scene,heightfield_y_m=torch.zeros_like(hf))
    assert approach_intent(dict(item,scene=flat)) is None


def test_live_backend_default_options_match_training_forward():
    _,model,_,history,task,scene=make_pair()
    with torch.no_grad():
        model.sdf_root.output.weight.normal_(std=.01)
        model.sdf_pose.output.weight.normal_(std=.01)
        model.sdf_decoder.output.weight.normal_(std=.01)
        expected=model(history,task,[scene])['world']
        actual=model(history,task,[scene],num_tokens=8,inference_steps=4,path_refinements=1,
            query_controls=history.new_tensor([[12.,2.,1.,.4]]))['world']
    torch.testing.assert_close(expected,actual,atol=1e-6,rtol=0)


def test_lateral_reference_admission_rejects_step_edge_and_unknown_space():
    from motionbricks.data.joint_sdf_training import verified_lateral_offsets
    heights=torch.zeros(7,7);heights[2:5]=1.
    scene={'heightfield_y_m':heights,'min_point_world_xyz_m':[-3.,0.,-3.],'dx_m':1.}
    points=torch.tensor([[[0.,1.02,0.],[.2,1.1,.2]]])
    admitted,report=verified_lateral_offsets(points,scene,[0.,.5,2.,10.])
    assert admitted==[0.,.5]
    assert report[2]['all_known'] and report[2]['max_surface_height_change_m']==1.
    assert not report[3]['all_known']


def test_native_codec_auxiliary_uses_its_own_source_boundary_and_leaves_actor_output_unchanged():
    from motionbricks.data.joint_sdf_codec_objective import native_codec_reconstruction
    _,model,source,history,task,scene=make_pair();model.freeze_base()
    model.codec.decoder.requires_grad_(True)
    result=model(history,task,[scene],source=source,straight_through=True)
    actor=result['world'].detach().clone()
    expected=native_codec_reconstruction(model,result)['loss']
    changed=dict(result,target=dict(result['target']))
    changed['target']['executed_history_positions']=result['target']['executed_history_positions']+100.
    actual=native_codec_reconstruction(model,changed)['loss']
    torch.testing.assert_close(expected,actual,atol=0,rtol=0)
    torch.testing.assert_close(actor,result['world'],atol=0,rtol=0)
    actual.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.codec.decoder.parameters())
    assert all(p.grad is None for p in model.root.parameters())
    assert all(p.grad is None for p in model.pose.parameters())
    assert all(p.grad is None for p in model.codec.encoder.parameters())
    assert all(p.grad is None for p in model.codec.quantizer.parameters())


def test_native_root_and_pose_receive_gradients_when_separately_unfrozen():
    for name in ('root','pose'):
        _,model,source,history,task,scene=make_pair();model.freeze_base()
        getattr(model,name).requires_grad_(True)
        result=model(history,task,[scene],source=source,straight_through=True)
        model.native_losses(result,source)['native_total'].backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(model,name).parameters())
        other='pose' if name=='root' else 'root'
        assert all(p.grad is None for p in getattr(model,other).parameters())
        assert all(p.grad is None for p in model.codec.parameters())
