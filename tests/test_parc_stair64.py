from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_parc_stair64_course import build_stair64_course
from generate_parc_stair64 import preserve_flags
from generate_parc_long_stairs import route_evidence
from parc_stair_command_schedule import StairCommandSchedule,stop_response_report


def test_eight_distinct_courses_have_real_up_and_down():
    hashes=[]
    for index in range(8):
        terrain,path,meta=build_stair64_course(index)
        hashes.append(meta['terrain_sha256'])
        torch.testing.assert_close(terrain.get_hf_val_from_points(path[:,:2]),path[:,2])
        rise=torch.diff(path[:,2])
        assert (rise>.05).sum()==meta['stair_flights'][0]['riser_count']
        assert (rise<-.05).sum()==meta['stair_flights'][0]['riser_count']
        assert route_evidence(path.repeat_interleave(2,0),path,meta)['all_segments_covered_in_order']
        assert not route_evidence(path[[0,-1]],path,meta)['all_segments_covered_in_order']
        assert path[0,2]==path[-1,2]==0
    assert len(set(hashes))==8


def test_flag_guard_restores_input_even_on_exception():
    conditions={'a':torch.tensor([True,True]),'b':torch.tensor([False,True])}
    before={key:value.clone() for key,value in conditions.items()}
    def predict(x,conditions):
        conditions['a'].fill_(False);conditions['b'].fill_(False)
        return x+1
    torch.testing.assert_close(preserve_flags(predict,('a','b'))(torch.tensor(3.),conditions),torch.tensor(4.))
    for key in before:assert torch.equal(before[key],conditions[key])
    def fail(x,conditions):
        conditions['a'].fill_(False)
        raise RuntimeError('intentional')
    with pytest.raises(RuntimeError,match='intentional'):
        preserve_flags(fail,('a','b'))(torch.tensor(3.),conditions)
    for key in before:assert torch.equal(before[key],conditions[key])


def test_command_stop_keeps_current_history_and_resumes():
    import copy
    terrain,nodes,course=build_stair64_course(0)
    class Frames:
        def __init__(self):self.root_pos=nodes[0].repeat(8,2,1)
        def get_copy(self,device):return copy.deepcopy(self)
    source=Frames();original=source.root_pos.clone()
    scheduler=StairCommandSchedule(nodes,course)
    first=scheduler.transform(dict(prev_frames=source,target_world_pos=nodes[1:2]))
    assert torch.equal(source.root_pos,original)
    assert len(torch.unique(first['prev_frames'].root_pos[:,-1,:2],dim=0))==8
    ascent=next(p for p in course['route_segments'] if p['stage']=='ascent')
    source.root_pos[:,-1]=nodes[ascent['path_node_start']]+.4*(nodes[ascent['path_node_end']]-nodes[ascent['path_node_start']])
    original=source.root_pos.clone()
    for _ in range(12):
        value=scheduler.transform(dict(prev_frames=source,target_world_pos=nodes[-1:]))
        assert torch.equal(source.root_pos,original)
        if scheduler.calls[-1]['labels'][0]=='ascent_stop':
            assert torch.equal(value['target_world_pos'][0,:2],source.root_pos[0,-1,:2])
    assert scheduler.calls[-1]['labels'][0]=='move'


def test_zero_input_does_not_certify_a_stopped_body():
    import numpy as np
    _,nodes,course=build_stair64_course(0)
    phases=[p for p in course['route_segments'] if p['stage'] in ('ascent','descent')]
    middle=[.65*nodes[p['path_node_start']]+.35*nodes[p['path_node_end']] for p in phases]
    def travel(a,b,count):return torch.stack([a+(b-a)*i/(count-1) for i in range(count)])
    root=torch.cat((travel(nodes[0],middle[0],20),middle[0].repeat(60,1),
        travel(middle[0],middle[1],25),middle[1].repeat(60,1),travel(middle[1],nodes[-1],20)))
    commands=np.tile([1.,0.],(len(root),1));commands[20:80]=0;commands[105:165]=0
    result=stop_response_report(SimpleNamespace(root_pos=root),commands,None,nodes,course,30)
    assert result['kinematic_stops_pass']
    root[20:80,0]+=torch.arange(60)*.01
    result=stop_response_report(SimpleNamespace(root_pos=root),commands,None,nodes,course,30)
    assert not result['kinematic_stops_pass']
