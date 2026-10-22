import torch
from scripts.review_position_joystick import planar_waypoints,joystick


def test_planar_driver_does_not_use_future_height_or_pose():
    source=torch.zeros(65,15);source[:,0]=torch.arange(65)*.05;source[:,4]=1
    first,ids=planar_waypoints(source)
    source[4:,1]=100;source[4:,3:]=torch.randn_like(source[4:,3:])*10
    second,other=planar_waypoints(source)
    assert ids==other
    torch.testing.assert_close(first,second,atol=0,rtol=0)
    history=source[None,:4].clone();before=history.clone()
    controls,index,reached=joystick(history,first,1,2.,torch.tensor(0.))
    torch.testing.assert_close(history,before,atol=0,rtol=0)
    assert controls.shape==(1,4) and index==1 and not reached


def test_position_driver_does_not_skip_an_unvisited_waypoint():
    history=torch.zeros(1,4,5);history[0,-1,0]=10
    points=torch.tensor([[0.,0.],[1.,0.],[10.,0.]])
    controls,index,reached=joystick(history,points,1,1.,torch.tensor(0.))
    assert index==1 and not reached and controls[0,0]<0
