import numpy as np

from motionbricks.data.terrain_braking_metrics import braking_kinematics,paired_braking_checks


def test_collision_free_freezing_does_not_pass_counterfactual_braking():
    value={'forward_progress_ratio':0.,'final_forward_speed_mps':0.,'desired_speed_mps':1.,
        'final_speed_mps':0.,'precontact_braking':True,'max_speed_jump_mps':0.,
        'unknown_body_fraction':0.,'body_penetration_p99_m':0.,'body_penetration_max_m':0.}
    checks=paired_braking_checks(value,value,{'max_speed_jump_mps':.1})
    assert not checks['open_ground_progress'] and not checks['obstacle_specific_slowdown']


def test_stop_at_first_predicted_frame_exposes_speed_jump():
    history=np.zeros((4,3));history[:,2]=np.arange(4)/30
    roots=np.repeat(history[-1:],180,axis=0)
    values=braking_kinematics(roots,history,[0.,1.,1.,0.],np.ones(180))
    assert values['max_speed_jump_mps']>.99
    assert values['final_speed_mps']==0
    good={**values,'forward_progress_ratio':1.,'final_forward_speed_mps':1.,
        'unknown_body_fraction':0.,'body_penetration_p99_m':0.,'body_penetration_max_m':0.}
    wall={**good,**values}
    assert not paired_braking_checks(wall,good,{'max_speed_jump_mps':.05})['smooth_stop']


def test_gradual_braking_requires_clearance_at_onset():
    history=np.zeros((4,3));history[:,2]=np.arange(4)/30
    velocities=np.concatenate((np.ones(15),np.linspace(1,0,30),np.zeros(135)))
    roots=np.zeros((180,3));roots[:,2]=history[-1,2]+np.cumsum(velocities)/30
    ahead=braking_kinematics(roots,history,[0.,1.,1.,0.],np.ones(180)*.2)
    inside=braking_kinematics(roots,history,[0.,1.,1.,0.],np.ones(180)*-.1)
    assert ahead['precontact_braking'] and not inside['precontact_braking']
    assert 0.<ahead['braking_onset_seconds']<2.
    assert ahead['max_speed_jump_mps']<.05
