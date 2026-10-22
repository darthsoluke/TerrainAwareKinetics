"""Task-level counterfactual braking measurements, with unchanged model output."""

import numpy as np


def braking_kinematics(roots,history_roots,control,clearance=None,fps=30.):
    roots=np.asarray(roots,dtype=np.float64);history=np.asarray(history_roots,dtype=np.float64)
    control=np.asarray(control,dtype=np.float64)
    if roots.ndim!=2 or roots.shape[1]!=3 or history.shape!=(4,3):raise ValueError('Root histories and rollout must be XYZ')
    speed=float(np.linalg.norm(control[:2]))
    if speed<=0:raise ValueError('Counterfactual braking requires a sustained nonzero command')
    direction=control[:2]/speed
    combined=np.concatenate((history,roots),axis=0)
    velocity=np.diff(combined[:,[0,2]],axis=0)*fps
    requested=velocity@direction
    forward=requested[3:]
    acceleration=np.diff(velocity,axis=0)*fps
    jerk=np.diff(acceleration,axis=0)*fps
    progress=float((roots[-1,[0,2]]-history[-1,[0,2]])@direction)
    initial=max(0.,float(requested[:3].mean()))
    onset=None
    if initial>.15:
        below=forward<.8*initial
        for i in range(len(below)-2):
            if below[i:i+3].all():onset=i;break
    measured={'desired_speed_mps':speed,'initial_speed_mps':initial,'net_forward_progress_m':progress,
        'forward_progress_ratio':progress/(len(roots)/fps*speed),
        'final_forward_speed_mps':float(forward[-int(fps):].mean()),
        'final_speed_mps':float(np.linalg.norm(velocity[-int(fps):],axis=-1).mean()),
        'max_speed_jump_mps':float(np.linalg.norm(np.diff(velocity,axis=0),axis=-1).max()),
        'acceleration_p95_mps2':float(np.quantile(np.linalg.norm(acceleration,axis=-1),.95)),
        'jerk_p95_mps3':float(np.quantile(np.linalg.norm(jerk,axis=-1),.95)),
        'braking_onset_seconds':None if onset is None else onset/fps,
        'braking_onset_clearance_m':None if onset is None or clearance is None else float(clearance[onset]),
        'precontact_braking':bool(onset is not None and clearance is not None and clearance[onset]>.05)}
    return measured


def paired_braking_checks(wall,clear,source):
    checks={
        'open_ground_progress':clear['forward_progress_ratio']>=.5,
        'open_ground_continues':clear['final_forward_speed_mps']>=.5*clear['desired_speed_mps'],
        'obstacle_specific_slowdown':clear['final_forward_speed_mps']-wall['final_forward_speed_mps']>=.25*clear['desired_speed_mps'],
        'wall_stops':wall['final_speed_mps']<=.15,
        'brakes_before_contact':wall['precontact_braking'],
        'smooth_stop':wall['max_speed_jump_mps']<=max(.10,2*source['max_speed_jump_mps']),
        'full_surface_known':wall['unknown_body_fraction']==0 and clear['unknown_body_fraction']==0,
        'wall_penetration':wall['body_penetration_p99_m']<=.03 and wall['body_penetration_max_m']<=.10,
        'open_penetration':clear['body_penetration_p99_m']<=.03 and clear['body_penetration_max_m']<=.10,
    }
    return checks
