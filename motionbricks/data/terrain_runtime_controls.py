"""Bounded online experiments; defaults preserve trained observation geometry."""

import math


RUNTIME_DEFAULTS = {'num_tokens': 12, 'inference_steps': 4, 'lookahead_m': -1.,
                    'lateral_halfwidth_m': .6, 'refresh_frames': 8,
                    'lookahead_gain':1.,'lookahead_base_m':.4,'path_refinements':1}

JOINT_RUNTIME_DEFAULTS = {**RUNTIME_DEFAULTS,'num_tokens':8,'lookahead_m':12.,
                          'lateral_halfwidth_m':2.,'refresh_frames':8}


def validate_runtime_controls(values):
    if not isinstance(values, dict) or set(values) != set(RUNTIME_DEFAULTS):
        raise ValueError('Runtime controls must include exactly the declared settings')
    result = dict(values)
    for name, low, high, stride in [('num_tokens',6,16,1),('inference_steps',1,8,1),
                                    ('refresh_frames',8,56,8),('path_refinements',0,2,1)]:
        value = values[name]
        if isinstance(value,bool) or not isinstance(value,int) or not low <= value <= high or value % stride:
            raise ValueError(f'{name} is outside its integer range')
    for name, low, high in [('lookahead_m',.8,20.),('lateral_halfwidth_m',.3,6.),
                            ('lookahead_gain',0.,4.),('lookahead_base_m',0.,10.)]:
        value = values[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError(f'{name} must be finite')
        if not (name == 'lookahead_m' and value == -1.) and not low <= value <= high:
            raise ValueError(f'{name} is outside its metric range')
        result[name] = float(value)
    return result
