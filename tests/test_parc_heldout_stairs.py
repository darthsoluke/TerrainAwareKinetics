import hashlib
import torch

from scripts.build_parc_heldout_stair_course import terrain_arrays
from scripts.build_parc_pose_entry_course import terrain_arrays as training_arrays


def test_heldout_geometry_has_real_approach_both_flights_and_long_exit():
    for index in range(8):
        hf,route,spec=terrain_arrays(index)
        level=hf[route[:,0],route[:,1]]
        changes=torch.diff(level)
        assert int((changes>0).sum())==spec['steps']
        assert int((changes<0).sum())==spec['steps']
        assert level[:60].eq(0).all() and level[-60:].eq(0).all()
        assert spec['split']==('validation' if index<4 else 'test')
        assert all(0<=first<last<len(route) for first,last,_ in spec['stages'])


def test_new_geometry_is_unique_and_disjoint_from_previous_entry_courses():
    digest=lambda value:hashlib.sha256(value.numpy().tobytes()).hexdigest()
    actual={digest(terrain_arrays(index)[0]) for index in range(8)}
    previous={digest(training_arrays(index)[0]) for index in range(8)}
    assert len(actual)==8 and not actual&previous
