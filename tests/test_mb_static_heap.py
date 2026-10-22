"""Static model collection policy preserves temporary collection and lifecycle."""

import gc
import weakref

import numpy as np

from motionbricks.mb_sonic.static_heap import StaticHeap


def test_static_heap_keeps_new_cycles_collectable_and_releases_permanent_generation():
    class Cycle:
        pass

    policy = StaticHeap()
    values = np.arange(128, dtype=np.float64)
    original = values.tobytes()
    try:
        report = policy.prepare()
        assert report["frozen_objects"] > 0
        assert report["temporary_collection_enabled"] and gc.isenabled()
        value = Cycle()
        value.self = value
        reference = weakref.ref(value)
        del value
        gc.collect()
        assert reference() is None
        assert values.tobytes() == original
        assert np.dot(values, values) == 690880.
        policy.prepare()
        assert gc.get_freeze_count() > 0
    finally:
        policy.release()
    assert gc.get_freeze_count() == 0
    assert gc.isenabled()
