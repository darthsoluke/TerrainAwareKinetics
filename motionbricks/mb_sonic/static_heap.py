"""Bound the live collector's scan to transient data, not preloaded model graphs."""

import gc
from time import perf_counter_ns


class StaticHeap:
    def __init__(self):
        self.active = False
        self.preparation = None

    def prepare(self):
        self.release()
        if not gc.isenabled() or gc.get_freeze_count():
            raise RuntimeError("Static heap preparation requires enabled GC and sole process ownership")
        started = perf_counter_ns()
        collected = gc.collect()
        collection_ns = perf_counter_ns()-started
        gc.freeze()
        self.active = True
        self.preparation = dict(strategy="freeze_preloaded_static_heap", frozen_objects=gc.get_freeze_count(),
            preparation_collection_ns=collection_ns, collected_before_freeze=collected,
            temporary_collection_enabled=gc.isenabled(), numerical_changes=False)
        return dict(self.preparation)

    def release(self):
        if self.active:
            gc.unfreeze()
            self.active = False
