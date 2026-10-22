import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def probe():
    spec=importlib.util.spec_from_file_location('parc_readiness_probe',
        Path(__file__).parents[1]/'scripts/probe_parc_long_route_physics.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_waits_for_both_original_memory_and_utilization_limits(probe,monkeypatch,tmp_path):
    responses=iter(['GPU-test, 19, 32\n','GPU-test, 3000, 0\n','GPU-test, 19, 0\n'])
    waits=[]
    monkeypatch.setattr(probe.subprocess,'check_output',lambda *a,**k:next(responses))
    monkeypatch.setattr(probe.time,'sleep',waits.append)
    assert probe.wait_for_idle_gpu(5,tmp_path/'readiness.json',attempts=4)=='GPU-test'
    report=json.loads((tmp_path/'readiness.json').read_text())
    assert waits==[3,3]
    assert [row['ready'] for row in report['observations']]==[False,False,True]
    assert report['limits']==dict(memory_mib=1024,utilization_percent=5)


def test_never_accepts_a_persistently_occupied_gpu(probe,monkeypatch,tmp_path):
    waits=[]
    monkeypatch.setattr(probe.subprocess,'check_output',lambda *a,**k:'GPU-test, 1025, 0\n')
    monkeypatch.setattr(probe.time,'sleep',waits.append)
    with pytest.raises(RuntimeError,match='occupied'):
        probe.wait_for_idle_gpu(5,tmp_path/'readiness.json',attempts=3)
    report=json.loads((tmp_path/'readiness.json').read_text())
    assert report['ready'] is False and len(report['observations'])==3
    assert waits==[3,3]
