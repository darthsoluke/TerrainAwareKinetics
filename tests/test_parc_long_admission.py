"""Parallel admission partitions sources, preserving the official SDF metric."""

import numpy as np
import pytest
import torch

from scripts.admit_parc_long_route_probe import partition_selection
from motionbricks.data.parc_soma_retarget import official_parc_character,terrain_surface_distance_zup


def test_parallel_partitions_cover_each_whole_motion_exactly_once():
    selected=[{'behavior':str(i)} for i in range(19)]
    parts=partition_selection(selected,8)
    assert len(parts)==8
    assert sorted(int(row['behavior']) for part in parts for row in part)==list(range(19))
    assert max(map(len,parts))-min(map(len,parts))<=1
    assert partition_selection(selected[:2],8)==[[selected[0]],[selected[1]]]
    with pytest.raises(ValueError):partition_selection([selected[0],selected[0]],2)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA backend required')
def test_official_sdf_cpu_cuda_parity_on_steps_and_outside_points():
    from pathlib import Path
    official_parc_character(Path(__file__).resolve().parents[1]/'out/vendor/PARC')
    hf=np.zeros((17,13),dtype=np.float32);hf[5:10,4:9]=.8;hf[10:,8:]=-.4
    points=np.random.default_rng(23).uniform([-.1,-.1,-.5],[3.3,2.5,1.2],(64,3))
    cpu,known=terrain_surface_distance_zup(points,hf,np.zeros(2),.2,chunk_size=16)
    gpu,gpu_known=terrain_surface_distance_zup(points,hf,np.zeros(2),.2,chunk_size=16,device='cuda:0')
    np.testing.assert_array_equal(known,gpu_known)
    np.testing.assert_allclose(cpu,gpu,atol=1e-6,rtol=1e-6,equal_nan=True)
