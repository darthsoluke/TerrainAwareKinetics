"""Bookkeeping regressions that do not require Isaac Gym or a GPU."""

from types import SimpleNamespace

import pytest

from scripts.run_parc_physics_record import (
    align_terrain_spacing,
    align_engine_origins,
    install_recorder_compatibility,
    is_complete_recording,
    validate_packed_terrain,
)


def test_engine_origins_update_shared_dm_view_without_grid_guess():
    import torch

    offsets = torch.tensor([[0., 0., 0.], [0., 4., 0.], [0., 8., 0.]])
    origins = [[0., 0., 0.], [4., 0., 0.], [0., 4., 0.]]
    dm = SimpleNamespace(_env_offsets=offsets[:])
    env = SimpleNamespace(
        _envs=list(range(3)), _env_offsets=offsets,
        _gym=SimpleNamespace(get_env_origin=lambda i: SimpleNamespace(
            **dict(zip(('x', 'y', 'z'), origins[i])))),
        get_dm_env=lambda: dm, _get_dm_slice=lambda data: data[:])
    pointer = offsets.data_ptr()
    evidence = align_engine_origins(env)
    assert evidence['before']['max_error_m'] == 4.0
    assert evidence['after']['max_error_m'] == 0.0
    assert offsets.data_ptr() == dm._env_offsets.data_ptr() == pointer
    torch.testing.assert_close(dm._env_offsets, torch.tensor(origins))
    assert align_engine_origins(env)['before']['max_error_m'] == 0.0
    dm._env_offsets = offsets.clone()
    with pytest.raises(ValueError, match='share'):
        align_engine_origins(env)


def test_legacy_library_alias_preserves_identity():
    library = object()
    dm = SimpleNamespace(_mlib=library)
    report = install_recorder_compatibility(dm)
    assert dm._motion_lib is dm._mlib is library
    assert report["same_object"] is True
    install_recorder_compatibility(dm)


def test_conflicting_library_is_not_silently_replaced():
    dm = SimpleNamespace(_mlib=object(), _motion_lib=object())
    with pytest.raises(ValueError, match="Conflicting"):
        install_recorder_compatibility(dm)


@pytest.mark.parametrize("fraction,count,source,complete", [
    (0.0, 408, 408, True), (0.1, 367, 408, False),
    (0.0, 407, 408, False), (0.5, 408, 408, False),
])
def test_complete_recording_needs_original_start_and_frame_count(
        fraction, count, source, complete):
    assert is_complete_recording(fraction, count, source) is complete


def test_spacing_follows_source_without_changing_policy_rays():
    config = {"env": {"dm": {"heightmap": {"horizontal_scale": .4, "padding": .4}},
                      "ray_dx": .05, "control_freq": 30}}
    evidence = align_terrain_spacing(config, [.2, .2])
    assert config["env"]["dm"]["heightmap"]["horizontal_scale"] == .2
    assert config["env"]["ray_dx"] == .05
    assert config["env"]["control_freq"] == 30
    assert evidence["archived_grid_spacing_m"] == .4


@pytest.mark.parametrize('spacing,count',[(.4000000059604645,1),(.20000000298023224,2)])
def test_float32_grid_spacing_preserves_intended_padding_cell_count(spacing,count):
    config={'env':{'dm':{'heightmap':{'horizontal_scale':.4,'padding':.4}}}}
    result=align_terrain_spacing(config,[spacing])
    heightmap=config['env']['dm']['heightmap']
    assert int(heightmap['padding']/heightmap['horizontal_scale'])==count
    assert result['padding_cells']==count
    assert heightmap['padding']==pytest.approx(.4,abs=1e-7)


@pytest.mark.parametrize("spacings", [[.2, .4], [], [float('nan')], [-.2]])
def test_invalid_or_mixed_grid_rejected(spacings):
    with pytest.raises(ValueError):
        align_terrain_spacing({"env": {"dm": {"heightmap": {
            "horizontal_scale": .4, "padding": .4}}}}, spacings)


def test_packed_cells_match_source_in_world_coordinates():
    import torch

    hf = torch.tensor([[0., .2], [.4, .6]])
    tile = SimpleNamespace(hf=hf, dxdy=torch.tensor([.2, .2]), min_point=torch.zeros(2))
    world = SimpleNamespace(dxdy=tile.dxdy.clone(), get_hf_val_from_points=lambda p: hf.clone())
    dm = SimpleNamespace(_mlib=SimpleNamespace(_terrains=[tile]), _terrain=world,
                         _dm_motion_offsets=torch.zeros((1, 1, 2)))
    assert validate_packed_terrain(dm)["all_tile_cells_checked"] == 4
    world.dxdy *= 2
    with pytest.raises(ValueError, match="spacing differ"):
        validate_packed_terrain(dm)
    world.dxdy = tile.dxdy.clone()
    world.get_hf_val_from_points = lambda p: torch.zeros_like(hf)
    with pytest.raises(ValueError, match="differs from collision/source"):
        validate_packed_terrain(dm)
