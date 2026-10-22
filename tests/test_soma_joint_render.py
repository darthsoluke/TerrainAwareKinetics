import json

import pytest

from scripts.render_bones_control import joint_worst_angular_event


def write_metrics(folder):
    for arm in ("Original", "A", "B"):
        rollouts = {str(history): {"worst_angular_step": {
            "destination_frame": 1512, "joint": "Hips", "deg": 30.}}
            for history in range(3)}
        if arm == "B":
            rollouts["2"]["worst_angular_step"]["deg"] = 173.
        rollouts["delay_12"] = {"worst_angular_step": {
            "destination_frame": 4, "joint": "Head", "deg": 179.}}
        path = folder / f"review_{arm}" / "metrics.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"rollouts": rollouts}))


def test_worst_excerpt_uses_all_histories_and_excludes_legacy_delay(tmp_path):
    write_metrics(tmp_path)
    assert joint_worst_angular_event(tmp_path) == {
        "arm": "B", "history": 2, "destination_frame": 1512,
        "joint": "Hips", "deg": 173.}


def test_missing_history_fails_instead_of_silently_selecting_first(tmp_path):
    write_metrics(tmp_path)
    path = tmp_path / "review_B" / "metrics.json"
    metrics = json.loads(path.read_text())
    del metrics["rollouts"]["2"]
    path.write_text(json.dumps(metrics))
    with pytest.raises(KeyError):
        joint_worst_angular_event(tmp_path)


def test_skinned_batch_loop_cannot_overwrite_source_command_range(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from scripts import render_bones_control as render
    from scripts.evaluate_soma_decoder_refinement import command_at
    from motionbricks.data import bones_seed, sole_geometry_losses, sole_region_contact
    class Rep:
        skeleton = None
        def inverse(self, features, *args, **kwargs):
            return {"local_joint_rots": torch.eye(3).expand(len(features), 30, 3, 3),
                    "posed_joints": torch.zeros(len(features), 30, 3)}
    monkeypatch.setattr(render, "load_official_skinner", lambda s: None)
    monkeypatch.setattr(bones_seed, "official_sole_surfaces", lambda s: (None, None, None, None))
    monkeypatch.setattr(sole_region_contact, "sole_region_indices", lambda s, side: [0])
    monkeypatch.setattr(sole_geometry_losses, "region_geometry_from_motion", lambda r, p, s, i:
                        (torch.zeros(len(r), 2), torch.zeros(len(r), 2, 2)))
    source = tmp_path / "source.npz"
    np.savez(source, global_features=np.zeros((1800, 366), dtype=np.float32))
    render.native_trial_inputs([("raw", source)], tmp_path / "render", Rep(), (1400, 1700))
    with np.load(tmp_path / "render/rollout_raw.npz") as data:
        assert len(data["command_velocity_xz"]) == len(data["joints"]) == 300
        np.testing.assert_array_equal(data["command_velocity_xz"], [command_at(f)[1] for f in range(1400, 1700)])
