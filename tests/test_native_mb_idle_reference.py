import numpy as np
import json
import pytest
import torch

from motionbricks.data.native_mb_data import build_native_representation
from scripts.author_native_mb_idle_reference import candidate_starts
from motionbricks.data.native_mb_frame import load_authored_idle_reference
from motionbricks.data.soma_replay import skeleton_manifest


def fixture():
    rep = build_native_representation()
    values = np.zeros((40, len(rep.indices['all'])), dtype=np.float32)
    values[:, 1] = 1.
    for side in ('Left', 'Right'):
        for part, height in (('Arm', 1.4), ('Hand', .9)):
            index = rep.skeleton.bone_order_names.index(side + part) - 1
            values[:, rep.indices['ric_data'][index * 3 + 1]] = height
    return rep, values


def test_stationary_calibration_is_not_a_relaxed_idle():
    rep, values = fixture()
    assert candidate_starts(values, rep)
    for side in ('Left', 'Right'):
        hand = rep.skeleton.bone_order_names.index(side + 'Hand') - 1
        values[:, rep.indices['ric_data'][hand * 3 + 1]] = 1.45
    assert candidate_starts(values, rep) == []


def test_translation_and_body_motion_are_not_stationary_references():
    rep, values = fixture()
    values[:, 0] = np.arange(40) / 30.
    assert candidate_starts(values, rep) == []
    assert candidate_starts(values[:4], rep) == []


def test_reference_loader_checks_identity_and_never_edits_observed_features(tmp_path):
    rep, values = fixture()
    artifact = dict(schema='motionbricks.authored_idle.v1', fps=30,
        source_id=skeleton_manifest()['source_id'], provenance={'split': 'train'}, features=values[:4].tolist())
    path = tmp_path / 'reference.json'
    path.write_text(json.dumps(artifact))
    result = load_authored_idle_reference(path, rep, 'cpu')
    torch.testing.assert_close(result[0], torch.from_numpy(values[:4]), atol=0., rtol=0.)
    artifact['fps'] = 60
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match='identity'):
        load_authored_idle_reference(path, rep, 'cpu')
