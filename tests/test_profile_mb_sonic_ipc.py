"""Standalone cached-data checks; no subprocess, model, physics or GPU startup."""

from pathlib import Path
import pickle

import numpy as np
import pytest

from scripts.profile_mb_sonic_ipc import (
    arrays, assert_equal, instrumented_restore_cost, load_cache, parse_args, production,
)


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "out/mb_sonic_e0_e3_20261018/wasd_v16/episode_0001_numeric/7313379dab02f2b45fba714560891b194ce4e3f055f8e9645bf50d82d693280b.npz"
RECORDING = ROOT / "out/mb_sonic_e0_e3_20261018/wasd_v16/episode_0001_000.jsonl"
SHA256 = "3fb19231ae8d3ec6e1ffc667376672a18696ac7084323e0fb8af4e5b1d43ee3e"


@pytest.fixture(scope="module")
def cache():
    return load_cache(RECORD, RECORDING, SHA256)


def test_actual_cached_update_preserves_every_array_and_metadata_through_production_restore(cache):
    response, provenance = cache
    restored = production.restore_ipc_value(pickle.loads(pickle.dumps(response, protocol=5)))
    assert_equal(response, restored)
    assert provenance["recorded_array_count"] == 28
    assert len(list(arrays(response))) >= 28
    assert len(response["value"].proposal.raw_blocks) == 2
    assert response["value"].proposal.source.scene_id == response["value"].command["scene_id"]
    assert restored["value"].packet.clocks_ns == response["value"].packet.clocks_ns


def test_cached_plan_hash_is_checked_before_reconstruction():
    with pytest.raises(ValueError, match="SHA256"):
        load_cache(RECORD, RECORDING, "0" * 64)


def test_restore_attribution_uses_actual_constructors_without_mutating_source(cache):
    response, _ = cache
    source_before = pickle.dumps(response, protocol=5)
    totals = instrumented_restore_cost(pickle.loads(source_before))
    assert totals["array_finite_check_and_owned_copy"]["calls"] >= 28
    assert totals["dataclass_constructor_validation_and_internal_copies"]["calls"] >= 8
    assert source_before == pickle.dumps(response, protocol=5)
    with pytest.raises(ValueError, match="finite"):
        production.restore_ipc_value(np.array([float("nan")]))


@pytest.mark.parametrize("count", [19, 51])
def test_exchange_allocation_is_bounded_and_explicit(count):
    with pytest.raises(SystemExit):
        parse_args(["--record", str(RECORD), "--recording", str(RECORDING), "--sha256", SHA256,
                    "--output", "out/tracking_isolation_v1/ipc_probe.json", "--exchanges", str(count)])
