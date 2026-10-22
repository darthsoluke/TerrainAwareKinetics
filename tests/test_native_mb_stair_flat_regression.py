from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.review_native_mb_stair_flat_regression import _run_hashes, summarize_metrics


def row(shard, behavior, start, baseline, child):
    return {"shard": shard, "behavior": behavior, "start": start,
            "metrics": {
                "source": {"scores": {"free_interior": {"mpjpe_m": 0.0}}},
                "baseline": {"scores": {"free_interior": {"mpjpe_m": baseline,
                                                              "root_error_m": None}}},
                "child": {"scores": {"free_interior": {"mpjpe_m": child,
                                                           "root_error_m": None}}},
            }}


def test_summary_aligns_arms_and_computes_child_minus_baseline():
    result = summarize_metrics([
        row("walk.npz", "walk", 0, 0.2, 0.3),
        row("walk.npz", "walk", 10, 0.4, 0.7),
    ])
    walk = result["walk"]
    assert walk["windows"] == 2
    assert walk["metrics"]["baseline"]["scores.free_interior.mpjpe_m"] == pytest.approx(0.3)
    assert walk["metrics"]["child"]["scores.free_interior.mpjpe_m"] == pytest.approx(0.5)
    assert walk["child_minus_baseline"]["scores.free_interior.mpjpe_m"] == pytest.approx(0.2)


def test_summary_rejects_duplicate_window_identity():
    with pytest.raises(ValueError, match="unique"):
        summarize_metrics([row("walk.npz", "walk", 0, 0.2, 0.3),
                           row("walk.npz", "walk", 0, 0.4, 0.5)])


def test_summary_keeps_null_when_metric_is_unavailable():
    result = summarize_metrics([row("run.npz", "run", 0, 0.2, 0.3)])
    assert result["run"]["metrics"]["baseline"]["scores.free_interior.root_error_m"] is None
    assert result["run"]["child_minus_baseline"]["scores.free_interior.root_error_m"] is None


def test_summary_is_order_independent():
    first = row("a.npz", "walk", 0, 1.0, 2.0)
    second = row("b.npz", "walk", 5, 3.0, 4.0)
    forward = summarize_metrics([first, second])
    reverse = summarize_metrics([second, first])
    assert forward == reverse
    assert forward["walk"]["metrics"]["baseline"]["scores.free_interior.mpjpe_m"] == 2.0
    assert forward["walk"]["child_minus_baseline"]["scores.free_interior.mpjpe_m"] == 1.0


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_measurements_are_not_presented_as_unknown(value):
    with pytest.raises(ValueError, match="Nonfinite measured metric"):
        summarize_metrics([row("walk.npz", "walk", 0, value, 1.0)])


def test_run_hashes_resolves_its_file_hash_helper(tmp_path):
    names = ("run_contract.json", "native_config.yaml", "training_result.json",
             "checkpoints/final.ckpt")
    (tmp_path / "checkpoints").mkdir()
    for name in names:
        (tmp_path / name).write_bytes(b"fixture")
    hashes = _run_hashes(tmp_path)
    assert set(hashes) == set(names)
    assert len(set(hashes.values())) == 1
    assert all(len(value) == 64 for value in hashes.values())


def test_cli_prints_serializable_completion(monkeypatch, capsys, tmp_path):
    import json
    from scripts import review_native_mb_stair_flat_regression as reviewer

    monkeypatch.setattr(reviewer, "review", lambda *args: {
        "status": "completed_comparative_review", "windows": [{}] * 6})
    arguments = []
    for name in ("source", "dataset", "baseline-root-run", "baseline-pose-run",
                 "root-run", "pose-run", "output"):
        arguments.extend(["--" + name, str(tmp_path / name)])
    reviewer.main(arguments + ["--gpu", "0"])
    output = json.loads(capsys.readouterr().out)
    assert output["windows"] == 6
    assert output["status"] == "completed_comparative_review"
