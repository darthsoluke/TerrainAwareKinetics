import json

import pytest

from scripts.summarize_soma_supervision import summarize
from test_soma_joint_review import fixture as joint_fixture, write


def fixture(folder, empty=False):
    joint_fixture(folder, empty)
    for arm, name in (("A", "legacy_splice"), ("B", "source_v2")):
        contract = json.loads((folder / f"full_v2_{arm}/contract.json").read_text())
        contract.update(steps=1000, supervision_policy=name, loss_arguments={
            "joint_vel_loss_coeff": 2., "rollout_rotation_increment_coeff": 0. if arm == "A" else 1.})
        result = json.loads((folder / f"full_v2_{arm}/result.json").read_text())
        result.update(steps=1000, root_changed=True)
        write(folder / name / "contract.json", contract)
        write(folder / name / "result.json", result)
        write(folder / f"review_{name}/metrics.json", json.loads((folder / f"review_{arm}/metrics.json").read_text()))
    write(folder / "review_original/metrics.json", json.loads((folder / "review_Original/metrics.json").read_text()))


def test_matched_supervision_report_never_automatically_approves(tmp_path):
    fixture(tmp_path)
    result = summarize(tmp_path)
    assert result["verified_matched"] and result["all_nonregression_gates"]
    assert not result["quality_approved"]
    assert (tmp_path / "REPORT.md").is_file()


@pytest.mark.parametrize("defect", ["sampling", "steps", "policy", "decoder", "masks", "initial"])
def test_unmatched_evidence_is_rejected(tmp_path, defect):
    fixture(tmp_path)
    name = ("source_v2/result.json" if defect in ("sampling", "steps") else
            "source_v2/contract.json" if defect == "policy" else "review_source_v2/metrics.json")
    if defect == "initial":
        name = "review_original/metrics.json"
    path = tmp_path / name
    data = json.loads(path.read_text())
    if defect == "sampling": data["sampling_sha256"] = "different"
    elif defect == "steps": data["steps"] = 999
    elif defect == "policy": data["supervision_policy"] = "legacy_splice"
    elif defect == "decoder": data["tokenizer_state"] = "changed"
    elif defect == "masks": data["windows"][0]["native_root"]["predicted_contact"]["valid_samples"]["frames"] = 9
    else: data["bundle_sha256"] = "different"
    write(path, data)
    with pytest.raises(ValueError):
        summarize(tmp_path)


def test_unknown_contact_metrics_do_not_pass(tmp_path):
    fixture(tmp_path, empty=True)
    result = summarize(tmp_path)
    assert not result["all_nonregression_gates"]
    assert not result["nonregression_vs_original"]["mean_window_measured_slip_mps"]


def decoder_fixture(folder):
    fixture(folder)
    for old, name in (("legacy_splice", "frozen_decoder"), ("source_v2", "train_decoder")):
        trained = name == "train_decoder"
        contract = json.loads((folder / old / "contract.json").read_text())
        contract.update(supervision_policy="source_v2", train_decoder=trained)
        contract["loss_arguments"]["rollout_rotation_increment_coeff"] = 1.
        result = json.loads((folder / old / "result.json").read_text())
        result["decoder_changed"] = trained
        report = json.loads((folder / f"review_{old}/metrics.json").read_text())
        if trained:
            report["frozen_state"]["decoder"] = "trained_decoder"
            report["tokenizer_state"] = "trained_codec"
        write(folder / name / "contract.json", contract)
        write(folder / name / "result.json", result)
        write(folder / f"review_{name}/metrics.json", report)


def test_decoder_comparison_accepts_only_intended_membership_change(tmp_path):
    decoder_fixture(tmp_path)
    result = summarize(tmp_path, decoder_comparison=True)
    assert result["verified_matched"] and result["decoder_comparison"]
    assert not result["quality_approved"]
    assert set(result["metrics"]) == {"original", "frozen_decoder", "train_decoder"}


@pytest.mark.parametrize("defect", ["decoder", "encoder", "quantizer", "flag", "change"])
def test_decoder_comparison_rejects_wrong_trained_or_frozen_modules(tmp_path, defect):
    decoder_fixture(tmp_path)
    path = tmp_path / ("train_decoder/contract.json" if defect == "flag" else
                       "train_decoder/result.json" if defect == "change" else
                       "review_train_decoder/metrics.json")
    data = json.loads(path.read_text())
    if defect == "flag": data["train_decoder"] = False
    elif defect == "change": data["decoder_changed"] = False
    else: data["frozen_state"][defect] = "same" if defect == "decoder" else "changed"
    write(path, data)
    with pytest.raises(ValueError):
        summarize(tmp_path, decoder_comparison=True)


def observation_fixture(folder):
    fixture(folder)
    for old, name in (("legacy_splice", "predicted_features"), ("source_v2", "executed_fk")):
        contract = json.loads((folder / old / "contract.json").read_text())
        contract.update(supervision_policy="source_v2", history_geometry=name, train_decoder=False)
        contract["loss_arguments"]["rollout_rotation_increment_coeff"] = 1.
        result = json.loads((folder / old / "result.json").read_text())
        result["decoder_changed"] = False
        report = json.loads((folder / f"review_{old}/metrics.json").read_text())
        report["history_geometry"] = name
        write(folder / name / "contract.json", contract)
        write(folder / name / "result.json", result)
        write(folder / f"review_{name}/metrics.json", report)
    path = folder / "review_original/metrics.json"
    report = json.loads(path.read_text())
    report["history_geometry"] = "predicted_features"
    write(path, report)


def test_observation_comparison_checks_both_training_and_inference(tmp_path):
    observation_fixture(tmp_path)
    result = summarize(tmp_path, observation_comparison=True)
    assert result["verified_matched"] and not result["quality_approved"]
    assert result["observation_comparison"]
    path = tmp_path / "review_executed_fk/metrics.json"
    data = json.loads(path.read_text())
    data["history_geometry"] = "predicted_features"
    write(path, data)
    with pytest.raises(ValueError, match="input policy"):
        summarize(tmp_path, observation_comparison=True)
