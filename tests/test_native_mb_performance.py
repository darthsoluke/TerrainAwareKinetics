"""Performance probes must not silently change the native training method."""

from scripts.profile_native_mb_control import profile_arguments, summarize_durations


def test_profile_preserves_batch_seed_schedule_and_frozen_tokenizer():
    contract = {"stage": "native_pose", "dataset": "/data", "batch_size": 32,
                "seed": 7, "original_schedule_steps": 2000001, "tokenizer": {"run": "/tokenizer"}}
    arguments = profile_arguments(contract, 12000, "/resume", "/output", 4, 8, 200)
    for name, value in {"--steps": "12200", "--batch-size": "32", "--seed": "7",
                        "--schedule-steps": "2000001", "--tokenizer-run": "/tokenizer",
                        "--workers": "8"}.items():
        assert arguments[arguments.index(name) + 1] == value
    assert "--skip-review" in arguments


def test_profile_excludes_warmup_only_when_enough_measurements_exist():
    summary = summarize_durations({"step": [.5, .5, .1, .1], "setup": [1.0], "empty": []}, warmup=2)
    assert summary["step"]["steady_mean_ms"] == 100
    assert summary["step"]["steady_calls"] == 2
    assert summary["setup"]["steady_mean_ms"] == 1000
    assert "empty" not in summary
