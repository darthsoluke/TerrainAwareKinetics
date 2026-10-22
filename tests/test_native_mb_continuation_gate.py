"""A milestone gate must never target the native training process or another job."""

from pathlib import Path

import pytest

from scripts.gate_native_mb_continuation import verify_wrapper_identity


def _identity():
    base = Path("/runs/current")
    state = {"stage": "root", "status": "running", "target_step": 100000,
             "pid": 100, "training_pid": 101}
    command = ["python", "-u", "scripts/continue_native_mb_control.py", "--stage", "root",
               "--output-parent", str(base), "--gpu", "0"]
    return state, command, base


def test_gate_identifies_only_the_wrapper_not_its_training_child():
    state, command, base = _identity()
    assert verify_wrapper_identity(state, command, "root", base) == 100


@pytest.mark.parametrize("defect", ["training_child", "other_run", "other_stage", "already_220k", "same_pid"])
def test_gate_refuses_wrong_process_or_milestone(defect):
    state, command, base = _identity()
    if defect == "training_child":
        command[2] = "scripts/train_native_mb_control.py"
    elif defect == "other_run":
        command[command.index("--output-parent") + 1] = "/other-run"
    elif defect == "other_stage":
        command[command.index("--stage") + 1] = "pose"
    elif defect == "already_220k":
        state["target_step"] = 220000
    else:
        state["pid"] = state["training_pid"]
    with pytest.raises(ValueError):
        verify_wrapper_identity(state, command, "root", base)
