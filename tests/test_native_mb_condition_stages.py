"""Stage-mask diagnosis preserves every output and the enclosing batch."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.trace_native_mb_condition_stages import sparse_stage_inputs


@pytest.mark.parametrize("stage", ["root", "pose"])
def test_only_input_masks_change_and_original_methods_restore_on_error(stage):
    batch = {key: torch.ones(1, 8, dtype=torch.bool) for key in
             ("has_global_root_values", "has_local_root_values", "has_local_poses")}
    batch["has_local_root_values"][:, 3] = False
    batch["local_poses"] = torch.randn(1, 8, 12)
    saved = {key: value.clone() for key, value in batch.items()}
    output, calls = torch.randn(1, 64, 5), []
    def original(inputs, config):
        calls.append(inputs)
        return output
    name = "_predict_root_trajectories" if stage == "root" else "_predict_pose_tokens"
    infer = SimpleNamespace(**{name: original})
    with pytest.raises(RuntimeError, match="intentional"):
        with sparse_stage_inputs(infer, stage) as observed:
            assert getattr(infer, name)(batch, {}) is output
            assert len(observed) == 1
            reduced = calls[-1]
            assert reduced["local_poses"] is batch["local_poses"]
            assert reduced["has_local_poses"].nonzero()[:, 1].tolist() == [3, 7]
            for key in ("has_global_root_values", "has_local_root_values"):
                if stage == "pose":
                    assert reduced[key] is batch[key]
                else:
                    assert reduced[key].nonzero()[:, 1].tolist() == ([2, 7] if "local" in key else [3, 7])
            raise RuntimeError("intentional")
    assert getattr(infer, name) is original
    for key in batch:
        torch.testing.assert_close(batch[key], saved[key], rtol=0, atol=0)
