"""Diagnose optional acceleration without admitting it to the live endpoint."""
import argparse
from pathlib import Path

import torch

from motionbricks.data.soma_terrain_execution import TerrainExecution
from scripts.soma_native_bank_server import NativeBankRuntime
from scripts.train_joint_terrain_v3 import write_json


@torch.inference_mode()
def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    runtime = NativeBankRuntime(a.bundle, a.artifact, entry='up')
    runner = TerrainExecution(runtime.inference, runtime.terrain_scene, runtime.skinner,
        runtime.terrain_initial_history, controller=runtime.unified_controller,
        independent_validation_seeds=True)
    write_json(a.output / 'complete.json', dict(validation=runner.validation, live_admitted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'artifact', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        write_json(args.output / 'failure.json', dict(error=repr(error), live_admitted=False))
        raise
