"""Read-only package inventory; importing Torch does not allocate a CUDA model."""

import importlib.metadata
import importlib.util
import json
import sys


PACKAGES = {
    'torch': 'torch', 'numpy': 'numpy', 'scipy': 'scipy', 'hydra-core': 'hydra',
    'omegaconf': 'omegaconf', 'pytorch-lightning': 'pytorch_lightning',
    'transformers': 'transformers', 'vector-quantize-pytorch': 'vector_quantize_pytorch',
    'adam-atan2-pytorch': 'adam_atan2_pytorch', 'matplotlib': 'matplotlib',
    'colorlog': 'colorlog', 'mujoco': 'mujoco', 'mink': 'mink', 'qpsolvers': 'qpsolvers',
}


def probe():
    installed = {}
    for distribution, module in PACKAGES.items():
        if importlib.util.find_spec(module) is None:
            installed[distribution] = None
        else:
            try:
                installed[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                installed[distribution] = 'module_present_distribution_unrecorded'
    return {'python': sys.version, 'executable': sys.executable,
            'python_supported': sys.version_info >= (3, 10), 'packages': installed}


if __name__ == '__main__':
    print(json.dumps(probe(), indent=2))
