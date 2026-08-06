from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from torch import Tensor


def test_tensorrt_runtime_import_does_not_load_pytorch_vit_stack() -> None:
    command: str = (
        "import sys; "
        "import wilor_nano.api.tensorrt_runtime; "
        "assert 'wilor_nano.models.vit' not in sys.modules, sorted(sys.modules); "
        "assert 'timm' not in sys.modules, sorted(sys.modules)"
    )
    completed_process: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed_process.returncode == 0, completed_process.stderr


def test_full_wilor_runner_requires_cuda_float16_inputs() -> None:
    from wilor_nano.api.tensorrt_runtime import TensorRtFullWilorRunner

    runner: TensorRtFullWilorRunner = TensorRtFullWilorRunner.__new__(TensorRtFullWilorRunner)
    inputs: Tensor = torch.empty((2, 256, 256, 3), dtype=torch.float16)

    with pytest.raises(ValueError, match="CUDA float16"):
        runner(inputs)


def test_raw_detector_runner_requires_cuda_float32_inputs() -> None:
    from wilor_nano.api.tensorrt_runtime import TensorRtRawDetectorRunner

    runner: TensorRtRawDetectorRunner = TensorRtRawDetectorRunner.__new__(TensorRtRawDetectorRunner)
    inputs: Tensor = torch.empty((2, 3, 512, 416), dtype=torch.float32)

    with pytest.raises(ValueError, match="CUDA float32"):
        runner(inputs)
