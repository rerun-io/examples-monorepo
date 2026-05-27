from __future__ import annotations

import subprocess
import sys
from typing import Any, cast

import pytest
import torch
from jaxtyping import Float
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


def test_static_tensorrt_runner_reuses_padded_input_buffer() -> None:
    from wilor_nano.api.tensorrt_runtime import _StaticTensorRtRunner

    runner: _StaticTensorRtRunner = _StaticTensorRtRunner.__new__(_StaticTensorRtRunner)
    runner._static_batch_size = 4
    runner._static_input_shape = (3, 8, 8)
    runner._input_buffer = None

    first: Float[Tensor, "batch=2 channels=3 height=8 width=8"] = torch.ones((2, 3, 8, 8), dtype=torch.float32)
    second: Float[Tensor, "batch=3 channels=3 height=8 width=8"] = 2.0 * torch.ones((3, 3, 8, 8), dtype=torch.float32)

    first_padded: Tensor = runner._prepare_input(first)
    second_padded: Tensor = runner._prepare_input(second)

    assert first_padded.shape == (4, 3, 8, 8)
    assert second_padded.shape == (4, 3, 8, 8)
    assert first_padded is second_padded
    torch.testing.assert_close(second_padded[:3], second)
    torch.testing.assert_close(second_padded[3:], torch.zeros((1, 3, 8, 8)))


def test_static_tensorrt_runner_rejects_oversized_batches() -> None:
    from wilor_nano.api.tensorrt_runtime import _StaticTensorRtRunner

    runner: _StaticTensorRtRunner = _StaticTensorRtRunner.__new__(_StaticTensorRtRunner)
    runner._static_batch_size = 1
    runner._static_input_shape = (3, 8, 8)
    runner._input_buffer = None

    with pytest.raises(ValueError, match="Static TensorRT batch"):
        runner._prepare_input(torch.zeros((2, 3, 8, 8)))


def test_full_wilor_runner_requires_cuda_inputs() -> None:
    from wilor_nano.api.tensorrt_runtime import TensorRtFullWilorRunner

    class FakeRunner:
        def run(self, inputs: Tensor) -> tuple[int, dict[str, Tensor]]:
            return int(inputs.shape[0]), {
                "global_orient": torch.zeros((4, 1, 3), dtype=torch.float16),
                "hand_pose": torch.zeros((4, 15, 3), dtype=torch.float16),
                "betas": torch.zeros((4, 10), dtype=torch.float16),
                "pred_cam": torch.arange(12, dtype=torch.float16).reshape(4, 3),
                "pred_keypoints_3d": torch.zeros((4, 21, 3), dtype=torch.float16),
                "pred_vertices": torch.zeros((4, 778, 3), dtype=torch.float16),
            }

    runner: TensorRtFullWilorRunner = TensorRtFullWilorRunner.__new__(TensorRtFullWilorRunner)
    runner._runner = cast(Any, FakeRunner())
    inputs: Tensor = torch.empty((2, 256, 256, 3), dtype=torch.float16)

    with pytest.raises(ValueError, match="CUDA float16"):
        runner(inputs)
