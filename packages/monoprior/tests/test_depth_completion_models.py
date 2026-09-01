"""Public-contract tests for depth-completion model-zoo predictors."""

from collections.abc import Callable

import pytest
import torch
from jaxtyping import Float32, UInt8
from torch import Tensor
from trtkit import RuntimeSpec, TensorSpec

from monopriors.models.depth_completion import PromptDAPredictor, ZipDepthPromptPredictor


class _RecordingRuntime:
    """Small tensor-runtime boundary double that records normalized inputs."""

    def __init__(self) -> None:
        self._spec = RuntimeSpec(
            inputs=(
                TensorSpec("image", (3, 2, 4), torch.float32),
                TensorSpec("prompt_depth", (1, 192, 256), torch.float32),
            ),
            outputs=(TensorSpec("depth", (1, 2, 4), torch.float32),),
            max_batch_size=2,
        )
        self.inputs: dict[str, Tensor] | None = None

    @property
    def spec(self) -> RuntimeSpec:
        """Return the fixed runtime contract."""
        return self._spec

    def __call__(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Record inputs and return a constant metric-depth grid."""
        self.inputs = inputs
        batch_size: int = int(inputs["image"].shape[0])
        depth_b1hw: Float32[Tensor, "b 1 2 4"] = torch.full(
            (batch_size, 1, 2, 4), 1.25, dtype=torch.float32, device=inputs["image"].device
        )
        return {"depth": depth_b1hw}


@pytest.mark.parametrize("predictor_factory", [PromptDAPredictor, ZipDepthPromptPredictor])
def test_completion_predictor_owns_layout_scale_and_output_resize(
    predictor_factory: Callable[..., PromptDAPredictor | ZipDepthPromptPredictor],
) -> None:
    """Every model-zoo predictor exposes the same backend-neutral family contract."""
    runtime = _RecordingRuntime()
    predictor = predictor_factory(runtime=runtime)
    rgb_bhw3: UInt8[Tensor, "1 4 8 3"] = torch.full((1, 4, 8, 3), 255, dtype=torch.uint8)
    prompt_bhw: Float32[Tensor, "1 192 256"] = torch.full((1, 192, 256), 0.75, dtype=torch.float32)

    depth_bhw: Float32[Tensor, "1 4 8"] = predictor(rgb_bhw3, prompt_bhw)

    assert runtime.inputs is not None
    assert runtime.inputs["image"].shape == (1, 3, 2, 4)
    assert runtime.inputs["image"].dtype == torch.float32
    assert torch.all(runtime.inputs["image"] == 1.0)
    assert runtime.inputs["prompt_depth"].shape == (1, 1, 192, 256)
    assert torch.equal(runtime.inputs["prompt_depth"][:, 0], prompt_bhw)
    assert depth_bhw.shape == (1, 4, 8)
    assert torch.all(depth_bhw == 1.25)
