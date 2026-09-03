"""Public-contract tests for depth-completion model-zoo predictors."""

from collections.abc import Callable

import pytest
import torch
from jaxtyping import Float32, UInt8
from torch import Tensor
from trtkit import RuntimeSpec, TensorSpec

from monopriors.models.depth_completion import PromptDAPredictor
from monopriors.models.depth_completion.base_completion_depth import upsample_depth_to_image


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
        depth_bchw: Float32[Tensor, "b 1 2 4"] = torch.full(
            (batch_size, 1, 2, 4), 1.25, dtype=torch.float32, device=inputs["image"].device
        )
        return {"depth": depth_bchw}


def test_upsample_depth_to_image_preserves_sampling_and_ownership_policy() -> None:
    """Depth upsampling uses the fixed bilinear policy and always returns owned storage."""
    depth_bchw: Float32[Tensor, "1 1 2 2"] = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
    expected_bhw: Float32[Tensor, "1 4 4"] = torch.tensor(
        [[[1.0, 1.25, 1.75, 2.0], [1.5, 1.75, 2.25, 2.5], [2.5, 2.75, 3.25, 3.5], [3.0, 3.25, 3.75, 4.0]]],
        dtype=torch.float32,
    )

    resized_bhw: Float32[Tensor, "1 4 4"] = upsample_depth_to_image(depth_bchw, (4, 4))
    owned_bhw: Float32[Tensor, "1 2 2"] = upsample_depth_to_image(depth_bchw, (2, 2))

    assert torch.equal(resized_bhw, expected_bhw)
    assert torch.equal(owned_bhw, depth_bchw[:, 0])
    assert owned_bhw.untyped_storage().data_ptr() != depth_bchw.untyped_storage().data_ptr()


@pytest.mark.parametrize("predictor_factory", [PromptDAPredictor])
def test_completion_predictor_owns_layout_scale_and_output_resize(
    predictor_factory: Callable[..., PromptDAPredictor],
) -> None:
    """Every model-zoo predictor exposes the same backend-neutral family contract."""
    runtime = _RecordingRuntime()
    predictor = predictor_factory(runtime=runtime)
    rgb_bhwc: UInt8[Tensor, "1 4 8 3"] = torch.full((1, 4, 8, 3), 255, dtype=torch.uint8)
    prompt_bhw: Float32[Tensor, "1 192 256"] = torch.full((1, 192, 256), 0.75, dtype=torch.float32)

    depth_bhw: Float32[Tensor, "1 4 8"] = predictor(rgb_bhwc, prompt_bhw)

    assert runtime.inputs is not None
    assert runtime.inputs["image"].shape == (1, 3, 2, 4)
    assert runtime.inputs["image"].dtype == torch.float32
    assert torch.all(runtime.inputs["image"] == 1.0)
    assert runtime.inputs["prompt_depth"].shape == (1, 1, 192, 256)
    assert torch.equal(runtime.inputs["prompt_depth"][:, 0], prompt_bhw)
    assert depth_bhw.shape == (1, 4, 8)
    assert torch.all(depth_bhw == 1.25)
