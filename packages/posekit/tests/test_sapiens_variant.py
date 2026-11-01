"""Sapiens2 variant selection is explicit at the public config boundary."""

import pytest

# ruff: noqa: E402

pytest.importorskip("torch")
from posekit.models.sapiens import SapiensPoseConfig
from posekit.runtimes import OnnxBackendConfig, TensorRtBackendConfig, TorchBackendConfig


def test_fast_sapiens_is_default() -> None:
    config = SapiensPoseConfig()
    assert config.variant == "nvfp4-768"
    assert isinstance(config.backend, TensorRtBackendConfig)
    assert (config.backend.opt_batch_size, config.backend.max_batch_size) == (4, 8)


@pytest.mark.parametrize("backend", [TorchBackendConfig(), OnnxBackendConfig()])
def test_fast_sapiens_requires_tensorrt(backend) -> None:
    with pytest.raises(ValueError, match="nvfp4-768.*TensorRT"):
        SapiensPoseConfig(backend=backend)
    assert SapiensPoseConfig(variant="bf16-1024", backend=backend).variant == "bf16-1024"
