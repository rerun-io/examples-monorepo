"""Both tools run the cached TensorRT engine with a batch-1 profile unless told otherwise; ``onnx`` is the fallback."""

import pytest

pytest.importorskip("rerun.catalog", reason="the tool configurations import the Rerun catalog dependencies")
pytest.importorskip("rerun.experimental.dataloader", reason="the tool configurations import the Rerun dataloader")
pytest.importorskip("torchcodec", reason="the tool configurations import TorchCodec")

import tyro  # noqa: E402
from simplecv.rerun_log_utils import RerunTyroConfig  # noqa: E402
from trtkit import OnnxBackendConfig, TensorRtBackendConfig  # noqa: E402

from mvs.apis import catalog_depth, live_mesh  # noqa: E402

TOOL_CONFIGS: list[type] = [live_mesh.Config, catalog_depth.Config]
HEADLESS: list[str] = ["--rr-config.headless"]
"""Parent flags come before the backend subcommand; headless keeps the parse from spawning a viewer."""


@pytest.mark.parametrize("config_type", TOOL_CONFIGS)
def test_the_default_backend_is_tensorrt_with_a_batch_one_profile(config_type: type) -> None:
    """No backend argument and a bare ``tensorrt`` both give the batch-1 profile, not trtkit's batch-32 default."""
    for backend in (
        config_type(rr_config=RerunTyroConfig(headless=True)).backend,
        tyro.cli(config_type, args=HEADLESS).backend,
        tyro.cli(config_type, args=[*HEADLESS, "tensorrt"]).backend,
    ):
        assert isinstance(backend, TensorRtBackendConfig)
        assert (backend.max_batch_size, backend.opt_batch_size) == (1, 1)


@pytest.mark.parametrize("config_type", TOOL_CONFIGS)
def test_onnx_is_the_onnx_runtime_fallback(config_type: type) -> None:
    assert isinstance(tyro.cli(config_type, args=[*HEADLESS, "onnx"]).backend, OnnxBackendConfig)


@pytest.mark.parametrize("config_type", TOOL_CONFIGS)
def test_tensorrt_takes_a_larger_batch_profile(config_type: type) -> None:
    backend = tyro.cli(config_type, args=[*HEADLESS, "tensorrt", "--backend.max-batch-size", "4"]).backend
    assert isinstance(backend, TensorRtBackendConfig)
    assert (backend.max_batch_size, backend.opt_batch_size) == (4, 1)
