"""One backend rule for every slam-rs entry point: ``auto`` by default, resolved by the core's ``select_backend``."""

import dataclasses

import pytest
from fixture_types import RigFactory

from slam_rs import Lane, _core
from slam_rs.apis import catalog_layer, gate, replay, robocap_fleet, robocap_probe
from slam_rs.catalog_feed import select_decode_device


@pytest.mark.parametrize("config_type", [replay.Config, gate.Config, catalog_layer.Config, robocap_fleet.Config, robocap_probe.Config])
def test_every_tool_defaults_to_auto(config_type: type) -> None:
    """No tool keeps its own default: each asks the shared selector for ``auto``."""
    defaults: dict[str, object] = {field.name: field.default for field in dataclasses.fields(config_type)}
    assert defaults["backend"] == "auto"
    assert "gpu" not in defaults


@pytest.mark.parametrize("config_type", [replay.Config, catalog_layer.Config])
def test_decoding_tools_default_to_auto(config_type: type) -> None:
    defaults: dict[str, object] = {field.name: field.default for field in dataclasses.fields(config_type)}
    assert defaults["decode_device"] == "auto"


def test_cpu_is_always_granted() -> None:
    assert _core.select_backend("cpu") == "cpu"
    assert select_decode_device("cpu") == "cpu"


def test_an_unknown_request_is_refused(rig: RigFactory) -> None:
    with pytest.raises(ValueError, match='unknown backend "cuda"'):
        _core.select_backend("cuda")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown backend"):
        _core.Vio(rig(1), _core.VioConfig(), backend="wgpu")  # type: ignore[arg-type]


def test_the_cpu_request_builds_a_cpu_pipeline(rig: RigFactory) -> None:
    vio: _core.Vio = _core.Vio(rig(2), _core.VioConfig(), backend="cpu")
    assert (vio.backend, vio.gpu) == ("cpu", False)


@pytest.mark.integration
def test_auto_and_gpu_agree_with_the_probe(rig: RigFactory) -> None:
    """Needs a GPU to say anything about the GPU arm: auto and an explicit gpu request follow the same probe."""
    lane: Lane = _core.select_backend("auto")
    if lane == "cpu":
        with pytest.raises(ValueError):
            _core.select_backend("gpu")
        pytest.skip("no usable GPU adapter here (or a core built without gpu-wgpu): auto chose the CPU")
    assert _core.gpu_backend == "wgpu"
    assert _core.select_backend("gpu") == "gpu"
    assert _core.Vio(rig(2), _core.VioConfig()).backend == "gpu", "the constructor's default is auto"
