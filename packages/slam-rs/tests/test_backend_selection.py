"""One backend rule for every slam-rs entry point: ``auto`` by default, resolved by the core's ``select_backend``."""

import dataclasses
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fixture_types import RigFactory

from slam_rs import Lane, _core
from slam_rs.apis import gate, replay, robocap_fleet, robocap_probe
from slam_rs.catalog_feed import select_decode_device


def _defaults(config_type: type) -> dict[str, object]:
    return {field.name: field.default for field in dataclasses.fields(config_type)}


@pytest.mark.parametrize("config_type", [replay.Config, gate.Config, robocap_fleet.Config, robocap_probe.Config])
def test_every_tool_defaults_to_auto(config_type: type) -> None:
    """No tool keeps its own default: each asks the shared selector for ``auto``."""
    defaults: dict[str, object] = _defaults(config_type)
    assert defaults["backend"] == "auto"
    assert "gpu" not in defaults
    if config_type in (replay.Config, robocap_probe.Config):
        assert defaults["decode_device"] == "auto"


def test_the_catalog_layer_defaults_to_auto() -> None:
    pytest.importorskip("dataforge", reason="catalog-layer publishes through dataforge, a Linux-only member")
    from slam_rs.apis import catalog_layer

    defaults: dict[str, object] = _defaults(catalog_layer.Config)
    assert (defaults["backend"], defaults["decode_device"]) == ("auto", "auto")


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


@pytest.fixture
def fresh_decode_policy(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Forget the cached NVDEC discovery and the printed notice for one test."""
    import threading

    from slam_rs import catalog_feed

    discovery = catalog_feed.nvdec_unavailable_reason  # a test may replace it; clear the real cache
    discovery.cache_clear()
    monkeypatch.setattr(catalog_feed, "_CPU_DECODE_NOTICE", threading.Event())
    yield
    discovery.cache_clear()


@pytest.mark.usefixtures("fresh_decode_policy")
@pytest.mark.parametrize(
    ("torch_present", "torchcodec_present", "cuda", "reason"),
    [
        (False, False, False, "no torch with torchcodec"),
        (True, False, False, "no torch with torchcodec"),
        (False, True, False, "no torch with torchcodec"),
        (True, True, False, "torch sees no CUDA device"),
        (True, True, True, None),
    ],
)
def test_nvdec_discovery_covers_every_combination(
    monkeypatch: pytest.MonkeyPatch, torch_present: bool, torchcodec_present: bool, cuda: bool, reason: str | None
) -> None:
    import sys
    from types import SimpleNamespace

    from slam_rs import catalog_feed

    present: dict[str, bool] = {"torch": torch_present, "torchcodec": torchcodec_present}
    monkeypatch.setattr(catalog_feed, "find_spec", lambda name: object() if present.get(name, True) else None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda)))
    found: str | None = catalog_feed.nvdec_unavailable_reason()
    assert (found is None) == (reason is None) and (reason is None or reason in str(found))
    if reason is None:
        assert select_decode_device("auto") == "cuda"
        assert select_decode_device("cuda") == "cuda"
    else:
        assert select_decode_device("auto") == "cpu"
        with pytest.raises(ValueError, match="NVDEC decoding is unavailable"):
            select_decode_device("cuda")
    assert select_decode_device("cpu") == "cpu"


@pytest.mark.usefixtures("fresh_decode_policy")
def test_the_cpu_decoding_notice_is_printed_once_per_process(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from slam_rs import catalog_feed

    monkeypatch.setattr(catalog_feed, "nvdec_unavailable_reason", lambda: "no CUDA here")
    assert select_decode_device() == select_decode_device("auto") == select_decode_device(requested="auto") == "cpu"
    with ThreadPoolExecutor(8) as pool:
        assert set(pool.map(lambda _: select_decode_device(), range(32))) == {"cpu"}
    assert capsys.readouterr().out.count("decoding on the CPU") == 1


def test_the_robocap_probe_routes_its_decoder_through_the_shared_policy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No catalog and no device: the feed is stubbed, so this checks only what the probe asks for."""
    from simplecv.rerun_log_utils import RerunTyroConfig

    requests: list[str] = []
    opened: dict[str, object] = {}

    class Opened(Exception):
        pass

    def select(requested: str) -> str:
        requests.append(requested)
        return "cuda"

    def open_segment(*_args: object, **kwargs: object) -> None:
        opened.update(kwargs)
        raise Opened

    monkeypatch.setattr(robocap_probe, "select_decode_device", select)
    monkeypatch.setattr(robocap_probe, "open_segment", open_segment)
    with pytest.raises(Opened):
        robocap_probe.main(robocap_probe.Config(rr_config=RerunTyroConfig(save=tmp_path / "probe.rrd"), output_csv=tmp_path / "probe.csv"))
    assert requests == ["auto"]
    assert opened["decode_device"] == "cuda"


@pytest.mark.skipif(sys.platform == "darwin", reason="Metal cannot be hidden from wgpu; the Vulkan loader honours VK_ICD_FILENAMES")
def test_without_an_adapter_every_default_vio_falls_back_with_one_warning(rig: RigFactory, tmp_path: Path) -> None:
    """No device is touched: the child process hides every Vulkan driver, so wgpu finds no adapter."""
    child: str = """
import sys
from slam_rs import _core
calibration = _core.Calibration.from_json(sys.argv[1])
vios = [_core.Vio(calibration, _core.VioConfig()) for _ in range(3)]
print([vio.backend for vio in vios], _core.select_backend("auto"), flush=True)
try:
    _core.Vio(calibration, _core.VioConfig(), backend="gpu")
except ValueError as error:
    print("gpu refused:", error, flush=True)
"""
    hidden: str = str(tmp_path / "no-driver.json")
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-c", child, rig(2).to_json()],
        capture_output=True,
        text=True,
        timeout=60.0,
        check=False,
        env={**os.environ, "VK_ICD_FILENAMES": hidden, "VK_DRIVER_FILES": hidden},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "['cpu', 'cpu', 'cpu'] cpu" in result.stdout
    assert "gpu refused: frontend: wgpu found no" in result.stdout
    assert result.stderr.count("tracking on the CPU because the GPU frontend cannot run here") == 1, result.stderr


def test_the_robocap_probe_names_its_raw_vio_output_as_the_layer_does() -> None:
    """The probe writes the raw causal VIO (it is compared with a raw-VIO regression reference), so its file is ``slam_rs_vio.csv``; ``slam_rs.csv`` is the final run."""
    assert robocap_probe.OUTPUT_NAME == "slam_rs_vio.csv"
