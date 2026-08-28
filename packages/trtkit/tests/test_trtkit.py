"""Tests for trtkit's additions over the runners it consolidates.

The moved behavior (three-backend parity, CUDA-graph replay, padding) stays
covered by posekit's test suite through its re-exports; this covers what
trtkit adds: the ``allow_tf32`` build axis.
"""

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from trtkit import TrtBuildConfig, build_engine, cached_engine_path, export_onnx, sweep_stale_onnx_exports

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_export_onnx_publishes_deterministic_sidecar_and_removes_temp_dir(tmp_path: Path) -> None:
    """External-data sidecars keep the `<name>.data` name the protobuf references."""
    out_path: Path = tmp_path / "model_v1.onnx"

    def fake_export_fn(model: object, inputs: object, path: str, **kwargs: object) -> None:
        target: Path = Path(path)
        target.write_bytes(b"protobuf")
        target.with_name(f"{target.name}.data").write_bytes(b"weights")

    export_onnx(
        torch.nn.Identity(),
        (torch.zeros(1),),
        out_path,
        input_names=["x"],
        output_names=["y"],
        export_fn=fake_export_fn,
    )

    assert out_path.read_bytes() == b"protobuf"
    assert out_path.with_name("model_v1.onnx.data").read_bytes() == b"weights"
    assert {path.name for path in tmp_path.iterdir()} == {"model_v1.onnx", "model_v1.onnx.data"}


def test_export_onnx_removes_temp_dir_when_export_fails(tmp_path: Path) -> None:
    """A crashing export leaves neither a truncated file nor a temp directory."""

    def failing_export_fn(model: object, inputs: object, path: str, **kwargs: object) -> None:
        Path(path).write_bytes(b"truncated")
        raise RuntimeError("export died")

    with pytest.raises(RuntimeError, match="export died"):
        export_onnx(
            torch.nn.Identity(),
            (torch.zeros(1),),
            tmp_path / "model_v1.onnx",
            input_names=["x"],
            output_names=["y"],
            export_fn=failing_export_fn,
        )

    assert list(tmp_path.iterdir()) == []


def test_sweep_stale_onnx_exports_removes_abandoned_temp_dirs(tmp_path: Path) -> None:
    """Abandoned pid-suffixed temp directories are swept like stale files."""
    abandoned_dir: Path = tmp_path / "model_v1.onnx.part456"
    abandoned_dir.mkdir()
    (abandoned_dir / "model_v1.onnx").write_bytes(b"orphan")
    stale_timestamp: float = time.time() - 7200.0
    os.utime(abandoned_dir, (stale_timestamp, stale_timestamp))
    current: Path = tmp_path / "model_v2.onnx"
    current.write_bytes(b"current")

    removed: list[Path] = sweep_stale_onnx_exports(tmp_path, "model_", keep_paths={current})

    assert removed == [abandoned_dir]
    assert {path.name for path in tmp_path.iterdir()} == {current.name}


def test_sweep_stale_onnx_exports_keeps_current_recent_and_unrelated_files(tmp_path: Path) -> None:
    """Only obsolete exports and abandoned partial files are removed."""
    current: Path = tmp_path / "model_v3.onnx"
    current_sidecar: Path = tmp_path / "model_v3.onnx.part123.data"
    old_export: Path = tmp_path / "model_v2.onnx"
    old_partial: Path = tmp_path / "model_v1.onnx.part456"
    recent_partial: Path = tmp_path / "model_v4.onnx.part789"
    unrelated: Path = tmp_path / "other_v1.onnx"
    for path in (current, current_sidecar, old_export, old_partial, recent_partial, unrelated):
        path.write_bytes(path.name.encode())
    stale_timestamp: float = time.time() - 7200.0
    os.utime(old_partial, (stale_timestamp, stale_timestamp))

    removed: list[Path] = sweep_stale_onnx_exports(
        tmp_path,
        "model_",
        keep_paths={current, current_sidecar},
    )

    assert removed == [old_partial, old_export]
    assert {path.name for path in tmp_path.iterdir()} == {
        current.name,
        current_sidecar.name,
        recent_partial.name,
        unrelated.name,
    }


@cuda_only
def test_allow_tf32_cache_key(tmp_path: Path) -> None:
    """Disabling TF32 is part of the engine identity; enabling it keeps legacy names."""
    onnx_path: Path = tmp_path / "dummy.onnx"
    onnx_path.write_bytes(b"cache-key probe")
    default_path: Path = cached_engine_path(onnx_path, TrtBuildConfig(), cache_dir=tmp_path)
    notf32_path: Path = cached_engine_path(onnx_path, TrtBuildConfig(allow_tf32=False), cache_dir=tmp_path)
    assert default_path != notf32_path
    assert "notf32" in notf32_path.name
    assert "notf32" not in default_path.name


def test_same_compute_capability_build_is_cache_keyed_and_recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A portable plan applies TensorRT's same-CC mode and records that build axis."""

    class FakeBuilderConfig:
        def __init__(self) -> None:
            self.builder_optimization_level: int = -1
            self.hardware_compatibility_level: object | None = None

        def set_memory_pool_limit(self, pool: object, limit: int) -> None:
            self.memory_pool_limit: tuple[object, int] = (pool, limit)

        def clear_flag(self, flag: object) -> None:
            self.cleared_flag: object = flag

        def add_optimization_profile(self, profile: object) -> None:
            self.optimization_profile: object = profile

    class FakeBuilder:
        def __init__(self, config: FakeBuilderConfig) -> None:
            self.config: FakeBuilderConfig = config

        def create_network(self, flags: int) -> SimpleNamespace:
            self.network_flags: int = flags
            return SimpleNamespace(num_inputs=0, num_layers=0)

        def create_builder_config(self) -> FakeBuilderConfig:
            return self.config

        def create_optimization_profile(self) -> SimpleNamespace:
            return SimpleNamespace()

        def build_serialized_network(self, network: object, config: FakeBuilderConfig) -> bytes:
            self.built_network: object = network
            self.build_config: FakeBuilderConfig = config
            return b"engine"

    class FakeParser:
        num_errors: int = 0

        def parse_from_file(self, path: str) -> bool:
            self.parsed_path: str = path
            return True

    class FakeLogger:
        WARNING: int = 1

        def __init__(self, severity: int) -> None:
            self.severity: int = severity

    fake_builder_config: FakeBuilderConfig = FakeBuilderConfig()
    fake_builder: FakeBuilder = FakeBuilder(fake_builder_config)
    fake_trt: SimpleNamespace = SimpleNamespace(
        __version__="11.2.1.2",
        Builder=lambda logger: fake_builder,
        BuilderFlag=SimpleNamespace(TF32=1),
        HardwareCompatibilityLevel=SimpleNamespace(SAME_COMPUTE_CAPABILITY="same-compute-capability"),
        Logger=FakeLogger,
        MemoryPoolType=SimpleNamespace(WORKSPACE=1),
        NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=0),
        OnnxParser=lambda network, logger: FakeParser(),
    )
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test-gpu")
    onnx_path: Path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    config: TrtBuildConfig = TrtBuildConfig(hardware_compatibility="same_compute_capability")

    engine_path: Path = cached_engine_path(onnx_path, config, cache_dir=tmp_path)
    build_engine(onnx_path, engine_path, config)

    manifest: dict[str, object] = json.loads(engine_path.with_suffix(".engine.json").read_text())
    assert "samecc" in engine_path.name
    assert fake_builder_config.hardware_compatibility_level == "same-compute-capability"
    assert manifest["portable_engine"] is True
    assert manifest["rebuild_from_onnx_on_target_machine"] is False
    assert manifest["hardware_compatibility"] == "same_compute_capability"


def test_fp32_transposed_conv_islands_share_weights_without_mutating() -> None:
    """bf16 exports isolate transposed convs in fp32 without touching the caller's model."""
    from trtkit.onnx_export import _Fp32Island, _with_fp32_transposed_convs

    model = torch.nn.Sequential(
        torch.nn.Conv2d(2, 4, 3, padding=1),
        torch.nn.ConvTranspose2d(4, 2, 2, stride=2),
    ).eval()
    wrapped: torch.nn.Sequential = cast(torch.nn.Sequential, _with_fp32_transposed_convs(model))

    assert isinstance(wrapped[1], _Fp32Island)
    assert wrapped[1].inner is model[1], "island must share the original conv (weights by reference)"
    assert not any(isinstance(m, _Fp32Island) for m in model.modules()), "caller's tree must stay unmodified"
    rewrapped: torch.nn.Sequential = cast(torch.nn.Sequential, _with_fp32_transposed_convs(wrapped))
    assert rewrapped[1] is wrapped[1], "re-wrapping must be idempotent"

    x = torch.randn(1, 2, 8, 8)
    with torch.inference_mode():
        torch.testing.assert_close(wrapped(x), model(x))
