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


def test_dynamic_dims_share_symbols_and_derive_multiples(tmp_path: Path) -> None:
    """One symbol per name across inputs, derived multiples, AUTO hints, and the autocast nesting."""
    from trtkit import DynamicDim

    captured: dict[str, object] = {}

    def fake_export_fn(model: object, inputs: object, path: str, **kwargs: object) -> None:
        captured.update(kwargs)
        Path(path).write_bytes(b"protobuf")

    export_onnx(
        torch.nn.Identity(),
        (torch.zeros(2, 3, 28, 42), torch.zeros(1, 3, 2, 3), torch.zeros(1, 9)),
        tmp_path / "dyn.onnx",
        input_names=["images", "grid", "tokens"],
        output_names=["y"],
        compute_dtype=torch.float16,
        dynamic_dims={
            "images": {0: DynamicDim("batch", 1, 4), 2: DynamicDim("rows", 2, 8, multiple=14), 3: DynamicDim("cols", 3, 9, multiple=14)},
            "grid": {2: DynamicDim("rows", 2, 8), 3: DynamicDim("cols", 3, 9)},
            "tokens": {1: DynamicDim("tokens", 7, 73, auto=True)},
        },
        export_fn=fake_export_fn,
    )

    shapes = cast(tuple[tuple[dict[int, object], ...]], captured["dynamic_shapes"])
    assert len(shapes) == 1, "autocast wrapper takes *inputs, so torch.export sees one varargs parameter"
    images_dims, grid_dims, token_dims = shapes[0]
    assert str(images_dims[2]) == "14*rows" and str(images_dims[3]) == "14*cols"
    assert images_dims[2].root is grid_dims[2] and images_dims[3].root is grid_dims[3], "derived dims must share the grid's symbols"
    assert images_dims[0].min == 1 and images_dims[0].max == 4
    assert "AUTO" in str(token_dims[1]) and "min=7" in str(token_dims[1]) and "max=73" in str(token_dims[1])


def test_dynamic_dims_with_a_range_of_one_stay_static() -> None:
    """``min == max`` (a batch pinned to 1 inside a dynamic spec) becomes a static dim instead of an invalid ``torch.export.Dim``."""
    from trtkit import DynamicDim
    from trtkit.onnx_export import _torch_dynamic_shapes

    (images_dims, grid_dims) = _torch_dynamic_shapes(
        ["images", "grid"],
        {"images": {0: DynamicDim("batch", 1, 1), 1: DynamicDim("views", 2, 6)}, "grid": {0: DynamicDim("batch", 1, 1, multiple=14)}},
    )
    assert 0 not in images_dims and 0 not in grid_dims, "a range of one is static"
    assert images_dims[1].min == 2 and images_dims[1].max == 6


def test_dynamic_dims_reject_conflicts() -> None:
    """The spec is complete and consistent: no batch shorthand beside it, no unknown inputs, no conflicting bounds."""
    from trtkit import DynamicDim
    from trtkit.onnx_export import _torch_dynamic_shapes

    with pytest.raises(ValueError, match="complete shape spec"):
        export_onnx(
            torch.nn.Identity(),
            (torch.zeros(2, 3),),
            Path("/nonexistent/dyn.onnx"),
            input_names=["x"],
            output_names=["y"],
            dynamic_batch_max=4,
            dynamic_dims={"x": {0: DynamicDim("batch", 1, 4)}},
            export_fn=lambda *_args, **_kwargs: None,
        )
    with pytest.raises(ValueError, match="not exported"):
        _torch_dynamic_shapes(["x"], {"y": {0: DynamicDim("batch", 1, 4)}})
    with pytest.raises(ValueError, match="bounds"):
        _torch_dynamic_shapes(["x", "z"], {"x": {0: DynamicDim("n", 1, 4)}, "z": {0: DynamicDim("n", 2, 4)}})
    with pytest.raises(ValueError, match="multiple"):
        _torch_dynamic_shapes(["x"], {"x": {0: DynamicDim("n", 1, 4, multiple=2, auto=True)}})


def test_shape_profiles_pin_dynamic_inputs_and_key_the_engine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Profiles set min/opt/max per dynamic input, skip static ones, reject gaps, and change the cache name."""
    from trtkit import InputShapeProfile
    from trtkit.trt_builder import _set_shape_profiles

    class FakeProfile:
        def __init__(self) -> None:
            self.shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = {}

        def set_shape(self, name: str, low: tuple[int, ...], opt: tuple[int, ...], high: tuple[int, ...]) -> None:
            self.shapes[name] = (low, opt, high)

    network = SimpleNamespace(
        num_inputs=2,
        get_input=lambda index: [SimpleNamespace(name="images", shape=(-1, 3, -1, -1)), SimpleNamespace(name="bias", shape=(1, 6, 9, 9))][index],
    )
    profile = FakeProfile()
    entry = InputShapeProfile(name="images", min_shape=(1, 3, 28, 28), opt_shape=(1, 3, 56, 56), max_shape=(4, 3, 112, 112))
    _set_shape_profiles(network, profile, (entry,))
    assert profile.shapes == {"images": (entry.min_shape, entry.opt_shape, entry.max_shape)}
    with pytest.raises(RuntimeError, match="no shape profile"):
        _set_shape_profiles(network, FakeProfile(), ())
    with pytest.raises(RuntimeError, match="missing from the ONNX graph"):
        _set_shape_profiles(network, FakeProfile(), (entry, InputShapeProfile(name="ghost", min_shape=(1,), opt_shape=(1,), max_shape=(1,))))
    with pytest.raises(RuntimeError, match="rank"):
        _set_shape_profiles(network, FakeProfile(), (InputShapeProfile(name="images", min_shape=(1, 3), opt_shape=(1, 3), max_shape=(4, 3)),))

    fake_trt = SimpleNamespace(__version__="11.2.1.2")
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    onnx_path: Path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    plain: Path = cached_engine_path(onnx_path, TrtBuildConfig(), cache_dir=tmp_path)
    profiled: Path = cached_engine_path(onnx_path, TrtBuildConfig(shape_profiles=(entry,)), cache_dir=tmp_path)
    other: Path = cached_engine_path(
        onnx_path, TrtBuildConfig(shape_profiles=(InputShapeProfile(name="images", min_shape=(1, 3, 28, 28), opt_shape=(1, 3, 56, 56), max_shape=(2, 3, 112, 112)),)), cache_dir=tmp_path
    )
    assert plain != profiled != other
    assert "_s" not in plain.name.split("_trt")[0].split("_w")[1]


class _ToyDynamic(torch.nn.Module):
    """Conv over a dynamic image plus a batch-1 bias input that broadcasts over the batch."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, images: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        return self.conv(images) + bias[:, :, None, None]


@cuda_only
def test_dynamic_runtime_runs_two_shapes_from_one_engine(tmp_path: Path) -> None:
    """A shape-profiled engine serves several batch and spatial sizes with batch-1 side inputs."""
    from trtkit import DynamicDim, InputShapeProfile, TensorRtDynamicRuntime, ensure_engine

    model = _ToyDynamic().eval().cuda()
    onnx_path: Path = tmp_path / "toy.onnx"
    export_onnx(
        model,
        (torch.zeros(2, 3, 28, 42, device="cuda"), torch.zeros(1, 4, device="cuda")),
        onnx_path,
        input_names=["images", "bias"],
        output_names=["y"],
        compute_dtype=torch.float16,
        dynamic_dims={"images": {0: DynamicDim("batch", 1, 4), 2: DynamicDim("rows", 2, 6, multiple=14), 3: DynamicDim("cols", 2, 6, multiple=14)}},
    )
    config = TrtBuildConfig(
        max_batch_size=4,
        opt_batch_size=1,
        workspace_gib=1.0,
        shape_profiles=(
            InputShapeProfile(name="images", min_shape=(1, 3, 28, 28), opt_shape=(1, 3, 56, 56), max_shape=(4, 3, 84, 84)),
            InputShapeProfile(name="bias", min_shape=(1, 4), opt_shape=(1, 4), max_shape=(1, 4)),
        ),
    )
    runtime = TensorRtDynamicRuntime(ensure_engine(onnx_path, config, cache_dir=tmp_path), use_cuda_graph=True)
    assert runtime.max_input_shapes == {"images": (4, 3, 84, 84), "bias": (1, 4)}
    assert runtime.spec.max_batch_size == 4
    bias = torch.randn(1, 4, device="cuda")
    for shape in ((1, 3, 28, 42), (3, 3, 70, 56), (4, 3, 84, 84)):
        images = torch.randn(*shape, device="cuda")
        with torch.inference_mode():
            expected = model(images, bias)
        outputs = runtime({"images": images, "bias": bias})
        assert outputs["y"].shape == expected.shape
        torch.testing.assert_close(outputs["y"].float(), expected, rtol=2e-2, atol=2e-2)
    with pytest.raises(ValueError, match="outside the engine profile"):
        runtime({"images": torch.randn(5, 3, 28, 28, device="cuda"), "bias": bias})
