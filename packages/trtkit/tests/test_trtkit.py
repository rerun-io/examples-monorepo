"""Tests for trtkit's additions over the runners it consolidates.

The moved behavior (three-backend parity, CUDA-graph replay, padding) stays
covered by posekit's test suite through its re-exports; this covers what
trtkit adds: the ``allow_tf32`` build axis.
"""

import os
import time
from pathlib import Path

import pytest
import torch

from trtkit import TrtBuildConfig, cached_engine_path, export_onnx, sweep_stale_onnx_exports

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
