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

from trtkit import TrtBuildConfig, cached_engine_path, sweep_stale_onnx_exports

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


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
