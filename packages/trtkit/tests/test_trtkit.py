"""Tests for trtkit's additions over the runners it consolidates.

The moved behavior (three-backend parity, CUDA-graph replay, padding) stays
covered by posekit's test suite through its re-exports; this covers what
trtkit adds: the ``allow_tf32`` build axis.
"""

from pathlib import Path

import pytest
import torch

from trtkit import TrtBuildConfig, cached_engine_path

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


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
