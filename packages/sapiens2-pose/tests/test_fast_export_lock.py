"""Concurrent callers publish one complete quantized fast-pose artifact."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

pytest.importorskip("torch", reason="requires the Sapiens2 environment")

# ruff: noqa: E402
import torch
import trtkit

from sapiens2_pose.api import fast_export, fast_onnx, runtime


def test_concurrent_fast_exports_share_one_completed_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    monkeypatch.setattr(fast_export, "calibration_path", lambda: calibration)
    monkeypatch.setattr(fast_export, "load_calibration", lambda: None)
    monkeypatch.setattr(fast_export, "make_fast_pose_exportable", lambda model: model)
    monkeypatch.setattr(runtime, "get_pose_model", lambda *_args: torch.nn.Identity())
    zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda shape, **_kwargs: zeros(shape))
    exports = []

    def export(_model, _inputs, target, **_kwargs):
        exports.append(target)
        target.write_bytes(b"bf16")

    def quantize(source, target, _calibration):
        assert source.read_bytes() == b"bf16"
        target.with_suffix(".onnx.data").write_bytes(b"weights")
        target.write_bytes(b"quantized")

    monkeypatch.setattr(trtkit, "export_onnx", export)
    monkeypatch.setattr(fast_onnx, "quantize_fast_pose", quantize)
    barrier = Barrier(2)

    def call():
        barrier.wait(timeout=10)
        return fast_export.ensure_fast_pose_onnx(tmp_path / "cache")

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _index: call(), range(2)))
    assert results[0] == results[1]
    assert results[0].read_bytes() == b"quantized"
    assert results[0].with_suffix(".onnx.data").read_bytes() == b"weights"
    assert len(exports) == 1
