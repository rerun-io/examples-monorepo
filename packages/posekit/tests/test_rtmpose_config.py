"""RTMW takes the same explicit artifact transform as YOLOX (e.g. trtkit's fp16 conv trunk)."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from posekit.models import rtmpose
from posekit.runtimes import TensorRuntime


def test_onnx_transform_receives_the_fetched_artifact_and_runtime_uses_result(tmp_path, monkeypatch):
    raw, transformed = tmp_path / "raw.onnx", tmp_path / "transformed.onnx"
    seen = []

    def transform(source):
        assert source == raw
        return transformed

    runtime = Mock(spec=TensorRuntime)
    runtime.spec = SimpleNamespace(
        inputs=[SimpleNamespace(shape=(8, 3, 256, 192), dtype=torch.float32)],
        outputs=[SimpleNamespace(shape=(8, 133, 384), name="simcc_x"), SimpleNamespace(shape=(8, 133, 512), name="simcc_y")],
    )

    def create(path, _backend):
        seen.append(path)
        return runtime

    monkeypatch.setattr(rtmpose, "fetch_openmmlab_onnx", lambda _url: raw)
    monkeypatch.setattr(rtmpose, "create_runtime_from_onnx", create)
    rtmpose.RtmPoseConfig(onnx_transform=transform).setup()
    assert seen == [transformed]
