"""Artifact transforms are explicit config, applied after detector NMS removal."""
from types import SimpleNamespace
from unittest.mock import Mock

from posekit.models import yolox
from posekit.runtimes import TensorRuntime


def test_onnx_transform_receives_stripped_artifact_and_runtime_uses_result(tmp_path, monkeypatch):
    raw, stripped, transformed = (tmp_path / name for name in ('raw.onnx', 'stripped.onnx', 'transformed.onnx'))
    seen = []
    def fetch(_url):
        return raw
    def strip(source):
        assert source == raw
        return stripped
    def transform(source):
        assert source == stripped
        return transformed
    runtime = Mock(spec=TensorRuntime)
    runtime.spec = SimpleNamespace(inputs=[SimpleNamespace(shape=(32, 32, 3))], outputs=[SimpleNamespace(shape=(4, 4), name='boxes'), SimpleNamespace(shape=(1, 4), name='scores')])
    def create(path, _backend):
        seen.append(path)
        return runtime
    monkeypatch.setattr(yolox, 'fetch_openmmlab_onnx', fetch)
    monkeypatch.setattr(yolox, 'strip_detector_nms', strip)
    monkeypatch.setattr(yolox, 'create_runtime_from_onnx', create)
    yolox.YoloxDetectorConfig(onnx_transform=transform).setup()
    assert seen == [transformed]
