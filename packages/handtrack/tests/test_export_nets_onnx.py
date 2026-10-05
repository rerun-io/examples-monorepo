"""The ONNX graphs robocap-live deploys reproduce DetNet-F and KeyNet-F (random weights, no assets)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from jaxtyping import Float32
from numpy import ndarray
from torch import Tensor

pytest.importorskip("onnxruntime", reason="onnxruntime comes with the cuda feature")
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402

from handtrack.apis.export_nets_onnx import DetNetGraph, KeyNetGraph, export_graph  # noqa: E402
from handtrack.models.detnet import DetNetF  # noqa: E402
from handtrack.models.keynet import KeyNetF  # noqa: E402


def randomise(model: torch.nn.Module) -> None:
    """Put ``model`` in eval mode with non-trivial BatchNorm statistics, so the test exercises them."""
    generator: torch.Generator = torch.Generator().manual_seed(7)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            assert module.running_mean is not None and module.running_var is not None
            module.running_mean.copy_(torch.randn(module.num_features, generator=generator) * 0.1)
            module.running_var.copy_(torch.rand(module.num_features, generator=generator) + 0.5)
    model.eval()


def test_keynet_graph_matches_forward_and_onnx(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model: KeyNetF = KeyNetF(pinch_head=True)
    randomise(model)
    crop: Float32[Tensor, "3 1 96 96"] = torch.rand(3, 1, 96, 96)
    keypoints: Float32[Tensor, "3 63"] = torch.rand(3, 63)
    with torch.inference_mode():
        expected = model(crop, keypoints)
        graph: tuple[Tensor, ...] = KeyNetGraph(model)(crop, keypoints)
    assert expected.pinch_logit is not None
    torch.testing.assert_close(graph[0], expected.heatmaps)
    torch.testing.assert_close(graph[1], expected.distance)
    torch.testing.assert_close(graph[2][:, 0], expected.presence_logit)
    torch.testing.assert_close(graph[3][:, 0], expected.pinch_logit)
    path: Path = tmp_path / "keynet.onnx"
    names: list[str] = ["heatmaps", "distance", "presence_logit", "pinch_logit"]
    export_graph(KeyNetGraph(model), (crop, keypoints), path, ["crop", "keypoints"], names, True, 17)
    ops: set[str] = {node.op_type for node in onnx.load(str(path)).graph.node}
    assert "ReduceMean" in ops and "Flatten" in ops
    for node in onnx.load(str(path)).graph.node:
        if node.op_type == "ReduceMean":
            assert {attribute.name: attribute.i for attribute in node.attribute}.get("keepdims", 1) == 1
    session: ort.InferenceSession = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    values: list[ndarray] = [np.asarray(value) for value in session.run(names, {"crop": crop[:2].numpy(), "keypoints": keypoints[:2].numpy()})]
    for value, reference in zip(values, graph, strict=True):
        np.testing.assert_allclose(value, reference[:2].numpy(), atol=1e-4, rtol=1e-4)


def test_keynet_graph_needs_pinch_head() -> None:
    with pytest.raises(ValueError, match="pinch head"):
        KeyNetGraph(KeyNetF())


def test_detnet_pooled_graph_equals_full_on_replicated_frames(tmp_path: Path) -> None:
    torch.manual_seed(5)
    model: DetNetF = DetNetF()
    randomise(model)
    pooled: Float32[Tensor, "2 1 120 160"] = torch.randint(0, 256, (2, 1, 120, 160)).float() / 255.0
    full: Float32[Tensor, "2 1 480 640"] = pooled.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)
    path: Path = tmp_path / "detnet_pooled.onnx"
    export_graph(DetNetGraph(model, pooled=True), (pooled[:1],), path, ["image"], ["center", "radius", "presence_logit"], False, 17)
    session: ort.InferenceSession = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected: tuple[Tensor, ...] = DetNetGraph(model, pooled=False)(full)
    for sample in range(2):
        values: list[ndarray] = [np.asarray(value) for value in session.run(None, {"image": pooled[sample : sample + 1].numpy()})]
        for value, reference in zip(values, expected, strict=True):
            np.testing.assert_allclose(value, reference[sample : sample + 1].numpy(), atol=1e-4, rtol=1e-4)
