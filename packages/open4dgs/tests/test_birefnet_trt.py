"""BiRefNet's TensorRT path without a GPU: the DeformConv export, the snapshot identity and the engine-or-torch choice."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")
pytest.importorskip("onnxscript")


class _Deform(torch.nn.Module):
    """BiRefNet's ``DeformableConv2d`` pattern: offsets and a 2*sigmoid modulation mask from plain convolutions."""

    def __init__(self) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        self.offset_conv = torch.nn.Conv2d(4, 18, 3, padding=1)
        self.modulator_conv = torch.nn.Conv2d(4, 9, 3, padding=1)
        self.weight = torch.nn.Parameter(torch.randn(6, 4, 3, 3, generator=gen))
        for conv in (self.offset_conv, self.modulator_conv):
            torch.nn.init.normal_(conv.weight, std=0.3, generator=gen)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(x)
        mask = 2.0 * torch.sigmoid(self.modulator_conv(x))
        return torchvision.ops.deform_conv2d(x, offset, self.weight, None, padding=(1, 1), mask=mask)


def test_deform_conv_exports_as_onnx_deformconv_with_torchvisions_values(tmp_path: Path) -> None:
    """The custom translation keeps torchvision's offset layout and modulation mask: ONNX Runtime's DeformConv gives the same output."""
    ort = pytest.importorskip("onnxruntime")
    from open4dgs.birefnet_trt import ONNX_OPSET, _deform_conv2d_onnx

    model = _Deform().eval()
    x = torch.randn(2, 4, 12, 10, generator=torch.Generator().manual_seed(1))
    path = tmp_path / "deform.onnx"
    torch.onnx.export(
        model,
        (x,),
        str(path),
        input_names=["x"],
        output_names=["y"],
        opset_version=ONNX_OPSET,
        dynamo=True,
        custom_translation_table={torch.ops.torchvision.deform_conv2d.default: _deform_conv2d_onnx},
    )
    import onnx

    assert "DeformConv" in {node.op_type for node in onnx.load(str(path)).graph.node}
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy()})[0]
    with torch.no_grad():
        want = model(x).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def _snapshot(directory: Path, weights: bytes) -> Path:
    directory.mkdir()
    for name in ("birefnet.py", "BiRefNet_config.py", "config.json"):
        (directory / name).write_text(name)
    (directory / "model.safetensors").write_bytes(weights)
    return directory


def test_onnx_path_follows_the_snapshot(tmp_path: Path) -> None:
    from open4dgs.birefnet_trt import onnx_path

    a = _snapshot(tmp_path / "a", b"\x00" * 3_000_000)
    b = _snapshot(tmp_path / "b", b"\x00" * 3_000_000)
    c = _snapshot(tmp_path / "c", b"\x00" * 2_999_999 + b"\x01")
    assert onnx_path(a, tmp_path) == onnx_path(b, tmp_path)
    assert onnx_path(a, tmp_path) != onnx_path(c, tmp_path)
    assert onnx_path(a, tmp_path).parent == tmp_path


def test_without_an_engine_the_torch_model_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No cached engine (or a CPU device): ``load_matte_model`` returns the torch model, never builds."""
    from open4dgs import birefnet_trt

    snapshot = _snapshot(tmp_path / "s", b"\x00" * 10)
    sentinel = torch.nn.Identity()
    monkeypatch.setattr(birefnet_trt, "load_birefnet", lambda model_dir, device: sentinel)
    assert birefnet_trt.cached_engine(snapshot, tmp_path / "cache") is None
    assert birefnet_trt.load_matte_model(snapshot, torch.device("cpu"), tmp_path / "cache") is sentinel
