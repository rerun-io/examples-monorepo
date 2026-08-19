"""Public-contract tests for MammaNet's posekit landmark role."""

from pathlib import Path

import pytest
import torch
from posekit.predictions import BoxDetections
from posekit.runtimes import TensorRtBackendConfig
from torch import Tensor
from trtkit import RuntimeSpec, TensorSpec

from mamma.landmarks.mammanet import MammaNet
from mamma.landmarks.posekit_role import MammaNetLandmarksConfig


def test_accelerated_role_requires_cuda_before_loading_weights() -> None:
    with pytest.raises(ValueError, match="require device='cuda'"):
        MammaNetLandmarksConfig(device="cpu", backend=TensorRtBackendConfig()).setup()


def test_tensorrt_role_builds_from_weights_and_returns_dense_landmark_heads(monkeypatch, tmp_path: Path) -> None:
    """The TensorRT role resolves weights and preserves MammaNet's image-space output contract."""

    class FakeTensorRtRuntime:
        def __init__(self) -> None:
            self.spec = RuntimeSpec(
                inputs=(
                    TensorSpec("crops", (3, 512, 384), torch.float32),
                    TensorSpec("masks", (1, 512, 384), torch.float32),
                ),
                outputs=(
                    TensorSpec("joints2d", (512, 3), torch.float32),
                    TensorSpec("visibility", (512, 1), torch.float32),
                    TensorSpec("contact", (512, 1), torch.float32),
                    TensorSpec("floor_contact", (512, 1), torch.float32),
                ),
                max_batch_size=32,
            )

        def __call__(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
            batch_size: int = int(inputs["crops"].shape[0])
            device: torch.device = inputs["crops"].device
            return {
                "joints2d": torch.zeros((batch_size, 512, 3), device=device),
                "visibility": torch.zeros((batch_size, 512, 1), device=device),
                "contact": torch.zeros((batch_size, 512, 1), device=device),
                "floor_contact": torch.zeros((batch_size, 512, 1), device=device),
            }

    weights_path: Path = tmp_path / "mammanet.safetensors"
    weights_path.touch()
    onnx_path: Path = tmp_path / "mammanet.onnx"
    model: MammaNet = MammaNet.__new__(MammaNet)

    def fake_load_mammanet(path: Path | None, *, device: str, config: object) -> MammaNet:
        del path, device, config
        return model

    def fake_ensure_mammanet_onnx(loaded_model: MammaNet, resolved_weights_path: Path) -> Path:
        del loaded_model, resolved_weights_path
        return onnx_path

    def fake_create_runtime(path: Path, backend: TensorRtBackendConfig) -> FakeTensorRtRuntime:
        del path, backend
        return FakeTensorRtRuntime()

    monkeypatch.setattr("mamma.landmarks.posekit_role.load_mammanet", fake_load_mammanet)
    monkeypatch.setattr("mamma.landmarks.posekit_role.ensure_mammanet_onnx", fake_ensure_mammanet_onnx)
    monkeypatch.setattr("mamma.landmarks.posekit_role.create_runtime_from_onnx", fake_create_runtime)
    backend = TensorRtBackendConfig()
    role = MammaNetLandmarksConfig(weights_path=weights_path, device="cuda", backend=backend).setup()
    frames_rgb: Tensor = torch.zeros((1, 18, 12, 3), dtype=torch.uint8)
    detections = BoxDetections(
        xyxy=torch.tensor([[2.0, 3.0, 10.0, 15.0]], dtype=torch.float32),
        scores=torch.ones((1,), dtype=torch.float32),
        frame_indices=torch.zeros((1,), dtype=torch.long),
        masks=torch.ones((1, 18, 12), dtype=torch.bool),
    )

    result = role(frames_rgb, detections)

    torch.testing.assert_close(result.xy, torch.tensor([6.0, 9.0]).expand(1, 512, 2))
    torch.testing.assert_close(result.log_variance, torch.zeros((1, 512)))
    torch.testing.assert_close(result.visibility, torch.full((1, 512), 0.5))
    torch.testing.assert_close(result.contact, torch.full((1, 512), 0.5))
    torch.testing.assert_close(result.floor_contact, torch.full((1, 512), 0.5))
