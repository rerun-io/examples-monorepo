"""Released-weight EgoHand checks; no synthetic weights or network downloads."""

from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

pytest.importorskip("egohand_inference", reason="requires the egohand Pixi environment")

from posekit.apis.egohand_selfcap import camera_from_window
from posekit.models.egohand import EgoHandConfig, EgoHandDetectorConfig

pytestmark = pytest.mark.integration


def test_tensorrt_detector_preserves_batched_anatomical_detections() -> None:
    """Changing the detector runtime preserves boxes, count and anatomical side."""
    from PIL import Image

    from posekit.runtimes import OnnxBackendConfig, TensorRtBackendConfig

    weights = Path("/tmp/egohand/weights/detector/hand_detector.onnx")
    image = Path("/tmp/egohand/frame-4.144.png")
    for asset in (weights, image):
        if not asset.is_file():
            pytest.skip(f"missing EgoHand detector fixture: {asset}")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    frames = torch.from_numpy(np.asarray(Image.open(image).convert("RGB")).copy()).cuda().unsqueeze(0).repeat(3, 1, 1, 1)
    reference = EgoHandDetectorConfig(weights, backend=OnnxBackendConfig(max_batch_size=3)).setup()(frames)
    result = EgoHandDetectorConfig(
        weights, backend=TensorRtBackendConfig(max_batch_size=8, opt_batch_size=4, cache_dir=Path("/tmp/egohand/engines"))
    ).setup()(frames)
    assert result.sides == reference.sides
    assert torch.equal(result.boxes.frame_indices, reference.boxes.frame_indices)
    low = torch.maximum(result.boxes.xyxy[:, :2], reference.boxes.xyxy[:, :2])
    high = torch.minimum(result.boxes.xyxy[:, 2:], reference.boxes.xyxy[:, 2:])
    intersection = (high - low).clamp_min(0).prod(-1)
    areas = (result.boxes.xyxy[:, 2:] - result.boxes.xyxy[:, :2]).prod(-1)
    reference_areas = (reference.boxes.xyxy[:, 2:] - reference.boxes.xyxy[:, :2]).prod(-1)
    assert torch.median(intersection / (areas + reference_areas - intersection)) >= 0.99


def test_released_models_return_aligned_hands() -> None:
    weights = Path("/tmp/egohand/weights")
    required = [weights / "egohand/egohand_s060000.ckpt", weights / "detector/hand_detector.onnx", weights / "soma/SOMAHand.npz"]
    required.extend(
        weights / name
        for name in (
            "soma/SOMA_neutral.npz",
            "soma/SOMA_procedural_transforms.json",
            "soma/SOMA_template_rig.usda",
            "soma/MHR/mhr_model_lod1.pt",
            "soma/MHR/base_body_lod1.obj",
            "soma/MHR/SOMA_wrap_lod1.obj",
        )
    )
    for asset in required:
        if not asset.is_file():
            pytest.skip(f"missing EgoHand asset: {asset}")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    image = Path("/tmp/egohand/frame-4.144.png")
    if not image.is_file():
        pytest.skip(f"missing one-segment RGB fixture: {image}")
    from PIL import Image

    rgb = np.asarray(Image.open(image).convert("RGB"))
    frames = torch.from_numpy(rgb.copy()).unsqueeze(0).cuda()
    window = Path("/tmp/egohand/window.parquet")
    if not window.is_file():
        pytest.skip(f"missing one-segment calibration fixture: {window}")
    camera = camera_from_window(pq.read_table(window))
    detections = EgoHandDetectorConfig(required[1]).setup()(frames)
    assert detections.sides == ("left", "right")
    assert float(detections.boxes.xyxy[0, 0]) < float(detections.boxes.xyxy[1, 0])
    model = EgoHandConfig(required[0], weights / "soma").setup()
    result = model(frames, detections, camera)
    assert len(result.sides) > 0
    assert result.keypoints.xyz_camera is not None
    assert torch.isfinite(result.keypoints.xyz_camera).all()
    assert result.keypoints.xy.shape == (len(result.sides), 25, 2)
    assert len(result.sides) == len(result.keypoints.frame_indices)
    assert result.scale_parameters is not None
    assert result.scale_parameters.shape == (len(result.sides), 26)
    assert result.rotations.shape == (len(result.sides), 25, 3, 3)
    for xyz, xy in zip(result.keypoints.xyz_camera.cpu().numpy(), result.keypoints.xy.cpu().numpy(), strict=True):
        expected, _ = cv2.projectPoints(xyz.astype(np.float64), np.zeros(3), np.zeros(3), camera.intrinsics, camera.distortion)
        np.testing.assert_allclose(xy, expected[:, 0], atol=1e-3)


def test_accelerated_predictor_retains_mesh_and_joint_parity() -> None:
    """TensorRT, GPU crops and batched SOMA stay within the geometry guard."""
    import gc

    from PIL import Image

    from posekit.runtimes import TensorRtBackendConfig

    root = Path("/tmp/egohand")
    required = [
        root / name
        for name in (
            "weights/egohand/egohand_s060000.ckpt",
            "weights/soma/SOMAHand.npz",
            "weights/soma/MHR/mhr_model_lod1.pt",
            "weights/detector/hand_detector.onnx",
            "dinov3-bf16.onnx",
            "window.parquet",
            "frame-4.144.png",
        )
    ]
    other_assets = [
        root / "weights/soma" / name
        for name in (
            "SOMA_neutral.npz",
            "SOMA_procedural_transforms.json",
            "SOMA_template_rig.usda",
            "MHR/base_body_lod1.obj",
            "MHR/SOMA_wrap_lod1.obj",
        )
    ]
    for asset in required + other_assets:
        if not asset.is_file():
            pytest.skip(f"missing accelerated EgoHand asset: {asset}")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    frames = torch.from_numpy(np.asarray(Image.open(required[-1]).convert("RGB")).copy()).cuda().unsqueeze(0).repeat(2, 1, 1, 1)
    camera = camera_from_window(pq.read_table(required[-2]))
    boxes = EgoHandDetectorConfig(required[3]).setup()(frames)
    eager = EgoHandConfig(required[0], root / "weights/soma").setup()
    expected = eager(frames, boxes, camera)
    assert expected.keypoints.xyz_camera is not None
    reference = expected.keypoints.xyz_camera.cpu()
    pixels = expected.keypoints.xy.cpu()
    del eager
    gc.collect()
    torch.cuda.empty_cache()
    accelerated = EgoHandConfig(
        required[0],
        root / "weights/soma",
        batch_size=32,
        gpu_crops=True,
        backbone_onnx=required[4],
        backbone_backend=TensorRtBackendConfig(max_batch_size=32, opt_batch_size=32, cache_dir=root / "engines"),
    ).setup()
    result = accelerated(frames, boxes, camera)
    assert result.keypoints.xyz_camera is not None
    assert result.sides == expected.sides
    assert result.scale_parameters is not None
    assert all(mesh.shape == original.shape and mesh.numel() > 0 for mesh, original in zip(result.vertices, expected.vertices, strict=True))
    errors = torch.linalg.vector_norm(result.keypoints.xyz_camera.cpu() - reference, dim=-1) * 1000.0
    assert errors.median() <= 1.0
    assert torch.quantile(errors, 0.99) <= 5.0
    assert torch.linalg.vector_norm(result.keypoints.xy.cpu() - pixels, dim=-1).median() <= 0.5
