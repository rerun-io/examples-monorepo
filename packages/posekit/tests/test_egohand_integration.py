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
