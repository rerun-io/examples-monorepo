"""Perspective sampling matches OpenCV's uint8 interpolation and black border."""

import cv2
import numpy as np
import pytest
import torch

from posekit.ops.crops import remap_frames


def test_perspective_remap_matches_opencv_with_mirrored_grid() -> None:
    rng = np.random.default_rng(42)
    frames = rng.integers(0, 256, (2, 19, 23, 3), dtype=np.uint8)
    coordinates = rng.uniform((-2, -2), (25, 21), (3, 13, 17, 2)).astype(np.float32)
    coordinates[1] = coordinates[1, :, ::-1]
    indices = np.array([1, 0, 1], dtype=np.int64)
    expected = np.stack(
        [
            cv2.remap(frames[i], grid[..., 0], grid[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            for i, grid in zip(indices, coordinates, strict=True)
        ]
    )
    result = remap_frames(torch.from_numpy(frames), torch.from_numpy(indices), torch.from_numpy(coordinates))
    np.testing.assert_array_equal(result.permute(0, 2, 3, 1).numpy(), expected)


@pytest.mark.integration
def test_gpu_perspective_crops_match_reference() -> None:
    pytest.importorskip("egohand_inference", reason="requires the egohand Pixi environment")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    from egohand_inference.camera.perspective_crop import perspective_crop

    from posekit.models.egohand_crops import prepare_gpu_crops
    from posekit.models.egohand_geometry import OAKCamera

    rng = np.random.default_rng(19)
    rgb = rng.integers(0, 256, (1, 72, 128, 3), dtype=np.uint8)
    camera = OAKCamera(np.array([[110.0, 0.0, 64.0], [0.0, 110.0, 36.0], [0.0, 0.0, 1.0]]), np.zeros(14), 128, 72)
    box = np.array([102.0, 30.0, 45.0], dtype=np.float32)
    prepared = prepare_gpu_crops(torch.from_numpy(rgb).cuda(), {"0": {"right": box}}, camera)
    expected, _, _, _ = perspective_crop(rgb[0, :, :, ::-1].copy(), box, camera.model, 256)
    expected = (expected[:, ::-1] - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    difference = np.abs(prepared[0]["crop_tensor"].cpu().numpy() - expected.transpose(2, 0, 1))
    # CPU/GPU arithmetic can straddle uint8 half-rounding at a few pixels.
    assert np.quantile(difference, 0.999) < 2e-6
    assert difference.max() <= 1.0 / (255.0 * 0.225) + 2e-6
