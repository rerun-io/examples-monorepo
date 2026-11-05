"""Sparse feature sampling retains grid_sample's boundary and coordinate rules."""

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("channels", [3, 1280])
def test_sparse_keypoint_sampling_matches_grid_sample(batch: int, channels: int) -> None:
    from posekit.models.sam3d_body_keypoints import sample_keypoint_features

    torch.manual_seed(19)
    features = torch.randn(batch, channels, 32, 32)
    points = torch.rand(batch, 70, 2) * 2.6 - 1.3
    points[:, :4] = torch.tensor([[-1., -1.], [1., 1.], [0., 0.], [1.3, -1.3]])
    expected = F.grid_sample(features, points[:, :, None], mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(3).transpose(1, 2)
    actual = sample_keypoint_features(features, points)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=0.0)


@pytest.mark.integration
def test_sparse_keypoint_sampling_cuda() -> None:
    from posekit.models.sam3d_body_keypoints import sample_keypoint_features

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    torch.manual_seed(31)
    features = torch.randn(2, 1280, 32, 32, device="cuda")
    points = torch.rand(2, 70, 2, device="cuda") * 2.6 - 1.3
    expected = F.grid_sample(features, points[:, :, None], mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(3).transpose(1, 2)
    actual = sample_keypoint_features(features, points)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=0.0)
