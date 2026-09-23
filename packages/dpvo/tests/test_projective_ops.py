import pytest
from beartype.roar import BeartypeException

torch = pytest.importorskip("torch", reason="DPVO requires PyTorch")
pytest.importorskip("lietorch", reason="DPVO requires lietorch")

from dpvo.projective_ops import iproj, proj  # noqa: E402


def test_proj_rejects_points_without_patch_axes() -> None:
    with pytest.raises(BeartypeException):
        proj(torch.ones((1, 5, 4)), torch.ones((1, 5, 4)))


@pytest.mark.parametrize("depth", [False, True])
def test_proj_preserves_patch_axes_and_pixel_coordinates(depth: bool) -> None:
    points = torch.tensor([2.0, 3.0, 2.0, 1.0]).expand(2, 5, 3, 3, 4)
    intrinsics = torch.tensor([4.0, 6.0, 10.0, 20.0]).expand(2, 5, 4)
    expected = torch.tensor([14.0, 29.0, 0.5] if depth else [14.0, 29.0])

    actual = proj(points, intrinsics, depth=depth)

    torch.testing.assert_close(actual, expected.expand(2, 5, 3, 3, len(expected)))


def test_iproj_round_trip_preserves_patch_coordinates() -> None:
    patches = torch.tensor([14.0, 29.0, 0.5]).view(1, 1, 3, 1, 1).expand(2, 5, 3, 3, 3)
    intrinsics = torch.tensor([4.0, 6.0, 10.0, 20.0]).expand(2, 5, 4)

    points = iproj(patches, intrinsics)

    assert points.shape == (2, 5, 3, 3, 4)
    torch.testing.assert_close(proj(points, intrinsics), patches[:, :, :2].permute(0, 1, 3, 4, 2))
