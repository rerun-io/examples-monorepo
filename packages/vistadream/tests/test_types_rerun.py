"""to_rerun -> catalog -> from_layer through an in-process Rerun server (integration lane: a server is a catalog)."""

import numpy as np
import pytest
import torch
from dagkit.types import Type, rerun_roundtrip
from serde import to_dict

from vistadream.types import CameraView, OutpaintCanvas, RelativeDepth, RgbImage, SceneViews, Splats

pytestmark = pytest.mark.integration

RNG = np.random.default_rng(3)
K = np.array([[60.0, 0.0, 16.0], [0.0, 60.0, 12.0], [0.0, 0.0, 1.0]])


def test_host_types_round_trip_through_a_catalog() -> None:
    rgb = RNG.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    depth = RNG.uniform(0.5, 3.0, (24, 32)).astype(np.float32)
    mask = RNG.random((24, 32)) > 0.5
    pose = np.eye(4)
    pose[:3, 3] = [0.125, -0.25, 0.5]  # float32-exact: Rerun stores transforms as float32
    view = CameraView(rgb, K, pose, depth, mask, ~mask)
    values: list[tuple[Type, str]] = [
        (RgbImage(rgb), "/image"),
        (OutpaintCanvas(rgb, mask, np.array([1, 2, 30, 20], dtype=np.int64)), "/canvas"),
        (RelativeDepth(depth, mask, K), "/depth"),
        (view, "/world/input"),
        (SceneViews((CameraView(rgb, K, np.eye(4), depth, mask, mask), view), 1), "/world"),
    ]
    for value, root in values:
        [(t, back)] = rerun_roundtrip(value, root, "video_time", 33_000_000)
        assert t == 33_000_000 and type(back) is type(value)
        np.testing.assert_equal(to_dict(back), to_dict(value))


def test_splats_round_trip_through_a_catalog() -> None:
    n = 50
    value = Splats(
        torch.as_tensor(RNG.normal(size=(n, 3)), dtype=torch.float32),
        torch.as_tensor(RNG.uniform(0.01, 0.1, (n, 3)), dtype=torch.float32),
        torch.nn.functional.normalize(torch.as_tensor(RNG.normal(size=(n, 4)), dtype=torch.float32), dim=1),
        torch.as_tensor(RNG.uniform(0, 1, (n, 3)), dtype=torch.float32),
        torch.as_tensor(RNG.uniform(0, 1, n), dtype=torch.float32),
    )
    [(_, back)] = rerun_roundtrip(value, "/world/splats", "video_time", 0)
    torch.testing.assert_close(back.centers, value.centers)
    torch.testing.assert_close(back.quats_wxyz, value.quats_wxyz)
    torch.testing.assert_close(back.rgb, value.rgb, atol=0.5 / 255 + 1e-6, rtol=0.0)
