"""UmeTrack ingests upright: a camera turned by whole quarter turns projects exactly where the turned image shows the point."""

import numpy as np
import pytest
from jaxtyping import Float64
from numpy import ndarray
from simplecv.sensors.camera.fisheye62 import project_fisheye62

from dataforge.datasets.umetrack_layers import fisheye_camera
from dataforge.datasets.umetrack_source import Camera, quarter_turn_rotation, quarter_turns_from_angle, turn_camera

REAL: Camera = Camera(ImageSizeX=636, ImageSizeY=480, fx=237.3, fy=237.5, cx=316.8, cy=239.6, DistortionModel="FishEye62",
                      k1=-0.0207, k2=0.1058, k3=-0.0927, k4=0.0332, p1=-0.00073, p2=0.00056, k5=-0.0066, k6=0.0011)


def turn_pixels(uv: Float64[ndarray, "n 2"], width: int, height: int, turns: int) -> Float64[ndarray, "n 2"]:
    """Where pixel centres land after ``turns`` clockwise turns of a W x H image: (u, v) -> (H - 1 - v, u) per turn."""
    for _ in range(turns):
        uv = np.stack((height - 1 - uv[:, 1], uv[:, 0]), axis=1)
        width, height = height, width
    return uv


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_a_turned_camera_projects_like_the_turned_image(turns: int) -> None:
    rng = np.random.default_rng(turns)
    xyz: Float64[ndarray, "n 3"] = np.column_stack((rng.uniform(-0.6, 0.6, (500, 2)), rng.uniform(0.3, 1.0, 500)))
    original: Float64[ndarray, "n 2"] = project_fisheye62(xyz, fisheye_camera(REAL, np.eye(4)))
    expected: Float64[ndarray, "n 2"] = turn_pixels(original, REAL.ImageSizeX, REAL.ImageSizeY, turns)
    turned_camera: Camera = turn_camera(REAL, turns)
    rotated: Float64[ndarray, "n 3"] = xyz @ quarter_turn_rotation(turns).T  # the same points in the turned camera's axes
    observed: Float64[ndarray, "n 2"] = project_fisheye62(rotated, fisheye_camera(turned_camera, np.eye(4)))
    seen = np.isfinite(original).all(axis=1) & np.isfinite(observed).all(axis=1)
    assert seen.sum() > 300
    np.testing.assert_allclose(observed[seen], expected[seen], atol=1e-9)
    # The image bound is half-open, [0, W): a turned pixel centre sits in (-1, W - 1], so only points within a pixel of the
    # border may be in one image and not the other.
    disagree = np.isfinite(original).all(axis=1) != np.isfinite(observed).all(axis=1)
    near_border = (np.nan_to_num(expected, nan=0.0) < 1.0).any(axis=1) | (np.nan_to_num(observed, nan=0.0) < 1.0).any(axis=1) | ~np.isfinite(observed).all(axis=1)
    assert disagree.sum() <= 5 and bool(near_border[disagree].all())
    assert (turned_camera.ImageSizeX, turned_camera.ImageSizeY) == ((480, 636) if turns % 2 else (636, 480))


def test_the_release_angles_name_the_turns_and_leave_no_roll() -> None:
    # UmeTrack's camera_angles for its four cameras: 0, 90, 90, 180 degrees.
    assert [quarter_turns_from_angle(a) for a in (0.0, 90.0, 90.0, 180.0)] == [0, 3, 3, 2]


def test_one_gpu_graph_crops_turns_and_names_every_camera() -> None:
    from dataforge.video_encoding import gpu_tile_graph

    crops = [(636, 480, 636 * k, 0) for k in range(4)]
    downloaded = gpu_tile_graph(crops, [0, 3, 3, 2], crop_on_gpu=False)
    assert downloaded == (
        "[0:v]hwdownload,format=nv12,split=4[tile0][tile1][tile2][tile3];"
        "[tile0]crop=636:480:0:0,hwupload_cuda[cam0];"
        "[tile1]crop=636:480:636:0,hwupload_cuda,transpose_cuda=dir=cclock[cam1];"
        "[tile2]crop=636:480:1272:0,hwupload_cuda,transpose_cuda=dir=cclock[cam2];"
        "[tile3]crop=636:480:1908:0,hwupload_cuda,transpose_cuda=dir=reversal[cam3]"
    )
    on_gpu = gpu_tile_graph(crops[:2], [0, 1], crop_on_gpu=True)
    assert on_gpu == "[0:v]split=2[tile0][tile1];[tile0]crop=636:480:0:0,scale_cuda=format=nv12[cam0];[tile1]crop=636:480:636:0,transpose_cuda=dir=clock[cam1]"


def test_a_turn_rebuild_can_skip_the_world_space_layers() -> None:
    """The upright turn changes the base layer (video, cameras) and the projections; hand_pose and hand_mesh are world-space."""
    from dataforge import paths
    from dataforge.datasets.umetrack import UmetrackConfig
    from dataforge.identity import SequenceIdentity

    dataset = UmetrackConfig(only_layers=(paths.BASE_LAYER, paths.PROJECTIONS_LAYER)).setup()
    identity = SequenceIdentity("umetrack", ("real", "hand_hand", "training", "user_03", "recording_05"))
    assert set(dataset.targets(identity)) == {paths.BASE_LAYER, paths.PROJECTIONS_LAYER}
