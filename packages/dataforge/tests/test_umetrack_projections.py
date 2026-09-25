"""Independent FishEye62 reference and derived projection contracts."""

import math
from pathlib import Path

import numpy as np
from conftest import read_chunks
from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion, apply_radial_tangential_distortion
from simplecv.umetrack_temp.cameras import Camera

from dataforge import hands, schema, writing
from dataforge.datasets.umetrack import pane_contents
from dataforge.datasets.umetrack_layers import project_fisheye62


def test_scalar_reference_agrees_with_simplecv_distortion() -> None:
    # Scalar FishEye62 definition, including tangential distortion AFTER radial.
    x, y, z = 0.3, -0.2, 0.8
    radial = (0.1, -0.02, 0.003, -0.0004, 0.00005, -0.000006)
    p1, p2 = 0.002, -0.003
    radius = math.hypot(x, y)
    theta = math.atan2(radius, z)
    theta_d = theta * (1 + sum(k * theta ** (2 * i) for i, k in enumerate(radial, 1)))
    xr, yr = theta_d * x / radius, theta_d * y / radius
    rho2 = xr * xr + yr * yr
    expected = [xr + 2 * p2 * xr * yr + p1 * (rho2 + 2 * xr * xr), yr + 2 * p1 * xr * yr + p2 * (rho2 + 2 * yr * yr)]
    lens = KannalaBrandtDistortion(k1=radial[0], k2=radial[1], k3=radial[2], k4=radial[3], k5=radial[4], k6=radial[5], p1=p1, p2=p2)
    actual = apply_radial_tangential_distortion(lens, np.array([[theta * x / radius, theta * y / radius]], dtype=np.float64))
    np.testing.assert_allclose(actual[0], expected, atol=1e-14, rtol=0)


def test_projection_matches_scalar_and_camera_reference() -> None:

    lens = KannalaBrandtDistortion(k1=0.1, k2=-0.02, k3=0.003, k4=-0.0004, k5=0.00005, k6=-0.000006, p1=0.002, p2=-0.003)
    camera = Fisheye62Parameters(
        name="real",
        distortion=lens,
        extrinsics=Extrinsics(world_R_cam=np.eye(3), world_t_cam=np.zeros(3)),
        intrinsics=Intrinsics.from_focal_principal_point(camera_conventions="RDF", fl_x=220.0, fl_y=225.0, cx=318.0, cy=240.0, width=636, height=480),
    )
    point = np.array([[0.3, -0.2, 0.8]], dtype=np.float64)
    x, y, z = point[0]
    r = math.hypot(x, y)
    theta = math.atan2(r, z)
    theta_d = theta * (1 + sum(k * theta ** (2 * i) for i, k in enumerate((lens.k1, lens.k2, lens.k3, lens.k4, lens.k5, lens.k6), 1)))
    xr, yr = theta_d * x / r, theta_d * y / r
    rho2 = xr * xr + yr * yr
    expected = [
        220.0 * (xr + 2 * lens.p2 * xr * yr + lens.p1 * (rho2 + 2 * xr * xr)) + 318.0,
        225.0 * (yr + 2 * lens.p1 * xr * yr + lens.p2 * (rho2 + 2 * yr * yr)) + 240.0,
    ]
    pixels, valid = project_fisheye62(point, camera)
    assert valid.tolist() == [True]
    np.testing.assert_allclose(pixels[0], expected, atol=1e-6, rtol=0)
    # The legacy wrapper returns float32 pixels; compare at its storage precision.
    np.testing.assert_allclose(
        pixels.astype(np.float32), Camera(camera).camera_to_image(point.astype(np.float32)), atol=float(np.spacing(np.float32(expected[0]))), rtol=0
    )


def test_invalid_projections_and_writer_confidence(tmp_path: Path) -> None:

    camera = Fisheye62Parameters(
        name="folding",
        distortion=KannalaBrandtDistortion(k1=-1.0, k2=0.0, k3=0.0, k4=0.0, k5=0.0, k6=0.0, p1=0.0, p2=0.0),
        extrinsics=Extrinsics(world_R_cam=np.eye(3), world_t_cam=np.zeros(3)),
        intrinsics=Intrinsics.from_focal_principal_point(camera_conventions="RDF", fl_x=220.0, fl_y=225.0, cx=50.0, cy=50.0, width=100, height=100),
    )
    # theta_max = sqrt(1/3); the folded ray maps BACK into the image and must still be rejected.
    points = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [math.sin(1.0), 0, math.cos(1.0)], [0.4, 0, 1], [np.nan, 0, 1]], dtype=np.float64)
    pixels, valid = project_fisheye62(points, camera)
    assert valid.tolist() == [True, False, False, False, False, False]
    np.testing.assert_array_equal(pixels[0], [50, 50])
    assert np.isnan(pixels[1:]).all()
    positions = np.full((1, 133, 2), np.nan, dtype=np.float32)
    positions[0, :6] = pixels
    target = tmp_path / "projected.rrd"
    times = np.array([123], dtype=np.int64)
    frames = np.array([7], dtype=np.int64)
    with writing.atomic_recording(target, recording_id="test", send_properties=False) as recording:
        hands.log_keypoints2d(
            recording,
            0,
            2,
            path=schema.coco133_uv_projected_path(0, 2),
            times_ns=times,
            frame_indices=frames,
            positions=positions,
            confidence=np.full((1, 133), 0.75, dtype=np.float32),
        )
    chunks = read_chunks(target)
    chunk = next(c for c in chunks if not c.is_static)
    assert str(chunk.entity_path) == schema.coco133_uv_projected_path(0, 2)
    batch = chunk.to_record_batch()
    scores = batch.column(next(n for n in batch.schema.names if n.endswith(":confidences"))).to_pylist()[0]
    assert scores == [0.75] + [0.0] * 132
    assert batch.column("frame_index").to_pylist() == [7]
    assert batch.column("video_time").cast("int64").to_pylist() == [123]


def test_panes_include_only_own_video_and_projections() -> None:

    for camera in range(4):
        contents = pane_contents(camera)
        assert "+ /world/**" not in contents
        assert {entry for entry in contents if entry.startswith("+")} == {
            f"+ {schema.video_path(0, camera)}",
            f"+ {schema.coco133_uv_projected_path(0, camera)}",
        }
