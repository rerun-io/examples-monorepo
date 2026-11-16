"""The calibration a camera node carries comes back out of the catalog as it was logged, distortion included."""

from pathlib import Path

import numpy as np
import pytest

rr = pytest.importorskip("rerun", reason="rerun-sdk is required by this test module")
pytest.importorskip("rerun.catalog", reason="rerun.catalog is required by this test module")

from simplecv.camera_parameters import BrownConradyDistortion, Extrinsics, Intrinsics, PinholeParameters  # noqa: E402
from simplecv.catalog_calibration import read_camera_calibration, read_camera_models  # noqa: E402
from simplecv.rerun_log_utils import log_pinhole  # noqa: E402

pytestmark = pytest.mark.integration


def _camera(name: str, distortion: BrownConradyDistortion | None) -> PinholeParameters:
    k_matrix = np.array([[271.7, 0.0, 231.3], [0.0, 271.6, 314.5], [0.0, 0.0, 1.0]])
    camera = PinholeParameters(
        name=name,
        intrinsics=Intrinsics(camera_conventions="RDF", k_matrix=k_matrix, width=480, height=640),
        extrinsics=Extrinsics(cam_R_world=np.eye(3), cam_t_world=np.array([0.1, 0.0, -0.05])),
    )
    camera.distortion = distortion
    return camera


def test_a_logged_distortion_reads_back_and_an_undistorted_camera_has_none(tmp_path: Path) -> None:
    """MGO07-style FULL_OPENCV coefficients (brown_conrady k1..k6) survive the round trip at float32 precision."""
    distortion = BrownConradyDistortion(k1=0.16062, k2=-0.46199, p1=-6e-05, p2=-0.00012, k3=0.02577, k4=0.4509, k5=-0.58754, k6=0.00296)
    path = tmp_path / "segment.rrd"
    with rr.RecordingStream("calibration", recording_id="segment") as recording:
        recording.save(path)
        log_pinhole(_camera("cam_00", distortion), Path("/world/rig_00/cam_00"), static=True, recording=recording)
        log_pinhole(_camera("cam_01", None), Path("/world/rig_00/cam_01"), static=True, recording=recording)
    entities = ["/world/rig_00/cam_00", "/world/rig_00/cam_01"]
    with rr.server.Server(datasets={"calibration": [path]}) as server:
        dataset = server.client().get_dataset("calibration")
        models = read_camera_models(dataset, "segment", entities)
        pinholes = read_camera_calibration(dataset, "segment", entities)
    distorted, plain = models["/world/rig_00/cam_00"], models["/world/rig_00/cam_01"]
    assert plain.distortion is None
    assert distorted.distortion is not None and distorted.distortion.model == "brown_conrady"
    expected = [distortion.k1, distortion.k2, distortion.p1, distortion.p2, distortion.k3, distortion.k4, distortion.k5, distortion.k6]
    np.testing.assert_allclose(distorted.distortion.coefficients[:8], expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(distorted.pinhole.intrinsics.k_matrix, _camera("x", None).intrinsics.k_matrix, rtol=1e-6)
    np.testing.assert_allclose(distorted.pinhole.extrinsics.cam_t_world, [0.1, 0.0, -0.05], atol=1e-7)
    assert (distorted.pinhole.intrinsics.width, distorted.pinhole.intrinsics.height) == (480, 640)
    np.testing.assert_allclose(pinholes["/world/rig_00/cam_00"].intrinsics.k_matrix, distorted.pinhole.intrinsics.k_matrix)
