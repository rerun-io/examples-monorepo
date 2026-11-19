"""A recording file's camera calibration reads back through the catalog decoder, without a catalog."""

from pathlib import Path

import numpy as np
import pytest

rr = pytest.importorskip("rerun", reason="rerun-sdk is required by this test module")

from simplecv.camera_parameters import BrownConradyDistortion, Extrinsics, Intrinsics, PinholeParameters  # noqa: E402
from simplecv.catalog_calibration import read_camera_models_from_store  # noqa: E402
from simplecv.rerun_log_utils import log_pinhole  # noqa: E402


def test_a_file_store_reads_back_what_log_pinhole_wrote(tmp_path: Path) -> None:
    import rerun.chunk as rrc

    k_matrix: np.ndarray = np.array([[271.7, 0.0, 231.3], [0.0, 271.6, 314.5], [0.0, 0.0, 1.0]])
    cam_R_world: np.ndarray = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    camera: PinholeParameters = PinholeParameters(
        name="cam_00",
        intrinsics=Intrinsics(camera_conventions="RDF", k_matrix=k_matrix, width=480, height=640),
        extrinsics=Extrinsics(cam_R_world=cam_R_world, cam_t_world=np.array([0.1, 0.0, -0.05])),
    )
    camera.distortion = BrownConradyDistortion(k1=0.16, k2=-0.46, p1=-6e-05, p2=-0.00012, k3=0.026)
    path: Path = tmp_path / "segment.rrd"
    with rr.RecordingStream("calibration", recording_id="segment") as recording:
        recording.save(path)
        log_pinhole(camera, Path("/world/rig_00/cam_00"), static=True, recording=recording)
    model = read_camera_models_from_store(rrc.RrdReader(path).store(), ["/world/rig_00/cam_00"], str(path))["/world/rig_00/cam_00"]
    assert model.distortion is not None and model.distortion.model == "brown_conrady"
    np.testing.assert_allclose(model.distortion.coefficients[:5], [0.16, -0.46, -6e-05, -0.00012, 0.026], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(model.pinhole.intrinsics.k_matrix, k_matrix, rtol=1e-6)
    np.testing.assert_allclose(model.pinhole.extrinsics.cam_R_world, cam_R_world, atol=1e-7)
    np.testing.assert_allclose(model.pinhole.extrinsics.cam_t_world, [0.1, 0.0, -0.05], atol=1e-7)
    assert (model.pinhole.intrinsics.width, model.pinhole.intrinsics.height) == (480, 640)
    with pytest.raises(ValueError, match="cam_01"):
        read_camera_models_from_store(rrc.RrdReader(path).store(), ["/world/rig_00/cam_01"], str(path))
