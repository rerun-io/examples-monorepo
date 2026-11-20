"""A Mei (unified sphere) camera as exoego:v2 logs it: the typed model in the distortion components, a paraxial Pinhole.

The X4 Air's lens 0 is the worked example: gamma 1835.37 / 1835.01 px, xi 2.31494, so the lens projects
gamma / (1 + xi) = 553.67 / 553.56 px per radian at its axis.
"""

from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float64
from numpy import ndarray

rr = pytest.importorskip("rerun", reason="rerun-sdk is required by this test module")

from simplecv.camera_parameters import MEI_CATALOG_MODELS, MeiParameters  # noqa: E402
from simplecv.catalog_calibration import read_camera_models_from_store  # noqa: E402
from simplecv.rerun_custom_types import PinholeWithDistortion  # noqa: E402

GAMMA: tuple[float, float] = (1835.37, 1835.01)
CENTRE: tuple[float, float] = (958.29, 965.87)
XI: float = 2.31494
HALF_FIELD_RAD: float = 1.7453292
D: list[float] = [0.9386, -1.6840, 2.1917, 3.8576, 0.0, -0.00496, 0.00156, -0.00331, 0.0, 0.0, 0.0, 0.0, 0.0]


def lens0() -> MeiParameters:
    return MeiParameters(
        name="cam_00", model="mei_v6", width=1920, height=1920, gamma_x=GAMMA[0], gamma_y=GAMMA[1], cx=CENTRE[0], cy=CENTRE[1],
        xi=XI, half_field_rad=HALF_FIELD_RAD, distortion=np.array(D, dtype=np.float64),
    )


def paraxial_k(camera: MeiParameters) -> Float64[ndarray, "3 3"]:
    """The viewer pinhole's matrix."""
    k: Float64[ndarray, "3 3"] | None = camera.paraxial_intrinsics().k_matrix
    assert k is not None
    return k


def k_matrix(fx: float, fy: float) -> Float64[ndarray, "3 3"]:
    return np.array([[fx, 0.0, CENTRE[0]], [0.0, fy, CENTRE[1]], [0.0, 0.0, 1.0]])


def test_the_viewer_pinhole_is_the_paraxial_focal() -> None:
    """A ray at a small angle t lands gamma * t / (1 + xi) pixels out: the Pinhole a viewer draws takes that focal."""
    camera: MeiParameters = lens0()
    np.testing.assert_allclose(np.diag(paraxial_k(camera))[:2], [553.67, 553.56], atol=0.01)
    np.testing.assert_allclose(paraxial_k(camera)[:2, 2], CENTRE)
    assert (camera.paraxial_intrinsics().width, camera.paraxial_intrinsics().height) == (1920, 1920)


def test_the_unified_name_carries_gamma_in_its_coefficients() -> None:
    """`mei_v6_unified` coefficients are (gamma_x, gamma_y, xi, half_field_rad, d0 .. d12); `mei_v3_unified` the same
    with k1 k2 k3 p1 p2. They read back to the same camera."""
    camera: MeiParameters = lens0()
    model, coefficients = camera.catalog_distortion()
    assert model == "mei_v6_unified" and MEI_CATALOG_MODELS == ("mei_v3", "mei_v6", "mei_v3_unified", "mei_v6_unified")
    np.testing.assert_allclose(coefficients, [*GAMMA, XI, HALF_FIELD_RAD, *D])
    back: MeiParameters = MeiParameters.from_catalog(model, coefficients, paraxial_k(camera), 1920, 1920, name="cam_00")
    assert back == camera


def test_the_old_name_reads_gamma_off_the_pinhole() -> None:
    """A base written before the unified name (`mei_v6`: xi, half_field_rad, d0 .. d12) states gamma as the Pinhole focal."""
    old: MeiParameters = MeiParameters.from_catalog("mei_v6", np.array([XI, HALF_FIELD_RAD, *D]), k_matrix(*GAMMA), 1920, 1920, name="cam_00")
    assert old == lens0()
    v3: MeiParameters = MeiParameters.from_catalog("mei_v3", np.array([XI, HALF_FIELD_RAD, 0.1, -0.05, 0.01, 0.02, -0.01]), k_matrix(*GAMMA), 1920, 1920)
    unified: MeiParameters = MeiParameters.from_catalog("mei_v3_unified", np.array(v3.catalog_distortion()[1]), paraxial_k(v3), 1920, 1920)
    assert v3.model == unified.model == "mei_v3" and unified == v3


@pytest.mark.parametrize(("model", "count"), [("mei_v6", 14), ("mei_v6_unified", 16), ("mei_v3_unified", 8)])
def test_a_short_coefficient_list_is_refused(model: str, count: int) -> None:
    with pytest.raises(ValueError, match=model):
        MeiParameters.from_catalog(model, np.zeros(count), k_matrix(*GAMMA), 1920, 1920)


def test_a_logged_mei_camera_reads_back_typed_from_a_file(tmp_path: Path) -> None:
    """Logged through `PinholeWithDistortion.from_camera`, the camera reads back as the same `MeiParameters`, and its
    Pinhole is the paraxial one. A non-Mei camera has no `mei`."""
    camera: MeiParameters = lens0()
    path: Path = tmp_path / "segment.rrd"
    with rr.RecordingStream("calibration", recording_id="segment") as recording:
        recording.save(path)
        recording.log("/world/rig_00/cam_00", rr.Transform3D(mat3x3=np.eye(3), translation=[0.0, 0.0, 0.0], relation=rr.TransformRelation.ChildFromParent), static=True)
        recording.log("/world/rig_00/cam_00/pinhole", PinholeWithDistortion.from_camera(camera, image_plane_distance=0.05), static=True)
    import rerun.chunk as rrc

    model = read_camera_models_from_store(rrc.RrdReader(path).store(), ["/world/rig_00/cam_00"], str(path))["/world/rig_00/cam_00"]
    assert model.distortion is not None and model.distortion.model == "mei_v6_unified"
    k: Float64[ndarray, "3 3"] | None = model.pinhole.intrinsics.k_matrix
    assert k is not None
    np.testing.assert_allclose(np.diag(k)[:2], [553.67, 553.56], atol=0.01)
    assert model.mei is not None
    # float32 in the file.
    np.testing.assert_allclose([model.mei.gamma_x, model.mei.gamma_y, model.mei.xi, model.mei.half_field_rad], [*GAMMA, XI, HALF_FIELD_RAD], rtol=1e-6)
    np.testing.assert_allclose(model.mei.distortion, D, rtol=1e-6, atol=1e-7)
    assert model.mei.model == "mei_v6"
