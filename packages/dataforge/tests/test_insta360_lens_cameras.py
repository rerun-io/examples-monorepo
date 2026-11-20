"""The lens cameras of a base read the same whichever Mei name the base was written with (exoego:v2 section 4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
from simplecv.rerun_custom_types import CameraDistortion, PinholeWithDistortion

from dataforge.datasets.insta360 import LensCameras, cameras_from_recording

GAMMA: tuple[float, float] = (1835.37, 1835.01)
COEFFICIENTS: list[float] = [2.31494, 1.7453292, 0.94, -1.68, 2.19, 3.86, 0.0, -0.005, 0.0016, -0.0033, 0.004, 0.0083, -0.0013, 0.024, -0.0014]


def write_base(path: Path, *, unified: bool) -> Path:
    """Two lenses as InsvReader writes them: the old name with gamma in the Pinhole, or the unified name with gamma in
    the coefficients and the paraxial focal in the Pinhole."""
    recording: rr.RecordingStream = rr.RecordingStream("dataforge", recording_id="test")
    recording.save(path)
    scale: float = 1.0 + COEFFICIENTS[0] if unified else 1.0
    for cam in range(2):
        node: str = f"/world/rig_00/cam_0{cam}"
        recording.log(node, rr.Transform3D(mat3x3=np.eye(3), translation=[0.0, 0.0, 0.016 * cam], relation=rr.TransformRelation.ChildFromParent), static=True)
        k: list[list[float]] = [[GAMMA[0] / scale, 0.0, 958.29], [0.0, GAMMA[1] / scale, 965.87], [0.0, 0.0, 1.0]]
        model, coefficients = ("mei_v6_unified", [*GAMMA, *COEFFICIENTS]) if unified else ("mei_v6", COEFFICIENTS)
        recording.log(
            f"{node}/pinhole",
            PinholeWithDistortion(rr.Pinhole(image_from_camera=k, width=1920, height=1920), CameraDistortion(model, np.asarray(coefficients))),
            static=True,
        )
    recording.flush()
    recording.disconnect()
    return path


@pytest.mark.parametrize("unified", [False, True])
def test_either_mei_name_gives_the_same_lens_cameras(tmp_path: Path, unified: bool) -> None:
    lenses: LensCameras = cameras_from_recording(write_base(tmp_path / "base.rrd", unified=unified))
    for camera in lenses.cameras:
        assert camera.model == "mei_v6"
        np.testing.assert_allclose([camera.fx, camera.fy, camera.cx, camera.cy], [*GAMMA, 958.29, 965.87], rtol=1e-6)
        np.testing.assert_allclose(camera.coefficients, COEFFICIENTS, rtol=1e-6, atol=1e-7)
    assert lenses.size == (1920, 1920)
