"""The synthetic bundle-adjustment benchmark generator at a tiny size."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pycolmap
import pytest

if TYPE_CHECKING:
    from colsfm.apis.ba_synth import SynthCameraModel


@pytest.mark.parametrize("model", ["PINHOLE", "SIMPLE_RADIAL", "OPENCV_FISHEYE", "FULL_OPENCV"])
def test_tiny_model_is_perturbed_with_distortion(model: SynthCameraModel) -> None:
    from colsfm.apis.ba_synth import DISTORTION, synthetic_reconstruction

    reconstruction = synthetic_reconstruction(5, 300, model)
    camera = reconstruction.camera(1)
    assert camera.model.name == model
    np.testing.assert_array_equal(camera.params[len(camera.params) - len(DISTORTION[model]) :], DISTORTION[model])
    assert reconstruction.num_reg_frames() == 5 and reconstruction.num_points3D() > 50
    reconstruction.update_point_3d_errors()
    # Perturbed start, well above the 0.5 px pixel noise.
    assert reconstruction.compute_mean_reprojection_error() > 2.0


def test_rig_second_camera_is_not_identity() -> None:
    from colsfm.apis.ba_synth import synthetic_reconstruction

    reconstruction = synthetic_reconstruction(4, 300, num_cameras=2)
    rig = reconstruction.rig(1)
    second = rig.sensor_from_rig(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, 2))
    assert second is not None and np.linalg.norm(np.asarray(second.translation)) > 0.2
    assert reconstruction.num_images() == 8 and all(len(reconstruction.frame(f).image_ids) == 2 for f in reconstruction.frames)


def test_same_seed_same_model(tmp_path: Path) -> None:
    from colsfm.apis.ba_synth import Config, main

    main(Config(output_dir=tmp_path / "a", num_frames=4, num_points=200))
    main(Config(output_dir=tmp_path / "b", num_frames=4, num_points=200))
    a, b = pycolmap.Reconstruction(tmp_path / "a"), pycolmap.Reconstruction(tmp_path / "b")
    assert a.num_points3D() == b.num_points3D()
    for point3D_id in a.point3D_ids():
        np.testing.assert_array_equal(a.point3D(point3D_id).xyz, b.point3D(point3D_id).xyz)
