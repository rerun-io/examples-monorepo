# SPDX-License-Identifier: Apache-2.0
"""Behaviour of `colsfm.rig_calibration`, the one description of a calibrated rig.

The two stock-COLMAP baseline tools each used to carry this geometry. These
tests pin the contracts they relied on: `cam_from_rig` is `cam_i_T_cam_ref`,
the reference sensor comes first in the config and carries no transform, and a
model built from the calibration measures a zero extrinsic delta against it.
"""

from __future__ import annotations

import numpy as np
import pycolmap
import pytest
from conftest import build_reconstruction

from colsfm.frames_meta import FramesMeta
from colsfm.reconstruction import RigReference, gauge_camera_params_id, rig_reference
from colsfm.rig_calibration import (
    ExtrinsicChange,
    calibrated_cam_from_rig,
    camera_params_id_by_camera_id,
    extrinsic_deltas,
    rig_config,
    rotation_degrees,
)


@pytest.fixture(scope="module")
def galileo_reference(three_samples: FramesMeta) -> RigReference:
    """The rig origin `build_reconstruction` uses: the lowest `camera_params_id`."""
    return rig_reference(three_samples, min(three_samples.cameras))


def test_the_reference_camera_owns_the_lowest_keyframe_id(three_samples: FramesMeta) -> None:
    """`FramesMeta.rig_frames`' rule, and the one both baseline tools applied.

    `colsfm.reconstruction.gauge_camera_params_id` is that rule; this module used to
    carry a second copy under its own name, which the two audit tools imported.
    """
    chosen: int = gauge_camera_params_id(three_samples)

    assert chosen == min(three_samples.keyframes, key=lambda item: item.keyframe_id).camera_params_id


def test_the_reference_camera_sits_at_identity(three_samples: FramesMeta, galileo_reference: RigReference) -> None:
    """`cam_ref_T_cam_ref` is the identity, by construction rather than by rounding."""
    calibrated: dict[int, pycolmap.Rigid3d] = calibrated_cam_from_rig(three_samples, galileo_reference)
    assert galileo_reference.camera_params_id is not None

    np.testing.assert_allclose(calibrated[galileo_reference.camera_params_id].matrix(), np.eye(3, 4), atol=1e-12)


def test_cam_from_rig_is_cam_from_vehicle_composed_with_the_reference_bridge(
    three_samples: FramesMeta, galileo_reference: RigReference
) -> None:
    """`cam_i_T_cam_ref = cam_i_T_vehicle * vehicle_T_cam_ref`, for every camera."""
    calibrated: dict[int, pycolmap.Rigid3d] = calibrated_cam_from_rig(three_samples, galileo_reference)

    assert sorted(calibrated) == sorted(three_samples.cameras)
    for camera_params_id, camera in three_samples.cameras.items():
        expected: pycolmap.Rigid3d = camera.vehicle_T_cam.inverse() * galileo_reference.vehicle_T_reference
        np.testing.assert_allclose(calibrated[camera_params_id].matrix(), expected.matrix(), atol=1e-12)


def test_rig_config_lists_the_reference_first_and_gives_it_no_transform(
    three_samples: FramesMeta, galileo_reference: RigReference
) -> None:
    """`Rig::AddSensor` refuses to run before a reference exists, so order matters."""
    config: pycolmap.RigConfig = rig_config(three_samples, galileo_reference)
    reference_camera = three_samples.cameras[galileo_reference.camera_params_id]

    assert len(config.cameras) == len(three_samples.cameras)
    assert config.cameras[0].ref_sensor
    assert config.cameras[0].image_prefix == f"{reference_camera.sensor_name}/"
    assert config.cameras[0].cam_from_rig is None
    assert [entry.ref_sensor for entry in config.cameras[1:]] == [False] * (len(config.cameras) - 1)


def test_rig_config_carries_the_calibrated_transform_on_every_other_camera(
    three_samples: FramesMeta, galileo_reference: RigReference
) -> None:
    """Each non-reference entry carries exactly the `cam_from_rig` of the calibration."""
    calibrated: dict[int, pycolmap.Rigid3d] = calibrated_cam_from_rig(three_samples, galileo_reference)
    prefix_by_params_id: dict[int, str] = {
        camera_params_id: f"{camera.sensor_name}/" for camera_params_id, camera in three_samples.cameras.items()
    }
    params_id_by_prefix: dict[str, int] = {prefix: params_id for params_id, prefix in prefix_by_params_id.items()}

    for entry in rig_config(three_samples, galileo_reference).cameras[1:]:
        expected: pycolmap.Rigid3d = calibrated[params_id_by_prefix[entry.image_prefix]]
        np.testing.assert_allclose(entry.cam_from_rig.matrix(), expected.matrix(), atol=1e-12)


def test_rotation_degrees_of_the_identity_is_zero() -> None:
    """A rig the mapper never moved reports no rotation at all."""
    assert rotation_degrees(pycolmap.Rigid3d()) == pytest.approx(0.0, abs=1e-12)


def test_rotation_degrees_measures_the_geodesic_angle() -> None:
    """Ninety degrees about +Z reads as ninety degrees, whichever way it is signed."""
    quarter_turn: pycolmap.Rigid3d = pycolmap.Rigid3d(
        pycolmap.Rotation3d(np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])), np.zeros(3)
    )

    assert rotation_degrees(quarter_turn) == pytest.approx(90.0)
    assert rotation_degrees(quarter_turn.inverse()) == pytest.approx(90.0)


def test_camera_params_ids_come_back_through_the_image_folders(three_samples: FramesMeta) -> None:
    """COLMAP's own camera ids map back onto `camera_params_id` through the folder name."""
    reconstruction: pycolmap.Reconstruction = build_reconstruction(three_samples, error_px=0.0)

    mapping: dict[int, int] = camera_params_id_by_camera_id(reconstruction, three_samples)

    # `build_reconstruction` numbers its COLMAP cameras `camera_params_id + 1`.
    assert mapping == {camera_params_id + 1: camera_params_id for camera_params_id in three_samples.cameras}


def test_a_model_built_from_the_calibration_has_no_extrinsic_delta(three_samples: FramesMeta, galileo_reference: RigReference) -> None:
    """The comparison is against the same composition the model was built with."""
    reconstruction: pycolmap.Reconstruction = build_reconstruction(three_samples, error_px=0.0)

    deltas: tuple[ExtrinsicChange, ...] = extrinsic_deltas(reconstruction, three_samples, galileo_reference)

    assert [delta.camera_params_id for delta in deltas] == sorted(set(three_samples.cameras) - {galileo_reference.camera_params_id})
    assert [delta.sensor_name for delta in deltas] == [
        three_samples.cameras[delta.camera_params_id].sensor_name for delta in deltas
    ]
    assert max(delta.translation_change_mm for delta in deltas) == pytest.approx(0.0, abs=1e-9)
    assert max(delta.rotation_change_deg for delta in deltas) == pytest.approx(0.0, abs=1e-9)


def test_a_moved_sensor_shows_up_in_millimetres_and_degrees(three_samples: FramesMeta, galileo_reference: RigReference) -> None:
    """Displacing one `sensor_from_rig` moves exactly that camera's delta."""
    reconstruction: pycolmap.Reconstruction = build_reconstruction(three_samples, error_px=0.0)
    rig: pycolmap.Rig = next(iter(reconstruction.rigs.values()))
    moved_params_id: int = max(three_samples.cameras)
    sensor_id: pycolmap.sensor_t = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, moved_params_id + 1)
    original: pycolmap.Rigid3d = rig.sensor_from_rig(sensor_id)
    nudge: pycolmap.Rigid3d = pycolmap.Rigid3d(pycolmap.Rotation3d(), np.array([0.005, 0.0, 0.0]))
    rig.set_sensor_from_rig(sensor_id, nudge * original)

    by_params_id: dict[int, ExtrinsicChange] = {
        delta.camera_params_id: delta for delta in extrinsic_deltas(reconstruction, three_samples, galileo_reference)
    }

    assert by_params_id[moved_params_id].translation_change_mm == pytest.approx(5.0, abs=1e-6)
    assert by_params_id[moved_params_id].rotation_change_deg == pytest.approx(0.0, abs=1e-9)
    others: list[float] = [
        delta.translation_change_mm for params_id, delta in by_params_id.items() if params_id != moved_params_id
    ]
    assert max(others) == pytest.approx(0.0, abs=1e-9)
