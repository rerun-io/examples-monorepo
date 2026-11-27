"""A calibration layer: `write_calibration_layer` writes cameras, `read_catalog_cameras(..., calibration_entity=...)` reads them back."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from colsfm.catalog import (
    CALIBRATION_ENTITY,
    CatalogCameraModel,
    CatalogInputConfig,
    read_catalog_cameras,
    serve_catalog_input,
    write_calibration_layer,
)


def _pose(yaw_deg: float, translation: list[float]) -> np.ndarray:
    yaw: float = np.deg2rad(yaw_deg)
    pose: np.ndarray = np.eye(4)
    pose[:3, :3] = [[np.cos(yaw), 0.0, np.sin(yaw)], [0.0, 1.0, 0.0], [-np.sin(yaw), 0.0, np.cos(yaw)]]
    pose[:3, 3] = translation
    return pose


RECORDED: tuple[CatalogCameraModel, ...] = (
    CatalogCameraModel(
        entity="/world/rig_00/cam_00",
        name="left_front",
        rig_T_cam=_pose(0.0, [0.0, 0.0, 0.0]),
        k_matrix=np.array([[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]]),
        width=640,
        height=480,
        projection="OPENCV_FISHEYE",
        distortion=(0.1, -0.02, 0.003, -0.0004),
    ),
    CatalogCameraModel(
        entity="/world/rig_00/cam_01",
        name="right_front",
        rig_T_cam=_pose(5.0, [0.1, 0.0, 0.0]),
        k_matrix=np.array([[500.0, 0.0, 330.0], [0.0, 505.0, 250.0], [0.0, 0.0, 1.0]]),
        width=640,
        height=480,
        projection="DISTORTED_PINHOLE",
        distortion=(-0.2, 0.05, 0.001, -0.002, 0.01, 0.0, 0.0, 0.0),
    ),
)
"""What the base recording's camera nodes carry."""


def _calibrated(camera: CatalogCameraModel) -> CatalogCameraModel:
    """The same camera with every calibration value moved, so a read of the base cannot pass for a read of the layer."""
    return dataclasses.replace(
        camera,
        rig_T_cam=camera.rig_T_cam @ _pose(2.0, [0.01, -0.02, 0.03]),
        k_matrix=camera.k_matrix + np.array([[3.0, 0.0, -2.0], [0.0, 4.0, 1.0], [0.0, 0.0, 0.0]]),
        distortion=tuple(value * 1.5 + 0.001 for value in camera.distortion),
    )


def _segment(tmp_path: Path) -> CatalogInputConfig:
    """A local segment: a base with each camera's name and video, the recorded calibration on the camera nodes, and a
    calibration layer that moved every value."""
    import rerun as rr

    base: Path = tmp_path / "base.rrd"
    with rr.RecordingStream("colsfm-test", recording_id="seg") as recording:
        recording.save(base)
        for camera in RECORDED:
            recording.log(camera.entity, rr.AnyValues(name=camera.name), static=True)
            video: str = f"{camera.entity}/pinhole/video"
            recording.log(video, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
            times = np.array([1_000, 2_000], dtype=np.int64).astype("timedelta64[ns]")
            recording.send_columns(
                video, indexes=[rr.TimeColumn("video_time", duration=times)], columns=rr.VideoStream.columns(sample=[b"\x00", b"\x01"])
            )
    # The camera nodes are `/world/rig_00/<node>`: the same writer, with the node as the name.
    nodes: tuple[CatalogCameraModel, ...] = tuple(dataclasses.replace(camera, name=camera.entity.rsplit("/", 1)[1]) for camera in RECORDED)
    write_calibration_layer(tmp_path / "recorded.rrd", "seg", nodes, rr.AnyValues(source="recorded"), root="/world/rig_00")
    write_calibration_layer(tmp_path / "calibrate.rrd", "seg", tuple(_calibrated(camera) for camera in RECORDED), rr.AnyValues(metric_scale=False))
    return CatalogInputConfig(
        dataset="robocap",
        rrd=base,
        layers={"recorded": tmp_path / "recorded.rrd", "calibrate": tmp_path / "calibrate.rrd"},
        output_root=tmp_path / "inputs",
    )


def _assert_same_cameras(read: tuple[CatalogCameraModel, ...], written: tuple[CatalogCameraModel, ...]) -> None:
    """Equal up to the float32 the Rerun components store."""
    assert [(camera.entity, camera.name, camera.projection, camera.width, camera.height) for camera in read] == [
        (camera.entity, camera.name, camera.projection, camera.width, camera.height) for camera in written
    ]
    for got, want in zip(read, written, strict=True):
        np.testing.assert_allclose(got.rig_T_cam, want.rig_T_cam, atol=1e-6)
        np.testing.assert_allclose(got.k_matrix, want.k_matrix, rtol=1e-6)
        np.testing.assert_allclose(got.distortion, want.distortion, rtol=1e-6, atol=1e-9)


def test_an_unset_calibration_entity_leaves_the_snapshot_name_alone(tmp_path: Path) -> None:
    recorded = CatalogInputConfig(dataset="robocap", segment="seg", output_root=tmp_path)
    assert "calibration_entity" not in recorded.spec()
    assert dataclasses.replace(recorded, calibration_entity=CALIBRATION_ENTITY).input_dir != recorded.input_dir


@pytest.mark.integration
def test_the_calibration_layer_replaces_the_camera_nodes_calibration(tmp_path: Path) -> None:
    """Needs a catalog server (an in-process one, on an OS port)."""
    from rerun.catalog import CatalogClient

    with serve_catalog_input(_segment(tmp_path)) as served:
        dataset = CatalogClient(served.catalog_url).get_dataset(name="robocap")
        recorded: tuple[CatalogCameraModel, ...] = read_catalog_cameras(dataset, served.segment, ())
        calibrated: tuple[CatalogCameraModel, ...] = read_catalog_cameras(dataset, served.segment, ("right_front",), CALIBRATION_ENTITY)
        with pytest.raises(ValueError, match="no_such_root"):
            read_catalog_cameras(dataset, served.segment, ("left_front",), "/world/runs/no_such_root/rig")
    _assert_same_cameras(recorded, RECORDED)
    _assert_same_cameras(calibrated, (_calibrated(RECORDED[1]),))
