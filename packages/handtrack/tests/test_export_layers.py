import socket
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa
import pytest
from dataforge import schema
from dataforge.writing import blueprint_views
from fake_track import fake_track, rotation_about
from jaxtyping import Float32
from numpy import ndarray
from rerun.chunk import RrdReader

from handtrack import rerun_layers
from handtrack.apis import export_layers
from handtrack.blueprint import handtrack_blueprint
from handtrack.geometry.camera import CameraRig
from handtrack.results import save_track

CATALOG_URL: str = "rerun+http://127.0.0.1:51235"
SEGMENT: str = "umetrack__real__hand_hand__testing__user_12__recording_13"


def _catalog_reachable() -> bool:
    address = urlparse(CATALOG_URL.replace("rerun+", ""))
    try:
        socket.create_connection((address.hostname, address.port), timeout=1.0).close()
    except OSError:
        return False
    return True


def _static_column(value: list[float] | int) -> pa.Array:
    return pa.array([[value]])


def test_read_rig_reads_the_column_major_extrinsics_and_intrinsics() -> None:
    rotation: Float32[ndarray, "3 3"] = rotation_about(1, 0.3) @ rotation_about(0, 0.2)
    columns: dict[str, pa.Array] = {}
    for camera in range(4):
        node: str = schema.cam_path(0, camera)
        pinhole: str = schema.pinhole_path(0, camera)
        columns[f"{node}:Transform3D:relation"] = _static_column(2)
        columns[f"{node}:Transform3D:mat3x3"] = _static_column(rotation.T.reshape(-1).tolist())  # column-major
        columns[f"{node}:Transform3D:translation"] = _static_column([0.01 * camera, 0.0, 0.0])
        intrinsics: list[float] = [240.0, 0.0, 0.0, 0.0, 241.0, 0.0, 318.0 + camera, 239.0, 1.0]  # column-major K
        columns[f"{pinhole}:Pinhole:image_from_camera"] = _static_column(intrinsics)
        columns[f"{pinhole}:Pinhole:resolution"] = _static_column([636.0, 480.0])
        columns[f"{pinhole}:simplecv.components.DistortionCoefficients"] = _static_column([0.1 * camera] * 8)
    rig: CameraRig = export_layers.read_rig(pa.table(columns))
    np.testing.assert_allclose(rig.cam_from_rig[2, :3, :3].numpy(), rotation, atol=1e-6)
    np.testing.assert_allclose(rig.cam_from_rig[3, :3, 3].numpy(), [0.03, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(rig.focal[1].numpy(), [240.0, 241.0])
    np.testing.assert_allclose(rig.principal[3].numpy(), [321.0, 239.0])
    assert rig.fisheye62 is not None
    np.testing.assert_allclose(rig.fisheye62[2].numpy(), [0.2] * 8, rtol=1e-6)


def test_dense_rows_are_nan_where_the_column_is_null_or_empty() -> None:
    column: pa.ChunkedArray = pa.chunked_array([pa.array([[[1.0, 2.0, 3.0]], None, [], [[4.0, 5.0, 6.0]]], type=pa.list_(pa.list_(pa.float32(), 3)))])
    rows: Float32[ndarray, "4 3"] = export_layers._dense(column, 3)
    np.testing.assert_array_equal(rows[[0, 3]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.isnan(rows[[1, 2]]).all()


def test_blueprint_lays_out_the_scene_four_cameras_and_three_plots(tmp_path: Path) -> None:
    origins: list[str] = [str(view.origin) for view in blueprint_views(handtrack_blueprint())]
    assert origins[0] == schema.rig_path(0)
    assert origins[1:5] == [schema.pinhole_path(0, camera) for camera in range(4)]
    assert len(origins) == 8
    handtrack_blueprint().save("dataforge", str(tmp_path / "handtrack.rbl"))
    assert (tmp_path / "handtrack.rbl").stat().st_size > 0


@pytest.mark.golden
def test_ground_truth_projects_onto_the_stored_projections() -> None:
    """The catalog reader, the rig and the skinning reproduce the ``projections`` layer (20 of the 21 COCO hand slots; slot 92/113 is derived)."""
    if not _catalog_reachable():
        pytest.skip(f"catalog {CATALOG_URL} is unreachable")
    from rerun.catalog import CatalogClient
    from simplecv.data.skeleton.coco_133 import LEFT_HAND_IDX, RIGHT_HAND_IDX

    entry = CatalogClient(CATALOG_URL).get_dataset("dataforge-umetrack")
    truth: rerun_layers.GroundTruth = export_layers.read_ground_truth(entry, SEGMENT)
    pixels: Float32[ndarray, "f 4 2 21 2"] = rerun_layers.camera_pixels(truth.rig, truth.world_from_rig, rerun_layers.gt_landmarks(truth))
    path: str = schema.coco133_uv_projected_path(0, 1)
    table: pa.Table = entry.filter_segments(SEGMENT).filter_contents([path]).reader(index=schema.TIMELINE).to_arrow_table().sort_by(schema.TIMELINE)
    stored: list[list[list[float] | None]] = table[f"{path}:Points2D:positions"].to_pylist()
    matched: list[float] = []
    for frame in range(0, len(stored), 30):
        uv: Float32[ndarray, "133 2"] = np.array([[np.nan, np.nan] if point is None else point for point in stored[frame]], dtype=np.float32)
        for side, slots in enumerate((LEFT_HAND_IDX, RIGHT_HAND_IDX)):
            shipped: Float32[ndarray, "21 2"] = uv[list(slots)]
            ours: Float32[ndarray, "21 2"] = pixels[frame, 1, side]
            if np.isfinite(shipped).all() and np.isfinite(ours).all():
                nearest: Float32[ndarray, "21"] = np.linalg.norm(shipped[:, None] - ours[None], axis=-1).min(axis=1)
                matched.append(float(np.sort(nearest)[19]))  # the 20 best-matching slots
    assert matched and max(matched) < 1e-3


@pytest.mark.integration
def test_export_writes_layers_and_a_standalone_clip(tmp_path: Path) -> None:
    """A fake track for one real test segment: both layer files, and a standalone rrd with the base video, the GT mesh and the prediction."""
    if not _catalog_reachable():
        pytest.skip(f"catalog {CATALOG_URL} is unreachable")
    from rerun.catalog import CatalogClient

    entry = CatalogClient(CATALOG_URL).get_dataset("dataforge-umetrack")
    truth: rerun_layers.GroundTruth = export_layers.read_ground_truth(entry, SEGMENT)
    save_track(fake_track(truth), tmp_path / "run")
    config = export_layers.Config(run_dir=tmp_path / "run", clips=(SEGMENT,), layers_root=tmp_path / "layers", export_dir=tmp_path / "export", catalog_url=CATALOG_URL)
    export_layers.main(config)
    clip: Path = tmp_path / "export" / f"handtrack__{SEGMENT}.rrd"
    reader: RrdReader = RrdReader(clip)
    assert [entry.recording_id for entry in reader.recordings()] == [SEGMENT]
    assert reader.blueprints()
    entities: set[str] = {str(chunk.entity_path) for chunk in reader.stream(store=reader.recordings()[0]).to_chunks()}
    assert {schema.video_path(0, 1), schema.hand_mesh_path("left"), rerun_layers.pred_mesh_path("left"), rerun_layers.error_path("right")} <= entities
