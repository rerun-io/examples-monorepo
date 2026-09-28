from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from beartype.roar import BeartypeException
from dataforge import schema
from dataforge.writing import blueprint_views
from fake_track import fake_track
from jaxtyping import Float32
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from rerun.chunk import RrdReader

from handtrack import rerun_layers
from handtrack.apis import export_layers
from handtrack.blueprint import handtrack_blueprint
from handtrack.data import catalog
from handtrack.data.catalog import SegmentInfo
from handtrack.results import save_track

SEGMENT: str = "umetrack__real__hand_hand__testing__user_12__recording_13"


def _umetrack() -> DatasetEntry:
    try:
        return CatalogClient(catalog.CATALOG_URL).get_dataset(catalog.UMETRACK)
    except BeartypeException:
        raise
    except Exception as error:  # any connection failure means the asset is absent
        pytest.skip(f"catalog {catalog.CATALOG_URL} dataset {catalog.UMETRACK} unreachable: {error}")


def _info(entry: DatasetEntry) -> SegmentInfo:
    return next(info for info in catalog.list_segments(entry, catalog.UMETRACK) if info.segment_id == SEGMENT)


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
    from simplecv.data.skeleton.coco_133 import LEFT_HAND_IDX, RIGHT_HAND_IDX

    entry: DatasetEntry = _umetrack()
    truth: rerun_layers.GroundTruth = export_layers.read_ground_truth(entry, _info(entry))
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
    entry: DatasetEntry = _umetrack()
    truth: rerun_layers.GroundTruth = export_layers.read_ground_truth(entry, _info(entry))
    save_track(fake_track(truth), tmp_path / "run")
    export_layers.main(export_layers.Config(run_dir=tmp_path / "run", clips=(SEGMENT,), layers_root=tmp_path / "layers", export_dir=tmp_path / "export"))
    clip: Path = tmp_path / "export" / f"handtrack__{SEGMENT}.rrd"
    reader: RrdReader = RrdReader(clip)
    assert [entry.recording_id for entry in reader.recordings()] == [SEGMENT]
    assert reader.blueprints()
    entities: set[str] = {str(chunk.entity_path) for chunk in reader.stream(store=reader.recordings()[0]).to_chunks()}
    assert {schema.video_path(0, 1), schema.hand_mesh_path("left"), rerun_layers.pred_mesh_path("left"), rerun_layers.error_path("right")} <= entities
