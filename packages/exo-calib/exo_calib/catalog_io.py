"""Catalog IO for the exo-calib pipeline.

Reads the registered ``assembly101`` dataset (layer ``base``) and writes
prediction layers back onto the same segment. ``simplecv.rerun_dataloader``
needs rerun-sdk's ``dataloader`` extra, which only the exo-calib/catalog envs
install — keep the import local to this module.

Entity map of the base layer (see /tmp/rerun-viewer-validation/assembly101-base/notes.md):
- exo videos:   /world/rig_{00..07}/cam_00/pinhole/video   (VideoStream, av1)
- intrinsics:   /world/rig_XX/cam_00/pinhole:Pinhole:image_from_camera (column-major)
- extrinsics:   /world/rig_XX/cam_00:Transform3D:{mat3x3,translation} — logged by
  simplecv ``log_pinhole`` with ``from_parent=True``, so they ARE cam_R_world / cam_t_world.
- GT 3D hands:  /world/gt/coco133_xyz (Points3D + simplecv.KeypointConfidence3D)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry

DEFAULT_CATALOG_URL: str = "rerun+http://127.0.0.1:9988"
DEFAULT_DATASET_NAME: str = "assembly101"
TIMELINE: str = "video_time"
EXO_CAMERA_NAMES: tuple[str, ...] = tuple(f"rig_{i:02d}/cam_00" for i in range(8))
EGO_CAMERA_NAMES: tuple[str, ...] = tuple(f"rig_08/cam_{i:02d}" for i in range(4))
EXOCALIB_ROOT: str = "/world/exocalib"


def pinhole_entity(camera_name: str) -> str:
    """Log-side entity path of a camera's pinhole node."""
    return f"/world/{camera_name}/pinhole"


def exocalib_entity(variant: str) -> str:
    """Entity path of an exocalib camera-rig variant (``init``/``refined``)."""
    return f"{EXOCALIB_ROOT}/{variant}"


def kp3d_entity(variant: str) -> str:
    """Entity path of an exocalib triangulated-keypoint variant."""
    return f"{EXOCALIB_ROOT}/kp3d_{variant}"


@dataclass(slots=True)
class RigCameras:
    """A camera rig read back from a catalog layer (GT base layer or an exocalib layer)."""

    names: tuple[str, ...]
    """Camera entity names, e.g. ``rig_00/cam_00``."""
    k_v33: Float64[ndarray, "v 3 3"]
    """Pinhole intrinsics at the native video resolution."""
    cam_T_world_v44: Float64[ndarray, "v 4 4"]
    """World-to-camera transforms (RDF, metric, +Z-up world)."""


def connect_dataset(catalog_url: str = DEFAULT_CATALOG_URL, dataset_name: str = DEFAULT_DATASET_NAME) -> DatasetEntry:
    """Connect to the running catalog and return the dataset entry.

    Args:
        catalog_url: Rerun catalog server URL.
        dataset_name: Existing dataset name.

    Returns:
        The bound dataset entry.
    """
    client: CatalogClient = CatalogClient(catalog_url)
    return client.get_dataset(dataset_name)


def only_segment_id(dataset: DatasetEntry) -> str:
    """Return the id of the dataset's single segment, failing on any other count."""
    table: pa.Table = pa.Table.from_batches(dataset.segment_table().collect())
    segment_ids: list[str] = [str(v) for v in table.column("rerun_segment_id").to_pylist()]
    if len(segment_ids) != 1:
        raise ValueError(f"expected exactly one segment, found {segment_ids}")
    return segment_ids[0]


def read_rig_cameras(
    dataset: DatasetEntry, segment_id: str, root: str = "/world", names: tuple[str, ...] = EXO_CAMERA_NAMES
) -> RigCameras:
    """Read a rig's intrinsics and extrinsics back from a catalog layer.

    ``root="/world"`` reads the GT base layer (evaluation only — never inside
    the pipeline); ``exocalib_entity("init")``/``exocalib_entity("refined")``
    read the pipeline's own frusta layers, which are the stage store for
    camera parameters.

    Args:
        dataset: Catalog dataset entry.
        segment_id: Segment to read.
        root: Entity prefix the rig's cameras live under.
        names: Camera entity names under ``root``.

    Returns:
        The rig's camera parameters.
    """
    view = dataset.filter_segments(segment_id).filter_contents([f"{root}/{name}/**" for name in names])
    # All rig calibration components are logged static, so read the static row.
    table: pa.Table = view.reader(index=None).to_arrow_table()

    def last_valid(column_name: str) -> ndarray:
        chunk: pa.ChunkedArray = table.column(column_name)
        values: list = [v for v in chunk.to_pylist() if v is not None]
        if not values:
            raise ValueError(f"column {column_name} has no data in segment {segment_id}")
        return np.asarray(values[-1], dtype=np.float64).reshape(-1)

    k_list: list[Float64[ndarray, "3 3"]] = []
    cam_T_world_list: list[Float64[ndarray, "4 4"]] = []
    for name in names:
        # Rerun stores mat3x3 column-major; both Transform3D and Pinhole need the transpose.
        cam_R_world_33: Float64[ndarray, "3 3"] = last_valid(f"{root}/{name}:Transform3D:mat3x3").reshape(3, 3).T
        cam_t_world_3: Float64[ndarray, "3"] = last_valid(f"{root}/{name}:Transform3D:translation").reshape(3)
        k_33: Float64[ndarray, "3 3"] = last_valid(f"{root}/{name}/pinhole:Pinhole:image_from_camera").reshape(3, 3).T
        cam_T_world_44: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
        cam_T_world_44[:3, :3] = cam_R_world_33
        cam_T_world_44[:3, 3] = cam_t_world_3
        k_list.append(k_33)
        cam_T_world_list.append(cam_T_world_44)
    return RigCameras(names=names, k_v33=np.stack(k_list), cam_T_world_v44=np.stack(cam_T_world_list))


def new_layer_recording(application_id: str, segment_id: str, rrd_path: Path) -> tuple[rr.RecordingStream, Path]:
    """Create a recording that saves near ``rrd_path`` and registers as a layer of ``segment_id``.

    The recording id must equal the source segment id so the catalog attaches the
    layer to the existing segment (mv-api ``catalog_prediction_layer`` pattern).
    If ``rrd_path`` exists, a ``-N``-suffixed sibling is written instead: the
    in-memory catalog server caches layer-file descriptors, so overwriting a
    registered rrd in place serves stale data. Returns the recording and the
    path actually written.
    """
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    actual_path: Path = rrd_path
    counter: int = 1
    while actual_path.exists():
        actual_path = rrd_path.with_name(f"{rrd_path.stem}-{counter}{rrd_path.suffix}")
        counter += 1
    recording: rr.RecordingStream = rr.RecordingStream(application_id=application_id, recording_id=segment_id)
    recording.save(actual_path)
    return recording, actual_path


def register_layer(dataset: DatasetEntry, rrd_path: Path, layer_name: str) -> None:
    """Register a generated RRD as a catalog layer, replacing any existing one."""
    from rerun.catalog import OnDuplicateSegmentLayer

    handle = dataset.register([rrd_path.resolve().as_uri()], layer_name=layer_name, on_duplicate=OnDuplicateSegmentLayer.REPLACE)
    handle.wait()


def log_coco133_class_context(
    recording: rr.RecordingStream,
    entity: str,
    classes: tuple[tuple[int, str, tuple[int, int, int]], ...],
    connections: bool = True,
) -> None:
    """Log a COCO-133 annotation context the simplecv exoego way.

    One class per skeleton instance: keypoint links render in the class color,
    so each logged keypoint set (GT / init / refined / 2D detections) is
    distinguishable by the color of its skeleton. ``connections=False`` keeps
    the keypoint names (hover) but draws no skeleton edges.
    """
    from simplecv.data.skeleton.coco_133 import COCO_133_ID2NAME, COCO_133_LINKS

    recording.log(
        entity,
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=class_id, label=label, color=color),
                    keypoint_annotations=[rr.AnnotationInfo(id=i, label=n) for i, n in COCO_133_ID2NAME.items()],
                    keypoint_connections=COCO_133_LINKS if connections else [],
                )
                for class_id, label, color in classes
            ]
        ),
        static=True,
    )
