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
import torch
from jaxtyping import Float64, Shaped
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from torchcodec.decoders import VideoDecoder

DEFAULT_CATALOG_URL: str = "rerun+http://127.0.0.1:9988"
DEFAULT_DATASET_NAME: str = "assembly101"
TIMELINE: str = "video_time"
EXO_CAMERA_NAMES: tuple[str, ...] = tuple(f"rig_{i:02d}/cam_00" for i in range(8))
EGO_CAMERA_NAMES: tuple[str, ...] = tuple(f"rig_08/cam_{i:02d}" for i in range(4))


@dataclass(slots=True)
class GtCameras:
    """Ground-truth exo camera parameters read from the base layer."""

    names: tuple[str, ...]
    """Camera entity names, e.g. ``rig_00/cam_00``."""
    k_v33: Float64[ndarray, "v 3 3"]
    """Pinhole intrinsics at the native video resolution."""
    cam_T_world_v44: Float64[ndarray, "v 4 4"]
    """World-to-camera transforms (RDF, metric, +Z-up world)."""
    resolution_wh_v2: Float64[ndarray, "v 2"]
    """Native video resolution (width, height) per camera."""


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


def read_gt_cameras(dataset: DatasetEntry, segment_id: str, names: tuple[str, ...] = EXO_CAMERA_NAMES) -> GtCameras:
    """Read GT intrinsics and extrinsics of the exo cameras from the base layer.

    Args:
        dataset: Catalog dataset entry.
        segment_id: Segment to read.
        names: Camera entity names under ``/world``.

    Returns:
        GT camera parameters (used for evaluation only — never inside the pipeline).
    """
    columns: list[str] = []
    for name in names:
        columns += [
            f"/world/{name}:Transform3D:mat3x3",
            f"/world/{name}:Transform3D:translation",
            f"/world/{name}/pinhole:Pinhole:image_from_camera",
            f"/world/{name}/pinhole:Pinhole:resolution",
        ]
    view = dataset.filter_segments(segment_id).filter_contents([f"/world/{name}/**" for name in names])
    table: pa.Table = view.reader(index=TIMELINE, fill_latest_at=True).select(TIMELINE, *columns).sort(TIMELINE).to_arrow_table()

    def last_valid(column_name: str) -> ndarray:
        chunk: pa.ChunkedArray = table.column(column_name)
        values: list = [v for v in chunk.to_pylist() if v is not None]
        if not values:
            raise ValueError(f"column {column_name} has no data in segment {segment_id}")
        return np.asarray(values[-1], dtype=np.float64)

    k_list: list[Float64[ndarray, "3 3"]] = []
    cam_T_world_list: list[Float64[ndarray, "4 4"]] = []
    resolution_list: list[Float64[ndarray, "2"]] = []
    for name in names:
        # Rerun stores mat3x3 column-major; both Transform3D and Pinhole need the transpose.
        cam_R_world_33: Float64[ndarray, "3 3"] = last_valid(f"/world/{name}:Transform3D:mat3x3").reshape(3, 3).T
        cam_t_world_3: Float64[ndarray, "3"] = last_valid(f"/world/{name}:Transform3D:translation").reshape(3)
        k_33: Float64[ndarray, "3 3"] = last_valid(f"/world/{name}/pinhole:Pinhole:image_from_camera").reshape(3, 3).T
        cam_T_world_44: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
        cam_T_world_44[:3, :3] = cam_R_world_33
        cam_T_world_44[:3, 3] = cam_t_world_3
        k_list.append(k_33)
        cam_T_world_list.append(cam_T_world_44)
        resolution_list.append(last_valid(f"/world/{name}/pinhole:Pinhole:resolution").reshape(2))
    return GtCameras(
        names=names,
        k_v33=np.stack(k_list),
        cam_T_world_v44=np.stack(cam_T_world_list),
        resolution_wh_v2=np.stack(resolution_list),
    )


@dataclass(slots=True)
class ExoVideoStreams:
    """Per-camera NVDEC decoders over one segment's exo videos."""

    names: tuple[str, ...]
    """Camera entity names, index-aligned with ``decoders``."""
    times_ns: list[Shaped[ndarray, " n_samples"]]
    """Per-camera sample timestamps (timedelta64[ns], timeline order)."""
    decoders: list[VideoDecoder]
    """One whole-segment torchcodec GPU decoder per camera."""

    def frame_rgb_hw3(self, cam_idx: int, sample_idx: int) -> ndarray:
        """Decode one frame to a uint8 RGB HWC numpy array."""
        frame_chw: torch.Tensor = self.decoders[cam_idx][sample_idx]  # pyrefly: ignore[bad-index]  # TorchCodec accepts Python int indices.
        return frame_chw.permute(1, 2, 0).contiguous().cpu().numpy()


def open_exo_streams(
    dataset: DatasetEntry,
    segment_id: str,
    device: str = "cuda",
    fps: int = 30,
    names: tuple[str, ...] = EXO_CAMERA_NAMES,
) -> ExoVideoStreams:
    """Open one NVDEC decoder per exo camera from catalog video packets.

    Args:
        dataset: Catalog dataset entry.
        segment_id: Segment whose packets are fetched.
        device: Decode device (``cuda`` uses NVDEC).
        fps: Nominal frame rate written into the wrapping MP4 track.
        names: Camera entity names under ``/world``.

    Returns:
        Decoders and per-camera sample timestamps.
    """
    from simplecv.rerun_dataloader import open_segment_decoder

    times_list: list[Shaped[ndarray, " n_samples"]] = []
    decoder_list: list[VideoDecoder] = []
    for name in names:
        times, _samples, _keyframes, decoder = open_segment_decoder(
            dataset, segment_id, f"world/{name}/pinhole/video", TIMELINE, torch.device(device), fps
        )
        times_list.append(times)
        decoder_list.append(decoder)
    return ExoVideoStreams(names=names, times_ns=times_list, decoders=decoder_list)


def new_layer_recording(application_id: str, segment_id: str, rrd_path: Path) -> rr.RecordingStream:
    """Create a recording that saves to ``rrd_path`` and registers as a layer of ``segment_id``.

    The recording id must equal the source segment id so the catalog attaches the
    layer to the existing segment (mv-api ``catalog_prediction_layer`` pattern).
    """
    rrd_path.parent.mkdir(parents=True, exist_ok=True)
    recording: rr.RecordingStream = rr.RecordingStream(application_id=application_id, recording_id=segment_id)
    recording.save(rrd_path)
    return recording


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
