"""Stage B's catalog record: the ``exocalib_kp2d`` layer's ``…_raw`` entities, written once and queried back by refinement.

Every array :func:`log_keypoints_record` writes round-trips exactly through
:func:`load_stage_b` (all components are logged at their native dtypes). The
gated skeleton overlay that shares the layer is visualization owned by the
Stage B tool, not part of this record.
"""


import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.rerun_custom_types import Points2DWithConfidence
from simplecv.rrd_query_utils import first_valid_value_as

from exo_calib.keypoints import BOX_SOURCE_BY_NAME, BoxSource, CameraKeypoints
from exo_calib.layer_io import KP2D_LAYER, TIMELINE, ClassSpec, log_coco133_class_context, pinhole_entity

KP2D_LABEL: str = "rtmw-x kp2d"
RAW_COLOR: tuple[int, int, int] = (110, 110, 110)


def kp2d_entity(camera_name: str) -> str:
    """Stage B's gated overlay entity under a camera's pinhole; ``_raw`` / ``_bbox`` / ``_crop`` / ``_rejected`` are its siblings."""
    return f"{pinhole_entity(camera_name)}/{KP2D_LAYER}"


def kp2d_raw_entity(camera_name: str) -> str:
    """Stage B's queryable data record for one camera."""
    return f"{kp2d_entity(camera_name)}_raw"


def log_keypoints_record(per_camera: dict[str, CameraKeypoints], recording: rr.RecordingStream) -> None:
    """Log every frame's raw keypoints, confidences, box and crop rectangle under each camera's ``…_raw`` entity.

    The person box and crop rectangle are also drawn as ``Boxes2D`` siblings so the
    tracked / detected provenance is visible per frame.
    """
    for name, cam in per_camera.items():
        entity: str = kp2d_entity(name)
        raw_entity: str = kp2d_raw_entity(name)
        log_coco133_class_context(recording, raw_entity, (ClassSpec(0, f"{KP2D_LABEL} raw", RAW_COLOR),), connections=False)
        recording.log(raw_entity, rr.AnyValues(crop_input_wh=cam.crop_input_wh, video_wh=cam.video_wh), static=True)
        for t in range(len(cam.sample_indices)):
            box_source: BoxSource = cam.box_source[t]
            recording.set_time(TIMELINE, duration=1e-9 * float(cam.times_ns[t]))
            recording.log(
                raw_entity,
                Points2DWithConfidence(positions=cam.kp_xy[t], confidences=cam.conf[t], class_ids=0, keypoint_ids=np.arange(133), radii=1.0),
            )
            recording.log(
                raw_entity,
                rr.AnyValues(
                    sample_idx=cam.sample_indices[t : t + 1],
                    box_source=box_source,
                    bbox_xyxy=cam.bbox_xyxy[t],
                    crop_origin_xy=cam.crop_origin_xy[t],
                    crop_size_wh=cam.crop_size_wh[t],
                ),
            )
            detected: bool = bool(np.isfinite(cam.bbox_xyxy[t]).all())
            recording.log(
                f"{entity}_bbox",
                rr.Boxes2D(
                    array=cam.bbox_xyxy[t : t + 1] if detected else np.zeros((0, 4), dtype=np.float32),
                    array_format=rr.Box2DFormat.XYXY,
                    colors=(0, 200, 255) if box_source == "tracked" else (240, 140, 60),
                    labels=[box_source] if detected else [],
                ),
            )
            recording.log(f"{entity}_bbox", rr.AnyValues(box_source=box_source))
            recording.log(
                f"{entity}_crop",
                rr.Boxes2D(
                    array=np.concatenate((cam.crop_origin_xy[t], cam.crop_size_wh[t]))[None] if detected else np.zeros((0, 4), dtype=np.float32),
                    array_format=rr.Box2DFormat.XYWH,
                    colors=RAW_COLOR,
                ),
            )


def load_stage_b(dataset: DatasetEntry, segment_id: str, names: tuple[str, ...]) -> dict[str, CameraKeypoints]:
    """Query Stage B's raw record back from the registered ``exocalib_kp2d`` layer.

    Raises:
        ValueError: If a camera's record is missing a component, or the reader returned rows out of timeline order.
    """
    return {name: _load_camera_keypoints(dataset, segment_id, name) for name in names}


def _load_camera_keypoints(dataset: DatasetEntry, segment_id: str, name: str) -> CameraKeypoints:
    raw_entity: str = kp2d_raw_entity(name)
    view = dataset.filter_segments(segment_id).filter_contents([raw_entity])
    # No client-side sort: the reader yields index order (simplecv's segment decoder relies on the same
    # contract), and a sort would re-materialize the per-frame keypoint columns. The guard below fails loudly.
    table: pa.Table = view.reader(index=TIMELINE).to_arrow_table()
    times_ns: Int64[ndarray, "t"] = table.column(TIMELINE).combine_chunks().cast(pa.int64()).to_numpy().astype(np.int64)
    if np.any(times_ns[1:] < times_ns[:-1]):
        raise ValueError(f"{raw_entity}: reader returned rows out of timeline order in segment {segment_id}")

    def column(field_name: str) -> pa.ChunkedArray:
        column_name: str = f"{raw_entity}:{field_name}"
        if column_name not in table.column_names:
            raise ValueError(f"{column_name} is missing from segment {segment_id}; rerun Stage B")
        return table.column(column_name)

    def static_wh(field_name: str) -> Int64[ndarray, "2"]:
        return np.asarray(first_valid_value_as(column(field_name), list, component_name=f"{raw_entity}:{field_name}"), dtype=np.int64).reshape(2)

    box_source: list[BoxSource] = []
    for name in np.asarray(column("box_source").to_pylist(), dtype=str).reshape(-1):
        source: BoxSource | None = BOX_SOURCE_BY_NAME.get(str(name))
        if source is None:
            raise ValueError(f"{raw_entity}: unknown box source {name!r}")
        box_source.append(source)
    return CameraKeypoints(
        sample_indices=np.asarray([v[0] for v in column("sample_idx").to_pylist()], dtype=np.int64),
        times_ns=times_ns,
        kp_xy=np.asarray(column("Points2D:positions").to_pylist(), dtype=np.float32),
        conf=np.asarray(column("simplecv.KeypointConfidence2D:confidences").to_pylist(), dtype=np.float32),
        bbox_xyxy=np.asarray(column("bbox_xyxy").to_pylist(), dtype=np.float32),
        box_source=tuple(box_source),
        crop_origin_xy=np.asarray(column("crop_origin_xy").to_pylist(), dtype=np.float32),
        crop_size_wh=np.asarray(column("crop_size_wh").to_pylist(), dtype=np.float32),
        crop_input_wh=static_wh("crop_input_wh"),
        video_wh=static_wh("video_wh"),
    )
