"""Deterministic hand samples from camera-scoped native catalog RRD caches."""

from pathlib import Path

import numpy as np
import pyarrow as pa
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from rerun.chunk import ChunkStore, RrdReader
from scipy.spatial.transform import Rotation
from simplecv.catalog_video import CatalogVideo, catalog_codec, catalog_keyframes, packet_views

from posekit.cameras import CalibratedCamera, LensModel
from posekit.eval.hand_gt import DERIVED_JOINTS, GT_XYZ, SIDE_SLOTS, HandGtSegment, _nearest

HAND_SLOTS: slice = slice(SIDE_SLOTS["left"], SIDE_SLOTS["right"] + 21)
"""Both hands' COCO-WholeBody slots, left then right."""


def select_eval_indices(valid: Bool[ndarray, "f 2 21"], *, max_frames: int, min_joints: int = 15) -> Int64[ndarray, "n"]:
    """Select uniformly across frames with a visible wrist and enough labelled joints.

    Selection uses ground truth only and never repeats a frame.
    """
    available: Int64[ndarray, "a"] = np.flatnonzero(((valid.sum(-1) >= min_joints) & valid[:, :, 0]).any(-1))
    if not len(available):
        raise ValueError("No visible hands in this recording.")
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    return available[np.linspace(0, len(available) - 1, min(max_frames, len(available)), dtype=np.int64)]


def load_cached_hand_gt(path: Path, *, max_frames: int = 100) -> HandGtSegment:
    """Read native catalog video, calibration and aligned COCO hand labels locally.

    No catalog request is made. Unsupported lenses fail rather than being treated
    as pinhole cameras. Original video indices remain the sample identity.
    """
    from datafusion import col

    store: ChunkStore = RrdReader(path).stream().collect()
    entities: list[str] = store.schema().entity_paths()
    video_entity: str = next(e for e in entities if e.endswith("/pinhole/video"))
    pinhole: str = video_entity.removesuffix("/video")
    camera_entity: str = pinhole.removesuffix("/pinhole")
    rig: str = camera_entity.rsplit("/", 1)[0]
    uv_entity: str = next(e for e in entities if e.startswith(pinhole + "/coco133"))
    static: pa.Table = store.reader(None).to_arrow_table()

    def value(name: str) -> list:
        return next(v for v in static[name].to_pylist() if v is not None)

    intrinsics: Float64[ndarray, "3 3"] = np.asarray(value(f"{pinhole}:Pinhole:image_from_camera")[0], np.float64).reshape(3, 3).T
    width, height = (int(v) for v in value(f"{pinhole}:Pinhole:resolution")[0])
    distortion_key: str = f"{pinhole}:simplecv.components.DistortionCoefficients"
    distortion: Float64[ndarray, "d"] = np.zeros(0, np.float64)
    lens: LensModel = "PINHOLE"
    if distortion_key in static.column_names:
        distortion = np.asarray(value(distortion_key)[0], np.float64)
        if len(distortion) != 8 or value(f"{pinhole}:simplecv.components.DistortionModel")[0] != "kannala_brandt":
            raise ValueError(f"{path}: unsupported distortion calibration")
        distortion = np.r_[distortion, np.zeros(4)]
        lens = "FISHEYE624"
    if f"{camera_entity}:ds_coefficients" in static.column_names:
        ds: Float64[ndarray, "6"] = np.asarray(value(f"{camera_entity}:ds_coefficients")[0], np.float64)
        intrinsics = np.array([[ds[0], 0.0, ds[2]], [0.0, ds[1], ds[3]], [0.0, 0.0, 1.0]])
        distortion, lens = ds[4:], "DOUBLE_SPHERE"
    camera: CalibratedCamera = CalibratedCamera(intrinsics, distortion, width, height, lens=lens)
    camera_transform: Float64[ndarray, "4 4"] = np.eye(4)
    camera_transform[:3, :3] = np.asarray(value(f"{camera_entity}:Transform3D:mat3x3")[0]).reshape(3, 3).T
    camera_transform[:3, 3] = value(f"{camera_entity}:Transform3D:translation")[0]
    relation: str = f"{camera_entity}:Transform3D:relation"
    if relation not in static.column_names or value(relation)[0] != 2:
        camera_transform = np.linalg.inv(camera_transform)
    video_table: pa.Table = (
        store.reader("video_time", contents=video_entity)
        .filter(col(f"{video_entity}:VideoStream:sample").is_not_null())
        .sort(col("video_time"))
        .to_arrow_table()
    )
    times: Int64[ndarray, "f"] = video_table["video_time"].cast(pa.int64()).to_numpy()
    video: CatalogVideo = CatalogVideo(
        times,
        packet_views(video_table[f"{video_entity}:VideoStream:sample"]),
        catalog_keyframes(video_table[f"{video_entity}:VideoStream:is_keyframe"]),
        catalog_codec(video_table[f"{video_entity}:VideoStream:codec"], str(path)),
    )
    gt_entity: str = GT_XYZ
    qa_entity: str = f"/world/gt/quality/{rig.rsplit('/', 1)[-1]}/{camera_entity.rsplit('/', 1)[-1]}/qa_pass"
    table: pa.Table = store.reader("video_time", contents=[rig, gt_entity, uv_entity, qa_entity]).to_arrow_table()
    row_times: Int64[ndarray, "r"] = table["video_time"].cast(pa.int64()).to_numpy()

    def aligned(name: str) -> tuple[Float64[ndarray, "f ..."], Bool[ndarray, "f"]]:
        values: list = table[name].to_pylist()
        rows: Int64[ndarray, "r"] = np.asarray([i for i, v in enumerate(values) if v is not None], dtype=np.int64)
        rows = rows[np.argsort(row_times[rows])]
        nearest: Int64[ndarray, "f"] = _nearest(row_times[rows], times)
        if not len(rows):
            raise ValueError(f"{path}: no rows for {name}")
        return np.asarray([values[rows[max(i, 0)]] for i in nearest], np.float64), nearest >= 0

    points, points_ok = aligned(f"{gt_entity}:Points3D:positions")
    confidence, confidence_ok = aligned(f"{gt_entity}:simplecv.KeypointConfidence3D:confidences")
    pixels, pixels_ok = aligned(f"{uv_entity}:Points2D:positions")
    translation, pose_ok = aligned(f"{rig}:Transform3D:translation")
    rotation_key: str = f"{rig}:Transform3D:quaternion"
    if rotation_key in table.column_names:
        quaternion, rotation_ok = aligned(rotation_key)
        quaternion = quaternion.reshape(-1, 4)
        rotation_ok &= np.isfinite(quaternion).all(-1) & (np.linalg.norm(quaternion, axis=-1) > 0.0)
        quaternion[~rotation_ok] = [0.0, 0.0, 0.0, 1.0]
        rotations: Float64[ndarray, "f 3 3"] = Rotation.from_quat(quaternion).as_matrix()
    else:
        matrices, rotation_ok = aligned(f"{rig}:Transform3D:mat3x3")
        rotations = matrices.reshape(-1, 3, 3).transpose(0, 2, 1)
    world_from_rig: Float64[ndarray, "f 4 4"] = np.tile(np.eye(4), (len(times), 1, 1))
    world_from_rig[:, :3, :3] = rotations
    world_from_rig[:, :3, 3] = translation.reshape(-1, 3)
    cam_from_world: Float64[ndarray, "f 4 4"] = camera_transform @ np.linalg.inv(world_from_rig)
    points = points.reshape(-1, 133, 3)[:, HAND_SLOTS].reshape(-1, 2, 21, 3)
    xyz: Float32[ndarray, "f 2 21 3"] = (
        np.einsum("fij,fskj->fski", cam_from_world[:, :3, :3], points) + cam_from_world[:, None, None, :3, 3]
    ).astype(np.float32)
    uv: Float32[ndarray, "f 2 21 2"] = pixels.reshape(-1, 133, 2)[:, HAND_SLOTS].reshape(-1, 2, 21, 2).astype(np.float32)
    valid: Bool[ndarray, "f 2 21"] = (
        (confidence.reshape(-1, 133)[:, HAND_SLOTS].reshape(-1, 2, 21) > 0) & np.isfinite(xyz).all(-1) & np.isfinite(uv).all(-1) & (xyz[..., 2] > 0)
    )
    valid &= (uv[..., 0] >= 0) & (uv[..., 0] < width) & (uv[..., 1] >= 0) & (uv[..., 1] < height)
    aligned_ok: Bool[ndarray, "f"] = points_ok & confidence_ok & pixels_ok & pose_ok & rotation_ok
    if f"{qa_entity}:Scalars:scalars" in table.column_names:
        qa, qa_ok = aligned(f"{qa_entity}:Scalars:scalars")
        aligned_ok &= qa_ok & (qa.reshape(-1) > 0.5)
    valid &= aligned_ok[:, None, None]
    valid[:, :, list(DERIVED_JOINTS)] = False
    chosen: Int64[ndarray, "n"] = select_eval_indices(valid, max_frames=max_frames)
    return HandGtSegment(path.stem, camera, video, chosen, times[chosen], xyz[chosen], uv[chosen], valid[chosen])
