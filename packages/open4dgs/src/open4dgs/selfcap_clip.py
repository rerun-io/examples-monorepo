"""Read a SelfCap segment (``selfcap-easy26-v4``) into a multi-view training clip: the four exo phones on ``video_time``.

SelfCap differs from the dataforge MAMMA layout that ``catalog_clip.read_catalog_clip`` reads:

- Timelines are durations (``capture_time``, ``video_time``, ``preview_time``); there is no ``frame_index``. Every phone
  is a 25 fps stream with its own sync offset on ``video_time`` (selfcap-curate ``time_sync``: tens of ms), so the
  phones' frames do not share timestamps. The clip takes the first camera's frames as the reference grid and, for
  every other camera, the frame nearest each reference time (at most half a frame period away; ``times_s`` keeps each
  camera's own frame time). ``frame_indices`` are the reference camera's packet numbers.
- The camera node's ``Transform3D`` (``mat3x3`` column-major + ``translation``) carries no relation column, so it is
  Rerun's default ``ParentFromChild``: camera-to-world (``world_T_cam``), the same reading dataforge
  ``oak_mount.read_static_cameras`` applies (``rig_T_cam``; the exo rigs carry no transform of their own). World is
  the Quest ``head_pose`` frame (metres, Y up), cameras OpenCV RDF (``property:calib:world``).
- Lens: selfcap-curate's exo calibration holds the phone lens at k1 = k2 = 0 (``exo_calibration.solve_camera``
  ``fixed``; ``keypoints.read_cameras`` passes ``radial=(0.0, 0.0)``: iOS video is already rectified), so the phones
  are pinhole and the logged ``Pinhole`` is the whole model. A ``DistortionCoefficients`` column with a non-zero
  value would mean a lens this reader does not undistort: it raises.
- People: no SMPL-X mesh. The body is given as 3D joints over time (``ClipJoints``), by default the triangulated
  COCO-133 keypoints of the ``calib`` layer (``/world/pred/rtmw_triangulated/coco133_xyz``, 25 Hz on ``video_time``),
  interpolated to any time.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import torch
from datafusion import col
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from simplecv.catalog_rig_layout import CatalogCamera, CatalogRigLayout, catalog_components, parse_rig_layout
from simplecv.catalog_video import CatalogVideo, catalog_keyframes, catalog_video_from_table
from simplecv.catalog_video_codec import wrap_mp4
from simplecv.rrd_query_utils import first_valid_value
from torch import Tensor
from torchcodec.decoders import VideoDecoder

from open4dgs.catalog_clip import RESAMPLE_CHUNK, MultiViewClip, fit_max_side, resample_frames, scale_intrinsics

VIDEO_TIMELINE: str = "video_time"
"""SelfCap's aligned clock (OAK-left video PTS as cut), a duration timeline."""
JOINTS_ENTITY: str = "/world/pred/rtmw_triangulated/coco133_xyz"
"""Triangulated RTMW COCO-133 joints (``calib`` layer), world metres, NaN where missing."""
JOINT_MAX_GAP_S: float = 0.1
"""Joint interpolation bridges at most this gap between two valid samples."""


@dataclass(frozen=True, slots=True)
class ClipJoints:
    """3D body joints over time (no mesh): ``positions[i]`` at ``times_s[i]``; NaN where the joint is missing."""

    entity: str
    """Source entity of the joints."""
    times_s: Float64[ndarray, " n"]
    """Sample times on ``video_time``, seconds."""
    positions: Float32[ndarray, "n joints 3"]
    """World positions, metres."""

    def at(self, times_s: Float64[ndarray, " t"]) -> Float32[ndarray, "t joints 3"]:
        """Positions linearly interpolated at ``times_s``; NaN when a neighbour is missing or the gap exceeds ``JOINT_MAX_GAP_S``."""
        right: Int64[ndarray, " t"] = np.clip(np.searchsorted(self.times_s, times_s), 1, len(self.times_s) - 1)
        left: Int64[ndarray, " t"] = right - 1
        t0: Float64[ndarray, " t"] = self.times_s[left]
        t1: Float64[ndarray, " t"] = self.times_s[right]
        weight: Float64[ndarray, " t"] = np.clip((times_s - t0) / np.maximum(t1 - t0, 1e-9), 0.0, 1.0)
        out: Float64[ndarray, "t j 3"] = self.positions[left] * (1.0 - weight)[:, None, None] + self.positions[right] * weight[:, None, None]
        bad: Bool[ndarray, " t"] = (t1 - t0 > JOINT_MAX_GAP_S) | (times_s < self.times_s[0] - 1e-6) | (times_s > self.times_s[-1] + 1e-6)
        out[bad] = np.nan
        return out.astype(np.float32)


@dataclass(frozen=True, slots=True)
class TimedClip:
    """A SelfCap clip: the training clip plus each camera's frame times and the body joints."""

    clip: MultiViewClip
    """Frames, pinholes and poses (``people`` is None: SelfCap has no SMPL-X mesh)."""
    times_s: Float64[ndarray, "cams frames"]
    """Each camera's ``video_time`` of each clip frame, seconds (camera 0 is the reference grid)."""
    joints: ClipJoints | None
    """Body joints over the window (and 0.2 s around it); None when the segment has no joints entity."""

    @property
    def reference_times_s(self) -> Float64[ndarray, " frames"]:
        """The reference camera's frame times."""
        return self.times_s[0]

    def joints_at_frames(self) -> Float32[ndarray, "frames joints 3"]:
        """Joints at the reference frame times."""
        if self.joints is None:
            raise ValueError("the clip has no joints")
        return self.joints.at(self.reference_times_s)


def _seconds(column: pa.ChunkedArray) -> Float64[ndarray, " n"]:
    return column.combine_chunks().to_numpy(zero_copy_only=False).view(np.int64).astype(np.float64) / 1e9


def _cell(table: pa.Table, name: str) -> object | None:
    if name not in table.column_names:
        return None
    return first_valid_value(table.column(name), allow_none=True, component_name=name)


def _floats(table: pa.Table, name: str) -> Float64[ndarray, " k"] | None:
    value: object | None = _cell(table, name)
    return None if value is None else np.asarray(value, dtype=np.float64).reshape(-1)


def rigid_from_columns(mat3x3: Float64[ndarray, " 9"], translation: Float64[ndarray, " 3"]) -> Float64[ndarray, "4 4"]:
    """A Rerun ``Transform3D`` (``mat3x3`` stored column-major) as a 4x4 matrix."""
    transform: Float64[ndarray, "4 4"] = np.eye(4)
    transform[:3, :3] = mat3x3.reshape(3, 3).T
    transform[:3, 3] = translation
    return transform


def static_world_T_cam(table: pa.Table, node: str) -> Float64[ndarray, "4 4"]:
    """Camera-to-world of a camera node: its ``Transform3D`` (ParentFromChild unless a relation says otherwise), composed
    with its rig node's transform when the rig has one.

    Raises:
        ValueError: If the node has no rotation/translation.
    """
    rotation: Float64[ndarray, " k"] | None = _floats(table, f"{node}:Transform3D:mat3x3")
    translation: Float64[ndarray, " k"] | None = _floats(table, f"{node}:Transform3D:translation")
    if rotation is None or translation is None or len(rotation) != 9 or len(translation) != 3:
        raise ValueError(f"{node}: no static Transform3D mat3x3 + translation")
    parent_T_cam: Float64[ndarray, "4 4"] = rigid_from_columns(rotation, translation)
    relation: object | None = _cell(table, f"{node}:Transform3D:relation")
    if isinstance(relation, list):
        relation = relation[0] if relation else None
    if relation is not None and relation not in (1, "ParentFromChild"):
        parent_T_cam = np.linalg.inv(parent_T_cam)
    rig: str = node.rsplit("/", 1)[0]
    rig_rotation = _floats(table, f"{rig}:Transform3D:mat3x3")
    rig_translation = _floats(table, f"{rig}:Transform3D:translation")
    if rig_rotation is not None and rig_translation is not None:
        parent_T_cam = rigid_from_columns(rig_rotation, rig_translation) @ parent_T_cam
    return parent_T_cam


def nearest_frames(times: Float64[ndarray, " n"], targets: Float64[ndarray, " t"], max_offset_s: float) -> Int64[ndarray, " t"]:
    """Index of the frame nearest each target time.

    Raises:
        ValueError: If a target has no frame within ``max_offset_s``.
    """
    right: Int64[ndarray, " t"] = np.clip(np.searchsorted(times, targets), 1, len(times) - 1)
    left: Int64[ndarray, " t"] = right - 1
    pick: Int64[ndarray, " t"] = np.where(np.abs(times[right] - targets) < np.abs(times[left] - targets), right, left)
    offset: Float64[ndarray, " t"] = np.abs(times[pick] - targets)
    if offset.max(initial=0.0) > max_offset_s:
        raise ValueError(f"a frame is {offset.max() * 1e3:.1f} ms from its reference time (limit {max_offset_s * 1e3:.1f} ms)")
    return pick


def read_joints(entry: DatasetEntry, segment_id: str, entity: str, start_s: float, end_s: float) -> ClipJoints | None:
    """The joints entity on ``video_time`` between ``start_s`` and ``end_s``; None when the segment has no such entity."""
    names: set[str] = {c.entity_path for c in catalog_components(entry.filter_segments(segment_id).schema())}
    if entity not in names:
        return None
    table: pa.Table = (
        entry.filter_segments(segment_id)
        .filter_contents([entity])
        .reader(index=VIDEO_TIMELINE)
        .filter((col(VIDEO_TIMELINE) >= np.timedelta64(int(start_s * 1e9), "ns")) & (col(VIDEO_TIMELINE) <= np.timedelta64(int(end_s * 1e9), "ns")))
        .select(VIDEO_TIMELINE, f"{entity}:Points3D:positions")
        .to_arrow_table()
    )
    times: Float64[ndarray, " n"] = _seconds(table.column(VIDEO_TIMELINE))
    cells: list[object | None] = table.column(f"{entity}:Points3D:positions").to_pylist()
    rows: list[int] = [i for i, cell in enumerate(cells) if isinstance(cell, list) and cell]
    if not rows:
        return None
    width: int = max(len(cells[i]) for i in rows)  # type: ignore[arg-type]
    positions: Float32[ndarray, "n j 3"] = np.full((len(rows), width, 3), np.nan, dtype=np.float32)
    for slot, i in enumerate(rows):
        values = np.asarray([[np.nan] * 3 if p is None else p for p in cells[i]], dtype=np.float32)  # type: ignore[union-attr]
        positions[slot, : len(values)] = values
    order: Int64[ndarray, " n"] = np.argsort(times[rows])
    return ClipJoints(entity=entity, times_s=times[rows][order], positions=positions[order])


def read_timed_clip(
    catalog_url: str,
    dataset: str,
    segment_id: str,
    time_start_s: float,
    num_frames: int,
    cameras: Sequence[str] | None = None,
    max_side: int | None = None,
    *,
    joints_entity: str = JOINTS_ENTITY,
    decode_device: torch.device | None = None,
    storage_device: torch.device | None = None,
) -> TimedClip:
    """Read ``num_frames`` consecutive frames from ``time_start_s`` (``video_time`` seconds) of a SelfCap segment's phones.

    Args:
        cameras: Camera names (``rig_00/cam_00``); None reads every camera whose node has ``calib_status == "ok"``
            (the four exo phones; the OAK and Quest rigs carry no such column).
        max_side: Longer side of the returned frames; None keeps the stored size.

    Raises:
        ValueError: If a camera is missing, uncalibrated or lensed, a camera has no frame near a reference time, or the
            window runs past the segment.
    """
    decode_on: torch.device = decode_device if decode_device is not None else torch.device("cuda")
    store_on: torch.device = storage_device if storage_device is not None else torch.device("cpu")
    entry: DatasetEntry = CatalogClient(catalog_url).get_dataset(dataset)
    layout: CatalogRigLayout = parse_rig_layout(catalog_components(entry.filter_segments(segment_id).schema()))
    by_name: dict[str, CatalogCamera] = {camera.name: camera for camera in layout.cameras}
    candidates: list[CatalogCamera] = [by_name[name] for name in cameras if name in by_name] if cameras is not None else list(layout.cameras)
    if cameras is not None and len(candidates) != len(cameras):
        raise ValueError(f"cameras {sorted(set(cameras) - set(by_name))} are not video cameras of {segment_id}")
    rigs: list[str] = sorted({camera.transform_entity.rsplit("/", 1)[0] for camera in candidates})
    static: pa.Table = (
        entry.filter_segments(segment_id)
        .filter_contents([camera.transform_entity for camera in candidates] + [f"{c.transform_entity}/pinhole" for c in candidates] + rigs)
        .reader(index=None)
        .to_arrow_table()
    )
    chosen: list[CatalogCamera] = (
        candidates if cameras is not None else [c for c in candidates if _cell(static, f"{c.transform_entity}:calib_status") in ("ok", ["ok"])]
    )
    if not chosen:
        raise ValueError(f"{segment_id}: no camera with calib_status ok")
    for camera in chosen:
        status: object | None = _cell(static, f"{camera.transform_entity}:calib_status")
        if status is not None and status not in ("ok", ["ok"]):
            raise ValueError(f"{segment_id} {camera.name}: calib_status {status!r}")
        distortion = _floats(static, f"{camera.transform_entity}/pinhole:simplecv.components.DistortionCoefficients")
        if distortion is not None and np.any(np.abs(distortion) > 0):
            raise ValueError(f"{segment_id} {camera.name}: lens coefficients {distortion[:4].tolist()}... are not undistorted here (pinhole only)")

    videos: list[str] = [camera.video_entity for camera in chosen]
    # is_keyframe is sparse (logged on keyframes only), so one query per camera: its rows are its frames.
    per_camera_times: list[Float64[ndarray, " m"]] = []
    per_camera_keyframes: list[Float64[ndarray, " k"]] = []
    for video in videos:
        flags: pa.Table = (
            entry.filter_segments(segment_id)
            .filter_contents([video])
            .reader(index=VIDEO_TIMELINE)
            .select(VIDEO_TIMELINE, f"{video}:VideoStream:is_keyframe")
            .to_arrow_table()
        )
        all_times: Float64[ndarray, " n"] = _seconds(flags.column(VIDEO_TIMELINE))
        key: Bool[ndarray, " n"] = np.asarray(catalog_keyframes(flags.column(f"{video}:VideoStream:is_keyframe")), dtype=bool)
        per_camera_times.append(np.unique(all_times))
        per_camera_keyframes.append(np.unique(all_times[key]))
    reference: Float64[ndarray, " m"] = per_camera_times[0]
    first: int = int(np.searchsorted(reference, time_start_s - 1e-6))
    if first + num_frames > len(reference):
        raise ValueError(f"{segment_id}: {num_frames} frames from {time_start_s} s run past the segment ({reference[-1]:.2f} s)")
    targets: Float64[ndarray, " t"] = reference[first : first + num_frames]
    period: float = float(np.median(np.diff(reference)))
    picks: list[Int64[ndarray, " t"]] = [nearest_frames(times, targets, 0.5 * period + 1e-4) for times in per_camera_times]
    times_s: Float64[ndarray, "c t"] = np.stack([times[pick] for times, pick in zip(per_camera_times, picks, strict=True)])
    window_start: float = min(float(keys[keys <= times_s[c].min() + 1e-9].max()) for c, keys in enumerate(per_camera_keyframes))
    packets: pa.Table = (
        entry.filter_segments(segment_id)
        .filter_contents(videos)
        .reader(index=VIDEO_TIMELINE)
        .filter(
            (col(VIDEO_TIMELINE) >= np.timedelta64(int(round(window_start * 1e9)) - 1000, "ns"))
            & (col(VIDEO_TIMELINE) <= np.timedelta64(int(round(times_s.max() * 1e9)) + 1000, "ns"))
        )
        .select(VIDEO_TIMELINE, *[f"{video}:VideoStream:{field}" for video in videos for field in ("sample", "is_keyframe", "codec")])
        .to_arrow_table()
    )
    fps: float = float(round(1.0 / period, 3))
    out_frames: list[UInt8[Tensor, "t h w 3"]] = []
    stored_sizes: list[tuple[int, int]] = []
    K: Float32[Tensor, "c 3 3"] = torch.empty((len(chosen), 3, 3), dtype=torch.float32)
    world_T_cam: Float32[Tensor, "c 4 4"] = torch.empty((len(chosen), 4, 4), dtype=torch.float32)
    for index, (camera, video) in enumerate(zip(chosen, videos, strict=True)):
        pinhole: str = f"{camera.transform_entity}/pinhole"
        matrix: Float64[ndarray, " k"] | None = _floats(static, f"{pinhole}:Pinhole:image_from_camera")
        resolution: Float64[ndarray, " k"] | None = _floats(static, f"{pinhole}:Pinhole:resolution")
        if matrix is None or resolution is None:
            raise ValueError(f"{segment_id} {camera.name}: no static Pinhole")
        stored_wh: tuple[int, int] = (int(resolution[0]), int(resolution[1]))
        target_wh: tuple[int, int] = fit_max_side(*stored_wh, max_side)
        stream: CatalogVideo = catalog_video_from_table(packets, video, VIDEO_TIMELINE, f"{segment_id} {video}")
        stream_times: Float64[ndarray, " n"] = stream.t_ns.astype(np.float64) / 1e9
        start: int = int(np.flatnonzero(np.asarray(stream.keyframes))[0])
        positions: list[int] = [int(np.argmin(np.abs(stream_times[start:] - t))) for t in times_s[index]]
        if np.abs(stream_times[start:][positions] - times_s[index]).max() > 1e-6:
            raise ValueError(f"{video}: some clip frames have no packet")
        decoder: VideoDecoder = VideoDecoder(
            wrap_mp4(stream.samples[start:], stream.keyframes[start:], max(1, round(fps)), codec=stream.codec),
            device=decode_on,
            seek_mode="exact",
            num_ffmpeg_threads=0,
        )
        decoded: UInt8[Tensor, "t 3 h w"] = decoder.get_frames_at(positions).data
        if (decoded.shape[3], decoded.shape[2]) != stored_wh:
            raise ValueError(f"{video}: decoded {decoded.shape[3]}x{decoded.shape[2]}, the pinhole says {stored_wh}")
        out_frames.append(
            torch.cat(
                [
                    resample_frames(decoded[s : s + RESAMPLE_CHUNK], None, target_wh).permute(0, 2, 3, 1).to(store_on)
                    for s in range(0, len(decoded), RESAMPLE_CHUNK)
                ]
            )
        )
        stored_sizes.append(stored_wh)
        K[index] = torch.from_numpy(scale_intrinsics(matrix.reshape(3, 3).T, stored_wh, target_wh).astype(np.float32))
        world_T_cam[index] = torch.from_numpy(static_world_T_cam(static, camera.transform_entity).astype(np.float32))
        del decoder, decoded
    clip = MultiViewClip(
        camera_names=tuple(camera.name for camera in chosen),
        frames=tuple(out_frames),
        valid=tuple(torch.ones((f.shape[1], f.shape[2]), dtype=torch.bool) for f in out_frames),
        K=K,
        world_T_cam=world_T_cam,
        fps=fps,
        frame_indices=torch.from_numpy((first + np.arange(num_frames)).astype(np.int64)),
        people=None,
        stored_sizes=tuple(stored_sizes),
        lenses=tuple(None for _ in chosen),
    )
    joints: ClipJoints | None = read_joints(entry, segment_id, joints_entity, float(times_s.min()) - 0.2, float(times_s.max()) + 0.2)
    return TimedClip(clip=clip, times_s=times_s, joints=joints)
