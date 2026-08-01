"""Load one ARKitScenes catalog segment into memory as 3DGS training views.

Reads RGB video frames, per-frame camera poses, and intrinsics from a Rerun
catalog through the experimental PyTorch dataloader, decodes them once
sequentially, and returns a list of in-memory views. Random-access AV1 decode
is slow (~0.5-1.3 s/frame measured), so training samples from this preloaded
list instead of hitting the dataloader per step.
"""

from dataclasses import dataclass, replace
from typing import Any, Literal, TypeAlias

import av.error
import cv2
import numpy as np
import pyarrow as pa
import torch
from arkitscenes_download.ingest.catalog import DEFAULT_CATALOG_URL
from arkitscenes_download.ingest.depth import ArkitDepthConfidence
from arkitscenes_download.ingest.paths import (
    CAM_ULTRAWIDE,
    CONFIDENCE,
    DEPTH,
    PINHOLE_ULTRAWIDE,
    PINHOLE_WIDE,
    PINHOLE_WIDE_LOWRES,
    RIG,
    VIDEO_ULTRAWIDE,
    VIDEO_WIDE,
)
from jaxtyping import Bool, Float32, Float64, UInt8
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry
from rerun.experimental.dataloader import DataSource, Field, FixedRateSampling, ImageDecoder, NumericDecoder, RerunMapDataset, VideoFrameDecoder
from scipy.spatial.transform import Rotation
from torch import Tensor

DEFAULT_SEGMENT_ID: str = "40753679"
"""Known-landscape ARKitScenes segment used as the v1 training default."""
VIDEO_FPS: float = 60.0
"""ARKitScenes wide-video frame rate; also the dataloader's densest sampling rate."""

Sample: TypeAlias = dict[str, Any]
"""One raw dataloader sample: video CHW uint8, pose_t (3,), pose_q xyzw (4,), k (9,)."""
CameraName: TypeAlias = Literal["wide", "ultrawide", "both"]
"""Which rig camera to train on. The wide camera carries the LiDAR depth."""
UndistortMaps: TypeAlias = tuple[Float32[ndarray, "h w"], Float32[ndarray, "h w"]]
"""cv2.remap source-coordinate maps (x, y) from ideal-pinhole to distorted pixels."""


class ResilientVideoFrameDecoder(VideoFrameDecoder):
    """A ``VideoFrameDecoder`` that survives per-sample AV1 decode failures.

    Copied from prompt-da's catalog tool. Root cause (verified 2026-07-31 on
    segment 40753679, rerun 0.36.0a1): ``FixedRateSampling`` draws grid
    timestamps algebraically and materializes windows with ``fill_latest_at``,
    so any packet that shares a grid slot with a later one is silently dropped
    (upstream RR-5087; camera cadence ~16.607 ms vs the 16.667 ms grid drops a
    packet wherever the phases cross — 2 per minute here). When the dropped
    packet is a reference frame, every window spanning it raises
    ``InvalidDataError`` from dav1d until the next true keyframe (GOP is 250
    frames ≈ 4.2 s; this segment got lucky — both drops were harmless
    ``show_existing_frame`` packets). The failure is per-sample, so degrade it
    to the decoder's ``None``-frame skip path. The caught exceptions are kept
    alive on purpose — releasing a failed decode's traceback frees the errored
    codec context un-drained, whose teardown can deadlock in ``dav1d_flush``.
    """

    def __init__(self, *, keyframe_interval: int = 30, fps_estimate: float = 30.0, codec: str = "h264") -> None:
        super().__init__(keyframe_interval=keyframe_interval, fps_estimate=fps_estimate, codec=codec)
        self.decode_failures: list[BaseException] = []

    def decode(self, raw: pa.ChunkedArray, index_value: int | np.datetime64 | np.timedelta64, segment_id: str) -> Tensor | None:
        """Decode one sample, degrading decoder errors to a skippable ``None``."""
        try:
            return super().decode(raw, index_value, segment_id)
        except av.error.InvalidDataError as error:
            self.decode_failures.append(error)
            return None


@dataclass(frozen=True, slots=True)
class SplatView:
    """One training view: half-res RGB, registered sensor depth, and the camera."""

    rgb_hwc: UInt8[Tensor, "h w 3"]
    """Decoded RGB frame, downscaled, on CPU."""
    cam_t_world_44: Float32[Tensor, "4 4"]
    """World→camera transform (gsplat's ``viewmats``); inverse of the logged rig pose."""
    k_33: Float32[Tensor, "3 3"]
    """Pinhole intrinsics rescaled to match ``rgb_hwc``'s resolution."""
    depth_m_hw: Float32[Tensor, "dh dw"]
    """ARKit LiDAR depth in meters, kept at its native 256x192 resolution."""
    depth_valid_hw: Bool[Tensor, "dh dw"]
    """Depth pixels that are nonzero AND high-confidence."""
    k_lowres_33: Float32[Tensor, "3 3"]
    """The depth map's own pinhole intrinsics (same camera, lower resolution)."""
    camera_index: int = 0
    """Physical camera this view came from (0 = wide, 1 = ultrawide in
    ``camera="both"`` loads; always 0 in single-camera loads)."""


@dataclass(frozen=True, slots=True)
class SegmentViewsConfig:
    """Where and how to load training views from the catalog."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """gRPC URL of the Rerun catalog server."""
    dataset_name: str = "arkitscenes"
    """Catalog dataset holding the registered segments."""
    segment_id: str = DEFAULT_SEGMENT_ID
    """Single segment (one coherent world frame) to train on."""
    camera: CameraName = "wide"
    """Rig camera: `wide` (1920x1440, has depth), `ultrawide` (640x480, no
    depth, frames undistorted at load from the stored Apple polynomial), or
    `both` (wide views first, then ultrawide, tagged via ``camera_index``)."""
    target_view_count: int = 300
    """Approximate number of views to sample uniformly over the segment."""
    downscale: int = 2
    """Integer factor applied to frame resolution and intrinsics. Use 1 for
    the already-small ultrawide."""
    fetch_chunk_size: int = 32
    """Views per batched dataloader fetch (one server query each)."""


@dataclass(frozen=True, slots=True)
class UltrawideCalibration:
    """Static ultrawide camera calibration read from the segment."""

    rig_t_cam_44: Float64[ndarray, "4 4"]
    """Rig←cam extrinsic (the rig frame is the wide camera)."""
    coefficients: Float64[ndarray, "8"]
    """Apple `LensDistortionCoefficients`: highest-power-first polynomial over
    radius/width giving a magnification delta, mapping ideal→distorted radii
    (direction verified empirically by render-overlay gradient correlation)."""
    center_xy_ref: Float64[ndarray, "2"]
    """Distortion center in calibration reference pixels."""
    reference_wh: Float64[ndarray, "2"]
    """Calibration reference dimensions in pixels."""


def load_ultrawide_calibration(dataset_entry: DatasetEntry, segment_id: str) -> UltrawideCalibration:
    """Read the static ultrawide extrinsic and distortion from the catalog."""
    table: pa.Table = (
        dataset_entry.filter_segments([segment_id]).filter_contents([f"/{CAM_ULTRAWIDE}", f"/{CAM_ULTRAWIDE}/**"]).reader(index=None).to_arrow_table()
    )

    def first_valid(column: str) -> Any:
        values: pa.ChunkedArray = table.column(column)
        for row in range(table.num_rows):
            if values[row].is_valid:
                return values[row].as_py()
        raise KeyError(f"no static value for {column!r} in segment {segment_id!r}")

    translation: Float64[ndarray, "3"] = np.asarray(first_valid(f"/{CAM_ULTRAWIDE}:Transform3D:translation")[0], dtype=np.float64)
    quat_xyzw: Float64[ndarray, "4"] = np.asarray(first_valid(f"/{CAM_ULTRAWIDE}:Transform3D:quaternion")[0], dtype=np.float64)
    rig_t_cam: Float64[ndarray, "4 4"] = np.eye(4)
    rig_t_cam[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    rig_t_cam[:3, 3] = translation
    return UltrawideCalibration(
        rig_t_cam_44=rig_t_cam,
        coefficients=np.asarray(first_valid(f"/{PINHOLE_ULTRAWIDE}:simplecv.components.DistortionCoefficients")[0], dtype=np.float64),
        center_xy_ref=np.asarray(first_valid(f"/{PINHOLE_ULTRAWIDE}:distortion_center_xy"), dtype=np.float64),
        reference_wh=np.asarray(first_valid(f"/{PINHOLE_ULTRAWIDE}:reference_dimensions_wh"), dtype=np.float64),
    )


def build_undistort_maps(calibration: UltrawideCalibration, frame_width: int, frame_height: int) -> UndistortMaps:
    """Build cv2.remap maps that rectify a distorted frame to its ideal pinhole.

    For each ideal target pixel, the source location in the distorted frame is
    ``r_src = r * (1 + polyval(coefficients, r / width))`` around the distortion
    center — Apple's polynomial is scale-invariant in ``r/width``.
    """
    scale: float = frame_width / float(calibration.reference_wh[0])
    center_xy: Float64[ndarray, "2"] = calibration.center_xy_ref * scale
    grid_x: Float64[ndarray, "h w"]
    grid_y: Float64[ndarray, "h w"]
    grid_x, grid_y = np.meshgrid(np.arange(frame_width, dtype=np.float64), np.arange(frame_height, dtype=np.float64))
    dx: Float64[ndarray, "h w"] = grid_x - center_xy[0]
    dy: Float64[ndarray, "h w"] = grid_y - center_xy[1]
    radius: Float64[ndarray, "h w"] = np.sqrt(dx * dx + dy * dy)
    magnification: Float64[ndarray, "h w"] = 1.0 + np.polyval(calibration.coefficients, radius / frame_width)
    return (center_xy[0] + dx * magnification).astype(np.float32), (center_xy[1] + dy * magnification).astype(np.float32)


def view_from_sample(sample: Sample, downscale: int, rig_t_cam_44: Float64[ndarray, "4 4"] | None = None, undistort: UndistortMaps | None = None) -> SplatView:
    """Convert one raw dataloader sample into an in-memory training view.

    Args:
        sample: Dataloader sample with ``video`` CHW uint8 tensor, ``pose_t``
            (3,) / ``pose_q`` xyzw (4,) world←rig pose, and ``k`` (9,)
            intrinsics — all torch CPU tensors. Wide samples also carry
            ``depth``/``conf``/``k_lo``; ultrawide samples don't.
        downscale: Integer resolution divisor.
        rig_t_cam_44: Optional static rig←cam extrinsic composed onto the rig
            pose (ultrawide). ``None`` means the camera IS the rig (wide).
        undistort: Optional cv2.remap maps applied to the frame before
            downscaling (ultrawide rectification).
    """
    rgb_hwc: UInt8[ndarray, "h w 3"] = sample["video"].permute(1, 2, 0).numpy()
    if undistort is not None:
        rgb_hwc = cv2.remap(rgb_hwc, undistort[0], undistort[1], interpolation=cv2.INTER_LINEAR)
    height: int = rgb_hwc.shape[0] // downscale
    width: int = rgb_hwc.shape[1] // downscale
    rgb_small: UInt8[ndarray, "h2 w2 3"] = cv2.resize(rgb_hwc, (width, height), interpolation=cv2.INTER_AREA) if downscale > 1 else rgb_hwc

    # Sensor depth stays at native resolution: uint16 millimeters -> float32
    # meters. Ultrawide has no registered depth; an all-invalid 1x1 stub keeps
    # the depth loss a no-op for those views.
    if "depth" in sample:
        depth_m_hw: Float32[Tensor, "dh dw"] = sample["depth"][0].float() / 1000.0
        conf_hw: Tensor = sample["conf"].reshape(depth_m_hw.shape)
        depth_valid_hw: Bool[Tensor, "dh dw"] = (depth_m_hw > 0.0) & (conf_hw == int(ArkitDepthConfidence.HIGH))
        k_lowres_33: Float32[ndarray, "3 3"] = sample["k_lo"].numpy().reshape(3, 3, order="F").astype(np.float32).copy()
    else:
        depth_m_hw = torch.zeros(1, 1)
        depth_valid_hw = torch.zeros(1, 1, dtype=torch.bool)
        k_lowres_33 = np.eye(3, dtype=np.float32)

    # The rig Transform3D is parent-from-child = world←rig (ingest names it
    # `world_from_camera`; cam_00 is a static identity under the rig).
    world_t_cam: Float64[ndarray, "4 4"] = np.eye(4)
    world_t_cam[:3, :3] = Rotation.from_quat(sample["pose_q"].numpy()).as_matrix()
    world_t_cam[:3, 3] = sample["pose_t"].numpy()
    if rig_t_cam_44 is not None:
        world_t_cam = world_t_cam @ rig_t_cam_44
    cam_t_world: Float32[ndarray, "4 4"] = np.linalg.inv(world_t_cam).astype(np.float32)

    # Rerun stores Pinhole image_from_camera flattened column-major.
    k_33: Float32[ndarray, "3 3"] = sample["k"].numpy().reshape(3, 3, order="F").astype(np.float32).copy()
    k_33[:2] /= float(downscale)
    return SplatView(
        rgb_hwc=torch.from_numpy(rgb_small),
        cam_t_world_44=torch.from_numpy(cam_t_world),
        k_33=torch.from_numpy(k_33),
        depth_m_hw=depth_m_hw,
        depth_valid_hw=depth_valid_hw,
        k_lowres_33=torch.from_numpy(k_lowres_33),
    )


def make_dataset(config: SegmentViewsConfig, dataset_entry: DatasetEntry) -> RerunMapDataset:
    """Build the map-style dataset for one segment at the config's sampling rate."""
    source: DataSource = DataSource(dataset=dataset_entry, segments=[config.segment_id])
    video_path: str = VIDEO_WIDE if config.camera == "wide" else VIDEO_ULTRAWIDE
    pinhole_path: str = PINHOLE_WIDE if config.camera == "wide" else PINHOLE_ULTRAWIDE
    fields: dict[str, Field] = {
        "video": Field(f"/{video_path}:VideoStream:sample", decode=ResilientVideoFrameDecoder(codec="av1", keyframe_interval=300, fps_estimate=VIDEO_FPS)),
        "pose_t": Field(f"/{RIG}:Transform3D:translation", decode=NumericDecoder()),
        "pose_q": Field(f"/{RIG}:Transform3D:quaternion", decode=NumericDecoder()),
        "k": Field(f"/{pinhole_path}:Pinhole:image_from_camera", decode=NumericDecoder()),
    }
    if config.camera == "wide":
        # LiDAR depth is registered to the wide camera only.
        fields["k_lo"] = Field(f"/{PINHOLE_WIDE_LOWRES}:Pinhole:image_from_camera", decode=NumericDecoder())
        fields["depth"] = Field(f"/{DEPTH}:EncodedDepthImage:blob", decode=ImageDecoder())
        fields["conf"] = Field(f"/{CONFIDENCE}:SegmentationImage:buffer", decode=NumericDecoder())
    # Always index at the full frame rate: a sparser FixedRateSampling grid
    # makes `fill_latest_at` drop most video packets from each decode window
    # (only near-keyframe samples survive; observed 15/24 lost). View
    # subsampling happens client-side by striding indices instead.
    return RerunMapDataset(source=source, index="video_time", fields=fields, timeline_sampling=FixedRateSampling(rate_hz=VIDEO_FPS))


def load_segment_views(config: SegmentViewsConfig) -> list[SplatView]:
    """Fetch, decode, and downscale every sampled view of the segment, in order.

    Samples whose video frame fails to decode are dropped with a notice.
    """
    if config.camera == "both":
        wide_views: list[SplatView] = load_segment_views(replace(config, camera="wide"))
        ultrawide_views: list[SplatView] = [replace(v, camera_index=1) for v in load_segment_views(replace(config, camera="ultrawide"))]
        return wide_views + ultrawide_views
    dataset_entry: DatasetEntry = CatalogClient(config.catalog_url).get_dataset(config.dataset_name)
    dataset: RerunMapDataset = make_dataset(config, dataset_entry)
    rig_t_cam: Float64[ndarray, "4 4"] | None = None
    calibration: UltrawideCalibration | None = None
    undistort: UndistortMaps | None = None
    required: tuple[str, ...] = ("video", "pose_t", "pose_q", "k", "k_lo", "depth", "conf")
    if config.camera == "ultrawide":
        calibration = load_ultrawide_calibration(dataset_entry, config.segment_id)
        rig_t_cam = calibration.rig_t_cam_44
        required = ("video", "pose_t", "pose_q", "k")
    stride: int = max(1, len(dataset) // config.target_view_count)
    wanted: list[int] = list(range(0, len(dataset), stride))
    views: list[SplatView] = []
    dropped: int = 0
    for start in range(0, len(wanted), config.fetch_chunk_size):
        indices: list[int] = wanted[start : start + config.fetch_chunk_size]
        samples: list[Sample] = dataset.__getitems__(indices)
        for sample in samples:
            if any(sample.get(name) is None for name in required):
                dropped += 1
                continue
            if calibration is not None and undistort is None:
                undistort = build_undistort_maps(calibration, int(sample["video"].shape[2]), int(sample["video"].shape[1]))
            views.append(view_from_sample(sample, config.downscale, rig_t_cam_44=rig_t_cam, undistort=undistort))
    if dropped:
        print(f"load_segment_views: dropped {dropped}/{len(wanted)} samples with missing fields")
    return views
