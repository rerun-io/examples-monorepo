"""Stream X-Lens fisheye rig depth from a Robocap catalog segment to Rerun.

The tool reads four calibrated outward-facing rig videos and the rig pose through the Rerun PyTorch dataloader,
which samples one ``video_time`` grid across all of them, and predicts per-view metric depth on the native fisheye
images. It logs the original views in 2D and same-axis, distortion-free pinhole twins whose remapped z-depth
unprojects natively in Rerun. The pinhole twins also provide valid inputs to the shared TSDF fuser. The tool never
registers a catalog layer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from jaxtyping import Float, Float32, Float64, Int64, UInt8, UInt16
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry, DatasetView
from scipy.spatial.transform import Rotation
from simplecv.camera_parameters import Fisheye62Parameters, Intrinsics, PinholeParameters, rescale_intri
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from simplecv.rerun_log_utils import RerunTyroConfig, log_open3d_mesh
from simplecv.rerun_rig_logger import log_rig_static
from simplecv.rig import CameraSensor, Rig, RigCalibration, entity_id
from torch import Tensor
from torch.utils.data import DataLoader

from monopriors.models.rig_depth import (
    AnnotatedRigDepthPredictorUnion,
    BaseRigDepthPredictor,
    RigDepthPrediction,
    XLensConfig,
    camera_type,
    unit_rays,
)
from monopriors.models.stereo_depth.rectify import remap_fisheye_image_and_depth, undistort_fisheye_to_pinhole
from monopriors.third_party.xlens.inference.geometry import fuse_point_cloud

TIMELINE: str = "video_time"
RIG_PATH: str = "world/rig_00"
DEPTH_RANGE_MM: tuple[float, float] = (0.0, 6000.0)


@dataclass(slots=True)
class RigBatch:
    """One dataloader batch of rig framesets, cameras in the configured order."""

    t_ns: list[int]
    """Grid timestamp per frameset, in nanoseconds on the ``video_time`` timeline."""
    images: UInt8[Tensor, "b s 3 h w"]
    """Decoded frames at video resolution, on the decoder's device."""
    world_T_rig: Float64[ndarray, "b 4 4"]
    """Rig pose per frameset, the latest one at or before its grid timestamp."""


@dataclass(slots=True)
class RigCollate:
    """Stack dataloader rows into one rig batch.

    A row is the dataloader's ``dict[str, Any]``: a ``TimedFrame`` under each camera key and a tensor
    under ``pose_t`` / ``pose_q``. Rerun 0.36.2 hands every grid slot to the collate, incomplete ones
    included, so the two kinds of gap are dropped here: a slot before a camera's first video packet
    carries ``None``, and a slot before the first rig pose carries an empty tensor. A batch of only
    those comes out empty, which the run loop skips.
    """

    cams: tuple[str, ...]
    """Rig cameras, in the order the predictor sees them."""

    def __call__(self, rows: list[dict[str, Any]]) -> RigBatch:
        """Stack the complete rows of one dataloader batch."""
        t_ns: list[int] = []
        framesets: list[UInt8[Tensor, "s 3 h w"]] = []
        poses: list[Float64[ndarray, "4 4"]] = []
        for row in rows:
            translation: Float[Tensor, " xyz"] = row["pose_t"]
            quaternion: Float[Tensor, " xyzw"] = row["pose_q"]
            if translation.numel() != 3 or quaternion.numel() != 4:
                continue
            frames: list[UInt8[Tensor, "3 h w"]] = []
            for cam in self.cams:
                timed_frame = row[cam]  # a TimedFrame, or None before this camera's first video packet
                if timed_frame is None:
                    break
                frames.append(timed_frame.rgb)
            if len(frames) != len(self.cams):
                continue
            pose: Float64[ndarray, "4 4"] = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(quaternion.numpy()).as_matrix()  # xyzw, as Rerun stores it
            pose[:3, 3] = translation.numpy()
            t_ns.append(row[self.cams[0]].t_ns)  # every camera answers the same grid slot
            framesets.append(torch.stack(frames))
            poses.append(pose)
        if not framesets:
            return RigBatch(t_ns=[], images=torch.empty((0, len(self.cams), 3, 0, 0), dtype=torch.uint8), world_T_rig=np.empty((0, 4, 4)))
        return RigBatch(t_ns=t_ns, images=torch.stack(framesets), world_T_rig=np.stack(poses))


@dataclass
class RigDepthCatalogConfig:
    """X-Lens inference over one calibrated Robocap rig segment."""

    rr_config: RerunTyroConfig
    """Rerun viewer, save, connect, or headless behavior."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Catalog server URL."""
    dataset: str = "robocap"
    """Catalog dataset name."""
    segment_id: str = "robocap__f408193e6447b3b0__s00000021"
    """Robocap segment recording id."""
    cams: tuple[str, ...] = ("cam_00", "cam_01", "cam_04", "cam_05")
    """Outward rig cameras; the downward wearer-facing eye cameras 02/03 are out of distribution for X-Lens."""
    fps: float = 5.0
    """Framesets per second on the catalog ``video_time`` timeline."""
    batch_size: int = 1
    """Framesets per dataloader batch and per prediction; the TRT predictor chunks it to its ``max_batch_size``."""
    start_s: float | None = None
    """Absolute ``video_time`` start in seconds; None starts at the first shared frame."""
    max_seconds: float = 60.0
    """Maximum interval processed after the chosen start."""
    width: int = 1120
    """Inference and logging width, divisible by the 14-pixel patch size."""
    height: int = 630
    """Inference and logging height, divisible by the 14-pixel patch size."""
    focal_scale: float = 0.8
    """Rectified pinhole focal length as a multiple of each fisheye camera's horizontal focal length."""
    max_depth_m: float = 20.0
    """Reject farther points and encode farther depth pixels as zero."""
    fov_max_deg: float = 85.0
    """Reject rays farther than this angle from the optical axis."""
    conf_drop_pct: float = 8.0
    """Reject this lowest global confidence percentile after geometric cleanup."""
    point_stride: int = 4
    """Log every Nth cleaned point."""
    log_points: bool = False
    """Log the cleaned world point cloud in addition to native pinhole-depth unprojection."""
    fuse: bool = True
    """TSDF-fuse the rectified pinhole twins and log a growing world mesh."""
    fusion_voxel_m: float = 0.04
    """TSDF voxel size in metres."""
    fusion_max_depth_m: float = 6.0
    """Depth beyond this is not integrated into the TSDF."""
    mesh_every: int = 300
    """Re-log the growing TSDF mesh every N framesets; the size-conscious default is once per 60-second run."""
    predictor: AnnotatedRigDepthPredictorUnion = field(default_factory=XLensConfig)
    """Calibrated rig-depth predictor."""

    def __post_init__(self) -> None:
        """Validate model geometry, sampling, and fisheye-fusion constraints."""
        if self.width < 28 or self.height < 28 or self.width % 14 != 0 or self.height % 14 != 0:
            raise ValueError(f"width and height must be multiples of 14 and at least 28, got {self.width}x{self.height}")
        if self.width * 9 != self.height * 16:
            raise ValueError(f"output must preserve the Robocap 16:9 aspect ratio, got {self.width}x{self.height}")
        if len(self.cams) < 2 or len(set(self.cams)) != len(self.cams):
            raise ValueError("cams must contain at least two unique camera entities")
        if self.fps <= 0.0 or self.max_seconds <= 0.0:
            raise ValueError("fps and max_seconds must be positive")
        if self.focal_scale <= 0.0:
            raise ValueError("focal_scale must be positive")
        if self.max_depth_m <= 0.0 or not 0.0 < self.fov_max_deg < 90.0:
            raise ValueError("max_depth_m must be positive and fov_max_deg must be between 0 and 90")
        if not 0.0 <= self.conf_drop_pct < 100.0 or self.point_stride < 1:
            raise ValueError("conf_drop_pct must be in [0, 100) and point_stride must be positive")
        if self.fusion_voxel_m <= 0.0 or self.fusion_max_depth_m <= 0.0 or self.mesh_every < 0:
            raise ValueError("fusion voxel/depth must be positive and mesh_every must be non-negative")


def rescaled_fisheye(camera: Fisheye62Parameters, *, width: int, height: int) -> Fisheye62Parameters:
    """Copy a fisheye camera with shared-helper rescaled intrinsics."""
    intrinsics: Intrinsics = rescale_intri(camera.intrinsics, target_width=width, target_height=height)
    return Fisheye62Parameters(name=camera.name, extrinsics=camera.extrinsics, intrinsics=intrinsics, distortion=camera.distortion)


def create_rig_depth_catalog_blueprint(cams: tuple[str, ...]) -> rrb.Blueprint:
    """Lay out native pinhole depth, rectified 2D grids, and secondary fisheye views."""
    rectified_cams: list[str] = [_rectified_camera_entity(cam) for cam in cams]
    rectified_images: list[rrb.Spatial2DView] = [
        rrb.Spatial2DView(origin=f"{RIG_PATH}/{cam}/pinhole/image", name=f"{cam} rectified image") for cam in rectified_cams
    ]
    rectified_depths: list[rrb.Spatial2DView] = [
        rrb.Spatial2DView(origin=f"{RIG_PATH}/{cam}/pinhole/depth", name=f"{cam} rectified depth") for cam in rectified_cams
    ]
    fisheye_images: list[rrb.Spatial2DView] = [
        rrb.Spatial2DView(origin=f"{RIG_PATH}/{cam}/rig_depth/image", name=f"{cam} fisheye image") for cam in cams
    ]
    exclusions: list[str] = [f"- $origin/rig_00/{cam}/rig_depth/**" for cam in cams]
    rectified_tab = rrb.Vertical(
        rrb.Grid(*rectified_images, grid_columns=2, name="rectified RGB"),
        rrb.Grid(*rectified_depths, grid_columns=2, name="rectified metric depth"),
        name="rectified pinhole twins",
    )
    fisheye_tab = rrb.Grid(*fisheye_images, grid_columns=2, name="original fisheye RGB")
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="moving rig + native depth + TSDF", contents=["$origin/**", *exclusions]),
            rrb.Tabs(rectified_tab, fisheye_tab, active_tab=0, name="camera views"),
            column_shares=(2, 3),
        ),
        rrb.TimePanel(timeline=TIMELINE),
        collapse_panels=True,
    )


def _depth_millimeters(depth_m_hw: Float32[ndarray, "height width"], max_depth_m: float) -> UInt16[ndarray, "height width"]:
    """Convert valid metric depth to 16-bit millimetres, with invalid pixels zeroed."""
    valid_depth: Float32[ndarray, "height width"] = np.where(
        np.isfinite(depth_m_hw) & (depth_m_hw > 0.0) & (depth_m_hw <= max_depth_m), depth_m_hw, 0.0
    ).astype(np.float32)
    return np.clip(valid_depth * 1000.0, 0.0, 65535.0).astype(np.uint16)


def _encode_depth(depth_m_hw: Float32[ndarray, "height width"], max_depth_m: float) -> bytes:
    """Encode valid metric depth as a 16-bit millimetre PNG."""
    depth_mm: UInt16[ndarray, "height width"] = _depth_millimeters(depth_m_hw, max_depth_m)
    encoded: ndarray
    ok: bool
    ok, encoded = cv2.imencode(
        ".png",
        depth_mm,
        # zlib level 1: 16 ms per 1120x630 map vs 355 ms at level 9 with the filtered
        # strategy, for a file only ~6 % larger (py-spy: level 9 was 60 % of the run).
        [cv2.IMWRITE_PNG_COMPRESSION, 1],
    )
    if not ok:
        raise RuntimeError("OpenCV failed to encode rig-depth PNG")
    return encoded.tobytes()


def _depth_colormap(depth_m_hw: Float32[ndarray, "height width"], max_depth_m: float) -> UInt8[ndarray, "height width 3"]:
    """Render metric depth as a compact 2D-only Turbo colormap."""
    valid: ndarray = np.isfinite(depth_m_hw) & (depth_m_hw > 0.0)
    normalized: UInt8[ndarray, "height width"] = np.clip(depth_m_hw / max_depth_m * 255.0, 0.0, 255.0).astype(np.uint8)
    color_bgr: UInt8[ndarray, "height width 3"] = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color_rgb: UInt8[ndarray, "height width 3"] = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    color_rgb[~valid] = 0
    return color_rgb


def _camera_index(cam: str) -> int:
    """Read the numeric index from a canonical ``cam_NN`` entity name."""
    prefix, separator, suffix = cam.partition("_")
    if prefix != "cam" or separator != "_" or not suffix.isdigit():
        raise ValueError(f"camera entity must use cam_NN form, got {cam!r}")
    return int(suffix)


def _rectified_camera_index(cam: str) -> int:
    """Return the rig sensor index reserved for one virtual pinhole twin."""
    return 10 + _camera_index(cam)


def _rectified_camera_entity(cam: str) -> str:
    """Map ``cam_0i`` to its same-pose virtual pinhole twin ``cam_1i``."""
    return entity_id("cam", _rectified_camera_index(cam))


def main(config: RigDepthCatalogConfig) -> None:
    """Decode, predict, and stream one calibrated multi-fisheye segment."""
    from rerun.experimental.dataloader import (
        DataSource,
        Field,
        FixedRateSampling,
        NoShuffle,
        NumericDecoder,
        RerunIterableDataset,
        SegmentMetadata,
    )
    from simplecv.rerun_dataloader import TimedNvdecDecoder

    from monopriors.apis.stereo_catalog import read_fisheye_camera, read_static

    if not torch.cuda.is_available():
        raise RuntimeError("rig-depth catalog inference requires CUDA for X-Lens and NVDEC")
    device = torch.device("cuda")
    dataset: DatasetEntry = CatalogClient(config.catalog_url).get_dataset(config.dataset)
    view: DatasetView = dataset.filter_segments(config.segment_id)
    source_cameras: dict[str, Fisheye62Parameters] = {cam: read_fisheye_camera(view, cam) for cam in config.cams}
    model_cameras: dict[str, Fisheye62Parameters] = {
        cam: rescaled_fisheye(camera, width=config.width, height=config.height) for cam, camera in source_cameras.items()
    }

    codec_value: int = int(np.asarray(read_static(view, f"{RIG_PATH}/{config.cams[0]}/pinhole/video", "VideoStream:codec")).ravel()[0])
    video_codec = rr.VideoCodec(codec_value)
    if video_codec == rr.VideoCodec.H264:
        codec = "h264"
    elif video_codec == rr.VideoCodec.AV1:
        codec = "av1"
    else:
        raise ValueError(f"unsupported catalog video codec: {video_codec}")
    fields: dict[str, Field] = {
        cam: Field(
            f"/{RIG_PATH}/{cam}/pinhole/video:VideoStream:sample",
            decode=TimedNvdecDecoder(dataset, f"{RIG_PATH}/{cam}/pinhole/video", TIMELINE, device, 30, codec),
        )
        for cam in config.cams
    }
    fields |= {
        "pose_t": Field(f"/{RIG_PATH}:Transform3D:translation", decode=NumericDecoder()),
        "pose_q": Field(f"/{RIG_PATH}:Transform3D:quaternion", decode=NumericDecoder()),
    }
    samples: RerunIterableDataset = RerunIterableDataset(
        DataSource(dataset, [config.segment_id]),
        index=TIMELINE,
        fields=fields,
        timeline_sampling=FixedRateSampling(rate_hz=config.fps),
        shuffle_strategy=NoShuffle(),
    )
    # One decoder holds a DatasetEntry and one CUDA decoder per camera, so neither survives a fork.
    loader: DataLoader = DataLoader(samples, batch_size=config.batch_size, num_workers=0, collate_fn=RigCollate(config.cams))

    segments: list[SegmentMetadata] = samples.sample_index.segments
    if len(segments) != 1:
        raise ValueError(f"the dataloader index holds {len(segments)} segments for {config.segment_id!r}, expected one")
    index_start_ns: int = int(segments[0].index_start)
    index_end_ns: int = int(segments[0].index_end)
    step_ns: int = round(1e9 / config.fps)
    # The dataloader lays its grid from the segment's first index value, so snap the requested
    # start up to the next grid point (ceil division) and keep the window a whole number of steps.
    start_offset_ns: int = 0 if config.start_s is None else round(config.start_s * 1e9) - index_start_ns
    start_ns: int = index_start_ns + max(0, -(-start_offset_ns // step_ns)) * step_ns
    end_ns: int = min(index_end_ns, start_ns + round(config.max_seconds * 1e9))
    if start_ns > end_ns:
        raise ValueError(f"requested interval [{start_ns / 1e9:.3f}, {end_ns / 1e9:.3f}] lies outside the segment")
    expected: int = (end_ns - start_ns) // step_ns + 1

    rays: Float32[ndarray, "views height width 3"] = np.stack([unit_rays(model_cameras[cam]) for cam in config.cams])
    cam_types: Int64[ndarray, "views"] = np.asarray([camera_type(model_cameras[cam]) for cam in config.cams], dtype=np.int64)
    rig_T_cam: Float64[ndarray, "views 4 4"] = np.stack(
        [np.asarray(source_cameras[cam].extrinsics.world_T_cam, dtype=np.float64) for cam in config.cams]
    )
    undistortion: dict[str, tuple[PinholeParameters, Float32[ndarray, "h w"], Float32[ndarray, "h w"]]] = {
        cam: undistort_fisheye_to_pinhole(
            model_cameras[cam], focal_scale=config.focal_scale, width=config.width, height=config.height
        )
        for cam in config.cams
    }
    rectified_cameras: dict[str, PinholeParameters] = {cam: values[0] for cam, values in undistortion.items()}

    rig = Rig(
        index=0,
        calibration=RigCalibration(
            cameras=(
                [
                    CameraSensor(index=_camera_index(cam), name=source_cameras[cam].name, kind="rgb", pinhole=source_cameras[cam])
                    for cam in config.cams
                ]
                + [
                    CameraSensor(
                        index=_rectified_camera_index(cam), name=rectified_cameras[cam].name, kind="rgb", pinhole=rectified_cameras[cam]
                    )
                    for cam in config.cams
                ]
            ),
            reference_index=_camera_index(config.cams[0]),
        ),
        image_plane_distance=0.1,
    )
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    log_rig_static(rig)
    rr.send_blueprint(create_rig_depth_catalog_blueprint(config.cams))

    predictor: BaseRigDepthPredictor = config.predictor.setup(device="cuda")
    fuser: Open3DFuser | None = (
        Open3DFuser(fusion_resolution=config.fusion_voxel_m, max_fusion_depth=config.fusion_max_depth_m) if config.fuse else None
    )
    decode_total_s: float = 0.0
    predict_total_s: float = 0.0
    log_total_s: float = 0.0
    processed: int = 0
    mesh_pending: bool = False
    run_started: float = time.perf_counter()
    print(
        f"{codec} segment {config.segment_id}: {expected} framesets, {len(config.cams)} cameras, "
        f"{config.width}x{config.height} at {config.fps:.3f} fps, {config.batch_size} per batch"
    )

    wait_started: float = time.perf_counter()
    for batch in loader:
        # "decode" now covers the wait on the loader (which owns the NVDEC decoders) plus the
        # host-side resize onto the network grid, so it stays comparable to the pre-loader runs.
        selected: list[int] = []
        resized: list[UInt8[ndarray, "views height width 3"]] = []
        for row, row_t_ns in enumerate(batch.t_ns):
            if not start_ns <= row_t_ns <= end_ns:
                continue
            views: list[UInt8[ndarray, "height width 3"]] = []
            for view_index in range(len(config.cams)):
                frame: UInt8[ndarray, "video_height video_width 3"] = rearrange(batch.images[row, view_index], "c h w -> h w c").cpu().numpy()
                views.append(cv2.resize(frame, (config.width, config.height), interpolation=cv2.INTER_AREA))
            selected.append(row)
            resized.append(np.stack(views))
        decode_s: float = time.perf_counter() - wait_started
        reached_end: bool = len(batch.t_ns) > 0 and batch.t_ns[-1] >= end_ns
        if not selected:
            if reached_end:
                break
            wait_started = time.perf_counter()
            continue
        batch_images: UInt8[ndarray, "framesets views height width 3"] = np.stack(resized)

        predict_started: float = time.perf_counter()
        predictions: list[RigDepthPrediction] = predictor.predict_batch(batch_images, rays, cam_types, rig_T_cam)
        torch.cuda.synchronize()
        predict_s: float = time.perf_counter() - predict_started

        log_started: float = time.perf_counter()
        points_count: int = 0
        for frameset_index, row in enumerate(selected):
            t_ns: int = batch.t_ns[row]
            images: UInt8[ndarray, "views height width 3"] = batch_images[frameset_index]
            prediction: RigDepthPrediction = predictions[frameset_index]
            world_T_rig: Float64[ndarray, "4 4"] = batch.world_T_rig[row]
            rig_T_world: Float64[ndarray, "4 4"] = np.linalg.inv(world_T_rig)
            depth: Float32[ndarray, "views height width"] = prediction.depth_m.detach().cpu().numpy()
            remapped: list[tuple[UInt8[ndarray, "height width 3"], Float32[ndarray, "height width"]]] = [
                remap_fisheye_image_and_depth(images[view_index], depth[view_index], undistortion[cam][1], undistortion[cam][2])
                for view_index, cam in enumerate(config.cams)
            ]
            rectified_images: UInt8[ndarray, "views height width 3"] = np.stack([values[0] for values in remapped])
            rectified_depth: Float32[ndarray, "views height width"] = np.stack([values[1] for values in remapped])

            rr.set_time(TIMELINE, duration=np.timedelta64(t_ns, "ns"))
            rr.log(RIG_PATH, rr.Transform3D(mat3x3=world_T_rig[:3, :3], translation=world_T_rig[:3, 3]))
            for view_index, cam in enumerate(config.cams):
                fisheye_path: str = f"{RIG_PATH}/{cam}/rig_depth"
                rectified_cam: str = _rectified_camera_entity(cam)
                pinhole_path: str = f"{RIG_PATH}/{rectified_cam}/pinhole"
                rr.log(f"{fisheye_path}/image", rr.Image(images[view_index]).compress(jpeg_quality=85))
                rr.log(
                    f"{fisheye_path}/depth",
                    rr.Image(_depth_colormap(depth[view_index], config.fusion_max_depth_m)).compress(jpeg_quality=85),
                )
                rr.log(f"{pinhole_path}/image", rr.Image(rectified_images[view_index]).compress(jpeg_quality=85))
                rr.log(
                    f"{pinhole_path}/depth",
                    rr.EncodedDepthImage(
                        blob=_encode_depth(rectified_depth[view_index], config.max_depth_m),
                        media_type="image/png",
                        meter=1000.0,
                        depth_range=DEPTH_RANGE_MM,
                    ),
                )
                if fuser is not None:
                    rectified_camera: PinholeParameters = rectified_cameras[cam]
                    cam_T_world: Float64[ndarray, "4 4"] = rectified_camera.extrinsics.cam_T_world @ rig_T_world
                    fuser.fuse_frames(
                        np.ascontiguousarray(_depth_millimeters(rectified_depth[view_index], config.max_depth_m)),
                        np.asarray(rectified_camera.intrinsics.k_matrix, dtype=np.float64),
                        cam_T_world,
                        np.ascontiguousarray(rectified_images[view_index]),
                    )
            if config.log_points:
                world_T_cam: Float64[ndarray, "views 4 4"] = np.stack([world_T_rig @ camera_pose for camera_pose in rig_T_cam])
                confidence: Float32[ndarray, "views height width"] = prediction.confidence.detach().cpu().numpy()
                points: Float32[ndarray, "points 3"]
                colors: UInt8[ndarray, "points 3"] | None
                points, colors = fuse_point_cloud(
                    depth,
                    rays,
                    world_T_cam,
                    rgb=images,
                    conf=confidence,
                    conf_drop_pct=config.conf_drop_pct,
                    max_depth=config.max_depth_m,
                    fov_max_deg=config.fov_max_deg,
                )
                points = points[:: config.point_stride]
                if colors is None:
                    raise RuntimeError("X-Lens point-cloud fusion did not return image colors")
                colors = colors[:: config.point_stride]
                points_count = len(points)
                rr.log("world/rig_depth/points", rr.Points3D(points, colors=colors))
            processed += 1
            mesh_pending = fuser is not None
            if fuser is not None and config.mesh_every and processed % config.mesh_every == 0:
                mesh = fuser.get_mesh()
                log_open3d_mesh("world/rig_depth/mesh", mesh)
                mesh_pending = False
        log_s: float = time.perf_counter() - log_started

        decode_total_s += decode_s
        predict_total_s += predict_s
        log_total_s += log_s
        print(
            f"framesets {processed - len(selected) + 1:04d}-{processed:04d}/{expected:04d} t={batch.t_ns[selected[0]] / 1e9:.3f}s "
            f"decode={decode_s * 1000.0:.1f}ms predict={predict_s * 1000.0:.1f}ms "
            f"log={log_s * 1000.0:.1f}ms points={points_count if config.log_points else 'off'}"
        )
        if reached_end:
            break
        wait_started = time.perf_counter()

    if processed == 0:
        raise RuntimeError(
            f"no grid slot in [{start_ns / 1e9:.3f}, {end_ns / 1e9:.3f}] carried all {len(config.cams)} camera frames and a rig pose"
        )
    if fuser is not None and mesh_pending:
        mesh = fuser.get_mesh()
        log_open3d_mesh("world/rig_depth/mesh", mesh)
        print(f"fused mesh logged: {len(mesh.vertices)} vertices")
    wall_s: float = time.perf_counter() - run_started
    print(
        f"totals: framesets={processed} decode={decode_total_s:.3f}s predict={predict_total_s:.3f}s "
        f"log={log_total_s:.3f}s wall={wall_s:.3f}s; "
        f"means={decode_total_s / processed * 1000.0:.2f}/{predict_total_s / processed * 1000.0:.2f}/"
        f"{log_total_s / processed * 1000.0:.2f} ms"
    )
