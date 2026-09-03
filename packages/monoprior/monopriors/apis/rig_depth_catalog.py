"""Stream X-Lens fisheye rig depth from a Robocap catalog segment to Rerun.

The tool samples the six calibrated rig videos on one ``video_time`` grid,
predicts per-view metric depth, and logs fisheye-safe 2D images plus a fused
world point cloud. It never registers a catalog layer. The shared TSDF fuser is
pinhole-only, so this tool deliberately does not synthesize a fisheye TSDF.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, cast, runtime_checkable

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from jaxtyping import Float32, Float64, Int64, Shaped, UInt8, UInt16
from numpy import ndarray
from rerun.catalog import CatalogClient, DatasetEntry, DatasetView
from simplecv.camera_parameters import Fisheye62Parameters, Intrinsics, rescale_intri
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.rerun_rig_logger import log_rig_static
from simplecv.rig import CameraSensor, Rig, RigCalibration

from monopriors.models.rig_depth import (
    AnnotatedRigDepthPredictorUnion,
    BaseRigDepthPredictor,
    RigDepthPrediction,
    XLensConfig,
    camera_type,
    unit_rays,
)
from monopriors.third_party.xlens.inference.geometry import fuse_point_cloud

TIMELINE: str = "video_time"
RIG_PATH: str = "world/rig_00"
DEPTH_RANGE_MM: tuple[float, float] = (0.0, 6000.0)
TimedeltaNs: TypeAlias = Shaped[ndarray, " samples"]


@runtime_checkable
class VideoFrame(Protocol):
    """Decoded RGB frame returned by TorchCodec."""

    data: UInt8[torch.Tensor, "3 height width"]


@runtime_checkable
class SegmentVideoDecoder(Protocol):
    """TorchCodec operation used by the frame sampler."""

    def get_frame_at(self, index: int) -> VideoFrame:
        """Decode one frame by sample index."""
        ...


DecoderBundle: TypeAlias = tuple[TimedeltaNs, list[bytes], list[bool], SegmentVideoDecoder]


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
    cams: tuple[str, ...] = ("cam_00", "cam_01", "cam_02", "cam_03", "cam_04", "cam_05")
    """Rig camera entities sampled together."""
    fps: float = 5.0
    """Framesets per second on the catalog ``video_time`` timeline."""
    start_s: float | None = None
    """Absolute ``video_time`` start in seconds; None starts at the first shared frame."""
    max_seconds: float = 60.0
    """Maximum interval processed after the chosen start."""
    width: int = 896
    """Inference and logging width, divisible by the 14-pixel patch size."""
    height: int = 504
    """Inference and logging height, divisible by the 14-pixel patch size."""
    max_depth_m: float = 20.0
    """Reject farther points and encode farther depth pixels as zero."""
    fov_max_deg: float = 85.0
    """Reject rays farther than this angle from the optical axis."""
    conf_drop_pct: float = 8.0
    """Reject this lowest global confidence percentile after geometric cleanup."""
    point_stride: int = 4
    """Log every Nth cleaned point."""
    fuse: bool = False
    """Reserved for TSDF; must remain false because the shared fuser is pinhole-only."""
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
        if self.max_depth_m <= 0.0 or not 0.0 < self.fov_max_deg < 90.0:
            raise ValueError("max_depth_m must be positive and fov_max_deg must be between 0 and 90")
        if not 0.0 <= self.conf_drop_pct < 100.0 or self.point_stride < 1:
            raise ValueError("conf_drop_pct must be in [0, 100) and point_stride must be positive")
        if self.fuse:
            raise ValueError("fisheye TSDF is unsupported: the shared Open3DFuser accepts pinhole depth only; use fused points")


def nearest_time_index(times: TimedeltaNs, target_ns: int) -> int:
    """Return the sample nearest one nanosecond timestamp, clamped to endpoints."""
    if len(times) == 0:
        raise ValueError("cannot sample an empty timestamp sequence")
    numeric_times: Int64[ndarray, "samples"] = np.asarray(times, dtype="timedelta64[ns]").astype(np.int64)
    insertion: int = int(np.searchsorted(numeric_times, target_ns))
    if insertion == 0:
        return 0
    if insertion == len(numeric_times):
        return len(numeric_times) - 1
    before: int = insertion - 1
    return before if target_ns - int(numeric_times[before]) <= int(numeric_times[insertion]) - target_ns else insertion


def rescaled_fisheye(camera: Fisheye62Parameters, *, width: int, height: int) -> Fisheye62Parameters:
    """Copy a fisheye camera with shared-helper rescaled intrinsics."""
    intrinsics: Intrinsics = rescale_intri(camera.intrinsics, target_width=width, target_height=height)
    return Fisheye62Parameters(name=camera.name, extrinsics=camera.extrinsics, intrinsics=intrinsics, distortion=camera.distortion)


def create_rig_depth_catalog_blueprint(cams: tuple[str, ...]) -> rrb.Blueprint:
    """Lay out a fisheye-safe 3D view and 3x2 image/depth grids."""
    image_views: list[rrb.Spatial2DView] = [rrb.Spatial2DView(origin=f"{RIG_PATH}/{cam}/rig_depth/image", name=f"{cam} image") for cam in cams]
    depth_views: list[rrb.Spatial2DView] = [rrb.Spatial2DView(origin=f"{RIG_PATH}/{cam}/rig_depth/depth", name=f"{cam} depth") for cam in cams]
    exclusions: list[str] = [
        exclusion for cam in cams for exclusion in (f"- $origin/rig_00/{cam}/rig_depth/image", f"- $origin/rig_00/{cam}/rig_depth/depth")
    ]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="moving rig + fused X-Lens points", contents=["$origin/**", *exclusions]),
            rrb.Vertical(
                rrb.Grid(*image_views, grid_columns=3, name="fisheye RGB"),
                rrb.Grid(*depth_views, grid_columns=3, name="metric depth"),
            ),
            column_shares=(2, 3),
        ),
        rrb.TimePanel(timeline=TIMELINE),
        collapse_panels=True,
    )


def _encode_depth(depth_m_hw: Float32[ndarray, "height width"], max_depth_m: float) -> bytes:
    """Encode valid metric depth as a 16-bit millimetre PNG."""
    valid_depth: Float32[ndarray, "height width"] = np.where(
        np.isfinite(depth_m_hw) & (depth_m_hw > 0.0) & (depth_m_hw <= max_depth_m), depth_m_hw, 0.0
    ).astype(np.float32)
    depth_mm: UInt16[ndarray, "height width"] = np.clip(valid_depth * 1000.0, 0.0, 65535.0).astype(np.uint16)
    encoded: ndarray
    ok: bool
    ok, encoded = cv2.imencode(".png", depth_mm, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError("OpenCV failed to encode rig-depth PNG")
    return encoded.tobytes()


def _camera_index(cam: str) -> int:
    """Read the numeric index from a canonical ``cam_NN`` entity name."""
    prefix, separator, suffix = cam.partition("_")
    if prefix != "cam" or separator != "_" or not suffix.isdigit():
        raise ValueError(f"camera entity must use cam_NN form, got {cam!r}")
    return int(suffix)


def main(config: RigDepthCatalogConfig) -> None:
    """Decode, predict, and stream one calibrated multi-fisheye segment."""
    from simplecv.rerun_dataloader import open_segment_decoder

    from monopriors.apis.stereo_catalog import read_fisheye_camera, read_rig_poses, read_static

    if not torch.cuda.is_available():
        raise RuntimeError("rig-depth catalog inference requires CUDA for X-Lens and NVDEC")
    device = torch.device("cuda")
    dataset: DatasetEntry = CatalogClient(config.catalog_url).get_dataset(config.dataset)
    view: DatasetView = dataset.filter_segments(config.segment_id)
    source_cameras: dict[str, Fisheye62Parameters] = {cam: read_fisheye_camera(view, cam) for cam in config.cams}
    model_cameras: dict[str, Fisheye62Parameters] = {
        cam: rescaled_fisheye(camera, width=config.width, height=config.height) for cam, camera in source_cameras.items()
    }
    pose_times, world_T_rig_values = read_rig_poses(view)

    codec_value: int = int(np.asarray(read_static(view, f"{RIG_PATH}/{config.cams[0]}/pinhole/video", "VideoStream:codec")).ravel()[0])
    video_codec = rr.VideoCodec(codec_value)
    if video_codec == rr.VideoCodec.H264:
        codec = "h264"
    elif video_codec == rr.VideoCodec.AV1:
        codec = "av1"
    else:
        raise ValueError(f"unsupported catalog video codec: {video_codec}")
    decoders: dict[str, DecoderBundle] = {
        cam: cast(
            DecoderBundle,
            open_segment_decoder(dataset, config.segment_id, f"{RIG_PATH}/{cam}/pinhole/video", TIMELINE, device, 30, codec),
        )
        for cam in config.cams
    }

    shared_start_ns: int = max(int(bundle[0][0].astype(np.int64)) for bundle in decoders.values())
    shared_end_ns: int = min(int(bundle[0][-1].astype(np.int64)) for bundle in decoders.values())
    start_ns: int = shared_start_ns if config.start_s is None else max(shared_start_ns, round(config.start_s * 1e9))
    end_ns: int = min(shared_end_ns, start_ns + round(config.max_seconds * 1e9))
    if start_ns >= end_ns:
        raise ValueError(f"requested interval [{start_ns / 1e9:.3f}, {end_ns / 1e9:.3f}] has no shared video samples")
    step_ns: int = round(1e9 / config.fps)
    grid_ns: Int64[ndarray, "framesets"] = np.arange(start_ns, end_ns, step_ns, dtype=np.int64)

    rays: Float32[ndarray, "views height width 3"] = np.stack([unit_rays(model_cameras[cam]) for cam in config.cams])
    cam_types: Int64[ndarray, "views"] = np.asarray([camera_type(model_cameras[cam]) for cam in config.cams], dtype=np.int64)
    rig_T_cam: Float64[ndarray, "views 4 4"] = np.stack(
        [np.asarray(source_cameras[cam].extrinsics.world_T_cam, dtype=np.float64) for cam in config.cams]
    )

    rig = Rig(
        index=0,
        calibration=RigCalibration(
            cameras=[
                CameraSensor(index=_camera_index(cam), name=source_cameras[cam].name, kind="rgb", pinhole=source_cameras[cam]) for cam in config.cams
            ],
            reference_index=_camera_index(config.cams[0]),
        ),
        image_plane_distance=0.1,
    )
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    log_rig_static(rig)
    rr.send_blueprint(create_rig_depth_catalog_blueprint(config.cams))

    predictor: BaseRigDepthPredictor = config.predictor.setup(device="cuda")
    decode_total_s: float = 0.0
    predict_total_s: float = 0.0
    log_total_s: float = 0.0
    run_started: float = time.perf_counter()
    print(
        f"{codec} segment {config.segment_id}: {len(grid_ns)} framesets, {len(config.cams)} cameras, "
        f"{config.width}x{config.height} at {config.fps:.3f} fps"
    )

    for frameset_index, t_ns_value in enumerate(grid_ns.tolist()):
        t_ns: int = int(t_ns_value)
        decode_started: float = time.perf_counter()
        images: UInt8[ndarray, "views height width 3"] = np.stack(
            [
                cv2.resize(
                    rearrange(decoders[cam][3].get_frame_at(nearest_time_index(decoders[cam][0], t_ns)).data, "c h w -> h w c").cpu().numpy(),
                    (config.width, config.height),
                    interpolation=cv2.INTER_AREA,
                )
                for cam in config.cams
            ]
        )
        decode_s: float = time.perf_counter() - decode_started

        predict_started: float = time.perf_counter()
        prediction: RigDepthPrediction = predictor(images, rays, cam_types, rig_T_cam)
        torch.cuda.synchronize()
        predict_s: float = time.perf_counter() - predict_started

        log_started: float = time.perf_counter()
        pose_index: int = nearest_time_index(pose_times, t_ns)
        world_T_rig: Float64[ndarray, "4 4"] = np.asarray(world_T_rig_values[pose_index], dtype=np.float64)
        world_T_cam: Float64[ndarray, "views 4 4"] = np.stack([world_T_rig @ camera_pose for camera_pose in rig_T_cam])
        depth: Float32[ndarray, "views height width"] = prediction.depth_m.detach().cpu().numpy()
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

        rr.set_time(TIMELINE, duration=np.timedelta64(t_ns, "ns"))
        rr.log(RIG_PATH, rr.Transform3D(mat3x3=world_T_rig[:3, :3], translation=world_T_rig[:3, 3]))
        for view_index, cam in enumerate(config.cams):
            output_path: str = f"{RIG_PATH}/{cam}/rig_depth"
            rr.log(f"{output_path}/image", rr.Image(images[view_index]).compress(jpeg_quality=85))
            rr.log(
                f"{output_path}/depth",
                rr.EncodedDepthImage(
                    blob=_encode_depth(depth[view_index], config.max_depth_m),
                    media_type="image/png",
                    meter=1000.0,
                    depth_range=DEPTH_RANGE_MM,
                ),
            )
        rr.log("world/rig_depth/points", rr.Points3D(points, colors=colors))
        log_s: float = time.perf_counter() - log_started

        decode_total_s += decode_s
        predict_total_s += predict_s
        log_total_s += log_s
        print(
            f"frameset {frameset_index + 1:04d}/{len(grid_ns):04d} t={t_ns / 1e9:.3f}s "
            f"decode={decode_s * 1000.0:.1f}ms predict={predict_s * 1000.0:.1f}ms "
            f"log={log_s * 1000.0:.1f}ms points={len(points)}"
        )

    wall_s: float = time.perf_counter() - run_started
    print(
        f"totals: framesets={len(grid_ns)} decode={decode_total_s:.3f}s predict={predict_total_s:.3f}s "
        f"log={log_total_s:.3f}s wall={wall_s:.3f}s; "
        f"means={decode_total_s / len(grid_ns) * 1000.0:.2f}/{predict_total_s / len(grid_ns) * 1000.0:.2f}/"
        f"{log_total_s / len(grid_ns) * 1000.0:.2f} ms"
    )
