import contextlib
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pyrealsense2 as rs  # type: ignore
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float, UInt8
from numpy import ndarray

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole


def _quat_to_rot_m33(x: float, y: float, z: float, w: float) -> Float[ndarray, "3 3"]:
    """Convert quaternion (x,y,z,w) to 3x3 rotation matrix."""
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    R: Float[ndarray, "3 3"] = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return R


def _setup_t265_input(serial: str | None):
    """Start a T265 pipeline configured for left/right fisheye streams.

    Returns (pipeline, rs, width, height)
    """

    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    if serial:
        rs_cfg.enable_device(serial)

    # Enable both fisheye streams (848x800 @ 30Hz, Y8)
    rs_cfg.enable_stream(rs.stream.fisheye, 1, 848, 800, rs.format.y8, 30)
    rs_cfg.enable_stream(rs.stream.fisheye, 2, 848, 800, rs.format.y8, 30)
    # Also enable pose for SLAM trajectory
    rs_cfg.enable_stream(rs.stream.pose)

    pipeline.start(rs_cfg)

    return pipeline


def _setup_output_stream(width: int, height: int, fps: int) -> av.VideoStream:
    """Create an H.264 encoder stream using PyAV, AnnexB bitstream.

    Keep it close to the Rerun example: minimal options, low latency, no B-frames.
    Returns an av.video.VideoStream instance.
    """
    output_container = av.open("/dev/null", "w", format="h264")  # Use AnnexB H.264 stream.
    # Explicitly pass intended framerate so timestamps align with fps
    output_stream: av.VideoStream = output_container.add_stream("libx264", rate=int(fps))
    output_stream.width = int(width)
    output_stream.height = int(height)
    # Constrain all related timing fields to desired FPS
    output_stream.time_base = Fraction(1, int(fps))
    output_stream.codec_context.framerate = Fraction(int(fps), 1)
    output_stream.codec_context.time_base = Fraction(1, int(fps))

    # Configure for low latency.
    output_stream.codec_context.options = {
        "tune": "zerolatency",
        "preset": "veryfast",
    }
    output_stream.max_b_frames = 0  # Avoid b-frames for lower latency.

    return output_stream


@dataclass
class RectificationSetup:
    """Precomputed rectification artifacts for fisheye→pinhole.

    Includes undistort/rectify maps, virtual pinhole intrinsics, rectification
    rotations R1/R2, and mid-frame transforms to each eye.
    """

    left_rect_map1: Float[ndarray, "h w"]  # x-map for left remap
    left_rect_map2: Float[ndarray, "h w"]  # y-map for left remap
    right_rect_map1: Float[ndarray, "h w"]  # x-map for right remap
    right_rect_map2: Float[ndarray, "h w"]  # y-map for right remap
    width_px: int
    height_px: int
    left_intrinsics: Intrinsics
    right_intrinsics: Intrinsics
    R1: Float[ndarray, "3 3"]
    R2: Float[ndarray, "3 3"]
    mid_T_left: Float[ndarray, "4 4"]
    mid_T_right: Float[ndarray, "4 4"]


def _init_rectification(pipeline: rs.pipeline, rect_fov_deg: float) -> RectificationSetup:
    """Initialize rectification and return maps/intrinsics/transforms.

    Uses current active pipeline profile to derive all needed matrices and maps.
    """
    profile = pipeline.get_active_profile()
    left_stream = profile.get_stream(rs.stream.fisheye, 1).as_video_stream_profile()
    right_stream = profile.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()
    pose_stream = profile.get_stream(rs.stream.pose)

    li = left_stream.get_intrinsics()
    ri = right_stream.get_intrinsics()
    height: int = int(li.height)

    K1: Float[ndarray, "3 3"] = np.array(
        [[li.fx, 0.0, li.ppx], [0.0, li.fy, li.ppy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    D1: Float[ndarray, "4"] = np.array(li.coeffs[:4], dtype=np.float64)
    K2: Float[ndarray, "3 3"] = np.array(
        [[ri.fx, 0.0, ri.ppx], [0.0, ri.fy, ri.ppy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    D2: Float[ndarray, "4"] = np.array(ri.coeffs[:4], dtype=np.float64)

    # Left->Right extrinsics
    lr_ex = left_stream.get_extrinsics_to(right_stream)
    R_lr_raw: Float[ndarray, "3 3"] = np.reshape(np.array(lr_ex.rotation, dtype=np.float64), (3, 3))
    R_lr: Float[ndarray, "3 3"] = R_lr_raw.T
    t_lr: Float[ndarray, "3"] = np.array(lr_ex.translation, dtype=np.float64)

    # Manual pinhole projection similar to Intel example
    rect_height_px: int = height
    rect_fov_rad: float = float(rect_fov_deg) * (np.pi / 180.0)
    rect_fx: float = rect_height_px / 2.0 / np.tan(rect_fov_rad / 2.0)
    rect_fy: float = rect_fx
    rect_width_px: int = rect_height_px
    cx: float = (rect_width_px - 1) / 2.0
    cy: float = (rect_height_px - 1) / 2.0

    R1: Float[ndarray, "3 3"] = np.eye(3, dtype=np.float64)
    R2: Float[ndarray, "3 3"] = R_lr.astype(np.float64)
    P1: Float[ndarray, "3 4"] = np.array(
        [[rect_fx, 0.0, cx, 0.0], [0.0, rect_fy, cy, 0.0], [0.0, 0.0, 1.0, 0.0]], dtype=np.float64
    )
    P2: Float[ndarray, "3 4"] = P1.copy()
    P2[0, 3] = t_lr[0] * rect_fx

    new_size: tuple[int, int] = (rect_width_px, rect_height_px)
    left_rect_map1, left_rect_map2 = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1[:, :3], new_size, cv2.CV_32FC1)
    right_rect_map1, right_rect_map2 = cv2.fisheye.initUndistortRectifyMap(
        K2, D2, R2, P2[:, :3], new_size, cv2.CV_32FC1
    )

    # Rectified intrinsics
    K1_rect: Float[ndarray, "3 3"] = P1[:, :3].astype(np.float32)
    K2_rect: Float[ndarray, "3 3"] = P2[:, :3].astype(np.float32)
    left_intri = Intrinsics(
        camera_conventions="RDF",
        fl_x=float(K1_rect[0, 0]),
        fl_y=float(K1_rect[1, 1]),
        cx=float(K1_rect[0, 2]),
        cy=float(K1_rect[1, 2]),
        width=new_size[0],
        height=new_size[1],
    )
    right_intri = Intrinsics(
        camera_conventions="RDF",
        fl_x=float(K2_rect[0, 0]),
        fl_y=float(K2_rect[1, 1]),
        cx=float(K2_rect[0, 2]),
        cy=float(K2_rect[1, 2]),
        width=new_size[0],
        height=new_size[1],
    )

    # Midpoint transforms (cam -> mid)
    mid_ex_left = left_stream.get_extrinsics_to(pose_stream)
    mid_R_left: Float[ndarray, "3 3"] = np.reshape(np.array(mid_ex_left.rotation, dtype=np.float64), (3, 3))
    mid_t_left: Float[ndarray, "3"] = np.array(mid_ex_left.translation, dtype=np.float64)
    mid_T_left: Float[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    mid_T_left[:3, :3] = mid_R_left
    mid_T_left[:3, 3] = mid_t_left

    mid_ex_right = right_stream.get_extrinsics_to(pose_stream)
    mid_R_right: Float[ndarray, "3 3"] = np.reshape(np.array(mid_ex_right.rotation, dtype=np.float64), (3, 3))
    mid_t_right: Float[ndarray, "3"] = np.array(mid_ex_right.translation, dtype=np.float64)
    mid_T_right: Float[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    mid_T_right[:3, :3] = mid_R_right
    mid_T_right[:3, 3] = mid_t_right

    return RectificationSetup(
        left_rect_map1=left_rect_map1,
        left_rect_map2=left_rect_map2,
        right_rect_map1=right_rect_map1,
        right_rect_map2=right_rect_map2,
        width_px=rect_width_px,
        height_px=rect_height_px,
        left_intrinsics=left_intri,
        right_intrinsics=right_intri,
        R1=R1,
        R2=R2,
        mid_T_left=mid_T_left,
        mid_T_right=mid_T_right,
    )


@dataclass
class T265Config:
    """Log T265 left/right fisheye images to Rerun."""

    rr_config: RerunTyroConfig
    # Note: runs until Ctrl+C; 'frames' is deprecated and ignored.

    serial: str | None = None
    """Optional device serial if multiple T265 are connected."""

    timeout_ms: int = 1000
    """Wait timeout (ms) for each frame."""

    jpeg_quality: int = 75
    """JPEG quality for image compression (lower = smaller & faster). Used only for debug image logging."""

    base_path: Path = Path("/world/ego")
    """Base entity path for logging (e.g., '/world/ego')."""

    rect_fov_deg: float = 110.0
    """Horizontal FOV (degrees) for rectified pinhole images. Typical: 80–110."""

    log_fisheye: bool = False
    """If True, also log raw fisheye images under '<base>/left|right/fisheye/image'."""

    target_fps: int = 30
    """Target FPS for encoder time base and PTS. T265 fisheye is typically 30 FPS."""

    run_seconds: float | None = None
    """Optional total run time in seconds. None runs until Ctrl+C."""

    rrd_save_dir: Path | None = Path("data/rrd-save-files")
    """Directory to save a .rrd recording (MultiSink). If None, don't save.

    The saved file name will be formatted as:
      {YYYYmmdd_HHMMSS}_t265_slam_rrd_{rerun_version}.rrd
    """


class _VideoStreamEncoder:
    """Threaded H.264 encoder + Rerun logger for a single grayscale stream.

    Non-blocking: frames are enqueued from the acquisition thread and encoded/logged
    on a separate worker to avoid stalling RealSense frame delivery.
    """

    def __init__(
        self,
        entity_path: str,
        timeline: str,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self.entity_path: str = entity_path
        self.timeline: str = timeline

        self.width: int = int(width)
        self.height: int = int(height)
        self.fps: int = int(fps)

        self.q: "queue.Queue[tuple[np.ndarray, int]]" = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self.output_stream = _setup_output_stream(width, height, fps=self.fps)

        # Log stream metadata once
        rr.log(self.entity_path, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)

        # Device-time bookkeeping for timeline normalization
        self._t0_ns: int | None = None  # First device timestamp (ns) used as zero-point for the timeline
        self._last_dt_ns: int = 0  # Cached elapsed (ns) since _t0_ns; reused during encoder flush

        self._thr = threading.Thread(target=self._run, name=f"Encoder[{entity_path}]", daemon=True)
        self._thr.start()

    def enqueue(self, frame_gray: UInt8[ndarray, "h w"], ts_ns: int) -> None:
        # Drop oldest if full to keep acquisition non-blocking
        if self.q.full():
            with contextlib.suppress(Exception):
                self.q.get_nowait()
        self.q.put((frame_gray, int(ts_ns)))

    def stop(self) -> None:
        self._stop.set()
        self._thr.join(timeout=2.0)
        # Flush remaining packets
        with contextlib.suppress(Exception):
            for packet in self.output_stream.encode(None):
                if packet.pts is None:
                    continue
                rr.set_time(self.timeline, duration=np.timedelta64(self._last_dt_ns, "ns"))
                rr.log(self.entity_path, rr.VideoStream.from_fields(sample=bytes(packet)))

    def _elapsed_since_start_ns(self, ts_ns: int) -> int:
        """Return elapsed nanoseconds since first device timestamp.

        Establishes a reference timestamp on first call and updates
        `_last_dt_ns` for reuse during flush.
        """
        if self._t0_ns is None:
            self._t0_ns = ts_ns
        elapsed_ns: int = int(ts_ns - self._t0_ns)
        self._last_dt_ns = elapsed_ns
        return elapsed_ns

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                queue_tuple: tuple[np.ndarray, int] = self.q.get(timeout=0.1)
                frame_np: UInt8[ndarray, "h w"] = queue_tuple[0]
                ts_ns: int = int(queue_tuple[1])
            except queue.Empty:
                continue
            # Construct AVFrame from grayscale and convert to encoder pix_fmt
            frame: av.VideoFrame = av.VideoFrame.from_ndarray(frame_np, format="gray8")

            # Normalize device timestamp to elapsed (ns) since stream start
            elapsed_ns = self._elapsed_since_start_ns(int(ts_ns))

            # Encode and stream to Rerun (packet-by-packet), timestamped by device clock
            for packet in self.output_stream.encode(frame):
                if packet.pts is None:
                    continue
                rr.set_time(self.timeline, duration=np.timedelta64(elapsed_ns, "ns"))
                rr.log(self.entity_path, rr.VideoStream.from_fields(sample=bytes(packet)))


def main(config: T265Config) -> int:
    """Stream T265 fisheye images to Rerun (left/right) as compressed images."""
    # Minimal setup: assume device is connected; no extra validation
    # Rerun is initialized via RerunTyroConfig.__post_init__
    # Enable MultiSink file saving if requested.
    if config.rrd_save_dir is not None:
        config.rrd_save_dir.mkdir(parents=True, exist_ok=True)
        rr_ver = getattr(rr, "__version__", "unknown")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rrd_file = config.rrd_save_dir / f"{ts}_t265_slam_rrd_{rr_ver}.rrd"
        # Set MultiSink sinks explicitly (file sink; add grpc sink if desired)
        sinks: list[rr.Sink] = [rr.GrpcSink(), rr.FileSink(str(rrd_file))]
        # If you want to also stream to a local viewer, uncomment this:
        # sinks.append(rr.GrpcSink())
        rr.set_sinks(*sinks)
        print(f"Saving Rerun recording to: {rrd_file}")

    # Set world coordinates and send a 3D+2D blueprint
    rr.log("/", rr.ViewCoordinates.RUB, static=True)
    timeline = "video_time"

    left_pinhole_video_path: Path = config.base_path / "left" / "pinhole" / "video"
    right_pinhole_video_path: Path = config.base_path / "right" / "pinhole" / "video"
    left_fisheye_image_path: Path = config.base_path / "left" / "fisheye" / "image"
    right_fisheye_image_path: Path = config.base_path / "right" / "fisheye" / "image"

    right_panel_contents = [
        rrb.Horizontal(
            rrb.Spatial2DView(origin=str(left_pinhole_video_path)),
            rrb.Spatial2DView(origin=str(right_pinhole_video_path)),
        )
    ]
    if config.log_fisheye:
        right_panel_contents.append(
            rrb.Horizontal(
                rrb.Spatial2DView(origin=str(left_fisheye_image_path)),
                rrb.Spatial2DView(origin=str(right_fisheye_image_path)),
            )
        )

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="/"),
                rrb.Vertical(*right_panel_contents),
                column_shares=[3, 2],
            ),
            collapse_panels=True,
        )
    )

    left_path = config.base_path / "left"
    right_path = config.base_path / "right"
    # Query active profile and prepare rectification from fisheye to pinhole
    pipeline = _setup_t265_input(config.serial)
    rect: RectificationSetup = _init_rectification(pipeline, rect_fov_deg=float(config.rect_fov_deg))

    # threaded H.264 video stream encoders for rectified images
    left_vs = right_vs = None
    left_vs = _VideoStreamEncoder(
        entity_path=str(left_pinhole_video_path),
        timeline=timeline,
        width=rect.width_px,
        height=rect.height_px,
        fps=config.target_fps,
    )
    right_vs = _VideoStreamEncoder(
        entity_path=str(right_pinhole_video_path),
        timeline=timeline,
        width=rect.width_px,
        height=rect.height_px,
        fps=config.target_fps,
    )

    try:
        start_ts: float = time.time()
        last_left_num: int = -1
        last_right_num: int = -1
        left_base_ts_ns: int | None = None

        while True:
            if config.run_seconds is not None and (time.time() - start_ts) >= float(config.run_seconds):
                break
            frames = pipeline.wait_for_frames(config.timeout_ms)

            # Retrieve left/right fisheye frames

            left = frames.get_fisheye_frame(1)
            right = frames.get_fisheye_frame(2)

            # Convert to numpy views (zero-copy where possible)
            left_np: UInt8[ndarray, "h=800 w=848"] = np.asanyarray(left.get_data())  # type: ignore[assignment]
            right_np: UInt8[ndarray, "h=800 w=848"] = np.asanyarray(right.get_data())  # type: ignore[assignment]

            # Rectify to pinhole
            left_rect: UInt8[ndarray, "new_h=800 new_w=800"] = cv2.remap(
                left_np, rect.left_rect_map1, rect.left_rect_map2, interpolation=cv2.INTER_LINEAR
            )
            right_rect: UInt8[ndarray, "new_h=800 new_w=800"] = cv2.remap(
                right_np, rect.right_rect_map1, rect.right_rect_map2, interpolation=cv2.INTER_LINEAR
            )

            # Use device timestamp (ms → ns) per stream; encoders normalize to their own start
            left_num = int(left.get_frame_number())
            right_num = int(right.get_frame_number())
            left_ts_ns = int(round(float(left.get_timestamp()) * 1_000_000.0))
            right_ts_ns = int(round(float(right.get_timestamp()) * 1_000_000.0))

            # Deduplicate per stream: framesets may arrive ~60 Hz alternating L/R
            if left_vs and left_num != last_left_num:
                left_vs.enqueue(left_rect, ts_ns=left_ts_ns)
                last_left_num = left_num
            if right_vs and right_num != last_right_num:
                right_vs.enqueue(right_rect, ts_ns=right_ts_ns)
                last_right_num = right_num

            # Also log original fisheye images for comparison (optional)
            if config.log_fisheye:
                rr.log(str(left_fisheye_image_path), rr.Image(left_np).compress(jpeg_quality=config.jpeg_quality))
                rr.log(
                    str(right_fisheye_image_path),
                    rr.Image(right_np).compress(jpeg_quality=config.jpeg_quality),
                )

            # Log pose transform (midpoint between fisheyes) in 3D
            pose = frames.get_pose_frame()
            if pose:
                data = pose.get_pose_data()
                t: Float[ndarray, "3"] = np.array(
                    [data.translation.x, data.translation.y, data.translation.z], dtype=np.float32
                )
                R_mid_world: Float[ndarray, "3 3"] = _quat_to_rot_m33(
                    data.rotation.x, data.rotation.y, data.rotation.z, data.rotation.w
                )
                # Align pose/camera logs with left stream device timeline (ns)
                if left_base_ts_ns is None:
                    left_base_ts_ns = left_ts_ns
                rr.set_time(timeline, duration=np.timedelta64(int(left_ts_ns - left_base_ts_ns), "ns"))
                # World_T_mid
                world_T_mid: Float[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
                world_T_mid[:3, :3] = R_mid_world
                world_T_mid[:3, 3] = t

                # Compose to eye frames, then invert to get cam_T_world
                world_T_left: Float[ndarray, "4 4"] = (world_T_mid @ rect.mid_T_left).astype(np.float32)
                world_T_right: Float[ndarray, "4 4"] = (world_T_mid @ rect.mid_T_right).astype(np.float32)

                left_cam_T_world: Float[ndarray, "4 4"] = np.linalg.inv(world_T_left).astype(np.float32)
                right_cam_T_world: Float[ndarray, "4 4"] = np.linalg.inv(world_T_right).astype(np.float32)

                # Apply rectification rotation to camera orientation (virtual pinholes)
                # New camera coords = R_rect * old_cam; so R_cam_world_new = R_cam_world_old @ R_rect^T
                left_cam_R_world: Float[ndarray, "3 3"] = (
                    left_cam_T_world[:3, :3] @ rect.R1.astype(np.float32).T
                ).astype(np.float32)
                right_cam_R_world: Float[ndarray, "3 3"] = (
                    right_cam_T_world[:3, :3] @ rect.R2.astype(np.float32).T
                ).astype(np.float32)
                left_cam_t_world: Float[ndarray, "3"] = left_cam_T_world[:3, 3]
                right_cam_t_world: Float[ndarray, "3"] = right_cam_T_world[:3, 3]

                # Log per-eye pinhole cameras via helper
                left_extri = Extrinsics(cam_R_world=left_cam_R_world, cam_t_world=left_cam_t_world)
                right_extri = Extrinsics(cam_R_world=right_cam_R_world, cam_t_world=right_cam_t_world)
                left_params = PinholeParameters(name="left", intrinsics=rect.left_intrinsics, extrinsics=left_extri)
                right_params = PinholeParameters(name="right", intrinsics=rect.right_intrinsics, extrinsics=right_extri)

                log_pinhole(camera=left_params, cam_log_path=left_path, image_plane_distance=0.01)
                log_pinhole(camera=right_params, cam_log_path=right_path, image_plane_distance=0.01)
    except KeyboardInterrupt:
        print("Interrupted by user. Stopping T265 stream…")
    except Exception as e:  # noqa: BLE001
        print(f"Error while streaming fisheye images: {e}")
        with contextlib.suppress(Exception):
            pipeline.stop()
        return 3
    finally:
        with contextlib.suppress(Exception):
            pipeline.stop()
        # Stop encoders
        with contextlib.suppress(Exception):
            if left_vs:
                left_vs.stop()
        with contextlib.suppress(Exception):
            if right_vs:
                right_vs.stop()

    return 0
