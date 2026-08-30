"""Stream stereo depth + an incremental TSDF mesh for a catalog rig segment into the Rerun viewer.

Reads one exoego:v2 segment straight from the catalog (video packets fetched once per camera, decoded on NVDEC),
rectifies the fisheye front pair, runs the selected stereo predictor, and logs everything to the ambient recording configured by
``RerunTyroConfig`` — the PromptDA Polycam pattern (viewer/save/connect, per-frame logging, mesh re-logged every
``mesh_every`` frames so scrubbing shows it grow), on the catalog's own ``video_time`` timeline so the relayed raw
videos and the slam poses line up. Nothing is registered back to the catalog.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

import cv2
import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from jaxtyping import Float32, Float64, UInt8, UInt16
from rerun.catalog import CatalogClient, DatasetEntry, DatasetView
from scipy.spatial.transform import Rotation
from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion, PinholeParameters, rescale_intri
from simplecv.catalog_video_codec import CatalogCodecName, catalog_codec_name
from simplecv.ops.tsdf_depth_fuser import Open3DFuser
from simplecv.rerun_dataloader import open_segment_decoder
from simplecv.rerun_log_utils import RerunTyroConfig, log_open3d_mesh
from simplecv.rerun_rig_logger import log_rig_static
from simplecv.rig import CameraSensor, Rig, RigCalibration
from simplecv.rrd_query_utils import first_valid_value

from monopriors.depth_utils import depth_edges_mask
from monopriors.models.stereo_depth import STEREO_PREDICTORS, BaseStereoPredictor, StereoDepthPrediction, get_stereo_predictor
from monopriors.models.stereo_depth.liteanystereo import LAS2ModelSize
from monopriors.models.stereo_depth.rectify import StereoRectification, fisheye_stereo_rectify, rectify_pair

TIMELINE: str = "video_time"
RIG: str = "world/rig_00"


class VideoFrame(Protocol):
    """Decoded RGB frame returned by the catalog video decoder."""

    data: UInt8[torch.Tensor, "3 h w"]


class SegmentVideoDecoder(Protocol):
    """Subset of the TorchCodec decoder used by this catalog tool."""

    def get_frame_at(self, index: int) -> VideoFrame:
        """Decode one frame by sample index."""
        ...


@dataclass
class StereoCatalogConfig:
    """Stereo depth over one catalog rig segment, streamed to Rerun."""

    rr_config: RerunTyroConfig
    """Rerun viewer / save / connect behaviour."""
    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Catalog server."""
    dataset_name: str = "robocap"
    """Dataset holding the exoego:v2 rig recordings (needs a temporal ``world_T_rig`` on the rig node, e.g. a slam layer)."""
    segment_id: str = "robocap__f408193e6447b3b0__s00000021"
    """Segment (recording id) to process."""
    left_cam: str = "cam_00"
    """Left camera of the rectified pair (robocap: ``left_front``)."""
    right_cam: str = "cam_01"
    """Right camera of the rectified pair (robocap: ``right_front``)."""
    fps: float = 5.0
    """Frames per second sampled from the video timeline."""
    max_seconds: float | None = 120.0
    """Only the first N seconds of the segment; None processes it all."""
    predictor_name: STEREO_PREDICTORS = "LiteAnyStereoPredictor"
    """Which stereo predictor to use."""
    model_size: LAS2ModelSize = "m"
    """LiteAnyStereo V2 variant; ignored by Fast-FoundationStereo."""
    focal_scale: float = 0.8
    """Rectified focal length as a multiple of the fisheye fx; below 1 keeps more of the wide FOV."""
    max_depth_m: float = 20.0
    """Depth beyond this is dropped."""
    colormap_max_m: float = 6.0
    """Upper end of the depth image's colour range (a viewer hint; indoor scenes are unreadable on a 0-20 m ramp)."""
    remove_flying_pixels: bool = True
    """Zero depth on depth edges so the backprojected cloud has no streaks."""
    depth_edge_threshold: float = 0.5
    """Depth-gradient magnitude (metres per pixel) that counts as an edge."""
    output_scale: float = 0.5
    """Resolution of the logged rectified image + depth relative to the video (inference stays at full res)."""
    fuse: bool = True
    """TSDF-fuse the depth along ``world_T_rig`` and log the mesh at ``world/stereo/mesh``."""
    fusion_voxel_m: float = 0.04
    """TSDF voxel size in metres."""
    fusion_max_depth_m: float = 4.0
    """Depth beyond this is not fused (stereo error grows with distance squared)."""
    mesh_every: int = 25
    """Re-log the growing mesh every N processed frames (0 = only at the end)."""
    device: Literal["cuda", "cpu"] = "cuda"
    """Where the network runs (decode is always NVDEC)."""


def read_static(view: DatasetView, entity: str, component: str) -> object:
    """First value of a static component on one entity."""
    table: pa.Table = view.filter_contents(entity).reader(index=None).select(f"/{entity}:{component}").to_arrow_table()
    return first_valid_value(table.column(0), component_name=component)


def read_fisheye_camera(view: DatasetView, cam: str) -> Fisheye62Parameters:
    """One rig camera as written by dataforge (exoego:v2 + simplecv distortion components); extrinsics are ``cam_T_rig``."""
    pinhole: str = f"{RIG}/{cam}/pinhole"
    model: str = str(np.asarray(read_static(view, pinhole, "simplecv.components.DistortionModel")).ravel()[0])
    if model != "kannala_brandt":
        raise ValueError(f"{pinhole}: expected kannala_brandt distortion, got {model!r}")
    K_33: Float64[np.ndarray, "3 3"] = np.asarray(read_static(view, pinhole, "Pinhole:image_from_camera"), dtype=np.float64).reshape(3, 3, order="F")
    resolution = np.asarray(read_static(view, pinhole, "Pinhole:resolution"), dtype=np.float64)
    k = np.asarray(read_static(view, pinhole, "simplecv.components.DistortionCoefficients"), dtype=np.float64)
    return Fisheye62Parameters(
        name=str(np.asarray(read_static(view, f"{RIG}/{cam}", "name")).ravel()[0]),
        extrinsics=Extrinsics(
            cam_R_world=np.asarray(read_static(view, f"{RIG}/{cam}", "Transform3D:mat3x3"), dtype=np.float64).reshape(3, 3, order="F"),
            cam_t_world=np.asarray(read_static(view, f"{RIG}/{cam}", "Transform3D:translation"), dtype=np.float64),
        ),
        intrinsics=Intrinsics.from_k_matrix(camera_conventions="RDF", k_matrix=K_33, height=int(resolution[1]), width=int(resolution[0])),
        distortion=KannalaBrandtDistortion(k1=float(k[0]), k2=float(k[1]), k3=float(k[2]), k4=float(k[3]), k5=float(k[4]), k6=float(k[5]), p1=float(k[6]), p2=float(k[7])),
    )


def read_rig_poses(view: DatasetView) -> tuple[np.ndarray, Float64[np.ndarray, "n 4 4"]]:
    """Temporal ``world_T_rig`` from the rig node (the slam layer): sample times (timedelta64[ns]) and 4x4 poses."""
    table: pa.Table = (
        view.filter_contents(RIG).reader(index=TIMELINE).select(TIMELINE, f"/{RIG}:Transform3D:translation", f"/{RIG}:Transform3D:quaternion").sort(TIMELINE).to_arrow_table()
    )
    if table.num_rows == 0:
        raise ValueError(f"{RIG} carries no temporal Transform3D — is the slam layer registered?")
    times: np.ndarray = np.array([t.value for t in table.column(0)], dtype="timedelta64[ns]")
    translations_n3: Float64[np.ndarray, "n 3"] = np.array([v[0] for v in table.column(1).to_pylist()], dtype=np.float64)
    quaternions_n4: Float64[np.ndarray, "n 4"] = np.array([v[0] for v in table.column(2).to_pylist()], dtype=np.float64)  # xyzw, as Rerun stores it
    poses_n44: Float64[np.ndarray, "n 4 4"] = np.tile(np.eye(4), (len(times), 1, 1))
    poses_n44[:, :3, :3] = Rotation.from_quat(quaternions_n4).as_matrix()
    poses_n44[:, :3, 3] = translations_n3
    return times, poses_n44


def scaled_pinhole(camera: PinholeParameters, scale: float) -> PinholeParameters:
    """The same camera for an image resized by ``scale``."""
    intrinsics: Intrinsics = rescale_intri(camera.intrinsics, target_width=round(camera.intrinsics.width * scale), target_height=round(camera.intrinsics.height * scale))
    return PinholeParameters(name=camera.name, extrinsics=camera.extrinsics, intrinsics=intrinsics)


LEFT_RECT_INDEX: int = 10
"""Rig sensor index of the rectified left camera (``cam_10``): above the six physical robocap cameras."""
RIGHT_RECT_INDEX: int = 11


def build_blueprint(left_cam: str, right_cam: str) -> rrb.Blueprint:
    rect_pinhole: str = f"{RIG}/cam_{LEFT_RECT_INDEX}/pinhole"
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="rig + stereo cloud + mesh", contents=["$origin/**"]),
            rrb.Vertical(
                rrb.Spatial2DView(origin=f"{rect_pinhole}/image", name="rectified left"),
                rrb.Spatial2DView(origin=f"{rect_pinhole}/depth", name="stereo depth (m)"),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin=f"{RIG}/{left_cam}/pinhole/video", name="left_front (fisheye)"),
                    rrb.Spatial2DView(origin=f"{RIG}/{right_cam}/pinhole/video", name="right_front (fisheye)"),
                ),
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(timeline=TIMELINE),
        collapse_panels=True,
    )


def main(config: StereoCatalogConfig) -> None:
    dataset: DatasetEntry = CatalogClient(config.catalog_url).get_dataset(config.dataset_name)
    view: DatasetView = dataset.filter_segments(config.segment_id)
    left: Fisheye62Parameters = read_fisheye_camera(view, config.left_cam)
    right: Fisheye62Parameters = read_fisheye_camera(view, config.right_cam)
    rect: StereoRectification = fisheye_stereo_rectify(left, right, focal_scale=config.focal_scale)
    left_rect_out: PinholeParameters = scaled_pinhole(rect.left_rect, config.output_scale)
    pose_times, world_T_rig_n44 = read_rig_poses(view)
    print(f"rectified pair: baseline {rect.baseline_m * 1000:.1f} mm, fx_rect {rect.left_rect.intrinsics.fl_x:.1f} px; {len(pose_times)} rig poses")

    device: torch.device = torch.device("cuda")
    codec_value: int = int(np.asarray(read_static(view, f"{RIG}/{config.left_cam}/pinhole/video", "VideoStream:codec")).ravel()[0])
    video_codec: rr.VideoCodec = rr.VideoCodec(codec_value)
    codec: CatalogCodecName = catalog_codec_name(codec_value)
    decoders: dict[str, tuple[np.ndarray, list[bytes], list[bool], SegmentVideoDecoder]] = {
        cam: open_segment_decoder(dataset, config.segment_id, f"{RIG}/{cam}/pinhole/video", TIMELINE, device, 30) for cam in (config.left_cam, config.right_cam)
    }
    left_times, right_times = decoders[config.left_cam][0], decoders[config.right_cam][0]
    t_start_ns: int = int(max(left_times[0], right_times[0]).astype("int64"))
    t_end_ns: int = int(min(left_times[-1], right_times[-1]).astype("int64"))
    if config.max_seconds is not None:
        t_end_ns = min(t_end_ns, t_start_ns + int(config.max_seconds * 1e9))
    grid_ns: np.ndarray = np.arange(t_start_ns, t_end_ns, int(1e9 / config.fps), dtype=np.int64)
    print(f"{codec} packets: left {len(left_times)}, right {len(right_times)}; sampling {len(grid_ns)} frames at {config.fps} fps over {(t_end_ns - t_start_ns) / 1e9:.1f} s")

    # ── static scene + blueprint, then the raw videos relayed as-is (no re-encode) ──
    rr.send_blueprint(build_blueprint(config.left_cam, config.right_cam))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    # The two fisheye cameras plus the two rectified virtual cameras, all as rig sensors (rig_T_cam extrinsics).
    rig: Rig = Rig(
        index=0,
        calibration=RigCalibration(
            cameras=[
                CameraSensor(index=int(config.left_cam.split("_")[1]), name=left.name, kind="grayscale", pinhole=left),
                CameraSensor(index=int(config.right_cam.split("_")[1]), name=right.name, kind="grayscale", pinhole=right),
                CameraSensor(index=LEFT_RECT_INDEX, name=rect.left_rect.name, kind="grayscale", pinhole=left_rect_out),
                CameraSensor(index=RIGHT_RECT_INDEX, name=rect.right_rect.name, kind="grayscale", pinhole=scaled_pinhole(rect.right_rect, config.output_scale)),
            ],
            reference_index=int(config.left_cam.split("_")[1]),
        ),
        image_plane_distance=0.1,
    )
    log_rig_static(rig)
    for cam, (times, samples, keyframes, _) in decoders.items():
        rr.log(f"{RIG}/{cam}/pinhole/video", rr.VideoStream(codec=video_codec), static=True)
        rr.send_columns(f"{RIG}/{cam}/pinhole/video", indexes=[rr.TimeColumn(TIMELINE, duration=times)], columns=rr.VideoStream.columns(sample=samples, is_keyframe=keyframes))

    match config.predictor_name:
        case "LiteAnyStereoPredictor":
            predictor: BaseStereoPredictor = get_stereo_predictor(config.predictor_name)(device=config.device, model_size=config.model_size)
        case "FastFoundationStereoPredictor":
            predictor = get_stereo_predictor(config.predictor_name)(device=config.device)
    rect_pinhole: str = f"{RIG}/cam_{LEFT_RECT_INDEX}/pinhole"
    output_wh: tuple[int, int] = (left_rect_out.intrinsics.width, left_rect_out.intrinsics.height)
    K_out_33: Float64[np.ndarray, "3 3"] = np.asarray(left_rect_out.intrinsics.k_matrix, dtype=np.float64)
    K_rect_32: Float32[np.ndarray, "3 3"] = np.asarray(rect.left_rect.intrinsics.k_matrix, dtype=np.float32)
    cam_rect_T_rig: Float64[np.ndarray, "4 4"] = rect.left_rect.extrinsics.cam_T_world
    colormap_max_mm: float = config.colormap_max_m * 1000.0
    fuser: Open3DFuser | None = Open3DFuser(fusion_resolution=config.fusion_voxel_m, max_fusion_depth=config.fusion_max_depth_m) if config.fuse else None

    def frame_at(cam: str, t_ns: int) -> UInt8[np.ndarray, "h w 3"]:
        times, _, _, decoder = decoders[cam]
        index: int = int(np.searchsorted(times, np.timedelta64(t_ns, "ns"), side="right")) - 1
        return rearrange(decoder.get_frame_at(index).data, "c h w -> h w c").cpu().numpy()

    for i, t_ns in enumerate(grid_ns.tolist()):
        world_T_rig: Float64[np.ndarray, "4 4"] = world_T_rig_n44[min(int(np.searchsorted(pose_times, np.timedelta64(t_ns, "ns"))), len(pose_times) - 1)]
        left_rect, right_rect = rectify_pair(rect, frame_at(config.left_cam, t_ns), frame_at(config.right_cam, t_ns))
        pred: StereoDepthPrediction = predictor(left_rect, right_rect, K_33=K_rect_32, baseline_m=rect.baseline_m)
        assert pred.depth_meters is not None
        depth_hw: Float32[np.ndarray, "h w"] = np.where(pred.depth_meters > config.max_depth_m, 0.0, pred.depth_meters).astype(np.float32)
        if config.remove_flying_pixels:
            depth_hw = np.asarray(depth_hw * ~depth_edges_mask(depth_hw, threshold=config.depth_edge_threshold), dtype=np.float32)
        depth_mm_hw: UInt16[np.ndarray, "h w"] = np.clip(depth_hw * 1000.0, 0.0, 65535.0).astype(np.uint16)
        if config.output_scale != 1.0:
            depth_mm_hw = cv2.resize(depth_mm_hw, output_wh, interpolation=cv2.INTER_NEAREST)
            left_rect = cv2.resize(left_rect, output_wh, interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", depth_mm_hw, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        assert ok

        rr.set_time(TIMELINE, duration=np.timedelta64(t_ns, "ns"))
        rr.log(RIG, rr.Transform3D(mat3x3=world_T_rig[:3, :3], translation=world_T_rig[:3, 3]))
        rr.log(f"{rect_pinhole}/image", rr.Image(left_rect).compress(jpeg_quality=85))
        rr.log(f"{rect_pinhole}/depth", rr.EncodedDepthImage(blob=png.tobytes(), media_type="image/png", meter=1000.0, depth_range=(0.0, colormap_max_mm)))
        if fuser is not None:
            fuser.fuse_frames(np.ascontiguousarray(depth_mm_hw), K_out_33, cam_rect_T_rig @ np.linalg.inv(world_T_rig), np.ascontiguousarray(left_rect))
            if config.mesh_every and (i + 1) % config.mesh_every == 0:
                log_open3d_mesh("world/stereo/mesh", fuser.get_mesh())
        if i % 50 == 0:
            print(f"  {i}/{len(grid_ns)}  t={(t_ns - t_start_ns) / 1e9:6.1f}s  disparity {pred.disparity.max():.1f} px")
    if fuser is not None:
        log_open3d_mesh("world/stereo/mesh", fuser.get_mesh())
        print(f"fused mesh logged: {len(fuser.get_mesh().vertices)} vertices")
