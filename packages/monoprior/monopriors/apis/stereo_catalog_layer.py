"""Stereo depth layer for a catalog rig segment: rectify the front fisheye pair, run LiteAnyStereo, register the result.

Follows the PromptDA ARKitScenes register tool: read one segment straight from the catalog (video packets
fetched once per camera, decoded on NVDEC), write one RRD whose ``recording_id`` equals the segment id so it
stacks as a layer, register it with ``OnDuplicateSegmentLayer.REPLACE``, and add a blueprint that shows it.

The rectified cameras are logged as two extra rig nodes ``cam_MM/rectified`` (exoego:v2 shape: ``Transform3D``
from the rig + a plain ``Pinhole``), so the original fisheye rig stays untouched and the viewer backprojects
``cam_00/rectified/pinhole/depth`` through the rectified intrinsics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
import torch
from einops import rearrange
from jaxtyping import Float32, Float64, UInt8, UInt16
from rerun.catalog import CatalogClient, DatasetEntry, DatasetView, OnDuplicateSegmentLayer
from simplecv.rerun_dataloader import open_segment_decoder
from torchcodec.decoders import VideoDecoder

from monopriors.depth_utils import depth_edges_mask
from monopriors.models.stereo_depth import LiteAnyStereoPredictor, StereoDepthPrediction
from monopriors.models.stereo_depth.liteanystereo import LAS2ModelSize
from monopriors.models.stereo_depth.rectify import FisheyeCamera, StereoRectification, fisheye_stereo_rectify, rectified_rig_extrinsics, rectify_pair

TIMELINE: str = "video_time"
RIG: str = "world/rig_00"


@dataclass
class StereoCatalogLayerConfig:
    """Stereo depth layer over one catalog rig segment."""

    catalog_url: str = "rerun+http://127.0.0.1:51235"
    """Catalog server."""
    dataset_name: str = "robocap"
    """Dataset holding the exoego:v2 rig recordings."""
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
    model_size: LAS2ModelSize = "m"
    """LiteAnyStereo V2 variant."""
    focal_scale: float = 0.8
    """Rectified focal length as a multiple of the fisheye fx; below 1 keeps more of the wide FOV."""
    max_depth_m: float = 20.0
    """Depth beyond this is dropped."""
    remove_flying_pixels: bool = True
    """Zero depth on depth edges so the backprojected cloud has no streaks."""
    depth_edge_threshold: float = 0.5
    """Depth-gradient magnitude (metres per pixel) that counts as an edge."""
    output_scale: float = 0.5
    """Resolution of the logged rectified image + depth relative to the video (inference stays at full res); 16-bit
    depth PNGs are 1.5 MB/frame at 1080p, 0.27 MB at half."""
    output_dir: Path = Path("data/stereo_layer")
    """Where the layer RRD and blueprint are written."""
    layer_name: str = "stereo_las2"
    """Catalog layer name."""
    register: bool = True
    """Register the RRD and the blueprint on the dataset."""
    default_blueprint: bool = False
    """Make the stereo blueprint the dataset default (otherwise it is registered alongside the existing default)."""
    device: Literal["cuda", "cpu"] = "cuda"
    """Where the network runs (decode is always NVDEC)."""


def read_static(view: DatasetView, entity: str, component: str) -> np.ndarray | str | int:
    """First value of a static component on one entity (the viewer's static columns are index-less)."""
    column: str = f"/{entity}:{component}"
    table: pa.Table = view.filter_contents(entity).reader(index=None).select(column).to_arrow_table()
    value = table.column(0)[0].as_py()
    if isinstance(value, list):
        value = value[0] if len(value) == 1 and not isinstance(value[0], (int, float)) else value
    return np.asarray(value) if isinstance(value, list) else value


def read_fisheye_camera(view: DatasetView, cam: str) -> tuple[FisheyeCamera, Float64[np.ndarray, "4 4"]]:
    """Fisheye intrinsics and ``cam_T_rig`` for one rig camera written by dataforge (exoego:v2 + simplecv distortion)."""
    pinhole: str = f"{RIG}/{cam}/pinhole"
    model = read_static(view, pinhole, "simplecv.components.DistortionModel")
    if model != "kannala_brandt":
        raise ValueError(f"{pinhole}: expected kannala_brandt distortion, got {model!r}")
    K_33: Float64[np.ndarray, "3 3"] = np.asarray(read_static(view, pinhole, "Pinhole:image_from_camera"), dtype=np.float64).reshape(3, 3, order="F")
    resolution = np.asarray(read_static(view, pinhole, "Pinhole:resolution"), dtype=np.float64)
    coefficients = np.asarray(read_static(view, pinhole, "simplecv.components.DistortionCoefficients"), dtype=np.float64)
    camera: FisheyeCamera = FisheyeCamera(K_33=K_33, dist_4=coefficients[:4], width=int(resolution[0]), height=int(resolution[1]))
    cam_R_rig: Float64[np.ndarray, "3 3"] = np.asarray(read_static(view, f"{RIG}/{cam}", "Transform3D:mat3x3"), dtype=np.float64).reshape(3, 3, order="F")
    cam_t_rig: Float64[np.ndarray, "3"] = np.asarray(read_static(view, f"{RIG}/{cam}", "Transform3D:translation"), dtype=np.float64)
    cam_T_rig: Float64[np.ndarray, "4 4"] = np.eye(4)
    cam_T_rig[:3, :3] = cam_R_rig
    cam_T_rig[:3, 3] = cam_t_rig
    return camera, cam_T_rig


def log_rectified_camera(
    rec: rr.RecordingStream, cam: str, name: str, cam_T_rig: Float64[np.ndarray, "4 4"], R_rect: Float64[np.ndarray, "3 3"], rect: StereoRectification, scale: float
) -> None:
    """Static rig node ``cam_MM/rectified`` = the rectified virtual pinhole camera, at ``scale`` × the video resolution."""
    cam_rect_R_rig, cam_rect_t_rig = rectified_rig_extrinsics(cam_T_rig[:3, :3], cam_T_rig[:3, 3], R_rect)
    path: str = f"{RIG}/{cam}/rectified"
    rr.log(path, rr.Transform3D(mat3x3=cam_rect_R_rig, translation=cam_rect_t_rig, from_parent=True), static=True, recording=rec)
    rr.log(path, rr.AnyValues(name=name, kind="grayscale"), static=True, recording=rec)
    rr.log(
        f"{path}/pinhole",
        rr.Pinhole(
            image_from_camera=scale_intrinsics(rect.K_rect_33, scale),
            width=round(rect.width * scale),
            height=round(rect.height * scale),
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=0.1,
        ),
        static=True,
        recording=rec,
    )


def scale_intrinsics(K_33: Float64[np.ndarray, "3 3"], scale: float) -> Float64[np.ndarray, "3 3"]:
    """Pinhole intrinsics for an image resized by ``scale`` (fx, fy, cx, cy scale; the last row does not)."""
    K_scaled_33: Float64[np.ndarray, "3 3"] = K_33.copy()
    K_scaled_33[:2] *= scale
    return K_scaled_33


def build_blueprint(left_cam: str, right_cam: str) -> rrb.Blueprint:
    """3D rig beside the rectified image, its depth, and the two raw fisheye videos."""
    rect_pinhole: str = f"{RIG}/{left_cam}/rectified/pinhole"
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="rig", contents=["$origin/**", f"- {RIG}/cam_*/pinhole/video"]),
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


def main(config: StereoCatalogLayerConfig) -> None:
    client: CatalogClient = CatalogClient(config.catalog_url)
    dataset: DatasetEntry = client.get_dataset(config.dataset_name)
    view: DatasetView = dataset.filter_segments(config.segment_id)
    left, left_T_rig = read_fisheye_camera(view, config.left_cam)
    right, right_T_rig = read_fisheye_camera(view, config.right_cam)
    right_T_left: Float64[np.ndarray, "4 4"] = right_T_rig @ np.linalg.inv(left_T_rig)
    rect: StereoRectification = fisheye_stereo_rectify(left, right, right_T_left, focal_scale=config.focal_scale)
    print(f"rectified pair: baseline {rect.baseline_m * 1000:.1f} mm, fx_rect {rect.K_rect_33[0, 0]:.1f} px, {rect.width}x{rect.height}")

    device: torch.device = torch.device("cuda")
    codec_fourcc: int = int(np.asarray(read_static(view, f"{RIG}/{config.left_cam}/pinhole/video", "VideoStream:codec")).ravel()[0])
    codec: str = "h264" if codec_fourcc.to_bytes(4, "big") == b"avc1" else "av1"  # VideoCodec is the big-endian fourcc
    times, _, _, left_decoder = open_segment_decoder(dataset, config.segment_id, f"{RIG}/{config.left_cam}/pinhole/video", TIMELINE, device, 30, codec)
    right_times, _, _, right_decoder = open_segment_decoder(dataset, config.segment_id, f"{RIG}/{config.right_cam}/pinhole/video", TIMELINE, device, 30, codec)
    t_start_ns: int = int(max(times[0], right_times[0]).astype("int64"))
    t_end_ns: int = int(min(times[-1], right_times[-1]).astype("int64"))
    if config.max_seconds is not None:
        t_end_ns = min(t_end_ns, t_start_ns + int(config.max_seconds * 1e9))
    grid_ns: np.ndarray = np.arange(t_start_ns, t_end_ns, int(1e9 / config.fps), dtype=np.int64)
    print(f"{codec} packets: left {len(times)}, right {len(right_times)}; sampling {len(grid_ns)} frames at {config.fps} fps over {(t_end_ns - t_start_ns) / 1e9:.1f} s")

    predictor: LiteAnyStereoPredictor = LiteAnyStereoPredictor(device=config.device, model_size=config.model_size)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rrd_path: Path = config.output_dir / f"{config.segment_id}.rrd"
    rec: rr.RecordingStream = rr.RecordingStream(application_id="dataforge", recording_id=config.segment_id)
    rec.save(rrd_path)
    log_rectified_camera(rec, config.left_cam, "left_front_rectified", left_T_rig, rect.R0_33, rect, config.output_scale)
    log_rectified_camera(rec, config.right_cam, "right_front_rectified", right_T_rig, rect.R1_33, rect, config.output_scale)
    output_wh: tuple[int, int] = (round(rect.width * config.output_scale), round(rect.height * config.output_scale))
    rect_pinhole: str = f"{RIG}/{config.left_cam}/rectified/pinhole"
    max_depth_mm: float = config.max_depth_m * 1000.0

    def frame_at(decoder: VideoDecoder, sample_times: np.ndarray, t_ns: int) -> UInt8[np.ndarray, "h w 3"]:
        index: int = int(np.searchsorted(sample_times, np.timedelta64(t_ns, "ns"), side="right")) - 1
        return rearrange(decoder.get_frame_at(index).data, "c h w -> h w c").cpu().numpy()

    for i, t_ns in enumerate(grid_ns.tolist()):
        left_rect, right_rect = rectify_pair(rect, frame_at(left_decoder, times, t_ns), frame_at(right_decoder, right_times, t_ns))
        pred: StereoDepthPrediction = predictor(left_rect, right_rect, K_33=rect.K_rect_33.astype(np.float32), baseline_m=rect.baseline_m)
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
        rr.set_time(TIMELINE, duration=np.timedelta64(t_ns, "ns"), recording=rec)
        rr.log(f"{rect_pinhole}/image", rr.Image(left_rect).compress(jpeg_quality=85), recording=rec)
        rr.log(f"{rect_pinhole}/depth", rr.EncodedDepthImage(blob=png.tobytes(), media_type="image/png", meter=1000.0, depth_range=(0.0, max_depth_mm)), recording=rec)
        if i % 50 == 0:
            print(f"  {i}/{len(grid_ns)}  t={(t_ns - t_start_ns) / 1e9:6.1f}s  disparity {pred.disparity.max():.1f} px")
    rr.send_blueprint(build_blueprint(config.left_cam, config.right_cam), recording=rec)
    rec.flush(timeout_sec=60.0)
    rec.disconnect()
    print(f"wrote {rrd_path} ({rrd_path.stat().st_size / 2**20:.1f} MB)")

    if config.register:
        dataset.register([rrd_path.resolve().as_uri()], layer_name=config.layer_name, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
        blueprint_path: Path = config.output_dir / f"{config.dataset_name}-stereo.rbl"
        build_blueprint(config.left_cam, config.right_cam).save(f"{config.dataset_name}-stereo", blueprint_path)
        dataset.register_blueprint(blueprint_path.resolve().as_uri(), set_default=config.default_blueprint)
        print(f"registered layer {config.layer_name!r} on {config.dataset_name}/{config.segment_id} and blueprint {blueprint_path.name}")
