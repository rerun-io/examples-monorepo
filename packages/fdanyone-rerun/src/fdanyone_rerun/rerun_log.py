"""Log a 4DAnyone run directory to Rerun: the source clip and camera, the SMPL-X body and the generated camera rig.

Entity layout (one recording per run, timelines ``time`` (seconds) and ``frame``):

- ``world``: the canonical human world, Y up.
- ``world/body/{mesh,keypoints,skeleton}``: the recovered SMPL-X body and its MHR70 conditioning keypoints.
- ``world/source`` (+ ``/video``): the moving source camera, fitted to the body, and the input clip.
- ``world/cameras/<NN>`` (+ ``/video``): each generated view's camera and its video.

Videos are ``rr.VideoStream`` samples (simplecv ``log_video``).
"""

import colorsys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, Float64, UInt32
from numpy import ndarray
from simplecv.rerun_log_utils import log_video

from fdanyone_rerun.assets import gvhmr_root
from fdanyone_rerun.outputs import CameraRecord, CameraRig, InputRecord, RunMetadata, read_camera_rig, read_run_metadata

TIMELINE: str = "time"
"""Seconds since the first canonical frame; the videos' timeline."""
FRAME_TIMELINE: str = "frame"
"""Canonical frame index (0..120)."""
SOURCE: str = "world/source"
BODY: str = "world/body"
CAMERAS: str = "world/cameras"
CAMERA_PLANE_DISTANCE: float = 0.35
"""Frustum depth of every camera in the 3D view, metres."""
BODY_RGBA: tuple[int, int, int, int] = (180, 180, 180, 90)
"""Translucent grey, so the skeleton shows through (upstream GUI's body colour)."""


def camera_entity(camera_id: int) -> str:
    """Entity of one generated view's camera."""
    return f"{CAMERAS}/{camera_id:02d}"


def camera_color(yaw_deg: float) -> tuple[int, int, int]:
    """Hue by yaw, so a frustum in the 3D view and its grid pane match."""
    rgb: tuple[float, float, float] = colorsys.hsv_to_rgb((yaw_deg % 360.0) / 360.0, 0.65, 1.0)
    return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


@dataclass(frozen=True, slots=True)
class BodyTrack:
    """The recovered body over the canonical clip, in the canonical world."""

    vertices: Float32[ndarray, "frames verts 3"]
    """SMPL-X mesh vertices."""
    faces: UInt32[ndarray, "faces 3"]
    """SMPL-X triangles."""
    keypoints: Float32[ndarray, "frames 70 3"]
    """MHR70 keypoints (the skeleton the views are conditioned on)."""
    world_T_source: Float64[ndarray, "frames 4 4"]
    """Source camera pose per frame, fitted between the in-camera and world keypoints."""
    intrinsics: Float64[ndarray, "3 3"]
    """Source camera pixel intrinsics (first frame)."""
    source_wh: tuple[int, int]
    """Source video size, pixels."""


def source_transforms(
    points_incam: Float32[ndarray, "frames points 3"], points_world: Float32[ndarray, "frames points 3"]
) -> Float64[ndarray, "frames 4 4"]:
    """The rigid source-camera pose per frame that maps the in-camera body onto the world body (Kabsch fit).

    From upstream 4DAnyone's GUI (fdanyone/space/body.py, not copied).
    """
    source: Float64[ndarray, "frames points 3"] = np.asarray(points_incam, dtype=np.float64)
    target: Float64[ndarray, "frames points 3"] = np.asarray(points_world, dtype=np.float64)
    source_mean: Float64[ndarray, "frames 3"] = source.mean(axis=1)
    target_mean: Float64[ndarray, "frames 3"] = target.mean(axis=1)
    covariance: Float64[ndarray, "frames 3 3"] = np.swapaxes(source - source_mean[:, None], 1, 2) @ (target - target_mean[:, None])
    left, _, right = np.linalg.svd(covariance)
    correction: Float64[ndarray, "frames 3 3"] = np.broadcast_to(np.eye(3), covariance.shape).copy()
    correction[:, 2, 2] = np.linalg.det(left @ right)
    row_rotation: Float64[ndarray, "frames 3 3"] = left @ correction @ right
    transforms: Float64[ndarray, "frames 4 4"] = np.broadcast_to(np.eye(4), (len(source), 4, 4)).copy()
    transforms[:, :3, :3] = np.swapaxes(row_rotation, 1, 2)
    transforms[:, :3, 3] = target_mean - np.einsum("fi,fij->fj", source_mean, row_rotation)
    return transforms


def load_body(output_dir: Path, models: Path) -> BodyTrack:
    """Rebuild the body track from the run's GVHMR motion (on the CPU) with upstream's own geometry code."""
    from fdanyone.assets import MHR70_REGRESSOR
    from fdanyone.motion.result import MotionResult
    from fdanyone.skeleton.pipeline import _body_geometry

    motion = MotionResult.load(output_dir / "gvhmr")
    geometry = _body_geometry(motion, models / MHR70_REGRESSOR, gvhmr_root(models), "cpu", include_mesh=True)
    assert geometry.mesh_world is not None and geometry.mesh_faces is not None
    world_T_source: Float64[ndarray, "frames 4 4"] = source_transforms(geometry.keypoints_incam, geometry.keypoints_world)
    return BodyTrack(
        vertices=geometry.mesh_world,
        faces=geometry.mesh_faces,
        keypoints=geometry.keypoints_world,
        world_T_source=world_T_source,
        intrinsics=motion.K_fullimg[0].numpy().astype(np.float64),
        source_wh=(int(motion.image_width), int(motion.image_height)),
    )


def _time_columns(num_frames: int, fps: float) -> list[rr.TimeColumn]:
    frames: ndarray = np.arange(num_frames)
    return [rr.TimeColumn(TIMELINE, duration=frames / fps), rr.TimeColumn(FRAME_TIMELINE, sequence=frames)]


def log_body(body: BodyTrack, fps: float, recording: rr.RecordingStream) -> None:
    """Log the mesh, keypoints and skeleton over the clip, and the moving source camera."""
    from fdanyone.skeleton.keypoints import LINKS, VISIBLE_KEYPOINT_IDS, keypoint_color

    num_frames: int = len(body.vertices)
    num_verts: int = body.vertices.shape[1]
    visible: list[int] = sorted(VISIBLE_KEYPOINT_IDS)
    pairs: list[tuple[int, int]] = [(first, second) for _, first, second, _, _ in LINKS]
    recording.log(f"{BODY}/mesh", rr.Mesh3D.from_fields(triangle_indices=body.faces, albedo_factor=BODY_RGBA), static=True)
    recording.log(f"{BODY}/keypoints", rr.Points3D.from_fields(radii=0.012, colors=[keypoint_color(i) for i in visible]), static=True)
    recording.log(f"{BODY}/skeleton", rr.LineStrips3D.from_fields(radii=0.008, colors=[color for _, _, _, color, _ in LINKS]), static=True)
    times: list[rr.TimeColumn] = _time_columns(num_frames, fps)
    recording.send_columns(
        f"{BODY}/mesh",
        indexes=times,
        columns=rr.Mesh3D.columns(vertex_positions=body.vertices.reshape(-1, 3)).partition([num_verts] * num_frames),
    )
    keypoints: Float32[ndarray, "frames visible 3"] = body.keypoints[:, visible]
    recording.send_columns(
        f"{BODY}/keypoints",
        indexes=times,
        columns=rr.Points3D.columns(positions=keypoints.reshape(-1, 3)).partition([len(visible)] * num_frames),
    )
    strips: Float32[ndarray, "frames links 2 3"] = body.keypoints[:, np.asarray(pairs)]
    recording.send_columns(
        f"{BODY}/skeleton",
        indexes=times,
        columns=rr.LineStrips3D.columns(strips=list(strips.reshape(-1, 2, 3))).partition([len(pairs)] * num_frames),
    )
    recording.log(
        SOURCE,
        rr.Pinhole(
            image_from_camera=body.intrinsics,
            resolution=body.source_wh,
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=CAMERA_PLANE_DISTANCE,
        ),
        static=True,
    )
    recording.send_columns(
        SOURCE,
        indexes=times,
        columns=rr.Transform3D.columns(translation=body.world_T_source[:, :3, 3], mat3x3=body.world_T_source[:, :3, :3]),
    )


def log_camera(camera: CameraRecord, output_dir: Path, recording: rr.RecordingStream) -> None:
    """Log one generated view: its fixed camera and its video."""
    entity: str = camera_entity(camera.camera_id)
    world_T_cam: Float64[ndarray, "4 4"] = camera.world_T_cam
    recording.log(entity, rr.Transform3D(translation=world_T_cam[:3, 3], mat3x3=world_T_cam[:3, :3]), static=True)
    recording.log(
        entity,
        rr.Pinhole(
            image_from_camera=camera.intrinsics,
            resolution=(camera.image_width, camera.image_height),
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=CAMERA_PLANE_DISTANCE,
            color=camera_color(camera.yaw),
        ),
        static=True,
    )
    log_video(output_dir / camera.video, Path(f"{entity}/video"), timeline=TIMELINE, recording=recording)


PROGRESS: str = "log"
"""Stage and step messages (TextLog)."""
PREVIEWS: str = "previews"
"""``previews/<NN>``: each view's x_0 estimate per denoising step, on ``diffusion_step``."""
DIFFUSION_TIMELINE: str = "diffusion_step"
"""Denoising step (0-3 for Turbo); only the previews live on it."""


def preview_entity(view: int) -> str:
    """Entity of one view's denoising preview."""
    return f"{PREVIEWS}/{view:02d}"


def _time_panel(timeline: str, *, playing: bool) -> rrb.TimePanel:
    return rrb.TimePanel(timeline=timeline, play_state="playing" if playing else "following", loop_mode="all" if playing else None, state="collapsed")


def motion_blueprint() -> rrb.Blueprint:
    """Phase 1 (motion recovery): the canonical clip (once cut) beside the progress log."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin=SOURCE, contents=["$origin/**"], name="Source"),
            rrb.TextLogView(origin=PROGRESS, name="Progress"),
            column_shares=[2.0, 1.0],
        ),
        _time_panel(TIMELINE, playing=True),
        collapse_panels=True,
    )


def body_blueprint() -> rrb.Blueprint:
    """Phase 2 (conditioning): the recovered body in 3D and over the source clip, while skeletons render."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="Canonical world"),
            rrb.Spatial2DView(origin=SOURCE, contents=["$origin/**", f"/{BODY}/**"], name="Source"),
            rrb.TextLogView(origin=PROGRESS, name="Progress"),
            column_shares=[2.0, 1.0, 1.0],
        ),
        _time_panel(TIMELINE, playing=True),
        collapse_panels=True,
    )


def _grid(num_views: int, entity: Callable[[int], str]) -> rrb.Grid:
    return rrb.Grid(
        *(rrb.Spatial2DView(origin=entity(index), contents=["$origin/**"], name=f"View {index:02d}") for index in range(num_views)),
        grid_columns=min(num_views, 6),
    )


def diffusion_blueprint(num_views: int) -> rrb.Blueprint:
    """Phase 3 (denoising): a grid of per-view previews that sharpens with every step, beside the progress log."""
    return rrb.Blueprint(
        rrb.Horizontal(_grid(num_views, preview_entity), rrb.TextLogView(origin=PROGRESS, name="Progress"), column_shares=[3.0, 1.0]),
        _time_panel(DIFFUSION_TIMELINE, playing=False),
        collapse_panels=True,
    )


def result_blueprint(num_views: int) -> rrb.Blueprint:
    """The rig and body in 3D beside the source camera, above a grid of the generated views, playing on a loop."""
    top: rrb.Horizontal = rrb.Horizontal(
        rrb.Spatial3DView(origin="world", name="Canonical world"),
        rrb.Spatial2DView(origin=SOURCE, contents=["$origin/**", f"/{BODY}/**"], name="Source"),
        column_shares=[3.0, 1.0],
    )
    rows: float = float(-(-num_views // min(num_views, 6)))
    return rrb.Blueprint(
        rrb.Vertical(top, _grid(num_views, camera_entity), row_shares=[2.0, max(1.0, rows / 2.0)]),
        _time_panel(TIMELINE, playing=True),
        collapse_panels=True,
    )


def log_canonical_clip(clip: Path, recording: rr.RecordingStream) -> None:
    """The run's canonical clip in the source camera: upstream's 121 frames from ``start_time``, frame k at k / fps, the
    clock of the body and the generated views. Upstream writes it RGB-lossless (4:4:4) for GVHMR; it is re-encoded
    to H.264, which every Rerun viewer decodes."""
    log_video(clip, Path(f"{SOURCE}/video"), timeline=TIMELINE, recording=recording, output_codec=rr.VideoCodec.H264)


def log_input_clip(video: Path, clip: InputRecord, recording: rr.RecordingStream) -> None:
    """Cut a finished run's canonical clip from its input video again (upstream's own sampling) and log it."""
    from fdanyone.video import decode_canonical_clip, write_gvhmr_video

    canonical = decode_canonical_clip(video, num_frames=clip.num_frames, start_time=clip.start_time_seconds, fps=Fraction(clip.fps))
    with tempfile.TemporaryDirectory() as scratch:
        log_canonical_clip(write_gvhmr_video(canonical, Path(scratch) / "canonical_clip.mp4"), recording)


def log_cameras(rig: CameraRig, output_dir: Path, recording: rr.RecordingStream) -> None:
    """Every generated view's camera and video."""
    for camera in rig.cameras:
        log_camera(camera, output_dir, recording)


def log_run(output_dir: Path, video: Path | None, models: Path, recording: rr.RecordingStream) -> RunMetadata:
    """Log a finished run directory: body, source camera (+ the input clip when given) and every generated view."""
    metadata: RunMetadata = read_run_metadata(output_dir)
    rig: CameraRig = read_camera_rig(output_dir)
    recording.send_blueprint(result_blueprint(len(rig.cameras)))
    recording.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    log_body(load_body(output_dir, models), float(metadata.input.frame_rate), recording)
    if video is not None:
        log_input_clip(video, metadata.input, recording)
    log_cameras(rig, output_dir, recording)
    return metadata
