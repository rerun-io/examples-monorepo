"""Log a 4DAnyone run directory to Rerun: the source clip and camera, the SMPL-X body and the generated camera rig.

Entity layout (one recording per run, timelines ``time`` (seconds) and ``frame``):

- ``world``: the canonical human world, Y up.
- ``world/body/{mesh,keypoints,skeleton}``: the recovered SMPL-X body and its MHR70 conditioning keypoints.
- ``world/source`` (+ ``/video``): the moving source camera, fitted to the body, and the run's canonical clip.
- ``world/cameras/<NN>`` (+ ``/video``): each generated view's camera and its video.
- ``world/person/splats``: the trained 4D person (``fdanyone_rerun.person4d``), Animated SOG, static.
- ``train/preview``, ``train/loss_rgb``: the person's training, on the ``train_step`` timeline.

Videos are ``rr.VideoStream`` samples (simplecv ``log_video``).
"""

import colorsys
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float32, Float64, UInt32
from numpy import ndarray
from simplecv.ops.umeyama import umeyama_alignment
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


def load_body(output_dir: Path, models: Path) -> BodyTrack:
    """Rebuild the body track from the run's GVHMR motion (on the CPU) with upstream's own geometry code."""
    from fdanyone.assets import MHR70_REGRESSOR
    from fdanyone.motion.result import MotionResult
    from fdanyone.skeleton.pipeline import _body_geometry

    motion = MotionResult.load(output_dir / "gvhmr")
    geometry = _body_geometry(motion, models / MHR70_REGRESSOR, gvhmr_root(models), "cpu", include_mesh=True)
    assert geometry.mesh_world is not None and geometry.mesh_faces is not None
    # The source camera per frame: the rigid fit of the in-camera body onto the world body.
    world_T_source: Float64[ndarray, "frames 4 4"] = np.stack(
        [umeyama_alignment(incam, world).as_matrix() for incam, world in zip(geometry.keypoints_incam, geometry.keypoints_world, strict=True)]
    )
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
PERSON: str = "world/person/splats"
"""The trained 4D person: ``AnimatedSogSplats3D`` + a one-pose ``GaussianSplats3D`` preview, static."""
PERSON_EYE_DISTANCE: float = 0.9
"""The 4D person view's eye: this many times the cameras' distance from the person's middle."""
TRAIN_TIMELINE: str = "train_step"
"""Optimizer steps of the person's training."""
TRAIN_PREVIEW: str = "train/preview"
"""The front view rendered from the model as it trains."""
TRAIN_LOSS: str = "train/loss_rgb"
"""TRiGS' RGB loss."""
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


def training_blueprint() -> rrb.Blueprint:
    """Phase 5 (the 4D person trains): the front view rendered every few hundred steps, the loss and the progress log."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin=TRAIN_PREVIEW, name="Front view, as it trains"),
            rrb.Vertical(rrb.TimeSeriesView(origin=TRAIN_LOSS, name="RGB loss"), rrb.TextLogView(origin=PROGRESS, name="Progress")),
            column_shares=[2.0, 1.0],
        ),
        _time_panel(TRAIN_TIMELINE, playing=False),
        collapse_panels=True,
    )


def person_blueprint(rig: CameraRig) -> rrb.Blueprint:
    """The trained 4D person in 3D beside the source camera, above the generated views. The 3D view looks from between the
    first two cameras at the ring's distance (no frustum in front of the person) and leaves out the body and every image
    plane, which would cover the person."""
    target: Float64[ndarray, " 3"] = np.array([0.0, 0.9, 0.0])
    positions: list[Float64[ndarray, " 3"]] = [camera.world_T_cam[:3, 3] for camera in rig.cameras[:2]]
    between: Float64[ndarray, " 3"] = np.mean(positions, axis=0) - target
    distance: float = float(np.linalg.norm(positions[0] - target))
    eye: Float64[ndarray, " 3"] = target + PERSON_EYE_DISTANCE * distance * between / max(float(np.linalg.norm(between)), 1e-6)
    person: rrb.Spatial3DView = rrb.Spatial3DView(
        origin="world",
        contents=["+ $origin/**", f"- /{BODY}/**", f"- /{SOURCE}/video", *(f"- /{camera_entity(camera.camera_id)}/video" for camera in rig.cameras)],
        name="4D person",
        eye_controls=rrb.EyeControls3D(position=eye.tolist(), look_target=target.tolist(), eye_up=[0.0, 1.0, 0.0]),
    )
    num_views: int = len(rig.cameras)
    top: rrb.Horizontal = rrb.Horizontal(
        person, rrb.Spatial2DView(origin=SOURCE, contents=["$origin/**", f"/{BODY}/**"], name="Source"), column_shares=[3.0, 1.0]
    )
    rows: float = float(-(-num_views // min(num_views, 6)))
    return rrb.Blueprint(
        rrb.Vertical(top, _grid(num_views, camera_entity), row_shares=[2.0, max(1.0, rows / 2.0)]),
        _time_panel(TIMELINE, playing=True),
        collapse_panels=True,
    )


def _log_clip_frames(frames: Iterator[ndarray] | tuple[ndarray, ...], fps: Fraction, recording: rr.RecordingStream) -> None:
    """The canonical clip in the source camera, frame k at k / fps: the clock of the body and the generated views.

    Encoded once as 4:2:0 H.264 without B-frames, which every Rerun viewer decodes (the browser's too): upstream's own
    clip is RGB-lossless High 4:4:4, which Rerun would pass through unchanged (~750 MB for a 4K clip).
    """
    from fdanyone.video import write_video

    with tempfile.TemporaryDirectory() as scratch:
        clip: Path = write_video(frames, Path(scratch) / "source.mp4", fps, crf=18, preset="ultrafast")
        log_video(clip, Path(f"{SOURCE}/video"), timeline=TIMELINE, recording=recording)


def log_canonical_clip(clip: Path, recording: rr.RecordingStream) -> None:
    """Log upstream's canonical clip mp4 (its 121 frames from ``start_time``, as the pipeline's stage record hands it over)."""
    import av

    from fdanyone.video import iter_rgb_video

    with av.open(str(clip)) as container:
        fps: Fraction = Fraction(str(container.streams.video[0].average_rate))
    _log_clip_frames(iter_rgb_video(clip), fps, recording)


def log_input_clip(video: Path, record: InputRecord, recording: rr.RecordingStream) -> None:
    """Cut a finished run's canonical clip from its input video again (upstream's own sampling) and log it."""
    from fdanyone.video import decode_canonical_clip

    canonical = decode_canonical_clip(video, num_frames=record.num_frames, start_time=record.start_time_seconds, fps=record.frame_rate)
    _log_clip_frames(canonical.rgb_frames, canonical.fps, recording)


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
