"""Rerun visualisation helpers for MASt3R-SLAM.

Provides blueprint creation, video logging, and shared constants
(``FRAME_TIMELINE``, ``VIDEO_TIMELINE``) used by both the pipeline
and the async logger.

The ``RerunLogger`` class is a **legacy synchronous fallback** retained
for debugging and tests.  The active logging path is
``AsyncRerunLogger`` in ``async_logger.py``.
"""

from pathlib import Path

import depth_png
import lietorch_mojo as lietorch
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Bool, Float, Float32, Int, UInt8, UInt16
from numpy import ndarray
from simplecv.camera_orient_utils import auto_orient_and_center_poses
from simplecv.rerun_log_utils import log_pinhole, log_video
from torch import Tensor

from mast3r_slam.dataloader import MonocularDataset, MP4Dataset
from mast3r_slam.frame import Frame, SharedKeyframes, SharedStates
from mast3r_slam.mast3r_utils import frame_to_extrinsics, frame_to_pinhole

FRAME_TIMELINE: str = "frame"
VIDEO_TIMELINE: str = "video_time"
# The source video, logged once as a VideoStream. Camera panels show the model's cropped frame instead.
VIDEO_LOG_PATH: Path = Path("/video")
# Entity layout. Under the parent log path (``/world``) every Pinhole owns only what projects in 3D: the current
# camera its image and depth, each keyframe its image. 2D-only maps live outside it, where no 3D view looks:
CURRENT_MAPS_PATH: Path = Path("/current_maps")
"""The current frame's pointmap and confidence."""
LAST_KEYFRAME_PATH: Path = Path("/last_keyframe")
"""The keyframe being tracked against: image, pointmap, confidence, depth."""
KEYFRAME_MAPS_PATH: Path = Path("/keyframe_maps")
"""Each keyframe's pointmap, confidence and depth, under a static copy of its Pinhole."""


# Chase eye of the Follow view, in the camera frame (RDF: x right, y down, z forward), like dpvo's. MASt3R's
# pointmaps are roughly metric, so this sits 0.6 m above and 1.5 m behind the camera, aimed 1.5 m ahead of it.
_FOLLOW_EYE_POSITION: tuple[float, float, float] = (0.0, -0.6, -1.5)
_FOLLOW_EYE_TARGET: tuple[float, float, float] = (0.0, 0.0, 1.5)
_FOLLOW_EYE_UP: tuple[float, float, float] = (0.0, -1.0, 0.0)


def create_blueprints(parent_log_path: Path, timeline: str = FRAME_TIMELINE) -> rrb.Blueprint:
    """Create the SLAM visualiser's blueprint, sent once at the start of a run.

    Overview and Follow 3D views beside the 2D camera panels, like dpvo. The
    overview is rooted at ``/`` so the gravity-alignment transform on
    ``parent_log_path`` shows (rerun#9244: a Transform3D on a view's own origin
    is invisible), with the ground grid. The Follow view is rooted at the
    current camera, so its eye, fixed in the camera frame, rides the camera; it
    has no grid because the camera frame has no ground plane.

    Both 3D views show ``parent_log_path`` and nothing else: the entity layout
    keeps 2D-only maps outside it (``CURRENT_MAPS_PATH``, ``LAST_KEYFRAME_PATH``,
    ``KEYFRAME_MAPS_PATH``), so the rule needs no exclusions.

    Args:
        parent_log_path: Root log path for all Rerun entities.
        timeline: Timeline shown in the Rerun time panel.

    Returns:
        A configured Rerun Blueprint.
    """
    root: str = f"/{str(parent_log_path).lstrip('/')}"
    camera_path: str = f"{root}/current_camera"
    contents_3d: str = f"+ {root}/**"
    overview = rrb.Spatial3DView(
        name="Overview",
        origin="/",
        contents=contents_3d,
        eye_controls=rrb.EyeControls3D(spin_speed=0.25),
    )
    follow = rrb.Spatial3DView(
        name="Follow",
        origin=camera_path,
        contents=contents_3d,
        line_grid=rrb.LineGrid3D(visible=False),
        eye_controls=rrb.EyeControls3D(
            kind=rrb.Eye3DKind.Orbital,
            position=_FOLLOW_EYE_POSITION,
            look_target=_FOLLOW_EYE_TARGET,
            eye_up=_FOLLOW_EYE_UP,
            spin_speed=0.0,
        ),
    )
    # The original video on top, then what MASt3R derives from the current frame and the keyframe it
    # tracks against. The model's cropped frame shows on the camera frustum in the 3D views.
    views: rrb.Vertical = rrb.Vertical(
        contents=[
            rrb.Spatial2DView(origin=str(VIDEO_LOG_PATH), name="Video"),
            rrb.Grid(
                rrb.Spatial2DView(origin=f"{camera_path}/pinhole/depth", name="Depth"),
                rrb.Spatial2DView(origin=str(CURRENT_MAPS_PATH / "confidence"), name="Confidence"),
                rrb.Spatial2DView(origin=str(CURRENT_MAPS_PATH / "pointmap"), name="Pointmap"),
                rrb.Spatial2DView(origin=str(LAST_KEYFRAME_PATH / "image"), name="Keyframe"),
                grid_columns=2,
            ),
        ],
        row_shares=[1, 1],
        name="Views",
    )
    logs: rrb.TextLogView = rrb.TextLogView(origin=f"{root}/logs", name="Logs")
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(overview, follow),
            rrb.Tabs(views, logs, active_tab=0),
            column_shares=(3, 2),
        ),
        rrb.TimePanel(timeline=timeline, state="collapsed"),
        collapse_panels=True,
    )


def log_video_for_dataset(dataset: MonocularDataset) -> Int[ndarray, "num_frames"] | None:
    """Log the source video once as a VideoStream on ``VIDEO_TIMELINE``.

    Returns the timestamps aligned to the dataset frames, or ``None`` for datasets that are not an MP4.
    """
    if not isinstance(dataset, MP4Dataset) or dataset.dataset_path is None:
        return None

    frame_timestamps_ns: Int[ndarray, "all_video_frames"] = log_video(
        video_source=dataset.dataset_path,
        video_log_path=VIDEO_LOG_PATH,
        timeline=VIDEO_TIMELINE,
    )
    return np.asarray(frame_timestamps_ns[:: dataset.stride][: len(dataset)], dtype=np.int64)


class RerunLogger:
    """Legacy synchronous logger.  Retained as a fallback for debugging and tests.

    The active logging path is :class:`~mast3r_slam.async_logger.AsyncRerunLogger`,
    which consumes event objects from a queue on a background thread.
    """

    def __init__(self, parent_log_path: Path, timeline: str = FRAME_TIMELINE) -> None:
        self.parent_log_path: Path = parent_log_path
        self.timeline: str = timeline

        self.path_list: list[list[float]] = []
        self.keyframe_logged_list: list[int] = []
        self.conf_thresh: int = 7
        self.image_plane_distance: float = 0.2
        # Number of keyframes used in the last orientation update.
        self._last_orient_n_kf: int = 0

    def _log_orient_transform(self, keyframes: SharedKeyframes, n_kf: int) -> None:
        """Recompute gravity-alignment from all keyframe poses and update the transform.

        Collects all ``n_kf`` keyframe world-from-camera poses in RUB convention,
        runs ``auto_orient_and_center_poses(method="up")``
        to align the reconstruction's up-vector with the Y axis, and logs the
        resulting rotation+translation on ``parent_log_path``.
        """
        world_T_cam_list: list[Float32[ndarray, "4 4"]] = []
        for i in range(n_kf):
            kf: Frame = keyframes[i]
            # auto_orient_and_center_poses(method="up") assumes camera column 1 is
            # the camera up axis. That is true for RUB, but RDF's column 1 is down.
            # We therefore orient the global scene from RUB poses even though the
            # logged pinhole cameras themselves stay in RDF.
            kf_ext = frame_to_extrinsics(kf, camera_conventions="RUB")
            world_T_cam_list.append(kf_ext.world_T_cam.astype(np.float32))

        world_T_cam: Float32[ndarray, "n_kf 4 4"] = np.stack(world_T_cam_list)
        orient_34: Float[ndarray, "3 4"] = auto_orient_and_center_poses(
            world_T_cam, method="up", center_method="poses"
        ).transform
        orient_R: Float[ndarray, "3 3"] = orient_34[:, :3]
        orient_t: Float[ndarray, "3"] = orient_34[:, 3]
        rr.log(
            f"{self.parent_log_path}",
            rr.Transform3D(mat3x3=orient_R, translation=orient_t),
        )
        self._last_orient_n_kf = n_kf

    def _log_rgb_image(self, image_path: Path, rgb: Float32[Tensor, "H W 3"]) -> UInt8[ndarray, "H W 3"]:
        rgb_uint8: UInt8[ndarray, "H W 3"] = (rgb * 255).numpy().astype(np.uint8)
        rr.log(
            str(image_path),
            rr.Image(image=rgb_uint8, color_model=rr.ColorModel.RGB).compress(jpeg_quality=75),
        )
        return rgb_uint8

    def _normalize_to_uint8(self, values: ndarray) -> UInt8[ndarray, "..."]:
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            return np.zeros(values.shape, dtype=np.uint8)

        finite_values = values[finite_mask]
        min_value = float(finite_values.min())
        max_value = float(finite_values.max())
        if max_value <= min_value:
            return np.zeros(values.shape, dtype=np.uint8)

        normalized = (values - min_value) / (max_value - min_value)
        return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)

    def _depth_meters_to_uint16_mm(self, depth_meters: Float32[ndarray, "H W"]) -> UInt16[ndarray, "H W"]:
        depth_mm = np.clip(depth_meters * 1000.0, 0.0, float(np.iinfo(np.uint16).max))
        return depth_mm.astype(np.uint16)

    def _log_pointmap_and_confidence(
        self,
        frame: Frame,
        log_path: Path,
        image_shape: tuple[int, int],
    ) -> None:
        if frame.X_canon is None or frame.C is None:
            return

        height, width = image_shape
        pointmap_hw3: Float32[ndarray, "H W 3"] = (
            frame.X_canon.detach().cpu().numpy().reshape(height, width, 3).astype(np.float32)
        )
        pointmap_hw3 = np.nan_to_num(pointmap_hw3, nan=0.0, posinf=0.0, neginf=0.0)
        depth_hw: Float32[ndarray, "H W"] = np.maximum(pointmap_hw3[..., 2], 0.0)
        confidence_mask: Bool[ndarray, "H W"] = frame.C.detach().cpu().numpy().reshape(height, width) > self.conf_thresh
        filtered_depth_hw: Float32[ndarray, "H W"] = np.where(confidence_mask, depth_hw, 0.0)

        conf = frame.get_average_conf()
        if conf is None:
            conf = frame.C
        confidence_hw: Float32[ndarray, "H W"] = conf.detach().cpu().numpy().reshape(height, width).astype(np.float32)
        confidence_hw = np.nan_to_num(confidence_hw, nan=0.0, posinf=0.0, neginf=0.0)
        confidence_uint8: UInt8[ndarray, "H W"] = self._normalize_to_uint8(confidence_hw)
        pointmap_uint8: UInt8[ndarray, "H W 3"] = self._normalize_to_uint8(pointmap_hw3)
        depth_uint16_mm: UInt16[ndarray, "H W"] = self._depth_meters_to_uint16_mm(filtered_depth_hw)

        rr.log(str(log_path / "pointmap"), rr.Image(pointmap_uint8, color_model=rr.ColorModel.RGB).compress())
        rr.log(str(log_path / "pointmap_depth"), rr.EncodedDepthImage(blob=depth_png.encode(depth_uint16_mm), media_type=rr.components.MediaType.PNG, meter=1000.0))
        rr.log(str(log_path / "confidence"), rr.Image(confidence_uint8, color_model=rr.ColorModel.L).compress())

    def log_frame(
        self,
        current_frame: Frame,
        keyframes: SharedKeyframes,
        states: SharedStates,
    ) -> None:
        """Log the current frame, all keyframes, camera path, and graph edges to Rerun.

        Args:
            current_frame: The most recent tracked frame.
            keyframes: Shared keyframe buffer.
            states: Shared system state (for edge lists).
        """
        # Recompute gravity-alignment whenever new keyframes appear.
        with keyframes.lock:
            n_kf: int = len(keyframes)
        if n_kf > 0 and n_kf != self._last_orient_n_kf:
            self._log_orient_transform(keyframes, n_kf)

        # ── Current camera ─────────────────────────────────────────────────
        current_pinhole = frame_to_pinhole(current_frame)
        cam_log_path: Path = self.parent_log_path / "current_camera"

        log_pinhole(
            camera=current_pinhole,
            cam_log_path=cam_log_path,
            image_plane_distance=self.image_plane_distance * 2,
        )

        rgb_uint8: UInt8[ndarray, "H W 3"] = self._log_rgb_image(cam_log_path / "pinhole" / "image", current_frame.rgb)
        self._log_pointmap_and_confidence(current_frame, cam_log_path / "pinhole", rgb_uint8.shape[:2])

        # Log camera path
        assert current_pinhole.extrinsics.world_t_cam is not None
        translation: Float32[ndarray, "3"] = current_pinhole.extrinsics.world_t_cam
        self.path_list.append(translation.tolist())
        rr.log(
            f"{self.parent_log_path}/path",
            rr.LineStrips3D(
                strips=self.path_list,
                colors=(255, 0, 0),
                labels=("Camera Path"),
            ),
        )

        # ── Keyframes ──────────────────────────────────────────────────────
        with keyframes.lock:
            N_keyframes: int = len(keyframes)

        for kf_idx in range(N_keyframes):
            keyframe: Frame = keyframes[kf_idx]
            kf_cam_log_path: Path = self.parent_log_path / "keyframes" / f"keyframe-{kf_idx}"

            # Log image + pointcloud only the first time a keyframe appears.
            if kf_idx not in self.keyframe_logged_list:
                kf_rgb_uint8: UInt8[ndarray, "H W 3"] = self._log_rgb_image(
                    kf_cam_log_path / "pinhole" / "image",
                    keyframe.rgb,
                )
                self._log_pointmap_and_confidence(
                    keyframe,
                    KEYFRAME_MAPS_PATH / f"keyframe-{kf_idx}",
                    kf_rgb_uint8.shape[:2],
                )
                # Confidence-filtered point cloud
                assert keyframe.C is not None
                mask: Bool[ndarray, "hw"] = (keyframe.C.cpu().numpy() > self.conf_thresh).squeeze()

                assert keyframe.X_canon is not None
                positions: Float32[ndarray, "num_points 3"] = keyframe.X_canon.cpu().numpy()
                colors: UInt8[ndarray, "num_points 3"] = kf_rgb_uint8.reshape(-1, 3)

                rr.log(
                    f"{kf_cam_log_path}/pointcloud",
                    rr.Points3D(positions=positions[mask], colors=colors[mask]),
                )
                self.keyframe_logged_list.append(kf_idx)

            # Always update the keyframe's camera; its pose may be refined by
            # the backend's global optimisation since the last log call.
            log_pinhole(
                camera=frame_to_pinhole(keyframe),
                cam_log_path=kf_cam_log_path,
                image_plane_distance=self.image_plane_distance,
            )

        # ── Last keyframe image ────────────────────────────────────────────
        if N_keyframes > 0:
            last_kf: Frame = keyframes[N_keyframes - 1]
            last_kf_rgb_uint8: UInt8[ndarray, "H W 3"] = self._log_rgb_image(
                self.parent_log_path / "last_keyframe" / "image", last_kf.rgb
            )
            self._log_pointmap_and_confidence(
                last_kf,
                self.parent_log_path / "last_keyframe",
                last_kf_rgb_uint8.shape[:2],
            )

        # ── Factor graph edges ─────────────────────────────────────────────
        world_sim3_cami: lietorch.Sim3 | None = None
        world_sim3_camj: lietorch.Sim3 | None = None
        with states.lock:
            ii: Int[Tensor, "num_edges"] = torch.tensor(states.edges_ii, dtype=torch.long)
            jj: Int[Tensor, "num_edges"] = torch.tensor(states.edges_jj, dtype=torch.long)
            if ii.numel() > 0 and jj.numel() > 0:
                world_sim3_cami = lietorch.Sim3(keyframes.world_sim3_cam[ii, 0])
                world_sim3_camj = lietorch.Sim3(keyframes.world_sim3_cam[jj, 0])
        if ii.numel() > 0 and jj.numel() > 0:
            assert world_sim3_cami is not None
            assert world_sim3_camj is not None
            t_world_cami: Float32[ndarray, "num_edges 3"] = world_sim3_cami.matrix()[:, :3, 3].cpu().numpy()
            t_world_camj: Float32[ndarray, "num_edges 3"] = world_sim3_camj.matrix()[:, :3, 3].cpu().numpy()
            line_strips: list[list[float]] = []
            for t_i, t_j in zip(t_world_cami, t_world_camj, strict=False):
                line_strips.append(t_i.tolist())
                line_strips.append(t_j.tolist())
            rr.log(
                f"{self.parent_log_path}/edges",
                rr.LineStrips3D(strips=line_strips, colors=(0, 255, 0), labels=("Factor Graph")),
            )
