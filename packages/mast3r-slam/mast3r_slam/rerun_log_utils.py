"""Rerun visualisation helpers for MASt3R-SLAM.

Logs camera frustums, keyframe point clouds, the camera path, and
factor-graph edges to a Rerun recording stream.  Uses ``simplecv``'s
``PinholeParameters`` and ``log_pinhole`` for camera logging so the
conversion from lietorch poses to Rerun transforms is done once in
``frame_to_pinhole`` rather than repeated inline.
"""

from pathlib import Path

import lietorch
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from jaxtyping import Bool, Float, Float32, Int, UInt8
from numpy import ndarray
from simplecv.camera_orient_utils import auto_orient_and_center_poses
from simplecv.rerun_log_utils import log_pinhole
from torch import Tensor

from mast3r_slam.frame import Frame, SharedKeyframes, SharedStates
from mast3r_slam.mast3r_utils import frame_to_extrinsics, frame_to_pinhole


def create_blueprints(parent_log_path: Path) -> rrb.Blueprint:
    """Create the default Rerun blueprint layout for the SLAM visualiser.

    Args:
        parent_log_path: Root log path for all Rerun entities.

    Returns:
        A configured Rerun Blueprint.
    """
    views: rrb.Vertical = rrb.Vertical(
        rrb.Spatial2DView(origin=str(parent_log_path / "current_camera" / "pinhole")),
        rrb.Spatial2DView(origin=str(parent_log_path / "last_keyframe")),
        name="Views",
    )
    logs: rrb.TextLogView = rrb.TextLogView(
        origin=str(parent_log_path / "logs"),
        name="Logs",
    )
    blueprint: rrb.Blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/",
                contents=[f"+ {parent_log_path}/**"],
            ),
            rrb.Tabs(views, logs, active_tab=0),
            column_shares=(3, 1),
        ),
        collapse_panels=True,
    )
    return blueprint


class RerunLogger:
    """Logs frames, keyframes, camera paths, and factor-graph edges to Rerun."""

    def __init__(self, parent_log_path: Path) -> None:
        self.parent_log_path: Path = parent_log_path

        self.path_list: list[list[float]] = []
        self.keyframe_logged_list: list[int] = []
        self.num_keyframes_logged: int = 0
        self.conf_thresh: int = 7
        self.image_plane_distance: float = 0.2
        self._last_orient_n_kf: int = 0
        """Number of keyframes used in the last orientation update."""

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

        # Log the current camera image
        rgb_float: Float32[Tensor, "H W 3"] = current_frame.rgb
        rgb_uint8: UInt8[ndarray, "H W 3"] = (rgb_float * 255).numpy().astype(np.uint8)
        rr.log(
            f"{cam_log_path}/pinhole/image",
            rr.Image(image=rgb_uint8, color_model=rr.ColorModel.RGB).compress(jpeg_quality=75),
        )

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
                kf_rgb_float: Float32[Tensor, "H W 3"] = keyframe.rgb
                kf_rgb_uint8: UInt8[ndarray, "H W 3"] = (kf_rgb_float * 255).numpy().astype(np.uint8)
                rr.log(
                    f"{kf_cam_log_path}/pinhole/image",
                    rr.Image(image=kf_rgb_uint8, color_model=rr.ColorModel.RGB).compress(),
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
            last_kf_rgb_float: Float32[Tensor, "H W 3"] = last_kf.rgb
            last_kf_rgb_uint8: UInt8[ndarray, "H W 3"] = (last_kf_rgb_float * 255).numpy().astype(np.uint8)
            rr.log(
                f"{self.parent_log_path}/last_keyframe",
                rr.Image(image=last_kf_rgb_uint8, color_model=rr.ColorModel.RGB).compress(),
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
