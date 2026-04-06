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
from torch import Tensor

from mast3r_slam.frame import Frame, SharedKeyframes, SharedStates
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.mast3r_utils import frame_to_extrinsics, frame_to_pinhole


def create_blueprints(parent_log_path: Path) -> rrb.Blueprint:
    """Create the default Rerun blueprint layout for the SLAM visualiser.

    Args:
        parent_log_path: Root log path for all Rerun entities.

    Returns:
        A configured Rerun Blueprint.
    """
    blueprint: rrb.Blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/",
                contents=[f"+ {parent_log_path}/**"],
            ),
            rrb.Vertical(
                rrb.Spatial2DView(origin=str(parent_log_path / "current_camera" / "pinhole")),
                rrb.Spatial2DView(origin=str(parent_log_path / "last_keyframe")),
                rrb.TextDocumentView(origin=str(parent_log_path)),
            ),
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

        Collects all ``n_kf`` keyframe world-from-camera poses (in GL convention
        via ``frame_to_pinhole``), runs ``auto_orient_and_center_poses(method="up")``
        to align the reconstruction's up-vector with the Y axis, and logs the
        resulting rotation+translation on ``parent_log_path``.
        """
        world_T_cam_gl_list: list[Float32[ndarray, "4 4"]] = []
        for i in range(n_kf):
            kf: Frame = keyframes[i]
            kf_ext = frame_to_extrinsics(kf)
            world_T_cam_gl_list.append(kf_ext.world_T_cam.astype(np.float32))

        world_T_cam_gl: Float32[ndarray, "n_kf 4 4"] = np.stack(world_T_cam_gl_list)
        orient_34: Float[ndarray, "3 4"] = auto_orient_and_center_poses(
            world_T_cam_gl, method="up", center_method="poses"
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

        # Log transform using world-from-camera (no from_parent) so camera
        # frustums are placed at their world position.
        rr.log(
            f"{cam_log_path}",
            rr.Transform3D(
                translation=current_pinhole.extrinsics.world_t_cam,
                mat3x3=current_pinhole.extrinsics.world_R_cam,
            ),
        )
        rr.log(
            f"{cam_log_path}/pinhole",
            rr.Pinhole(
                image_from_camera=current_pinhole.intrinsics.k_matrix,
                height=current_pinhole.intrinsics.height,
                width=current_pinhole.intrinsics.width,
                camera_xyz=rr.ViewCoordinates.RUB,
                image_plane_distance=self.image_plane_distance * 2,
            ),
        )

        # Log the current camera image
        rgb_img_float: Float32[Tensor, "H W 3"] = current_frame.uimg
        rgb_img: UInt8[ndarray, "H W 3"] = (rgb_img_float * 255).numpy().astype(np.uint8)
        rr.log(
            f"{cam_log_path}/pinhole/image",
            rr.Image(image=rgb_img, color_model=rr.ColorModel.RGB).compress(jpeg_quality=75),
        )

        # Log camera path
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
                kf_img_float: Float32[Tensor, "H W 3"] = keyframe.uimg
                kf_img: UInt8[ndarray, "H W 3"] = (kf_img_float * 255).numpy().astype(np.uint8)
                rr.log(
                    f"{kf_cam_log_path}/pinhole/image",
                    rr.Image(image=kf_img, color_model=rr.ColorModel.RGB).compress(),
                )
                # Confidence-filtered point cloud
                assert keyframe.C is not None
                mask: Bool[ndarray, "hw"] = (keyframe.C.cpu().numpy() > self.conf_thresh).squeeze()

                assert keyframe.X_canon is not None
                positions: Float32[ndarray, "num_points 3"] = keyframe.X_canon.cpu().numpy()
                colors: UInt8[ndarray, "num_points 3"] = kf_img.reshape(-1, 3)

                rr.log(
                    f"{kf_cam_log_path}/pointcloud",
                    rr.Points3D(positions=positions[mask], colors=colors[mask]),
                )
                self.keyframe_logged_list.append(kf_idx)

            # Always update the keyframe's pose (it may have been refined by
            # the backend's global optimisation since the last log call).
            # Keyframes use CV convention (RDF) — no GL conversion needed.
            kf_se3: lietorch.SE3 = as_SE3(keyframe.world_T_cam.cpu())
            kf_mat4x4: Float32[ndarray, "4 4"] = kf_se3.matrix().numpy().astype(np.float32)[0]
            rr.log(
                f"{kf_cam_log_path}",
                rr.Transform3D(
                    translation=kf_mat4x4[:3, 3],
                    mat3x3=kf_mat4x4[:3, :3],
                ),
            )
            rr.log(
                f"{kf_cam_log_path}/pinhole",
                rr.Pinhole(
                    focal_length=current_pinhole.intrinsics.fl_x,
                    principal_point=[current_pinhole.intrinsics.cx, current_pinhole.intrinsics.cy],
                    height=current_pinhole.intrinsics.height,
                    width=current_pinhole.intrinsics.width,
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=self.image_plane_distance,
                ),
            )

        # ── Last keyframe image ────────────────────────────────────────────
        if N_keyframes > 0:
            last_kf: Frame = keyframes[N_keyframes - 1]
            last_kf_img_float: Float32[Tensor, "H W 3"] = last_kf.uimg
            last_kf_img: UInt8[ndarray, "H W 3"] = (last_kf_img_float * 255).numpy().astype(np.uint8)
            rr.log(
                f"{self.parent_log_path}/last_keyframe",
                rr.Image(image=last_kf_img, color_model=rr.ColorModel.RGB).compress(),
            )

        # ── Factor graph edges ─────────────────────────────────────────────
        world_T_cami: lietorch.Sim3 | None = None
        world_T_camj: lietorch.Sim3 | None = None
        with states.lock:
            ii: Int[Tensor, "num_edges"] = torch.tensor(states.edges_ii, dtype=torch.long)
            jj: Int[Tensor, "num_edges"] = torch.tensor(states.edges_jj, dtype=torch.long)
            if ii.numel() > 0 and jj.numel() > 0:
                world_T_cami = lietorch.Sim3(keyframes.world_T_cam[ii, 0])
                world_T_camj = lietorch.Sim3(keyframes.world_T_cam[jj, 0])
        if ii.numel() > 0 and jj.numel() > 0:
            assert world_T_cami is not None
            assert world_T_camj is not None
            t_world_cami: Float32[ndarray, "num_edges 3"] = world_T_cami.matrix()[:, :3, 3].cpu().numpy()
            t_world_camj: Float32[ndarray, "num_edges 3"] = world_T_camj.matrix()[:, :3, 3].cpu().numpy()
            line_strips: list[list[float]] = []
            for t_i, t_j in zip(t_world_cami, t_world_camj):
                line_strips.append(t_i.tolist())
                line_strips.append(t_j.tolist())
            rr.log(
                f"{self.parent_log_path}/edges",
                rr.LineStrips3D(strips=line_strips, colors=(0, 255, 0), labels=("Factor Graph")),
            )
