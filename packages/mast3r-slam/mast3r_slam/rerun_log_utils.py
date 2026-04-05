import numpy as np
import torch
import rerun as rr
from jaxtyping import UInt8, Float32, Int, Bool
from pathlib import Path

from mast3r_slam.frame import Frame, SharedKeyframes, SharedStates
from mast3r_slam.mast3r_utils import estimate_focal_knowing_depth
import lietorch
from mast3r_slam.lietorch_utils import as_SE3
from simplecv.ops import conventions
import rerun.blueprint as rrb


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
                rrb.Spatial2DView(
                    origin=parent_log_path / "current_camera" / "pinhole"
                ),
                rrb.Spatial2DView(origin=parent_log_path / "last_keyframe"),
                rrb.TextDocumentView(origin=parent_log_path),
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
        # Set up-axis convention on the root so the 3D view (origin="/") is correctly oriented.
        # RFU = Right-Forward-Up (Y-up convention matching Rerun's default grid).
        rr.log("/", rr.ViewCoordinates.RFU, static=True)

        self.path_list: list[list[float]] = []
        self.keyframe_logged_list: list[int] = []
        self.num_keyframes_logged: int = 0
        self.conf_thresh: int = 7
        self.image_plane_distance: float = 0.2

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
        # Add your rerun logging logic here
        H: int = current_frame.img_shape.squeeze()[0].item()
        W: int = current_frame.img_shape.squeeze()[1].item()

        pp: Float32[torch.Tensor, "2"] = torch.tensor((W / 2, H / 2))
        pts3d: Float32[torch.Tensor, "H W 3"] = (
            current_frame.X_canon.clone().cpu().reshape(H, W, 3)
        )
        focal: float = float(
            estimate_focal_knowing_depth(pts3d[None], pp, focal_mode="weiszfeld")
        )

        rgb_img_float: Float32[torch.Tensor, "H W 3"] = current_frame.uimg
        rgb_img: UInt8[np.ndarray, "H W 3"] = (rgb_img_float * 255).numpy().astype(np.uint8)

        se3_pose: lietorch.SE3 = as_SE3(current_frame.T_WC.cpu())
        matb4x4: Float32[np.ndarray, "1 4 4"] = (
            se3_pose.matrix().numpy().astype(dtype=np.float32)
        )
        mat4x4: Float32[np.ndarray, "4 4"] = matb4x4[
            0
        ]  # Extract the first batch element

        mat4x4 = conventions.convert_pose(
            mat4x4, src_convention=conventions.CC.CV, dst_convention=conventions.CC.GL
        )

        # Extract rotation (3x3) and translation (1x3) from the 4x4 transformation matrix
        rotation_matrix: Float32[np.ndarray, "3 3"] = mat4x4[
            :3, :3
        ]  # Top-left 3x3 block
        translation_vector: Float32[np.ndarray, "3"] = mat4x4[
            :3, 3
        ]  # Right column, first 3 elements

        cam_log_path: Path = self.parent_log_path / "current_camera"
        rr.log(
            f"{cam_log_path}",
            rr.Transform3D(translation=translation_vector, mat3x3=rotation_matrix),
        )
        rr.log(
            f"{cam_log_path}/pinhole",
            rr.Pinhole(
                focal_length=focal,
                principal_point=pp.numpy(),
                height=H,
                width=W,
                camera_xyz=rr.ViewCoordinates.RUB,
                image_plane_distance=self.image_plane_distance * 2,
            ),
        )
        rr.log(
            f"{cam_log_path}/pinhole/image",
            rr.Image(image=rgb_img, color_model=rr.ColorModel.RGB).compress(
                jpeg_quality=75
            ),
        )
        self.path_list.append(translation_vector.tolist())
        rr.log(
            f"{self.parent_log_path}/path",
            rr.LineStrips3D(
                strips=self.path_list,
                colors=(255, 0, 0),
                labels=("Camera Path"),
            ),
        )

        with keyframes.lock:
            N_keyframes: int = len(keyframes)
            # dirty_idx = keyframes.get_dirty_idx()

        for kf_idx in range(N_keyframes):
            keyframe: Frame = keyframes[kf_idx]
            se3_pose = as_SE3(keyframe.T_WC.cpu())
            matb4x4 = (
                se3_pose.matrix().numpy().astype(dtype=np.float32)
            )
            mat4x4 = matb4x4[
                0
            ]  # Extract the first batch element

            # Extract rotation (3x3) and translation (1x3) from the 4x4 transformation matrix
            rotation_matrix = mat4x4[
                :3, :3
            ]  # Top-left 3x3 block
            translation_vector = mat4x4[
                :3, 3
            ]  # Right column, first 3 elements
            cam_log_path = self.parent_log_path / "keyframes" / f"keyframe-{kf_idx}"
            if kf_idx not in self.keyframe_logged_list:
                kf_img_float: Float32[torch.Tensor, "H W 3"] = keyframe.uimg
                kf_img: UInt8[np.ndarray, "H W 3"] = (
                    (kf_img_float * 255).numpy().astype(np.uint8)
                )
                rr.log(
                    f"{cam_log_path}/pinhole/image",
                    rr.Image(image=kf_img, color_model=rr.ColorModel.RGB).compress(),
                )
                # create a mask based on the confidence values
                mask_raw: Bool[np.ndarray, "hw 1"] = keyframe.C.cpu().numpy() > self.conf_thresh

                # Convert the mask from shape (h*w, 1) to shape (h*w,)
                mask: Bool[np.ndarray, "hw"] = (
                    mask_raw.squeeze()
                )  # Remove the trailing dimension to get a 1D boolean array

                # Now apply the mask to both positions and colors
                positions: Float32[np.ndarray, "num_points 3"] = (
                    keyframe.X_canon.cpu().numpy()
                )
                colors: UInt8[np.ndarray, "num_points 3"] = kf_img.reshape(-1, 3)

                masked_positions: Float32[np.ndarray, "n_valid 3"] = positions[
                    mask
                ]  # Now selects entire rows where mask is True
                masked_colors: UInt8[np.ndarray, "n_valid 3"] = colors[mask]
                rr.log(
                    f"{cam_log_path}/pointcloud",
                    rr.Points3D(
                        positions=masked_positions,
                        colors=masked_colors,
                    ),
                )
                self.keyframe_logged_list.append(kf_idx)
            rr.log(
                f"{cam_log_path}",
                rr.Transform3D(translation=translation_vector, mat3x3=rotation_matrix),
            )
            rr.log(
                f"{cam_log_path}/pinhole",
                rr.Pinhole(
                    focal_length=focal,
                    principal_point=pp.numpy(),
                    height=H,
                    width=W,
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=self.image_plane_distance,
                ),
            )

        # log the last keyframe image
        if N_keyframes > 0:
            last_kf: Frame = keyframes[N_keyframes - 1]
            last_kf_img_float: Float32[torch.Tensor, "H W 3"] = last_kf.uimg
            last_kf_img: UInt8[np.ndarray, "H W 3"] = (
                (last_kf_img_float * 255).numpy().astype(np.uint8)
            )
            rr.log(
                f"{self.parent_log_path}/last_keyframe",
                rr.Image(image=last_kf_img, color_model=rr.ColorModel.RGB).compress(),
            )

        # Log the edges
        with states.lock:
            ii: Int[torch.Tensor, "num_edges"] = torch.tensor(
                states.edges_ii, dtype=torch.long
            )
            jj: Int[torch.Tensor, "num_edges"] = torch.tensor(
                states.edges_jj, dtype=torch.long
            )
            if ii.numel() > 0 and jj.numel() > 0:
                T_WCi: lietorch.Sim3 = lietorch.Sim3(keyframes.T_WC[ii, 0])
                T_WCj: lietorch.Sim3 = lietorch.Sim3(keyframes.T_WC[jj, 0])
        if ii.numel() > 0 and jj.numel() > 0:
            t_WCi: Float32[np.ndarray, "num_edges 3"] = T_WCi.matrix()[:, :3, 3].cpu().numpy()
            t_WCj: Float32[np.ndarray, "num_edges 3"] = T_WCj.matrix()[:, :3, 3].cpu().numpy()
            line_strips: list[list[float]] = []
            for t_i, t_j in zip(t_WCi, t_WCj):
                line_strips.append(t_i.tolist())
                line_strips.append(t_j.tolist())
            rr.log(
                f"{self.parent_log_path}/edges",
                rr.LineStrips3D(
                    strips=line_strips, colors=(0, 255, 0), labels=("Factor Graph")
                ),
            )
