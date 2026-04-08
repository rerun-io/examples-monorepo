import lietorch
import rerun as rr
import torch
from jaxtyping import Bool, Float, Int
from mast3r.model import AsymmetricMASt3R
from torch import Tensor

try:
    from beartype.roar import BeartypeException
except ImportError:
    BeartypeException = ()

from mast3r_slam.config import config
from mast3r_slam.frame import Frame, SharedKeyframes
from mast3r_slam.geometry import (
    act_Sim3,
    constrain_points_to_ray,
    get_pixel_coords,
    point_to_ray_dist,
    project_calib,
)
from mast3r_slam.mast3r_utils import mast3r_match_asymmetric
from mast3r_slam.nonlinear_optimizer import check_convergence, huber
from mast3r_slam.perf import timed_section


class FrameTracker:
    """Tracks the pose of incoming frames against the most recent keyframe.

    Uses iterative Gauss-Newton optimisation on either ray-distance or
    calibrated reprojection residuals.
    """

    def __init__(
        self,
        model: AsymmetricMASt3R,
        frames: SharedKeyframes,
        device: str,
    ) -> None:
        self.cfg: dict = config["tracking"]
        self.model: AsymmetricMASt3R = model
        self.keyframes: SharedKeyframes = frames
        self.device: str = device

        self.last_profile: dict[str, float | int | str] = {}
        self.last_opt_steps: int = 0
        self.reset_idx_f2k()

    # Initialize with identity indexing of size (1,n)
    def reset_idx_f2k(self) -> None:
        """Reset the frame-to-keyframe correspondence cache to identity."""
        self.idx_f2k: Int[Tensor, "1 hw"] | None = None

    def track(self, frame: Frame) -> tuple[bool, list, bool]:
        """Track the frame pose against the last keyframe.

        Args:
            frame: The current Frame to track; its ``world_sim3_cam`` is updated in place.

        Returns:
            A tuple of (new_kf, match_info, try_reloc) where new_kf is True
            when a new keyframe should be created, match_info contains
            diagnostic tensors, and try_reloc is True when relocalization
            should be attempted.
        """
        profile: dict[str, float | int | str] = {}
        last_kf: Frame | None = self.keyframes.last_keyframe()
        assert last_kf is not None, "Cannot track without a keyframe"
        keyframe: Frame = last_kf

        with timed_section(profile, "mast3r_match_total_ms", sync_cuda=True):
            idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
                self.model, frame, keyframe, idx_i2j_init=self.idx_f2k, profile=profile
            )
        # Save idx for next
        self.idx_f2k = idx_f2k.clone()

        # Get rid of batch dim
        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0]

        Qk: Float[Tensor, "hw 1"] = torch.sqrt(Qff[idx_f2k] * Qkf)

        # Update keyframe pointmap after registration (need pose)
        frame.update_pointmap(Xff, Cff)

        use_calib: bool = config["use_calib"]
        img_size: tuple[int, int] = (int(frame.rgb_tensor.shape[-2]), int(frame.rgb_tensor.shape[-1]))
        K: Float[Tensor, "3 3"] | None = keyframe.K if use_calib else None

        # Get poses and point correspondences and confidences
        Xf: Float[Tensor, "hw 3"]
        Xk: Float[Tensor, "hw 3"]
        world_sim3_camf: lietorch.Sim3
        world_sim3_camk: lietorch.Sim3
        Cf: Float[Tensor, "hw 1"]
        Ck: Float[Tensor, "hw 1"]
        meas_k: Float[Tensor, "hw 3"] | None
        valid_meas_k: Bool[Tensor, "hw 1"] | None
        with timed_section(profile, "points_pose_prep_ms", sync_cuda=False):
            Xf, Xk, world_sim3_camf, world_sim3_camk, Cf, Ck, meas_k, valid_meas_k = self.get_points_poses(
                frame, keyframe, idx_f2k, img_size, use_calib, K
            )

        # Get valid
        # Use canonical confidence average
        valid_Cf: Bool[Tensor, "hw 1"] = Cf > self.cfg["C_conf"]
        valid_Ck: Bool[Tensor, "hw 1"] = Ck > self.cfg["C_conf"]
        valid_Q: Bool[Tensor, "hw 1"] = Qk > self.cfg["Q_conf"]

        valid_opt: Bool[Tensor, "hw 1"] = valid_match_k & valid_Cf & valid_Ck & valid_Q
        valid_kf: Bool[Tensor, "hw 1"] = valid_match_k & valid_Q

        match_frac: Float[Tensor, ""] = valid_opt.sum() / valid_opt.numel()
        profile["match_fraction"] = float(match_frac.item())
        if match_frac < self.cfg["min_match_frac"]:
            self.last_profile = profile
            rr.log("/world/logs", rr.TextLog(f"Skipped frame {frame.frame_id}", level="DEBUG"))
            return False, [], True

        try:
            # Track
            if not use_calib:
                with timed_section(profile, "optimize_ms", sync_cuda=True):
                    world_sim3_camf, camk_sim3_camf = self.opt_pose_ray_dist_sim3(
                        Xf, Xk, world_sim3_camf, world_sim3_camk, Qk, valid_opt
                    )
            else:
                assert meas_k is not None
                assert valid_meas_k is not None
                assert K is not None
                with timed_section(profile, "optimize_ms", sync_cuda=True):
                    world_sim3_camf, camk_sim3_camf = self.opt_pose_calib_sim3(
                        Xf,
                        Xk,
                        world_sim3_camf,
                        world_sim3_camk,
                        Qk,
                        valid_opt,
                        meas_k,
                        valid_meas_k,
                        K,
                        img_size,
                    )
        except BeartypeException:
            raise
        except torch.linalg.LinAlgError:
            self.last_profile = profile
            rr.log("/world/logs", rr.TextLog(f"Cholesky failed {frame.frame_id}", level="WARN"))
            return False, [], True

        frame.world_sim3_cam = world_sim3_camf

        # Use pose to transform points to update keyframe
        with timed_section(profile, "keyframe_update_ms", sync_cuda=True):
            Xkk_raw = camk_sim3_camf.act(Xkf)
            assert isinstance(Xkk_raw, torch.Tensor)
            Xkk: Float[Tensor, "hw 3"] = Xkk_raw
            keyframe.update_pointmap(Xkk, Ckf)
            # write back the fitered pointmap
            self.keyframes[len(self.keyframes) - 1] = keyframe

        # Keyframe selection
        n_valid: Int[Tensor, ""] = valid_kf.sum()
        match_frac_k: Float[Tensor, ""] = n_valid / valid_kf.numel()
        unique_frac_f: float = (
            torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
        )

        new_kf: bool = bool(min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"])
        profile["match_frac_k"] = float(match_frac_k.item())
        profile["unique_frac_f"] = float(unique_frac_f)
        profile["opt_steps"] = self.last_opt_steps
        profile["new_keyframe"] = int(new_kf)

        # Rest idx if new keyframe
        if new_kf:
            self.reset_idx_f2k()

        self.last_profile = profile
        return (
            new_kf,
            [
                keyframe.X_canon,
                keyframe.get_average_conf(),
                frame.X_canon,
                frame.get_average_conf(),
                Qkf,
                Qff,
            ],
            False,
        )

    def get_points_poses(
        self,
        frame: Frame,
        keyframe: Frame,
        idx_f2k: Int[Tensor, "hw"],
        img_size: tuple[int, int],
        use_calib: bool,
        K: Float[Tensor, "3 3"] | None = None,
    ) -> tuple[
        Float[Tensor, "hw 3"],
        Float[Tensor, "hw 3"],
        lietorch.Sim3,
        lietorch.Sim3,
        Float[Tensor, "hw 1"],
        Float[Tensor, "hw 1"],
        Float[Tensor, "hw 3"] | None,
        Bool[Tensor, "hw 1"] | None,
    ]:
        """Extract matched points, poses, confidences, and optional pixel measurements.

        Args:
            frame: The current tracked frame.
            keyframe: The reference keyframe.
            idx_f2k: Linear correspondence indices from frame pixels to keyframe pixels.
            img_size: (height, width) of the image.
            use_calib: Whether calibrated mode is active.
            K: 3x3 intrinsic matrix (only used when ``use_calib`` is True).

        Returns:
            A tuple of (Xf, Xk, world_sim3_camf, world_sim3_camk, Cf, Ck, meas_k, valid_meas_k).
        """
        assert frame.X_canon is not None
        assert keyframe.X_canon is not None
        Xf: Float[Tensor, "hw 3"] = frame.X_canon
        Xk: Float[Tensor, "hw 3"] = keyframe.X_canon
        world_sim3_camf: lietorch.Sim3 = frame.world_sim3_cam
        world_sim3_camk: lietorch.Sim3 = keyframe.world_sim3_cam

        # Average confidence
        Cf_opt: Float[Tensor, "hw 1"] | None = frame.get_average_conf()
        Ck_opt: Float[Tensor, "hw 1"] | None = keyframe.get_average_conf()
        assert Cf_opt is not None
        assert Ck_opt is not None
        Cf: Float[Tensor, "hw 1"] = Cf_opt
        Ck: Float[Tensor, "hw 1"] = Ck_opt

        meas_k: Float[Tensor, "hw 3"] | None = None
        valid_meas_k: Bool[Tensor, "hw 1"] | None = None

        if use_calib:
            assert K is not None
            Xf = constrain_points_to_ray(img_size, Xf[None], K).squeeze(0)
            Xk = constrain_points_to_ray(img_size, Xk[None], K).squeeze(0)

            # Setup pixel coordinates
            uv_k: Float[Tensor, "hw 2"] = get_pixel_coords(1, img_size, device=Xf.device, dtype=Xf.dtype)
            uv_k = uv_k.view(-1, 2)
            meas_k = torch.cat((uv_k, torch.log(Xk[..., 2:3])), dim=-1)
            # Avoid any bad calcs in log
            valid_meas_k = Xk[..., 2:3] > self.cfg["depth_eps"]
            meas_k[~valid_meas_k.repeat(1, 3)] = 0.0

        return Xf[idx_f2k], Xk, world_sim3_camf, world_sim3_camk, Cf[idx_f2k], Ck, meas_k, valid_meas_k

    def solve(
        self,
        sqrt_info: Float[Tensor, "n r"],
        r: Float[Tensor, "n r"],
        J: Float[Tensor, "n r m"],
    ) -> tuple[Float[Tensor, "1 m"], float]:
        """Solve one Gauss-Newton step with Huber-weighted residuals.

        Args:
            sqrt_info: Square-root information (weighting) matrix diagonal.
            r: Residual vector.
            J: Jacobian of residuals w.r.t. the state.

        Returns:
            A tuple of (tau_j, cost) where tau_j is the update step and
            cost is the weighted squared cost.
        """
        whitened_r: Float[Tensor, "n r"] = sqrt_info * r
        robust_sqrt_info: Float[Tensor, "n r"] = sqrt_info * torch.sqrt(
            huber(whitened_r, k=self.cfg["huber"])
        )
        mdim: int = J.shape[-1]
        A: Float[Tensor, "N m"] = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # dr_dX
        b: Float[Tensor, "N 1"] = (robust_sqrt_info * r).view(-1, 1)  # z-h
        H: Float[Tensor, "m m"] = A.T @ A
        g: Float[Tensor, "m 1"] = -A.T @ b
        cost: float = 0.5 * (b.T @ b).item()

        L: Float[Tensor, "m m"] = torch.linalg.cholesky(H, upper=False)
        tau_j: Float[Tensor, "1 m"] = torch.cholesky_solve(g, L, upper=False).view(1, -1)

        return tau_j, cost

    def opt_pose_ray_dist_sim3(
        self,
        Xf: Float[Tensor, "hw 3"],
        Xk: Float[Tensor, "hw 3"],
        world_sim3_camf: lietorch.Sim3,
        world_sim3_camk: lietorch.Sim3,
        Qk: Float[Tensor, "hw 1"],
        valid: Bool[Tensor, "hw 1"],
    ) -> tuple[lietorch.Sim3, lietorch.Sim3]:
        """Optimise the relative Sim3 pose using ray-distance residuals.

        Args:
            Xf: Frame points in the frame's canonical coordinate system.
            Xk: Keyframe points in the keyframe's canonical coordinate system.
            world_sim3_camf: Current world-from-frame Sim3 pose.
            world_sim3_camk: World-from-keyframe Sim3 pose.
            Qk: Matching quality weights.
            valid: Boolean validity mask.

        Returns:
            A tuple of (world_sim3_camf, camk_sim3_camf) with the updated world-from-frame
            pose and the optimised relative pose.
        """
        last_error: float = float("inf")
        sqrt_info_ray: Float[Tensor, "hw 1"] = 1 / self.cfg["sigma_ray"] * valid * torch.sqrt(Qk)
        sqrt_info_dist: Float[Tensor, "hw 1"] = 1 / self.cfg["sigma_dist"] * valid * torch.sqrt(Qk)
        sqrt_info: Float[Tensor, "hw 4"] = torch.cat((sqrt_info_ray.repeat(1, 3), sqrt_info_dist), dim=1)

        # Solving for relative pose without scale!
        camk_sim3_camf_raw = world_sim3_camk.inv() * world_sim3_camf
        assert isinstance(camk_sim3_camf_raw, lietorch.Sim3)
        camk_sim3_camf: lietorch.Sim3 = camk_sim3_camf_raw

        # Precalculate distance and ray for obs k
        rd_k_result = point_to_ray_dist(Xk, jacobian=False)
        assert isinstance(rd_k_result, torch.Tensor)
        rd_k: Float[Tensor, "hw 4"] = rd_k_result

        old_cost: float = float("inf")
        for step in range(self.cfg["max_iters"]):
            act_result = act_Sim3(camk_sim3_camf, Xf, jacobian=True)
            assert isinstance(act_result, tuple)
            Xf_Ck: Float[Tensor, "hw 3"] = act_result[0]
            dXf_Ck_dcamk_sim3_camf: Float[Tensor, "hw 3 7"] = act_result[1]
            rd_result = point_to_ray_dist(Xf_Ck, jacobian=True)
            assert isinstance(rd_result, tuple)
            rd_f_Ck: Float[Tensor, "hw 4"] = rd_result[0]
            drd_f_Ck_dXf_Ck: Float[Tensor, "hw 4 3"] = rd_result[1]
            # r = z-h(x)
            r: Float[Tensor, "hw 4"] = rd_k - rd_f_Ck
            # Jacobian
            J: Float[Tensor, "hw 4 7"] = -drd_f_Ck_dXf_Ck @ dXf_Ck_dcamk_sim3_camf

            tau_ij_sim3: Float[Tensor, "1 7"]
            new_cost: float
            tau_ij_sim3, new_cost = self.solve(sqrt_info, r, J)
            last_error = new_cost
            camk_sim3_camf_new = camk_sim3_camf.retr(tau_ij_sim3)
            assert isinstance(camk_sim3_camf_new, lietorch.Sim3)
            camk_sim3_camf = camk_sim3_camf_new

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                rr.log("/world/logs", rr.TextLog(f"Max iters reached {last_error}", level="DEBUG"))

        self.last_opt_steps = step + 1
        # Assign new pose based on relative pose
        world_sim3_camf_new = world_sim3_camk * camk_sim3_camf
        assert isinstance(world_sim3_camf_new, lietorch.Sim3)

        return world_sim3_camf_new, camk_sim3_camf

    def opt_pose_calib_sim3(
        self,
        Xf: Float[Tensor, "hw 3"],
        Xk: Float[Tensor, "hw 3"],
        world_sim3_camf: lietorch.Sim3,
        world_sim3_camk: lietorch.Sim3,
        Qk: Float[Tensor, "hw 1"],
        valid: Bool[Tensor, "hw 1"],
        meas_k: Float[Tensor, "hw 3"],
        valid_meas_k: Bool[Tensor, "hw 1"],
        K: Float[Tensor, "3 3"],
        img_size: tuple[int, int],
    ) -> tuple[lietorch.Sim3, lietorch.Sim3]:
        """Optimise the relative Sim3 pose using calibrated reprojection residuals.

        Args:
            Xf: Frame points in frame canonical coordinates.
            Xk: Keyframe points in keyframe canonical coordinates.
            world_sim3_camf: Current world-from-frame Sim3 pose.
            world_sim3_camk: World-from-keyframe Sim3 pose.
            Qk: Matching quality weights.
            valid: Boolean validity mask.
            meas_k: Keyframe measurements (u, v, log_z).
            valid_meas_k: Validity mask for the keyframe measurements.
            K: 3x3 camera intrinsic matrix.
            img_size: (height, width) of the image.

        Returns:
            A tuple of (world_sim3_camf, camk_sim3_camf) with the updated world-from-frame
            pose and the optimised relative pose.
        """
        last_error: float = float("inf")
        sqrt_info_pixel: Float[Tensor, "hw 1"] = 1 / self.cfg["sigma_pixel"] * valid * torch.sqrt(Qk)
        sqrt_info_depth: Float[Tensor, "hw 1"] = 1 / self.cfg["sigma_depth"] * valid * torch.sqrt(Qk)
        sqrt_info: Float[Tensor, "hw 3"] = torch.cat((sqrt_info_pixel.repeat(1, 2), sqrt_info_depth), dim=1)

        # Solving for relative pose without scale!
        camk_sim3_camf_raw = world_sim3_camk.inv() * world_sim3_camf
        assert isinstance(camk_sim3_camf_raw, lietorch.Sim3)
        camk_sim3_camf: lietorch.Sim3 = camk_sim3_camf_raw

        old_cost: float = float("inf")
        for step in range(self.cfg["max_iters"]):
            act_result = act_Sim3(camk_sim3_camf, Xf, jacobian=True)
            assert isinstance(act_result, tuple)
            Xf_Ck: Float[Tensor, "hw 3"] = act_result[0]
            dXf_Ck_dcamk_sim3_camf: Float[Tensor, "hw 3 7"] = act_result[1]
            proj_result = project_calib(
                Xf_Ck,
                K,
                img_size,
                jacobian=True,
                border=self.cfg["pixel_border"],
                z_eps=self.cfg["depth_eps"],
            )
            assert len(proj_result) == 3
            pzf_Ck: Float[Tensor, "hw 3"] = proj_result[0]
            dpzf_Ck_dXf_Ck: Float[Tensor, "hw 3 3"] = proj_result[1]
            valid_proj: Bool[Tensor, "hw 1"] = proj_result[2]
            valid2: Bool[Tensor, "hw 1"] = valid_proj & valid_meas_k
            sqrt_info2: Float[Tensor, "hw 3"] = valid2 * sqrt_info

            # r = z-h(x)
            r: Float[Tensor, "hw 3"] = meas_k - pzf_Ck
            # Jacobian
            J: Float[Tensor, "hw 3 7"] = -dpzf_Ck_dXf_Ck @ dXf_Ck_dcamk_sim3_camf

            tau_ij_sim3: Float[Tensor, "1 7"]
            new_cost: float
            tau_ij_sim3, new_cost = self.solve(sqrt_info2, r, J)
            last_error = new_cost
            camk_sim3_camf_new = camk_sim3_camf.retr(tau_ij_sim3)
            assert isinstance(camk_sim3_camf_new, lietorch.Sim3)
            camk_sim3_camf = camk_sim3_camf_new

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                rr.log("/world/logs", rr.TextLog(f"Max iters reached {last_error}", level="DEBUG"))

        self.last_opt_steps = step + 1
        # Assign new pose based on relative pose
        world_sim3_camf_new = world_sim3_camk * camk_sim3_camf
        assert isinstance(world_sim3_camf_new, lietorch.Sim3)

        return world_sim3_camf_new, camk_sim3_camf
