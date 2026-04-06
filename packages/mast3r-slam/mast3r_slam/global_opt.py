import lietorch
import torch
from jaxtyping import Bool, Float, Float32, Int
from torch import Tensor

from mast3r_slam import _backends
from mast3r_slam.config import config
from mast3r_slam.frame import Frame, SharedKeyframes
from mast3r_slam.geometry import (
    constrain_points_to_ray,
)
from mast3r_slam.mast3r_utils import mast3r_match_symmetric


class FactorGraph:
    """Maintains pairwise factors between keyframes and solves global pose optimisation.

    Stores symmetric matching edges (correspondences, quality scores, validity
    masks) and provides Gauss-Newton solvers for both ray-distance and
    calibrated reprojection formulations.
    """

    def __init__(
        self,
        model: object,
        frames: SharedKeyframes,
        K: Float[Tensor, "3 3"] | None = None,
        device: str = "cuda",
    ) -> None:
        self.model: object = model
        self.frames: SharedKeyframes = frames
        self.device: str = device
        self.cfg: dict = config["local_opt"]
        self.ii: Int[Tensor, "..."] = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj: Int[Tensor, "..."] = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.idx_ii2jj: Int[Tensor, "..."] = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.idx_jj2ii: Int[Tensor, "..."] = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.valid_match_j: Bool[Tensor, "..."] = torch.as_tensor([], dtype=torch.bool, device=self.device)
        self.valid_match_i: Bool[Tensor, "..."] = torch.as_tensor([], dtype=torch.bool, device=self.device)
        self.Q_ii2jj: Float32[Tensor, "..."] = torch.as_tensor([], dtype=torch.float32, device=self.device)
        self.Q_jj2ii: Float32[Tensor, "..."] = torch.as_tensor([], dtype=torch.float32, device=self.device)
        self.window_size: float = self.cfg["window_size"]

        self.K: Float[Tensor, "3 3"] | None = K

    def add_factors(
        self,
        ii: list[int],
        jj: list[int],
        min_match_frac: float,
        is_reloc: bool = False,
    ) -> bool:
        """Add pairwise matching factors between keyframe pairs.

        Runs symmetric MASt3R matching between keyframes indexed by ``ii`` and
        ``jj``, filters by match quality, and appends valid edges to the graph.

        Args:
            ii: List of source keyframe indices.
            jj: List of target keyframe indices.
            min_match_frac: Minimum fraction of valid matches to accept an edge.
            is_reloc: If True, return False immediately on any invalid edge.

        Returns:
            True if at least one valid edge was added, False otherwise.
        """
        kf_ii: list[Frame] = [self.frames[idx] for idx in ii]
        kf_jj: list[Frame] = [self.frames[idx] for idx in jj]
        feat_i_list: list[Float[Tensor, "1 n_patches feat_dim"]] = []
        feat_j_list: list[Float[Tensor, "1 n_patches feat_dim"]] = []
        pos_i_list: list[Int[Tensor, "1 n_patches 2"]] = []
        pos_j_list: list[Int[Tensor, "1 n_patches 2"]] = []
        for kf_i in kf_ii:
            assert kf_i.feat is not None
            assert kf_i.pos is not None
            feat_i_list.append(kf_i.feat)
            pos_i_list.append(kf_i.pos)
        for kf_j in kf_jj:
            assert kf_j.feat is not None
            assert kf_j.pos is not None
            feat_j_list.append(kf_j.feat)
            pos_j_list.append(kf_j.pos)
        feat_i: Float[Tensor, "b n_patches feat_dim"] = torch.cat(feat_i_list)
        feat_j: Float[Tensor, "b n_patches feat_dim"] = torch.cat(feat_j_list)
        pos_i: Int[Tensor, "b n_patches 2"] = torch.cat(pos_i_list)
        pos_j: Int[Tensor, "b n_patches 2"] = torch.cat(pos_j_list)
        shape_i: list[Int[Tensor, "1 2"]] = [kf_i.img_true_shape for kf_i in kf_ii]
        shape_j: list[Int[Tensor, "1 2"]] = [kf_j.img_true_shape for kf_j in kf_jj]

        (
            idx_i2j,
            idx_j2i,
            valid_match_j,
            valid_match_i,
            Qii,
            Qjj,
            Qji,
            Qij,
        ) = mast3r_match_symmetric(
            self.model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
        )

        batch_inds: Int[Tensor, "b hw"] = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[
            :, None
        ].repeat(1, idx_i2j.shape[1])
        Qj: Float[Tensor, "b hw 1"] = torch.sqrt(Qii[batch_inds, idx_i2j] * Qji)
        Qi: Float[Tensor, "b hw 1"] = torch.sqrt(Qjj[batch_inds, idx_j2i] * Qij)

        valid_Qj: Bool[Tensor, "b hw 1"] = Qj > self.cfg["Q_conf"]
        valid_Qi: Bool[Tensor, "b hw 1"] = Qi > self.cfg["Q_conf"]
        valid_j: Bool[Tensor, "b hw 1"] = valid_match_j & valid_Qj
        valid_i: Bool[Tensor, "b hw 1"] = valid_match_i & valid_Qi
        nj: int = valid_j.shape[1] * valid_j.shape[2]
        ni: int = valid_i.shape[1] * valid_i.shape[2]
        match_frac_j: Float[Tensor, "b"] = valid_j.sum(dim=(1, 2)) / nj
        match_frac_i: Float[Tensor, "b"] = valid_i.sum(dim=(1, 2)) / ni

        ii_tensor: Int[Tensor, "b"] = torch.as_tensor(ii, device=self.device)
        jj_tensor: Int[Tensor, "b"] = torch.as_tensor(jj, device=self.device)

        # NOTE: Saying we need both edge directions to be above thrhreshold to accept either
        invalid_edges: Bool[Tensor, "b"] = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
        consecutive_edges: Bool[Tensor, "b"] = ii_tensor == (jj_tensor - 1)
        invalid_edges = (~consecutive_edges) & invalid_edges

        if invalid_edges.any() and is_reloc:
            return False

        valid_edges: Bool[Tensor, "b"] = ~invalid_edges
        ii_tensor = ii_tensor[valid_edges]
        jj_tensor = jj_tensor[valid_edges]
        idx_i2j = idx_i2j[valid_edges]
        idx_j2i = idx_j2i[valid_edges]
        valid_match_j = valid_match_j[valid_edges]
        valid_match_i = valid_match_i[valid_edges]
        Qj = Qj[valid_edges]
        Qi = Qi[valid_edges]

        self.ii = torch.cat([self.ii, ii_tensor])
        self.jj = torch.cat([self.jj, jj_tensor])
        self.idx_ii2jj = torch.cat([self.idx_ii2jj, idx_i2j])
        self.idx_jj2ii = torch.cat([self.idx_jj2ii, idx_j2i])
        self.valid_match_j = torch.cat([self.valid_match_j, valid_match_j])
        self.valid_match_i = torch.cat([self.valid_match_i, valid_match_i])
        self.Q_ii2jj = torch.cat([self.Q_ii2jj, Qj])
        self.Q_jj2ii = torch.cat([self.Q_jj2ii, Qi])

        added_new_edges: bool = bool(valid_edges.sum() > 0)
        return added_new_edges

    def get_unique_kf_idx(self) -> Int[Tensor, "n_unique"]:
        """Return sorted unique keyframe indices from all edges.

        Returns:
            1-D tensor of unique keyframe indices.
        """
        return torch.unique(torch.cat([self.ii, self.jj]), sorted=True)

    def prep_two_way_edges(
        self,
    ) -> tuple[
        Int[Tensor, "n_edges"],
        Int[Tensor, "n_edges"],
        Int[Tensor, "n_edges hw"],
        Bool[Tensor, "n_edges hw 1"],
        Float[Tensor, "n_edges hw 1"],
    ]:
        """Concatenate forward and backward edges into symmetric form.

        Returns:
            A tuple of (ii, jj, idx_ii2jj, valid_match, Q_ii2jj) with
            doubled edges (original + reversed).
        """
        ii: Int[Tensor, "n_edges"] = torch.cat((self.ii, self.jj), dim=0)
        jj: Int[Tensor, "n_edges"] = torch.cat((self.jj, self.ii), dim=0)
        idx_ii2jj: Int[Tensor, "n_edges hw"] = torch.cat((self.idx_ii2jj, self.idx_jj2ii), dim=0)
        valid_match: Bool[Tensor, "n_edges hw 1"] = torch.cat((self.valid_match_j, self.valid_match_i), dim=0)
        Q_ii2jj: Float[Tensor, "n_edges hw 1"] = torch.cat((self.Q_ii2jj, self.Q_jj2ii), dim=0)
        return ii, jj, idx_ii2jj, valid_match, Q_ii2jj

    def get_poses_points(
        self,
        unique_kf_idx: Int[Tensor, "n_unique"],
    ) -> tuple[
        Float[Tensor, "n_unique hw 3"],
        lietorch.Sim3,
        Float[Tensor, "n_unique hw 1"],
    ]:
        """Read poses, point maps, and confidences for a set of keyframe indices.

        Args:
            unique_kf_idx: Tensor of keyframe buffer indices.

        Returns:
            A tuple of (Xs, world_T_cams, Cs) with stacked point maps, Sim3 poses,
            and average confidence values.
        """
        kfs: list[Frame] = [self.frames[idx] for idx in unique_kf_idx]
        X_list: list[Float[Tensor, "hw 3"]] = []
        C_list: list[Float[Tensor, "hw 1"]] = []
        for kf in kfs:
            assert kf.X_canon is not None
            X_list.append(kf.X_canon)
            avg_conf: Float[Tensor, "hw 1"] | None = kf.get_average_conf()
            assert avg_conf is not None
            C_list.append(avg_conf)
        Xs: Float[Tensor, "n_unique hw 3"] = torch.stack(X_list)
        world_T_cams: lietorch.Sim3 = lietorch.Sim3(torch.stack([kf.world_T_cam.data for kf in kfs]))

        Cs: Float[Tensor, "n_unique hw 1"] = torch.stack(C_list)

        return Xs, world_T_cams, Cs

    def solve_GN_rays(self) -> None:
        """Run Gauss-Newton optimisation using ray-distance residuals.

        Pins the first ``pin`` keyframes and optimises the rest.  Writes
        the updated poses back into the shared keyframe buffer.
        """
        pin: int = self.cfg["pin"]
        unique_kf_idx: Int[Tensor, "n_unique"] = self.get_unique_kf_idx()
        n_unique_kf: int = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs: Float[Tensor, "n_unique hw 3"]
        world_T_cams: lietorch.Sim3
        Cs: Float[Tensor, "n_unique hw 1"]
        Xs, world_T_cams, Cs = self.get_poses_points(unique_kf_idx)

        ii: Int[Tensor, "n_edges"]
        jj: Int[Tensor, "n_edges"]
        idx_ii2jj: Int[Tensor, "n_edges hw"]
        valid_match: Bool[Tensor, "n_edges hw 1"]
        Q_ii2jj: Float[Tensor, "n_edges hw 1"]
        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh: float = self.cfg["C_conf"]
        Q_thresh: float = self.cfg["Q_conf"]
        max_iter: int = self.cfg["max_iters"]
        sigma_ray: float = self.cfg["sigma_ray"]
        sigma_dist: float = self.cfg["sigma_dist"]
        delta_thresh: float = self.cfg["delta_norm"]

        pose_data: Float[Tensor, "n_unique sim3_dim"] = world_T_cams.data[:, 0, :]
        _backends.gauss_newton_rays(
            pose_data,
            Xs,
            Cs,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            sigma_ray,
            sigma_dist,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # Update the keyframe world_T_cam
        self.frames.update_world_T_cams(world_T_cams[pin:], unique_kf_idx[pin:])

    def solve_GN_calib(self) -> None:
        """Run Gauss-Newton optimisation using calibrated reprojection residuals.

        Constrains points to calibrated rays, then pins the first ``pin``
        keyframes and optimises the rest.  Writes the updated poses back
        into the shared keyframe buffer.
        """
        assert self.K is not None
        K: Float[Tensor, "3 3"] = self.K
        pin: int = self.cfg["pin"]
        unique_kf_idx: Int[Tensor, "n_unique"] = self.get_unique_kf_idx()
        n_unique_kf: int = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs: Float[Tensor, "n_unique hw 3"]
        world_T_cams: lietorch.Sim3
        Cs: Float[Tensor, "n_unique hw 1"]
        Xs, world_T_cams, Cs = self.get_poses_points(unique_kf_idx)

        # Constrain points to ray
        img_size: tuple[int, int] = (int(self.frames[0].img.shape[-2]), int(self.frames[0].img.shape[-1]))
        Xs = constrain_points_to_ray(img_size, Xs, K)

        ii: Int[Tensor, "n_edges"]
        jj: Int[Tensor, "n_edges"]
        idx_ii2jj: Int[Tensor, "n_edges hw"]
        valid_match: Bool[Tensor, "n_edges hw 1"]
        Q_ii2jj: Float[Tensor, "n_edges hw 1"]
        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh: float = self.cfg["C_conf"]
        Q_thresh: float = self.cfg["Q_conf"]
        pixel_border: float = self.cfg["pixel_border"]
        z_eps: float = self.cfg["depth_eps"]
        max_iter: int = self.cfg["max_iters"]
        sigma_pixel: float = self.cfg["sigma_pixel"]
        sigma_depth: float = self.cfg["sigma_depth"]
        delta_thresh: float = self.cfg["delta_norm"]

        pose_data: Float[Tensor, "n_unique sim3_dim"] = world_T_cams.data[:, 0, :]

        img_size = (int(self.frames[0].img.shape[-2]), int(self.frames[0].img.shape[-1]))
        height: int
        width: int
        height, width = img_size

        _backends.gauss_newton_calib(
            pose_data,
            Xs,
            Cs,
            K,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            height,
            width,
            pixel_border,
            z_eps,
            sigma_pixel,
            sigma_depth,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # Update the keyframe world_T_cam
        self.frames.update_world_T_cams(world_T_cams[pin:], unique_kf_idx[pin:])
