from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

import numpy as np
from einops import rearrange, repeat
from huggingface_hub import hf_hub_download
from jaxtyping import Float, Float32, Float64, Int64
from numpy import ndarray
from serde.pickle import from_pickle

from simplecv.ops.mano.mano_utils import MANOData


def quat2mat(quat: Float[ndarray, "_ 4"]) -> Float[ndarray, "_ 3 3"]:
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: shape (b, 4) with ordering (w, x, y, z)
    Returns:
        Rotation matrices of shape (b, 3, 3)
    """
    quat = quat.astype(np.float32)
    # Match torch behavior: divide by norm without epsilon (zero-norm -> NaN); silence related warnings
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = quat / norm
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        w2, x2, y2, z2 = w * w, x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        rot = np.stack(
            [
                w2 + x2 - y2 - z2,
                2 * xy - 2 * wz,
                2 * wy + 2 * xz,
                2 * wz + 2 * xy,
                w2 - x2 + y2 - z2,
                2 * yz - 2 * wx,
                2 * xz - 2 * wy,
                2 * wx + 2 * yz,
                w2 - x2 - y2 + z2,
            ],
            axis=1,
        ).reshape((-1, 3, 3))
    return rot


def batch_rodrigues(axisang: Float[ndarray, "_ 3"]) -> Float[ndarray, "_ 9"]:
    aa = axisang.astype(np.float32)
    aa_norm = np.linalg.norm(aa + 1e-8, axis=1)
    angle = aa_norm[:, None]
    axis = aa / np.maximum(angle, 1e-8)
    angle_half = angle * 0.5
    v_cos = np.cos(angle_half)
    v_sin = np.sin(angle_half)
    quat = np.concatenate([v_cos, v_sin * axis], axis=1)
    rot = quat2mat(quat)
    return rot.reshape((rot.shape[0], 9))


def th_posemap_axisang(pose_vectors: Float[ndarray, "b betas=48"]) -> Float[ndarray, "b n_rotmat_flat=144"]:
    rot_nb = int(pose_vectors.shape[1] / 3)
    pose_vec_reshaped: Float[ndarray, "_ 3"] = pose_vectors.reshape((-1, 3)).astype(np.float32)
    rot_mats: Float[ndarray, "_ 9"] = batch_rodrigues(pose_vec_reshaped)
    rot_mats = rot_mats.reshape((pose_vectors.shape[0], rot_nb * 9))
    return rot_mats


def th_with_zeros(tensor: Float[ndarray, "b 3 4"]) -> Float[ndarray, "b 4 4"]:
    b = tensor.shape[0]
    padding = np.array([0.0, 0.0, 0.0, 1.0], dtype=tensor.dtype)
    return np.concatenate([tensor, padding.reshape(1, 1, 4).repeat(b, axis=0)], axis=1)


def subtract_flat_id(rot_mats: Float[ndarray, "b 144"]) -> Float[ndarray, "b 144"]:
    """Subtract flattened identity per 3x3 block to match torch logic exactly."""
    b, total = rot_mats.shape
    rot_nb = total // 9
    rm = rot_mats.reshape(b * rot_nb, 9)
    id9 = np.eye(3, dtype=rot_mats.dtype).reshape(1, 9).repeat(b * rot_nb, axis=0)
    out = rm - id9
    return out.reshape(b, total)


class ManoSimpleLayerNP:
    def __init__(
        self,
        ncomps: Literal[45] = 45,
        side: Literal["left", "right"] = "right",
        mano_root: Path = Path("mano/models"),
        use_pca: bool = True,
    ) -> None:
        self.side: Literal["left", "right"] = side
        self.use_pca: bool = use_pca
        self.ncomps = ncomps

        self.mano_path: Path = mano_root / f"MANO_{side.upper()}.pkl"
        with open(self.mano_path, "rb") as f:
            binary_data: bytes = f.read()
        mano_data: MANOData = from_pickle(MANOData, binary_data)

        hands_components: Float64[ndarray, "45 45"] = mano_data.hands_components

        self.th_shapedirs: Float32[ndarray, "n_verts=778 dim=3 n_betas=10"] = mano_data.shapedirs.astype(np.float32)
        self.th_posedirs: Float32[ndarray, "n_verts=778 dim=3 n_pose_dims=135"] = mano_data.posedirs.astype(np.float32)
        self.th_v_template: Float32[ndarray, "1 n_verts=778 dim=3"] = mano_data.v_template.astype(np.float32)[
            None, :, :
        ]
        self.th_J_regressor: Float32[ndarray, "n_joints=16 n_verts=778"] = mano_data.J_regressor.astype(np.float32)
        self.th_weights: Float32[ndarray, "n_verts=778 n_joints=16"] = mano_data.weights.astype(np.float32)
        self.th_faces: Int64[ndarray, "n_faces=1538 3"] = mano_data.f.astype(np.int64)
        self.n_verts: int = self.th_v_template.shape[1]

        self.th_hands_mean: Float32[ndarray, "1 45"] = mano_data.hands_mean.astype(np.float32)[None, :]
        self.th_comps: Float32[ndarray, "45 45"] = hands_components.astype(np.float32)

        self.kintree_table: Int64[ndarray, "2 16"] = mano_data.kintree_table
        parents: list[int] = list(self.kintree_table[0].tolist())
        self.kintree_parents: list[int] = parents

    def __call__(
        self,
        th_pose_coeffs: Float32[ndarray, "b n_poses=48"],
        th_betas: Float32[ndarray, "b n_betas=10"],
        th_trans: Float32[ndarray, "b dim=3"],
    ) -> tuple[Float32[ndarray, "b n_verts=778 3"], Float32[ndarray, "b joints_and_tips=21 3"]]:
        return self.forward(th_pose_coeffs, th_betas, th_trans)

    def forward(
        self,
        th_pose_coeffs: Float32[ndarray, "b n_poses=48"],
        th_betas: Float32[ndarray, "b n_betas=10"],
        th_trans: Float32[ndarray, "b dim=3"],
    ) -> tuple[Float32[ndarray, "b n_verts=778 3"], Float32[ndarray, "b joints_and_tips=21 3"]]:
        bsz: int = th_pose_coeffs.shape[0]

        # Step 1: PCA pose -> axis-angle
        if self.use_pca:
            hand_pose_coeffs: Float32[ndarray, "b 45"] = th_pose_coeffs[:, 3 : 3 + 45]
            full_hand_pose: Float32[ndarray, "b 45"] = hand_pose_coeffs @ self.th_comps
            full_pose: Float32[ndarray, "b 48"] = np.concatenate(
                [th_pose_coeffs[:, :3], self.th_hands_mean + full_hand_pose], axis=1
            )
        else:
            full_pose: Float32[ndarray, "b 48"] = th_pose_coeffs

        # Step 2: rotation matrices and pose maps
        rot_map_all: Float32[ndarray, "b 144"] = th_posemap_axisang(full_pose)
        pose_map_all: Float32[ndarray, "b 144"] = subtract_flat_id(rot_map_all)

        root_rot: Float32[ndarray, "b 3 3"] = rot_map_all[:, :9].reshape(bsz, 3, 3)
        rot_map: Float32[ndarray, "b 135"] = rot_map_all[:, 9:]
        pose_map: Float32[ndarray, "b 135"] = pose_map_all[:, 9:]

        # Step 3: Shape blend
        v_shaped_tmp: Float32[ndarray, "n_verts=778 dim=3 b"] = np.tensordot(
            self.th_shapedirs, th_betas.T, axes=([2], [0])
        )
        v_shaped: Float32[ndarray, "b n_verts=778 dim=3"] = (
            rearrange(v_shaped_tmp, "n_verts dim b -> b n_verts dim") + self.th_v_template
        )

        # Step 4: Joint regression
        j: Float32[ndarray, "b 16 3"] = np.einsum("jk,bkq->bjq", self.th_J_regressor, v_shaped)

        # Step 5: Pose-dependent corrective offsets
        pose_offsets_tmp: Float32[ndarray, "n_verts=778 dim=3 b"] = np.tensordot(
            self.th_posedirs, pose_map.T, axes=([2], [0])
        )
        v_posed: Float32[ndarray, "b n_verts=778 dim=3"] = v_shaped + rearrange(
            pose_offsets_tmp, "n_verts dim b -> b n_verts dim"
        )

        # Step 6: Forward kinematics
        root_j: Float32[ndarray, "b dim=3 1"] = j[:, 0, :].reshape(bsz, 3, 1)
        root_trans: Float32[ndarray, "b 4 4"] = th_with_zeros(np.concatenate([root_rot, root_j], axis=2))

        all_rots: Float32[ndarray, "b 15 3 3"] = rot_map.reshape(bsz, 15, 3, 3)
        lev1_idxs: list[int] = [1, 4, 7, 10, 13]
        lev2_idxs: list[int] = [2, 5, 8, 11, 14]
        lev3_idxs: list[int] = [3, 6, 9, 12, 15]

        lev1_rots: Float32[ndarray, "b 5 3 3"] = all_rots[:, [i - 1 for i in lev1_idxs]]
        lev2_rots: Float32[ndarray, "b 5 3 3"] = all_rots[:, [i - 1 for i in lev2_idxs]]
        lev3_rots: Float32[ndarray, "b 5 3 3"] = all_rots[:, [i - 1 for i in lev3_idxs]]
        lev1_j: Float32[ndarray, "b 5 3"] = j[:, lev1_idxs]
        lev2_j: Float32[ndarray, "b 5 3"] = j[:, lev2_idxs]
        lev3_j: Float32[ndarray, "b 5 3"] = j[:, lev3_idxs]

        all_transforms: list[Float32[ndarray, "b n_T 4 4"]] = [root_trans[:, None]]
        lev1_j_rel: Float32[ndarray, "b 5 3"] = lev1_j - rearrange(root_j, "b dim n -> b n dim")
        lev1_rel_transform_flt: Float32[ndarray, "_ 4 4"] = th_with_zeros(
            np.concatenate([lev1_rots, lev1_j_rel[:, :, :, None]], axis=3).reshape(-1, 3, 4)
        )
        root_trans_flt: Float32[ndarray, "_ 4 4"] = repeat(root_trans, "b m1 m2 -> (b f) m1 m2", f=5)
        lev1_flt: Float32[ndarray, "_ 4 4"] = root_trans_flt @ lev1_rel_transform_flt
        all_transforms.append(lev1_flt.reshape(bsz, 5, 4, 4))

        lev2_j_rel: Float32[ndarray, "b 5 3"] = lev2_j - lev1_j
        lev2_rel_transform_flt: Float32[ndarray, "_ 4 4"] = th_with_zeros(
            np.concatenate([lev2_rots, lev2_j_rel[:, :, :, None]], axis=3).reshape(-1, 3, 4)
        )
        lev2_flt: Float32[ndarray, "_ 4 4"] = lev1_flt @ lev2_rel_transform_flt
        all_transforms.append(lev2_flt.reshape(bsz, 5, 4, 4))

        lev3_j_rel: Float32[ndarray, "b 5 3"] = lev3_j - lev2_j
        lev3_rel_transform_flt: Float32[ndarray, "_ 4 4"] = th_with_zeros(
            np.concatenate([lev3_rots, lev3_j_rel[:, :, :, None]], axis=3).reshape(-1, 3, 4)
        )
        lev3_flt: Float32[ndarray, "_ 4 4"] = lev2_flt @ lev3_rel_transform_flt
        all_transforms.append(lev3_flt.reshape(bsz, 5, 4, 4))

        reorder_idxs: list[int] = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        results: Float32[ndarray, "b n_joints=16 4 4"] = np.concatenate(all_transforms, axis=1)[:, reorder_idxs]
        results_global: Float32[ndarray, "b n_joints=16 4 4"] = results

        # Step 7 – Linear Blend Skinning
        joint_js: Float32[ndarray, "b 16 4"] = np.concatenate([j, np.zeros((bsz, 16, 1), dtype=j.dtype)], axis=2)
        tmp2: Float32[ndarray, "b 16 4 1"] = results @ rearrange(joint_js, "b n m -> b n m 1")
        zeros_mat: Float32[ndarray, "b 16 4 3"] = np.zeros((bsz, 16, 4, 3), dtype=results.dtype)
        zeros_mat4: Float32[ndarray, "b 16 4 4"] = np.concatenate([zeros_mat, tmp2], axis=3)
        results2: Float32[ndarray, "b 4 4 16"] = rearrange((results - zeros_mat4), "b n m n2 -> b m n2 n")
        th_T: Float32[ndarray, "b 4 4 778"] = results2 @ self.th_weights.T

        rest_shape_h: Float32[ndarray, "b 4 778"] = np.concatenate(
            [v_posed.transpose(0, 2, 1), np.ones((bsz, 1, v_posed.shape[1]), dtype=th_T.dtype)], axis=1
        )
        rest_shape_h = rearrange(rest_shape_h, "b m n_verts -> b 1 m n_verts")
        verts_h: Float32[ndarray, "b 4 778"] = (th_T * rest_shape_h).sum(axis=2)
        verts: Float32[ndarray, "b 778 3"] = verts_h.transpose(0, 2, 1)[:, :, :3]

        # Step 8 – Fingertip pseudo-joints
        jtr: Float32[ndarray, "b 16 3"] = results_global[:, :, :3, 3]
        tips: Float32[ndarray, "b 5 3"] = verts[:, [745, 319, 444, 556, 673]]
        jtr = np.concatenate([jtr, tips], axis=1)

        # Step 9 – Re-order joints
        jtr = jtr[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]

        # Step 10 – Apply global translation
        jtr = jtr + rearrange(th_trans, "b d -> b 1 d")
        verts = verts + rearrange(th_trans, "b d -> b 1 d")

        # Step 11 – Convert m -> mm
        verts = verts * 1000.0
        jtr = jtr * 1000.0

        return verts.astype(np.float32), jtr.astype(np.float32)


class MANOLayerNP:
    """NumPy implementation mirroring MANOLayerTorch's interface (meters output)."""

    def __init__(
        self,
        side: Literal["left", "right"],
        betas: Float32[np.ndarray, "10"],
        mano_root_dir: Path | None = None,
        use_pca: bool = True,
    ) -> None:
        if mano_root_dir is None:
            repo_root: Path = Path(__file__).resolve().parents[2]
            data_dir: Path = repo_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            mano_filename: str = f"MANO_{side.upper()}.pkl"
            dest_pkl_path: Path = data_dir / mano_filename

            if not dest_pkl_path.exists():
                try:
                    downloaded_path: Path = Path(
                        hf_hub_download(
                            repo_id="pablovela5620/wilor-nano",
                            subfolder="pretrained_models/mano_clean",
                            filename=mano_filename,
                            local_dir=data_dir,
                            local_dir_use_symlinks=False,
                        )
                    )
                    if downloaded_path != dest_pkl_path:
                        shutil.copy2(downloaded_path, dest_pkl_path)
                except Exception as e:  # pragma: no cover - optional network
                    raise RuntimeError(
                        "Failed to download MANO model. Provide 'mano_root_dir' manually or ensure network access. "
                        f"Original error: {e}"
                    ) from e
            mano_root_dir = data_dir

        assert mano_root_dir.exists() and mano_root_dir.is_dir(), f"Invalid MANO root {mano_root_dir}"

        self._side: Literal["left", "right"] = side
        self._betas: Float32[np.ndarray, "10"] = betas
        self._use_pca: bool = use_pca

        self._mano_layer = ManoSimpleLayerNP(
            side=side,
            mano_root=mano_root_dir,
            ncomps=45,
            use_pca=use_pca,
        )

        # Store faces
        self.f: Int64[ndarray, "num_faces=1538 3"] = self._mano_layer.th_faces

        # Precompute root translation (for parity with torch layer; not used directly here)
        shapedirs: Float32[ndarray, "n_verts=778 3 10"] = self._mano_layer.th_shapedirs
        v_template: Float32[ndarray, "b=1 n_verts=778 dim=3"] = self._mano_layer.th_v_template
        v = np.tensordot(shapedirs, betas.reshape(1, -1).T, axes=([2], [0]))  # [778,3,1]
        v = rearrange(v, "n_verts dim b -> b n_verts dim") + v_template  # [1,778,3]
        j_regressor: Float32[ndarray, "16 n_verts=778"] = self._mano_layer.th_J_regressor
        # Ensure shape (1, 3) to match torch parity and annotations
        r: Float32[ndarray, "1 3"] = (j_regressor[0][None, :] @ v[0]).astype(np.float32)
        self.root_trans: Float32[ndarray, "1 3"] = r

    def __call__(
        self,
        poses: Float32[ndarray, "b n_poses=48"],
        translations: Float32[ndarray, "b dim=3"],
    ) -> tuple[Float32[ndarray, "b n_verts=778 dim=3"], Float32[ndarray, "b n_joints=21 dim=3"]]:
        return self.forward(poses, translations)

    def forward(
        self,
        poses: Float32[ndarray, "b n_poses=48"],
        translations: Float32[ndarray, "b dim=3"],
    ) -> tuple[Float32[ndarray, "b n_verts=778 dim=3"], Float32[ndarray, "b n_joints=21 dim=3"]]:
        bsz: int = poses.shape[0]
        verts_mm, joints_mm = self._mano_layer(
            poses, np.repeat(self._betas[None, :], bsz, axis=0), translations
        )
        # Convert to meters
        return verts_mm / 1000.0, joints_mm / 1000.0

    @property
    def th_hands_mean(self) -> ndarray:
        return self._mano_layer.th_hands_mean

    @property
    def th_selected_comps(self) -> ndarray:
        return self._mano_layer.th_comps

    @property
    def th_v_template(self) -> Float32[ndarray, "b=1 n_verts=778 dim=3"]:
        return self._mano_layer.th_v_template

    @property
    def side(self) -> Literal["left", "right"]:
        return self._side

    @property
    def num_verts(self) -> int:
        return 778
