"""Torch SMPL/SMPL-X body layer wrapping the pip ``smplx`` package.

Mirrors ``simplecv.ops.mano.mano_torch.MANOLayerTorch``: betas are baked in at
construction, the forward is fully batched over frames, and the license-gated
model file is lazily downloaded when no explicit root is given (SMPL-X neutral
from the private HF dataset ``pablovela5620/mamma-streaming-data`` — requires
``hf auth login``; plain SMPL has no hosted copy and must be provided via
``model_root_dir``).
"""

import shutil
from pathlib import Path
from typing import Literal, NamedTuple, cast

import numpy as np
import torch
from beartype.roar import BeartypeException
from huggingface_hub import hf_hub_download
from jaxtyping import Float32, Int64
from numpy import ndarray
from torch import Tensor
from torch.nn import Module

SMPLX_HF_REPO_ID: str = "pablovela5620/mamma-streaming-data"
"""Private HF dataset hosting ``body_models/smplx/SMPLX_NEUTRAL.npz`` (SMPL-X v1.1)."""

# Standard SMPL-X 165-dim axis-angle full-pose layout (matches MAMMA pred/params NPZs).
_SMPLX_GLOBAL: slice = slice(0, 3)
_SMPLX_BODY: slice = slice(3, 66)
_SMPLX_JAW: slice = slice(66, 69)
_SMPLX_LEYE: slice = slice(69, 72)
_SMPLX_REYE: slice = slice(72, 75)
_SMPLX_LHAND: slice = slice(75, 120)
_SMPLX_RHAND: slice = slice(120, 165)

SMPLX_NUM_POSE: int = 165
SMPL_NUM_POSE: int = 72

# SMPL-X LBS materializes (frames, n_verts, 4, 4) skinning transforms, so a
# whole multi-thousand-frame sequence in one forward peaks at several GB. Chunk
# batched forwards to bound peak memory (EPFL sessions run to ~100k frames).
SMPLX_FORWARD_CHUNK_FRAMES: int = 512


class SmplxForwardResult(NamedTuple):
    """Host-numpy vertices and joints from a batched SMPL/SMPL-X forward."""

    vertices: Float32[ndarray, "n_frames n_verts 3"]
    joints: Float32[ndarray, "n_frames n_joints 3"]


class SmplxLayerTorch(Module):
    """Batched SMPL/SMPL-X forward returning world-space vertices and joints."""

    def __init__(
        self,
        *,
        betas: Float32[ndarray, "n_betas"],
        model_type: Literal["smplx", "smpl"] = "smplx",
        gender: str = "neutral",
        flat_hand_mean: bool = False,
        use_face_contour: bool = False,
        model_root_dir: Path | None = None,
        v_template: Float32[ndarray, "n_verts 3"] | None = None,
    ) -> None:
        """
        Args:
            betas: Shape coefficients baked into every forward (SMPL-X commonly 16, SMPL 10).
            model_type: Body model family; both are served by the ``smplx`` package.
            gender: Model gender ("neutral", "male", "female").
            flat_hand_mean: SMPL-X only — False means a zero hand pose rests at the
                MANO mean (the MAMMA convention), matching the ``smplx`` package flag.
            use_face_contour: SMPL-X only — also regress the 17 jawline contour
                landmarks (joints output grows from 127 to 144). Needed for the
                full COCO-133 face mapping.
            model_root_dir: Body-model root containing ``smplx/SMPLX_<GENDER>.npz``
                or ``smpl/SMPL_<GENDER>.{npz,pkl}``. If None, defaults to the module
                data dir ``simplecv/data/body_models`` and lazily downloads the
                SMPL-X neutral model from Hugging Face when missing.
            v_template: Optional subject-specific rest mesh replacing the model's
                mean template (MoSh ``mosh_v_template`` GT, e.g. MAMMA eval). Shape
                blendshapes still apply on top, so pair with zero ``betas``.
        """
        super().__init__()
        import smplx

        if model_root_dir is None:
            module_root: Path = Path(__file__).resolve().parents[2]
            model_root_dir = module_root / "data" / "body_models"

        ext: str = self._ensure_model_file(model_root_dir, model_type=model_type, gender=gender)
        create_kwargs: dict = dict(
            model_type=model_type,
            gender=gender,
            ext=ext,
            num_betas=int(betas.shape[0]),
            flat_hand_mean=flat_hand_mean,
            use_pca=False,
            use_face_contour=use_face_contour,
        )
        if v_template is not None:
            create_kwargs["v_template"] = np.ascontiguousarray(v_template, dtype=np.float32)
        # The smplx package's SMPL loader hardcodes a .pkl path + pickle.load and
        # ignores ``ext`` (unlike SMPLX, which honors npz), so a chumpy-free
        # SMPL .npz would never be found. Load it ourselves and hand smplx a
        # pre-built ``data_struct`` to bypass its file loader.
        if model_type == "smpl" and ext == "npz":
            from smplx.utils import Struct

            npz_path: Path = model_root_dir / f"smpl/SMPL_{gender.upper()}.npz"
            model_npz = np.load(npz_path, allow_pickle=True)
            create_kwargs["data_struct"] = Struct(**{key: model_npz[key] for key in model_npz.files})
        self.model = smplx.create(str(model_root_dir), **create_kwargs)
        self.model_type: Literal["smplx", "smpl"] = model_type
        betas_torch: Float32[Tensor, "1 n_betas"] = torch.from_numpy(np.ascontiguousarray(betas, dtype=np.float32)).reshape(1, -1)
        self.register_buffer("betas_buffer", betas_torch)
        # Cache faces on host before any .to(device) move.
        self._faces: Int64[ndarray, "n_faces 3"] = np.asarray(self.model.faces, dtype=np.int64)

    @staticmethod
    def _ensure_model_file(model_root_dir: Path, *, model_type: str, gender: str) -> str:
        """Resolve (downloading if needed) the model file; returns the smplx ``ext``."""
        model_stem: str = f"{model_type}/{model_type.upper()}_{gender.upper()}"
        # Prefer .npz: stock SMPL/SMPL-X .pkl releases embed chumpy arrays that
        # need chumpy (not in the env) to unpickle, so a dropped-in .pkl would
        # raise ModuleNotFoundError deep inside the smplx package.
        for ext in ("npz", "pkl"):
            if (model_root_dir / f"{model_stem}.{ext}").exists():
                return ext
        if model_type != "smplx":
            raise RuntimeError(
                f"No {model_type.upper()} model file at {model_root_dir / model_stem}.npz. "
                f"Download {model_type.upper()}_{gender.upper()} from https://smpl.is.tue.mpg.de/, convert it to a "
                f"chumpy-free .npz, and place it there (or pass 'model_root_dir')."
            )
        hf_filename: str = f"body_models/{model_stem}.npz"
        dest_path: Path = model_root_dir / f"{model_stem}.npz"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            downloaded_path: Path = Path(
                hf_hub_download(
                    repo_id=SMPLX_HF_REPO_ID,
                    repo_type="dataset",
                    filename=hf_filename,
                    local_dir=model_root_dir.parent,
                )
            )
            if downloaded_path != dest_path:
                shutil.copy2(downloaded_path, dest_path)
        except BeartypeException:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to download the SMPL-X model from {SMPLX_HF_REPO_ID} (license-gated; run 'hf auth login'). "
                "Provide 'model_root_dir' manually or download from https://smpl-x.is.tue.mpg.de/. Original error: " + str(e)
            ) from e
        return "npz"

    @property
    def faces(self) -> Int64[ndarray, "n_faces 3"]:
        """Triangle indices of the body mesh (host numpy)."""
        return self._faces

    def forward(
        self,
        poses: Float32[Tensor, "b n_pose"],
        transl: Float32[Tensor, "b 3"],
    ) -> tuple[Float32[Tensor, "b n_verts 3"], Float32[Tensor, "b n_joints 3"]]:
        """Run the batched body forward.

        Args:
            poses: Axis-angle full pose per frame — 165-dim for SMPL-X
                (global/body/jaw/eyes/hands), 72-dim for SMPL (global/body).
            transl: World translation per frame in meters.

        Returns:
            Vertices (10475 for SMPL-X, 6890 for SMPL) and regressed joints, meters.
        """
        batch_size: int = poses.shape[0]
        betas: Float32[Tensor, "b n_betas"] = cast(Tensor, self.betas_buffer).expand(batch_size, -1)
        if self.model_type == "smplx":
            assert poses.shape[1] == SMPLX_NUM_POSE, f"SMPL-X expects {SMPLX_NUM_POSE}-dim poses, got {poses.shape[1]}"
            expression: Float32[Tensor, "b n_expr"] = torch.zeros(
                (batch_size, self.model.num_expression_coeffs), dtype=poses.dtype, device=poses.device
            )
            output = self.model(
                betas=betas,
                global_orient=poses[:, _SMPLX_GLOBAL],
                body_pose=poses[:, _SMPLX_BODY],
                jaw_pose=poses[:, _SMPLX_JAW],
                leye_pose=poses[:, _SMPLX_LEYE],
                reye_pose=poses[:, _SMPLX_REYE],
                left_hand_pose=poses[:, _SMPLX_LHAND],
                right_hand_pose=poses[:, _SMPLX_RHAND],
                expression=expression,
                transl=transl,
            )
        else:
            assert poses.shape[1] == SMPL_NUM_POSE, f"SMPL expects {SMPL_NUM_POSE}-dim poses, got {poses.shape[1]}"
            output = self.model(
                betas=betas,
                global_orient=poses[:, 0:3],
                body_pose=poses[:, 3:SMPL_NUM_POSE],
                transl=transl,
            )
        vertices: Float32[Tensor, "b n_verts 3"] = output.vertices
        joints: Float32[Tensor, "b n_joints 3"] = output.joints
        return vertices, joints

    def forward_batched(
        self,
        poses: Float32[ndarray, "n_frames n_pose"],
        transl: Float32[ndarray, "n_frames 3"],
    ) -> SmplxForwardResult:
        """Forward a whole numpy sequence in frame chunks, returning host arrays.

        Chunks the batch by ``SMPLX_FORWARD_CHUNK_FRAMES`` to bound peak GPU
        memory, moves each chunk to the layer's device, and concatenates on the
        host. Returns both vertices and joints so callers that need only one
        avoid a second forward.

        Args:
            poses: Axis-angle full pose per frame (165 SMPL-X / 72 SMPL).
            transl: World translation per frame in meters.

        Returns:
            Vertices and joints for all frames, in meters, on the host.
        """
        device: torch.device = cast(Tensor, self.betas_buffer).device
        num_frames: int = poses.shape[0]
        verts_chunks: list[Float32[ndarray, "chunk n_verts 3"]] = []
        joints_chunks: list[Float32[ndarray, "chunk n_joints 3"]] = []
        for chunk_start in range(0, num_frames, SMPLX_FORWARD_CHUNK_FRAMES):
            chunk_end: int = min(chunk_start + SMPLX_FORWARD_CHUNK_FRAMES, num_frames)
            with torch.no_grad():
                chunk_output: tuple[Float32[Tensor, "chunk n_verts 3"], Float32[Tensor, "chunk n_joints 3"]] = self.forward(
                    torch.from_numpy(np.ascontiguousarray(poses[chunk_start:chunk_end])).float().to(device),
                    torch.from_numpy(np.ascontiguousarray(transl[chunk_start:chunk_end])).float().to(device),
                )
            verts_chunks.append(chunk_output[0].cpu().numpy())
            joints_chunks.append(chunk_output[1].cpu().numpy())
        return SmplxForwardResult(vertices=np.concatenate(verts_chunks, axis=0), joints=np.concatenate(joints_chunks, axis=0))

    def rest_root_joint(self) -> Float32[ndarray, "3"]:
        """Root (pelvis) joint position at zero pose and zero translation.

        Used to convert origin-pivot translations (EasyMocap convention:
        ``x = R @ lbs(pose) + Th``) into the smplx root-joint-pivot convention:
        ``transl = Th + (R - I) @ root_joint``.
        """
        n_pose: int = SMPLX_NUM_POSE if self.model_type == "smplx" else SMPL_NUM_POSE
        device: torch.device = cast(Tensor, self.betas_buffer).device
        with torch.no_grad():
            rest_output: tuple[Float32[Tensor, "1 n_verts 3"], Float32[Tensor, "1 n_joints 3"]] = self.forward(
                torch.zeros((1, n_pose), dtype=torch.float32, device=device),
                torch.zeros((1, 3), dtype=torch.float32, device=device),
            )
        root_joint: Float32[ndarray, "3"] = rest_output[1][0, 0].cpu().numpy()
        return root_joint
