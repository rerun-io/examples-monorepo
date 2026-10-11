"""MAMMA SMPL-X bodies: the posed meshes and COCO-133 keypoints of every person's fit, and the model files they need.

The release's reference model is SMPL-X with the head bun removed ("locked head", 2023), whose shape basis differs from
v1.1's at every vertex. Every fit with betas is neutral, and those use the locked-head neutral. The gendered fits
(eval Singles) carry a subject template and zero betas, which make the shape basis and template of the model irrelevant:
those use SMPL-X v1.1 male and female from a public mirror. ``docs/mamma.md`` gives the measurements.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float32, Int64, UInt32
from numpy import ndarray
from simplecv.ops.smplx.smplx_coco133 import smplx_joints_to_coco133_xyzc

from dataforge import meshes, transports
from dataforge.datasets.mamma_source import Gender, PersonFit
from dataforge.transports import HfRepoType


@dataclass(frozen=True, slots=True)
class ModelFile:
    """One SMPL-X model file pinned on the Hub."""

    repo: str
    """Hub repo holding it."""
    repo_type: HfRepoType
    revision: str
    """Pinned commit."""
    path: str
    """Repo-relative path."""
    local_dir: str
    """Directory under the model root that ``path`` lands in."""

    def local(self, model_root: Path) -> Path:
        return model_root / self.local_dir / self.path


V1_1_MIRROR: str = "camenduru/SMPLer-X"
"""Public model repo with SMPL-X v1.1 for all three genders at its root."""
V1_1_REVISION: str = "9e5548b70b48efe6992faa3b2a418df108542bf7"
MODEL_FILES: dict[Gender, ModelFile] = {
    "neutral": ModelFile(
        "pablovela5620/mamma-streaming-data",
        "dataset",
        "0a961eedf02e2fb3d32929b11cef625661dffc60",
        "body_models/smplx_lockedhead/SMPLX_NEUTRAL.npz",
        "",
    ),
    "male": ModelFile(V1_1_MIRROR, "model", V1_1_REVISION, "SMPLX_MALE.npz", "smplx_v1_1"),
    "female": ModelFile(V1_1_MIRROR, "model", V1_1_REVISION, "SMPLX_FEMALE.npz", "smplx_v1_1"),
}
"""The model each gender's fits use: the locked-head neutral (private repo), v1.1 male and female (public mirror)."""
EXPRESSION_COEFFS: int = 10
"""Expression coefficients of the model; the release fits none, so they stay zero."""
FORWARD_BATCH: int = 256
"""Frames per SMPL-X forward."""


def fetch_models(model_root: Path) -> int:
    """Fetch the three model files under ``model_root`` unless already there with their pinned sha256; return how many were fetched."""
    fetched: int = 0
    for model in MODEL_FILES.values():
        (file,) = transports.hf_lfs_files(model.repo, [model.path], revision=model.revision, repo_type=model.repo_type)
        fetched += transports.hf_fetch_verified(
            model.repo, file, local_dir=model_root / model.local_dir, revision=model.revision, repo_type=model.repo_type
        )
    return fetched


@dataclass(frozen=True, slots=True)
class PosedPerson:
    """One person's forward over a sequence: keypoints at every frame, meshes at the display rows."""

    coco133: Float32[ndarray, "t 133 3"]
    """COCO-133 keypoints regressed from the posed model (all 133 mapped, face contour included), world metres."""
    mesh_frames: Int64[ndarray, "k"]
    """Frames of ``vertices``: every ``meshes.BODY_MESH_STRIDE``-th one."""
    vertices: Float32[ndarray, "k 10475 3"]
    """Posed vertices at ``mesh_frames``, world metres."""
    faces: UInt32[ndarray, "20908 3"]
    """The mesh triangles (one array every person shares)."""


class SmplxBody:
    """The ``MODEL_FILES`` under ``model_root``; one smplx layer per person, built from the fit's own settings."""

    def __init__(self, model_root: Path) -> None:
        missing: list[str] = [str(model.local(model_root)) for model in MODEL_FILES.values() if not model.local(model_root).is_file()]
        if missing:
            raise FileNotFoundError(f"MAMMA needs the SMPL-X files {missing}; run dataforge-download mamma or set --model-root")
        self.model_root: Path = model_root
        with np.load(MODEL_FILES["neutral"].local(model_root), allow_pickle=False) as model:
            self.faces: UInt32[ndarray, "20908 3"] = model["f"].astype(np.uint32)  # every SMPL-X file shares this topology

    def pose(self, person: PersonFit, frames: Int64[ndarray, "t"]) -> PosedPerson:
        """Pose ``person`` at ``frames`` (sequence frame indices), keeping the meshes of the stride frames."""
        import smplx

        layer = smplx.create(
            str(MODEL_FILES[person.gender].local(self.model_root)),
            model_type="smplx",
            gender=person.gender,
            ext="npz",
            use_pca=False,
            flat_hand_mean=person.flat_hand_mean,
            num_betas=len(person.betas),
            num_expression_coeffs=EXPRESSION_COEFFS,
            use_face_contour=True,
            v_template=None if person.v_template is None else torch.from_numpy(person.v_template.astype(np.float32)),
        )
        kept: Int64[ndarray, "k"] = np.flatnonzero(frames % meshes.BODY_MESH_STRIDE == 0)
        coco133: list[Float32[ndarray, "b 133 3"]] = []
        vertices: list[Float32[ndarray, "c 10475 3"]] = []
        betas: torch.Tensor = torch.from_numpy(person.betas.astype(np.float32))[None]
        for start in range(0, len(frames), FORWARD_BATCH):
            rows: Int64[ndarray, "b"] = frames[start : start + FORWARD_BATCH]
            pose: torch.Tensor = torch.from_numpy(person.pose[rows].astype(np.float32))
            with torch.no_grad():
                output = layer(
                    betas=betas.expand(len(rows), -1),
                    global_orient=pose[:, 0:3],
                    body_pose=pose[:, 3:66],
                    jaw_pose=pose[:, 66:69],
                    leye_pose=pose[:, 69:72],
                    reye_pose=pose[:, 72:75],
                    left_hand_pose=pose[:, 75:120],
                    right_hand_pose=pose[:, 120:165],
                    transl=torch.from_numpy(person.trans[rows].astype(np.float32)),
                    expression=torch.zeros(len(rows), EXPRESSION_COEFFS),
                )
            coco133.append(smplx_joints_to_coco133_xyzc(output.joints.numpy())[..., :3])
            in_batch: Int64[ndarray, "c"] = kept[(kept >= start) & (kept < start + len(rows))] - start
            vertices.append(output.vertices.numpy()[in_batch])
        return PosedPerson(coco133=np.concatenate(coco133), mesh_frames=frames[kept], vertices=np.concatenate(vertices), faces=self.faces)
