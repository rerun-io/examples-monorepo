"""MAMMA SMPL-X bodies: the posed meshes and COCO-133 keypoints of every person's fit, and the model files they need.

The release's reference model is SMPL-X with the head bun removed ("locked head", 2023), whose shape basis differs from
v1.1's at every vertex. Every fit with betas is neutral, and those use the locked-head neutral. The gendered fits
(eval Singles) carry a subject template and zero betas, which make the shape basis and template of the model irrelevant:
those use SMPL-X v1.1 male and female from a public mirror. ``docs/mamma.md`` gives the measurements.
"""

import io
import pickle
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Float32, Int64, UInt32
from numpy import ndarray
from simplecv.ops.smplx.smplx_coco133 import smplx_joints_to_coco133_xyzc

from dataforge import meshes, transports
from dataforge.datasets.mamma_source import PersonFit, SmplxModel, flags
from dataforge.smpl import BODY_MODELS_REPO, BODY_MODELS_REVISION
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
MODEL_FILES: dict[SmplxModel, ModelFile] = {
    "locked_head_neutral": ModelFile(BODY_MODELS_REPO, "dataset", BODY_MODELS_REVISION, "body_models/smplx_lockedhead/SMPLX_NEUTRAL.npz", ""),
    "v1_1_male": ModelFile(V1_1_MIRROR, "model", V1_1_REVISION, "SMPLX_MALE.npz", "smplx_v1_1"),
    "v1_1_female": ModelFile(V1_1_MIRROR, "model", V1_1_REVISION, "SMPLX_FEMALE.npz", "smplx_v1_1"),
    "v1_1_neutral": ModelFile(V1_1_MIRROR, "model", V1_1_REVISION, "SMPLX_NEUTRAL.npz", "smplx_v1_1"),
}
"""Every model a MAMMA fit uses: the locked-head neutral (private repo; MPI neutral fits, MammaSyn renders) and v1.1 male,
female (MPI gendered template fits) and neutral (BEDLAM) from the public mirror."""
EXPRESSION_COEFFS: int = 10
"""Expression coefficients of the model; the release fits none, so they stay zero."""
FORWARD_BATCH: int = 256
"""Frames per SMPL-X forward."""


LANDMARK_FILE: ModelFile = ModelFile(BODY_MODELS_REPO, "dataset", BODY_MODELS_REVISION, "body_models/downsampled_verts/verts_512.pkl", "")
"""MammaNet's 512 surface landmarks (farthest-point samples of SMPL-X, MAMMA paper Fig. 3): a pickled torch [512, 10475]
sampling matrix, one 1 per row."""


class TensorUnpickler(pickle.Unpickler):
    """Unpickles a pickled torch tensor and nothing else: its storage bytes go through ``torch.load(weights_only=True)``."""

    def find_class(self, module: str, name: str) -> object:
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return torch._utils._rebuild_tensor_v2
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return lambda payload: torch.load(io.BytesIO(payload), weights_only=True)
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        raise pickle.UnpicklingError(f"the landmark pickle refers to {module}.{name}, which is not part of a torch tensor")


def landmark_vertices(model_root: Path) -> Int64[ndarray, "512"]:
    """MammaNet's landmarks as SMPL-X vertex ids, in its landmark order.

    Raises:
        FileNotFoundError: The file is not under ``model_root`` (``dataforge-download mamma`` fetches it).
        ValueError: A row of the matrix does not pick exactly one vertex.
    """
    with LANDMARK_FILE.local(model_root).open("rb") as handle:
        picks: Bool[ndarray, "512 10475"] = flags(TensorUnpickler(handle).load().numpy())
    if not (picks.sum(axis=1) == 1).all():
        raise ValueError(f"{LANDMARK_FILE.local(model_root)}: a row of the landmark matrix does not pick exactly one vertex")
    return picks.argmax(axis=1).astype(np.int64)


BODY_FILES: tuple[ModelFile, ...] = (*MODEL_FILES.values(), LANDMARK_FILE)
"""Every file posing a MAMMA body needs."""


def fetch_models(model_root: Path) -> int:
    """Fetch ``BODY_FILES`` under ``model_root`` unless already there with their pinned sha256; return how many were fetched."""
    fetched: int = 0
    for model in BODY_FILES:
        (file,) = transports.hf_lfs_files(model.repo, [model.path], revision=model.revision, repo_type=model.repo_type)
        fetched += transports.hf_fetch_verified(
            model.repo, file, local_dir=model_root / model.local_dir, revision=model.revision, repo_type=model.repo_type
        )
    return fetched


@dataclass(frozen=True, slots=True)
class PosedPerson:
    """One person's forward over a sequence's rows: keypoints at every row, meshes at the display rows."""

    index: int
    """The person's entity, ``person_NN`` (``PersonFit.index``)."""
    coco133: Float32[ndarray, "t 133 3"]
    """COCO-133 keypoints regressed from the posed model (all 133 mapped, face contour included), world metres; NaN where the
    person is absent."""
    joints: Float32[ndarray, "t 144 3"]
    """The model's joints (127, then the 17 face-contour points), world metres; NaN where absent."""
    mesh_rows: Int64[ndarray, "k"]
    """Rows of the mesh timeline: every ``mesh_stride``-th one, absent rows included (they clear the mesh)."""
    mesh_present: Bool[ndarray, "k"]
    """Whether the person is there at each mesh row."""
    vertices: Float32[ndarray, "m 10475 3"]
    """Posed vertices at the mesh rows where the person is present, world metres."""
    faces: UInt32[ndarray, "20908 3"]
    """The mesh triangles (one array every person shares)."""
    landmarks: Float32[ndarray, "t 512 3"]
    """The posed landmark vertices (``SmplxBody.landmark_ids``) at every row, world metres; NaN where the person is absent."""


class SmplxBody:
    """The ``BODY_FILES`` under ``model_root``; one smplx layer per person, built from the fit's own settings."""

    def __init__(self, model_root: Path) -> None:
        missing: list[str] = [str(model.local(model_root)) for model in BODY_FILES if not model.local(model_root).is_file()]
        if missing:
            raise FileNotFoundError(f"MAMMA needs the SMPL-X files {missing}; run dataforge-download mamma or set --model-root")
        self.model_root: Path = model_root
        self.landmark_ids: Int64[ndarray, "512"] = landmark_vertices(model_root)
        """MammaNet's landmarks as vertex ids; every pose keeps them at every row."""
        with np.load(MODEL_FILES["locked_head_neutral"].local(model_root), allow_pickle=False) as model:
            self.faces: UInt32[ndarray, "20908 3"] = model["f"].astype(np.uint32)  # every SMPL-X file shares this topology
        self.layers: dict[tuple[SmplxModel, bool, int], torch.nn.Module] = {}
        """Layers by (model, flat hand mean, betas): loading one costs 0.35 s, a 26-frame forward 0.02 s."""

    def layer(self, person: PersonFit) -> torch.nn.Module:
        """The smplx layer for ``person``'s settings; one of its own when it carries a template, which replaces the model's."""
        import smplx

        key: tuple[SmplxModel, bool, int] = (person.model, person.flat_hand_mean, len(person.betas))
        if person.v_template is None and key in self.layers:
            return self.layers[key]
        layer = smplx.create(
            str(MODEL_FILES[person.model].local(self.model_root)),
            model_type="smplx",
            ext="npz",
            use_pca=False,
            flat_hand_mean=person.flat_hand_mean,
            num_betas=len(person.betas),
            num_expression_coeffs=EXPRESSION_COEFFS,
            use_face_contour=True,
            v_template=None if person.v_template is None else torch.from_numpy(person.v_template.astype(np.float32)),
        )
        if person.v_template is None:
            self.layers[key] = layer
        return layer

    def pose(self, person: PersonFit, rows: Int64[ndarray, "t"], *, mesh_stride: int = meshes.BODY_MESH_STRIDE) -> PosedPerson:
        """Pose ``person`` at ``rows`` (indices into its arrays, from 0), keeping the meshes of every ``mesh_stride``-th row and the
        landmarks at every row."""
        layer: torch.nn.Module = self.layer(person)
        present: Bool[ndarray, "t"] = person.present[rows]
        posed: Int64[ndarray, "p"] = np.flatnonzero(present)
        mesh_rows: Int64[ndarray, "k"] = np.flatnonzero(rows % mesh_stride == 0)
        coco133: Float32[ndarray, "t 133 3"] = np.full((len(rows), 133, 3), np.nan, dtype=np.float32)
        joints: Float32[ndarray, "t 144 3"] = np.full((len(rows), 144, 3), np.nan, dtype=np.float32)
        vertices: list[Float32[ndarray, "c 10475 3"]] = []
        landmarks: Float32[ndarray, "t 512 3"] = np.full((len(rows), len(self.landmark_ids), 3), np.nan, dtype=np.float32)
        betas: torch.Tensor = torch.from_numpy(person.betas.astype(np.float32))[None]
        for start in range(0, len(posed), FORWARD_BATCH):
            batch: Int64[ndarray, "b"] = posed[start : start + FORWARD_BATCH]
            pose: torch.Tensor = torch.from_numpy(person.pose[rows[batch]].astype(np.float32))
            with torch.no_grad():
                output = layer(
                    betas=betas.expand(len(batch), -1),
                    global_orient=pose[:, 0:3],
                    body_pose=pose[:, 3:66],
                    jaw_pose=pose[:, 66:69],
                    leye_pose=pose[:, 69:72],
                    reye_pose=pose[:, 72:75],
                    left_hand_pose=pose[:, 75:120],
                    right_hand_pose=pose[:, 120:165],
                    transl=torch.from_numpy(person.trans[rows[batch]].astype(np.float32)),
                    expression=torch.zeros(len(batch), EXPRESSION_COEFFS),
                )
            joints[batch] = output.joints.numpy()
            coco133[batch] = smplx_joints_to_coco133_xyzc(output.joints.numpy())[..., :3]
            posed_vertices: Float32[ndarray, "b 10475 3"] = output.vertices.numpy()
            vertices.append(posed_vertices[np.isin(batch, mesh_rows)])
            landmarks[batch] = posed_vertices[:, self.landmark_ids]
        return PosedPerson(
            index=person.index,
            coco133=coco133,
            joints=joints,
            mesh_rows=mesh_rows,
            mesh_present=present[mesh_rows],
            vertices=np.concatenate(vertices) if vertices else np.zeros((0, 10475, 3), dtype=np.float32),
            faces=self.faces,
            landmarks=landmarks,
        )
