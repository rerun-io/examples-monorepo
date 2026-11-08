"""MoSh++ ground truth of the MammaEval sets, posed to world-frame SMPL-X joints and vertices.

The eval ``gt/global.npz`` holds, per person, a full 165-d axis-angle pose and a root translation per frame in the
world frame (metres), plus either a subject template (``mosh_v_template``: Singles, Dance; zero betas) or 300 betas
(``mosh_betas``: Extra, neutral). Hand poses are absolute (``flat_hand_mean=True``).

Model per fit, matching the release (``packages/dataforge/docs/mamma.md`` 'Body models'):

- neutral fits use the locked-head neutral SMPL-X (``smplx_lockedhead/SMPLX_NEUTRAL.npz``), whose shape basis differs from
  v1.1's at every vertex, so a betas fit must use it;
- gendered fits (Singles) use SMPL-X v1.1 male/female (``smplx_v1_1/SMPLX_{MALE,FEMALE}.npz``). A subject template
  replaces the model template and zero betas leave the shape basis unused, but the joint regressor, pose blendshapes
  and skinning weights still come from the model, so the gender is not cosmetic: see :func:`pose_mosh_person`'s
  ``variant`` override to measure it.

The pipeline's own fitter uses the v1.1 neutral (``smplx/SMPLX_NEUTRAL.npz``, 16 betas, ``flat_hand_mean=False``);
all variants share the 55-joint kinematic tree and the 10475-vertex topology, so joints[:55] and vertices compare
index for index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import torch
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray

SmplxVariant: TypeAlias = Literal["locked_head_neutral", "v1_1_neutral", "v1_1_male", "v1_1_female"]
"""SMPL-X model files a MoSh fit can belong to."""
FitMethod: TypeAlias = Literal["mosh_v_template", "mosh_betas"]
"""How the eval GT encodes body shape."""

VARIANT_FILES: dict[SmplxVariant, str] = {
    "locked_head_neutral": "smplx_lockedhead/SMPLX_NEUTRAL.npz",
    "v1_1_neutral": "smplx_v1_1/SMPLX_NEUTRAL.npz",
    "v1_1_male": "smplx_v1_1/SMPLX_MALE.npz",
    "v1_1_female": "smplx_v1_1/SMPLX_FEMALE.npz",
}
"""Model file of each variant, relative to the body-models root (``data/body_models``)."""
GENDER_VARIANTS: dict[str, SmplxVariant] = {"neutral": "locked_head_neutral", "male": "v1_1_male", "female": "v1_1_female"}
"""The model each gender's MoSh fits use in the MPI release."""
NUM_KINEMATIC_JOINTS: int = 55
"""SMPL-X kinematic-tree joints scored by MPJPE (body 22 + jaw + 2 eyes + 2x15 hand)."""
NUM_VERTICES: int = 10475
"""SMPL-X vertices scored by PVE."""
FORWARD_BATCH: int = 128
"""Frames per SMPL-X forward."""


@dataclass(frozen=True, slots=True)
class MoshPerson:
    """One person's MoSh++ fit over the whole sequence."""

    subject: str
    """Eval subject id (``subject_name``)."""
    gender: str
    """Gender as the GT states it (``male`` / ``female`` / ``neutral``)."""
    flat_hand_mean: bool
    """True when hand poses are absolute (no MANO mean added)."""
    pose: Float64[ndarray, "f 165"]
    """Axis-angle pose: root, 21 body, jaw, leye, reye, 15 left hand, 15 right hand."""
    trans: Float64[ndarray, "f 3"]
    """World-frame root translation (smplx ``transl``), metres."""
    betas: Float64[ndarray, "nb"]
    """Shape coefficients (300; all zero for template fits)."""
    v_template: Float64[ndarray, "10475 3"] | None
    """Registered subject template replacing the model's; None for betas fits."""

    @property
    def variant(self) -> SmplxVariant:
        """The SMPL-X file this fit was made with."""
        return GENDER_VARIANTS[self.gender]


@dataclass(frozen=True, slots=True)
class MoshSequence:
    """All MoSh++ fits of one eval sequence."""

    name: str
    """Sequence name (``seq_name``)."""
    fps: float
    """Frame rate."""
    frames_len: int
    """Number of GT frames; GT row ``i`` is video frame ``i``."""
    method: FitMethod
    """Shape encoding (``method_used``)."""
    people: list[MoshPerson]
    """Fits in ``subject_name`` order."""


def load_mosh_gt(global_npz: Path) -> MoshSequence:
    """Read an eval ``gt/global.npz`` (raw release or a prepared copy).

    Args:
        global_npz: Path to ``gt/global.npz``.

    Raises:
        ValueError: The file is not a MoSh++ eval GT this module knows how to pose.
    """
    with np.load(global_npz, allow_pickle=False) as npz:
        method_used: str = str(npz["method_used"])
        if method_used not in ("mosh_v_template", "mosh_betas"):
            raise ValueError(f"{global_npz}: unsupported method_used {method_used!r}")
        method: FitMethod = "mosh_v_template" if method_used == "mosh_v_template" else "mosh_betas"
        templates: Float64[ndarray, "p 10475 3"] | None = npz["v_template"].astype(np.float64) if "v_template" in npz.files else None
        people: list[MoshPerson] = []
        for index, subject in enumerate(npz["subject_name"]):
            gender: str = str(npz["gender"][index])
            if gender not in GENDER_VARIANTS:
                raise ValueError(f"{global_npz}: unknown gender {gender!r}")
            people.append(
                MoshPerson(
                    subject=str(subject),
                    gender=gender,
                    flat_hand_mean=bool(npz["flat_hand_mean"]),
                    pose=npz["pose_world"][:, index].astype(np.float64),
                    trans=npz["pose_trans_world"][:, index].astype(np.float64),
                    betas=npz["shape"][index].astype(np.float64),
                    v_template=None if templates is None else templates[index],
                )
            )
        return MoshSequence(name=str(npz["seq_name"]), fps=float(npz["fps"]), frames_len=int(npz["frames_len"]), method=method, people=people)


@dataclass(frozen=True, slots=True)
class PosedBody:
    """World-frame SMPL-X output at a set of frames."""

    frames: Int64[ndarray, "t"]
    """GT rows (= video frame indices) posed."""
    joints: Float32[ndarray, "t 55 3"]
    """Kinematic joints, world metres."""
    vertices: Float32[ndarray, "t 10475 3"]
    """Mesh vertices, world metres."""


def build_mosh_layer(person: MoshPerson, body_models_dir: Path, variant: SmplxVariant | None = None, device: str = "cpu"):
    """The smplx layer that reproduces ``person``'s fit.

    Args:
        person: The fit.
        body_models_dir: Body-models root holding ``VARIANT_FILES``.
        variant: Override the model file (diagnostics only); default is the one the fit was made with.
        device: Torch device.
    """
    import smplx

    model_path: Path = body_models_dir / VARIANT_FILES[variant or person.variant]
    if not model_path.is_file():
        raise FileNotFoundError(f"missing SMPL-X model {model_path} (v1.1: /mnt/nas/datasets/models_smplx_v1_1/models/smplx)")
    v_template: torch.Tensor | None = None if person.v_template is None else torch.from_numpy(person.v_template.astype(np.float32))
    return smplx.create(
        str(model_path),
        model_type="smplx",
        ext="npz",
        use_pca=False,
        flat_hand_mean=person.flat_hand_mean,
        num_betas=len(person.betas),
        num_expression_coeffs=10,
        v_template=v_template,
    ).to(device)


def pose_mosh_person(
    person: MoshPerson,
    frames: Int64[ndarray, "t"],
    body_models_dir: Path,
    variant: SmplxVariant | None = None,
    device: str = "cpu",
) -> PosedBody:
    """Forward ``person`` through SMPL-X at GT rows ``frames``.

    Args:
        person: The fit.
        frames: GT rows to pose.
        body_models_dir: Body-models root holding ``VARIANT_FILES``.
        variant: Override the model file (diagnostics only).
        device: Torch device.
    """
    layer = build_mosh_layer(person, body_models_dir, variant=variant, device=device)
    joints: Float32[ndarray, "t 55 3"] = np.zeros((len(frames), NUM_KINEMATIC_JOINTS, 3), dtype=np.float32)
    vertices: Float32[ndarray, "t 10475 3"] = np.zeros((len(frames), NUM_VERTICES, 3), dtype=np.float32)
    betas: Float32[torch.Tensor, "1 nb"] = torch.from_numpy(person.betas.astype(np.float32))[None].to(device)
    for start in range(0, len(frames), FORWARD_BATCH):
        rows: Int64[ndarray, "b"] = frames[start : start + FORWARD_BATCH]
        pose: Float32[torch.Tensor, "b 165"] = torch.from_numpy(person.pose[rows].astype(np.float32)).to(device)
        transl: Float32[torch.Tensor, "b 3"] = torch.from_numpy(person.trans[rows].astype(np.float32)).to(device)
        n: int = len(rows)
        with torch.no_grad():
            out = layer(
                betas=betas.expand(n, -1),
                global_orient=pose[:, 0:3],
                body_pose=pose[:, 3:66],
                jaw_pose=pose[:, 66:69],
                leye_pose=pose[:, 69:72],
                reye_pose=pose[:, 72:75],
                left_hand_pose=pose[:, 75:120],
                right_hand_pose=pose[:, 120:165],
                transl=transl,
                expression=torch.zeros(n, 10, device=device),
            )
        joints[start : start + n] = out.joints[:, :NUM_KINEMATIC_JOINTS].cpu().numpy()
        vertices[start : start + n] = out.vertices.cpu().numpy()
    return PosedBody(frames=frames.astype(np.int64), joints=joints, vertices=vertices)
