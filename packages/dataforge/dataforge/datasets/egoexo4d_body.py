"""Ego-Exo4D-HM: per-take SMPL-H fits of the camera wearer, as the body_pose and body_mesh layers.

The release (``Ego-Exo4D-HM/npz-datasets``) ships one npz per take with SLAHMR's optimized parameters and the
67 OpenPose keypoints regressed from them. The model is SLAHMR's own: SMPL-H male with 16 shape
coefficients, and hands as 45 PCA coefficients per side in SMPL-X's MANO basis with the mean hand added
(``flat_hand_mean=False``). With it the regressed joints match the shipped ``joints3d`` to 1e-6 m, which
``test_smplh_reproduces_shipped_joints`` holds.
"""

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
import torch
from jaxtyping import Bool, Float32, Int8, Int64, UInt32
from numpy import ndarray
from serde import SerdeError, from_dict, serde
from simplecv.data.skeleton.coco_133 import LEFT_HAND_IDX, RIGHT_HAND_IDX
from simplecv.ops.smplx.smplx_torch import SmplxForwardResult

from dataforge import hands, meshes, schema
from dataforge.logging_toolkit import frame_index_column, time_column

HM_REPO: str = "Ego-Exo4D-HM/npz-datasets"
"""Hugging Face dataset of the release: ``<take_name>/<HM_FILE>``."""
HM_REVISION: str = "6e1cd8627a0c5751b02493a87172dd3709c0c652"
"""Pinned release commit (2,649 takes)."""
HM_FILE: str = "points_triangulated_world_results_merged.npz"
"""Basename of every take's fit."""
MODEL_REPO: str = "pablovela5620/mamma-streaming-data"
"""Private dataset holding the body models; contacted only when they are not already in place."""
MODEL_REVISION: str = "03a1f4b53f56eafd1002f70184014375edc84bfe"
SMPLH_FILE: str = "body_models/smplh/SMPLH_MALE.npz"
"""SMPL-H male, the AMASS "extended SMPL+H" release with 16 shape coefficients (the file SLAHMR ships)."""
SMPLX_FILE: str = "body_models/smplx/SMPLX_NEUTRAL.npz"
"""SMPL-X neutral; only its MANO hand PCA basis and mean are read."""
NUM_BETAS: int = 16
HAND_PCA_COMPONENTS: int = 45
MESH_BATCH_ROWS: int = 64
"""Mesh rows per chunk: one chunk is ~5 MB and carries its own copy of the 165 KB triangle list."""
SMPLH_VERTICES: int = 6890

OPENPOSE67_SOURCE: Int64[ndarray, "k"] = np.array(
    [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11, 19, 20, 21, 22, 23, 24, *range(25, 67)]
)
"""Release keypoints (OpenPose BODY_25, left hand 21, right hand 21) that have a COCO-133 slot; neck (1) and mid-hip (8) do not."""
OPENPOSE67_DESTINATION: Int64[ndarray, "k"] = np.array([*range(23), *LEFT_HAND_IDX, *RIGHT_HAND_IDX])
"""COCO-133 slot of each ``OPENPOSE67_SOURCE`` keypoint: body 0-16, feet 17-22, hands 91-132. The face (23-90) stays empty."""


@serde
@dataclass(frozen=True, slots=True)
class HmFit:
    """One take's fit, with the release's leading person axis (always 1) dropped."""

    trans: Float32[ndarray, "t 3"]
    """Root translation in the Ego-Exo4D world frame, metres (smplx ``transl``)."""
    root_orient: Float32[ndarray, "t 3"]
    """Root orientation, axis-angle."""
    pose_body: Float32[ndarray, "t 63"]
    """21 body joints, axis-angle."""
    hand_pose: Float32[ndarray, "t 90"]
    """45 PCA coefficients per hand, left then right (SMPL-X MANO basis); the release README calls them axis-angle."""
    betas: Float32[ndarray, "t 16"]
    """Shape coefficients, constant within each optimization chunk (``betas_per_frame``)."""
    joints3d: Float32[ndarray, "t 67 3"]
    """OpenPose BODY_25 + left hand + right hand keypoints regressed from the fit, world metres."""
    valid: Bool[ndarray, "t"]
    """False on frames whose optimization chunk hit a NaN/Inf loss."""
    chunk_ranges: Int64[ndarray, "c 2"]
    """Half-open frame ranges of the optimization chunks, tiling the take."""

    def __post_init__(self) -> None:
        count: int = len(self.trans)
        lengths: set[int] = {len(getattr(self, field.name)) for field in fields(self) if field.name != "chunk_ranges"}
        if lengths != {count}:
            raise ValueError(f"per-frame arrays disagree on the frame count: {sorted(lengths)}")
        starts: Int64[ndarray, "c"] = self.chunk_ranges[:, 0]
        stops: Int64[ndarray, "c"] = self.chunk_ranges[:, 1]
        if starts[0] != 0 or stops[-1] != count or not np.array_equal(starts[1:], stops[:-1]):
            raise ValueError(f"chunk_ranges {self.chunk_ranges.tolist()} do not tile [0, {count})")


def read_fit(path: Path) -> HmFit:
    """Decode one release npz; the per-camera arrays (undistorted pinhole views) and VPoser latents are not read."""
    with np.load(path, allow_pickle=False) as npz:
        valid: Int8[ndarray, "1 t"] = npz["valid"]
        values: dict[str, ndarray] = {
            "trans": npz["trans"][0],
            "root_orient": npz["root_orient"][0],
            "pose_body": npz["pose_body"][0],
            "hand_pose": npz["hand_pose"][0],
            "betas": npz["betas_per_frame"][0],
            "joints3d": npz["joints3d"][0],
            "valid": valid[0] == 1,
            "chunk_ranges": npz["chunk_ranges"],
        }
    try:
        return from_dict(HmFit, values)
    except (SerdeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error


def coco133_from_openpose67(joints: Float32[ndarray, "t 67 3"]) -> Float32[ndarray, "t 133 3"]:
    """Place the release keypoints in COCO-133 slots; slots with no source keypoint are NaN."""
    coco: Float32[ndarray, "t 133 3"] = np.full((len(joints), 133, 3), np.nan, dtype=np.float32)
    coco[:, OPENPOSE67_DESTINATION] = joints[:, OPENPOSE67_SOURCE]
    return coco


def fit_keypoints(fit: HmFit, frames: Int64[ndarray, "t"]) -> Float32[ndarray, "t 133 3"]:
    """Rows ``frames`` of the fit's keypoints as COCO-133; invalid frames are NaN."""
    keypoints: Float32[ndarray, "t 133 3"] = coco133_from_openpose67(fit.joints3d[frames])
    keypoints[~fit.valid[frames]] = np.nan
    return keypoints


class SmplhModel:
    """The SMPL-H male model SLAHMR fit, built from the two model files under ``model_root``."""

    def __init__(self, model_root: Path) -> None:
        import smplx
        from smplx.utils import Struct

        for name in (SMPLH_FILE, SMPLX_FILE):
            if not (model_root / name).is_file():
                raise FileNotFoundError(f"Ego-Exo4D-HM needs {model_root / name}; run dataforge-download egoexo4d or set --model-root")
        with np.load(model_root / SMPLH_FILE, allow_pickle=False) as smplh, np.load(model_root / SMPLX_FILE, allow_pickle=False) as smplx_model:
            parts: dict[str, ndarray] = {key: smplh[key] for key in smplh.files}
            for key in ("hands_componentsl", "hands_componentsr", "hands_meanl", "hands_meanr"):
                parts[key] = smplx_model[key]
        # smplx keeps only 10 betas unless the shape basis has SHAPE_SPACE_DIM (300) columns; SLAHMR pads it the same way.
        # Padding cannot stand in for a missing basis: the fit's extra betas would do nothing.
        shapedirs: ndarray = parts["shapedirs"]
        if shapedirs.shape[2] < NUM_BETAS:
            raise ValueError(f"{model_root / SMPLH_FILE}: {shapedirs.shape[2]} shape bases, the fit needs {NUM_BETAS}")
        padding: ndarray = np.zeros((*shapedirs.shape[:2], smplx.SMPLH.SHAPE_SPACE_DIM - shapedirs.shape[2]), dtype=shapedirs.dtype)
        parts["shapedirs"] = np.concatenate([shapedirs, padding], axis=-1)
        self.layer = smplx.SMPLH(
            str(model_root),
            data_struct=Struct(**parts),
            num_betas=NUM_BETAS,
            use_pca=True,
            num_pca_comps=HAND_PCA_COMPONENTS,
            flat_hand_mean=False,
        )
        self.faces: UInt32[ndarray, "13776 3"] = np.asarray(self.layer.faces, dtype=np.uint32)

    def forward(self, fit: HmFit, rows: Int64[ndarray, "k"]) -> SmplxForwardResult:
        """Pose ``rows`` of ``fit`` (one mesh batch), each with its own betas.

        Joints are SMPL-H's 52 plus smplx's 21 vertex-selector keypoints; both outputs are world metres.
        """
        inputs: dict[str, torch.Tensor] = {
            name: torch.from_numpy(np.ascontiguousarray(getattr(fit, name)[rows]))
            for name in ("betas", "root_orient", "pose_body", "hand_pose", "trans")
        }
        with torch.no_grad():
            output = self.layer(
                betas=inputs["betas"],
                global_orient=inputs["root_orient"],
                body_pose=inputs["pose_body"],
                left_hand_pose=inputs["hand_pose"][:, :HAND_PCA_COMPONENTS],
                right_hand_pose=inputs["hand_pose"][:, HAND_PCA_COMPONENTS:],
                transl=inputs["trans"],
            )
        return SmplxForwardResult(vertices=output.vertices.numpy(), joints=output.joints.numpy())


def write_body_pose(recording: rr.RecordingStream, fit: HmFit, times: Int64[ndarray, "t"], frames: Int64[ndarray, "t"]) -> None:
    """Rows ``frames`` of the fit, stamped ``times``: raw parameters at full rate (invalid frames included, flagged), and the keypoints as COCO-133 (invalid frames empty)."""
    path: str = schema.body_path("smplh")
    rr.log(
        path,
        rr.AnyValues(
            model="SMPL-H male, 16 betas",
            hand_pose_encoding=f"{HAND_PCA_COMPONENTS} PCA coefficients per hand (left, right), SMPL-X MANO basis, mean hand added",
            translation_pivot="root joint (smplx transl)",
            source=f"{HM_REPO}@{HM_REVISION[:8]}",
        ),
        static=True,
        recording=recording,
    )
    columns: dict[str, pa.FixedSizeListArray] = {}
    for name in ("trans", "root_orient", "pose_body", "hand_pose", "betas"):
        values: Float32[ndarray, "t d"] = getattr(fit, name)[frames]
        columns[name] = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])
    indexes: list[rr.TimeColumn] = [time_column(times), frame_index_column(frames)]
    rr.send_columns(path, indexes=indexes, columns=rr.AnyValues.columns(**columns), recording=recording)
    valid: Bool[ndarray, "t"] = fit.valid[frames]
    rr.send_columns(f"{path}/valid", indexes=indexes, columns=rr.Scalars.columns(scalars=valid.astype(np.float64)), recording=recording)
    hands.log_keypoints3d(recording, times_ns=times, frame_indices=frames, positions=fit_keypoints(fit, frames), confidence=None)


def write_body_mesh(recording: rr.RecordingStream, model: SmplhModel, fit: HmFit, times: Int64[ndarray, "t"], frames: Int64[ndarray, "t"]) -> None:
    """The posed mesh every ``meshes.BODY_MESH_STRIDE``-th frame and wherever validity changes; invalid frames are empty rows.

    The change rows keep a stale mesh from showing through an invalid stretch that starts or ends between two stride frames.
    """
    path: str = schema.body_path("mesh")
    meshes.log_mesh_static(recording, path, albedo_factor=hands.BODY_ALBEDO)
    valid: Bool[ndarray, "t"] = fit.valid[frames]
    changed: Bool[ndarray, "t"] = np.concatenate([[False], valid[1:] != valid[:-1]])
    kept: Int64[ndarray, "k"] = np.flatnonzero((frames % meshes.BODY_MESH_STRIDE == 0) | changed)
    for start in range(0, len(kept), MESH_BATCH_ROWS):
        rows: Int64[ndarray, "b"] = kept[start : start + MESH_BATCH_ROWS]
        trusted: Bool[ndarray, "b"] = valid[rows]
        meshes.log_mesh_batch(
            recording,
            path,
            times_ns=times[rows],
            frame_indices=frames[rows],
            vertices=model.forward(fit, frames[rows][trusted]).vertices if trusted.any() else np.empty((0, SMPLH_VERTICES, 3), dtype=np.float32),
            trusted=trusted.tolist(),
            topology=model.faces,
        )
