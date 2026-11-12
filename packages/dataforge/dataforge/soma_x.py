"""The ``soma_x`` layer every dataforge dataset shares: a SOMA-X fit at the XLO level of detail, drawn like EgoHand.

One writer for every body and hand source (SMPL, SMPL-H, MANO ...). Under a model root (``/world/gt/hands/<side>``,
``/world/gt/body``, ``/world/gt/people/person_NN/body``) it writes ``soma_x/``:

- ``mesh``: the XLO surface (612 body / 134 hand vertices), semi-transparent; triangles ride on the vertex chunks
  (``dataforge.meshes`` convention), so only the albedo is static.
- ``joints``: the SOMA-X joints in ``JOINT_COLOR`` (green); the ``AnnotationContext`` on ``soma_x`` links each joint to
  its parent in ``LINK_COLOR`` (white), the look of posekit's EgoHand demo.
- ``params``: what generates the mesh: the SOMA-X identity (static), the per-frame absolute local rotations and root
  translation, the backend, LOD and asset revision, and the per-frame fit error against the source mesh.
- ``source``: the source model's own parameters the fit started from, in one vocabulary across datasets
  (``betas``, ``global_orient``, ``hand_pose`` / ``body_pose``, ``transl``), with their encoding spelled out.

This module is pure Rerun and numpy: the fit itself needs CUDA and lives in ``dataforge.soma_x_fit``.
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float32, Int64, UInt32
from numpy import ndarray

from dataforge import meshes
from dataforge.logging_toolkit import frame_index_column, time_column

SOMA_X_REPO_ID: str = "nvidia/SOMA-X"
"""Hugging Face repository of the SOMA-X model files."""
SOMA_X_REVISION: str = "f81a01796f652260ff930d5318d6c6b845a505ed"
"""Pinned SOMA-X v0.3.0 asset revision (the one selfcap-soma fits with)."""
LOD: str = "xlo"
"""Level of detail of every logged mesh: the extra-low one, small enough to keep at full rate."""

Kind: TypeAlias = Literal["body", "hand"]
Rgba: TypeAlias = tuple[int, int, int, int]

MESH_ALPHA: int = 60
"""Mesh opacity (0-255), EgoHand's: the video and the joints show through."""
HAND_ALBEDO: dict[str, Rgba] = {"left": (66, 135, 245, MESH_ALPHA), "right": (245, 135, 66, MESH_ALPHA)}
"""EgoHand's per-side hand colours."""
BODY_ALBEDO: Rgba = (160, 190, 200, MESH_ALPHA)
"""Body colour: dataforge's neutral body tint at EgoHand's opacity; the fingers of a body mesh are not coloured apart."""
LINK_COLOR: tuple[int, int, int] = (255, 255, 255)
"""Skeleton links: white, as in EgoHand."""
JOINT_COLOR: tuple[int, int, int] = (0, 255, 0)
"""Joints: the green that simplecv's confidence colours give a fit (confidence 1.0)."""
JOINT_RADIUS: dict[Kind, float] = {"hand": 0.004, "body": 0.012}
"""Joint radius in metres."""
JOINT_CLASS: int = 0
"""Annotation class of the joints inside ``soma_x``'s own context."""
ROTATION_ENCODING: str = "axis-angle, absolute local joint rotations (SOMA pose input, joint-orient convention), one row per joint"


def soma_x_path(root: str) -> str:
    """``<root>/soma_x``: the SOMA-X fit of one body or hand; carries the annotation context of its joints."""
    return f"{root}/soma_x"


def mesh_path(root: str) -> str:
    """The XLO mesh."""
    return f"{soma_x_path(root)}/mesh"


def joints_path(root: str) -> str:
    """The SOMA-X joints."""
    return f"{soma_x_path(root)}/joints"


def params_path(root: str) -> str:
    """The SOMA-X parameters that generate the mesh."""
    return f"{soma_x_path(root)}/params"


def source_path(root: str) -> str:
    """The source model parameters the fit started from."""
    return f"{soma_x_path(root)}/source"


def rgba_u32(rgba: Rgba) -> int:
    """Rerun's packed ``0xRRGGBBAA`` colour."""
    red, green, blue, alpha = rgba
    return (red << 24) | (green << 16) | (blue << 8) | alpha


@dataclass(frozen=True, slots=True)
class SomaXTopology:
    """What a SOMA-X layer's mesh and skeleton look like; fixed per model and side."""

    faces: UInt32[ndarray, "f 3"]
    """XLO triangles."""
    parents: Int64[ndarray, "j"]
    """Parent joint of each joint; a root is its own parent."""
    joint_names: tuple[str, ...]
    """Name of each joint, in joint order."""

    def __post_init__(self) -> None:
        if len(self.joint_names) != len(self.parents):
            raise ValueError(f"{len(self.joint_names)} joint names for {len(self.parents)} joints")

    @property
    def links(self) -> list[tuple[int, int]]:
        """(parent, child) of every non-root joint."""
        return [(int(parent), child) for child, parent in enumerate(self.parents) if parent != child]


@dataclass(frozen=True, slots=True)
class SomaXFit:
    """A SOMA-X fit over every row of a segment; ``vertices`` and ``joints`` hold the trusted rows only."""

    times_ns: Int64[ndarray, "t"]
    """``video_time`` of each row."""
    frame_indices: Int64[ndarray, "t"]
    """Source frame index of each row."""
    trusted: Bool[ndarray, "t"]
    """Rows whose fit passed the error bar; the others log an empty mesh and no joints."""
    vertices: Float32[ndarray, "k v 3"]
    """XLO vertices of the trusted rows, world metres."""
    joints: Float32[ndarray, "k j 3"]
    """SOMA-X joints of the trusted rows, world metres."""
    rotations: Float32[ndarray, "t j 3"]
    """Fitted axis-angle rotations (``ROTATION_ENCODING``); NaN where nothing was fitted."""
    root_translation: Float32[ndarray, "t 3"]
    """Fitted root translation, world metres; NaN where nothing was fitted."""
    identity_coeffs: Float32[ndarray, "t c"]
    """SOMA-X identity of each row (the source shape in its backend); NaN where nothing was fitted."""
    fit_error_mm: Float32[ndarray, "t"]
    """Median per-vertex distance to the source mesh, mm; NaN where nothing was fitted."""

    def __post_init__(self) -> None:
        trusted: int = int(self.trusted.sum())
        if len(self.vertices) != trusted or len(self.joints) != trusted:
            raise ValueError(f"{len(self.vertices)} vertex rows and {len(self.joints)} joint rows for {trusted} trusted rows")
        rows: set[int] = {
            len(self.times_ns),
            len(self.frame_indices),
            len(self.trusted),
            len(self.rotations),
            len(self.root_translation),
            len(self.identity_coeffs),
            len(self.fit_error_mm),
        }
        if len(rows) != 1:
            raise ValueError(f"per-row arrays disagree on the row count: {sorted(rows)}")


@dataclass(frozen=True, slots=True)
class SomaXModel:
    """The SOMA-X model a fit used."""

    kind: Kind
    """A full body or one hand."""
    identity_model_type: str
    """SOMA-X identity backend (``mano``, ``smpl``, ``smplh`` ...): the source model's shape space."""
    max_median_error_mm: float
    """Error bar a row's fit must meet to be trusted."""


@dataclass(frozen=True, slots=True)
class SourceParameters:
    """The source model's parameters the fit started from, under the shared vocabulary."""

    model: str
    """Source model (``MANO``, ``SMPL neutral`` ...)."""
    encoding: dict[str, str]
    """How each field is encoded, by field name (``hand_pose``: ``45 PCA coefficients, hands_mean added``)."""
    static: dict[str, Float32[ndarray, "c"]]
    """Fields that do not change per row (``betas``)."""
    temporal: dict[str, Float32[ndarray, "t c"]]
    """Per-row fields (``global_orient``, ``hand_pose``, ``transl``); NaN rows where the source has none."""


def _fixed_rows(values: Float32[ndarray, "t c"]) -> pa.FixedSizeListArray:
    """One fixed-size list per row."""
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), values.shape[1])


def _send(recording: rr.RecordingStream, path: str, fit: SomaXFit, columns: rr.ComponentColumnList | list[rr.ComponentColumn]) -> None:
    rr.send_columns(path, indexes=[time_column(fit.times_ns), frame_index_column(fit.frame_indices)], columns=columns, recording=recording)


MESH_ROWS: dict[Kind, int] = {"hand": 256, "body": 64}
"""Mesh rows per chunk: ~0.4 MB of hand or ~0.5 MB of body vertices, so the viewer streams a long segment in small pieces."""


def window(fit: SomaXFit, rows: slice) -> SomaXFit:
    """The fit's rows ``rows`` (a contiguous slice), with the matching trusted vertex and joint rows."""
    before: int = int(fit.trusted[: rows.start].sum())
    count: int = int(fit.trusted[rows].sum())
    trusted_rows: slice = slice(before, before + count)
    return SomaXFit(
        times_ns=fit.times_ns[rows],
        frame_indices=fit.frame_indices[rows],
        trusted=fit.trusted[rows],
        vertices=fit.vertices[trusted_rows],
        joints=fit.joints[trusted_rows],
        rotations=fit.rotations[rows],
        root_translation=fit.root_translation[rows],
        identity_coeffs=fit.identity_coeffs[rows],
        fit_error_mm=fit.fit_error_mm[rows],
    )


def log_soma_x_static(
    recording: rr.RecordingStream, root: str, *, topology: SomaXTopology, model: SomaXModel, source: SourceParameters, albedo: Rgba
) -> None:
    """Everything of one body's or hand's ``soma_x`` subtree that never changes: once per segment, before ``log_soma_x_rows``."""
    meshes.log_mesh_static(recording, mesh_path(root), albedo_factor=albedo)
    rr.log(
        soma_x_path(root),
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=JOINT_CLASS, label=f"SOMA-X {model.kind} joints", color=LINK_COLOR),
                    keypoint_annotations=[rr.AnnotationInfo(id=index, label=name) for index, name in enumerate(topology.joint_names)],
                    keypoint_connections=topology.links,
                )
            ]
        ),
        static=True,
        recording=recording,
    )
    rr.log(
        joints_path(root),
        rr.Points3D.from_fields(
            class_ids=JOINT_CLASS,
            keypoint_ids=np.arange(len(topology.parents), dtype=np.uint16),
            colors=[JOINT_COLOR],
            radii=JOINT_RADIUS[model.kind],
            show_labels=False,  # EgoHand draws no joint names; they would cover the hand
        ),
        static=True,
        recording=recording,
    )
    rr.log(
        params_path(root),
        rr.AnyValues(
            kind=model.kind,
            identity_model_type=model.identity_model_type,
            lod=LOD,
            assets_repo=SOMA_X_REPO_ID,
            assets_revision=SOMA_X_REVISION,
            rotation_encoding=ROTATION_ENCODING,
            joint_names=pa.array(topology.joint_names),
            max_median_error_mm=model.max_median_error_mm,
        ),
        static=True,
        recording=recording,
    )
    fields: dict[str, str | pa.FixedSizeListArray] = {
        "model": source.model,
        **{f"{name}_encoding": text for name, text in source.encoding.items()},
        **{name: _fixed_rows(values[None]) for name, values in source.static.items()},
    }
    rr.log(
        source_path(root),
        rr.AnyValues(**fields),  # type: ignore[arg-type]  # pyrefly matches the unpacked str values to drop_untyped_nones
        static=True,
        recording=recording,
    )


def log_soma_x_rows(
    recording: rr.RecordingStream,
    root: str,
    *,
    fit: SomaXFit,
    topology: SomaXTopology,
    source_temporal: dict[str, Float32[ndarray, "t c"]],
    mesh_rows: int,
) -> None:
    """One window of rows: mesh chunks of ``mesh_rows`` rows (triangles on each chunk's first trusted row), joints, parameters."""
    trusted: list[bool] = fit.trusted.tolist()
    before: int = 0
    for start in range(0, len(trusted), mesh_rows):
        rows: list[bool] = trusted[start : start + mesh_rows]
        meshes.log_mesh_batch(
            recording,
            mesh_path(root),
            times_ns=fit.times_ns[start : start + mesh_rows],
            frame_indices=fit.frame_indices[start : start + mesh_rows],
            vertices=fit.vertices[before : before + sum(rows)],
            trusted=rows,
            topology=topology.faces,
        )
        before += sum(rows)
    _send(
        recording,
        joints_path(root),
        fit,
        rr.Points3D.columns(positions=fit.joints.reshape(-1, 3)).partition([len(topology.parents) if ok else 0 for ok in trusted]),
    )
    _send(
        recording,
        params_path(root),
        fit,
        rr.AnyValues.columns(
            rotations=_fixed_rows(fit.rotations.reshape(len(fit.rotations), -1)),
            root_translation=_fixed_rows(fit.root_translation),
            identity_coeffs=_fixed_rows(fit.identity_coeffs),
            fit_error_mm=pa.array(fit.fit_error_mm, type=pa.float32()),
        ),
    )
    if source_temporal:
        _send(recording, source_path(root), fit, rr.AnyValues.columns(**{name: _fixed_rows(values) for name, values in source_temporal.items()}))


def log_soma_x(
    recording: rr.RecordingStream,
    root: str,
    *,
    fit: SomaXFit,
    topology: SomaXTopology,
    model: SomaXModel,
    source: SourceParameters,
    albedo: Rgba,
) -> None:
    """A whole segment of one body or hand at once: ``log_soma_x_static`` then ``log_soma_x_rows`` over every row.

    Args:
        recording: Destination stream, a ``soma_x`` layer of the segment.
        root: Model root (``schema.hands_path(side)``, ``/world/gt/body``, ``schema.person_path(...)/body``).
        fit: The fit over every segment row.
        topology: XLO triangles and the skeleton.
        model: Backend and error bar.
        source: The source parameters the fit started from.
        albedo: Mesh RGBA (``HAND_ALBEDO[side]`` or ``BODY_ALBEDO``).
    """
    log_soma_x_static(recording, root, topology=topology, model=model, source=source, albedo=albedo)
    log_soma_x_rows(recording, root, fit=fit, topology=topology, source_temporal=source.temporal, mesh_rows=MESH_ROWS[model.kind])
