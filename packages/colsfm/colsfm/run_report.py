"""What a finished run says about itself: `summary.json` and `loop_edges.json`.

The reporting layer. `PipelineSummary` is the run's own account of what it
produced and how long each stage took, written once at the end from the stage
ledger; `MappingStats` freezes the reconstruction counters at that moment,
because `MappingResult` computes them from a live model that later passes adjust.
`LoopEdgeRecord` is a pose-graph loop constraint flattened to plain numbers, so a
viewer can draw the loops without pycolmap.

Separate from the runner because these are the *contract with a reader* -- a
benchmark, a viewer, a report -- rather than anything the run needs to execute.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
from jaxtyping import Float, Float64
from numpy import ndarray
from serde import serde
from serde.json import to_json

from colsfm.ba_backend import BaBackend
from colsfm.cameras import colmap_camera_parameters
from colsfm.frames_meta import CameraParams, FramesMeta
from colsfm.geometry import MILLIMETRES_PER_METRE, relative_rotation_degrees
from colsfm.mapping import MappingResult, PolishStats
from colsfm.pose_graph import EdgeKind, PoseGraphEdge
from colsfm.rig_calibration import ExtrinsicChange
from colsfm.run_config import PipelineOptions
from colsfm.run_lifecycle import StageName


@serde
@dataclass(frozen=True, slots=True)
class MappingStats:
    """The reconstruction counters `summary.json` reports, as one block.

    A snapshot of `colsfm.mapping.MappingResult`'s properties at the moment the
    summary was written: the result itself computes them from the live
    reconstruction, which is exactly why they cannot be stored there.
    """

    num_registered_images: int
    """Images belonging to a registered frame after mapping."""
    num_images_with_observations: int
    """Images observing at least one point; the count a COLMAP export writes out."""
    num_points3D: int
    """Triangulated points in the final model."""
    num_observations: int
    """Point-to-image observations in the final model."""
    mean_reprojection_error_px: float
    """Mean reprojection error over every observation, in pixels."""
    mean_track_length: float
    """Mean observations per point."""
    ba_backend: BaBackend = "ceres"
    """Which backend the bundle adjustments actually ran on, fallbacks applied; the one
    field here that is not a counter, and the only record a finished run keeps of it."""
    polish_seconds: float | None = None
    """Seconds the closing Ceres polish of a CASPAR run took, or None when none ran.

    They are already inside the `reconstruction` stage of `runtime.csv` — the polish
    solves inside `run_mapping` — so this names them rather than adding a row."""
    polish_iterations: int | None = None
    """Ceres iterations that polish took; the measure of how far CASPAR stopped short."""
    ba_fallback_reason: str | None = None
    """Why `ba_backend` is not `options.ba_backend`, or None when it is.

    A `--ba-backend caspar` run that solved on Ceres used to say so on stdout and
    nowhere else, so a finished run could not be asked which backend produced it.
    The sentences are `colsfm.ba_backend`'s: an unsupported camera model,
    `--no-regularised-extrinsics`, or a pycolmap built without CASPAR_ENABLED. The
    last is the ordinary outcome of the default `--ba-backend caspar` outside a
    `colsfm` environment, which is exactly why it is written down here."""

    @staticmethod
    def of(mapping: MappingResult) -> MappingStats:
        """Snapshot a mapping result's counters.

        Args:
            mapping: What the last mapping pass produced.

        Returns:
            The six counters, the backend that produced them and its closing polish,
            frozen at this point in the run.
        """
        polish: PolishStats | None = mapping.polish
        return MappingStats(
            num_registered_images=mapping.num_registered_images,
            num_images_with_observations=mapping.num_images_with_observations,
            num_points3D=mapping.num_points3D,
            num_observations=mapping.num_observations,
            mean_reprojection_error_px=mapping.mean_reprojection_error_px,
            mean_track_length=mapping.mean_track_length,
            ba_backend=mapping.ba_backend,
            polish_seconds=None if polish is None else polish.seconds,
            polish_iterations=None if polish is None else polish.report.num_iterations,
            ba_fallback_reason=mapping.ba_fallback_reason,
        )


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class PriorAlignment:
    """The similarity `prior ~ scale * R @ x + t` applied to a run, and how well it fits."""

    scale: float
    rotation_deg: float
    rotation_xyzw: list[float]
    translation_m: list[float]
    residual_median_m: float
    """Camera-centre residual after the fit, over the fitted cameras."""
    residual_p90_m: float
    num_cameras: int
    """Registered images with a prior pose."""
    num_fitted: int
    """Of those, the ones the refit kept."""

    def sim3d(self) -> pycolmap.Sim3d:
        """The transform as pycolmap's `Sim3d` (new world from old world)."""
        return pycolmap.Sim3d(self.scale, pycolmap.Rotation3d(np.asarray(self.rotation_xyzw)), np.asarray(self.translation_m))


@serde
@dataclass(frozen=True, slots=True)
class FocalChange:
    """How far one camera's focal moved in a mapper that refined it (`--refine-camera-model`)."""

    camera_params_id: int
    """The camera, by its `frames_meta.json` id."""
    sensor_name: str
    """The camera folder name, for reading the table."""
    nominal_focal_px: tuple[float, float]
    """fx, fy the run started from."""
    refined_focal_px: tuple[float, float]
    """fx, fy the mapper ended on."""
    focal_change_percent: float
    """The larger of the two relative changes, in percent."""


@serde
@dataclass(frozen=True, slots=True)
class PipelineSummary:
    """`summary.json`: what one run produced and how long each stage took."""

    options: PipelineOptions
    """The options the run was given, verbatim; the run's own provenance."""
    git_sha: str
    """`git rev-parse HEAD` of the working tree, or `"unknown"` outside a checkout."""
    num_input_keyframes: int
    """Keyframes in the input `frames_meta.json`."""
    num_selected_keyframes: int
    """Keyframes that survived keyframe selection."""
    num_rig_frames: int
    """Rig frames (`synced_sample_id` groups) among the selected keyframes."""
    num_pairs: int
    """Image pairs stage 3 selected, before any loop pair."""
    num_loop_pairs: int
    """Extra image pairs matched for loop closure; 0 when the stage is off. Only the
    pairs stage 5 had to match itself are counted — most candidates are consecutive or
    stereo pairs stage 4 already matched, and matching them again would only cost time."""
    num_loop_edges: int
    """Loop constraints that survived verification and gating."""
    num_pose_graph_edges: int
    """Constraints in the solved pose graph, sequential and loop together."""
    pose_graph_translation_change_m: float
    """Largest rig position change the pose-graph solve produced, in metres."""
    mapping: MappingStats
    """The final model's counters, after the extrinsic refinement when one ran."""
    stage_seconds: dict[StageName, float]
    """Wall-clock seconds per stage, in stage order; the same rows as `runtime.csv`."""
    total_seconds: float
    """Wall-clock seconds for the whole run."""
    extrinsic_changes: tuple[ExtrinsicChange, ...] = ()
    """Per-camera extrinsic movement the refinement produced; empty when the flag is off."""
    extrinsic_refinement_rounds_run: int = 0
    """Rounds the regularised refinement actually took before it converged."""
    focal_changes: tuple[FocalChange, ...] = ()
    """Per-camera focal movement the mapper produced under `--refine-camera-model`; empty when the flag is off."""
    prior_alignment: PriorAlignment | None = None
    """The similarity `align_to_prior` applied to every exported product; None when it did not run."""


@serde
@dataclass(frozen=True, slots=True)
class LoopEdgeRecord:
    """One gated loop constraint, in a form a viewer can read without pycolmap.

    `PoseGraphEdge` carries a `pycolmap.Rigid3d` and a 6x6 NumPy array, neither of
    which pyserde can write; this is the same measurement flattened to plain
    numbers. The information matrix is reduced to its diagonal because that is all
    `colsfm.pose_graph.default_information` ever puts in it — the off-diagonal
    blocks are zero by construction — and the diagonal is what a viewer would
    render as an edge weight.
    """

    source: int
    """`rig_id` (cuSFM's `synced_sample_id`) of the source rig frame."""
    target: int
    """`rig_id` of the target rig frame."""
    source_T_target_quaternion_xyzw: tuple[float, float, float, float]
    """Rotation of the measurement, as `pycolmap.Rotation3d.quat` orders it: x, y, z, w."""
    source_T_target_translation: tuple[float, float, float]
    """Translation of the measurement, in metres."""
    information_diagonal: tuple[float, float, float, float, float, float]
    """Diagonal of the 6x6 weight, rotation block first then translation."""
    kind: EdgeKind
    """`PoseGraphEdge.kind`; always `"loop"` in this file, kept so the record is self-describing."""


def loop_edge_record(edge: PoseGraphEdge) -> LoopEdgeRecord:
    """Flatten one pose-graph edge into its serialisable form.

    Args:
        edge: The constraint, as `colsfm.loop_closure` gated it.

    Returns:
        The same measurement as plain numbers.
    """
    quaternion: Float[ndarray, " 4"] = np.asarray(edge.source_T_target.rotation.quat, dtype=np.float64)
    translation: Float[ndarray, " 3"] = np.asarray(edge.source_T_target.translation, dtype=np.float64)
    diagonal: Float[ndarray, " 6"] = np.asarray(edge.information, dtype=np.float64).diagonal()
    return LoopEdgeRecord(
        source=int(edge.source),
        target=int(edge.target),
        source_T_target_quaternion_xyzw=(
            float(quaternion[0]),
            float(quaternion[1]),
            float(quaternion[2]),
            float(quaternion[3]),
        ),
        source_T_target_translation=(float(translation[0]), float(translation[1]), float(translation[2])),
        information_diagonal=(
            float(diagonal[0]),
            float(diagonal[1]),
            float(diagonal[2]),
            float(diagonal[3]),
            float(diagonal[4]),
            float(diagonal[5]),
        ),
        kind=edge.kind,
    )


def write_loop_edges(path: Path, loop_edges: Sequence[PoseGraphEdge]) -> None:
    """Write the gated loop constraints beside the pose-graph outputs.

    Written on every run, empty list included: a viewer that draws loop edges needs
    to tell "the stage ran and found none" from "the file is not there yet", and an
    absent file cannot say the first.

    Args:
        path: Destination `.json` file; parent directories are created.
        loop_edges: The constraints stage 5 produced; may be empty.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json([loop_edge_record(edge) for edge in loop_edges]) + "\n")


def extrinsic_changes(
    frames_meta: FramesMeta,
    refined_by_camera_params_id: Mapping[int, pycolmap.Rigid3d],
    reference_camera_params_id: int,
) -> tuple[ExtrinsicChange, ...]:
    """Measure what the refinement did to each camera's extrinsic.

    The per-camera translation and rotation are the same two quantities
    `colsfm.extrinsic_refinement.extrinsic_deltas` maximises over, computed the
    same way — `MILLIMETRES_PER_METRE` times the origin distance, and
    `colsfm.geometry.relative_rotation_degrees` for the angle.

    Args:
        frames_meta: The collection the refinement started from.
        refined_by_camera_params_id: `vehicle_T_cam` per camera after the solve.
        reference_camera_params_id: The rig origin, which is also the fixed camera.

    Returns:
        One record per camera, ordered by `camera_params_id`.
    """
    changes: list[ExtrinsicChange] = []
    for camera_params_id in sorted(refined_by_camera_params_id):
        camera: CameraParams = frames_meta.cameras[camera_params_id]
        refined: pycolmap.Rigid3d = refined_by_camera_params_id[camera_params_id]
        translation_m: float = float(
            np.linalg.norm(np.asarray(refined.translation) - np.asarray(camera.vehicle_T_cam.translation))
        )
        changes.append(
            ExtrinsicChange(
                camera_params_id=camera_params_id,
                sensor_name=camera.sensor_name,
                translation_change_mm=MILLIMETRES_PER_METRE * translation_m,
                rotation_change_deg=relative_rotation_degrees(refined, camera.vehicle_T_cam),
                is_reference=camera_params_id == reference_camera_params_id,
            )
        )
    return tuple(changes)




def focal_changes(nominal: FramesMeta, refined: FramesMeta) -> tuple[FocalChange, ...]:
    """Measure what a refining mapper did to each camera's focal.

    Args:
        nominal: The collection the mapper started from.
        refined: The same collection after `colsfm.cameras.with_colmap_intrinsics`.

    Returns:
        One record per camera, ordered by `camera_params_id`.
    """
    changes: list[FocalChange] = []
    for camera_params_id in sorted(refined.cameras):
        before: Float64[ndarray, " 2"] = colmap_camera_parameters(nominal.cameras[camera_params_id])[:2]
        after: Float64[ndarray, " 2"] = colmap_camera_parameters(refined.cameras[camera_params_id])[:2]
        changes.append(
            FocalChange(
                camera_params_id=camera_params_id,
                sensor_name=refined.cameras[camera_params_id].sensor_name,
                nominal_focal_px=(float(before[0]), float(before[1])),
                refined_focal_px=(float(after[0]), float(after[1])),
                focal_change_percent=float(100.0 * np.max(np.abs(after / before - 1.0))),
            )
        )
    return tuple(changes)
