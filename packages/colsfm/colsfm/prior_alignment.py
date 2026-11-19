"""Put a finished run back into its pose prior's gauge: one similarity, fitted on the registered cameras, applied to
everything the run exports.

Bundle adjustment pins the gauge at the start and has no prior residual, so on a long sequence the reconstruction's
scale and heading drift away from the prior's (an Insta360 walk-through: scale 0.80, 4.2 m median offset). When the prior is
the metric frame the rest of a pipeline lives in (a visual-inertial SLAM run), `align_run_directory` fits
`prior ~ s R x + t` (Umeyama, `colsfm.rigid_fit.align_rigid`) on the camera centres of every registered image against
the same images' prior poses (the run's input `frames_meta.json`), refits once without the residual outliers, and
applies it to `sparse/` (a similarity keeps every reprojection), the `kpmap/` metadata, the TUM pose files and
`summary.json` (`prior_alignment`). The catalog layer is written from the aligned directory afterwards.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pycolmap
from jaxtyping import Float64
from numpy import ndarray
from serde import serde, to_dict

from colsfm.export import EVERY_FRAME_METADATA_SUBPATH, KEYFRAME_METADATA_SUBPATH, SPARSE_DIR_NAME, write_tum_file
from colsfm.frames_meta import FRAMES_META_NAME, FramesMeta, read_frames_meta, write_frames_meta
from colsfm.geometry import TumPose, rigid3d_from_matrix
from colsfm.rigid_fit import RigidAlignment, align_rigid
from colsfm.run_lifecycle import SUMMARY_NAME

OUTLIER_FACTOR: Final[float] = 3.0
"""The refit drops cameras whose first-fit residual exceeds this many times the median residual."""

SKIPPED_ENTRIES: Final[frozenset[str]] = frozenset({"database.db", "colsfm.rrd"})
"""Not copied into a new run directory: the database is linked (GBs on a long run), the layer is rewritten."""


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


def fit_prior_alignment(estimate_xyz: Float64[ndarray, "n 3"], prior_xyz: Float64[ndarray, "n 3"]) -> PriorAlignment:
    """Fit the similarity taking estimated camera centres onto their prior ones, refit once without outliers.

    Args:
        estimate_xyz: Camera centres in the run's frame.
        prior_xyz: The same cameras' centres in the prior's frame.

    Returns:
        The fit and its residuals.
    """
    first: RigidAlignment = align_rigid(estimate_xyz, prior_xyz, estimate_scale=True)
    residual: Float64[ndarray, " n"] = np.linalg.norm(first.apply(estimate_xyz) - prior_xyz, axis=1)
    keep: ndarray = residual <= OUTLIER_FACTOR * max(float(np.median(residual)), 1e-9)
    fit: RigidAlignment = align_rigid(estimate_xyz[keep], prior_xyz[keep], estimate_scale=True) if int(keep.sum()) >= 3 else first
    kept_residual: Float64[ndarray, " k"] = np.linalg.norm(fit.apply(estimate_xyz[keep]) - prior_xyz[keep], axis=1)
    rotation: pycolmap.Rotation3d = rigid3d_from_matrix(fit.target_R_source, np.zeros(3)).rotation
    return PriorAlignment(
        scale=float(fit.scale),
        rotation_deg=float(np.rad2deg(rotation.angle())),
        rotation_xyzw=[float(value) for value in rotation.quat],
        translation_m=[float(value) for value in fit.target_t_source],
        residual_median_m=float(np.median(kept_residual)),
        residual_p90_m=float(np.percentile(kept_residual, 90)),
        num_cameras=len(estimate_xyz),
        num_fitted=int(keep.sum()),
    )


def _transform_pose(alignment: pycolmap.Sim3d, world_T_body: pycolmap.Rigid3d) -> pycolmap.Rigid3d:
    rotation: Float64[ndarray, "3 3"] = alignment.rotation.matrix()
    position: Float64[ndarray, " 3"] = alignment.scale * rotation @ np.asarray(world_T_body.translation) + np.asarray(alignment.translation)
    return rigid3d_from_matrix(rotation @ world_T_body.rotation.matrix(), position)


def _transform_tum(path: Path, alignment: pycolmap.Sim3d) -> None:
    poses: list[TumPose] = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields: list[float] = [float(value) for value in line.split()]
        world_T_body = pycolmap.Rigid3d(pycolmap.Rotation3d(np.asarray(fields[4:8])), np.asarray(fields[1:4]))
        poses.append(TumPose(round(fields[0] * 1e6), _transform_pose(alignment, world_T_body)))
    write_tum_file(path, poses)


def _transform_frames_meta(path: Path, alignment: pycolmap.Sim3d) -> None:
    frames_meta: FramesMeta = read_frames_meta(path)
    write_frames_meta(
        path,
        frames_meta.with_camera_to_world(
            {keyframe.keyframe_id: _transform_pose(alignment, keyframe.world_T_cam) for keyframe in frames_meta.keyframes}
        ),
    )


def align_run_directory(run_dir: Path, output_dir: Path, prior_frames_meta: Path | None = None) -> PriorAlignment:
    """Align a finished run to its pose prior and write the aligned run (in place when `output_dir` is `run_dir`).

    Args:
        run_dir: A finished run directory (`sparse/`, `kpmap/`, `output_poses/`, `summary.json`).
        output_dir: Where the aligned run goes; a new directory gets copies, with `database.db` linked and no layer.
        prior_frames_meta: The prior's `frames_meta.json`; the run's input (`summary.json` options.input_dir) when None.

    Returns:
        The alignment applied.

    Raises:
        ValueError: When fewer than three registered images have a prior pose.
    """
    summary: dict[str, object] = json.loads((run_dir / SUMMARY_NAME).read_text())
    if prior_frames_meta is None:
        options = summary["options"]
        assert isinstance(options, dict)
        prior_frames_meta = Path(str(options["input_dir"])) / FRAMES_META_NAME
    prior: dict[str, pycolmap.Rigid3d] = {keyframe.image_name: keyframe.world_T_cam for keyframe in read_frames_meta(prior_frames_meta).keyframes}
    reconstruction = pycolmap.Reconstruction(run_dir / SPARSE_DIR_NAME)
    names: list[str] = [image.name for image in reconstruction.images.values() if image.has_pose and image.name in prior]
    if len(names) < 3:
        raise ValueError(f"{run_dir}: {len(names)} registered images have a prior pose in {prior_frames_meta}")
    by_name = {image.name: image for image in reconstruction.images.values()}
    estimate_xyz = np.asarray([by_name[name].projection_center() for name in names], dtype=np.float64)
    prior_xyz = np.asarray([prior[name].translation for name in names], dtype=np.float64)
    alignment: PriorAlignment = fit_prior_alignment(estimate_xyz, prior_xyz)
    sim3d: pycolmap.Sim3d = alignment.sim3d()

    if output_dir.resolve() != run_dir.resolve():
        output_dir.mkdir(parents=True, exist_ok=False)
        for entry in run_dir.iterdir():
            if entry.name in SKIPPED_ENTRIES:
                continue
            (shutil.copytree if entry.is_dir() else shutil.copy2)(entry, output_dir / entry.name)
        if (run_dir / "database.db").exists():
            os.symlink((run_dir / "database.db").resolve(), output_dir / "database.db")
    reconstruction.transform(sim3d)
    reconstruction.write_text(output_dir / SPARSE_DIR_NAME)
    for subpath in (KEYFRAME_METADATA_SUBPATH, EVERY_FRAME_METADATA_SUBPATH):
        if (output_dir / subpath).is_file():
            _transform_frames_meta(output_dir / subpath, sim3d)
    for tum in sorted((output_dir / "output_poses").rglob("*.tum")):
        _transform_tum(tum, sim3d)
    summary["prior_alignment"] = to_dict(alignment)
    (output_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=1) + "\n")
    print(
        f"[colsfm] aligned to the prior ({prior_frames_meta}): scale {alignment.scale:.4f}, rotation {alignment.rotation_deg:.2f} deg, "
        f"translation {np.round(alignment.translation_m, 3).tolist()} m, residual median {alignment.residual_median_m:.3f} m / "
        f"p90 {alignment.residual_p90_m:.3f} m over {alignment.num_fitted} of {alignment.num_cameras} cameras"
    )
    return alignment
