"""Stage E: score init and refined cameras against the dataset ground truth.

GT enters here and nowhere else. Both variants are aligned to the GT rig with
SE(3) (primary — tests the metric claim) and Sim(3) (secondary — isolates the
scale error), then per-camera rotation/translation errors and focal errors are
reported and written to ``eval.json``. Datasets without ground truth still get
the refinement diagnostics and the self-contained rig geometry.
"""

import json
from dataclasses import asdict, dataclass

import numpy as np
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray
from simplecv.ops.umeyama import SimilarityTransform

from exo_calib.cameras import RigCameras, camera_centers
from exo_calib.catalog_io import StageConfig, StageContext, stage_context
from exo_calib.eval import RIG_MODES, CameraErrors, RigGeometry, RigMode, align_rigs, evaluate_rig, rig_geometry
from exo_calib.layer_io import (
    ALIGN_LAYER,
    CALIBRATION_VARIANTS,
    VARIANT_COLOR,
    CalibrationVariant,
    RefinementDiagnostics,
    exocalib_entity,
    kp3d_entity,
    new_layer_recording,
    read_refinement_diagnostics,
    read_rig_cameras,
    register_layer,
)


@dataclass(slots=True, frozen=True)
class ErrorStats:
    """Per-camera errors and their summary, rounded for ``eval.json``."""

    per_camera: list[float]
    mean: float
    median: float
    max: float


def error_stats(values: Float64[ndarray, " v"], digits: int) -> ErrorStats:
    """Summarize one per-camera error vector."""
    return ErrorStats(
        per_camera=[round(float(x), digits) for x in values],
        mean=round(float(np.mean(values)), digits),
        median=round(float(np.median(values)), digits),
        max=round(float(np.max(values)), digits),
    )


@dataclass(slots=True, frozen=True)
class AlignedErrors:
    """Pose errors after aligning the rig onto GT under one alignment model."""

    rotation_deg: ErrorStats
    translation_cm: ErrorStats
    scale: float
    """Scale the alignment applied (1.0 under SE(3))."""


@dataclass(slots=True, frozen=True)
class FocalErrors:
    """Focal-length error of the estimated intrinsics against GT, in percent."""

    per_camera: list[float]
    mean_abs: float
    max_abs: float


@dataclass(slots=True, frozen=True)
class VariantMetrics:
    """Everything ``eval.json`` records about one camera set."""

    se3: AlignedErrors
    sim3: AlignedErrors
    focal_error_pct: FocalErrors


def variant_metrics(pred: RigCameras, gt: RigCameras) -> VariantMetrics:
    """Score one camera set against GT under SE(3) and Sim(3) alignment."""
    aligned: dict[RigMode, AlignedErrors] = {}
    for mode in RIG_MODES:
        errors: CameraErrors = evaluate_rig(pred.cam_T_world, gt.cam_T_world, mode=mode)
        aligned[mode] = AlignedErrors(
            rotation_deg=error_stats(errors.rotation_error_deg, 4),
            translation_cm=error_stats(errors.translation_error_cm, 4),
            scale=round(float(errors.alignment_scale), 6),
        )
    focal_pct: Float64[ndarray, " v"] = (pred.intrinsics[:, 0, 0] - gt.intrinsics[:, 0, 0]) / gt.intrinsics[:, 0, 0] * 100.0
    return VariantMetrics(
        se3=aligned["se3"],
        sim3=aligned["sim3"],
        focal_error_pct=FocalErrors(
            per_camera=[round(float(x), 3) for x in focal_pct],
            mean_abs=round(float(np.abs(focal_pct).mean()), 3),
            max_abs=round(float(np.abs(focal_pct).max()), 3),
        ),
    )


@dataclass(slots=True, frozen=True)
class RigGeometryReport:
    """``RigGeometry`` as JSON lists (metres), rounded to 6 digits."""

    camera_centers_m: list[list[float]]
    pairwise_distance_m: list[list[float]]
    height_above_floor_m: list[float]


@dataclass(slots=True, frozen=True)
class EvalReport:
    """The ``eval.json`` schema: identity, per-variant GT metrics when GT exists, and the GT-free diagnostics."""

    segment_id: str
    dataset_name: str
    camera_names: list[str]
    init: VariantMetrics | None
    refined: VariantMetrics | None
    refinement_diagnostics: RefinementDiagnostics
    refined_rig_geometry: RigGeometryReport


def main(config: StageConfig) -> None:
    """Score both variants, write ``eval.json``, and register the viewer alignment layer."""
    context: StageContext = stage_context(config)
    dataset, segment_id = context.dataset, context.segment_id
    names: tuple[str, ...] = context.layout.exo_camera_names
    missing_gt: tuple[str, ...] = tuple(name for name in names if name not in context.layout.calibrated_camera_names)
    gt: RigCameras | None = None
    if missing_gt:
        print(f"ground truth: unavailable for {len(missing_gt)}/{len(names)} cameras — evaluation and alignment skipped")
    else:
        gt = read_rig_cameras(dataset, segment_id, names, source="ground_truth")
    variants: dict[CalibrationVariant, RigCameras] = {variant: read_rig_cameras(dataset, segment_id, names, source=variant) for variant in CALIBRATION_VARIANTS}

    metrics: dict[CalibrationVariant, VariantMetrics] = {}
    if gt is not None:
        for variant, pred in variants.items():
            metrics[variant] = variant_metrics(pred, gt)
            for mode in RIG_MODES:
                m: AlignedErrors = getattr(metrics[variant], mode)
                print(
                    f"{variant:8s} {mode:4s}: trans cm mean {m.translation_cm.mean:6.2f} med {m.translation_cm.median:6.2f} "
                    f"max {m.translation_cm.max:6.2f} | rot deg mean {m.rotation_deg.mean:5.2f} med {m.rotation_deg.median:5.2f} "
                    f"max {m.rotation_deg.max:5.2f} | scale {m.scale:.4f}"
                )
    diagnostics: RefinementDiagnostics = read_refinement_diagnostics(dataset, segment_id)
    print(f"refinement diagnostics: {json.dumps(asdict(diagnostics), separators=(',', ':'))}")
    geometry: RigGeometry = rig_geometry(variants["refined"].cam_T_world)
    print(f"refined pairwise camera distances (m): {geometry.pairwise_distance_m.round(3).tolist()}")
    print(f"refined camera heights above fitted floor (m): {geometry.height_above_floor_m.round(3).tolist()}")
    report: EvalReport = EvalReport(
        segment_id=segment_id,
        dataset_name=config.dataset_name,
        camera_names=list(names),
        init=metrics.get("init"),
        refined=metrics.get("refined"),
        refinement_diagnostics=diagnostics,
        refined_rig_geometry=RigGeometryReport(
            camera_centers_m=geometry.camera_centers_m.round(6).tolist(),
            pairwise_distance_m=geometry.pairwise_distance_m.round(6).tolist(),
            height_above_floor_m=geometry.height_above_floor_m.round(6).tolist(),
        ),
    )
    context.segment_dir.mkdir(parents=True, exist_ok=True)
    eval_path = context.segment_dir / "eval.json"
    eval_path.write_text(json.dumps({key: value for key, value in asdict(report).items() if value is not None}, indent=2))
    print(f"wrote {eval_path}")

    # Viewer alignment: give each variant its own SE(3) pred-world -> GT-world
    # transform (on its frusta and kp3d subtrees) so everything overlays GT, and
    # draw labeled error lines from each aligned camera center to its GT center.
    recording, rrd_path = new_layer_recording(segment_id, context.segment_dir / f"{ALIGN_LAYER}.rrd")
    if gt is None:
        recording.log("/world/exocalib_report", rr.AnyValues(ground_truth_available=False), static=True)
    else:
        gt_centers: Float64[ndarray, "v 3"] = camera_centers(gt.cam_T_world)
        for variant, pred in variants.items():
            gt_T_pred: SimilarityTransform = align_rigs(pred.cam_T_world, gt.cam_T_world, with_scale=False)
            transform: rr.Transform3D = rr.Transform3D(translation=gt_T_pred.dst_t_src, mat3x3=gt_T_pred.dst_R_src)
            recording.log(exocalib_entity(variant), transform, static=True)
            recording.log(kp3d_entity(variant), transform, static=True)
            aligned_centers: Float64[ndarray, "v 3"] = gt_T_pred.apply(camera_centers(pred.cam_T_world))
            errors_cm: Float64[ndarray, " v"] = np.linalg.norm(aligned_centers - gt_centers, axis=1) * 100.0
            recording.log(
                f"/world/exocalib_error/{variant}",
                rr.LineStrips3D(
                    strips=[np.stack([aligned_centers[i], gt_centers[i]]) for i in range(len(gt.names))],
                    labels=[f"{name}: {errors_cm[i]:.1f} cm" for i, name in enumerate(gt.names)],
                    colors=VARIANT_COLOR[variant],
                    radii=0.004,
                    show_labels=True,
                ),
                static=True,
            )
            print(f"{variant}: alignment + error lines logged (mean {errors_cm.mean():.1f} cm)")
    recording.flush(timeout_sec=30.0)
    print(f"wrote {rrd_path}")
    if config.register:
        register_layer(dataset, rrd_path, ALIGN_LAYER)
        print(f"registered layer {ALIGN_LAYER}")
