"""Stage E: score init and refined cameras against the dataset ground truth.

GT enters here and nowhere else. Both variants are aligned to the GT rig with
SE(3) (primary — tests the metric claim) and Sim(3) (secondary — isolates the
scale error), then per-camera rotation/translation errors and focal errors are
reported and written to ``eval.json``.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rerun as rr
from jaxtyping import Float64
from numpy import ndarray

from exo_calib.apis.calibrate_init import InitCameras
from exo_calib.catalog_io import (
    DEFAULT_CATALOG_URL,
    DEFAULT_DATASET_NAME,
    GtCameras,
    connect_dataset,
    new_layer_recording,
    only_segment_id,
    read_gt_cameras,
    register_layer,
)


@dataclass
class ReportConfig:
    """Config for the evaluation report tool."""

    catalog_url: str = DEFAULT_CATALOG_URL
    """Rerun catalog server URL."""
    dataset_name: str = DEFAULT_DATASET_NAME
    """Catalog dataset holding the registered segment."""
    segment_id: str | None = None
    """Segment to score; ``None`` uses the dataset's single segment."""
    output_dir: Path = Path("data/outputs")
    """Directory holding Stage A/C outputs; ``eval.json`` lands beside them."""
    align_layer_name: str = "exocalib_align"
    """Catalog layer carrying the SE(3) alignment transform on ``/world/exocalib``,
    so the pipeline-frame frusta and 3D tracks display in the GT world frame.
    The pipeline outputs themselves stay unaligned (self-contained)."""
    application_id: str = "exocalib"
    """Application id of generated layer recordings."""
    register: bool = True
    """Register the alignment layer into the catalog."""


def _variant_metrics(pred: InitCameras, gt: GtCameras) -> dict:
    """Score one camera set against GT under SE(3) and Sim(3) alignment."""
    from exo_calib.eval import evaluate_rig

    metrics: dict = {}
    for mode in ("se3", "sim3"):
        errors = evaluate_rig(pred.cam_T_world_v44, gt.cam_T_world_v44, mode=mode)
        metrics[mode] = {
            "rotation_deg": {
                "per_camera": [round(float(x), 4) for x in errors.rotation_deg_v],
                "mean": round(float(np.mean(errors.rotation_deg_v)), 4),
                "median": round(float(np.median(errors.rotation_deg_v)), 4),
                "max": round(float(np.max(errors.rotation_deg_v)), 4),
            },
            "translation_cm": {
                "per_camera": [round(float(x), 4) for x in errors.translation_cm_v],
                "mean": round(float(np.mean(errors.translation_cm_v)), 4),
                "median": round(float(np.median(errors.translation_cm_v)), 4),
                "max": round(float(np.max(errors.translation_cm_v)), 4),
            },
            "scale": round(float(errors.scale), 6),
        }
    focal_pct: Float64[ndarray, " v"] = (pred.k_v33[:, 0, 0] - gt.k_v33[:, 0, 0]) / gt.k_v33[:, 0, 0] * 100.0
    metrics["focal_error_pct"] = {
        "per_camera": [round(float(x), 3) for x in focal_pct],
        "mean_abs": round(float(np.abs(focal_pct).mean()), 3),
        "max_abs": round(float(np.abs(focal_pct).max()), 3),
    }
    return metrics


def _gt_kp3d_by_time(dataset, segment_id: str) -> dict[int, Float64[ndarray, "133 3"]]:
    """Read the GT 3D COCO-133 keypoints (hands) per timeline nanosecond."""
    from exo_calib.catalog_io import TIMELINE

    column: str = "/world/gt/coco133_xyz:Points3D:positions"
    view = dataset.filter_segments(segment_id).filter_contents(["/world/gt/**"])
    table = view.reader(index=TIMELINE, fill_latest_at=False).select(TIMELINE, column).sort(TIMELINE).to_arrow_table()
    time_column: str = table.column_names[0]
    gt_by_ns: dict[int, Float64[ndarray, "133 3"]] = {}
    for row in table.to_pylist():
        if row[column] is not None and len(row[column]) == 133:
            gt_by_ns[int(row[time_column].value)] = np.asarray(row[column], dtype=np.float64)
    return gt_by_ns


def _kp3d_metrics(
    points_npz: Path,
    alignment,
    gt_by_ns: dict[int, Float64[ndarray, "133 3"]],
    max_time_gap_ns: int = 25_000_000,
) -> dict | None:
    """Score aligned triangulated keypoints against GT 3D hands (MPJPE, cm)."""
    data = np.load(points_npz)
    points_n3: Float64[ndarray, "n 3"] = data["points_xyz_n3"]
    frame_idx_n = data["frame_idx_n"]
    joint_idx_n = data["joint_idx_n"]
    frame_times_ns = data["frame_times_ns"]
    gt_times: Float64[ndarray, " g"] = np.array(sorted(gt_by_ns.keys()), dtype=np.float64)
    if gt_times.size == 0:
        return None
    aligned_n3: Float64[ndarray, "n 3"] = (alignment.scale * (alignment.rotation_33 @ points_n3.T)).T + alignment.translation_3
    point_times_n: Float64[ndarray, " n"] = frame_times_ns[frame_idx_n].astype(np.float64)
    nearest_gt_pos_n = np.clip(np.searchsorted(gt_times, point_times_n), 1, gt_times.size - 1)
    nearest_gt_ns_n = np.where(
        np.abs(gt_times[nearest_gt_pos_n - 1] - point_times_n) <= np.abs(gt_times[nearest_gt_pos_n] - point_times_n),
        gt_times[nearest_gt_pos_n - 1],
        gt_times[nearest_gt_pos_n],
    )
    errors_cm: list[float] = []
    for i in range(points_n3.shape[0]):
        if abs(nearest_gt_ns_n[i] - point_times_n[i]) > max_time_gap_ns or not np.isfinite(aligned_n3[i]).all():
            continue
        gt_xyz_1333: Float64[ndarray, "133 3"] = gt_by_ns[int(nearest_gt_ns_n[i])]
        gt_joint_3: Float64[ndarray, "3"] = gt_xyz_1333[int(joint_idx_n[i])]
        if not np.isfinite(gt_joint_3).all():
            continue
        errors_cm.append(float(np.linalg.norm(aligned_n3[i] - gt_joint_3) * 100.0))
    if not errors_cm:
        return None
    errors: Float64[ndarray, " e"] = np.asarray(errors_cm)
    return {
        "matched_points": int(errors.size),
        "mean_cm": round(float(errors.mean()), 3),
        "median_cm": round(float(np.median(errors)), 3),
        "p90_cm": round(float(np.percentile(errors, 90)), 3),
    }


def main(config: ReportConfig) -> None:
    """Score both variants and write ``eval.json``."""
    dataset = connect_dataset(config.catalog_url, config.dataset_name)
    segment_id: str = config.segment_id or only_segment_id(dataset)
    segment_dir: Path = config.output_dir / segment_id
    gt: GtCameras = read_gt_cameras(dataset, segment_id)

    report: dict = {"segment_id": segment_id}
    for variant, filename in (("init", "init_cameras.npz"), ("refined", "refined_cameras.npz")):
        npz_path: Path = segment_dir / filename
        if not npz_path.exists():
            print(f"{variant}: {npz_path} missing — skipped")
            continue
        pred: InitCameras = InitCameras.load(npz_path)
        report[variant] = _variant_metrics(pred, gt)
        for mode in ("se3", "sim3"):
            m: dict = report[variant][mode]
            print(
                f"{variant:8s} {mode:4s}: trans cm mean {m['translation_cm']['mean']:6.2f} med {m['translation_cm']['median']:6.2f} "
                f"max {m['translation_cm']['max']:6.2f} | rot deg mean {m['rotation_deg']['mean']:5.2f} med {m['rotation_deg']['median']:5.2f} "
                f"max {m['rotation_deg']['max']:5.2f} | scale {m['scale']:.4f}"
            )

    # Viewer alignment: give each variant its own SE(3) pred-world -> GT-world
    # transform (on its frusta and kp3d subtrees) so everything overlays GT, and
    # draw labeled error lines from each aligned camera center to its GT center.
    # The same alignment scores the triangulated keypoints against the GT hands.
    from exo_calib.eval import align_rigs

    gt_kp3d_by_ns: dict[int, Float64[ndarray, "133 3"]] = _gt_kp3d_by_time(dataset, segment_id)
    rrd_path: Path = segment_dir / f"{config.align_layer_name}.rrd"
    recording: rr.RecordingStream = new_layer_recording(config.application_id, segment_id, rrd_path)
    gt_centers_v3: Float64[ndarray, "v 3"] = np.stack(
        [-gt.cam_T_world_v44[i, :3, :3].T @ gt.cam_T_world_v44[i, :3, 3] for i in range(len(gt.names))]
    )
    for variant, filename in (("init", "init_cameras.npz"), ("refined", "refined_cameras.npz")):
        npz_path = segment_dir / filename
        if not npz_path.exists():
            continue
        pred = InitCameras.load(npz_path)
        alignment = align_rigs(pred.cam_T_world_v44, gt.cam_T_world_v44, with_scale=False)
        points_npz: Path = segment_dir / f"points_{variant}.npz"
        if points_npz.exists() and variant in report:
            kp3d_metrics: dict | None = _kp3d_metrics(points_npz, alignment, gt_kp3d_by_ns)
            if kp3d_metrics is not None:
                report[variant]["kp3d_hands_cm"] = kp3d_metrics
                print(
                    f"{variant:8s} kp3d vs GT hands: {kp3d_metrics['matched_points']} pts | "
                    f"mean {kp3d_metrics['mean_cm']:.2f} cm med {kp3d_metrics['median_cm']:.2f} cm p90 {kp3d_metrics['p90_cm']:.2f} cm"
                )
        transform: rr.Transform3D = rr.Transform3D(translation=alignment.translation_3, mat3x3=alignment.rotation_33)
        recording.log(f"/world/exocalib/{variant}", transform, static=True)
        recording.log(f"/world/exocalib/kp3d_{variant}", transform, static=True)
        pred_centers_v3: Float64[ndarray, "v 3"] = np.stack(
            [-pred.cam_T_world_v44[i, :3, :3].T @ pred.cam_T_world_v44[i, :3, 3] for i in range(len(pred.names))]
        )
        aligned_centers_v3: Float64[ndarray, "v 3"] = (alignment.rotation_33 @ pred_centers_v3.T).T + alignment.translation_3
        errors_cm_v: Float64[ndarray, " v"] = np.linalg.norm(aligned_centers_v3 - gt_centers_v3, axis=1) * 100.0
        recording.log(
            f"/world/exocalib_error/{variant}",
            rr.LineStrips3D(
                strips=[np.stack([aligned_centers_v3[i], gt_centers_v3[i]]) for i in range(len(gt.names))],
                labels=[f"{name}: {errors_cm_v[i]:.1f} cm" for i, name in enumerate(gt.names)],
                colors=(255, 214, 0) if variant == "init" else (0, 200, 255),
                radii=0.004,
                show_labels=True,
            ),
            static=True,
        )
        print(f"{variant}: alignment + error lines logged (mean {errors_cm_v.mean():.1f} cm)")
    recording.flush(timeout_sec=30.0)
    print(f"wrote {rrd_path}")
    if config.register:
        register_layer(dataset, rrd_path, config.align_layer_name)
        print(f"registered layer {config.align_layer_name}")

    eval_path: Path = segment_dir / "eval.json"
    eval_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {eval_path}")
