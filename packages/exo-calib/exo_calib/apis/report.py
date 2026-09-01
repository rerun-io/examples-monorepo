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
    log_coco133_skeleton_context,
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

    eval_path: Path = segment_dir / "eval.json"
    eval_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {eval_path}")

    # Viewer alignment: give each variant its own SE(3) pred-world -> GT-world
    # transform (on its frusta and kp3d subtrees) so everything overlays GT, and
    # draw labeled error lines from each aligned camera center to its GT center.
    from exo_calib.eval import align_rigs

    rrd_path: Path = segment_dir / f"{config.align_layer_name}.rrd"
    recording: rr.RecordingStream = new_layer_recording(config.application_id, segment_id, rrd_path)
    log_coco133_skeleton_context(recording, "/world/gt", connections=True)
    gt_centers_v3: Float64[ndarray, "v 3"] = np.stack(
        [-gt.cam_T_world_v44[i, :3, :3].T @ gt.cam_T_world_v44[i, :3, 3] for i in range(len(gt.names))]
    )
    for variant, filename in (("init", "init_cameras.npz"), ("refined", "refined_cameras.npz")):
        npz_path = segment_dir / filename
        if not npz_path.exists():
            continue
        pred = InitCameras.load(npz_path)
        alignment = align_rigs(pred.cam_T_world_v44, gt.cam_T_world_v44, with_scale=False)
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
