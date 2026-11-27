"""The fork's proof numbers as parity bands: Galileo against cuSFM, and KITTI 06 against the paper's Table 4 setup.

Reference numbers (RTX 5090, pablovela5620/pyCuSFM feat/pycolmap-open-pipeline, NOTES.md and
docs/full-pipeline-results.md):

| Run | Reference |
|---|---|
| Galileo, COLMAP ALIKED + LightGlue (pycolmap) + Ceres ablation, no loops, no refinement (the engine-free case before ORT; its bounds carry over) | ATE vs GT 4.33 mm (cuSFM 5.00 mm), 19.35 s (cuSFM 40.07 s), 225/226 registered, 1.334 px |
| Galileo, default pipeline | ATE vs GT 5.08 mm, 20.0 s (cuSFM with refinement 32.4 s) |
| KITTI 06, default pipeline, cuVSLAM start | Sim(3) ATE 0.928 m, 155.7 s, 2156 registered (cuSFM 1.328 m, 1596.1 s) |

The bands are relative, as the fork's acceptance bounds are: accuracy within 10 % of the fork's own number (or the
fork's bound against cuSFM where it had one), wall time below the cuSFM binary's own time — runtimes move with the GPU's
other tenants, and the claim the fork makes is "faster than cuSFM", not a stopwatch reading.

Assets: the Galileo sample (`colsfm-download-galileo`), the RaCo graphs (`colsfm-raco-export`) and the KITTI 06 input
folder `packages/colsfm/data/kitti/06_input` (KITTI odometry 06 PNGs + the cuVSLAM SLAM start trajectory that the cuSFM
binary writes; see the README). A missing asset skips with its name.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Final

import pytest

from colsfm import DATA_DIR, RESOURCES_DIR
from colsfm.alignment import RigTrack, read_ground_truth, rig_track_from_frames_meta
from colsfm.benchmark import RunMetrics, compute_run_metrics, read_run, reconstruction_metrics
from colsfm.export import MERGED_POSE_FILE_NAME, OUTPUT_POSES_DIR_NAME
from colsfm.frames_meta import FRAMES_META_NAME, read_frames_meta
from colsfm.kitti_eval import Trajectory, TrajectoryReport, evaluate_trajectory, read_kitti_ground_truth
from colsfm.matching import FLEXGLUE_MODEL_DIR, FLEXGLUE_MODEL_FILES
from colsfm.model_assets import RACO_DYNAMIC_ONNX_PATH, RACO_ONNX_PATH, required_raco_graphs
from colsfm.pipeline import run_pipeline
from colsfm.run_config import PipelineOptions

pytestmark = pytest.mark.golden

GALILEO_DIR: Final[Path] = DATA_DIR / "r2b_galileo"
"""The downloaded Galileo sample."""

KITTI_INPUT_DIR: Final[Path] = DATA_DIR / "kitti" / "06_input"
"""KITTI 06 PNGs plus the cuVSLAM-posed `frames_meta.json`."""

KITTI_SEQUENCE_DIR: Final[Path] = DATA_DIR / "kitti" / "06"
"""`times.txt` and `poses_gt_06.txt`."""

CUSFM_GALILEO_REFINED_SECONDS: Final[float] = 32.4
"""The cuSFM binary on Galileo with loops and extrinsic refinement."""

CUSFM_KITTI_SECONDS: Final[float] = 1596.1
"""The cuSFM binary on KITTI 06, cuVSLAM SLAM start, loops."""

GALILEO_CUSFM_BOUND_ATE_MM: Final[float] = 5.495
"""The fork's acceptance bound against cuSFM: 1.10 x cuSFM's 5.00 mm. The fork's engine-free ablation (COLMAP's ALIKED +
LightGlue) measured 4.33 mm; the engine-free run here takes the same bound on RaCo + FlexGlue."""

GALILEO_DEFAULT_MAX_ATE_MM: Final[float] = 5.08 * 1.10
"""Within 10 % of the fork's 5.08 mm."""

GALILEO_MAX_REPROJECTION_PX: Final[float] = 1.704
"""The fork's acceptance bound: 1.10 x cuSFM's 1.550 px."""

GALILEO_MIN_REGISTERED: Final[int] = 220
"""Of 226 images; the fork's acceptance bound."""

KITTI_SIM3_ATE_M: Final[float] = 0.928
"""The fork's post-refactor default-pipeline number."""

KITTI_MIN_REGISTERED: Final[int] = 2100
"""Of 2200 keyframes in; the fork registered 2156."""


def _require(*paths: Path) -> None:
    missing: list[str] = [str(path) for path in paths if not path.exists()]
    if missing:
        pytest.skip(f"missing asset(s): {', '.join(missing)}")


GALILEO_EVERY_FRAME_M: Final[float] = 0.0
"""Galileo keeps every keyframe, as cuSFM's demo does: it is a 0.93 s, 0.66 m sweep of 29 rig frames, so the pipeline's
0.10 m / 15 deg default would leave a handful of rig frames. 0.0 turns both gates off (they are OR-combined with `>=`)."""


def _galileo_metrics(output_dir: Path) -> RunMetrics:
    input_track: RigTrack = rig_track_from_frames_meta(read_frames_meta(GALILEO_DIR / FRAMES_META_NAME))
    return compute_run_metrics(read_run(output_dir, "colsfm"), input_track, read_ground_truth(GALILEO_DIR))


def test_galileo_engine_free_run_stays_in_the_cusfm_bounds(tmp_path: Path) -> None:
    """The default RaCo + FlexGlue graphs on ONNX Runtime's CPU provider, Ceres, no refinement, no loops: within cuSFM's
    accuracy bounds with no GPU and no engine. No wall-time bound: the claim here is the models without TensorRT, not speed."""
    _require(GALILEO_DIR / FRAMES_META_NAME, *required_raco_graphs("raco", "flexglue", "onnxruntime"))
    output_dir: Path = tmp_path / "galileo_engine_free"
    started: float = time.perf_counter()
    run_pipeline(
        PipelineOptions(
            input_dir=GALILEO_DIR,
            output_dir=output_dir,
            min_inter_frame_distance=GALILEO_EVERY_FRAME_M,
            features_backend="raco",
            matching_backend="flexglue",
            inference_runtime="onnxruntime",
            ba_backend="ceres",
            optimize_extrinsics=False,
            loop_closure=False,
        )
    )
    wall: float = time.perf_counter() - started
    metrics: RunMetrics = _galileo_metrics(output_dir)
    assert metrics.vs_ground_truth is not None
    print(
        f"[golden] galileo engine-free: ATE {metrics.vs_ground_truth.rmse_millimeters:.2f} mm, "
        f"{metrics.reconstruction.registered_images} registered, {metrics.reconstruction.mean_reprojection_error_px:.3f} px, {wall:.1f} s"
    )
    assert metrics.reconstruction.registered_images >= GALILEO_MIN_REGISTERED
    assert metrics.reconstruction.mean_reprojection_error_px <= GALILEO_MAX_REPROJECTION_PX
    assert metrics.vs_ground_truth.rmse_millimeters <= GALILEO_CUSFM_BOUND_ATE_MM


def test_galileo_default_pipeline_stays_in_its_band(tmp_path: Path) -> None:
    """RaCo + FlexGlue, loops, CASPAR + Ceres polish, regularised refinement: within 10 % of the fork's 5.08 mm (4.48 measured), under cuSFM's time."""
    _require(GALILEO_DIR / FRAMES_META_NAME, RACO_ONNX_PATH, *(FLEXGLUE_MODEL_DIR / name for name in FLEXGLUE_MODEL_FILES.values()))
    output_dir: Path = tmp_path / "galileo_default"
    started: float = time.perf_counter()
    run_pipeline(PipelineOptions(input_dir=GALILEO_DIR, output_dir=output_dir, min_inter_frame_distance=GALILEO_EVERY_FRAME_M))
    wall: float = time.perf_counter() - started
    metrics: RunMetrics = _galileo_metrics(output_dir)
    assert metrics.vs_ground_truth is not None
    print(
        f"[golden] galileo default: ATE {metrics.vs_ground_truth.rmse_millimeters:.2f} mm, "
        f"{metrics.reconstruction.registered_images} registered, {metrics.reconstruction.mean_reprojection_error_px:.3f} px, {wall:.1f} s"
    )
    assert metrics.reconstruction.registered_images >= GALILEO_MIN_REGISTERED
    assert metrics.vs_ground_truth.rmse_millimeters <= GALILEO_DEFAULT_MAX_ATE_MM
    assert wall < CUSFM_GALILEO_REFINED_SECONDS


def test_kitti06_default_pipeline_reproduces_the_fork(tmp_path: Path) -> None:
    """KITTI 06 from the cuVSLAM start: Sim(3) ATE within 10 % of 0.928 m, far under cuSFM's 1596 s."""
    _require(KITTI_INPUT_DIR / FRAMES_META_NAME, KITTI_SEQUENCE_DIR / "poses_gt_06.txt", RACO_ONNX_PATH, RACO_DYNAMIC_ONNX_PATH, *(FLEXGLUE_MODEL_DIR / name for name in FLEXGLUE_MODEL_FILES.values()))
    output_dir: Path = tmp_path / "kitti06"
    started: float = time.perf_counter()
    run_pipeline(
        PipelineOptions(
            input_dir=KITTI_INPUT_DIR,
            output_dir=output_dir,
            config_dir=RESOURCES_DIR / "configs" / "kitti",
            min_inter_frame_distance=0.5,
        )
    )
    wall: float = time.perf_counter() - started
    ground_truth: Trajectory = read_kitti_ground_truth(
        poses_path=KITTI_SEQUENCE_DIR / "poses_gt_06.txt", times_path=KITTI_SEQUENCE_DIR / "times.txt"
    )
    report: TrajectoryReport = evaluate_trajectory(
        label="colsfm",
        path=output_dir / OUTPUT_POSES_DIR_NAME / MERGED_POSE_FILE_NAME,
        ground_truth=ground_truth,
        tolerance_microseconds=2000,
    )
    sim3: float = report.ate_by_alignment["sim3"].rmse
    registered: int = reconstruction_metrics(read_run(output_dir, "colsfm").reconstruction).registered_images
    print(f"[golden] kitti06 default: Sim(3) ATE {sim3:.3f} m, {registered} registered, {wall:.1f} s")
    assert registered >= KITTI_MIN_REGISTERED
    assert abs(sim3 - KITTI_SIM3_ATE_M) <= 0.10 * KITTI_SIM3_ATE_M
    assert wall < CUSFM_KITTI_SECONDS / 4
