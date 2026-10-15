"""Graph-tier check of exo-calib, driven by ``VERIFY.toml`` through ``python -m dagkit.verify``.

``run`` writes a golden rrd; ``score`` reads only that golden plus the GT rig of the segment from the catalog.
Golden contract (paths on ``video_time``, as the pre-port goldens): static rigs at ``/world/exocalib/{init,refined}/<rig>/<cam>``
(``Transform3D`` child-from-parent = cam_T_world, ``Pinhole`` on the ``pinhole`` child), kp2d boxes and the gated
skeleton under each camera's ``pinhole/exocalib_kp2d*``, kp3d tracks at ``/world/exocalib/kp3d_{init,refined}``, GT
alignment at ``/world/exocalib_error``. Since the port the queryable kp2d record sits at ``/world/exocalib/kp2d/<cam>``.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from dagkit.app import RUNNERS
from dagkit.sources import SegmentRef
from dagkit.types import read_column, served
from dagkit.verify import Case, Phase, load_card
from jaxtyping import Float64
from numpy import ndarray
from simplecv.ops.umeyama import SimilarityTransform, umeyama_alignment
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.types.rig import RigCameras

from exo_calib.apis.calibrate import Config, build

CARD: Path = Path(__file__).resolve().parents[1] / "VERIFY.toml"


def case_config(case: Case) -> Config:
    """The App config of a catalog case; it registers nothing, and its files go to the package's ``data/outputs`` from any cwd."""
    return Config(RerunTyroConfig(headless=True), case.segment_ref(), frame_budget=int(case.params["frame_budget"]), output_dir=CARD.parent / "data" / "outputs", register="off")


def examples() -> list[Config]:
    """The card's short cases as App configs: the gradio app's examples, one click from a known-good run."""
    return [case_config(case) for block in load_card(CARD).graph for case in block.cases if case.clip == "short"]


def run(case: Case, phase: Phase, recording: rr.RecordingStream) -> None:
    """The App's graphs (``apis/calibrate.py``, lossless, threaded): every final output goes into ``recording``.

    Nothing is registered and no layer file is written, so the shared catalog is only read. ``phase`` does not change what runs.
    """
    if case.window_s is not None:  # fail closed: Stage A calibrates from Config.frame_index, which a window may exclude
        raise ValueError(f"case {case.name!r}: exo-calib does not run windowed cases (window_s); use frame_budget for a short clip")
    config: Config = case_config(case)
    for graph in build(config, recording, golden=True):
        RUNNERS[config.runner](graph, mode=config.mode)
        del graph  # its models free before the next graph is built (as the App's loop does)
    print(f"{case.name} {phase}: graph outputs written into the golden")


def _rig_errors(
    prefix: str, pred: tuple[Float64[ndarray, "v 3 3"], Float64[ndarray, "v 4 4"]], gt: tuple[Float64[ndarray, "v 3 3"], Float64[ndarray, "v 4 4"]]
) -> dict[str, float]:
    """Mean SE(3)-aligned rotation (deg) and centre (cm) errors and mean absolute focal error (%) of one rig against GT."""
    centers: Float64[ndarray, "v 3"] = -np.einsum("vji,vj->vi", pred[1][:, :3, :3], pred[1][:, :3, 3])
    gt_centers: Float64[ndarray, "v 3"] = -np.einsum("vji,vj->vi", gt[1][:, :3, :3], gt[1][:, :3, 3])
    gt_T_pred: SimilarityTransform = umeyama_alignment(centers, gt_centers, allow_scaling=False)
    aligned_R: Float64[ndarray, "v 3 3"] = pred[1][:, :3, :3] @ gt_T_pred.dst_R_src.T
    cosine: Float64[ndarray, " v"] = (np.trace(gt[1][:, :3, :3] @ aligned_R.transpose(0, 2, 1), axis1=1, axis2=2) - 1.0) / 2.0
    return {
        f"{prefix}rot_deg_mean": float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).mean()),
        f"{prefix}trans_cm_mean": float(np.linalg.norm(gt_T_pred.apply(centers) - gt_centers, axis=1).mean() * 100.0),
        f"{prefix}focal_pct_mean_abs": float(np.abs(pred[0][:, 0, 0] / gt[0][:, 0, 0] - 1.0).mean() * 100.0),
    }


def score(case: Case, golden: Path) -> dict[str, float]:
    """Score the golden's refined (banded) and init (reported) rigs against the segment's GT base calibration."""
    segment: SegmentRef = case.segment_ref()
    with served(golden) as dataset:
        table: pa.Table = dataset.filter_contents(["/world/exocalib/**"]).reader(index=None).to_arrow_table()
    _, refined = next(RigCameras.from_layer(table, "/world/exocalib/refined", segment.timeline))
    _, init = next(RigCameras.from_layer(table, "/world/exocalib/init", segment.timeline))
    gt_table: pa.Table = read_column(segment.entry(), segment.segment, RigCameras, "/world", segment.timeline, static=True)
    _, gt = next(RigCameras.from_layer(gt_table, "/world", segment.timeline))
    init, gt = init.select(refined.names), gt.select(refined.names)
    truth: tuple[Float64[ndarray, "v 3 3"], Float64[ndarray, "v 4 4"]] = (gt.intrinsics, gt.cam_T_world)
    return _rig_errors("", (refined.intrinsics, refined.cam_T_world), truth) | _rig_errors("init_", (init.intrinsics, init.cam_T_world), truth)
