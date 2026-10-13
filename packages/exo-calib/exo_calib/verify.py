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
from dagkit.sources import SegmentRef
from dagkit.verify import Case, Phase
from jaxtyping import Float64
from numpy import ndarray
from rerun.catalog import DatasetEntry
from simplecv.ops.umeyama import SimilarityTransform, umeyama_alignment
from simplecv.rerun_log_utils import RerunTyroConfig

CHILD_FROM_PARENT: int = 2
"""``Transform3D:relation`` value of a camera node logged as cam_T_world."""


def run(case: Case, phase: Phase, recording: rr.RecordingStream) -> None:
    """The ported graph (``apis/calibrate.py``, lossless, threaded): every final output goes into ``recording``.

    Nothing is registered and no layer file is written, so the shared catalog is only read. ``phase`` does not change what runs.
    """
    from exo_calib.apis.calibrate import Config, calibrate

    assert case.dataset is not None and case.segment is not None, "exo-calib cases are catalog segments"
    if case.window_s is not None:  # fail closed: Stage A calibrates from Config.frame_index, which a window may exclude
        raise ValueError(f"case {case.name!r}: exo-calib does not run windowed cases (window_s); use frame_budget for a short clip")
    config = Config(RerunTyroConfig(headless=True), SegmentRef(case.dataset, case.segment), frame_budget=int(case.params["frame_budget"]), register="off")
    calibrate(config, golden=recording)
    print(f"{case.name} {phase}: graph outputs written into the golden")


def _rig(table: pa.Table, root: str, names: list[str]) -> tuple[Float64[ndarray, "v 3 3"], Float64[ndarray, "v 4 4"]]:
    """Intrinsics and cam_T_world of the static camera nodes ``<root>/<name>`` (Rerun stores 3x3 matrices column-major)."""

    def value(column: str) -> Float64[ndarray, " k"]:
        return np.asarray(table.column(column).to_pylist()[0], dtype=np.float64).reshape(-1)

    intrinsics: Float64[ndarray, "v 3 3"] = np.stack([value(f"{root}/{name}/pinhole:Pinhole:image_from_camera").reshape(3, 3).T for name in names])
    cam_T_world: Float64[ndarray, "v 4 4"] = np.tile(np.eye(4), (len(names), 1, 1))
    for i, name in enumerate(names):
        cam_T_world[i, :3, :3] = value(f"{root}/{name}:Transform3D:mat3x3").reshape(3, 3).T
        cam_T_world[i, :3, 3] = value(f"{root}/{name}:Transform3D:translation")
        relation: str = f"{root}/{name}:Transform3D:relation"
        if relation not in table.column_names or int(value(relation)[0]) != CHILD_FROM_PARENT:
            cam_T_world[i] = np.linalg.inv(cam_T_world[i])
    return intrinsics, cam_T_world


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
    assert case.dataset is not None and case.segment is not None, "exo-calib cases are catalog segments"
    with rr.server.Server(host="127.0.0.1", datasets={"golden": [str(golden)]}) as server:
        golden_dataset: DatasetEntry = server.client().get_dataset("golden")  # an entry must outlive its queries
        table: pa.Table = golden_dataset.filter_contents(["/world/exocalib/**"]).reader(index=None).to_arrow_table()
    suffix: str = ":Transform3D:translation"
    names: list[str] = sorted(
        c.removeprefix("/world/exocalib/refined/").removesuffix(suffix)
        for c in table.column_names
        if c.startswith("/world/exocalib/refined/") and c.endswith(suffix)
    )
    source: DatasetEntry = SegmentRef(case.dataset, case.segment).entry()
    gt_table: pa.Table = (
        source.filter_segments(case.segment).filter_contents([f"/world/{name}/**" for name in names]).reader(index=None).to_arrow_table()
    )
    gt: tuple[Float64[ndarray, "v 3 3"], Float64[ndarray, "v 4 4"]] = _rig(gt_table, "/world", names)
    return _rig_errors("", _rig(table, "/world/exocalib/refined", names), gt) | _rig_errors("init_", _rig(table, "/world/exocalib/init", names), gt)
