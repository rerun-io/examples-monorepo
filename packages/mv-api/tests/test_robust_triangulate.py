from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float
from numpy import ndarray

from mv_api.robust_triangulate import batch_triangulate, robust_batch_triangulate


def _load_hocap_root() -> Path | None:
    # Try common env vars or fallbacks
    candidates = [
        os.environ.get("HOCAP_ROOT"),
        os.environ.get("MV_API_HOCAP_ROOT"),
        str(Path("/mnt/8tb/data/hocap/datasets")),
        str(Path("data/hocap")),
        str(Path("data/hocap-datasets")),
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.exists():
            return p
    return None


def _build_hocap_from_config(root: Path) -> tuple[Float[ndarray, "n_views 3 4"], Float[ndarray, "133 4"]]:
    """Use simplecv dataset config to build Pall and ground-truth xyzc for a frame."""
    from simplecv.data.exoego.hocap import HocapConfig

    # Pick the first available sequence to avoid hardcoding
    seq_dir, subject_id, seq_name = _pick_first_sequence(root)
    assert seq_dir.exists()

    cfg = HocapConfig(root_directory=root, subject_id=subject_id, sequence_name=seq_name)
    exoego_sequence = cfg.setup()
    exo_sequence = exoego_sequence.exo_sequence
    assert exo_sequence is not None

    Pall: Float[ndarray, "n_views 3 4"] = np.array([cam.projection_matrix for cam in exo_sequence.exo_cam_list])
    labels = exoego_sequence.exoego_labels
    assert labels is not None
    xyzc_gt: Float[ndarray, "133 4"] = labels.xyzc_stack[0]
    return Pall, xyzc_gt


def _pick_first_sequence(root: Path) -> tuple[Path, str, str]:
    # Return (sequence_dir, subject_id, sequence_name)
    subjects = sorted([p for p in root.glob("subject_*") if p.is_dir()])
    assert subjects, f"No subject_* dirs under {root}"
    for subj in subjects:
        seqs = sorted([p for p in subj.iterdir() if p.is_dir()])
        if not seqs:
            continue
        return seqs[0], subj.name.split("_")[-1], seqs[0].name
    raise AssertionError("No sequences found in HoCap root")


    # No longer needed: ground truth comes from exoego_sequence.exoego_labels
    raise NotImplementedError


def _project_uvc(Pall: Float[ndarray, "n_views 3 4"], xyzc: Float[ndarray, "133 4"]) -> Float[ndarray, "n_views 133 3"]:
    n_views = Pall.shape[0]
    uvc = np.zeros((n_views, 133, 3), dtype=np.float64)
    for i in range(n_views):
        P = Pall[i]
        X = np.hstack([xyzc[:, :3], np.ones((133, 1))])
        x = (P @ X.T).T
        z = x[:, 2:3]
        uv = x[:, :2] / z
        # mark invisible if behind camera
        vis = (z[:, 0] > 0) & (xyzc[:, 3] > 0)
        uvc[i, :, 0:2] = uv
        uvc[i, ~vis, 2] = 0.0
        uvc[i, vis, 2] = 1.0
    return uvc


def _median_3d_error(pred: Float[ndarray, "133 4"], gt: Float[ndarray, "133 4"]) -> float:
    mask = gt[:, 3] > 0
    if not np.any(mask):
        return np.inf
    return float(np.median(np.linalg.norm(pred[mask, :3] - gt[mask, :3], axis=-1)))


@pytest.mark.slow
def test_hocap_robust_beats_naive_with_outliers():
    root = _load_hocap_root()
    if root is None:
        pytest.skip("HoCap dataset root not found; set HOCAP_ROOT to run this test")

    Pall, xyzc_gt = _build_hocap_from_config(root)

    uvc = _project_uvc(Pall, xyzc_gt)
    # Introduce outliers in two views (shift all joints by 50px)
    uvc[0, :, 0:2] += 50.0
    uvc[1, :, 0:2] += 50.0

    naive: Float[ndarray, "133 4"] = batch_triangulate(uvc, Pall, min_views=2)
    robust: Float[ndarray, "133 4"] = robust_batch_triangulate(uvc, Pall, min_views=2, reproj_error_thresh=5.0, init_subset=2)

    naive_med = _median_3d_error(naive, xyzc_gt)
    robust_med = _median_3d_error(robust, xyzc_gt)

    assert robust_med < naive_med, f"robust should beat naive: robust={robust_med:.3f} naive={naive_med:.3f}"


@pytest.mark.slow
def test_hocap_robust_matches_naive_without_outliers():
    root = _load_hocap_root()
    if root is None:
        pytest.skip("HoCap dataset root not found; set HOCAP_ROOT to run this test")

    Pall, xyzc_gt = _build_hocap_from_config(root)
    uvc = _project_uvc(Pall, xyzc_gt)

    naive: Float[ndarray, "133 4"] = batch_triangulate(uvc, Pall, min_views=2)
    robust: Float[ndarray, "133 4"] = robust_batch_triangulate(uvc, Pall, min_views=2, reproj_error_thresh=5.0, init_subset=2)

    naive_med = _median_3d_error(naive, xyzc_gt)
    robust_med = _median_3d_error(robust, xyzc_gt)

    assert abs(naive_med - robust_med) < 1e-3
