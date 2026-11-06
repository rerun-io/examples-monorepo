"""The paper's scoring rule for each MetricTrail table that VSLAM-LAB covers.

Sources (all read 2026-11-06):

* paper: MetricTrail whitepaper (sha256 e6fa3270...), Sec. 4.1 p. 7 and Tables 8, 9, 1/12;
  extracted in /tmp/fleet-artifacts/metrictrail/paper-tables.md;
* evaluator: nv-tlabs/vipe ``metrictrail_eval`` @ 2825861 (``tum.py``, ``sevenscenes.py``, ``kitti.py``,
  ``vendored/trajectory_eval.py``): Sim(3) Umeyama on positions, translation RMSE, GT rows that are NaN dropped,
  unweighted mean over sequences;
* TUM GT association: LingBot-Map ``benchmark/dataset/tum.py`` (nearest GT pose within 0.02 s, inclusive,
  pose stored float32), which prepared the GT the evaluator reads.

VSLAM-LAB's rule for every dataset is ``evo_ape tum gt traj -va -as --t_max_diff 1e9/rgb_hz`` on nanosecond
stamps (``Evaluate/evaluate_functions.py:63``, ``Evaluate/evo_functions.py:43``).
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

VslamlabDataset: TypeAlias = Literal["rgbdtum", "7scenes", "kitti"]
"""VSLAM-LAB dataset names (``Datasets/dataset_files/dataset_<name>.yaml``) with a MetricTrail paper table."""
GtMatch: TypeAlias = Literal["nearest_time", "frame_index"]
"""How a tracked frame finds its GT row: nearest GT stamp within a tolerance, or GT row = source frame id."""
TimestampFallback: TypeAlias = Literal["none", "filename_seconds"]
"""Timestamp of a frame missing from rgb.csv: refuse, or VSLAM-LAB's ``int(float(stem) * 1e9)``."""
Units: TypeAlias = Literal["m", "cm"]
"""Units the paper prints."""


@dataclass(frozen=True, slots=True)
class PaperProtocol:
    """The paper's trajectory scoring for one dataset, plus how its frames meet VSLAM-LAB's files."""

    dataset: VslamlabDataset
    """VSLAM-LAB dataset name."""
    benchmark_folder: str
    """Folder under the VSLAM-LAB benchmark root (VSLAM-LAB's ``dataset_name.upper()``)."""
    rgb_hz: float
    """VSLAM-LAB's ``rgb_hz`` for the dataset; its evo ``t_max_diff`` is ``1e9 / rgb_hz`` ns."""
    gt_match: GtMatch
    """How the paper's evaluator pairs a tracked frame with GT."""
    max_dt_ns: int
    """Tolerance of ``nearest_time`` matching, nanoseconds (unused for ``frame_index``)."""
    gt_float32: bool
    """The paper's GT positions went through float32 (LingBot-Map TUM adapter; 7-Scenes ``pose.txt`` cast)."""
    frame_stride: int
    """Tracked frames are source frames 0, k, 2k, ... (7-Scenes: 5)."""
    timestamp_fallback: TimestampFallback
    """Timestamp rule for frames that VSLAM-LAB's rgb.csv does not list (TUM RGB frames without depth)."""
    units: Units
    """Units of the paper's table."""
    paper_table: str
    """Table and page of the headline number."""
    paper_value: float
    """Headline dataset mean in ``units`` (the upstream command's setting)."""
    upstream_cases: dict[str, str]
    """VSLAM-LAB sequence name -> upstream eval case name (names ``pose/<case>.npz`` and seeds DPVO)."""
    frame_root: str
    """Default parent of the MetricTrail frame directories (``<frame_root>/<case>``)."""


TUM: PaperProtocol = PaperProtocol(
    dataset="rgbdtum",
    benchmark_folder="RGBDTUM",
    rgb_hz=30.0,
    gt_match="nearest_time",
    max_dt_ns=20_000_000,
    gt_float32=True,
    frame_stride=1,
    timestamp_fallback="filename_seconds",
    units="cm",
    paper_table="Table 9, p. 20 (Uncalibrated w/ Global BA, Refine)",
    paper_value=4.4,
    upstream_cases={f"rgbd_dataset_freiburg1_{case}": case for case in ("360", "desk", "desk2", "floor", "plant", "room", "rpy", "teddy", "xyz")},
    frame_root="/mnt/nas/ssd-8tb/metrictrail/data/TUM/rgb",
)

SEVEN_SCENES: PaperProtocol = PaperProtocol(
    dataset="7scenes",
    benchmark_folder="7SCENES",
    rgb_hz=30.0,
    gt_match="frame_index",
    max_dt_ns=0,
    gt_float32=True,
    frame_stride=5,
    timestamp_fallback="none",
    units="cm",
    paper_table="Table 8, p. 19 (Ours (Uncalibrated) w/ Global BA)",
    paper_value=6.3,
    upstream_cases={
        name.replace("__", "_"): name
        for name in (
            "chess__seq-03",
            "chess__seq-05",
            "fire__seq-03",
            "fire__seq-04",
            "heads__seq-01",
            "office__seq-02",
            "office__seq-06",
            "office__seq-07",
            "office__seq-09",
            "pumpkin__seq-01",
            "pumpkin__seq-07",
            "redkitchen__seq-03",
            "redkitchen__seq-04",
            "redkitchen__seq-06",
            "redkitchen__seq-12",
            "redkitchen__seq-14",
            "stairs__seq-01",
            "stairs__seq-04",
        )
    },
    frame_root="/mnt/nas/ssd-8tb/metrictrail/data/7-Scenes/rgb",
)

KITTI: PaperProtocol = PaperProtocol(
    dataset="kitti",
    benchmark_folder="KITTI",
    rgb_hz=10.0,
    gt_match="frame_index",
    max_dt_ns=0,
    gt_float32=False,
    frame_stride=1,
    timestamp_fallback="none",
    units="m",
    paper_table="Tables 1/12, p. 8/22 (Ours (Uncalibrated))",
    paper_value=14.05,
    upstream_cases={f"{index:02d}": f"{index:02d}" for index in range(11)},
    frame_root="/mnt/nas/ssd-8tb/metrictrail/data/KITTI/rgb",
)

PROTOCOLS: dict[VslamlabDataset, PaperProtocol] = {protocol.dataset: protocol for protocol in (TUM, SEVEN_SCENES, KITTI)}
"""Paper protocol per VSLAM-LAB dataset."""


BENCHMARK_DATASETS: dict[str, VslamlabDataset] = {"tum": "rgbdtum", "7scenes": "7scenes", "kitti": "kitti", "kitti_calibrated": "kitti"}
"""Upstream benchmark preset name (``metrictrail.benchmarks``) -> the VSLAM-LAB dataset whose paper protocol scores it."""


def units_per_metre(units: Units) -> float:
    """Factor from metres to ``units``."""
    return 100.0 if units == "cm" else 1.0
