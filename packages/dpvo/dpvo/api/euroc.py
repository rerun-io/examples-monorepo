"""EuRoC MAV dataset reader for DPV-SLAM evaluation.

Provides utilities to load EuRoC sequences (image paths, timestamps,
ground truth trajectories) and a download helper for the Machine Hall
sequences (MH_01 through MH_05) which contain loop revisits.

EuRoC dataset: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
"""

import glob
import os
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from evo.core.trajectory import PoseTrajectory3D
from jaxtyping import Float64
from numpy import ndarray

# EuRoC cam0 intrinsics (after undistortion) — same for all sequences
EUROC_CAM0_INTRINSICS: Float64[ndarray, "4"] = np.array([458.654, 457.296, 367.215, 248.375], dtype=np.float64)

EUROC_MH_SEQUENCES: list[str] = [
    "MH_01_easy",
    "MH_02_easy",
    "MH_03_medium",
    "MH_04_difficult",
    "MH_05_difficult",
]

EUROC_ALL_SEQUENCES: list[str] = [
    *EUROC_MH_SEQUENCES,
    "V1_01_easy",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_02_medium",
    "V2_03_difficult",
]

# ETH Research Collection URLs for EuRoC dataset (grouped by category)
# These are more reliable than the old robotics.ethz.ch per-sequence URLs
_EUROC_CATEGORY_URLS = {
    "MH": "https://www.research-collection.ethz.ch/server/api/core/bitstreams/7b2419c1-62b5-4714-b7f8-485e5fe3e5fe/content",
    "V1": "https://www.research-collection.ethz.ch/server/api/core/bitstreams/02ecda9a-298f-498b-970c-b7c44334d880/content",
    "V2": "https://www.research-collection.ethz.ch/server/api/core/bitstreams/ea12bc01-3677-4b4c-853d-87c7870b8c44/content",
}


@dataclass
class EuRoCSequence:
    """A parsed EuRoC MAV sequence ready for evaluation."""

    name: str
    image_dir: str
    image_files: list[str]
    timestamps_ns: Float64[ndarray, "n"]
    intrinsics: Float64[ndarray, "4"]
    gt_traj: PoseTrajectory3D


def load_euroc_groundtruth(csv_path: str | Path) -> PoseTrajectory3D:
    """Load EuRoC ground truth from the state estimate CSV.

    EuRoC CSV format (after comment lines):
    ``timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,...``

    Timestamps are nanoseconds. Quaternions are wxyz order.
    """
    data = np.loadtxt(str(csv_path), delimiter=",", comments="#")
    timestamps_ns = data[:, 0]
    timestamps_sec = timestamps_ns / 1e9
    positions = data[:, 1:4]  # p_x, p_y, p_z
    quaternions_wxyz = data[:, 4:8]  # q_w, q_x, q_y, q_z

    return PoseTrajectory3D(
        positions_xyz=positions,
        orientations_quat_wxyz=quaternions_wxyz,
        timestamps=timestamps_sec,
    )


def load_euroc_sequence(data_root: str | Path, seq_name: str) -> EuRoCSequence:
    """Load a single EuRoC sequence.

    Expected layout::

        data_root/seq_name/mav0/cam0/data/*.png
        data_root/seq_name/mav0/state_groundtruth_estimate0/data.csv
    """
    data_root = Path(data_root)
    seq_dir = data_root / seq_name
    image_dir = str(seq_dir / "mav0" / "cam0" / "data")
    gt_csv = seq_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv"

    assert os.path.isdir(image_dir), f"Image directory not found: {image_dir}"
    assert gt_csv.exists(), f"Ground truth CSV not found: {gt_csv}"

    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    assert len(image_files) > 0, f"No PNG images found in {image_dir}"

    # Extract nanosecond timestamps from filenames
    timestamps_ns = np.array([float(Path(f).stem) for f in image_files])

    gt_traj = load_euroc_groundtruth(gt_csv)

    return EuRoCSequence(
        name=seq_name,
        image_dir=image_dir,
        image_files=image_files,
        timestamps_ns=timestamps_ns,
        intrinsics=EUROC_CAM0_INTRINSICS.copy(),
        gt_traj=gt_traj,
    )


def write_euroc_calib(output_path: str | Path) -> None:
    """Write EuRoC cam0 intrinsics to a calibration text file.

    Format: ``fx fy cx cy`` on a single line (matches ``stream.load_calib``).
    """
    np.savetxt(str(output_path), EUROC_CAM0_INTRINSICS.reshape(1, -1), fmt="%.6f")


def download_euroc(
    data_root: str | Path = "data/euroc",
    sequences: list[str] | None = None,
) -> None:
    """Download EuRoC MAV sequences from ETH Research Collection.

    Downloads category-level zip archives (MH, V1, V2) and extracts
    the requested sequences.  Skips sequences that already exist on disk.

    Args:
        data_root: Directory to store extracted sequences.
        sequences: List of sequence names to download. Defaults to MH_01-MH_05.
    """
    if sequences is None:
        sequences = list(EUROC_MH_SEQUENCES)

    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    # Group requested sequences by category to minimize downloads
    needed_categories: dict[str, list[str]] = {}
    for seq_name in sequences:
        seq_dir = data_root / seq_name
        if (seq_dir / "mav0" / "cam0" / "data").is_dir():
            print(f"  {seq_name}: already exists, skipping")
            continue
        prefix = seq_name[:2]
        needed_categories.setdefault(prefix, []).append(seq_name)

    for prefix, seq_names in needed_categories.items():
        url = _EUROC_CATEGORY_URLS.get(prefix)
        if url is None:
            print(f"  Unknown category prefix '{prefix}' for sequences: {seq_names}")
            continue

        print(f"  Downloading {prefix} category ({', '.join(seq_names)}) ...")
        with TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, f"{prefix}.zip")
            subprocess.run(["curl", "-L", "--progress-bar", "-o", zip_path, url], check=True)

            # Category zip extracts to category_name/seq_name/{seq.bag, seq.zip}
            print("  Extracting category archive ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)

            # Extract each sequence's inner zip (contains mav0/ structure)
            for seq_name in seq_names:
                inner_matches = glob.glob(os.path.join(tmpdir, "**", f"{seq_name}.zip"), recursive=True)
                if not inner_matches:
                    print(f"  {seq_name}: inner zip not found, skipping")
                    continue

                print(f"  Extracting {seq_name} ...")
                seq_dir = data_root / seq_name
                seq_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(inner_matches[0], "r") as zf:
                    zf.extractall(str(seq_dir))

                assert (seq_dir / "mav0" / "cam0" / "data").is_dir(), f"Extraction failed for {seq_name}"
                print(f"  {seq_name}: done")
