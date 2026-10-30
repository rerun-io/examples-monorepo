"""Lossless Quest hand CSV boundary, including the optional OpenXR orientations."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
from jaxtyping import Float32, Int64
from numpy import ndarray


@dataclass(frozen=True, slots=True)
class QuestJointPoses:
    times_ns: Int64[ndarray, "n"]
    """Raw episode-relative clock, without latency correction."""
    positions: Float32[ndarray, "n 26 3"]
    """All OpenXR joint positions, including palm and metacarpals."""
    orientations: Float32[ndarray, "n 26 4"] | None
    """Raw world-frame XYZW orientations, or None for position-only CSVs."""


def read_quest_joint_poses(source: Path) -> QuestJointPoses:
    """Preserve the 3- or 7-wide source layout; zero tracking rows become missing."""
    try:
        table = pacsv.read_csv(source, convert_options=pacsv.ConvertOptions(column_types={"ts_ns": pa.int64()}))
        width = (len(table.column_names) - 1) // 26
        if table.column_names[0] != "ts_ns" or width not in (3, 7) or len(table.column_names) != 1 + 26 * width:
            raise ValueError("expected ts_ns followed by 26 xyz or xyz+xyzw joints")
        times = table.column(0).to_numpy().astype(np.int64)
        joints = np.column_stack([column.to_numpy() for column in table.columns[1:]]).astype(np.float32).reshape(-1, 26, width)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, OSError, ValueError, TypeError) as error:
        raise ValueError(f"{source}: invalid Quest joint CSV: {error}") from error
    if len(np.unique(times)) != len(times):
        raise ValueError(f"{source}: duplicate hand timestamp")
    missing = np.abs(joints[:, :, :3]).sum(axis=(1, 2)) == 0
    joints[missing] = np.nan
    return QuestJointPoses(times, joints[:, :, :3], joints[:, :, 3:7] if width == 7 else None)
