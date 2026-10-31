"""Lossless Quest hand CSV boundary, including the optional OpenXR orientations."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from jaxtyping import Float32, Int64
from numpy import ndarray
from rerun.catalog import DatasetEntry

from dataforge.schema import CAPTURE_TIMELINE

QUEST_HAND_SIDES: tuple[str, str] = ("left", "right")
QUEST_HANDS_ROOT: str = "/world/gt/quest_hands"
"""The Quest hand tracking as recorded in a base layer: one entity per hand, 26 OpenXR joint poses per sample."""


def quest_hand_path(side: str) -> str:
    """``/world/gt/quest_hands/{left,right}``."""
    return f"{QUEST_HANDS_ROOT}/{side}"


@dataclass(frozen=True, slots=True)
class QuestJointPoses:
    times_ns: Int64[ndarray, "n"]
    """Raw episode-relative clock, without latency correction."""
    positions: Float32[ndarray, "n 26 3"]
    """All OpenXR joint positions, including palm and metacarpals."""
    orientations: Float32[ndarray, "n 26 4"] | None
    """Raw world-frame XYZW orientations, or None for position-only CSVs."""
    joint_names: tuple[str, ...] = ()
    """OpenXR joint names in CSV order (``palm``, ``wrist``, ``thumb_metacarpal``, …), from the header."""


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
    names = tuple(name.rsplit("_", 1)[0] for name in table.column_names[1::width])
    return QuestJointPoses(times, joints[:, :, :3], joints[:, :, 3:7] if width == 7 else None, names)


def joint_poses_from_table(table: pa.Table, entity: str, index: str, joint_names: tuple[str, ...] = ()) -> QuestJointPoses:
    """Read one hand back from a Rerun query (``InstancePoses3D`` rows on ``entity``, ``index`` as the clock).

    The inverse of the base layer's hand logging: an empty pose batch is an untracked sample and becomes NaN, exactly
    as the CSV reader turns an all-zero row into NaN.
    """
    translations, quaternions = f"{entity}:InstancePoses3D:translations", f"{entity}:InstancePoses3D:quaternions"
    rows = table.filter(pc.is_valid(table[translations])).sort_by(index)
    times = rows[index].combine_chunks().cast(pa.int64()).to_numpy()
    positions = np.full((len(times), 26, 3), np.nan, dtype=np.float32)
    has_orientation = quaternions in rows.column_names and rows[quaternions].null_count < len(rows)
    orientations = np.full((len(times), 26, 4), np.nan, dtype=np.float32) if has_orientation else None
    for i, (xyz, quat) in enumerate(zip(rows[translations].to_pylist(),
                                        rows[quaternions].to_pylist() if has_orientation else [None] * len(times), strict=True)):
        if xyz:
            positions[i] = np.asarray(xyz, dtype=np.float32).reshape(26, 3)
            if orientations is not None and quat:
                orientations[i] = np.asarray(quat, dtype=np.float32).reshape(26, 4)
    return QuestJointPoses(times, positions, orientations, joint_names)


def read_segment_hand_poses(segment: DatasetEntry) -> dict[str, QuestJointPoses]:
    """A catalog segment's Quest joint poses (one query), on ``capture_time``: the raw Quest stamps."""
    entities = {side: quest_hand_path(side) for side in QUEST_HAND_SIDES}
    table = segment.filter_contents(list(entities.values())).reader(index=CAPTURE_TIMELINE).to_arrow_table()
    return {
        side: joint_poses_from_table(table, entity, CAPTURE_TIMELINE)
        for side, entity in entities.items()
        if f"{entity}:InstancePoses3D:translations" in table.column_names
    }
