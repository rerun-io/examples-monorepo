"""The shared hand logger keeps capture stamps while shifting only alignment."""

from pathlib import Path

import numpy as np
import pyarrow as pa
from rerun.chunk import RrdReader

from dataforge.datasets.selfcap import QuestHands, log_quest_hands
from dataforge.writing import recording_to


def test_quest_hands_logger_shifts_only_video_time(tmp_path: Path) -> None:
    target = tmp_path / "hands.rrd"
    hands = QuestHands(np.array([17, 27], dtype=np.int64), np.ones((2, 133, 3), dtype=np.float32))
    with recording_to(target, recording_id="hands", send_properties=False) as recording:
        log_quest_hands(recording, hands, shift_ns=61_300_000)
    reader = RrdReader(target)
    entities: set[str] = set()
    for chunk in reader.stream():
        entities.add(str(chunk.entity_path))
        if chunk.is_static:
            continue
        batch = chunk.to_record_batch()
        assert batch.column("capture_time").cast(pa.int64()).to_pylist() == [17, 27]
        assert batch.column("video_time").cast(pa.int64()).to_pylist() == [61_300_017, 61_300_027]
    assert {"/world/gt", "/world/gt/coco133_xyz"} <= entities
