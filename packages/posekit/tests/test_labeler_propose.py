"""The labeler's proposer: detections -> proposals, the hand_proposals layer rrd, and reading proposals back."""

import socket
import stat
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa
import pytest
import torch
from rerun.chunk import RrdReader

from posekit.labeler.paths import proposal_path
from posekit.labeler.proposals import SCORE_ARCHETYPE, egohand_proposals, hands_layer_proposals, keynet_path, read_proposals
from posekit.labeler.propose import (
    concat_detections,
    detections_to_proposals,
    proposal_label,
    raw_detections,
    resolve_each_frame,
    write_proposal_layer,
)
from posekit.labeler.schema import Box, CameraInfo, Proposal
from posekit.models.egohand import EgoHandDetector, EgoHandDetectorConfig
from posekit.predictions import BoxDetections, HandDetections

CAMS: list[CameraInfo] = [CameraInfo(cam=0, name="left_front", width=1920, height=1080), CameraInfo(cam=1, name="right_front", width=1920, height=1080)]
GRID: np.ndarray = np.array([1_000_000_000, 1_200_000_000, 1_400_000_000], dtype=np.int64)


def _detections(rows: list[tuple[list[float], float, int, str]]) -> HandDetections:
    return HandDetections(
        BoxDetections(
            torch.tensor([row[0] for row in rows], dtype=torch.float32).reshape(-1, 4),
            torch.tensor([row[1] for row in rows], dtype=torch.float32),
            torch.tensor([row[2] for row in rows], dtype=torch.int64),
        ),
        tuple(row[3] for row in rows),  # type: ignore[arg-type]
    )


def test_detections_become_clipped_proposals_on_their_grid_times() -> None:
    found = _detections([([10.0, 20.0, 110.0, 220.0], 0.9, 0, "left"), ([1900.0, 1000.0, 2000.0, 1200.0], 0.7, 2, "right"), ([-50.0, 5.0, -1.0, 9.0], 0.8, 2, "left")])
    proposals = detections_to_proposals(found, GRID, CAMS[1])
    assert sorted(proposals) == GRID.tolist()
    assert proposals[1_200_000_000] == []
    assert proposals[1_000_000_000] == [Proposal(cam=1, side="left", box=Box(10.0, 20.0, 110.0, 220.0), score=0.9, model="egohand")]
    # Clipped to the image; the box fully left of the image is dropped.
    assert [p.box for p in proposals[1_400_000_000]] == [Box(1900.0, 1000.0, 1920.0, 1080.0)]
    assert proposal_label(proposals[1_000_000_000][0]) == "L 0.90"


def test_concat_shifts_frame_indices_by_batch_offset() -> None:
    first = _detections([([0.0, 0.0, 1.0, 1.0], 0.9, 1, "left")])
    second = _detections([([0.0, 0.0, 2.0, 2.0], 0.8, 0, "right"), ([0.0, 0.0, 3.0, 3.0], 0.7, 3, "left")])
    joined = concat_detections([first, HandDetections.empty("cpu"), second], [0, 4, 8], "cpu")
    assert joined.boxes.frame_indices.tolist() == [1, 8, 11]
    assert joined.sides == ("left", "right", "left")
    assert concat_detections([HandDetections.empty("cpu")], [0], "cpu").num_detections == 0


def test_each_frame_keeps_at_most_one_box_per_side_and_its_frame_index() -> None:
    raw = _detections(
        [
            ([100.0, 100.0, 200.0, 200.0], 0.9, 0, "left"),
            ([110.0, 100.0, 210.0, 200.0], 0.8, 0, "left"),
            ([800.0, 100.0, 900.0, 200.0], 0.9, 0, "right"),
            ([500.0, 500.0, 600.0, 600.0], 0.9, 3, "right"),
        ]
    )
    resolved = resolve_each_frame(raw)
    frames = resolved.boxes.frame_indices.tolist()
    assert sorted(set(frames)) == [0, 3]
    for frame in (0, 3):
        sides = [side for side, f in zip(resolved.sides, frames, strict=True) if f == frame]
        assert len(sides) == len(set(sides))
    assert sorted(side for side, f in zip(resolved.sides, frames, strict=True) if f == 0) == ["left", "right"]
    assert resolve_each_frame(HandDetections.empty("cpu")).num_detections == 0


class _FakeRuntime:
    """Two frames, three queries: class 1 (left) and 2 (right) above threshold, one low score."""

    spec = SimpleNamespace(max_batch_size=8)

    def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch = inputs["images"].shape[0]
        labels = torch.tensor([[1, 2, 2]] * batch, dtype=torch.int64)
        scores = torch.tensor([[0.9, 0.8, 0.2]] * batch, dtype=torch.float32)
        boxes = torch.tensor([[[10.0, 10.0, 50.0, 60.0], [100.0, 10.0, 150.0, 60.0], [0.0, 0.0, 5.0, 5.0]]] * batch)
        return {"labels": labels, "scores": scores, "boxes": boxes}


def test_raw_detections_keep_both_classes_above_threshold() -> None:
    detector = EgoHandDetector.__new__(EgoHandDetector)
    detector.config = EgoHandDetectorConfig(gpu_preprocessing=True, threshold=0.5)
    detector.runtime = _FakeRuntime()  # type: ignore[assignment]
    found = raw_detections(detector, torch.zeros((2, 48, 64, 3), dtype=torch.uint8))
    assert found.num_detections == 4
    assert found.sides == ("left", "right", "left", "right")
    assert found.boxes.frame_indices.tolist() == [0, 0, 1, 1]


def _reader_table(columns: dict[str, list], times: list[int]) -> pa.Table:
    data = {"rerun_segment_id": pa.array(["seg"] * len(times)), "video_time": pa.array(times, pa.duration("ns"))}
    vec2 = pa.list_(pa.list_(pa.float32(), 2))
    for name, values in columns.items():
        if name.endswith(("centers", "half_sizes", "positions")):
            data[name] = pa.array(values, vec2)
        elif name.endswith("class_ids"):
            data[name] = pa.array(values, pa.list_(pa.uint16()))
        else:
            data[name] = pa.array(values, pa.list_(pa.float32()))
    return pa.table(data)


def test_egohand_rows_parse_into_proposals() -> None:
    path = proposal_path(0, "egohand")
    table = _reader_table(
        {
            f"{path}:Boxes2D:centers": [[[60.0, 120.0], [300.0, 300.0]], [], None],
            f"{path}:Boxes2D:half_sizes": [[[50.0, 100.0], [20.0, 20.0]], [], None],
            f"{path}:Boxes2D:class_ids": [[100, 101], [], None],
            f"{path}:{SCORE_ARCHETYPE}:score": [[0.9, 0.6], [], None],
        },
        [1_000_000_000, 1_200_000_000, 1_400_000_000],
    )
    proposals = egohand_proposals(table, CAMS, GRID)
    assert [(p.side, p.box, p.model) for p in proposals[1_000_000_000]] == [
        ("left", Box(10.0, 20.0, 110.0, 220.0), "egohand"),
        ("right", Box(280.0, 280.0, 320.0, 320.0), "egohand"),
    ]
    assert proposals[1_200_000_000] == [] and proposals[1_400_000_000] == []


def test_hands_layer_keynet_points_make_boxes_at_the_nearest_grid_time() -> None:
    path = keynet_path(1, "right")
    square = [[500.0, 500.0], [600.0, 500.0], [600.0, 600.0], [500.0, 600.0]]
    table = _reader_table(
        {
            f"{path}:Points2D:positions": [square, square, square, square],
            f"{path}:simplecv.KeypointConfidence2D:average_confidence": [[0.9], [0.95], [0.2], [0.8]],
        },
        # 26 us after the first grid time; the second row lies between grid times; the third is not confident.
        [1_000_026_000, 1_100_000_000, 1_200_000_000, 1_400_000_000 + 30_000_000],
    )
    proposals = hands_layer_proposals(table, CAMS, GRID)
    assert proposals[1_000_000_000] == [Proposal(cam=1, side="right", box=Box(485.0, 485.0, 615.0, 615.0), score=0.9, model="hands_layer")]
    assert proposals[1_200_000_000] == [] and proposals[1_400_000_000] == []


def _proposals() -> dict[int, list[Proposal]]:
    return {
        1_000_000_000: [
            Proposal(cam=0, side="left", box=Box(10.0, 20.0, 110.0, 220.0), score=0.9, model="egohand"),
            Proposal(cam=1, side="right", box=Box(300.0, 300.0, 400.0, 420.0), score=0.75, model="egohand"),
        ],
        1_200_000_000: [],
        1_400_000_000: [Proposal(cam=0, side="right", box=Box(500.0, 500.0, 600.0, 650.0), score=0.5, model="egohand")],
    }


def test_layer_is_written_atomically_with_the_base_ids(tmp_path: Path) -> None:
    path = tmp_path / "layers" / "seg.rrd"
    write_proposal_layer(path, application_id="dataforge", segment_id="seg", cams=CAMS, proposals=_proposals(), properties={"model": "egohand"})
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert [p.name for p in path.parent.iterdir()] == ["seg.rrd"]
    recordings = RrdReader(path).recordings()
    assert [(r.application_id, r.recording_id) for r in recordings] == [("dataforge", "seg")]


@pytest.mark.integration
def test_layer_round_trips_through_a_catalog(tmp_path: Path) -> None:
    from rerun.server import Server

    path = tmp_path / "seg.rrd"
    write_proposal_layer(path, application_id="dataforge", segment_id="seg", cams=CAMS, proposals=_proposals(), properties={"model": "egohand"})
    with Server(port=0, datasets={"labeling": [path]}) as server:
        dataset = server.client().get_dataset("labeling")
        read = read_proposals(dataset, "seg", CAMS, GRID.tolist())
    expected = _proposals()
    assert sorted(read) == sorted(expected)
    for t_ns, proposals in expected.items():
        assert sorted(read[t_ns], key=lambda p: (p.cam, p.side)) == sorted(proposals, key=lambda p: (p.cam, p.side))


CATALOG_URL: str = "rerun+http://127.0.0.1:9897"
CAP_B_SEGMENT: str = "robocap__fe6fede545c972fa__s00000013"


@pytest.mark.integration
def test_labeling_catalog_serves_both_proposers_on_a_cap_b_segment() -> None:
    parsed = urlparse(CATALOG_URL.replace("rerun+", ""))
    try:
        socket.create_connection((parsed.hostname, parsed.port), timeout=1.0).close()
    except OSError:
        pytest.skip(f"labeling catalog {CATALOG_URL} (robocap-labeling with the hands and hand_proposals layers) is not reachable")
    from rerun.catalog import CatalogClient

    from posekit.labeler.frames import FrameSource

    dataset = CatalogClient(CATALOG_URL).get_dataset("robocap-labeling")
    rows = [row for row in dataset.segment_table().to_arrow_table().to_pylist() if row["rerun_segment_id"] == CAP_B_SEGMENT]
    if not rows or "hand_proposals" not in rows[0]["rerun_layer_names"]:
        pytest.skip("the hand_proposals layer is not registered yet (run tools/apps/label_propose.py)")
    source = FrameSource.from_catalog(dataset, CAP_B_SEGMENT)
    proposals = read_proposals(dataset, CAP_B_SEGMENT, source.cams, source.times_ns)
    models = {p.model for found in proposals.values() for p in found}
    assert models == {"egohand", "hands_layer"}
    assert sorted(proposals) == source.times_ns
