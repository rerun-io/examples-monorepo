"""The visual carrier's run rules with a fake tracker and fake frames (posekit.labeler.carry); no OpenCV tracker runs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from jaxtyping import UInt8
from numpy import ndarray

from posekit.labeler.carry import VisualCarrier, clip_carried
from posekit.labeler.schema import Box

STEP: int = 200_000_000
TIMES: list[int] = [5_000 + i * STEP for i in range(30)]
WIDTH: int = 400
HEIGHT: int = 200
PER_STEP: int = 2
"""Video frames per grid step (grid time ``i`` shows frame ``2 i``)."""


class Frames:
    """Grey frames; the camera has no frame at the grid indices in ``missing``; records every decode pass."""

    def __init__(self, missing: frozenset[int] = frozenset()) -> None:
        self.missing: frozenset[int] = missing
        self.passes: list[tuple[int, int, int]] = []

    def sample_index(self, cam: int, t_ns: int) -> int:
        if TIMES.index(t_ns) in self.missing:
            raise KeyError(t_ns)
        return PER_STEP * TIMES.index(t_ns)

    def frame_count(self, cam: int) -> int:
        return PER_STEP * len(TIMES)

    def frame_range(self, cam: int, first: int, stop: int) -> UInt8[ndarray, "b h w 3"]:
        self.passes.append((cam, first, stop))
        return np.full((stop - first, HEIGHT, WIDTH, 3), 90, dtype=np.uint8)


class Slide:
    """Fake tracker: the box moves ``dx`` tracking px right per update; reports failure on update ``fail_on``."""

    def __init__(self, dx: int = 5, fail_on: int | None = None) -> None:
        self.dx: int = dx
        self.fail_on: int | None = fail_on
        self.rect: list[int] = [0, 0, 0, 0]
        self.updates: int = 0

    def init(self, image: UInt8[ndarray, "h w 3"], bounding_box: Sequence[int], /) -> None:
        assert image.shape == (HEIGHT // 2, WIDTH // 2, 3)  # tracked at half size
        self.rect = list(bounding_box)

    def update(self, image: UInt8[ndarray, "h w 3"], /) -> tuple[bool, Sequence[int]]:
        self.updates += 1
        if self.updates == self.fail_on:
            return False, [0, 0, 0, 0]
        self.rect = [self.rect[0] + self.dx, *self.rect[1:]]
        return True, self.rect


def test_a_run_carries_the_box_and_answers_again_without_tracking() -> None:
    frames: Frames = Frames()
    trackers: list[Slide] = []

    def make() -> Slide:
        trackers.append(Slide())
        return trackers[-1]

    carrier: VisualCarrier = VisualCarrier(frames, TIMES, make)
    start: Box = Box(20.0, 40.0, 60.0, 80.0)
    # The tracker sees every frame: 2 updates of 5 tracking px per grid step = 20 native px.
    assert carrier.carry(0, TIMES[2], start, TIMES[5]) == Box(80.0, 40.0, 120.0, 80.0)
    assert carrier.carry(0, TIMES[2], start, TIMES[3]) == Box(40.0, 40.0, 80.0, 80.0)
    assert len(trackers) == 1 and trackers[0].updates == 6
    # One forward pass decoded every frame from the start frame to ten grid steps later.
    assert frames.passes == [(0, 4, 25)]
    # Another run on the same camera reuses those frames; another camera decodes its own.
    carrier.carry(0, TIMES[4], start, TIMES[6])
    carrier.carry(1, TIMES[4], start, TIMES[5])
    assert frames.passes == [(0, 4, 25), (1, 8, 29)]
    with pytest.raises(ValueError, match="grid"):
        carrier.carry(0, TIMES[4], start, TIMES[4])


def test_failure_and_missing_frames_lose_the_box_for_good() -> None:
    carrier: VisualCarrier = VisualCarrier(Frames(), TIMES, lambda: Slide(fail_on=3))
    start: Box = Box(20.0, 40.0, 60.0, 80.0)
    assert carrier.carry(0, TIMES[0], start, TIMES[1]) is not None
    assert carrier.carry(0, TIMES[0], start, TIMES[2]) is None  # the third update (in step 2) fails
    assert carrier.carry(0, TIMES[0], start, TIMES[3]) is None
    gap: VisualCarrier = VisualCarrier(Frames(missing=frozenset({2})), TIMES, lambda: Slide())
    assert gap.carry(0, TIMES[0], start, TIMES[1]) is not None
    assert gap.carry(0, TIMES[0], start, TIMES[3]) is None
    assert gap.carry(0, TIMES[2], start, TIMES[3]) is None  # no start frame


def test_a_box_that_leaves_the_image_is_lost() -> None:
    # 20 native px per step from x0=320 (box 60 wide, image 400 wide): x1 = 400 at step 1 (inside), 420 at step 2
    # (2/3 inside, clipped), 440 at step 3 (1/3 inside: lost on the frame where it falls below a half).
    carrier: VisualCarrier = VisualCarrier(Frames(), TIMES, lambda: Slide())
    start: Box = Box(320.0, 40.0, 380.0, 80.0)
    assert carrier.carry(0, TIMES[0], start, TIMES[1]) == Box(340.0, 40.0, 400.0, 80.0)
    assert carrier.carry(0, TIMES[0], start, TIMES[2]) == Box(360.0, 40.0, 400.0, 80.0)
    assert carrier.carry(0, TIMES[0], start, TIMES[3]) is None


def test_clip_carried() -> None:
    assert clip_carried([10.0, 10.0, 20.0, 20.0], 0.5, WIDTH, HEIGHT) == Box(20.0, 20.0, 60.0, 60.0)
    assert clip_carried([-5.0, 10.0, 20.0, 20.0], 0.5, WIDTH, HEIGHT) == Box(0.0, 20.0, 30.0, 60.0)  # 3/4 inside
    assert clip_carried([-15.0, 10.0, 20.0, 20.0], 0.5, WIDTH, HEIGHT) is None  # 1/4 inside
    assert clip_carried([10.0, 10.0, 1.0, 20.0], 0.5, WIDTH, HEIGHT) is None  # 2 px wide


# -- integration: NanoTrack on a real RoboCap segment (read-only use of the labeling catalog) -------------------------

CATALOG_URL: str = "rerun+http://127.0.0.1:9897"
SEGMENT: str = "robocap__fe6fede545c972fa__s00000013"


@pytest.mark.integration
def test_nanotrack_carries_egohand_boxes_on_robocap_deterministically() -> None:
    import socket
    import time
    import urllib.error
    from urllib.parse import urlparse

    from posekit.labeler.carry import nanotrack_file, opencv_tracker
    from posekit.labeler.frames import FrameSource
    from posekit.labeler.proposals import iou, read_proposals
    from posekit.labeler.schema import Proposal
    from posekit.labeler.tracker import CARRY_STEPS

    parsed = urlparse(CATALOG_URL.replace("rerun+", ""))
    try:
        socket.create_connection((parsed.hostname, parsed.port), timeout=1.0).close()
    except OSError:
        pytest.skip(f"labeling catalog {CATALOG_URL} (dataset robocap-labeling) is not reachable")
    try:
        nanotrack_file("nanotrack_backbone_sim")
        nanotrack_file("nanotrack_head_sim")
    except urllib.error.URLError as error:
        pytest.skip(f"NanoTrack v2 ONNX files could not be downloaded: {error}")
    from rerun.catalog import CatalogClient

    dataset = CatalogClient(CATALOG_URL).get_dataset("robocap-labeling")
    source: FrameSource = FrameSource.from_catalog(dataset, SEGMENT)
    times: list[int] = source.times_ns
    proposals = read_proposals(dataset, SEGMENT, source.cams, times)
    starts: list[tuple[int, Proposal]] = []
    for index in range(0, len(times) - CARRY_STEPS, 40):
        confident: list[Proposal] = [p for p in proposals.get(times[index], []) if p.model == "egohand" and p.score >= 0.85]
        if confident:
            starts.append((index, max(confident, key=lambda p: p.score)))
    assert len(starts) >= 4
    carrier: VisualCarrier = VisualCarrier(source, times, opencv_tracker("nano"))
    again: VisualCarrier = VisualCarrier(source, times, opencv_tracker("nano"))
    overlaps: list[float] = []
    for index, start in starts[:4]:
        began: float = time.perf_counter()
        boxes: list[Box | None] = [carrier.carry(start.cam, times[index], start.box, times[index + k]) for k in range(1, CARRY_STEPS + 1)]
        cold_s: float = time.perf_counter() - began
        began = time.perf_counter()
        assert [carrier.carry(start.cam, times[index], start.box, times[index + k]) for k in range(1, CARRY_STEPS + 1)] == boxes
        warm_s: float = time.perf_counter() - began
        # Same frames, same initialisation -> the same boxes from a fresh carrier.
        assert [again.carry(start.cam, times[index], start.box, times[index + k]) for k in range(1, CARRY_STEPS + 1)] == boxes
        assert boxes[0] is not None
        assert cold_s < 2.0 and warm_s < 0.05, (cold_s, warm_s)
        for k, box in enumerate(boxes, start=1):
            redetected: list[Proposal] = [p for p in proposals.get(times[index + k], []) if p.model == "egohand" and p.cam == start.cam]
            if box is not None and redetected:
                assert 0.0 <= box.x0 < box.x1 <= 1920.0 and 0.0 <= box.y0 < box.y1 <= 1080.0
                overlaps.append(max(iou(box, p.box) for p in redetected))
    # The carried box sits on a hand EgoHand finds again in most steps.
    assert len(overlaps) >= 10
    assert sum(overlap >= 0.3 for overlap in overlaps) >= 0.7 * len(overlaps), overlaps
