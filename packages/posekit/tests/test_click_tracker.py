"""Kineo-style single-object click tracking over a real clip (CUDA only)."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from posekit.apis.click_tracker import ClickTracker
from posekit.models.sam2_video import Sam2VideoSegmenterConfig

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
CLIP: Path = Path(__file__).resolve().parents[2] / "wilor-nano" / "assets" / "video.mp4"
CHEST: tuple[float, float] = (360.0, 450.0)


@pytest.fixture(scope="module")
def tracker() -> Iterator[ClickTracker]:
    # -ti drifts on point-only seeding (0.39–0.91 end scores); -s holds at 0.95+.
    click_tracker = ClickTracker(CLIP, Sam2VideoSegmenterConfig(variant="efficienttam-s-512").setup().predictor)
    yield click_tracker
    click_tracker.close()
    click_tracker.close()


@cuda_only
def test_add_point_masks_the_click_and_preview_follows(tracker: ClickTracker) -> None:
    tracker.clear()
    result = tracker.add_point(0, *CHEST, positive=True)
    assert result.frame_idx == 0 and result.score > 0.5
    assert bool(result.mask[int(CHEST[1]), int(CHEST[0])])
    body_area = int(result.mask.sum())
    preview = tracker.preview(30)
    assert preview is not None and preview.frame_idx == 30
    assert 0.5 * body_area < int(preview.mask.sum()) < 1.5 * body_area
    # Previews write no memory: the only memory is the conditional one on frame 0.
    assert tracker._state.memory_bank.count_non_conditional_memories() == 0


@cuda_only
def test_first_click_matches_explicit_resegment(tracker: ClickTracker) -> None:
    tracker.clear()
    first = tracker.add_point(0, *CHEST, positive=True)
    tracker.clear()
    replaced = tracker.add_point(0, *CHEST, positive=True, resegment=True)
    assert torch.equal(first.mask, replaced.mask)
    assert first.score == replaced.score


@cuda_only
def test_negative_point_shrinks_and_removal_restores(tracker: ClickTracker) -> None:
    tracker.clear()
    full = tracker.add_point(0, *CHEST, positive=True)
    shrunk = tracker.add_point(0, 320.0, 320.0, positive=False)  # on the face
    assert int(shrunk.mask.sum()) < int(full.mask.sum())
    removed, restored = tracker.remove_point_near(0, 325.0, 318.0)
    assert removed is not None and not removed.positive and restored is not None
    assert abs(int(restored.mask.sum()) - int(full.mask.sum())) < 0.02 * int(full.mask.sum())
    assert tracker.remove_point_near(0, 10.0, 10.0) == (None, None)


@cuda_only
def test_undo_last_point_clears_its_frame_memory(tracker: ClickTracker) -> None:
    tracker.clear()
    tracker.add_point(0, *CHEST, positive=True)
    tracker.add_point(40, *CHEST, positive=True)
    assert tracker.prompted_frames() == [0, 40]
    last, result = tracker.undo()
    assert last is not None and last.frame_idx == 40 and result is None
    assert tracker.prompted_frames() == [0]
    assert tracker._state.memory_bank.count_conditional_memories() == 1


@cuda_only
@pytest.mark.parametrize("click", [(353.0, 405.0), (390.0, 480.0), CHEST])
def test_track_holds_object_through_clip(tracker: ClickTracker, click: tuple[float, float]) -> None:
    tracker.clear()
    tracker.add_point(0, *click, positive=True)
    results = list(tracker.track())
    assert [r.frame_idx for r in results] == list(range(tracker.num_frames))
    assert results[-1].score > 0.7


@cuda_only
def test_refinement_preserves_propagated_object(tracker: ClickTracker) -> None:
    tracker.clear()
    tracker.add_point(0, *CHEST, positive=True)
    propagated = tracker.preview(200)
    assert propagated is not None
    area = int(propagated.mask.sum())

    refined = tracker.add_point(200, 330.0, 520.0, positive=True)
    assert int(refined.mask.sum()) >= 0.85 * area
    end = tracker.preview(328)
    assert end is not None and int(end.mask.sum()) >= 0.85 * area

    tracker.clear()
    tracker.add_point(0, *CHEST, positive=True)
    propagated = tracker.preview(200)
    assert propagated is not None
    area = int(propagated.mask.sum())
    excluded = tracker.add_point(200, 320.0, 320.0, positive=False)
    assert int(excluded.mask.sum()) >= 0.7 * area


@cuda_only
def test_track_runs_bidirectionally_with_confidence(tracker: ClickTracker) -> None:
    tracker.clear()
    tracker.add_point(60, *CHEST, positive=True)
    results = list(tracker.track())
    assert {result.frame_idx for result in results} == set(range(tracker.num_frames))
    by_frame = {result.frame_idx: result for result in results}
    assert by_frame[0].score > 0.5
    assert by_frame[328].score > 0.5
    assert all(0.0 <= result.object_score <= 1.0 for result in results)


@cuda_only
def test_decoding_from_other_threads_works(tracker: ClickTracker) -> None:
    # torchcodec's NVDEC decoder fails from any thread but its creator; web callbacks
    # run on pool threads, so the tracker must route decodes to its own thread.
    tracker.clear()
    tracker.add_point(0, *CHEST, positive=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(lambda f: tracker.frame(f).shape, [0, 30, 300, 5, 200, 100, 328, 50]))
        previews = list(pool.map(tracker.preview, [10, 300, 20]))
    assert all(tuple(shape) == (*tracker.frame_hw, 3) for shape in frames)
    assert all(p is not None and p.score > 0.5 for p in previews)
