"""People tracks on real Insta360 lens frames (GPU, SAM 3 + EfficientTAM weights, an Insta360 base recording)."""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

BASE_RRD: str | None = os.environ.get("POSEKIT_PEOPLE_MASKS_RRD")
"""A recording whose two lenses show the camera operator close to the camera."""


def test_people_tracker_keeps_the_operator_between_redetects() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    if BASE_RRD is None or not Path(BASE_RRD).exists():
        pytest.skip(f"asset missing: Insta360 base .rrd (set POSEKIT_PEOPLE_MASKS_RRD; got {BASE_RRD})")
    import rerun as rr
    from simplecv.catalog_video import read_catalog_videos

    from posekit.apis.people_masks import MaskOptions
    from posekit.apis.people_tracks import PeopleTracker, StreamDecoder, TrackOptions
    from posekit.ops.people_tracks import mask_iou

    entities: list[str] = ["/world/rig_00/cam_00/pinhole/video", "/world/rig_00/cam_01/pinhole/video"]
    with rr.server.Server(datasets={"base": [str(BASE_RRD)]}) as server:
        dataset = server.client().get_dataset("base")
        videos = read_catalog_videos(dataset, dataset.segment_ids()[0], entities, "video_time")
    indices: list[int] = [0, 10, 20, 30]
    decoders = [StreamDecoder(v, 0, 31, "cuda") for v in videos]
    frames = torch.stack([d.frames(indices) for d in decoders])
    tracker = PeopleTracker(TrackOptions(redetect_every=3), MaskOptions(), "cuda")

    unions = [tracker.step(frames[:, i]) for i in range(len(indices))]
    reference = torch.from_numpy(tracker.masker(frames[:, 2])).cuda()  # SAM 3 on the last tracker-only step

    assert tracker.redetects == 2  # steps 0 and 3
    assert all(u.shape == (2, 1920, 1920) for u in unions)
    assert all(len(live) >= 1 for live in tracker.live)
    iou = mask_iou(unions[2], reference).diagonal()  # per lens, tracker output vs SAM 3 at frame 20
    assert float(iou.min()) > 0.6


def test_prefetched_chunks_match_the_sequential_decode() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    if BASE_RRD is None or not Path(BASE_RRD).exists():
        pytest.skip(f"asset missing: Insta360 base .rrd (set POSEKIT_PEOPLE_MASKS_RRD; got {BASE_RRD})")
    import rerun as rr
    from simplecv.catalog_video import read_catalog_videos

    from posekit.apis.people_tracks import StreamDecoder, prefetched_chunks

    entities: list[str] = ["/world/rig_00/cam_00/pinhole/video", "/world/rig_00/cam_01/pinhole/video"]
    with rr.server.Server(datasets={"base": [str(BASE_RRD)]}) as server:
        dataset = server.client().get_dataset("base")
        videos = read_catalog_videos(dataset, dataset.segment_ids()[0], entities, "video_time")
    chunks: list[list[int]] = [[10, 20, 30], [40, 50, 60], [70]]
    decoders = [StreamDecoder(v, 10, 71, "cuda") for v in videos]
    expected = [torch.stack([d.frames(chunk) for d in decoders]) for chunk in chunks]

    got = list(prefetched_chunks(videos, chunks, 10, 71, "cuda"))

    assert len(got) == len(chunks)
    assert all(torch.equal(g, e) for g, e in zip(got, expected, strict=True))


def test_people_tracker_lookahead_gives_the_same_masks() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    if BASE_RRD is None or not Path(BASE_RRD).exists():
        pytest.skip(f"asset missing: Insta360 base .rrd (set POSEKIT_PEOPLE_MASKS_RRD; got {BASE_RRD})")
    import rerun as rr
    from simplecv.catalog_video import read_catalog_videos

    from posekit.apis.people_masks import MaskOptions
    from posekit.apis.people_tracks import PeopleTracker, StreamDecoder, TrackOptions

    entities: list[str] = ["/world/rig_00/cam_00/pinhole/video", "/world/rig_00/cam_01/pinhole/video"]
    with rr.server.Server(datasets={"base": [str(BASE_RRD)]}) as server:
        dataset = server.client().get_dataset("base")
        videos = read_catalog_videos(dataset, dataset.segment_ids()[0], entities, "video_time")
    indices: list[int] = list(range(0, 70, 10))
    decoders = [StreamDecoder(v, 0, 61, "cuda") for v in videos]
    frames = torch.stack([d.frames(indices) for d in decoders])

    def unions(lookahead: bool) -> list:
        tracker = PeopleTracker(TrackOptions(redetect_every=2, lookahead=lookahead), MaskOptions(), "cuda")
        ahead = [frames[:, i + 2] if i + 2 < len(indices) else None for i in range(len(indices))]
        out = [tracker.step(frames[:, i], ahead[i]).cpu() for i in range(len(indices))]
        tracker.close()
        assert tracker.redetects == 4  # steps 0, 2, 4, 6
        return out

    inline = unions(lookahead=False)
    overlapped = unions(lookahead=True)

    assert all(torch.equal(a, b) for a, b in zip(inline, overlapped, strict=True))
    assert any(bool(u.any()) for u in overlapped)
