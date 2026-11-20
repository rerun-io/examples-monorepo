"""SAM 3 people masks on a real Insta360 lens frame (GPU, SAM 3 weights, an Insta360 base recording)."""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

BASE_RRD: str | None = os.environ.get("POSEKIT_PEOPLE_MASKS_RRD")
"""A recording whose lens 0 frame 0 shows the camera operator close to the camera."""


def test_people_masker_finds_the_operator_in_a_lens_frame() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    if BASE_RRD is None or not Path(BASE_RRD).exists():
        pytest.skip(f"asset missing: Insta360 base .rrd (set POSEKIT_PEOPLE_MASKS_RRD; got {BASE_RRD})")
    import rerun as rr
    from simplecv.catalog_video import read_catalog_videos

    from posekit.apis.people_masks import MaskOptions, PeopleMasker, decode_selected

    entity: str = "/world/rig_00/cam_00/pinhole/video"
    with rr.server.Server(datasets={"base": [str(BASE_RRD)]}) as server:
        dataset = server.client().get_dataset("base")
        video = read_catalog_videos(dataset, dataset.segment_ids()[0], [entity], "video_time")[0]
    decoded = decode_selected(video, [0], "cuda")
    masker = PeopleMasker(MaskOptions(dilate_px=4, tiles=2, batch_size=4), "cuda")

    detections = masker.detect(decoded[1])
    masks = masker(decoded[1])

    assert decoded[0] == [0]
    assert detections.num_detections >= 1
    assert masks.shape == tuple(decoded[1].shape[:3])
    assert 0.05 < float(masks.mean()) < 0.5  # the operator fills a large part of lens 0, not all of it


def test_sam3_tensorrt_engines_match_the_torch_backend() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorrt", reason="the tensorrt backend needs TensorRT")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    if BASE_RRD is None or not Path(BASE_RRD).exists():
        pytest.skip(f"asset missing: Insta360 base .rrd (set POSEKIT_PEOPLE_MASKS_RRD; got {BASE_RRD})")
    import rerun as rr
    from simplecv.catalog_video import read_catalog_videos

    from posekit.apis.people_masks import decode_selected
    from posekit.models.base import SegmentationPrompts
    from posekit.models.sam3_segmenter import Sam3SegmenterConfig

    entities: list[str] = ["/world/rig_00/cam_00/pinhole/video", "/world/rig_00/cam_01/pinhole/video"]
    with rr.server.Server(datasets={"base": [str(BASE_RRD)]}) as server:
        dataset = server.client().get_dataset("base")
        videos = read_catalog_videos(dataset, dataset.segment_ids()[0], entities, "video_time")
    frames = torch.cat([decode_selected(video, [0], "cuda")[1] for video in videos])
    prompts = SegmentationPrompts(frame_indices=torch.zeros(0, dtype=torch.int64), text="person")
    unions: dict[str, list] = {}
    for backend in ("torch", "tensorrt"):
        segmenter = Sam3SegmenterConfig(dtype="float16", backend=backend).setup()
        detections = segmenter(frames, prompts)
        assert detections.masks is not None
        unions[backend] = [detections.masks[detections.frame_indices == lens].any(dim=0) for lens in range(len(entities))]
        del segmenter
        torch.cuda.empty_cache()

    for eager, engine in zip(unions["torch"], unions["tensorrt"], strict=True):
        assert float((eager & engine).sum()) >= 0.98 * float((eager | engine).sum())  # the fp16 engines: union IoU 0.998 on clip B
