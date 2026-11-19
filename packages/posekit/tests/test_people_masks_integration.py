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
