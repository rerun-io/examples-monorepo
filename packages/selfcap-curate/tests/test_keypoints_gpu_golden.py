"""GPU producer parity, exact AV1 frame checks and downstream 5a accuracy."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from frozen_v2 import ENTITY, GOLDEN_REFS, ROOT, SERVED, calib_files, dense, load_keypoint_reference
from simplecv.catalog_video_codec import wrap_mp4
from torchcodec.decoders import VideoDecoder

from selfcap_curate.catalog_io import DATASET_NAME, open_segment, read_time_sync
from selfcap_curate.keypoints import read_anchors, read_cameras, triangulate_episode
from selfcap_curate.keypoints_av1 import filter_video
from selfcap_curate.keypoints_gpu import detect_keypoints, keypoint_models, plan_episode
from selfcap_curate.records import read_frozen_sync

pytestmark = pytest.mark.golden
KEY = "2025-12-30__65d00748__ep001"


def test_gpu_keypoints_match_frozen_and_served_triangulation(tmp_path: Path, private_server, private_catalog):
    assets = [
        ROOT / "v2/sync" / f"{KEY}.json",
        GOLDEN_REFS / "rtmw-kp2d" / f"selfcap__{KEY}.rrd",
        SERVED / "base" / f"selfcap__{KEY}.rrd",
        SERVED / "calib-v2" / f"selfcap__{KEY}.rrd",
        ROOT / "calib/sessions/2025-12-30__65d00748.json",
    ]
    for path in assets:
        if not os.access(path, os.R_OK):
            pytest.skip(f"keypoints GPU golden asset missing: {path}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for RTMW TensorRT golden")
    sync = read_frozen_sync(assets[0])
    frozen = load_keypoint_reference(assets[1])
    expected = frozen.detections
    models = keypoint_models()
    with private_catalog([sync], tmp_path / "base", layers=False) as server:
        dataset = server.client().get_dataset(DATASET_NAME)
        segment = f"selfcap__{KEY}"
        calib = calib_files(server, [sync], tmp_path / "calib")[0]
        handle = open_segment(dataset, segment)
        sync = read_time_sync(dataset, segment)
        plan = plan_episode(handle, expected.rate, sync=sync)
        np.testing.assert_array_equal(plan.times, expected.times)
        for phone, video in plan.videos.items():
            original = plan.indices[phone]
            np.testing.assert_allclose(
                video.t_ns[original] * 1e-9, frozen.pts_s[phone] + sync.offsets_ms["exo/" + phone] / 1000.0, rtol=0.0, atol=1e-9
            )
            fps = round(1e9 / float(np.median(np.diff(video.t_ns))))
            np.testing.assert_array_equal(
                original,
                np.rint((frozen.pts_s[phone] + sync.offsets_ms["exo/" + phone] / 1000.0 - video.t_ns[0] * 1e-9) * fps).astype(np.int64),
            )
            source, target = tmp_path / "source.mp4", tmp_path / "filtered.mp4"
            source.write_bytes(wrap_mp4(video.samples, video.keyframes, fps, video.codec))
            indices = filter_video(source, target, original.astype(np.float64) / fps)
            full = VideoDecoder(str(source), device="cuda", dimension_order="NHWC", seek_mode="exact")
            skip = VideoDecoder(str(target), device="cuda", dimension_order="NHWC", seek_mode="exact")
            assert len(skip) < len(full)
            for start in range(0, len(original), 8):
                full_frames = full.get_frames_at(original[start : start + 8].tolist())
                skip_frames = skip.get_frames_at(indices[start : start + 8].tolist())
                assert torch.equal(full_frames.data, skip_frames.data)
                assert torch.equal(full_frames.pts_seconds, skip_frames.pts_seconds)
            print("AV1_GPU_EXACT", phone, len(original), len(full), len(skip), flush=True)
            del full, skip, full_frames, skip_frames
        actual = detect_keypoints(handle, *models, rate=expected.rate, sync=sync)
        np.testing.assert_array_equal(actual.times, expected.times)
        assert actual.views.keys() == expected.views.keys()
        for phone, view in actual.views.items():
            reference = expected.views[phone]
            good = reference.scores >= 0.3
            delta = np.linalg.norm(view.xy[good] - reference.xy[good], axis=-1)
            assert np.isfinite(delta).all()
            print("KP2D_GPU", phone, len(actual.times), float(np.median(delta)), float(np.max(delta)), flush=True)
            assert np.median(delta) <= 1.0
        with private_server({DATASET_NAME: [calib]}) as camera_server:
            cameras = read_cameras(camera_server.client().get_dataset(DATASET_NAME), segment)
        result = triangulate_episode(actual, cameras, read_anchors(handle, actual.times, sync=sync))
    stamps, truth = dense(assets[3], ENTITY, 3)
    np.testing.assert_array_equal(np.rint(result.times * 1e9).astype(np.int64), stamps)
    distance = np.linalg.norm(result.xyz - truth, axis=-1)
    common = np.isfinite(distance)
    assert common.any()
    print(
        "KP2D_3D",
        int(common.sum()),
        int(np.count_nonzero(np.isfinite(result.xyz).all(2) != np.isfinite(truth).all(2))),
        float(np.nanmedian(distance)),
        float(np.nanmax(distance)),
        flush=True,
    )
    assert np.nanmedian(distance) <= 0.001
