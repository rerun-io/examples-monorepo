"""CPU triangulation against served masks, positions and overlays."""

from pathlib import Path

import numpy as np
import pytest
from frozen_v2 import ENTITY, GOLDEN_REFS, ROOT, SERVED, calib_files, chunk_bytes, dense, load_keypoints2d
from rerun.chunk import RrdReader

from selfcap_curate.catalog_io import DATASET_NAME, open_segment, read_time_sync
from selfcap_curate.keypoints import read_anchors, read_cameras, triangulate_episode, write_keypoints
from selfcap_curate.records import read_frozen_sync

pytestmark = pytest.mark.golden
SESSIONS = ("2025-12-20__25aeb66a", "2025-12-30__65d00748")


@pytest.mark.parametrize("session", SESSIONS)
def test_triangulated_episode_matches_served_masks_and_overlays(session: str, tmp_path: Path, private_server, private_catalog) -> None:
    sync_paths = sorted((ROOT / "v2/sync").glob(f"{session}__ep*.json"))
    if not sync_paths:
        pytest.skip(f"keypoint golden sync missing: {session}")
    syncs = [read_frozen_sync(path) for path in sync_paths]
    for sync in syncs:
        for asset in (GOLDEN_REFS / "rtmw-kp2d" / sync.episode_key.rrd_name, SERVED / "calib-v2" / f"selfcap__{sync.key}.rrd"):
            if not asset.is_file():
                pytest.skip(f"keypoint golden asset missing: {asset}")
    with private_catalog(syncs, tmp_path / "base", layers=False) as server:
        dataset = server.client().get_dataset(DATASET_NAME)
        calibs = calib_files(server, syncs, tmp_path / "calib")
        for sync, calib in zip(syncs, calibs, strict=True):
            segment = f"selfcap__{sync.key}"
            detections = load_keypoints2d(GOLDEN_REFS / "rtmw-kp2d" / sync.episode_key.rrd_name)
            with private_server({DATASET_NAME: [calib]}) as camera_server:
                cameras = read_cameras(camera_server.client().get_dataset(DATASET_NAME), segment)
            result = triangulate_episode(
                detections, cameras, read_anchors(open_segment(dataset, segment), detections.times, sync=read_time_sync(dataset, segment))
            )
            output = tmp_path / "complete" / calib.name
            before, original = calib.read_bytes(), chunk_bytes(calib)
            write_keypoints(calib, output, result)
            assert calib.read_bytes() == before
            after = chunk_bytes(output)
            assert all(after[key] == value for key, value in original.items())
            assert {str(c.entity_path) for c in RrdReader(output).stream() if str(c.entity_path).startswith("/__properties")} == {
                "/__properties/calib"
            }
            served = SERVED / "calib-v2" / calib.name
            stamps, truth = dense(served, ENTITY, 3)
            times, actual = dense(output, ENTITY, 3)
            np.testing.assert_array_equal(times, stamps)
            np.testing.assert_array_equal(np.isfinite(actual).all(2), np.isfinite(truth).all(2))
            distance = np.linalg.norm(actual - truth, axis=2)
            assert np.nanmedian(distance) <= 0.001
            overlay_max = 0.0
            for camera in cameras.values():
                entity = f"{camera.entity}/pinhole/pred/rtmw/coco133_uv"
                t, xy = dense(output, entity, 2)
                ts, expected = dense(served, entity, 2)
                np.testing.assert_array_equal(t, ts)
                np.testing.assert_array_equal(np.isfinite(xy), np.isfinite(expected))
                overlay_max = max(overlay_max, float(np.nanmax(np.abs(xy - expected))))
            assert overlay_max <= 0.01
            print(
                "KP_COMPARE",
                sync.key,
                int(np.isfinite(truth).all(2).sum()),
                float(np.nanmedian(distance)),
                float(np.nanmax(distance)),
                overlay_max,
                flush=True,
            )
