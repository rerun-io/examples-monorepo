from __future__ import annotations

from pathlib import Path

from simplecv.data.ego.rrd_ego import ordered_video_sources


def test_ordered_video_sources_uses_blobs_and_path_bytes(tmp_path: Path) -> None:
    """RRD ego video sources preserve camera order and fall back to path bytes."""
    disk_video_path: Path = tmp_path / "disk.mp4"
    disk_video_bytes: bytes = b"disk-video"
    blob_video_bytes: bytes = b"blob-video"
    disk_video_path.write_bytes(disk_video_bytes)

    sources: list[Path | bytes] = ordered_video_sources(
        ordered_names=["blob_cam", "disk_cam"],
        ordered_video_map={"blob_cam": tmp_path / "unused.mp4", "disk_cam": disk_video_path},
        video_blobs={"blob_cam": blob_video_bytes},
    )

    assert sources == [blob_video_bytes, disk_video_bytes]
