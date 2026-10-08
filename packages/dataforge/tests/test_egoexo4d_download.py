"""Ego-Exo4D release access against an in-memory S3: manifests, per-take file selection, size-checked fetch."""

import json
import threading
from pathlib import Path
from typing import Any, cast

import pytest

from dataforge.datasets.egoexo4d import Egoexo4dConfig, Egoexo4dDataset
from dataforge.datasets.egoexo4d_download import RELEASE, Release
from dataforge.datasets.egoexo4d_source import read_takes

TAKE: dict[str, object] = {
    "take_name": "cmu_bike02_4",
    "take_uid": "take-uid",
    "root_dir": "takes/cmu_bike02_4",
    "capture_uid": "capture-uid",
    "timesync_start_idx": 0,
    "timesync_end_idx": 10,
    "capture": {"capture_name": "cmu_bike02", "cameras": [{"cam_id": "cam01", "is_ego": False}, {"cam_id": "aria01", "is_ego": True}]},
    "frame_aligned_videos": {
        "cam01": {"0": {"relative_path": "frame_aligned_videos/cam01.mp4", "readable_stream_id": "0"}},
        "aria01": {
            stream: {"relative_path": f"frame_aligned_videos/aria01_{stream_id}.mp4", "readable_stream_id": stream}
            for stream, stream_id in (("rgb", "214-1"), ("slam-left", "1201-1"), ("slam-right", "1201-2"), ("et", "211-1"))
        },
    },
}


class FakeS3:
    """The three s3fs calls Release makes, over a dict of object bytes; ``denied`` raises like an unauthorized profile."""

    def __init__(self, objects: dict[str, bytes], *, denied: bool = False) -> None:
        self.objects: dict[str, bytes] = objects
        self.denied: bool = denied

    def read_text(self, path: str) -> str:
        if self.denied:
            raise PermissionError("Access Denied")
        return self.objects[path].decode()

    def size(self, path: str) -> int:
        return len(self.objects[path])

    def get_file(self, path: str, dest: str) -> None:
        Path(dest).write_bytes(self.objects[path])


def entry(uid: str, paths: list[tuple[str, int]]) -> dict[str, object]:
    return {
        "uid": uid,
        "paths": [{"source_path": f"s3://bucket/{path}", "relative_path": path, "size": size, "checksum": None} for path, size in paths],
    }


def release(objects: dict[str, bytes], manifests: dict[str, list[dict[str, object]]], *, denied: bool = False) -> Release:
    for part, entries in manifests.items():
        objects[f"{RELEASE}/{part}/manifest.json"] = json.dumps(entries).encode()
    result: Release = Release.__new__(Release)
    result.profile = "egoexo4d"
    result.fs = cast(Any, FakeS3(objects, denied=denied))
    result.manifests = {}
    result.lock = threading.Lock()
    return result


def test_take_files_select_what_base_reads(tmp_path: Path) -> None:
    videos = ["cam01.mp4", "aria01_214-1.mp4", "aria01_1201-1.mp4", "aria01_1201-2.mp4", "aria01_211-1.mp4", "collage.mp4"]
    manifests = {
        "takes": [entry("take-uid", [(f"takes/cmu_bike02_4/frame_aligned_videos/{name}", 10) for name in videos])],
        "take_trajectory": [
            entry(
                "take-uid",
                [
                    (f"takes/cmu_bike02_4/trajectory/{name}", 5)
                    for name in ("closed_loop_trajectory.csv", "gopro_calibs.csv", "online_calibration.jsonl")
                ],
            )
        ],
        "take_vrs_noimagestream": [entry("take-uid", [("takes/cmu_bike02_4/aria01_noimagestreams.vrs", 7)])],
        "captures": [entry("capture-uid", [("captures/cmu_bike02/timesync.csv", 3), ("captures/cmu_bike02/post_surveys.csv", 3)])],
    }
    (tmp_path / "takes.json").write_text(json.dumps([TAKE]))
    take = read_takes(tmp_path / "takes.json")["cmu_bike02_4"]
    dataset: Egoexo4dDataset = Egoexo4dDataset(Egoexo4dConfig(root=tmp_path))
    dataset.release = release({}, manifests)
    take_files = [
        *(f"takes/cmu_bike02_4/frame_aligned_videos/{name}" for name in videos[:5]),
        "takes/cmu_bike02_4/trajectory/closed_loop_trajectory.csv",
        "takes/cmu_bike02_4/trajectory/gopro_calibs.csv",
        "takes/cmu_bike02_4/aria01_noimagestreams.vrs",
    ]
    assert [path.relative_path for path in dataset.take_files(take)] == take_files
    fetched: list[str] = []
    dataset.release.fetch = lambda path, root: fetched.append(path.relative_path) or True  # type: ignore[method-assign]
    dataset.fetch_take(take)
    assert sorted(fetched) == sorted([*take_files, "captures/cmu_bike02/timesync.csv"])  # the capture's other files stay remote


def test_fetch_lands_whole_files_once(tmp_path: Path) -> None:
    objects = {"s3://bucket/takes/t/a.mp4": b"0123456789"}
    source = release(objects, {"takes": [entry("u", [("takes/t/a.mp4", 10)])]})
    (path,) = source.manifest("takes")["u"].paths
    assert source.fetch(path, tmp_path)
    assert (tmp_path / "takes/t/a.mp4").read_bytes() == b"0123456789"
    assert not source.fetch(path, tmp_path)  # already complete
    objects["s3://bucket/takes/t/a.mp4"] = b"short"
    (tmp_path / "takes/t/a.mp4").unlink()
    with pytest.raises(ValueError, match="10"):
        source.fetch(path, tmp_path)  # a truncated object never lands
    assert not (tmp_path / "takes/t/a.mp4").exists()
    assert not any((tmp_path / ".dataforge-staging").rglob("*.mp4*"))


def test_denied_access_names_the_profile_and_the_sign_up() -> None:
    source = release({}, {"takes": []}, denied=True)
    with pytest.raises(PermissionError, match="'egoexo4d'.*ego4ddataset"):
        source.manifest("takes")
