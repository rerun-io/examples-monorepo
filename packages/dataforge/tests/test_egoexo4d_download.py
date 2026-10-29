"""Ego-Exo4D release access against an in-memory S3: manifests, per-take file selection, size-checked fetch."""

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, cast

import pytest
from fsspec.asyn import get_loop
from test_egoexo4d_source import GOPRO_HEADER, GOPRO_ROWS

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
    "capture": {
        "capture_name": "cmu_bike02",
        "cameras": [{"cam_id": "cam01", "is_ego": False}, {"cam_id": "cam02", "is_ego": False}, {"cam_id": "aria01", "is_ego": True}],
    },
    "frame_aligned_videos": {
        "cam01": {"0": {"relative_path": "frame_aligned_videos/cam01.mp4", "readable_stream_id": "0"}},
        "cam02": {"0": {"relative_path": "frame_aligned_videos/cam02.mp4", "readable_stream_id": "0"}},
        "aria01": {
            stream: {"relative_path": f"frame_aligned_videos/aria01_{stream_id}.mp4", "readable_stream_id": stream}
            for stream, stream_id in (("rgb", "214-1"), ("slam-left", "1201-1"), ("slam-right", "1201-2"), ("et", "211-1"))
        },
    },
}


class FakeBody:
    """An async S3 body whose reads can be replaced with a stalled stream."""

    def __init__(self, payload: bytes) -> None:
        self.payload: bytes = payload
        self.offset: int = 0
        self.closed: bool = False

    async def read(self, amount: int) -> bytes:
        chunk: bytes = self.payload[self.offset : self.offset + amount]
        self.offset += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


class FakeS3:
    """The s3fs session, size and GetObject calls, over in-memory bytes; ``denied`` raises like an unauthorized profile."""

    def __init__(self, objects: dict[str, bytes], *, denied: bool = False) -> None:
        self.objects: dict[str, bytes] = objects
        self.denied: bool = denied
        self.loop = get_loop()
        self.s3 = self

    async def set_session(self) -> None:
        pass

    async def get_object(self, *, Bucket: str, Key: str, Range: str) -> dict[str, FakeBody]:
        start, end = (int(value) for value in Range.removeprefix("bytes=").split("-"))
        return {"Body": FakeBody(self.objects[f"s3://{Bucket}/{Key}"][start : end + 1])}

    async def _size(self, path: str) -> int:
        return self.size(path)

    def size(self, path: str) -> int:
        if self.denied:
            raise PermissionError("Access Denied")
        return len(self.objects[path])



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
    result.stream_slots = threading.BoundedSemaphore(16)
    return result


def test_fetch_take_fetches_what_base_reads(tmp_path: Path) -> None:
    """cam02 has a video but no localization (gopro_calibs.csv quality 0): base never logs it, so it is never downloaded."""
    videos = ["cam01.mp4", "aria01_214-1.mp4", "aria01_1201-1.mp4", "aria01_1201-2.mp4", "aria01_211-1.mp4", "cam02.mp4", "collage.mp4"]
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
    calibs: Path = tmp_path / "takes/cmu_bike02_4/trajectory/gopro_calibs.csv"
    calibs.parent.mkdir(parents=True)
    calibs.write_text("\n".join([GOPRO_HEADER, *GOPRO_ROWS[:2]]) + "\n")  # cam01 localized, cam02 not
    take = read_takes(tmp_path / "takes.json")["cmu_bike02_4"]
    dataset: Egoexo4dDataset = Egoexo4dDataset(Egoexo4dConfig(root=tmp_path))
    dataset.release = release({}, manifests)
    take_files = [
        *(f"takes/cmu_bike02_4/frame_aligned_videos/{name}" for name in videos[:5]),
        "takes/cmu_bike02_4/trajectory/closed_loop_trajectory.csv",
        "takes/cmu_bike02_4/trajectory/gopro_calibs.csv",
        "takes/cmu_bike02_4/aria01_noimagestreams.vrs",
    ]
    assert [path.relative_path for path in dataset.take_files(take, ["cam01"])] == take_files
    listed = [path.relative_path for path in dataset.take_files(take, take.exo_cameras)]  # --list-remote: every exo video
    assert listed == [*take_files[:5], "takes/cmu_bike02_4/frame_aligned_videos/cam02.mp4", *take_files[5:]]
    for part in manifests:
        dataset.release.manifest(cast(Any, part))
    fetched: list[str] = []
    dataset.release.fetch = lambda path, root: fetched.append(path.relative_path) or True  # type: ignore[method-assign]
    dataset.fetch_take(take)
    assert fetched[:2] == take_files[5:7]  # the trajectory first: gopro_calibs.csv names the GoPros to fetch
    assert sorted(fetched) == sorted([*take_files, "captures/cmu_bike02/timesync.csv"])  # once each; the capture's other files stay remote


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


def test_large_fetch_reads_ranges_concurrently_and_verifies_each(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A large object uses overlapping bounded reads and never publishes a short range."""
    from dataforge.datasets import egoexo4d_download as download

    monkeypatch.setattr(download, "RANGE_BYTES", 4, raising=False)
    payload = b"0123456789abcdef"
    source = release({"s3://bucket/a": payload}, {"takes": [entry("u", [("a", len(payload))])]})
    source.stream_slots = threading.BoundedSemaphore(16)
    barrier = asyncio.Barrier(4)
    reads: list[tuple[int, int]] = []
    (file,) = source.manifest("takes")["u"].paths

    async def read(**kwargs: str) -> dict[str, FakeBody]:
        start, end = (int(value) for value in kwargs["Range"].removeprefix("bytes=").split("-"))
        reads.append((start, end + 1))
        await asyncio.wait_for(barrier.wait(), timeout=2.0)
        return {"Body": FakeBody(payload[start : end + 1])}

    monkeypatch.setattr(source.fs.s3, "get_object", read)
    assert source.fetch(file, tmp_path)
    assert (tmp_path / "a").read_bytes() == payload
    assert sorted(reads) == [(0, 4), (4, 8), (8, 12), (12, 16)]
    assert not source.fetch(file, tmp_path)
    (tmp_path / "a").unlink()
    async def short(**kwargs: str) -> dict[str, FakeBody]:
        return {"Body": FakeBody(b"x")}

    monkeypatch.setattr(source.fs.s3, "get_object", short)
    with pytest.raises(ValueError, match="range"):
        source.fetch(file, tmp_path)
    assert not (tmp_path / "a").exists()


def test_prefetch_leaves_oversized_take_for_foreground(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Speculation has a raw-byte ceiling; an oversized take can still convert in the foreground."""
    from dataforge.datasets.egoexo4d_download import ManifestPath
    from dataforge.identity import SequenceIdentity

    (tmp_path / "takes.json").write_text(json.dumps([TAKE]))
    take = read_takes(tmp_path / "takes.json")["cmu_bike02_4"]
    dataset = Egoexo4dDataset(Egoexo4dConfig(root=tmp_path))
    dataset.config.raw_budget_gb = 0.00000001
    dataset.release = release({}, {"captures": [entry("capture-uid", [])]})
    files = [ManifestPath("s3://bucket/video", "video", 20)]
    monkeypatch.setattr(dataset, "take_files", lambda take, gopros: files)
    fetched: list[str] = []
    monkeypatch.setattr(dataset, "fetch_take", lambda take: fetched.append(take.take_name))
    identity = SequenceIdentity("egoexo4d", (take.take_name,))
    dataset.prefetch(identity, take, force=True)
    assert fetched == []
    files[0] = ManifestPath("s3://bucket/video", "video", 5)
    dataset.prefetch(identity, take, force=True)
    assert fetched == [take.take_name]


def test_stalled_stream_is_cancelled_closed_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """After one chunk, a body that never sends again cannot hold a fetch indefinitely."""
    from dataforge.datasets import egoexo4d_download as download

    monkeypatch.setattr(download, "NO_PROGRESS_S", 0.03, raising=False)
    monkeypatch.setattr(download, "RETRY_DELAY_S", 0.0, raising=False)
    source = release({"s3://bucket/a": b"0123456789"}, {"takes": [entry("u", [("a", 10)])]})
    (file,) = source.manifest("takes")["u"].paths
    stalled = FakeBody(b"012")
    cancelled = threading.Event()
    original_read = stalled.read

    async def read(amount: int) -> bytes:
        if stalled.offset:
            try:
                try:
                    await asyncio.wait_for(asyncio.Event().wait(), timeout=1.0)
                except TimeoutError as error:
                    raise RuntimeError("no-progress watchdog did not cancel the stopped read") from error
            finally:
                cancelled.set()
        return await original_read(amount)

    monkeypatch.setattr(stalled, "read", read)
    calls: list[str] = []

    async def get_object(**kwargs: str) -> dict[str, FakeBody]:
        calls.append(kwargs["Range"])
        return {"Body": stalled if len(calls) == 1 else FakeBody(b"0123456789")}

    monkeypatch.setattr(source.fs.s3, "get_object", get_object)
    assert source.fetch(file, tmp_path)
    assert cancelled.is_set() and stalled.closed
    assert calls == ["bytes=0-9", "bytes=0-9"]
    assert (tmp_path / "a").read_bytes() == b"0123456789"
    assert "attempt 1/3" in capsys.readouterr().out


@pytest.mark.parametrize("stall_at", ["headers", "body", "trickle"])
def test_stall_attempts_are_bounded_and_never_publish(
    stall_at: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dead GET, a silent body, and a continuously trickling body all terminate."""
    from dataforge.datasets import egoexo4d_download as download

    source = release({"s3://bucket/a": b"0123456789"}, {"takes": [entry("u", [("a", 10)])]})
    (file,) = source.manifest("takes")["u"].paths
    monkeypatch.setattr(download, "NO_PROGRESS_S", 0.03)
    monkeypatch.setattr(download, "RANGE_DEADLINE_S", 0.06)
    monkeypatch.setattr(download, "RETRY_DELAY_S", 0.0)
    bodies: list[FakeBody] = []
    calls: list[str] = []

    async def read(amount: int) -> bytes:
        if stall_at == "trickle":
            await asyncio.sleep(0.02)
            return b"x"
        await asyncio.Event().wait()
        return b""

    async def get_object(**kwargs: str) -> dict[str, FakeBody]:
        calls.append(kwargs["Range"])
        if stall_at == "headers":
            await asyncio.Event().wait()
        body = FakeBody(b"")
        monkeypatch.setattr(body, "read", read)
        bodies.append(body)
        return {"Body": body}

    monkeypatch.setattr(source.fs.s3, "get_object", get_object)
    with pytest.raises(TimeoutError):
        source.fetch(file, tmp_path)
    assert calls == ["bytes=0-9"] * 3
    assert all(body.closed for body in bodies)
    assert not (tmp_path / "a").exists()
    assert not any(path.is_file() for path in (tmp_path / ".dataforge-staging").rglob("*"))
    assert "attempt 3/3" in capsys.readouterr().out
