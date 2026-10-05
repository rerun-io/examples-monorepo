"""Aria Gen2 Pilot download from a synthetic URL file, with the CDN replaced by an in-memory transport."""

import hashlib
import json
from pathlib import Path

import pytest
from conftest import zip_bytes

from dataforge import transports
from dataforge.datasets.aria_gen2_pilot import MANIFEST_FILE, AriaGen2PilotConfig

FUTURE: str = "oe=7FFFFFFF"
"""An ``oe`` expiry in 2038."""
READ: tuple[str, ...] = ("video.vrs", "mps/slam/closed_loop_trajectory.csv", "mps/hand_tracking/hand_tracking_results.csv")
"""What the converter reads per sequence, relative to its folder."""


def entry(bodies: dict[str, bytes], name: str, data: bytes) -> dict[str, object]:
    """A URL-file entry serving ``data`` from a fake signed URL."""
    url = f"https://cdn.invalid/{name}?{FUTURE}&oh=SECRETSIG"
    bodies[url] = data
    return {"filename": name, "sha1sum": hashlib.sha1(data).hexdigest(), "file_size_bytes": len(data), "download_url": url}


def url_file_for(tmp_path: Path, bodies: dict[str, bytes], dataset_name: str = "AriaGen2PilotDataset") -> Path:
    """Two sequences as projectaria.com lists them: the main VRS, the MPS zips, and files the port never reads."""
    prefix = "AriaGen2PilotDataset_v1.0"
    sequences = {
        name: {
            "main_vrs": entry(bodies, f"{prefix}_{name}_main_recording.vrs", name.encode() * 10),
            "mps_slam_trajectories": entry(
                bodies, f"{prefix}_{name}_mps_slam_trajectories.zip", zip_bytes({"closed_loop_trajectory.csv": b"c", "open_loop_trajectory.csv": b"o"})
            ),
            "mps_hand_tracking": entry(bodies, f"{prefix}_{name}_mps_hand_tracking.zip", zip_bytes({"hand_tracking_results.csv": b"h", "summary.json": b"{}"})),
            "video_main_rgb": entry(bodies, f"{prefix}_{name}_preview_rgb.mp4", b"never fetched"),
            "depth": entry(bodies, f"{prefix}_{name}_depth.zip", b"never fetched"),
        }
        for name in ("clean_0", "cook_0")
    }
    config = {"dataset_name": dataset_name, "main": {"recording": "video.vrs", "mps": "mps"}, "data_groups": {"depth": ["depth.mp4"]}, "release": "v1.0"}
    path = tmp_path / "urls" / f"{dataset_name}_download_urls.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"sequences": sequences, "sequence_config": config}))
    return path


@pytest.fixture
def cdn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, bytes], Path, list[str]]:
    """The served bodies, the URL file, and the names of the files fetched so far."""
    bodies: dict[str, bytes] = {}
    fetched: list[str] = []

    def fake_fetch(url: str, *, dest: Path, label: str | None = None) -> Path:
        assert label is not None and "SECRETSIG" not in label
        fetched.append(url.split("?")[0].rsplit("/", 1)[1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(bodies[url])
        return dest

    monkeypatch.setattr(transports, "http_fetch", fake_fetch)
    return bodies, url_file_for(tmp_path, bodies), fetched


def test_download_fetches_only_what_convert_reads_then_rerun_skips(cdn: tuple[dict[str, bytes], Path, list[str]], tmp_path: Path) -> None:
    _, url_file, fetched = cdn
    root = tmp_path / "raw"
    AriaGen2PilotConfig(root=root, url_file=url_file, sequences=("clean_0",)).setup().download()
    assert sorted(fetched) == [
        "AriaGen2PilotDataset_v1.0_clean_0_main_recording.vrs",
        "AriaGen2PilotDataset_v1.0_clean_0_mps_hand_tracking.zip",
        "AriaGen2PilotDataset_v1.0_clean_0_mps_slam_trajectories.zip",
    ]
    folder = root / "clean_0"
    assert sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()) == sorted(READ)
    assert (folder / "video.vrs").read_bytes() == b"clean_0" * 10
    manifest = (root / MANIFEST_FILE).read_text()
    assert "SECRETSIG" not in manifest and "download_url" not in manifest

    # Discovery and the verify-only download need no URL file.
    dataset = AriaGen2PilotConfig(root=root).setup()
    assert [identity.recording_id for identity, _ in dataset.discover()] == ["aria_gen2_pilot__clean_0"]
    dataset.download()

    fetched.clear()
    AriaGen2PilotConfig(root=root, url_file=url_file, sequences=("clean_0",)).setup().download()
    assert fetched == []


def test_listing_comes_from_the_url_file_or_the_manifest(cdn: tuple[dict[str, bytes], Path, list[str]], tmp_path: Path) -> None:
    bodies, url_file, fetched = cdn
    root = tmp_path / "raw"
    listing = AriaGen2PilotConfig(root=root, url_file=url_file).setup().remote_sequences()
    assert [row.key for row in listing] == ["clean_0", "cook_0"]
    assert listing[0].files == tuple(f"clean_0/{name}" for name in READ)
    served = json.loads(url_file.read_text())["sequences"]["clean_0"]
    assert listing[0].size_bytes == sum(served[kind]["file_size_bytes"] for kind in ("main_vrs", "mps_slam_trajectories", "mps_hand_tracking"))
    assert not fetched
    AriaGen2PilotConfig(root=root, url_file=url_file, sequences=("cook_0",)).setup().download()
    assert AriaGen2PilotConfig(root=root).setup().remote_sequences() == listing


def test_wrong_url_files_and_names_fail_before_any_fetch(cdn: tuple[dict[str, bytes], Path, list[str]], tmp_path: Path) -> None:
    bodies, url_file, fetched = cdn
    root = tmp_path / "raw"
    with pytest.raises(ValueError, match="AriaGen2PilotDataset_download_urls"):
        AriaGen2PilotConfig(root=root, url_file=url_file_for(tmp_path, bodies, "Hot3DAria")).setup().download()
    with pytest.raises(ValueError, match="unknown sequence"):
        AriaGen2PilotConfig(root=root, url_file=url_file, sequences=("walk_0",)).setup().download()
    with pytest.raises(FileNotFoundError, match="--url-file"):
        AriaGen2PilotConfig(root=root).setup().discover()
    assert not fetched
