"""EgoHumans: Drive fetch (quota page, resume), upstream configs, the untrusted tar extraction, discovery, and (integration) a
synthetic take converted from its tar.gz."""

from __future__ import annotations

import gzip
import io
import tarfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from conftest import column_rows, raw_asset, read_back, serve, static_values
from egohumans_synthetic import COLMAP_FROM_WORLD, EXO_POSES, FRAMES, write_sequence

from dataforge import paths, schema
from dataforge.datasets import egohumans_download
from dataforge.datasets.egohumans import EgohumansConfig, EgohumansDataset, extract_sequence
from dataforge.datasets.egohumans_download import DriveFile, DriveQuotaError, config_path, fetch_drive_file, read_manifest
from dataforge.egohumans_layout import CaptureConfig, DirectorySequenceFiles, read_sequence
from dataforge.records import decode

UPSTREAM_CONFIG: str = """SEQUENCE: '001_basketball'
INVALID_ARIAS: []
INVALID_EXOS: ['cam01']

GEOMETRY:
  MANUAL_GROUND_PLANE_POINTS: "array([[-14.00048087,   5.81472378,  -1.10658211],
       [ -7.74524121,  10.04410562,  -1.22805929]])
  "

CALIBRATION:
  MANUAL_EXO_CAMERAS:
  - cam02

BBOX:
  VIS_CAMERAS: ['cam02', 'aria01']
"""
"""Shaped like the release's configs: block and flow lists, a multi-line string, many keys this reader ignores."""


def tar_gz(members: dict[str, bytes], *, symlink: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if symlink is not None:
            link = tarfile.TarInfo(symlink)
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            archive.addfile(link)
    return buffer.getvalue()


def release_tar(directory: Path, name: str) -> bytes:
    """A sequence directory packed the way the release packs it: under the authors' absolute path."""
    prefix = f"media/rawalk/disk1/rawalk/datasets/ego_exo/main/01_tagging/{name}"
    return tar_gz({f"{prefix}/{path.relative_to(directory)}": path.read_bytes() for path in sorted(directory.rglob("*")) if path.is_file()})


def test_manifest_lists_every_sequence_archive() -> None:
    manifest = read_manifest()
    assert len(manifest.sequences) == 133 and len(manifest.files) == 142
    assert sum(file.size_bytes for file in manifest.sequences) == pytest.approx(597.6e9, rel=1e-3)
    assert manifest.sequences[0].key == "01_tagging/001_tagging"
    assert config_path("02_lego/006_legoassemble") == "configs/legoassemble/006_legoassemble.yaml"


def test_upstream_config_reads_the_lists_and_ignores_the_rest() -> None:
    config = decode(CaptureConfig, UPSTREAM_CONFIG, source="001_basketball.yaml", fmt="yaml")
    assert config.INVALID_EXOS == ["cam01"] and config.INVALID_ARIAS == []
    assert config.CALIBRATION.MANUAL_EXO_CAMERAS == ["cam02"]
    assert decode(CaptureConfig, "SEQUENCE: '002'\n", source="minimal.yaml", fmt="yaml") == CaptureConfig()


def test_invalid_cameras_are_dropped_and_hand_posed_ones_use_their_npy(tmp_path: Path) -> None:
    directory = write_sequence(tmp_path / "01_tagging" / "001_tagging")
    moved = EXO_POSES["cam02"].copy()
    moved[:3, 3] += (0.5, 0.0, 0.0)
    cam_T_colmap = np.linalg.inv(COLMAP_FROM_WORLD @ moved)
    cam_T_colmap /= np.cbrt(np.linalg.det(cam_T_colmap[:3, :3]))  # COLMAP poses are rigid in COLMAP units
    cam_T_colmap[3] = (0.0, 0.0, 0.0, 1.0)
    np.save(directory / "colmap/workplace/cam02.npy", cam_T_colmap[:3])
    config = decode(CaptureConfig, UPSTREAM_CONFIG, source="001_basketball.yaml", fmt="yaml")
    sequence = read_sequence(DirectorySequenceFiles(directory), config=config)
    assert [camera.name for camera in sequence.exo] == ["cam02"]
    np.testing.assert_allclose(sequence.exo[0].world_T_cam, moved, atol=1e-9)
    no_glasses = read_sequence(DirectorySequenceFiles(directory), config=CaptureConfig(INVALID_ARIAS=["aria01"]))
    assert no_glasses.glasses == [] and [person.name for person in no_glasses.people] == ["aria01", "aria02"]


def test_extraction_strips_the_authors_path_and_refuses_escapes(tmp_path: Path) -> None:
    archive = tmp_path / "004_tagging.tar.gz"
    archive.write_bytes(
        tar_gz(
            {
                "media/x/main/01_tagging/004_tagging/exo/cam01/images/00001.jpg": b"jpeg",
                "media/x/main/01_tagging/004_tagging/../../escape.txt": b"no",
                "media/x/main/01_tagging/other/readme.txt": b"elsewhere",
            },
            symlink="media/x/main/01_tagging/004_tagging/link",
        )
    )
    root = extract_sequence(archive, "004_tagging", tmp_path / "work")
    assert sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()) == ["exo/cam01/images/00001.jpg"]
    assert not (tmp_path / "escape.txt").exists() and not (tmp_path / "work" / "escape.txt").exists()


def test_drive_quota_page_is_refused_and_a_short_transfer_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = gzip.compress(np.random.default_rng(0).integers(0, 255, 1 << 16, dtype=np.uint8).tobytes())
    bodies = {"/download?id=open": payload, "/download?id=busy": b"<!DOCTYPE html><title>Google Drive - Quota exceeded</title>"}
    types = {"/download?id=open": "application/x-gzip", "/download?id=busy": "text/html; charset=utf-8"}
    with serve(bodies, content_types=types, stall_once="id=open") as server:
        monkeypatch.setattr(egohumans_download, "DRIVE_URL", server.base_url + "/download?id={file_id}")
        busy = DriveFile("01_tagging/001_tagging.tar.gz", "busy", 999)
        with pytest.raises(DriveQuotaError, match="quota"):
            fetch_drive_file(busy, tmp_path / busy.path)
        assert not (tmp_path / busy.path).exists()
        open_file = DriveFile("01_tagging/002_tagging.tar.gz", "open", len(payload))
        with pytest.raises(ValueError, match="rerun to resume"):
            fetch_drive_file(open_file, tmp_path / open_file.path)  # the server hangs up half way
        assert fetch_drive_file(open_file, tmp_path / open_file.path)
        assert server.served[-1].range_header == f"bytes={len(payload) // 2}-"
        assert not fetch_drive_file(open_file, tmp_path / open_file.path)  # complete: nothing fetched
    assert (tmp_path / open_file.path).read_bytes() == payload


def test_drive_parts_that_are_complete_long_or_not_gzip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = gzip.compress(b"archive" * 1000)
    with serve({"/download?id=gz": payload, "/download?id=text": b"plain text, no gzip"}) as server:
        monkeypatch.setattr(egohumans_download, "DRIVE_URL", server.base_url + "/download?id={file_id}")
        done = DriveFile("01_tagging/001_tagging.tar.gz", "gz", len(payload))
        part = tmp_path / (done.path + ".part")
        part.parent.mkdir(parents=True)
        part.write_bytes(payload)  # a run killed between its last write and the rename
        assert fetch_drive_file(done, tmp_path / done.path)
        assert server.served == [] and (tmp_path / done.path).read_bytes() == payload
        long_part = tmp_path / "01_tagging/002_tagging.tar.gz.part"
        long_part.write_bytes(payload + b"junk")
        assert fetch_drive_file(DriveFile("01_tagging/002_tagging.tar.gz", "gz", len(payload)), tmp_path / "01_tagging/002_tagging.tar.gz")
        assert server.served[-1].range_header is None  # the oversized part was discarded, not resumed past its end
        with pytest.raises(ValueError, match="not gzip"):
            fetch_drive_file(DriveFile("01_tagging/003_tagging.tar.gz", "text", 19), tmp_path / "01_tagging/003_tagging.tar.gz")


def test_drive_resume_with_another_range_restarts_the_part(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = gzip.compress(b"archive" * 1000)

    class WrongRange:
        """A 206 that answers ``bytes=10-`` with the file from byte 0."""

        status_code = 206
        headers = {"Content-Type": "application/x-gzip", "Content-Range": f"bytes 0-{len(payload) - 1}/{len(payload)}"}

        def __enter__(self) -> WrongRange:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int) -> Iterator[bytes]:
            del chunk_size
            return iter([payload])

    monkeypatch.setattr(egohumans_download.requests, "get", lambda *_, **__: WrongRange())
    file = DriveFile("01_tagging/001_tagging.tar.gz", "gz", len(payload))
    part = tmp_path / (file.path + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(payload[:10])
    assert fetch_drive_file(file, tmp_path / file.path)
    assert (tmp_path / file.path).read_bytes() == payload  # rewritten, not the head appended twice


def stage_release(root: Path, tmp_path: Path) -> None:
    directory = write_sequence(tmp_path / "staging" / "001_tagging", pose_dir="fit_poses3d", box_dir="bboxes")
    (root / "01_tagging").mkdir(parents=True)
    (root / "01_tagging" / "001_tagging.tar.gz").write_bytes(release_tar(directory, "001_tagging"))
    (root / "01_tagging" / "002_tagging.tar.gz").write_bytes(release_tar(directory, "002_tagging"))
    (root / "configs" / "tagging").mkdir(parents=True)
    (root / "configs" / "tagging" / "001_tagging.yaml").write_text("SEQUENCE: '001_tagging'\nINVALID_ARIAS: []\n")


def test_discover_needs_the_upstream_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "raw"
    stage_release(root, tmp_path)
    found = EgohumansConfig(root=root).setup().discover()
    assert [identity.recording_id for identity, _ in found] == ["egohumans__01_tagging__001_tagging"]
    assert "skip 01_tagging/002_tagging: no upstream config" in capsys.readouterr().out
    assert EgohumansConfig(root=root, sequences=("02_lego",)).setup().discover() == []


@pytest.mark.integration
def test_convert_extracts_and_writes_four_layers(tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    smpl_root = raw_asset("official neutral SMPL (dataforge-download harmony4d)", paths.raw_root() / "harmony4d/body_models/smpl/SMPL_NEUTRAL.pkl").parents[1]
    root = tmp_path / "raw"
    stage_release(root, tmp_path)
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    dataset = EgohumansConfig(root=root, smpl_model_root=smpl_root).setup()
    assert isinstance(dataset, EgohumansDataset)
    ((identity, source),) = dataset.discover()
    dataset.convert(identity, source, force=False)
    targets = dataset.targets(identity)
    assert all(target.is_file() for target in targets.values())
    assert not (paths.work_root() / "egohumans" / "001_tagging").exists()  # the extraction is gone
    base = read_back(targets["base"])
    samples = column_rows(base, f"{schema.video_path(91, 0)}:VideoStream:sample")
    assert samples[schema.TIMELINE].cast("int64").to_pylist() == [(frame - FRAMES[0]) * 50_000_000 for frame in FRAMES]
    pose = read_back(targets["body_pose"])
    assert static_values(pose, schema.person_path(0))["keypoints_source"] == "processed_data/fit_poses3d"
