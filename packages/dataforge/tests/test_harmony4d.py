"""Harmony4D: zip discovery and key selection, the Hub download, and (integration) a synthetic take converted into four layers."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from conftest import column_rows, raw_asset, read_back, read_chunks, recording_properties, static_values
from egohumans_synthetic import FRAMES, write_sequence
from hub_fake import HubStore

from dataforge import paths, schema, transports
from dataforge.datasets import harmony4d
from dataforge.datasets.harmony4d import SOURCE_REPO, SOURCE_REVISION, Harmony4dConfig, Harmony4dDataset
from dataforge.smpl import SMPL_FILE, SMPL_REPO, SMPL_REVISION

SMPL_ROOT: Path = paths.raw_root() / "harmony4d" / "body_models"


def write_release(root: Path, tmp_path: Path) -> None:
    """``test/01_hugging.zip`` with two sequences (one with glasses, one exo-only) and ``train/02_grappling.zip`` with one."""
    for split, scene, sequences in (("test", "01_hugging", ("002_hugging", "003_hugging")), ("train", "02_grappling", ("001_grappling",))):
        (root / split).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(root / split / f"{scene}.zip", "w") as archive:
            for name in sequences:
                directory = write_sequence(tmp_path / "staging" / scene / name, glasses=name != "003_hugging")
                for path in sorted(directory.rglob("*")):
                    if path.is_file():
                        archive.write(path, f"{scene}/{name}/{path.relative_to(directory)}")


def test_discover_finds_every_sequence_in_the_zips_and_selects_by_archive_or_sequence(tmp_path: Path) -> None:
    root = tmp_path / "raw"
    write_release(root, tmp_path)
    found = Harmony4dConfig(root=root).setup().discover()
    assert [identity.recording_id for identity, _ in found] == [
        "harmony4d__test__01_hugging__002_hugging",
        "harmony4d__test__01_hugging__003_hugging",
        "harmony4d__train__02_grappling__001_grappling",
    ]
    assert found[0][1].prefix == "01_hugging/002_hugging"
    by_archive = Harmony4dConfig(root=root, sequences=("test/01_hugging",)).setup().discover()
    assert [identity.sequence_key for identity, _ in by_archive] == ["test/01_hugging/002_hugging", "test/01_hugging/003_hugging"]
    one = Harmony4dConfig(root=root, sequences=("train/02_grappling/001_grappling",)).setup().discover()
    assert [identity.sequence_key for identity, _ in one] == ["train/02_grappling/001_grappling"]
    with pytest.raises(ValueError, match="download the archive first"):
        Harmony4dConfig(root=root, sequences=("train/09_karate",)).setup().discover()


def harmony_hub(monkeypatch: pytest.MonkeyPatch) -> HubStore:
    store = HubStore({SOURCE_REPO: SOURCE_REVISION, SMPL_REPO: SMPL_REVISION})
    for path in ("test/01_hugging.zip", "train/02_grappling.zip", "README.md"):
        store.add(SOURCE_REPO, path, f"bytes of {path}".encode(), lfs=path.endswith(".zip"))
    store.add(SMPL_REPO, SMPL_FILE, b"smpl", lfs=True)
    store.install(monkeypatch, harmony4d, transports)
    return store


def test_remote_sequences_are_the_release_zips(monkeypatch: pytest.MonkeyPatch) -> None:
    harmony_hub(monkeypatch)
    listed = Harmony4dConfig(root=Path("/nonexistent")).setup().remote_sequences()
    assert [(sequence.key, sequence.files) for sequence in listed] == [
        ("test/01_hugging", ("test/01_hugging.zip",)),
        ("train/02_grappling", ("train/02_grappling.zip",)),
    ]
    assert listed[0].size_bytes == len(b"bytes of test/01_hugging.zip")


def test_download_fetches_the_selected_zip_and_the_smpl_model_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hub = harmony_hub(monkeypatch)
    dataset = Harmony4dConfig(root=tmp_path / "raw", sequences=("test/01_hugging",)).setup()
    dataset.download()
    assert (tmp_path / "raw/test/01_hugging.zip").read_bytes() == b"bytes of test/01_hugging.zip"
    assert not (tmp_path / "raw/train").exists()
    assert (tmp_path / "raw/body_models/smpl/SMPL_NEUTRAL.pkl").read_bytes() == b"smpl"
    fetched = [call for call in hub.calls if call.kind == "fetch"]
    dataset.download()
    assert [call for call in hub.calls if call.kind == "fetch"] == fetched  # complete files are not fetched again
    with pytest.raises(ValueError, match="no archive"):
        Harmony4dConfig(root=tmp_path / "raw", sequences=("test/99_nothing",)).setup().download()


@pytest.mark.integration
def test_convert_writes_four_layers_on_the_release_clock(tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_asset("official neutral SMPL (dataforge-download harmony4d)", SMPL_ROOT / "smpl/SMPL_NEUTRAL.pkl")
    root = tmp_path / "raw"
    write_release(root, tmp_path)
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    dataset = Harmony4dConfig(root=root, sequences=("test/01_hugging",), smpl_model_root=SMPL_ROOT).setup()
    assert isinstance(dataset, Harmony4dDataset)
    expected_times = [(frame - FRAMES[0]) * 50_000_000 for frame in FRAMES]
    for identity, source in dataset.discover():
        dataset.convert(identity, source, force=False)
        targets = dataset.targets(identity)
        assert all(target.is_file() for target in targets.values())
        glasses = identity.parts[-1] == "002_hugging"

        base = read_back(targets["base"])
        for rig in (1, 2):
            samples = column_rows(base, f"{schema.video_path(rig, 0)}:VideoStream:sample")
            assert samples[schema.TIMELINE].cast("int64").to_pylist() == expected_times
            by_frame = base.reader(index=schema.FRAME_INDEX).to_arrow_table().select([schema.FRAME_INDEX, f"{schema.video_path(rig, 0)}:VideoStream:sample"])
            assert sorted(by_frame.drop_null()[schema.FRAME_INDEX].to_pylist()) == list(FRAMES)  # the release's own frame numbers
            values = static_values(base, schema.cam_path(rig, 0))
            assert (values["source_width"], values["stored_width"], values["stored_height"]) == (3840, 1920, 1080)
            assert values["camera_model"] == "OPENCV_FISHEYE"
        aria_streams = [f"{schema.video_path(91, cam)}:VideoStream:sample" for cam in range(3)]
        for column in aria_streams:
            if glasses:
                assert len(column_rows(base, column)) == len(FRAMES)
        if glasses:
            assert static_values(base, schema.cam_path(91, 0))["image_rotation_cw_deg"] == 90
            poses = column_rows(base, f"{schema.rig_path(91)}:Transform3D:translation")
            assert len(poses) == len(FRAMES)
        capture = recording_properties(base, "capture")
        assert capture["num_frames"] == len(FRAMES) and capture["num_cameras"] == (5 if glasses else 2)

        pose = read_back(targets["body_pose"])
        for person in (0, 1):
            keypoints = column_rows(pose, f"{schema.person_coco133_xyz_path(person)}:Points3D:positions")
            assert keypoints[schema.TIMELINE].cast("int64").to_pylist() == expected_times
            assert static_values(pose, schema.person_path(person))["name"] == f"aria0{person + 1}"
        betas = column_rows(pose, f"{schema.person_body_path(0, 'smpl')}:betas")
        np.testing.assert_allclose(np.array(betas.column(betas.column_names[-1]).to_pylist())[:, 0], 0.1, rtol=1e-6)

        mesh = read_back(targets["body_mesh"])
        rows = column_rows(mesh, f"{schema.person_body_path(0, 'mesh')}:Mesh3D:vertex_positions")
        assert rows[schema.TIMELINE].cast("int64").to_pylist() == expected_times[::2]  # 10 Hz of the 20 fps release

        projected = {str(chunk.entity_path) for chunk in read_chunks(targets["projections"]) if not chunk.is_static}
        cameras = [(1, 0), (2, 0)] + ([(91, cam) for cam in range(3)] if glasses else [])
        expected = {schema.person_projection_path(rig, cam, person) for rig, cam in cameras for person in (0, 1)}
        expected |= {schema.boxes_path(1, 0, f"person_{person:02d}") for person in (0, 1)}
        assert projected == expected
