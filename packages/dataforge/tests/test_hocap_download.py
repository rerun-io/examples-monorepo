"""HOCap download and listing against a fake Hub: selection and skip/refetch (hf_fetch_verified's staging, size and hash checks are in test_transports.py)."""

from pathlib import Path
from zipfile import ZipFile

import pytest
import requests
from conftest import zip_bytes
from hub_fake import HubStore

from dataforge import transports
from dataforge.datasets import hocap
from dataforge.datasets.base import RemoteSequence
from dataforge.datasets.hocap import HocapConfig
from dataforge.datasets.hocap_source import SOURCE_REPO, SOURCE_REVISION
from dataforge.transports import HfFileInfo, hf_fetch_verified, hf_lfs_files


def archive(name: str, hub: HubStore) -> bytes:
    """What the Hub holds for one archive."""
    return hub.files[SOURCE_REPO, name]


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> HubStore:
    keys = ("subject_1/a", "subject_1/b", "subject_2/a")
    store = HubStore({SOURCE_REPO: SOURCE_REVISION})
    members: dict[str, list[str]] = {
        "calibration.zip": ["calibration/x"],
        "models.zip": ["models/x"],
        "labels.zip": ["labels/x"],
        "poses.zip": [f"{key}/poses_m.npy" for key in keys],
        "subject_1.zip": ["subject_1/a/meta.yaml", "subject_1/b/meta.yaml"],
        "subject_2.zip": ["subject_2/a/meta.yaml"],
    }
    for name, names in members.items():
        store.add(SOURCE_REPO, name, zip_bytes(dict.fromkeys(names, b"")), lfs=True)
    store.install(monkeypatch, hocap, transports)
    return store


def test_listing_names_each_subject_archive_and_fetches_nothing(tmp_path: Path, hub: HubStore) -> None:
    listed = HocapConfig(root=tmp_path).setup().remote_sequences()
    sizes = {name: len(archive(f"{name}.zip", hub)) for name in ("subject_1", "subject_2")}
    assert listed == [RemoteSequence(name, size, (f"{name}.zip",)) for name, size in sizes.items()]
    assert hub.fetched == []


def test_a_listed_subject_key_selects_all_its_sequences(tmp_path: Path, hub: HubStore) -> None:
    HocapConfig(root=tmp_path, sequences=("subject_1",)).setup().download()
    assert hub.fetched == ["poses.zip", "calibration.zip", "models.zip", "labels.zip", "subject_1.zip"]
    dataset = HocapConfig(root=tmp_path, sequences=("subject_1",)).setup()
    assert [identity.recording_id for identity, _ in dataset.discover()] == ["hocap__subject_1__a", "hocap__subject_1__b"]
    # Deleting the listed files after conversion leaves only the shared archives (and the empty staging dir).
    (tmp_path / "subject_1.zip").unlink()
    assert sorted(path.name for path in tmp_path.iterdir() if path.is_file()) == ["calibration.zip", "labels.zip", "models.zip", "poses.zip"]


def test_subset_fetches_shared_archives_and_its_subject_only(tmp_path: Path, hub: HubStore, capsys: pytest.CaptureFixture[str]) -> None:
    HocapConfig(root=tmp_path, sequences=("subject_2/a",)).setup().download()
    assert hub.fetched == ["poses.zip", "calibration.zip", "models.zip", "labels.zip", "subject_2.zip"]
    assert "fetched 5 file(s)" in capsys.readouterr().out
    dataset = HocapConfig(root=tmp_path).setup()
    assert [identity.recording_id for identity, _ in dataset.discover()] == ["hocap__subject_2__a"]


def test_rerun_skips_complete_files_and_fetches_the_rest(tmp_path: Path, hub: HubStore, capsys: pytest.CaptureFixture[str]) -> None:
    HocapConfig(root=tmp_path, sequences=("subject_1/a",)).setup().download()
    hub.calls.clear()
    HocapConfig(root=tmp_path).setup().download()
    assert hub.fetched == ["subject_2.zip"]
    assert "fetched 1 file(s)" in capsys.readouterr().out.splitlines()[-1]
    # A deleted, converted subject is fetched again; the others stay untouched.
    (tmp_path / "subject_1.zip").unlink()
    hub.calls.clear()
    HocapConfig(root=tmp_path).setup().download()
    assert hub.fetched == ["subject_1.zip"]


def test_unknown_selection_fails_after_only_the_pose_index(tmp_path: Path, hub: HubStore) -> None:
    with pytest.raises(ValueError, match="subject_3/a"):
        HocapConfig(root=tmp_path, sequences=("subject_3/a",)).setup().download()
    assert hub.fetched == ["poses.zip"]


@pytest.mark.integration
def test_real_fetch_of_the_calibration_archive(tmp_path: Path) -> None:
    try:
        requests.head("https://huggingface.co", timeout=10)
    except requests.RequestException as error:
        pytest.skip(f"HuggingFace unreachable ({error}); needs {SOURCE_REPO} calibration.zip")
    remote: HfFileInfo = hf_lfs_files(SOURCE_REPO, ["calibration.zip"], revision=SOURCE_REVISION)[0]
    assert hf_fetch_verified(SOURCE_REPO, remote, local_dir=tmp_path, revision=SOURCE_REVISION)
    assert not hf_fetch_verified(SOURCE_REPO, remote, local_dir=tmp_path, revision=SOURCE_REVISION)
    listed = HocapConfig(root=tmp_path).setup().remote_sequences()
    assert [row.key for row in listed] == [f"subject_{n}" for n in range(1, 10)]
    assert all(row.size_bytes > 3e9 for row in listed)
    with ZipFile(tmp_path / "calibration.zip") as archive:
        assert any(name.startswith("calibration/extrinsics/") for name in archive.namelist())
