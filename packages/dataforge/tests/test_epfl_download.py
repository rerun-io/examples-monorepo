"""EPFL download: Hub listing, subset selection, resume/skip and verification, against a fake Hub."""

from pathlib import Path

import httpx
import pytest
from hub_fake import HubStore
from huggingface_hub.errors import HfHubHTTPError

from dataforge import transports
from dataforge.datasets import epfl_download
from dataforge.datasets.epfl import EpflConfig
from dataforge.datasets.epfl_download import SMPL_FILE, session_files
from dataforge.datasets.epfl_source import SOURCE_REVISION
from dataforge.transports import matches_file

KEYS = ("test/YH2003/2023_05_17_09_08_58", "train/YH2007/2023_10_30_10_05_27")


def content(path: str) -> bytes:
    return f"bytes of {path}\n".encode()


def epfl_hub(monkeypatch: pytest.MonkeyPatch, *, incomplete: str = "train/YH2099/2023_01_01_00_00_00") -> HubStore:
    """The pose/video repo with KEYS complete, plus the SMPL model repo; content(path) is every file's bytes."""
    store = HubStore({epfl_download.SOURCE_REPO: SOURCE_REVISION, epfl_download.SMPL_REPO: epfl_download.SMPL_REVISION})
    # The incomplete session ships its poses but no videos.
    for path in [path for key in KEYS for path in session_files(key)] + list(session_files(incomplete)[:4]) + ["manifests/sequences.csv"]:
        store.add(epfl_download.SOURCE_REPO, path, content(path), lfs=path.endswith((".mp4", ".csv")))
    store.add(epfl_download.SMPL_REPO, SMPL_FILE, content(SMPL_FILE), lfs=True)
    store.install(monkeypatch, epfl_download, transports)
    return store


def corrupt(hub: HubStore, path: str) -> None:
    """Make every download of ``path`` write bytes that fail its check."""
    hub.served[epfl_download.SOURCE_REPO, path] = b"corrupt"


def test_remote_sequences_lists_complete_sessions(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    hub = epfl_hub(monkeypatch)
    listed = EpflConfig(root=Path("/nonexistent")).setup().remote_sequences()
    assert [sequence.key for sequence in listed] == list(KEYS)
    sizes = {path: len(data) for (_, path), data in hub.files.items()}
    for sequence in listed:
        assert sequence.files == session_files(sequence.key)
        assert sequence.size_bytes == sum(sizes[path] for path in sequence.files)
    assert "YH2099" in capsys.readouterr().err
    assert hub.fetched == [] and epfl_download.SMPL_REPO not in hub.listed


def test_download_subset_then_resume_skips_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    hub = epfl_hub(monkeypatch)
    config = EpflConfig(root=tmp_path / "raw", sequences=(KEYS[1],))
    dataset = config.setup()
    dataset.download()
    assert hub.listed.count(epfl_download.SMPL_REPO) == 1
    assert sorted(hub.fetched) == sorted([SMPL_FILE, *session_files(KEYS[1])])
    assert (tmp_path / "raw/body_models/smpl/SMPL_NEUTRAL.pkl").read_bytes() == content(SMPL_FILE)
    assert [identity.parts for identity, _ in dataset.discover()] == [tuple(KEYS[1].split("/"))]
    assert "fetched 18 file(s)" in capsys.readouterr().out  # the session's 17 and SMPL

    hub.calls.clear()
    dataset.download()
    assert hub.fetched == [] and epfl_download.SMPL_REPO not in hub.listed
    assert "fetched 0 file(s)" in capsys.readouterr().out

    # A truncated file is fetched again (forced past the Hub's staging metadata); others and the in-place SMPL stay skipped.
    truncated = tmp_path / "raw" / session_files(KEYS[1])[0]
    truncated.write_bytes(b"x")
    dataset.download()
    assert hub.fetched == hub.forced == [session_files(KEYS[1])[0]]
    assert epfl_download.SMPL_REPO not in hub.listed
    assert truncated.read_bytes() == content(session_files(KEYS[1])[0])


def test_download_all_splits_videos_and_smpl_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = epfl_hub(monkeypatch)
    config = EpflConfig(root=tmp_path / "raw", video_root=tmp_path / "videos", smpl_model_root=tmp_path / "models")
    config.setup().download()
    assert hub.listed.count(epfl_download.SMPL_REPO) == 1
    assert len(hub.fetched) == 1 + len(KEYS) * len(session_files(KEYS[0]))
    assert (tmp_path / "models/smpl/SMPL_NEUTRAL.pkl").is_file()
    assert (tmp_path / "videos" / session_files(KEYS[0])[-1]).is_file()
    assert (tmp_path / "raw" / session_files(KEYS[0])[0]).is_file()
    assert not (tmp_path / "raw/Public_release_videos").exists()
    assert len(config.setup().discover()) == len(KEYS)


def test_download_rejects_unknown_selection_and_bad_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = epfl_hub(monkeypatch)
    with pytest.raises(ValueError, match=r"epfl: unknown sequence\(s\)"):
        EpflConfig(root=tmp_path, sequences=("train/YH2099/2023_01_01_00_00_00",)).setup().download()
    bad = session_files(KEYS[0])[4:6]
    for path in bad:
        corrupt(hub, path)
    dataset = EpflConfig(root=tmp_path, sequences=(KEYS[0],)).setup()
    for _ in range(2):
        # Both bad files are reported in one run; the good ones still land.
        with pytest.raises(ValueError, match="2 failed") as error:
            dataset.download()
        assert all(path in str(error.value) for path in bad)
        assert all(not (tmp_path / path).exists() for path in bad)
        assert (tmp_path / session_files(KEYS[0])[0]).is_file()
    # Every failed copy is kept under its own name.
    assert len(list((tmp_path / ".download" / bad[0]).parent.glob(f"{Path(bad[0]).name}.*.mismatch"))) == 2
    hub.served.clear()
    dataset.download()
    assert all((tmp_path / path).read_bytes() == content(path) for path in bad)


def test_interrupted_replacement_keeps_old_bytes_and_drops_stale_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = epfl_hub(monkeypatch)
    dataset = EpflConfig(root=tmp_path, sequences=(KEYS[0],)).setup()
    dataset.download()
    target = session_files(KEYS[0])[0]
    (tmp_path / target).write_bytes(b"old")
    stale = tmp_path / ".download/.cache/huggingface/download" / Path(target).parent / "abc.etag.incomplete"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"partial")

    def interrupted(repo_id: str, filename: str, **kwargs: object) -> str:
        if filename == target:
            raise KeyboardInterrupt
        return hub.hf_hub_download(repo_id, filename, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(epfl_download, "hf_hub_download", interrupted)
    with pytest.raises(KeyboardInterrupt):
        dataset.download()
    assert (tmp_path / target).read_bytes() == b"old"
    assert not stale.exists()


def test_smpl_in_place_skips_private_repo_and_denied_repo_names_the_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub = epfl_hub(monkeypatch)
    # A synthetic signed query stands in for the redirect the real client would carry.
    signed = "https://cdn.invalid/smpl?X-Signature=synthetic"
    hub.refused[epfl_download.SMPL_REPO] = HfHubHTTPError(f"403 for {signed}", response=httpx.Response(403, request=httpx.Request("GET", signed)))
    with pytest.raises(ValueError, match="place the official SMPL_NEUTRAL.pkl at") as error:
        EpflConfig(root=tmp_path, sequences=(KEYS[0],)).setup().download()
    assert "X-Signature" not in str(error.value)
    # The sessions landed before the model failure.
    assert all((tmp_path / path).is_file() for path in session_files(KEYS[0]))
    model = tmp_path / "body_models/smpl/SMPL_NEUTRAL.pkl"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"user-placed model")
    hub.calls.clear()
    EpflConfig(root=tmp_path, sequences=(KEYS[0],)).setup().download()
    assert hub.fetched == [] and epfl_download.SMPL_REPO not in hub.listed


def test_discover_skips_pruned_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    epfl_hub(monkeypatch)
    dataset = EpflConfig(root=tmp_path).setup()
    dataset.download()
    for path in session_files(KEYS[0]):
        (tmp_path / path).unlink()
    assert [key for _, key in dataset.discover()] == [KEYS[1]]
    assert "17 of 17 raw files missing" in capsys.readouterr().out
    # Register's blueprints read nothing from the raw tree.
    dataset.default_blueprint()
    dataset.table_blueprint()


@pytest.mark.integration
def test_real_hub_fetches_one_small_file(tmp_path: Path) -> None:
    folder = f"{epfl_download.POSE_DIR}/{KEYS[0]}/annotations"
    try:
        (file,) = [file for file in transports.hf_list_files(epfl_download.SOURCE_REPO, SOURCE_REVISION, folder) if file.path.endswith(".json")]
        assert epfl_download.fetch(epfl_download.SOURCE_REPO, SOURCE_REVISION, file, tmp_path / file.path, tmp_path / ".download")
    except (httpx.TransportError, HfHubHTTPError, epfl_download.FetchError, OSError) as error:
        pytest.skip(f"HuggingFace Hub unreachable: {error}")
    assert matches_file(tmp_path / file.path, file.integrity)
    assert not epfl_download.fetch(epfl_download.SOURCE_REPO, SOURCE_REVISION, file, tmp_path / file.path, tmp_path / ".download")
