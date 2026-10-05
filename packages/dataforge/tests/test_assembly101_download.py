"""Assembly101 download and listing against a fake Hub; the integration test fetches one tiny real file."""

import hashlib
from pathlib import Path
from zipfile import ZipFile

import httpx
import pytest
from hub_fake import HubStore
from huggingface_hub import RepoFile
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from dataforge import transports
from dataforge.datasets import assembly101, assembly101_download
from dataforge.datasets.assembly101 import Assembly101Config
from dataforge.datasets.assembly101_download import RemoteFile, fetch_hub_files
from dataforge.datasets.assembly101_source import (
    EXO_SERIALS,
    MIRROR_REPO,
    MIRROR_REVISION,
    OFFICIAL_REPO,
    OFFICIAL_REVISION,
    POSE_MEMBERS,
    SHARED_POSE_MEMBER,
    pose_path,
)
from dataforge.transports import FetchReport, FileIntegrity

SEQUENCES: tuple[str, ...] = ("seq-a", "seq-b", "seq-video-only")
EGO: tuple[str, ...] = ("21110305", "21176623", "21176875", "21179183")


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> HubStore:
    mirror: Path = tmp_path / "remote/mirror"
    for sequence in SEQUENCES:
        folder: Path = mirror / "videos/av1-720-new" / sequence
        folder.mkdir(parents=True)
        for name in [*(f"{serial}_rgb_low" for serial in EXO_SERIALS), *(f"HMC_{serial}_mono10bit_low" for serial in EGO)]:
            (folder / f"{name}.mp4").write_bytes(f"{sequence}/{name}".encode())
    calibration: Path = mirror / "assemblyhands-toolkit/calib/nimble_json_calib/seq-a.json"
    calibration.parent.mkdir(parents=True)
    calibration.write_text("[]")
    (mirror / "manifests").mkdir()
    (mirror / "manifests/sequences.csv").write_text("sequence_name,video_only\nseq-a,False\nseq-b,False\nseq-video-only,True\n")
    with ZipFile(mirror / "AssemblyPoses.zip", "w") as archive:
        for sequence in SEQUENCES[:2]:
            for member in (SHARED_POSE_MEMBER, *POSE_MEMBERS, "hand_bboxes"):
                archive.writestr(f"assembly101_camera_and_hand_poses/{member}/{sequence}.json", f'{{"{member}": "{sequence}"}}')
    notes: Path = tmp_path / "remote/official/annotations/fine-grained-annotations/train.csv"
    notes.parent.mkdir(parents=True)
    notes.write_text("id,video,start_frame,end_frame,action_cls\n")
    store = HubStore({MIRROR_REPO: MIRROR_REVISION, OFFICIAL_REPO: OFFICIAL_REVISION})
    store.add_tree(MIRROR_REPO, mirror, lfs=(".mp4",))
    store.add_tree(OFFICIAL_REPO, tmp_path / "remote/official")
    store.install(monkeypatch, assembly101, assembly101_download)
    return store


def local_config(tmp_path: Path, sequences: tuple[str, ...] | None = None) -> Assembly101Config:
    return Assembly101Config(root=tmp_path / "raw", annotations_root=tmp_path / "raw/official/annotations", sequences=sequences)


def test_listing_splits_per_sequence_and_shared_files(tmp_path: Path, hub: HubStore) -> None:
    listing = local_config(tmp_path).setup().remote_sequences()
    assert [remote.key for remote in listing] == list(SEQUENCES)
    assert len(listing[0].files) == 12 + len(POSE_MEMBERS)
    assert len(listing[2].files) == 12
    assert not any(SHARED_POSE_MEMBER in path or "hand_bboxes" in path for remote in listing for path in remote.files)
    assert listing[0].size_bytes == sum(len(hub.files[MIRROR_REPO, path]) for path in listing[0].files if path.endswith(".mp4")) + sum(
        len(f'{{"{member}": "seq-a"}}') for member in POSE_MEMBERS
    )
    assert hub.fetched == []


def test_subset_download_fetches_shared_once_and_resumes(tmp_path: Path, hub: HubStore, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = local_config(tmp_path).setup()
    assert dataset.discover() == []
    local_config(tmp_path, ("seq-b",)).setup().download()
    root: Path = tmp_path / "raw"
    assert not (root / "videos/av1-720-new/seq-a").exists()
    assert pose_path(root, SHARED_POSE_MEMBER, "seq-a").is_file()
    assert (root / "official/annotations/fine-grained-annotations/train.csv").is_file()
    assert [source.sequence for _, source in dataset.discover()] == ["seq-b"]  # the manifest is read after the download
    first: list[str] = list(hub.fetched)
    assert len(first) == 12 + 3  # videos, nimble file, manifest, annotation; pose members come from the zip

    video: Path = root / "videos/av1-720-new/seq-b/C10095_rgb_low.mp4"
    video.write_bytes(b"partial")
    local_config(tmp_path, ("seq-b", "seq-video-only")).setup().download()
    assert sorted(hub.fetched[len(first) :]) == sorted(
        [str(video.relative_to(root)), *(f"videos/av1-720-new/seq-video-only/{path.name}" for path in video.parent.iterdir())]
    )
    assert hub.forced == []  # the partial file is replaced by a verified staged copy, never downloaded onto
    assert "21 already complete" in capsys.readouterr().out
    assert [source.sequence for _, source in local_config(tmp_path).setup().discover()] == ["seq-b", "seq-video-only"]


def test_discover_skips_pruned_sequences_and_register_surface_survives(tmp_path: Path, hub: HubStore) -> None:
    config = local_config(tmp_path, ("seq-a", "seq-b"))
    config.setup().download()
    for path in next(remote for remote in config.setup().remote_sequences() if remote.key == "seq-a").files:
        (tmp_path / "raw" / path).unlink()
    dataset = local_config(tmp_path).setup()
    assert [source.sequence for _, source in dataset.discover()] == ["seq-b"]
    dataset.default_blueprint()
    dataset.table_blueprint()
    with pytest.raises(FileNotFoundError, match="seq-a"):
        local_config(tmp_path, ("seq-a",)).setup().discover()


def test_unknown_sequence_and_nas_root_are_refused(tmp_path: Path, hub: HubStore) -> None:
    with pytest.raises(ValueError, match=r"assembly101: unknown sequence\(s\)"):
        local_config(tmp_path, ("seq-z",)).setup().download()
    with pytest.raises(ValueError, match="never the NAS"):
        Assembly101Config(root=Path("/mnt/nas/datasets/assembly101")).setup().download()
    assert hub.fetched == []


def test_gated_annotations_warn_and_the_rest_lands(tmp_path: Path, hub: HubStore, capsys: pytest.CaptureFixture[str]) -> None:
    hub.refused[OFFICIAL_REPO] = GatedRepoError("gated", response=httpx.Response(401, request=httpx.Request("GET", "https://huggingface.co")))
    local_config(tmp_path, ("seq-a",)).setup().download()
    root: Path = tmp_path / "raw"
    assert not (root / "official").exists()
    assert pose_path(root, POSE_MEMBERS[0], "seq-a").is_file()
    assert len(list((root / "videos/av1-720-new/seq-a").iterdir())) == 12
    output: str = capsys.readouterr().out
    assert "warning: no access to the gated" in output and "no annotations" in output


def test_transfer_errors_hide_the_signed_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    signed: str = "https://cdn-lfs.hf.co/x.bin?X-Amz-Signature=SECRET"

    def denied(repo_id: str, filename: str, **_: object) -> str:
        raise HfHubHTTPError(f"403 Forbidden for url: {signed}", response=httpx.Response(403, request=httpx.Request("GET", signed)))

    monkeypatch.setattr(assembly101_download, "hf_hub_download", denied)
    with pytest.raises(RuntimeError, match=r"x.bin: HfHubHTTPError \(HTTP 403\)") as caught:
        fetch_hub_files(MIRROR_REPO, MIRROR_REVISION, [RemoteFile("mirror", "x.bin", FileIntegrity(4, None, None))], tmp_path, FetchReport())
    assert "SECRET" not in str(caught.value) and caught.value.__cause__ is None and caught.value.__suppress_context__


def test_size_mismatch_after_fetch_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def short(repo_id: str, filename: str, *, local_dir: str, **_: object) -> str:
        (Path(local_dir) / filename).parent.mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / filename).write_bytes(b"abc")
        return filename

    monkeypatch.setattr(assembly101_download, "hf_hub_download", short)
    with pytest.raises(ValueError, match="source lists 4"):
        fetch_hub_files(MIRROR_REPO, MIRROR_REVISION, [RemoteFile("mirror", "x.bin", FileIntegrity(4, None, None))], tmp_path, FetchReport())
    assert not (tmp_path / "x.bin").exists()
    with pytest.raises(ValueError, match="sha256"):
        fetch_hub_files(MIRROR_REPO, MIRROR_REVISION, [RemoteFile("mirror", "x.bin", FileIntegrity(3, "sha256", "0" * 64))], tmp_path, FetchReport())
    assert not (tmp_path / "x.bin").exists()  # a rejected file never survives for the size-only skip of the next run


def test_an_interrupted_verification_publishes_nothing_and_the_retry_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data: bytes = b"abcd"
    remote = RemoteFile("mirror", "videos/x.mp4", FileIntegrity(len(data), "sha256", hashlib.sha256(data).hexdigest()))
    fetched: list[str] = []

    def fetch(repo_id: str, filename: str, *, local_dir: str, **_: object) -> str:
        fetched.append(filename)
        (Path(local_dir) / filename).parent.mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / filename).write_bytes(data)
        return filename

    def killed(path: Path, expected: FileIntegrity) -> None:
        raise RuntimeError("killed between transfer and verification")

    monkeypatch.setattr(assembly101_download, "hf_hub_download", fetch)
    verify_file = transports.verify_file
    monkeypatch.setattr(transports, "verify_file", killed)
    with pytest.raises(RuntimeError, match="killed"):
        fetch_hub_files(MIRROR_REPO, MIRROR_REVISION, [remote], tmp_path, FetchReport())
    assert not (tmp_path / remote.path).exists()  # nothing unverified where the size-only skip of the next run looks
    monkeypatch.setattr(transports, "verify_file", verify_file)
    report = FetchReport()
    fetch_hub_files(MIRROR_REPO, MIRROR_REVISION, [remote], tmp_path, report)
    assert (tmp_path / remote.path).read_bytes() == data
    assert fetched == [remote.path]  # the retry published the staged transfer instead of fetching again
    assert report.fetched == 1


@pytest.mark.integration
def test_real_manifest_fetch(tmp_path: Path) -> None:
    """One 44 KB mirror file at the pinned revision, through the real transport."""
    try:
        info = assembly101_download.HfApi().get_paths_info(MIRROR_REPO, ["manifests/sequences.csv"], repo_type="dataset", revision=MIRROR_REVISION)
    except (OSError, httpx.TransportError) as error:
        pytest.skip(f"Hugging Face Hub unreachable: {error}")
    assert isinstance(info[0], RepoFile)
    remote = assembly101_download.repo_file("mirror", info[0])
    report = FetchReport()
    fetch_hub_files(MIRROR_REPO, MIRROR_REVISION, [remote], tmp_path, report)
    assert report.fetched_bytes == remote.integrity.size_bytes > 0
    assert (tmp_path / "manifests/sequences.csv").read_text().startswith("sequence_name,")
