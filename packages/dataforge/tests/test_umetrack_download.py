"""UmeTrack's GitHub download: index, listing, selection, resume and verification, with a fake transport."""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests
from serde.json import to_json

from dataforge import transports
from dataforge.apis import download
from dataforge.datasets import umetrack_remote
from dataforge.datasets.umetrack import UmetrackConfig
from dataforge.datasets.umetrack_remote import INDEX_NAME, REVISION, RemoteFile, RemoteIndex, exclusive, fetch_file, load_index

KEYS: tuple[str, str] = ("real/hand_hand/testing/user_05/recording_00", "synthetic/separate_hand/training/user_52/recording_27")
CONTENT: dict[str, bytes] = {f"{key}.{ext}": f"{key} {ext} bytes".encode() for key in KEYS for ext in ("json", "mp4")}


def digest_of(path: str, data: bytes) -> str:
    if path.endswith(".mp4"):
        return f"sha256:{hashlib.sha256(data).hexdigest()}"
    return f"git-sha1:{hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()}"


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A raw root holding only the cached index, as after one ``--list-remote``."""
    monkeypatch.setattr(umetrack_remote, "RECORDINGS", len(KEYS))
    root = tmp_path / "raw_data"
    root.mkdir()
    files = [RemoteFile(path, len(data), digest_of(path, data)) for path, data in sorted(CONTENT.items())]
    (root / INDEX_NAME).write_text(to_json(RemoteIndex(REVISION, files)))
    return root


@pytest.fixture
def fetched(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace ``http_fetch`` with a local writer that resumes like the real one; record the URLs."""
    urls: list[str] = []

    def fake_http_fetch(url: str, *, dest: Path) -> Path:
        urls.append(url)
        data = CONTENT[url.split("/raw_data/", 1)[1]]
        dest.parent.mkdir(parents=True, exist_ok=True)
        have = dest.stat().st_size if dest.is_file() else 0
        with dest.open("ab") as sink:
            sink.write(data[have:])
        return dest

    monkeypatch.setattr(umetrack_remote, "http_fetch", fake_http_fetch)
    monkeypatch.setattr(umetrack_remote, "build_index", lambda: pytest.fail("a cached index must not be rebuilt"))
    return urls


def test_listing_comes_from_the_cached_index(root: Path, fetched: list[str]) -> None:
    listing = UmetrackConfig(root=root).setup().remote_sequences()
    assert [sequence.key for sequence in listing] == sorted(KEYS)
    for sequence in listing:
        assert sequence.files == (f"{sequence.key}.json", f"{sequence.key}.mp4")
        assert sequence.size_bytes == sum(len(CONTENT[path]) for path in sequence.files)
    assert fetched == []


def test_download_fetches_the_selection_at_the_pinned_revision_then_skips_it(root: Path, fetched: list[str]) -> None:
    dataset = UmetrackConfig(root=root, sequences=(KEYS[1],)).setup()
    dataset.download()
    assert set(fetched) == {
        f"https://raw.githubusercontent.com/facebookresearch/UmeTrack_data/{REVISION}/raw_data/{KEYS[1]}.json",
        f"https://media.githubusercontent.com/media/facebookresearch/UmeTrack_data/{REVISION}/raw_data/{KEYS[1]}.mp4",
    }
    assert not (root / f"{KEYS[0]}.json").exists()
    assert [identity.recording_id for identity, _ in dataset.discover()] == ["umetrack__" + KEYS[1].replace("/", "__")]
    dataset.download()
    assert len(fetched) == 2, "complete files cost no request"


def test_download_all_resumes_a_partial_file(root: Path, fetched: list[str]) -> None:
    partial = root / f"{KEYS[0]}.mp4.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(CONTENT[f"{KEYS[0]}.mp4"][:5])
    UmetrackConfig(root=root).setup().download()
    assert len(fetched) == 4
    assert all((root / path).read_bytes() == data for path, data in CONTENT.items())
    assert not partial.exists()


def test_a_repeated_key_is_fetched_once(root: Path, fetched: list[str]) -> None:
    UmetrackConfig(root=root, sequences=(KEYS[0], KEYS[0])).setup().download()
    assert len(fetched) == 2


def test_a_second_download_on_the_same_root_is_refused(root: Path, fetched: list[str]) -> None:
    with exclusive(root), pytest.raises(RuntimeError, match="another UmeTrack download"):
        UmetrackConfig(root=root).setup().download()
    assert fetched == []


def test_unknown_selection_is_refused(root: Path, fetched: list[str]) -> None:
    with pytest.raises(ValueError, match="recording_99"):
        UmetrackConfig(root=root, sequences=("real/hand_hand/testing/user_05/recording_99",)).setup().download()
    assert fetched == []


def test_size_and_hash_mismatches_never_land_under_the_final_name(root: Path, fetched: list[str]) -> None:
    wrong_size = root / f"{KEYS[0]}.json"
    wrong_size.parent.mkdir(parents=True)
    wrong_size.write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="1 UmeTrack files failed(.|\n)*delete it and rerun"):
        UmetrackConfig(root=root, sequences=(KEYS[0],)).setup().download()
    assert wrong_size.read_bytes() == b"truncated", "download never overwrites or deletes a final file"
    assert (root / f"{KEYS[0]}.mp4").is_file(), "the other transfers still complete"

    path = f"{KEYS[1]}.mp4"
    corrupt = RemoteFile(path, len(CONTENT[path]), "sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="next run restarts it"):
        fetch_file(root, corrupt)
    assert not (root / path).exists()
    assert not (root / f"{path}.partial").exists(), "a partial proven wrong is removed, or every rerun would stick on it"
    assert fetch_file(root, RemoteFile(path, len(CONTENT[path]), digest_of(path, CONTENT[path])))


def test_a_partial_longer_than_the_remote_is_restarted(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def stale(url: str, *, dest: Path) -> Path:
        raise transports.StaleLocalFile(f"{dest} is too long")

    monkeypatch.setattr(umetrack_remote, "http_fetch", stale)
    path = f"{KEYS[0]}.json"
    partial = root / f"{path}.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"x" * 1000)
    with pytest.raises(ValueError, match="next run restarts it"):
        fetch_file(root, RemoteFile(path, len(CONTENT[path]), digest_of(path, CONTENT[path])))
    assert not partial.exists()


@pytest.mark.parametrize(
    ("path", "digest"),
    [
        ("../../main/raw_data/real/a/b/user_00/recording_00.json", "git-sha1:" + "a" * 40),
        ("real/a/b/user_00/recording_00.json", "md5:" + "a" * 32),
        ("real/a/b/user_00/recording_00.json", "git-sha1:abc"),
    ],
)
def test_malformed_index_rows_are_refused(path: str, digest: str) -> None:
    with pytest.raises(ValueError, match="malformed"):
        RemoteFile(path, 1, digest)


def test_an_index_must_pair_every_recording_and_list_each_path_once() -> None:
    json_file = RemoteFile("real/a/b/user_00/recording_00.json", 1, "git-sha1:" + "a" * 40)
    with pytest.raises(ValueError, match="one JSON and one MP4"):
        RemoteIndex(REVISION, [json_file])
    with pytest.raises(ValueError, match="twice"):
        RemoteIndex(REVISION, [json_file, json_file])


def test_load_index_builds_caches_and_rereads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = [RemoteFile(path, len(data), digest_of(path, data)) for path, data in sorted(CONTENT.items())]
    monkeypatch.setattr(umetrack_remote, "RECORDINGS", len(KEYS))
    monkeypatch.setattr(umetrack_remote, "build_index", lambda: RemoteIndex(REVISION, files))
    root = tmp_path / "fresh"
    assert load_index(root).files == files
    assert [p.name for p in root.iterdir()] == [INDEX_NAME], "no temp file is left behind"
    monkeypatch.setattr(umetrack_remote, "build_index", lambda: pytest.fail("a cached index must not be rebuilt"))
    assert load_index(root).files == files
    monkeypatch.setattr(umetrack_remote, "RECORDINGS", len(KEYS) + 1)
    with pytest.raises(ValueError, match="expected 6; delete it to rebuild"):
        load_index(root)
    (root / INDEX_NAME).write_text("{not json")
    with pytest.raises(ValueError, match="delete it to rebuild"):
        load_index(root)


def test_build_index_reads_the_tree_and_lfs_pointers(monkeypatch: pytest.MonkeyPatch) -> None:
    video = b"stacked video"
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{hashlib.sha256(video).hexdigest()}\nsize {len(video)}\n"
    tree = (
        '{"sha":"x","truncated":false,"tree":['
        '{"path":"README.md","type":"blob","sha":"r","size":10},'
        '{"path":"raw_data/real","type":"tree","sha":"t"},'
        f'{{"path":"raw_data/real/a/b/user_00/recording_00.json","type":"blob","sha":"{"ab" * 20}","size":7,"url":"u"}},'
        '{"path":"raw_data/real/a/b/user_00/recording_00.mp4","type":"blob","sha":"def","size":133,"url":"u"}]}'
    )
    monkeypatch.setattr(umetrack_remote, "http_text", lambda url, **_: tree if "api.github.com" in url else pointer)
    index = umetrack_remote.build_index()
    assert index.files == [
        RemoteFile("real/a/b/user_00/recording_00.json", 7, f"git-sha1:{'ab' * 20}"),
        RemoteFile("real/a/b/user_00/recording_00.mp4", len(video), f"sha256:{hashlib.sha256(video).hexdigest()}"),
    ]


def ok_response(text: str) -> requests.Response:
    return Mock(spec=requests.Response, text=text)  # passes isinstance; raise_for_status() does nothing


def test_listing_stdout_stays_json_lines_when_the_build_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    video = b"stacked video"
    pointer = f"version https://git-lfs.github.com/spec/v1\noid sha256:{hashlib.sha256(video).hexdigest()}\nsize {len(video)}\n"
    entry = '{{"path":"raw_data/real/a/b/user_00/recording_00.{ext}","type":"blob","sha":"{sha}","size":7}}'
    tree = f'{{"truncated":false,"tree":[{entry.format(ext="json", sha="ab" * 20)},{entry.format(ext="mp4", sha="cd" * 20)}]}}'
    failed: list[str] = []

    def get(url: str, **_: object) -> requests.Response:
        if url not in failed:
            failed.append(url)
            raise requests.ConnectionError("reset")
        return ok_response(tree if "api.github.com" in url else pointer)

    monkeypatch.setattr(umetrack_remote, "RECORDINGS", 1)
    monkeypatch.setattr(transports.time, "sleep", lambda _: None)
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests.Session, "get", lambda self, url, **kwargs: get(url, **kwargs))
    download.main(download.Config(dataset=UmetrackConfig(root=tmp_path / "raw"), list_remote=True))
    captured = capsys.readouterr()
    assert len(failed) == 2 and "warning" in captured.err and "waiting" in captured.err
    assert [json.loads(line)["key"] for line in captured.out.splitlines()] == ["real/a/b/user_00/recording_00"]


def test_blueprints_need_no_raw_files(tmp_path: Path) -> None:
    dataset = UmetrackConfig(root=tmp_path / "pruned").setup()
    dataset.default_blueprint()
    dataset.table_blueprint()


@pytest.mark.integration
@pytest.mark.parametrize(
    "remote",
    [
        RemoteFile("synthetic/hand_hand/training/user_08/recording_01.json", 2604536, "git-sha1:25dce18cbbea07d312384865bc2d9b8d322269e6"),
        RemoteFile(
            "synthetic/hand_hand/training/user_08/recording_01.mp4",
            3964019,
            "sha256:8eaa5bec2a8afb2fea57e0bd55a58756ebaa43b68a817a18a514414a2ff08956",
        ),
    ],
    ids=["label-raw-host", "video-lfs-media-host"],
)
def test_fetch_one_real_recording_file_from_github(tmp_path: Path, remote: RemoteFile) -> None:
    try:
        requests.head(remote.url, timeout=10)
    except requests.RequestException as error:
        pytest.skip(f"GitHub unreachable for {remote.url}: {error}")
    assert fetch_file(tmp_path, remote)
    assert (tmp_path / remote.path).stat().st_size == remote.size_bytes
    assert not fetch_file(tmp_path, remote)
