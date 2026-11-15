"""HOT3D download from a synthetic URL file, with the CDN replaced by an in-memory transport."""

import hashlib
import json
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path

import pytest
import requests
import tyro
from conftest import zip_bytes
from serde import from_dict

from dataforge import meta_cdn, transports
from dataforge.apis import download as download_api
from dataforge.datasets.hot3d import Hot3dAriaConfig, Hot3dQuest3Config
from dataforge.datasets.hot3d_download import SOURCE, read_url_file
from dataforge.meta_cdn import CdnFile

FUTURE: str = "oe=7FFFFFFF"
"""An ``oe`` expiry in 2038."""


def entry(bodies: dict[str, bytes], name: str, data: bytes, *, oe: str = FUTURE, size: int | None = None) -> dict[str, object]:
    """A URL-file entry serving ``data`` from a fake signed URL."""
    url = f"https://cdn.invalid/{name}?{oe}&oh=SECRETSIG"
    bodies[url] = data
    return {"filename": name, "sha1sum": hashlib.sha1(data).hexdigest(), "file_size_bytes": len(data) if size is None else size, "download_url": url}


def gt_members() -> dict[str, bytes]:
    metadata = json.dumps({"have_hand_object_pose_gt": True, "participant_id": "P0001", "object_uids": ["7"]}).encode()
    return {"metadata.json": metadata, "camera_models.json": b"[]", "masks/mask_qa_pass.csv": b"m", "box2d_hands.csv": b"b", "license.txt": b"l"}


@pytest.fixture
def cdn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, bytes], Path, Path, list[str]]:
    """Two Aria sequences + assets; returns the served bodies, both URL files and the fetch log."""
    bodies: dict[str, bytes] = {}
    fetched: list[str] = []
    sequences = {
        name: {
            "main_vrs": entry(bodies, f"{name}_main_recording.vrs", name.encode() * 10),
            "ground_truth": entry(bodies, f"{name}_ground_truth.zip", zip_bytes(gt_members())),
            "hand_data": entry(bodies, f"{name}_hand_data.zip", zip_bytes({"hands.jsonl": b"h", "license.txt": b"l"})),
            "mps_artifacts": entry(bodies, f"{name}_mps_artifacts.zip", b"never fetched"),
        }
        for name in ("P0001_aaaa", "P0001_bbbb")
    }
    groups = {
        "ground_truth": ["box2d_hands.csv", "metadata.json", "camera_models.json", "masks/mask_qa_pass.csv", "license.txt"],
        "hand_data": ["hands.jsonl", "license.txt"],
    }
    url_file = tmp_path / "urls/Hot3DAria_download_urls.json"
    url_file.parent.mkdir()
    url_file.write_text(
        json.dumps({"sequences": sequences, "sequence_config": {"dataset_name": "Hot3DAria", "data_groups": groups, "release": "v4.0.0"}})
    )
    census = json.dumps({"7": {"instance_id": "7", "instance_name": "mug"}}).encode()
    assets = {"7.glb": b"glb", "instance.json": census, "license.txt": b"l"}
    assets_file = tmp_path / "urls/Hot3DAssets_download_urls.json"
    assets_file.write_text(
        json.dumps(
            {
                "sequences": {"assets": {"assets": entry(bodies, "assets.zip", zip_bytes(assets))}},
                "sequence_config": {"dataset_name": "Hot3DAssets", "data_groups": {"assets": list(assets)}},
            }
        )
    )

    def fake_fetch(url: str, *, dest: Path, label: str | None = None) -> Path:
        assert label is not None and "SECRETSIG" not in label
        fetched.append(url.split("?")[0].rsplit("/", 1)[1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        have = dest.stat().st_size if dest.is_file() else 0
        with dest.open("ab") as sink:
            sink.write(bodies[url][have:])
        return dest

    monkeypatch.setattr(transports, "http_fetch", fake_fetch)
    return bodies, url_file, assets_file, fetched


def test_listing_counts_only_fetched_types(cdn: tuple[dict[str, bytes], Path, Path, list[str]], tmp_path: Path) -> None:
    bodies, url_file, _, fetched = cdn
    listing = Hot3dAriaConfig(root=tmp_path / "raw", url_file=url_file).setup().remote_sequences()
    assert [row.key for row in listing] == ["P0001_aaaa", "P0001_bbbb"]
    urls = read_url_file(url_file)
    assert listing[0].size_bytes == sum(urls.sequences["P0001_aaaa"][kind].file_size_bytes for kind in ("main_vrs", "ground_truth", "hand_data"))
    assert listing[0].files == (
        "aria/P0001_aaaa/recording.vrs",
        "aria/P0001_aaaa/metadata.json",
        "aria/P0001_aaaa/camera_models.json",
        "aria/P0001_aaaa/masks/mask_qa_pass.csv",
        "aria/P0001_aaaa/hands.jsonl",
    )
    assert not fetched


def test_subset_download_then_discover_and_rerun_skips(
    cdn: tuple[dict[str, bytes], Path, Path, list[str]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, url_file, assets_file, fetched = cdn
    root = tmp_path / "raw"
    config = Hot3dAriaConfig(root=root, url_file=url_file, assets_url_file=assets_file, sequences=("P0001_aaaa",))
    dataset = config.setup()
    dataset.download()
    assert sorted(fetched) == ["P0001_aaaa_ground_truth.zip", "P0001_aaaa_hand_data.zip", "P0001_aaaa_main_recording.vrs", "assets.zip"]
    folder = root / "aria/P0001_aaaa"
    assert (folder / "recording.vrs").read_bytes() == b"P0001_aaaa" * 10
    assert sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()) == [
        "camera_models.json",
        "hands.jsonl",
        "masks/mask_qa_pass.csv",
        "metadata.json",
        "recording.vrs",
    ]
    assert sorted(path.name for path in (root / "assets").iterdir()) == [".lock", "7.glb", "instance.json"]
    assert "SECRETSIG" not in (root / "Hot3DAria_manifest.json").read_text()
    # Discovery lists only the complete sequence and does not need the URL file.
    found = Hot3dAriaConfig(root=root).setup().discover()
    assert [identity.recording_id for identity, _ in found] == ["hot3d-aria__P0001_aaaa"]
    listing = Hot3dAriaConfig(root=root).setup().remote_sequences()
    assert [row.key for row in listing] == ["P0001_aaaa", "P0001_bbbb"]

    fetched.clear()
    Hot3dAriaConfig(root=root, url_file=url_file, sequences=("P0001_aaaa",)).setup().download()
    assert fetched == []
    output = capsys.readouterr().out
    assert "fetched 0 file(s)" in output and "3 already complete" in output
    assert "SECRETSIG" not in output

    # Deleting a converted sequence's files leaves discovery (and the other sequence) unharmed.
    for relative in listing[0].files:
        (root / relative).unlink()
    assert Hot3dAriaConfig(root=root).setup().discover() == []


def test_partial_file_resumes(cdn: tuple[dict[str, bytes], Path, Path, list[str]], tmp_path: Path) -> None:
    bodies, url_file, _, _ = cdn
    urls = read_url_file(url_file)
    vrs: CdnFile = urls.sequences["P0001_bbbb"]["main_vrs"]
    dest = tmp_path / "recording.vrs"
    dest.with_name("recording.vrs.part").write_bytes(bodies[vrs.download_url][:7])
    meta_cdn.fetch_verified(vrs, dest, SOURCE)
    assert dest.read_bytes() == b"P0001_bbbb" * 10
    assert not dest.with_name("recording.vrs.part").exists()


def test_size_and_sha1_mismatch(cdn: tuple[dict[str, bytes], Path, Path, list[str]], tmp_path: Path) -> None:
    bodies, _, _, _ = cdn
    short = from_dict(CdnFile, entry(bodies, "short.vrs", b"abc", size=4))
    with pytest.raises(ValueError, match="holds 3 bytes"):
        meta_cdn.fetch_verified(short, tmp_path / "short.vrs", SOURCE)
    assert (tmp_path / "short.vrs.part").exists() and not (tmp_path / "short.vrs").exists()
    corrupt = CdnFile(filename="bad.vrs", sha1sum="0" * 40, file_size_bytes=3, download_url=next(iter(bodies)))
    bodies[corrupt.download_url] = b"xyz"
    with pytest.raises(ValueError, match="sha1 differs"):
        meta_cdn.fetch_verified(corrupt, tmp_path / "bad.vrs", SOURCE)
    assert not (tmp_path / "bad.vrs.part").exists() and not (tmp_path / "bad.vrs").exists()


def test_expired_and_missing_url_files(cdn: tuple[dict[str, bytes], Path, Path, list[str]], tmp_path: Path) -> None:
    bodies, _, _, fetched = cdn
    old = from_dict(CdnFile, entry(bodies, "old.vrs", b"abc", oe="oe=5F5E1000"))
    with pytest.raises(ValueError, match="expired on 2020-09-13"):
        meta_cdn.fetch_verified(old, tmp_path / "old.vrs", SOURCE)
    assert not fetched
    with pytest.raises(FileNotFoundError, match="projectaria.com"):
        Hot3dAriaConfig(root=tmp_path / "raw").setup().download()
    with pytest.raises(FileNotFoundError, match="--url-file"):
        Hot3dAriaConfig(root=tmp_path / "raw").setup().remote_sequences()


def test_error_flags_parse() -> None:
    """The flags the errors name are the ones the CLI accepts."""
    config = tyro.cli(download_api.Config, args=["hot3d-aria", "--url-file", "u.json", "--assets-url-file", "a.json"])
    assert isinstance(config.dataset, Hot3dAriaConfig)
    assert config.dataset.url_file == Path("u.json") and config.dataset.assets_url_file == Path("a.json")


def test_no_gt_sequence_downloads_base_only(cdn: tuple[dict[str, bytes], Path, Path, list[str]], tmp_path: Path) -> None:
    """A test-split sequence: ground_truth without masks/, 0-byte hand members; download, discover and a rerun all succeed."""
    bodies, url_file, assets_file, fetched = cdn
    urls = json.loads(url_file.read_text())
    metadata = json.dumps({"have_hand_object_pose_gt": False, "participant_id": "P0020", "object_uids": []}).encode()
    urls["sequences"]["P0020_test"] = {
        "main_vrs": entry(bodies, "P0020_test_main_recording.vrs", b"vrs"),
        "ground_truth": entry(bodies, "P0020_test_ground_truth.zip", zip_bytes({"metadata.json": metadata, "camera_models.json": b"[]"})),
        "hand_data": entry(bodies, "P0020_test_hand_data.zip", zip_bytes({"hands.jsonl": b"", "license.txt": b""})),
    }
    url_file.write_text(json.dumps(urls))
    root = tmp_path / "raw"
    Hot3dAriaConfig(root=root, url_file=url_file, assets_url_file=assets_file, sequences=("P0020_test",)).setup().download()
    assert sorted(fetched) == ["P0020_test_ground_truth.zip", "P0020_test_main_recording.vrs", "assets.zip"]
    found = Hot3dAriaConfig(root=root).setup().discover()
    assert [(identity.recording_id, source.metadata.have_hand_object_pose_gt) for identity, source in found] == [("hot3d-aria__P0020_test", False)]
    fetched.clear()
    Hot3dAriaConfig(root=root, url_file=url_file, sequences=("P0020_test",)).setup().download()
    assert fetched == []


def test_interrupted_extraction_reuses_the_verified_zip(cdn: tuple[dict[str, bytes], Path, Path, list[str]], tmp_path: Path) -> None:
    bodies, url_file, assets_file, fetched = cdn
    root = tmp_path / "raw"
    gt: CdnFile = read_url_file(url_file).sequences["P0001_aaaa"]["ground_truth"]
    folder = root / "aria/P0001_aaaa"
    folder.mkdir(parents=True)
    (folder / gt.filename).write_bytes(bodies[gt.download_url])
    Hot3dAriaConfig(root=root, url_file=url_file, assets_url_file=assets_file, sequences=("P0001_aaaa",)).setup().download()
    assert "P0001_aaaa_ground_truth.zip" not in fetched
    assert (folder / "masks/mask_qa_pass.csv").is_file() and not (folder / gt.filename).exists()


def test_wrong_files_are_rejected_without_leaking_urls(cdn: tuple[dict[str, bytes], Path, Path, list[str]], tmp_path: Path) -> None:
    _, url_file, assets_file, fetched = cdn
    root = tmp_path / "raw"
    with pytest.raises(ValueError, match="Hot3DAria's; hot3d-quest3 needs Hot3DQuest"):
        Hot3dQuest3Config(root=root, url_file=url_file, assets_url_file=assets_file).setup().download()
    with pytest.raises(ValueError, match="Hot3DAria's; --assets-url-file needs Hot3DAssets"):
        Hot3dAriaConfig(root=root, url_file=url_file, assets_url_file=url_file).setup().download()
    assert not fetched
    broken = json.loads(url_file.read_text())
    broken["sequences"]["P0001_aaaa"]["main_vrs"]["download_url"] = ["https://cdn.invalid/x?oh=SECRETSIG"]
    url_file.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="not a HOT3D URL file") as failure:
        read_url_file(url_file)
    assert "SECRETSIG" not in "".join(traceback.format_exception(failure.value))


def test_unfinished_assets_need_the_assets_url_file(cdn: tuple[dict[str, bytes], Path, Path, list[str]], tmp_path: Path) -> None:
    _, url_file, assets_file, fetched = cdn
    root = tmp_path / "raw"
    Hot3dAriaConfig(root=root, url_file=url_file, assets_url_file=assets_file, sequences=("P0001_aaaa",)).setup().download()
    (root / "assets/instance.json").unlink()
    with pytest.raises(FileNotFoundError, match="--assets-url-file"):
        Hot3dAriaConfig(root=root, url_file=url_file, sequences=("P0001_aaaa",)).setup().download()


def test_unknown_sequence_is_an_error(cdn: tuple[dict[str, bytes], Path, Path, list[str]], tmp_path: Path) -> None:
    _, url_file, _, _ = cdn
    with pytest.raises(ValueError, match=r"hot3d-aria: unknown sequence\(s\) \['P9999_zzzz'\]"):
        Hot3dAriaConfig(root=tmp_path / "raw", url_file=url_file, sequences=("P9999_zzzz",)).setup().download()


def test_labelled_http_fetch_never_prints_the_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    url = "https://cdn.invalid/file?oh=SECRETSIG"

    def refuse(*_args: object, **_kwargs: object) -> requests.Response:
        raise requests.ConnectionError(f"cannot reach {url}")

    monkeypatch.setattr(requests, "head", refuse)
    monkeypatch.setattr(transports, "RETRY_BACKOFF_S", (0.0,))
    with pytest.raises(RuntimeError) as failure:
        transports.http_fetch(url, dest=tmp_path / "file", attempts=2, label="HOT3D file")
    assert "SECRETSIG" not in str(failure.value) and "SECRETSIG" not in capsys.readouterr().out


@pytest.mark.integration
def test_real_cdn_fetch_of_a_tiny_file(tmp_path: Path) -> None:
    """Fetch the 726-byte MPS summary zip of the first Aria sequence and check its sha1."""
    path = os.environ.get("HOT3D_URL_FILE")
    if path is None or not Path(path).is_file():
        pytest.skip("HOT3D URL file absent: set HOT3D_URL_FILE to a Hot3DAria_download_urls*.json from projectaria.com")
    urls = read_url_file(Path(path))
    tiny: CdnFile = min((files[kind] for files in urls.sequences.values() for kind in files), key=lambda item: item.file_size_bytes)
    stamp = meta_cdn.expiry(tiny.download_url)
    if stamp is not None and stamp <= datetime.now(UTC):
        pytest.skip(f"HOT3D URL file absent: {path} expired")
    try:
        requests.head("https://scontent.xx.fbcdn.net", timeout=10)
    except requests.RequestException:
        pytest.skip("offline: the HOT3D CDN is unreachable")
    meta_cdn.fetch_verified(tiny, tmp_path / tiny.filename, SOURCE)
    assert (tmp_path / tiny.filename).stat().st_size == tiny.file_size_bytes
