import hashlib
from pathlib import Path
from typing import Any

import pytest
from conftest import serve, zip_bytes  # pyrefly: ignore[missing-import]
from hub_fake import HubStore

from dataforge import transports
from dataforge.transports import (
    FetchReport,
    FileIntegrity,
    HfFileInfo,
    IntegrityError,
    StaleLocalFile,
    content_digest,
    hf_fetch,
    hf_fetch_verified,
    hf_lfs_files,
    http_fetch,
    http_index,
    local_verify,
    matches_file,
    parse_apache_index,
    publish_verified,
    verify_file,
)


def test_local_verify_reports_missing_globs(tmp_path: Path) -> None:
    (tmp_path / "sess").mkdir()
    (tmp_path / "sess" / "video_dev0.mp4").write_bytes(b"x")
    missing: list[str] = local_verify(tmp_path, required=("sess/video_*.mp4", "sess/IMUWriter_*.db"))
    assert missing == ["sess/IMUWriter_*.db"]


def test_local_verify_ok_when_all_present(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"x")
    assert local_verify(tmp_path, required=("a.txt",)) == []


def test_local_verify_missing_root(tmp_path: Path) -> None:
    missing: list[str] = local_verify(tmp_path / "nope", required=("a.txt",))
    assert missing == ["a.txt"]


@pytest.fixture
def recorded_snapshot(monkeypatch) -> dict[str, Any]:
    """Replace ``snapshot_download`` with a recorder that touches one file."""
    recorded: dict[str, Any] = {}

    def fake_snapshot_download(repo_id: str, **kwargs: Any) -> str:
        recorded["repo_id"] = repo_id
        recorded.update(kwargs)
        local_dir: Path = Path(kwargs["local_dir"])
        landed: Path = local_dir / "MI_valid_01" / "camera_calibration.json"
        landed.parent.mkdir(parents=True, exist_ok=True)
        landed.write_text("{}")
        return str(local_dir)

    monkeypatch.setattr(transports, "snapshot_download", fake_snapshot_download)
    return recorded


def test_hf_fetch_lands_files_at_local_dir(tmp_path: Path, recorded_snapshot: dict[str, Any]) -> None:
    returned: Path = hf_fetch("collabora/monado-slam-datasets", allow_patterns=("MI_valid_01/**",), local_dir=tmp_path, revision="a" * 40)
    assert returned == tmp_path
    # local_dir mode, not the symlinked cache tree: the file sits at <local_dir>/<path-in-repo>.
    assert (tmp_path / "MI_valid_01" / "camera_calibration.json").is_file()
    assert recorded_snapshot["repo_id"] == "collabora/monado-slam-datasets"
    assert recorded_snapshot["repo_type"] == "dataset"
    assert recorded_snapshot["allow_patterns"] == ["MI_valid_01/**"]
    assert recorded_snapshot["local_dir"] == str(tmp_path)
    assert recorded_snapshot["revision"] == "a" * 40


def test_hf_fetch_passes_the_revision_through(tmp_path: Path, recorded_snapshot: dict[str, Any]) -> None:
    hf_fetch("collabora/monado-slam-datasets", allow_patterns=("*.json",), local_dir=tmp_path, repo_type="model", revision="b" * 40)
    assert recorded_snapshot["repo_type"] == "model"
    assert recorded_snapshot["revision"] == "b" * 40


# ── parse_apache_index ────────────────────────────────────────────────────

# Verbatim rows from https://cvg-data.inf.ethz.ch/lamaria/raw_data/training/ and
# .../aria_calibrations/training/, kept as a literal so the test never needs the network.
APACHE_INDEX: str = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<html>
 <head>
  <title>Index of /lamaria/raw_data/training</title>
 </head>
 <body>
<h1>Index of /lamaria/raw_data/training</h1>
  <table>
   <tr><th valign="top"><img src="/isginf/icons/blank.gif" alt="[ICO]"></th><th><a href="?C=N;O=D">Name</a></th><th><a href="?C=M;O=A">Last modified</a></th><th><a href="?C=S;O=A">Size</a></th><th><a href="?C=D;O=A">Description</a></th></tr>
   <tr><th colspan="5"><hr></th></tr>
<tr><td valign="top"><img src="/isginf/icons/back.gif" alt="[PARENTDIR]"></td><td><a href="/lamaria/raw_data/">Parent Directory</a></td><td>&nbsp;</td><td align="right">  - </td><td>&nbsp;</td></tr>
<tr><td valign="top"><img src="/isginf/icons/unknown.gif" alt="[   ]"></td><td><a href="R_01_easy.vrs">R_01_easy.vrs</a></td><td align="right">2025-08-29 14:39  </td><td align="right">897M</td><td>&nbsp;</td></tr>
<tr><td valign="top"><img src="/isginf/icons/unknown.gif" alt="[   ]"></td><td><a href="R_04_medium.vrs">R_04_medium.vrs</a></td><td align="right">2025-08-29 14:39  </td><td align="right">1.9G</td><td>&nbsp;</td></tr>
<tr><td valign="top"><img src="/isginf/icons/unknown.gif" alt="[   ]"></td><td><a href="R_01_easy.json">R_01_easy.json</a></td><td align="right">2025-08-29 17:40  </td><td align="right">2.7K</td><td>&nbsp;</td></tr>
<tr><td valign="top"><img src="/isginf/icons/unknown.gif" alt="[   ]"></td><td><a href="sequence_3_17.vrs">sequence_3_17.vrs</a></td><td align="right">2025-08-29 14:15  </td><td align="right"> 10G</td><td>&nbsp;</td></tr>
   <tr><th colspan="5"><hr></th></tr>
</table>
<address>Apache Server at cvg-data.inf.ethz.ch Port 443</address>
</body></html>
"""


def test_parse_apache_index_reads_name_and_size() -> None:
    listed: list[transports.IndexEntry] = parse_apache_index(APACHE_INDEX)
    # Literal byte counts, worked out by hand from Apache's binary multiples:
    # 897 MiB, 1.9 GiB, 2.7 KiB, 10 GiB.
    assert listed == [
        ("R_01_easy.vrs", 940_572_672),
        ("R_04_medium.vrs", 2_040_109_465),
        ("R_01_easy.json", 2_764),
        ("sequence_3_17.vrs", 10_737_418_240),
    ]


def test_parse_apache_index_skips_the_parent_link_and_sort_headers() -> None:
    names: list[str] = [name for name, _ in parse_apache_index(APACHE_INDEX)]
    assert "Parent Directory" not in names
    assert not any(name.startswith("?C=") for name in names)


# ── http_fetch ────────────────────────────────────────────────────────────

PAYLOAD: bytes = bytes(range(256)) * 41  # 10,496 bytes, so every byte offset is checkable
FILE_PATH: str = "/file.bin"
"""Where the loopback archive serves ``PAYLOAD``."""


@pytest.mark.integration
def test_http_fetch_downloads_a_fresh_file(tmp_path: Path) -> None:
    dest: Path = tmp_path / "nested" / "file.bin"
    with serve({FILE_PATH: PAYLOAD}) as archive:
        returned: Path = http_fetch(f"{archive.base_url}{FILE_PATH}", dest=dest, timeout_s=5.0)
    assert returned == dest
    assert dest.read_bytes() == PAYLOAD
    # HEAD first (that is where the size comes from), then one whole-file GET.
    assert [(entry.method, entry.range_header) for entry in archive.served] == [("HEAD", None), ("GET", None)]


@pytest.mark.integration
def test_http_fetch_skips_a_file_that_is_already_complete(tmp_path: Path) -> None:
    dest: Path = tmp_path / "file.bin"
    dest.write_bytes(PAYLOAD)
    with serve({FILE_PATH: PAYLOAD}) as archive:
        assert http_fetch(f"{archive.base_url}{FILE_PATH}", dest=dest, timeout_s=5.0) == dest
    assert dest.read_bytes() == PAYLOAD
    assert [entry.method for entry in archive.served] == ["HEAD"], "a complete file must cost one HEAD and no body"


@pytest.mark.integration
def test_http_fetch_resumes_a_partial_file(tmp_path: Path) -> None:
    dest: Path = tmp_path / "file.bin"
    already: int = 4_096
    dest.write_bytes(PAYLOAD[:already])
    with serve({FILE_PATH: PAYLOAD}) as archive:
        http_fetch(f"{archive.base_url}{FILE_PATH}", dest=dest, timeout_s=5.0)
    assert dest.read_bytes() == PAYLOAD, "the tail must be appended, not prepended to a second copy of the head"
    assert [(entry.method, entry.range_header) for entry in archive.served] == [("HEAD", None), ("GET", f"bytes={already}-")]


@pytest.mark.integration
def test_http_fetch_restarts_when_the_server_ignores_the_range(tmp_path: Path) -> None:
    """A 200 answer to a ``Range`` request carries the whole file, so appending would double the head."""
    dest: Path = tmp_path / "file.bin"
    dest.write_bytes(PAYLOAD[:1_000])
    with serve({FILE_PATH: PAYLOAD}, honor_ranges=False) as archive:
        http_fetch(f"{archive.base_url}{FILE_PATH}", dest=dest, timeout_s=5.0)
    assert dest.read_bytes() == PAYLOAD


@pytest.mark.integration
def test_http_fetch_refuses_a_local_file_longer_than_the_remote_one(tmp_path: Path) -> None:
    """No retry can fix this one, so it must not spend the budget either."""
    dest: Path = tmp_path / "file.bin"
    dest.write_bytes(PAYLOAD + b"extra")
    with serve({FILE_PATH: PAYLOAD}) as archive, pytest.raises(StaleLocalFile, match="delete it and refetch"):
        http_fetch(f"{archive.base_url}{FILE_PATH}", dest=dest, timeout_s=5.0)
    assert [entry.method for entry in archive.served] == ["HEAD"], "one attempt, and no body transferred"


@pytest.mark.integration
def test_http_fetch_gives_up_after_its_attempts_and_keeps_the_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transports, "RETRY_BACKOFF_S", (0.0,))
    dest: Path = tmp_path / "file.bin"
    served: int = 2_048
    # Every attempt is truncated the same way, so the budget runs out.
    with serve({FILE_PATH: PAYLOAD}, body_limit=served) as archive, pytest.raises(RuntimeError, match="2 attempts"):
        http_fetch(f"{archive.base_url}{FILE_PATH}", dest=dest, timeout_s=5.0, attempts=2)
    assert dest.stat().st_size >= served, "the partial file is the point: the next attempt resumes from it"


@pytest.mark.integration
def test_http_fetch_retries_a_stalled_transfer_and_resumes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The first GET hangs up halfway; the transport's own retry appends the rest."""
    monkeypatch.setattr(transports, "RETRY_BACKOFF_S", (0.0,))
    dest: Path = tmp_path / "file.bin"
    with serve({FILE_PATH: PAYLOAD}, stall_once=".bin") as archive:
        assert http_fetch(f"{archive.base_url}{FILE_PATH}", dest=dest, timeout_s=5.0) == dest
    assert dest.read_bytes() == PAYLOAD
    assert [entry.range_header for entry in archive.served if entry.method == "GET"] == [None, f"bytes={len(PAYLOAD) // 2}-"]
    assert "attempt 1/4 stalled" in capsys.readouterr().out, "a silent retry looks like a slow link"


# ── http_index ────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_http_index_reads_a_page_and_parses_it() -> None:
    with serve({"/training/": APACHE_INDEX.encode()}) as archive:
        listed: list[transports.IndexEntry] = http_index(f"{archive.base_url}/training/")
    assert [entry.name for entry in listed] == ["R_01_easy.vrs", "R_04_medium.vrs", "R_01_easy.json", "sequence_3_17.vrs"]


@pytest.mark.integration
def test_http_index_gives_up_on_a_page_the_archive_does_not_have(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transports, "RETRY_BACKOFF_S", (0.0,))
    with serve({}) as archive, pytest.raises(RuntimeError, match="2 attempts"):
        http_index(f"{archive.base_url}/gone/", attempts=2)
    assert [entry.method for entry in archive.served] == ["GET", "GET"], "a 404 is retried, not treated as an empty directory"


# ── file integrity ────────────────────────────────────────────────────────


def test_content_digest_knows_each_algorithm(tmp_path: Path) -> None:
    path: Path = tmp_path / "hello.txt"
    path.write_bytes(b"hello\n")
    assert content_digest(path, "sha256") == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
    assert content_digest(path, "sha1") == "f572d396fae9206628714fb2ce00f72e94f2258f"
    assert content_digest(path, "git-sha1") == "ce013625030ba8dba906f756967f9e9ca394464a"  # git hash-object


def test_verify_file_checks_size_before_the_hash(tmp_path: Path) -> None:
    path: Path = tmp_path / "hello.txt"
    path.write_bytes(b"hello\n")
    verify_file(path, FileIntegrity(6, None, None))
    verify_file(path, FileIntegrity(6, "git-sha1", "ce013625030ba8dba906f756967f9e9ca394464a"))
    with pytest.raises(IntegrityError, match="holds 6 bytes, the source lists 7"):
        verify_file(path, FileIntegrity(7, "sha1", "0" * 40))
    with pytest.raises(IntegrityError, match="sha1 f572d396"):
        verify_file(path, FileIntegrity(6, "sha1", "0" * 40))
    assert not matches_file(path, FileIntegrity(6, "sha1", "0" * 40))
    assert not matches_file(tmp_path / "absent", FileIntegrity(6, None, None))
    assert not matches_file(tmp_path, FileIntegrity(6, None, None))
    with pytest.raises(ValueError, match="needs its algorithm"):
        FileIntegrity(6, None, "0" * 40)


def test_publish_verified_renames_only_verified_bytes(tmp_path: Path) -> None:
    staged: Path = tmp_path / "staging/a.bin"
    staged.parent.mkdir()
    staged.write_bytes(b"new")
    dest: Path = tmp_path / "out/a.bin"
    with pytest.raises(IntegrityError):
        publish_verified(staged, dest, FileIntegrity(3, "sha256", "0" * 64))
    assert staged.read_bytes() == b"new" and not dest.exists(), "a rejected copy is left for the caller's policy"
    publish_verified(staged, dest, FileIntegrity(3, None, None))
    assert dest.read_bytes() == b"new" and not staged.exists()

    staged.write_bytes(b"newer")
    publish_verified(staged, dest, FileIntegrity(5, None, None), set_aside_replaced=True)
    assert dest.read_bytes() == b"newer" and (tmp_path / "out/a.bin.stale-0").read_bytes() == b"new"


def test_publish_verified_restores_the_set_aside_file_when_the_rename_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staged: Path = tmp_path / "staging/a.bin"
    staged.parent.mkdir()
    staged.write_bytes(b"new")
    dest: Path = tmp_path / "a.bin"
    dest.write_bytes(b"old")

    def refuse(source: Path, target: Path) -> None:
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr("dataforge.transports.os.replace", refuse)
    with pytest.raises(OSError, match="cross-device"):
        publish_verified(staged, dest, FileIntegrity(3, None, None), set_aside_replaced=True)
    assert dest.read_bytes() == b"old" and staged.read_bytes() == b"new"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.bin", "staging"]


def test_fetch_report_summary_counts_files_and_bytes() -> None:
    report: FetchReport = FetchReport(started_s=0.0)
    report.count(True, 2_000_000_000)
    report.count(True, 500_000_000)
    for _ in range(3):
        report.count(False, 1)
    assert (report.fetched, report.fetched_bytes) == (2, 2_500_000_000)
    assert report.summary().startswith("fetched 2 file(s), 2.50 GB in ")
    assert report.summary().endswith("; 3 already complete")


HUB_REPO: str = "org/archives"
HUB_REVISION: str = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> HubStore:
    """Two LFS zip archives at a pinned revision, served to ``transports``."""
    store = HubStore({HUB_REPO: HUB_REVISION})
    store.add(HUB_REPO, "calibration.zip", zip_bytes({"calibration/x": b"intrinsics"}), lfs=True)
    store.add(HUB_REPO, "models.zip", zip_bytes({"models/x": b"mesh"}), lfs=True)
    store.install(monkeypatch, transports)
    return store


def listed(name: str) -> HfFileInfo:
    """The archive's pinned listing entry, through the real lookup."""
    return hf_lfs_files(HUB_REPO, [name], revision=HUB_REVISION)[0]


def test_hf_lfs_files_keeps_the_requested_order_and_names_a_missing_file(hub: HubStore) -> None:
    infos = hf_lfs_files(HUB_REPO, ["models.zip", "calibration.zip"], revision=HUB_REVISION)
    assert [info.path for info in infos] == ["models.zip", "calibration.zip"]
    assert infos[0].integrity == FileIntegrity(len(hub.files[HUB_REPO, "models.zip"]), "sha256", hashlib.sha256(hub.files[HUB_REPO, "models.zip"]).hexdigest())
    with pytest.raises(FileNotFoundError, match="subject_9.zip"):
        hf_lfs_files(HUB_REPO, ["models.zip", "subject_9.zip"], revision=HUB_REVISION)


def test_a_same_size_corrupt_file_is_refetched_and_kept_aside(tmp_path: Path, hub: HubStore) -> None:
    good = hub.files[HUB_REPO, "models.zip"]
    (tmp_path / "models.zip").write_bytes(bytes(len(good)))
    assert hf_fetch_verified(HUB_REPO, listed("models.zip"), local_dir=tmp_path, revision=HUB_REVISION)
    assert (tmp_path / "models.zip").read_bytes() == good
    assert (tmp_path / "models.zip.stale-0").read_bytes() == bytes(len(good))
    assert hub.fetched == ["models.zip"] and hub.forced == []  # staged beside, never downloaded onto the stale file


def test_a_failed_replacement_leaves_the_old_file_in_place(tmp_path: Path, hub: HubStore) -> None:
    (tmp_path / "models.zip").write_bytes(b"old")
    hub.served[HUB_REPO, "models.zip"] = hub.files[HUB_REPO, "models.zip"][:-1]
    with pytest.raises(ValueError, match="kept as"):
        hf_fetch_verified(HUB_REPO, listed("models.zip"), local_dir=tmp_path, revision=HUB_REVISION)
    assert (tmp_path / "models.zip").read_bytes() == b"old"


def test_every_hash_mismatch_is_kept_and_a_rerun_refetches(tmp_path: Path, hub: HubStore) -> None:
    good = hub.files[HUB_REPO, "models.zip"]
    remote = listed("models.zip")
    hub.served[HUB_REPO, "models.zip"] = bytes(len(good))
    for _ in range(2):
        with pytest.raises(ValueError, match="sha256"):
            hf_fetch_verified(HUB_REPO, remote, local_dir=tmp_path, revision=HUB_REVISION)
    staging = tmp_path / transports.STAGING_DIR
    assert sorted(path.name for path in staging.iterdir()) == ["models.zip.sha256-mismatch-0", "models.zip.sha256-mismatch-1"]
    assert not (tmp_path / "models.zip").exists()
    hub.served[HUB_REPO, "models.zip"] = good
    assert hf_fetch_verified(HUB_REPO, remote, local_dir=tmp_path, revision=HUB_REVISION)
    assert (tmp_path / "models.zip").read_bytes() == good


def test_a_verified_staged_file_is_published_without_refetching(tmp_path: Path, hub: HubStore) -> None:
    # An interrupt after the Hub finished but before the rename leaves a complete staged file.
    staged = tmp_path / transports.STAGING_DIR / "models.zip"
    staged.parent.mkdir()
    staged.write_bytes(hub.files[HUB_REPO, "models.zip"])
    assert hf_fetch_verified(HUB_REPO, listed("models.zip"), local_dir=tmp_path, revision=HUB_REVISION)
    assert hub.fetched == []
    assert (tmp_path / "models.zip").read_bytes() == hub.files[HUB_REPO, "models.zip"]
    assert not hf_fetch_verified(HUB_REPO, listed("models.zip"), local_dir=tmp_path, revision=HUB_REVISION)


@pytest.mark.parametrize('revision', ['main', 'abcdef0', 'A' * 40, 'g' * 40, 'refs/pr/1'])
def test_hf_requires_commit_before_network(revision, tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('network called for an unpinned revision')
    monkeypatch.setattr(transports, 'hf_hub_download', forbidden)
    with pytest.raises(ValueError, match='pin a full commit sha'):
        transports.hf_fetch_files('repo', ['file'], local_dir=tmp_path, revision=revision)


def test_hf_parallel_files(tmp_path, monkeypatch):
    from threading import Barrier
    barrier = Barrier(4, timeout=3)
    def download(*_args, **_kwargs):
        barrier.wait()
    monkeypatch.setattr(transports, 'hf_hub_download', download)
    assert transports.hf_fetch_files('repo', ['a', 'b', 'c', 'd'], local_dir=tmp_path, revision='a' * 40, workers=4) == tmp_path


def test_hf_file_failure_propagates(tmp_path, monkeypatch):
    def download(*_args, **_kwargs):
        raise OSError('download failed')
    monkeypatch.setattr(transports, 'hf_hub_download', download)
    with pytest.raises(OSError, match='download failed'):
        transports.hf_fetch_files('repo', ['a', 'b'], local_dir=tmp_path, revision='a' * 40)


def test_dataset_pins():
    from dataforge.datasets.msd import REVISION as MSD_REVISION
    from dataforge.datasets.msd import MsdConfig
    from dataforge.datasets.show3d import REVISION, Show3dConfig
    from dataforge.datasets.show3d_mesh_source import MESH_REVISION
    for pin in (REVISION, MSD_REVISION, MESH_REVISION):
        assert transports.require_commit_sha(pin) == pin
    assert Show3dConfig().revision == REVISION
    assert MsdConfig().revision == MSD_REVISION


@pytest.mark.parametrize('entry', ['snapshot', 'files', 'lfs', 'verified'])
def test_all_hf_entry_points_reject_floating_revision(entry, tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('network entry reached')
    monkeypatch.setattr(transports, 'HfApi', forbidden)
    monkeypatch.setattr(transports, 'snapshot_download', forbidden)
    monkeypatch.setattr(transports, 'hf_hub_download', forbidden)
    with pytest.raises(ValueError, match='main.*pin a full commit sha'):
        if entry == 'snapshot':
            transports.hf_fetch('repo', allow_patterns=[], local_dir=tmp_path, revision='main')
        elif entry == 'files':
            transports.hf_fetch_files('repo', [], local_dir=tmp_path, revision='main')
        elif entry == 'lfs':
            transports.hf_lfs_files('repo', [], revision='main')
        else:
            transports.hf_fetch_verified('repo', transports.HfFileInfo('file', 0, None, 'hash'), local_dir=tmp_path, revision='main')
