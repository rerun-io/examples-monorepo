import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from dataforge import transports
from dataforge.transports import gdrive_fetch, hf_fetch, http_fetch, local_verify, parse_apache_index


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


def test_unbuilt_transports_raise() -> None:
    with pytest.raises(NotImplementedError):
        gdrive_fetch()


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
    returned: Path = hf_fetch("collabora/monado-slam-datasets", allow_patterns=("MI_valid_01/**",), local_dir=tmp_path)
    assert returned == tmp_path
    # local_dir mode, not the symlinked cache tree: the file sits at <local_dir>/<path-in-repo>.
    assert (tmp_path / "MI_valid_01" / "camera_calibration.json").is_file()
    assert recorded_snapshot["repo_id"] == "collabora/monado-slam-datasets"
    assert recorded_snapshot["repo_type"] == "dataset"
    assert recorded_snapshot["allow_patterns"] == ["MI_valid_01/**"]
    assert recorded_snapshot["local_dir"] == str(tmp_path)
    assert recorded_snapshot["revision"] is None


def test_hf_fetch_passes_the_revision_through(tmp_path: Path, recorded_snapshot: dict[str, Any]) -> None:
    hf_fetch("collabora/monado-slam-datasets", allow_patterns=("*.json",), local_dir=tmp_path, repo_type="model", revision="refs/pr/1")
    assert recorded_snapshot["repo_type"] == "model"
    assert recorded_snapshot["revision"] == "refs/pr/1"


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
    listed: list[tuple[str, int]] = parse_apache_index(APACHE_INDEX)
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


@dataclass(slots=True)
class ServedRequest:
    """One request the test server answered."""

    method: str
    """``"HEAD"`` or ``"GET"``."""
    range_header: str | None
    """Verbatim ``Range`` request header, or ``None`` when the client asked for the whole file."""


def build_handler(payload: bytes, log: list[ServedRequest], *, body_limit: int | None, honor_ranges: bool) -> type[BaseHTTPRequestHandler]:
    """A handler serving ``payload`` with ``Content-Length``, HEAD, and byte ranges.

    Args:
        payload: The file's bytes.
        log: Appended to on every request, so a test can assert *how* it was fetched.
        body_limit: Truncate every GET body to this many bytes while still
            advertising the full length in HEAD — a server that hangs up early.
        honor_ranges: When False, answer every GET with the whole file and a 200,
            the way a range-blind server (or a CDN in front of one) does.
    """

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _requested_start(self) -> int:
            requested: str | None = self.headers.get("Range")
            log.append(ServedRequest(method=self.command, range_header=requested))
            if requested is None or not honor_ranges:
                return 0
            return int(requested.removeprefix("bytes=").split("-")[0])

        def do_HEAD(self) -> None:
            self._requested_start()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self) -> None:
            start: int = self._requested_start()
            body: bytes = payload[start:]
            if body_limit is not None:
                body = body[:body_limit]
            self.send_response(206 if start else 200)
            if start:
                self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


@contextmanager
def serving(*, body_limit: int | None = None, honor_ranges: bool = True) -> Iterator[tuple[str, list[ServedRequest]]]:
    """Run the handler on a loopback port for the duration of the block."""
    log: list[ServedRequest] = []
    handler: type[BaseHTTPRequestHandler] = build_handler(PAYLOAD, log, body_limit=body_limit, honor_ranges=honor_ranges)
    server: ThreadingHTTPServer = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread: threading.Thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/file.bin", log
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def test_http_fetch_downloads_a_fresh_file(tmp_path: Path) -> None:
    dest: Path = tmp_path / "nested" / "file.bin"
    with serving() as (url, log):
        returned: Path = http_fetch(url, dest=dest, timeout_s=5.0)
    assert returned == dest
    assert dest.read_bytes() == PAYLOAD
    # HEAD first (that is where the size comes from), then one whole-file GET.
    assert [(entry.method, entry.range_header) for entry in log] == [("HEAD", None), ("GET", None)]


def test_http_fetch_skips_a_file_that_is_already_complete(tmp_path: Path) -> None:
    dest: Path = tmp_path / "file.bin"
    dest.write_bytes(PAYLOAD)
    with serving() as (url, log):
        assert http_fetch(url, dest=dest, timeout_s=5.0) == dest
    assert dest.read_bytes() == PAYLOAD
    assert [entry.method for entry in log] == ["HEAD"], "a complete file must cost one HEAD and no body"


def test_http_fetch_resumes_a_partial_file(tmp_path: Path) -> None:
    dest: Path = tmp_path / "file.bin"
    already: int = 4_096
    dest.write_bytes(PAYLOAD[:already])
    with serving() as (url, log):
        http_fetch(url, dest=dest, timeout_s=5.0)
    assert dest.read_bytes() == PAYLOAD, "the tail must be appended, not prepended to a second copy of the head"
    assert [(entry.method, entry.range_header) for entry in log] == [("HEAD", None), ("GET", f"bytes={already}-")]


def test_http_fetch_restarts_when_the_server_ignores_the_range(tmp_path: Path) -> None:
    """A 200 answer to a ``Range`` request carries the whole file, so appending would double the head."""
    dest: Path = tmp_path / "file.bin"
    dest.write_bytes(PAYLOAD[:1_000])
    with serving(honor_ranges=False) as (url, _):
        http_fetch(url, dest=dest, timeout_s=5.0)
    assert dest.read_bytes() == PAYLOAD


def test_http_fetch_rejects_a_stale_expected_size(tmp_path: Path) -> None:
    dest: Path = tmp_path / "file.bin"
    with serving() as (url, log), pytest.raises(ValueError, match="the manifest is stale"):
        http_fetch(url, dest=dest, expected_size=len(PAYLOAD) + 1, timeout_s=5.0)
    assert not dest.exists(), "a wrong size must be caught by the HEAD, before any body is transferred"
    assert [entry.method for entry in log] == ["HEAD"]


def test_http_fetch_keeps_a_short_transfer_for_the_next_resume(tmp_path: Path) -> None:
    dest: Path = tmp_path / "file.bin"
    served: int = 2_048
    with serving(body_limit=served) as (url, _), pytest.raises(ValueError, match=f"holds {served} of {len(PAYLOAD)} bytes"):
        http_fetch(url, dest=dest, timeout_s=5.0)
    assert dest.stat().st_size == served, "the partial file is the point: the next call resumes from it"
