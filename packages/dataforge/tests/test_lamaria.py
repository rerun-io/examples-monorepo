"""LaMAria: the manifest, discovery, remote index resolution, blueprints, and convert.

Nothing here touches the network or a VRS. ``download`` runs against a threaded
``http.server`` serving verbatim Apache index pages, and ``convert`` runs against
the ``open_streams`` seam, so the orchestration (temp mp4s, deletion vs
``--keep-raw``, capture properties) is exercised with synthetic frames while the
real encoder and the real writers do their jobs.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
import rerun.blueprint as rrb
import serde.json
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Intrinsics, KannalaBrandtDistortion

from dataforge import aria, paths, schema
from dataforge.datasets import dataset_defaults, lamaria
from dataforge.datasets.lamaria import (
    DEFAULT_SEQUENCES,
    LamariaConfig,
    LamariaDataset,
    LamariaManifest,
    LamariaSource,
    SequenceRecord,
)
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import FrameSource, ImuChannel, require_av1_nvenc, resolve_ffmpeg

REFERENCE_DIR: Path = Path(__file__).parent / "reference_data" / "lamaria"
"""Verbatim excerpts of published LaMAria files, shared with ``test_aria.py``."""


# ── config and registration ───────────────────────────────────────────────


def test_lamaria_is_registered_under_its_own_command() -> None:
    assert dataset_defaults["lamaria"].command == "lamaria"
    # The catalog dataset is the command: one Aria layout serves every sequence.
    assert dataset_defaults["lamaria"].name == "lamaria"


def test_the_default_selection_is_the_five_surveyed_training_sequences() -> None:
    config: LamariaConfig = LamariaConfig()
    assert config.sequences == DEFAULT_SEQUENCES
    assert config.sequences == ("R_01_easy", "R_04_medium", "R_11_5cp", "sequence_1_19", "sequence_4_11")
    assert config.root == paths.raw_root() / "lamaria"
    assert config.base_url == "https://cvg-data.inf.ethz.ch/lamaria/"
    assert config.keep_raw is False


# ── the manifest ──────────────────────────────────────────────────────────


def manifest_fixture(base_url: str = "https://cvg-data.inf.ethz.ch/lamaria/") -> LamariaManifest:
    """A two-sequence manifest covering both GT shapes and both splits."""
    return LamariaManifest(
        base_url=base_url,
        sequences=[
            SequenceRecord(
                sequence="R_01_easy",
                split="training",
                vrs_url=f"{base_url}raw_data/training/R_01_easy.vrs",
                vrs_display_bytes=940_572_672,
                has_pseudo_gt=True,
                has_control_points=False,
            ),
            SequenceRecord(
                sequence="R_11_5cp",
                split="training",
                vrs_url=f"{base_url}raw_data/training/R_11_5cp.vrs",
                vrs_display_bytes=2_791_728_742,
                has_pseudo_gt=True,
                has_control_points=True,
            ),
        ],
    )


def test_the_manifest_round_trips_through_json(tmp_path: Path) -> None:
    written: LamariaManifest = manifest_fixture()
    path: Path = tmp_path / "manifest.json"
    path.write_text(serde.json.to_json(written))
    assert serde.json.from_json(LamariaManifest, path.read_text()) == written


# ── discovery ─────────────────────────────────────────────────────────────


def small_files(root: Path, record: SequenceRecord) -> None:
    """Touch the files ``download`` leaves on disk for one sequence."""
    sequence_dir: Path = root / record.split / record.sequence
    (sequence_dir / "aria_calibrations").mkdir(parents=True, exist_ok=True)
    (sequence_dir / "aria_calibrations" / f"{record.sequence}.json").write_text("{}")
    if record.has_pseudo_gt:
        (sequence_dir / "ground_truth" / "pGT").mkdir(parents=True, exist_ok=True)
        (sequence_dir / "ground_truth" / "pGT" / f"{record.sequence}.txt").write_text("")
    if record.has_control_points:
        (sequence_dir / "ground_truth" / "control_points").mkdir(parents=True, exist_ok=True)
        (sequence_dir / "ground_truth" / "control_points" / f"{record.sequence}.json").write_text("{}")


def downloaded_root(tmp_path: Path, *, manifest: LamariaManifest, complete: tuple[str, ...]) -> Path:
    """A raw root holding ``manifest`` plus the small files of the named sequences."""
    root: Path = tmp_path / "raw"
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(serde.json.to_json(manifest))
    for record in manifest.sequences:
        if record.sequence in complete:
            small_files(root, record)
    return root


def test_discover_without_a_manifest_names_the_download_verb(tmp_path: Path) -> None:
    dataset: LamariaDataset = LamariaDataset(LamariaConfig(root=tmp_path / "raw"))
    with pytest.raises(FileNotFoundError, match="dataforge-download lamaria"):
        dataset.discover()


def test_discover_pairs_each_selected_sequence_with_its_local_paths(tmp_path: Path) -> None:
    manifest: LamariaManifest = manifest_fixture()
    root: Path = downloaded_root(tmp_path, manifest=manifest, complete=("R_01_easy", "R_11_5cp"))
    config: LamariaConfig = LamariaConfig(root=root, sequences=("R_11_5cp", "R_01_easy"))

    discovered: list[tuple[SequenceIdentity, LamariaSource]] = LamariaDataset(config).discover()

    # Sorted by name whatever order --sequences named them in.
    assert [identity.sequence_key for identity, _ in discovered] == ["R_01_easy", "R_11_5cp"]
    assert discovered[0][0].recording_id == "lamaria__R_01_easy"
    easy: LamariaSource = discovered[0][1]
    assert easy.split == "training"
    assert easy.vrs_path == root / "training" / "R_01_easy" / "raw_data" / "R_01_easy.vrs"
    assert easy.calibration_path == root / "training" / "R_01_easy" / "aria_calibrations" / "R_01_easy.json"
    assert easy.pseudo_gt_path == root / "training" / "R_01_easy" / "ground_truth" / "pGT" / "R_01_easy.txt"
    assert easy.control_points_path is None, "R_01_easy has no surveyed control points"
    assert discovered[1][1].control_points_path == root / "training" / "R_11_5cp" / "ground_truth" / "control_points" / "R_11_5cp.json"


def test_discover_skips_a_sequence_whose_small_files_are_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest: LamariaManifest = manifest_fixture()
    root: Path = downloaded_root(tmp_path, manifest=manifest, complete=("R_01_easy",))
    config: LamariaConfig = LamariaConfig(root=root, sequences=("R_01_easy", "R_11_5cp"))

    discovered: list[tuple[SequenceIdentity, LamariaSource]] = LamariaDataset(config).discover()

    assert [identity.sequence_key for identity, _ in discovered] == ["R_01_easy"]
    # Silence would look like the sequence was never selected.
    assert "R_11_5cp" in capsys.readouterr().out


def test_discover_ignores_a_manifest_sequence_the_config_did_not_select(tmp_path: Path) -> None:
    manifest: LamariaManifest = manifest_fixture()
    root: Path = downloaded_root(tmp_path, manifest=manifest, complete=("R_01_easy", "R_11_5cp"))
    config: LamariaConfig = LamariaConfig(root=root, sequences=("R_01_easy",))

    discovered: list[tuple[SequenceIdentity, LamariaSource]] = LamariaDataset(config).discover()

    assert [identity.sequence_key for identity, _ in discovered] == ["R_01_easy"]


def test_a_selected_sequence_the_manifest_never_saw_is_announced_and_skipped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The selection defaults to all five, so a narrower download must still convert.

    A name the archive genuinely does not have is a hard error at ``download``;
    here it only means this raw root has not been asked for it yet.
    """
    manifest: LamariaManifest = manifest_fixture()
    root: Path = downloaded_root(tmp_path, manifest=manifest, complete=("R_01_easy",))
    config: LamariaConfig = LamariaConfig(root=root, sequences=("R_01_easy", "R_99_nonesuch"))

    discovered: list[tuple[SequenceIdentity, LamariaSource]] = LamariaDataset(config).discover()

    assert [identity.sequence_key for identity, _ in discovered] == ["R_01_easy"]
    output: str = capsys.readouterr().out
    assert "R_99_nonesuch" in output
    assert "--sequences R_99_nonesuch" in output, "the message has to say how to fix it"


# ── the archive, on loopback ──────────────────────────────────────────────

APACHE_HEAD: str = """<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<html>
 <head>
  <title>Index of /lamaria/{directory}</title>
 </head>
 <body>
<h1>Index of /lamaria/{directory}</h1>
  <table>
   <tr><th valign="top"><img src="/isginf/icons/blank.gif" alt="[ICO]"></th><th><a href="?C=N;O=D">Name</a></th><th><a href="?C=M;O=A">Last modified</a></th><th><a href="?C=S;O=A">Size</a></th><th><a href="?C=D;O=A">Description</a></th></tr>
   <tr><th colspan="5"><hr></th></tr>
<tr><td valign="top"><img src="/isginf/icons/back.gif" alt="[PARENTDIR]"></td><td><a href="/lamaria/">Parent Directory</a></td><td>&nbsp;</td><td align="right">  - </td><td>&nbsp;</td></tr>
"""
"""Verbatim head of a cvg-data.inf.ethz.ch fancy index, parent link included."""

APACHE_TAIL: str = """   <tr><th colspan="5"><hr></th></tr>
</table>
<address>Apache Server at cvg-data.inf.ethz.ch Port 443</address>
</body></html>
"""
"""Verbatim tail of the same page."""


def apache_page(directory: str, rows: str) -> str:
    """One index page: the archive's own wrapper around verbatim listing rows."""
    return APACHE_HEAD.format(directory=directory) + rows + APACHE_TAIL


# Every row below is copied verbatim from the saved index pages of the real archive.
RAW_TRAINING_ROWS: str = """<tr><td valign="top"><img src="/isginf/icons/unknown.gif" alt="[   ]"></td><td><a href="R_01_easy.vrs">R_01_easy.vrs</a></td><td align="right">2025-08-29 14:39  </td><td align="right">897M</td><td>&nbsp;</td></tr>
<tr><td valign="top"><img src="/isginf/icons/unknown.gif" alt="[   ]"></td><td><a href="R_04_medium.vrs">R_04_medium.vrs</a></td><td align="right">2025-08-29 14:39  </td><td align="right">1.9G</td><td>&nbsp;</td></tr>
<tr><td valign="top"><img src="/isginf/icons/unknown.gif" alt="[   ]"></td><td><a href="R_11_5cp.vrs">R_11_5cp.vrs</a></td><td align="right">2025-08-29 14:44  </td><td align="right">2.6G</td><td>&nbsp;</td></tr>
"""
RAW_TEST_ROWS: str = """<tr><td valign="top"><img src="/isginf/icons/unknown.gif" alt="[   ]"></td><td><a href="sequence_1_1.vrs">sequence_1_1.vrs</a></td><td align="right">2025-08-29 11:32  </td><td align="right">9.3G</td><td>&nbsp;</td></tr>
"""
CALIBRATION_TRAINING_ROWS: str = """<tr><td valign="top"><img src="/isginf/icons/unknown.gif" alt="[   ]"></td><td><a href="R_01_easy.json">R_01_easy.json</a></td><td align="right">2025-08-29 17:40  </td><td align="right">2.7K</td><td>&nbsp;</td></tr>
<tr><td valign="top"><img src="/isginf/icons/unknown.gif" alt="[   ]"></td><td><a href="R_04_medium.json">R_04_medium.json</a></td><td align="right">2025-08-29 17:40  </td><td align="right">2.7K</td><td>&nbsp;</td></tr>
<tr><td valign="top"><img src="/isginf/icons/unknown.gif" alt="[   ]"></td><td><a href="R_11_5cp.json">R_11_5cp.json</a></td><td align="right">2025-08-29 17:40  </td><td align="right">2.7K</td><td>&nbsp;</td></tr>
"""
CALIBRATION_TEST_ROWS: str = """<tr><td valign="top"><img src="/isginf/icons/unknown.gif" alt="[   ]"></td><td><a href="sequence_1_1.json">sequence_1_1.json</a></td><td align="right">2025-08-29 17:41  </td><td align="right">2.8K</td><td>&nbsp;</td></tr>
"""
PSEUDO_DENSE_ROWS: str = """<tr><td valign="top"><img src="/isginf/icons/text.gif" alt="[TXT]"></td><td><a href="R_01_easy.txt">R_01_easy.txt</a></td><td align="right">2025-09-30 21:43  </td><td align="right">429K</td><td>&nbsp;</td></tr>
<tr><td valign="top"><img src="/isginf/icons/text.gif" alt="[TXT]"></td><td><a href="R_11_5cp.txt">R_11_5cp.txt</a></td><td align="right">2025-09-30 21:50  </td><td align="right">235K</td><td>&nbsp;</td></tr>
"""
SPARSE_ROWS: str = """<tr><td valign="top"><img src="/isginf/icons/unknown.gif" alt="[   ]"></td><td><a href="R_11_5cp.json">R_11_5cp.json</a></td><td align="right">2025-09-05 01:08  </td><td align="right">1.8M</td><td>&nbsp;</td></tr>
"""

CALIBRATION_BODY: bytes = b'{"cam0": {"model": "RAD_TAN_THIN_PRISM_FISHEYE"}}'
"""Stand-in for a published calibration; ``download`` only moves the bytes."""
PSEUDO_GT_BODY: bytes = b"1389350666375 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n"
CONTROL_POINTS_BODY: bytes = b'{"control_points": {}, "images": {}, "timestamps": {}}'
VRS_BODY: bytes = bytes(range(256)) * 64
"""16 384 bytes standing in for a VRS, long enough that a half-served body really resumes."""


def archive_bodies() -> dict[str, bytes]:
    """The whole loopback archive: index pages plus the small files they list."""
    return {
        "/lamaria/raw_data/training/": apache_page("raw_data/training", RAW_TRAINING_ROWS).encode(),
        "/lamaria/raw_data/test/": apache_page("raw_data/test", RAW_TEST_ROWS).encode(),
        "/lamaria/aria_calibrations/training/": apache_page("aria_calibrations/training", CALIBRATION_TRAINING_ROWS).encode(),
        "/lamaria/aria_calibrations/test/": apache_page("aria_calibrations/test", CALIBRATION_TEST_ROWS).encode(),
        "/lamaria/ground_truth/pseudo_dense/": apache_page("ground_truth/pseudo_dense", PSEUDO_DENSE_ROWS).encode(),
        "/lamaria/ground_truth/sparse/": apache_page("ground_truth/sparse", SPARSE_ROWS).encode(),
        "/lamaria/aria_calibrations/training/R_01_easy.json": CALIBRATION_BODY,
        "/lamaria/aria_calibrations/training/R_11_5cp.json": CALIBRATION_BODY,
        "/lamaria/aria_calibrations/test/sequence_1_1.json": CALIBRATION_BODY,
        "/lamaria/ground_truth/pseudo_dense/R_01_easy.txt": PSEUDO_GT_BODY,
        "/lamaria/ground_truth/pseudo_dense/R_11_5cp.txt": PSEUDO_GT_BODY,
        "/lamaria/ground_truth/sparse/R_11_5cp.json": CONTROL_POINTS_BODY,
        # A stand-in for the 897 MB VRS: the reader is replaced by the ``open_streams``
        # seam, so what matters is that the fetch, the resume and the deletion are real.
        "/lamaria/raw_data/training/R_01_easy.vrs": VRS_BODY,
    }


def build_archive_handler(bodies: dict[str, bytes], requested: list[str], *, stall_first_get: bool) -> type[BaseHTTPRequestHandler]:
    """A handler answering GET (with byte ranges) and HEAD for the paths ``bodies`` names.

    Args:
        bodies: URL path → whole file, exactly as the archive serves it.
        requested: Appended to on every request, so a test can assert *how* a file was fetched.
        stall_first_get: Truncate the first GET of a ``.vrs`` to half its length
            while still advertising the full one, the way the real archive hangs
            up mid-transfer. The next GET (a ``Range`` resume) is answered in full.
    """
    stalled: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _body(self) -> bytes | None:
            requested.append(f"{self.command} {self.path}")
            return bodies.get(self.path)

        def do_HEAD(self) -> None:
            body: bytes | None = self._body()
            if body is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self) -> None:
            whole: bytes | None = self._body()
            if whole is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            requested_range: str | None = self.headers.get("Range")
            start: int = 0 if requested_range is None else int(requested_range.removeprefix("bytes=").split("-")[0])
            body: bytes = whole[start:]
            if stall_first_get and self.path.endswith(".vrs") and not stalled:
                stalled.append(self.path)
                body = body[: len(body) // 2]
            self.send_response(206 if start else 200)
            if start:
                self.send_header("Content-Range", f"bytes {start}-{len(whole) - 1}/{len(whole)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Keep pytest's captured output about dataforge, not about HTTP."""

    return Handler


@contextmanager
def archive(bodies: dict[str, bytes] | None = None, *, stall_first_get: bool = False) -> Iterator[tuple[str, list[str]]]:
    """Serve the archive on a loopback port; yields its base URL and the request log."""
    requested: list[str] = []
    handler: type[BaseHTTPRequestHandler] = build_archive_handler(
        archive_bodies() if bodies is None else bodies, requested, stall_first_get=stall_first_get
    )
    server: ThreadingHTTPServer = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread: threading.Thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/lamaria/", requested
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


# ── download ──────────────────────────────────────────────────────────────


def test_download_resolves_splits_and_ground_truth_from_the_index_pages(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root: Path = tmp_path / "raw"
    with archive() as (base_url, _):
        LamariaDataset(LamariaConfig(root=root, base_url=base_url, sequences=("R_11_5cp", "R_01_easy", "sequence_1_1"))).download()

    manifest: LamariaManifest = serde.json.from_json(LamariaManifest, (root / "manifest.json").read_text())
    assert [record.sequence for record in manifest.sequences] == ["R_01_easy", "R_11_5cp", "sequence_1_1"]
    easy, surveyed, held_out = manifest.sequences
    # The split is *resolved*: which raw_data index listed the VRS decides it.
    assert (easy.split, surveyed.split, held_out.split) == ("training", "training", "test")
    assert easy.vrs_url == f"{base_url}raw_data/training/R_01_easy.vrs"
    assert held_out.vrs_url == f"{base_url}raw_data/test/sequence_1_1.vrs"
    # 897 MiB and 9.3 GiB, Apache's rounded display sizes.
    assert (easy.vrs_display_bytes, held_out.vrs_display_bytes) == (940_572_672, 9_985_798_963)
    assert (easy.has_pseudo_gt, easy.has_control_points) == (True, False)
    assert (surveyed.has_pseudo_gt, surveyed.has_control_points) == (True, True)
    # The test split ships no ground truth at all.
    assert (held_out.has_pseudo_gt, held_out.has_control_points) == (False, False)

    output: str = capsys.readouterr().out
    assert "3 sequence(s)" in output
    assert "R_11_5cp" in output and "control points" in output
    # 0.9 + 2.8 + 10.0 GB of VRS the convert verb will fetch one at a time.
    assert "13.7 GB" in output


def test_download_lands_the_small_files_in_the_official_layout(tmp_path: Path) -> None:
    root: Path = tmp_path / "raw"
    with archive() as (base_url, _):
        LamariaDataset(LamariaConfig(root=root, base_url=base_url, sequences=("R_01_easy", "R_11_5cp", "sequence_1_1"))).download()

    assert (root / "training" / "R_01_easy" / "aria_calibrations" / "R_01_easy.json").read_bytes() == CALIBRATION_BODY
    assert (root / "training" / "R_01_easy" / "ground_truth" / "pGT" / "R_01_easy.txt").read_bytes() == PSEUDO_GT_BODY
    assert (root / "training" / "R_11_5cp" / "ground_truth" / "control_points" / "R_11_5cp.json").read_bytes() == CONTROL_POINTS_BODY
    # R_01_easy was never surveyed, and the test split has no ground truth.
    assert not (root / "training" / "R_01_easy" / "ground_truth" / "control_points").exists()
    assert (root / "test" / "sequence_1_1" / "aria_calibrations" / "sequence_1_1.json").is_file()
    assert not (root / "test" / "sequence_1_1" / "ground_truth").exists()
    # The VRS files themselves are the whole point of fetching on demand.
    assert not list(root.rglob("*.vrs"))


def test_download_then_discover_yields_every_downloaded_sequence(tmp_path: Path) -> None:
    root: Path = tmp_path / "raw"
    config: LamariaConfig = LamariaConfig(root=root, sequences=("R_01_easy", "R_11_5cp", "sequence_1_1"))
    with archive() as (base_url, _):
        dataset: LamariaDataset = LamariaDataset(LamariaConfig(root=root, base_url=base_url, sequences=config.sequences))
        dataset.download()
        discovered: list[tuple[SequenceIdentity, LamariaSource]] = dataset.discover()

    assert [identity.sequence_key for identity, _ in discovered] == ["R_01_easy", "R_11_5cp", "sequence_1_1"]
    assert [source.split for _, source in discovered] == ["training", "training", "test"]


def test_a_sequence_no_raw_index_lists_is_named_in_the_error(tmp_path: Path) -> None:
    with archive() as (base_url, _), pytest.raises(ValueError, match="R_99_nonesuch"):
        LamariaDataset(LamariaConfig(root=tmp_path / "raw", base_url=base_url, sequences=("R_01_easy", "R_99_nonesuch"))).download()


def test_a_sequence_with_no_published_calibration_is_named_in_the_error(tmp_path: Path) -> None:
    """Every LaMAria sequence ships one, so a missing entry means the archive changed."""
    bodies: dict[str, bytes] = archive_bodies()
    bodies["/lamaria/aria_calibrations/training/"] = apache_page("aria_calibrations/training", CALIBRATION_TRAINING_ROWS.splitlines(True)[0]).encode()
    with archive(bodies) as (base_url, _), pytest.raises(ValueError, match="R_11_5cp"):
        LamariaDataset(LamariaConfig(root=tmp_path / "raw", base_url=base_url, sequences=("R_01_easy", "R_11_5cp"))).download()


# ── the follow frame and the blueprints ───────────────────────────────────


def test_the_declared_follow_frame_is_the_calibration_own_forward_and_up() -> None:
    """The eye's axes are typed in by hand, so the calibration has to agree with them.

    ``T_b_s`` for cam0 is ``rig_T_cam`` of camera-slam-left in the imu-right
    frame, and the Aria device frame *is* that camera's frame (RDF: x right, y
    down, z along the optical axis). So the rotation's third column is where the
    wearer looks and the negated second column is the wearer's up.
    """
    published: dict[str, aria.PublishedCamera] = aria.read_calibration_json(REFERENCE_DIR / "R_01_easy.calibration.json")
    rig_R_cam0: Float64[ndarray, "3 3"] = published["cam0"].rig_T_cam.to_matrix()[:3, :3]

    np.testing.assert_allclose(lamaria.FOLLOW_FORWARD, rig_R_cam0[:, 2], atol=1e-3)
    np.testing.assert_allclose(lamaria.FOLLOW_UP, -rig_R_cam0[:, 1], atol=1e-3)
    assert np.linalg.norm(lamaria.FOLLOW_FORWARD) == pytest.approx(1.0, abs=1e-3)
    assert np.linalg.norm(lamaria.FOLLOW_UP) == pytest.approx(1.0, abs=1e-3)
    assert float(np.dot(lamaria.FOLLOW_FORWARD, lamaria.FOLLOW_UP)) == pytest.approx(0.0, abs=1e-3)


def eye_vector(batch: rr.components.Position3DBatch | rr.components.Vector3DBatch | None) -> list[float]:
    """Read one three-component field back out of an ``EyeControls3D`` archetype."""
    assert batch is not None, "follow_eye_controls sets every field of the eye"
    return [float(value) for value in batch.as_arrow_array().flatten().to_pylist()]


def blueprint_views(blueprint: rrb.Blueprint) -> list[rrb.View]:
    """Every view in a blueprint, depth-first, whatever containers nest them."""
    found: list[rrb.View] = []

    def walk(node: rrb.View | rrb.Container) -> None:
        if isinstance(node, rrb.View):
            found.append(node)
            return
        for child in node.contents or ():
            walk(child)

    walk(blueprint.root_container)
    return found


def test_the_follow_eye_chases_the_wearer_from_behind_and_above() -> None:
    eye: rrb.EyeControls3D = lamaria.follow_eye_controls()
    forward: Float64[ndarray, "3"] = np.array(lamaria.FOLLOW_FORWARD, dtype=np.float64)
    up: Float64[ndarray, "3"] = np.array(lamaria.FOLLOW_UP, dtype=np.float64)

    assert eye_vector(eye.look_target) == pytest.approx((lamaria.FOLLOW_AHEAD_M * forward).tolist(), abs=1e-6)
    assert eye_vector(eye.eye_up) == pytest.approx(list(lamaria.FOLLOW_UP), abs=1e-6)
    position: list[float] = eye_vector(eye.position)
    assert float(np.dot(position, forward)) < 0.0, "the eye leans against forward"
    assert float(np.dot(position, up)) > 0.0, "and with up"


def test_the_default_blueprint_shows_three_cameras_over_the_imu_plots() -> None:
    views: list[rrb.View] = blueprint_views(LamariaDataset(LamariaConfig()).default_blueprint())

    assert [view.name for view in views if isinstance(view, rrb.Spatial3DView)] == ["World", "Follow"]
    assert [view.name for view in views if isinstance(view, rrb.Spatial2DView)] == ["camera-slam-left", "camera-slam-right", "camera-rgb"]
    assert [view.name for view in views if isinstance(view, rrb.TimeSeriesView)] == ["Gyroscope", "Accelerometer"]
    plots: list[rrb.View] = [view for view in views if isinstance(view, rrb.TimeSeriesView)]
    # The plots follow imu_00 (imu-right), the sensor the rig frame coincides with.
    assert str(plots[0].contents) == schema.gyro_path(0, 0)
    assert str(plots[1].contents) == schema.accel_path(0, 0)


def test_the_default_blueprint_already_names_the_gt_layer_paths() -> None:
    """The overrides are inert in a base-only rrd, and correct once the gt layer stacks on."""
    views: list[rrb.View] = blueprint_views(LamariaDataset(LamariaConfig()).default_blueprint())
    follow: rrb.View = next(view for view in views if view.name == "Follow")
    world: rrb.View = next(view for view in views if view.name == "World")

    assert set(follow.visualizer_overrides) == {schema.trajectory_path("gt"), schema.trail_path("gt")}
    assert set(world.visualizer_overrides) == {schema.trail_path("gt")}


def test_the_table_card_decodes_only_the_slam_left_stream() -> None:
    """Every visible table row renders through this at once, so it excludes the rest."""
    views: list[rrb.View] = blueprint_views(LamariaDataset(LamariaConfig()).table_blueprint())
    follow: rrb.View = next(view for view in views if view.name == "Follow")
    contents: str = str(follow.contents)

    assert [view.name for view in views if isinstance(view, rrb.Spatial2DView)] == ["camera-slam-left"]
    for index in range(3):
        assert f"- {schema.video_path(0, index)}/**" in contents, "a card must not decode video in the 3D view"
    assert f"- {schema.trail_path('gt')}/**" in contents
    pane: rrb.View = next(view for view in views if isinstance(view, rrb.Spatial2DView))
    assert pane.origin == schema.pinhole_path(0, 0)


def test_both_blueprints_serialize_to_a_non_empty_rbl(tmp_path: Path) -> None:
    dataset: LamariaDataset = LamariaDataset(LamariaConfig())
    default_path: Path = tmp_path / "lamaria.rbl"
    table_path: Path = tmp_path / "lamaria-table.rbl"

    dataset.default_blueprint().save("lamaria", str(default_path))
    dataset.table_blueprint().save("lamaria", str(table_path))

    assert default_path.stat().st_size > 0
    assert table_path.stat().st_size > 0


# ── convert, against the VRS seam ─────────────────────────────────────────

FRAME_WIDTH: int = 192
"""Synthetic frame width; NVENC refuses anything much smaller."""
FRAME_HEIGHT: int = 160
"""Synthetic frame height, likewise above NVENC's minimum."""
SLAM_FRAMES: int = 8
"""Frames the synthetic SLAM cameras carry, at 20 Hz."""
RGB_FRAMES: int = 4
"""Frames the synthetic RGB camera carries, at 10 Hz — half the rate, as on a real Aria."""
DEVICE_T0_NS: int = 1_389_350_666_375
"""R_01_easy's first slam-left device timestamp; ``video_time`` must show it unshifted."""
SLAM_PERIOD_NS: int = 50_000_000
"""20 Hz."""
IMU_SAMPLES: int = 40
"""Samples per synthetic IMU channel."""
IMU_PERIOD_NS: int = 1_000_000
"""1 kHz."""
IMU_T0_NS: int = DEVICE_T0_NS - 9_000_000
"""The IMUs start before the first frame, as they do in a real VRS: this is ``start_time_ns``."""
IMU_LEFT_TRANSLATION_M: tuple[float, float, float] = (0.005, -0.102, -0.086)
"""imu-left's offset in the rig frame, rounded from R_01_easy's device calibration."""


def synthetic_frames(count: int, *, channels: int) -> Iterator[bytes]:
    """One camera's whole clip, lazily, the way the VRS reader hands frames over.

    A function and not a generator expression on purpose: an expression would
    look ``channels`` up when it is finally consumed, long after the loop that
    built it moved on.
    """
    for index in range(count):
        yield synthetic_frame(index, channels=channels)


def synthetic_frame(index: int, *, channels: int) -> bytes:
    """One raw plane with a moving square, so no two frames compress to the same bytes."""
    shape: tuple[int, ...] = (FRAME_HEIGHT, FRAME_WIDTH) if channels == 1 else (FRAME_HEIGHT, FRAME_WIDTH, channels)
    frame: aria.AriaImage = np.full(shape, np.uint8(40 + 3 * index), dtype=np.uint8)
    left: int = (index * 6) % (FRAME_WIDTH - 16)
    frame[8:24, left : left + 16] = np.uint8(220)
    return frame.tobytes()


def synthetic_camera(stream_id: aria.AriaStreamId, rig_T_cam: Float64[ndarray, "4 4"]) -> Fisheye62Parameters:
    """A Fisheye62 camera at ``rig_T_cam`` with the synthetic frame size."""
    return Fisheye62Parameters(
        name=aria.STREAM_LABELS[stream_id],
        extrinsics=Extrinsics(world_R_cam=rig_T_cam[:3, :3].copy(), world_t_cam=rig_T_cam[:3, 3].copy()),
        intrinsics=Intrinsics.from_focal_principal_point(
            camera_conventions="RDF", fl_x=120.0, fl_y=120.0, cx=96.0, cy=80.0, width=FRAME_WIDTH, height=FRAME_HEIGHT
        ),
        distortion=KannalaBrandtDistortion(k1=-0.02, k2=0.09, k3=-0.06, k4=0.006, k5=0.003, k6=-0.0007, p1=0.0008, p2=0.0003),
    )


def published_rig_T_cam(name: str) -> Float64[ndarray, "4 4"]:
    """``rig_T_cam`` of one published camera, straight out of the reference calibration."""
    published: dict[str, aria.PublishedCamera] = aria.read_calibration_json(REFERENCE_DIR / "R_01_easy.calibration.json")
    return published[name].rig_T_cam.to_matrix()


def synthetic_streams(_: Path) -> lamaria.SequenceStreams:
    """Stand in for ``open_streams``: the same shapes, without a 900 MB VRS.

    The two SLAM cameras are placed at R_01_easy's *published* extrinsics, so an
    assertion about what landed on ``cam_00`` has a source of truth outside this
    module. camera-rgb has no published entry (it exists only in the VRS), so it
    reuses cam1's pose.
    """
    poses: dict[aria.AriaStreamId, Float64[ndarray, "4 4"]] = {
        aria.SLAM_LEFT_STREAM_ID: published_rig_T_cam("cam0"),
        aria.SLAM_RIGHT_STREAM_ID: published_rig_T_cam("cam1"),
        aria.RGB_STREAM_ID: published_rig_T_cam("cam1"),
    }
    cameras: list[lamaria.CameraStream] = []
    for stream_id in aria.CAMERA_STREAM_IDS:
        rgb: bool = stream_id == aria.RGB_STREAM_ID
        count: int = RGB_FRAMES if rgb else SLAM_FRAMES
        period_ns: int = SLAM_PERIOD_NS * 2 if rgb else SLAM_PERIOD_NS
        cameras.append(
            lamaria.CameraStream(
                stream_id=stream_id,
                camera=synthetic_camera(stream_id, poses[stream_id]),
                frames=synthetic_frames(count, channels=3 if rgb else 1),
                times_ns=DEVICE_T0_NS + np.arange(count, dtype=np.int64) * period_ns,
                frame_source=FrameSource("rgb24" if rgb else "gray8", width=FRAME_WIDTH, height=FRAME_HEIGHT),
                fps=10 if rgb else 20,
            )
        )

    times_ns: Int64[ndarray, "n_samples"] = IMU_T0_NS + np.arange(IMU_SAMPLES, dtype=np.int64) * IMU_PERIOD_NS
    rig_T_imu_left: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    rig_T_imu_left[:3, :3] = Rotation.from_euler("z", 8.0, degrees=True).as_matrix()
    rig_T_imu_left[:3, 3] = IMU_LEFT_TRANSLATION_M
    imus: list[lamaria.ImuStream] = []
    for stream_id, rig_T_imu in ((aria.IMU_RIGHT_STREAM_ID, np.eye(4, dtype=np.float64)), (aria.IMU_LEFT_STREAM_ID, rig_T_imu_left)):
        imus.append(
            lamaria.ImuStream(
                stream_id=stream_id,
                gyro=ImuChannel(times_ns=times_ns, values_xyz=np.tile([0.01, -0.02, 0.03], (IMU_SAMPLES, 1))),
                accel=ImuChannel(times_ns=times_ns, values_xyz=np.tile([0.1, -0.2, 9.81], (IMU_SAMPLES, 1))),
                rig_T_imu=rig_T_imu,
            )
        )
    return lamaria.SequenceStreams(cameras=tuple(cameras), imus=tuple(imus))


@dataclass(frozen=True, slots=True)
class FakeArchive:
    """A loopback archive plus the raw root and config a convert works against."""

    root: Path
    """``LamariaConfig.root``, already holding the manifest and the small files."""
    config: LamariaConfig
    """Config pointed at ``root`` and the loopback base URL."""
    requested: list[str]
    """Every request the archive answered, in order."""
    vrs_path: Path
    """Where ``convert`` fetches the VRS to."""


@contextmanager
def converting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, keep_raw: bool = False, stall_first_get: bool = False
) -> Iterator[FakeArchive]:
    """Download R_01_easy from a loopback archive, then hand ``convert`` the seam."""
    root: Path = tmp_path / "raw"
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "rrd"))
    monkeypatch.setattr(lamaria, "open_streams", synthetic_streams)
    with archive(stall_first_get=stall_first_get) as (base_url, requested):
        config: LamariaConfig = LamariaConfig(root=root, base_url=base_url, sequences=("R_01_easy",), keep_raw=keep_raw)
        LamariaDataset(config).download()
        requested.clear()
        yield FakeArchive(
            root=root,
            config=config,
            requested=requested,
            vrs_path=root / "training" / "R_01_easy" / "raw_data" / "R_01_easy.vrs",
        )


def convert_one(fake: FakeArchive, *, force: bool = False) -> tuple[SequenceIdentity, Path]:
    """Discover R_01_easy and convert it, returning its identity and its rrd."""
    dataset: LamariaDataset = LamariaDataset(fake.config)
    identity, source = dataset.discover()[0]
    return identity, dataset.convert(identity, source, force=force)


def read_back(rrd: Path) -> rr.experimental.ChunkStore:
    """Load a saved rrd the way a consumer does: reader → store → queryable views."""
    return rr.experimental.ChunkStore.from_chunks(list(rr.experimental.RrdReader(rrd).stream()))


def column_rows(store: rr.experimental.ChunkStore, column: str) -> pa.Table:
    """Non-null rows of one component column, index-sorted."""
    table: pa.Table = store.reader(index=schema.TIMELINE).to_arrow_table().sort_by(schema.TIMELINE)
    return table.select([schema.TIMELINE, column]).drop_null()


def recording_properties(store: rr.experimental.ChunkStore, group: str) -> dict[str, object]:
    """One property group's values (``property:<group>:*``), unwrapped from their one-row lists."""
    table: pa.Table = store.reader(index=None, contents="/__properties/**").to_arrow_table()
    row: dict[str, list[object] | None] = table.to_pylist()[0]
    prefix: str = f"property:{group}:"
    return {name.removeprefix(prefix): values[0] for name, values in row.items() if name.startswith(prefix) and values}


@pytest.fixture(scope="module")
def nvenc() -> Path:
    """The resolved ffmpeg, or a skip when this machine cannot encode AV1 on the GPU."""
    ffmpeg: Path = resolve_ffmpeg()
    try:
        require_av1_nvenc(ffmpeg)
    except RuntimeError as error:
        pytest.skip(f"no av1_nvenc: {error}")
    return ffmpeg


def test_convert_writes_three_camera_streams_and_two_imus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc: Path) -> None:
    with converting(tmp_path, monkeypatch) as fake:
        identity, target = convert_one(fake)

    assert target.name == "lamaria__R_01_easy.rrd"
    store: rr.experimental.ChunkStore = read_back(target)
    assert [column_rows(store, f"{schema.video_path(0, index)}:VideoStream:sample").num_rows for index in range(3)] == [
        SLAM_FRAMES,
        SLAM_FRAMES,
        RGB_FRAMES,
    ]
    for imu in range(2):
        assert column_rows(store, f"{schema.gyro_path(0, imu)}:Scalars:scalars").num_rows == IMU_SAMPLES
        assert column_rows(store, f"{schema.accel_path(0, imu)}:Scalars:scalars").num_rows == IMU_SAMPLES
    rig: dict[str, list[object]] = store.reader(index=None, contents=schema.rig_path(0)).to_arrow_table().to_pylist()[0]
    assert rig[f"{schema.rig_path(0)}:schema_version"][0] == schema.EXOEGO_SCHEMA_VERSION
    assert rig[f"{schema.rig_path(0)}:reference"][0] == "imu_00"
    assert rig[f"{schema.rig_path(0)}:name"][0] == "aria"
    assert rig[f"{schema.rig_path(0)}:kind"][0] == "ego"
    assert identity.recording_id == "lamaria__R_01_easy"


def test_video_time_is_the_raw_device_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc: Path) -> None:
    """No shift anywhere: a pGT row's own timestamp must land on its frame."""
    with converting(tmp_path, monkeypatch) as fake:
        _, target = convert_one(fake)

    samples: pa.Table = column_rows(read_back(target), f"{schema.video_path(0, 0)}:VideoStream:sample")
    times_ns: list[int] = samples.column(schema.TIMELINE).combine_chunks().cast(pa.int64()).to_pylist()
    assert times_ns[0] == DEVICE_T0_NS
    assert times_ns[-1] == DEVICE_T0_NS + (SLAM_FRAMES - 1) * SLAM_PERIOD_NS


def test_the_logged_cam_00_node_carries_the_published_rig_T_cam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc: Path) -> None:
    """``log_pinhole`` stores the child-from-parent step, so inverting it gives ``T_b_s`` back."""
    with converting(tmp_path, monkeypatch) as fake:
        _, target = convert_one(fake)

    node: str = schema.cam_path(0, 0)
    row: dict[str, list[object]] = read_back(target).reader(index=None, contents=node).to_arrow_table().to_pylist()[0]
    assert row[f"{node}:Transform3D:relation"][0] == rr.components.TransformRelation.ChildFromParent.value
    # Rerun stores mat3x3 column-major, so the read-back needs one transpose.
    cam_R_rig: Float64[ndarray, "3 3"] = np.asarray(row[f"{node}:Transform3D:mat3x3"][0], dtype=np.float64).reshape(3, 3).T
    cam_t_rig: Float64[ndarray, "3"] = np.asarray(row[f"{node}:Transform3D:translation"][0], dtype=np.float64)

    expected: Float64[ndarray, "4 4"] = published_rig_T_cam("cam0")
    # float32 on the wire, so a loose tolerance is the honest one.
    np.testing.assert_allclose(cam_R_rig.T, expected[:3, :3], atol=1e-6)
    np.testing.assert_allclose(-cam_R_rig.T @ cam_t_rig, expected[:3, 3], atol=1e-6)
    assert row[f"{node}:name"][0] == "camera-slam-left"
    assert row[f"{node}:kind"][0] == "grayscale"


def test_the_rgb_camera_says_so_and_the_slam_pair_does_not(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc: Path) -> None:
    with converting(tmp_path, monkeypatch) as fake:
        _, target = convert_one(fake)

    store: rr.experimental.ChunkStore = read_back(target)
    kinds: list[object] = []
    for index in range(3):
        node: str = schema.cam_path(0, index)
        row: dict[str, list[object]] = store.reader(index=None, contents=node).to_arrow_table().to_pylist()[0]
        kinds.append(row[f"{node}:kind"][0])
    assert kinds == ["grayscale", "grayscale", "rgb"]


def test_imu_01_carries_its_real_pose_while_imu_00_is_the_rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc: Path) -> None:
    """imu-right *is* the rig frame; imu-left sits 13 cm away and rotated."""
    with converting(tmp_path, monkeypatch) as fake:
        _, target = convert_one(fake)

    store: rr.experimental.ChunkStore = read_back(target)
    poses: list[Float64[ndarray, "3"]] = []
    for imu in range(2):
        node: str = schema.imu_path(0, imu)
        row: dict[str, list[object]] = store.reader(index=None, contents=node).to_arrow_table().to_pylist()[0]
        poses.append(np.asarray(row[f"{node}:Transform3D:translation"][0], dtype=np.float64))
    np.testing.assert_allclose(poses[0], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(poses[1], IMU_LEFT_TRANSLATION_M, atol=1e-6)


def test_the_base_layer_owns_no_world_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc: Path) -> None:
    """The gt layer establishes the world, so it owns the root axes and the rig transform."""
    with converting(tmp_path, monkeypatch) as fake:
        _, target = convert_one(fake)

    store: rr.experimental.ChunkStore = read_back(target)
    assert "/:ViewCoordinates:xyz" not in store.reader(index=None, contents="/").to_arrow_table().column_names
    rig_columns: list[str] = store.reader(index=None, contents=schema.rig_path(0)).to_arrow_table().column_names
    # A static transform here would permanently shadow the temporal world_T_rig.
    assert not [name for name in rig_columns if "Transform3D" in name]


def test_the_capture_properties_describe_the_sequence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc: Path) -> None:
    with converting(tmp_path, monkeypatch) as fake:
        vrs_bytes: int = len(archive_bodies()["/lamaria/raw_data/training/R_01_easy.vrs"])
        _, target = convert_one(fake)

    capture: dict[str, object] = recording_properties(read_back(target), "capture")
    assert capture["schema"] == schema.DATAFORGE_SCHEMA_VERSION
    assert capture["num_cameras"] == 3
    assert capture["num_frames"] == SLAM_FRAMES, "num_frames is the slam-left count, not the RGB one"
    assert capture["split"] == "training"
    assert capture["set"] == "controlled"
    assert capture["challenge"] == "easy"
    assert capture["has_pseudo_gt"] is True
    assert capture["control_point_count"] == 0, "R_01_easy was never surveyed"
    assert capture["start_time_ns"] == IMU_T0_NS, "the IMUs start before the first frame"
    assert capture["vrs_bytes"] == vrs_bytes
    duration_s: object = capture["duration_s"]
    assert isinstance(duration_s, float)
    assert duration_s == pytest.approx((DEVICE_T0_NS + (SLAM_FRAMES - 1) * SLAM_PERIOD_NS - IMU_T0_NS) / 1e9, abs=1e-9)


def test_convert_deletes_the_vrs_and_the_mp4s_but_keeps_the_small_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc: Path
) -> None:
    with converting(tmp_path, monkeypatch) as fake:
        convert_one(fake)
        assert not fake.vrs_path.exists(), "raw is scratch: 18 GB of VRS must not accumulate"
        assert not list(fake.root.rglob("*.mp4"))
        assert (fake.root / "training" / "R_01_easy" / "aria_calibrations" / "R_01_easy.json").is_file()
        assert (fake.root / "training" / "R_01_easy" / "ground_truth" / "pGT" / "R_01_easy.txt").is_file()


def test_keep_raw_leaves_the_vrs_and_the_encoded_mp4s(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc: Path) -> None:
    with converting(tmp_path, monkeypatch, keep_raw=True) as fake:
        convert_one(fake)
        assert fake.vrs_path.is_file()
        assert sorted(path.name for path in fake.root.rglob("*.mp4")) == ["cam_00.mp4", "cam_01.mp4", "cam_02.mp4"]


def test_an_existing_recording_is_skipped_without_fetching_the_vrs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with converting(tmp_path, monkeypatch) as fake:
        dataset: LamariaDataset = LamariaDataset(fake.config)
        identity, source = dataset.discover()[0]
        target: Path = paths.rrd_path(paths.output_root(), layer=paths.BASE_LAYER, identity=identity)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"already done")

        assert dataset.convert(identity, source, force=False) == target
        assert fake.requested == [], "an existing rrd must cost neither a request nor an encode"
        assert target.read_bytes() == b"already done"


def test_force_rewrites_an_existing_recording(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc: Path) -> None:
    with converting(tmp_path, monkeypatch) as fake:
        dataset: LamariaDataset = LamariaDataset(fake.config)
        identity, source = dataset.discover()[0]
        target: Path = paths.rrd_path(paths.output_root(), layer=paths.BASE_LAYER, identity=identity)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"stale")

        assert dataset.convert(identity, source, force=True) == target
        assert target.read_bytes() != b"stale"


def test_a_failed_encode_keeps_the_vrs_and_clears_the_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], nvenc: Path
) -> None:
    def explode(*arguments: object, **keywords: object) -> int:
        raise RuntimeError("nvenc fell over")

    with converting(tmp_path, monkeypatch) as fake:
        monkeypatch.setattr(lamaria, "encode_frames_to_mp4", explode)
        dataset: LamariaDataset = LamariaDataset(fake.config)
        identity, source = dataset.discover()[0]
        with pytest.raises(RuntimeError, match="nvenc fell over"):
            dataset.convert(identity, source, force=False)

        # The download is the expensive half, so it survives; the scratch does not.
        assert fake.vrs_path.is_file()
        assert not list(fake.root.rglob("*.mp4"))
        assert not paths.rrd_path(paths.output_root(), layer=paths.BASE_LAYER, identity=identity).exists()
        assert "kept" in capsys.readouterr().out


def test_a_stalled_vrs_fetch_is_retried_and_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], nvenc: Path
) -> None:
    """The archive hangs up mid-transfer; the retry must append, never restart."""
    with converting(tmp_path, monkeypatch, stall_first_get=True, keep_raw=True) as fake:
        convert_one(fake)

    assert fake.vrs_path.read_bytes() == archive_bodies()["/lamaria/raw_data/training/R_01_easy.vrs"]
    ranges: list[str] = [entry for entry in fake.requested if entry.endswith(".vrs")]
    assert len(ranges) >= 3, f"expected HEAD, a stalled GET and a resumed GET, got {ranges}"
    assert "resuming" in capsys.readouterr().out


# ── the gt world frame, the trajectory, and the measured up axis ───────────


def test_the_first_ten_controlled_sequences_are_posed_in_the_mps_frame() -> None:
    """Upstream posed ``R_01``…``R_10`` in MPS's own gravity-aligned frame and everything else in LV95/LN02."""
    assert lamaria.gt_world("R_01_easy") == "mps"
    assert lamaria.gt_world("R_10_hard") == "mps"
    assert lamaria.gt_world("R_11_5cp") == "lv95"
    assert lamaria.gt_world("sequence_1_19") == "lv95"


def test_a_controlled_name_without_an_index_is_an_error_not_a_guess() -> None:
    with pytest.raises(ValueError, match="index"):
        lamaria.gt_world("R_x_easy")


def test_the_rig_pose_is_the_published_camera_pose_seen_from_the_rig() -> None:
    """``world_T_rig = world_T_cam0 @ cam0_T_rig``, and the fixture pins which way simplecv stores it.

    ``Fisheye62Parameters.extrinsics.world_T_cam`` holds ``rig_T_cam0`` here (its
    "world" is the rig), so the composition has to invert it. At an identity
    ``world_T_cam0`` the rig therefore lands on ``cam0_T_rig``'s translation,
    which is minus the rotated ``T_b_s`` translation of cam0.
    """
    rig_T_cam0: Float64[ndarray, "4 4"] = published_rig_T_cam("cam0")
    identity_pose: aria.PseudoGt = aria.PseudoGt(
        times_ns=np.array([DEVICE_T0_NS], dtype=np.int64), world_T_cam0=np.eye(4, dtype=np.float64).reshape(1, 4, 4)
    )

    trajectory: lamaria.GtTrajectory = lamaria.rig_trajectory(identity_pose, rig_T_cam0=rig_T_cam0)

    expected_xyz: Float64[ndarray, "3"] = -rig_T_cam0[:3, :3].T @ rig_T_cam0[:3, 3]
    np.testing.assert_allclose(trajectory.translations_xyz[0], expected_xyz, atol=1e-12)
    # And the rotation is cam0's, inverted: the rig is rolled out of the camera frame.
    np.testing.assert_allclose(
        Rotation.from_quat(trajectory.quaternions_xyzw[0]).as_matrix(), rig_T_cam0[:3, :3].T, atol=1e-12
    )
    assert trajectory.length_m == 0.0, "one pose covers no distance"


def test_a_constant_rotation_leaves_the_path_length_the_camera_walked() -> None:
    """The rig sits a fixed offset from cam0, so a rigid walk has one length in both frames."""
    rig_T_cam0: Float64[ndarray, "4 4"] = published_rig_T_cam("cam0")
    world_T_cam0: Float64[ndarray, "n_poses 4 4"] = np.tile(np.eye(4, dtype=np.float64), (4, 1, 1))
    world_T_cam0[:, :3, 3] = np.column_stack([0.25 * np.arange(4.0), np.zeros(4), np.zeros(4)])

    trajectory: lamaria.GtTrajectory = lamaria.rig_trajectory(
        aria.PseudoGt(times_ns=DEVICE_T0_NS + np.arange(4, dtype=np.int64) * SLAM_PERIOD_NS, world_T_cam0=world_T_cam0),
        rig_T_cam0=rig_T_cam0,
    )

    assert trajectory.length_m == pytest.approx(0.75)
    assert trajectory.duration_s == pytest.approx(3 * SLAM_PERIOD_NS / 1e9)


def test_an_empty_pseudo_gt_yields_an_empty_trajectory() -> None:
    """A sequence with control points but no pGT still gets a gt layer, without poses."""
    empty: aria.PseudoGt = aria.PseudoGt(times_ns=np.zeros(0, dtype=np.int64), world_T_cam0=np.zeros((0, 4, 4)))

    trajectory: lamaria.GtTrajectory = lamaria.rig_trajectory(empty, rig_T_cam0=published_rig_T_cam("cam0"))

    assert trajectory.times_ns.size == 0
    assert trajectory.length_m == 0.0
    assert trajectory.duration_s == 0.0


def resting_accel(times_ns: Int64[ndarray, "n_samples"], accel_xyz: Float64[ndarray, "3"]) -> ImuChannel:
    """An accelerometer reading one fixed vector, in the rig frame, at ``times_ns``."""
    return ImuChannel(times_ns=times_ns, values_xyz=np.tile(accel_xyz, (times_ns.size, 1)))


def constant_rotation_trajectory(times_ns: Int64[ndarray, "n_poses"], world_R_rig: Rotation) -> lamaria.GtTrajectory:
    """A gt trajectory that holds one orientation at the world origin."""
    return lamaria.GtTrajectory(
        times_ns=times_ns,
        translations_xyz=np.zeros((times_ns.size, 3)),
        quaternions_xyzw=np.tile(np.asarray(world_R_rig.as_quat(), dtype=np.float64), (times_ns.size, 1)),
        length_m=0.0,
        duration_s=float((times_ns[-1] - times_ns[0]) / 1e9),
    )


def test_the_world_up_axis_is_measured_by_rotating_the_accelerometer_into_the_world() -> None:
    """An accelerometer at rest reads +g pointing *up*, so ``world_R_rig @ a_rig`` averages to the up axis."""
    times_ns: Int64[ndarray, "n_samples"] = DEVICE_T0_NS + np.arange(4_000, dtype=np.int64) * IMU_PERIOD_NS
    accel: ImuChannel = resting_accel(times_ns, np.array([0.1, -0.2, 9.81]))
    # The second half of the capture points the other way; the 2 s window must ignore it.
    accel.values_xyz[times_ns >= DEVICE_T0_NS + lamaria.MEASURED_UP_WINDOW_NS] = [0.1, -0.2, -9.81]

    measured: lamaria.WorldUp = lamaria.measured_world_up(constant_rotation_trajectory(times_ns, Rotation.identity()), accel)

    assert measured.axis == "+z"
    assert measured.fraction_of_g == pytest.approx(1.0, abs=0.01)


def test_a_rig_lying_on_its_side_measures_the_axis_its_own_gravity_points_along() -> None:
    """The mapping is a real rotation, not a relabelling: +90 deg about x sends the rig's +z onto the world's -y."""
    times_ns: Int64[ndarray, "n_samples"] = DEVICE_T0_NS + np.arange(1_000, dtype=np.int64) * IMU_PERIOD_NS
    world_R_rig: Rotation = Rotation.from_euler("x", 90.0, degrees=True)

    measured: lamaria.WorldUp = lamaria.measured_world_up(
        constant_rotation_trajectory(times_ns, world_R_rig), resting_accel(times_ns, np.array([0.0, 0.0, 9.80665]))
    )

    assert measured.axis == "-y"
    assert measured.fraction_of_g == pytest.approx(1.0, abs=1e-6)


def test_measuring_the_world_up_axis_needs_both_a_pose_and_a_sample() -> None:
    empty_times: Int64[ndarray, "n_samples"] = np.zeros(0, dtype=np.int64)
    times_ns: Int64[ndarray, "n_samples"] = DEVICE_T0_NS + np.arange(10, dtype=np.int64) * IMU_PERIOD_NS
    with pytest.raises(ValueError, match="both a gt pose and an accelerometer sample"):
        lamaria.measured_world_up(
            constant_rotation_trajectory(times_ns, Rotation.identity()), resting_accel(empty_times, np.array([0.0, 0.0, 9.8]))
        )


def test_an_accelerometer_that_stops_before_the_ground_truth_starts_is_an_error() -> None:
    """The window opens where both streams are live, so an IMU that quit first leaves it empty."""
    times_ns: Int64[ndarray, "n_poses"] = DEVICE_T0_NS + np.arange(10, dtype=np.int64) * IMU_PERIOD_NS
    far_earlier: Int64[ndarray, "n_samples"] = times_ns - 60_000_000_000
    with pytest.raises(ValueError, match="no accelerometer sample within"):
        lamaria.measured_world_up(
            constant_rotation_trajectory(times_ns, Rotation.identity()), resting_accel(far_earlier, np.array([0.0, 0.0, 9.8]))
        )
