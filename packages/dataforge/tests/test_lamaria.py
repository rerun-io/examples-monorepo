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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import rerun as rr
import rerun.blueprint as rrb
import serde.json
from jaxtyping import Float64
from numpy import ndarray

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


def test_a_selected_sequence_the_manifest_never_saw_is_an_error(tmp_path: Path) -> None:
    manifest: LamariaManifest = manifest_fixture()
    root: Path = downloaded_root(tmp_path, manifest=manifest, complete=("R_01_easy",))
    config: LamariaConfig = LamariaConfig(root=root, sequences=("R_01_easy", "R_99_nonesuch"))

    with pytest.raises(ValueError, match="R_99_nonesuch"):
        LamariaDataset(config).discover()


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
    }


def build_archive_handler(bodies: dict[str, bytes], requested: list[str]) -> type[BaseHTTPRequestHandler]:
    """A handler answering GET and HEAD for exactly the paths ``bodies`` names."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _body(self) -> bytes | None:
            requested.append(f"{self.command} {self.path}")
            return bodies.get(self.path)

        def _respond(self, body: bytes | None, *, with_body: bool) -> None:
            if body is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if with_body:
                self.wfile.write(body)

        def do_HEAD(self) -> None:
            self._respond(self._body(), with_body=False)

        def do_GET(self) -> None:
            self._respond(self._body(), with_body=True)

        def log_message(self, format: str, *args: object) -> None:
            """Keep pytest's captured output about dataforge, not about HTTP."""

    return Handler


@contextmanager
def archive(bodies: dict[str, bytes] | None = None) -> Iterator[tuple[str, list[str]]]:
    """Serve the archive on a loopback port; yields its base URL and the request log."""
    requested: list[str] = []
    handler: type[BaseHTTPRequestHandler] = build_archive_handler(archive_bodies() if bodies is None else bodies, requested)
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
    contents: list[str] = [str(part) for part in follow.contents or ()]

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
