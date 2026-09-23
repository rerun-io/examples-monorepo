"""What the dataforge test modules share: the GPU-encoder gate, rrd read-back, synthetic frames, a loopback archive, and the published calibration reader.

Each piece is here because two or more modules need exactly it: the read-back
helpers go through the *public* reader (``RrdReader`` → ``ChunkStore`` → a
datafusion view over one index), so a test asserts what a consumer sees rather
than what the writer intended; the frame builders only have to make consecutive
frames differ, and the sizes stay with the modules, which each have their own
reason for theirs; ``test_transports.py`` and ``test_lamaria.py`` fetch from the
same loopback server; and three modules read the published LaMAria calibration.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
import rerun.blueprint as rrb
import rerun.chunk as rrc
from jaxtyping import UInt8
from numpy import ndarray
from serde import field, from_dict, serde

from dataforge import schema
from dataforge.aria import PublishedTransform
from dataforge.datasets.show3d_hands import HandFrame, HandProfileDoc, read_hand_frames, read_hand_profile
from dataforge.datasets.show3d_layers import Scene, read_scene
from dataforge.datasets.show3d_source import CAMERAS, FrameClock, IndexRow, hand_pose_file, hand_profile_file, read_frame_clock
from dataforge.identity import SequenceIdentity
from dataforge.video_encoding import require_av1_nvenc, resolve_ffmpeg

NOISE_CEILING: int = 96
"""Upper bound of the per-pixel noise, low enough that gradient + noise cannot wrap."""

FIXTURES: Path = Path(__file__).parent / "fixtures"
"""Checked-in binaries and real upstream files; ``fixtures/README.md`` says where each came from."""


def calibration_fixture(device: str) -> Path:
    """One MSD device's **real** ``calibration.json``, copied verbatim from upstream.

    Shared because two modules need the same three files for opposite reasons:
    ``test_basalt`` asserts what the format holds, and ``test_msd`` builds its
    synthetic sequence's calibration out of the real one rather than out of the
    device constant it is checking.

    Args:
        device: ``"index"``, ``"g2"`` or ``"odyssey"``.

    Returns:
        The fixture path, or a skip naming the missing calibration.
    """
    path: Path = FIXTURES / "msd" / f"{device}-calibration.json"
    if not path.is_file():
        pytest.skip(f"MSD calibration fixture is absent: {path}; see tests/fixtures/README.md")
    return path


@pytest.fixture(scope="session")
def nvenc_ffmpeg() -> Path:
    """The resolved ffmpeg, or a skip when this machine cannot encode AV1 on the GPU."""
    ffmpeg: Path = resolve_ffmpeg()
    try:
        require_av1_nvenc(ffmpeg)
    except RuntimeError as error:
        pytest.skip(f"no av1_nvenc: {error}")
    return ffmpeg


def gray_frame(index: int, *, width: int, height: int, noisy: bool = False) -> UInt8[ndarray, "height width"]:
    """A horizontal gradient with a square that moves with ``index``, so no two frames match.

    Args:
        index: Frame number; it drives the square's position and its tint.
        width: Frame width in pixels.
        height: Frame height in pixels.
        noisy: Add sensor-like noise. A flat gradient compresses to a couple of
            kilobytes, and msd's split-archive fixture needs PNG frames that
            really do span volumes.

    Returns:
        One grayscale frame.
    """
    ceiling: int = 255 - NOISE_CEILING if noisy else 255
    frame: UInt8[ndarray, "height width"] = np.tile(np.linspace(0, ceiling, width, dtype=np.uint8), (height, 1))
    if noisy:
        frame = frame + np.random.default_rng(index).integers(0, NOISE_CEILING, (height, width), dtype=np.uint8)
    left: int = (index * 4) % (width - 16)
    frame[8:24, left : left + 16] = np.uint8(255 - (index * 7) % 256)
    return frame


def png_frame(index: int, *, width: int, height: int, noisy: bool = False) -> bytes:
    """One ``gray_frame`` as encoded PNG bytes — the ``image2pipe`` input form."""
    success, buffer = cv2.imencode(".png", gray_frame(index, width=width, height=height, noisy=noisy))
    assert success
    return buffer.tobytes()


def read_back(rrd: Path) -> rrc.ChunkStore:
    """Load a saved rrd the way a consumer does: reader → store → queryable views.

    The stream is materialized because ``from_chunks`` declares ``Sequence[Chunk]``;
    these recordings are a few dozen rows, so the list costs nothing.
    """
    return rrc.ChunkStore.from_chunks(read_chunks(rrd))


def recording_properties(store: rrc.ChunkStore, group: str) -> dict[str, object]:
    """One property group's values (``property:<group>:*``), unwrapped from their one-row lists.

    Properties live on the static ``/__properties`` entity, off every index, so
    they need their own content-filtered read.
    """
    table: pa.Table = store.reader(index=None, contents="/__properties/**").to_arrow_table()
    row: dict[str, list[object] | None] = table.to_pylist()[0]
    prefix: str = f"property:{group}:"
    return {name.removeprefix(prefix): values[0] for name, values in row.items() if name.startswith(prefix) and values}


def eye_vector(batch: rr.components.Position3DBatch | rr.components.Vector3DBatch | None) -> list[float]:
    """Read one three-component field back out of an ``EyeControls3D`` archetype.

    Every field of the archetype is optional, so an unset one is a wiring failure
    rather than a value worth asserting on.
    """
    assert batch is not None, "the follow eye sets every field it is read for"
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


def column_rows(store: rrc.ChunkStore, column: str) -> pa.Table:
    """Non-null rows of one component column, index-sorted."""
    table: pa.Table = store.reader(index=schema.TIMELINE).to_arrow_table().sort_by(schema.TIMELINE)
    return table.select([schema.TIMELINE, column]).drop_null()

# ── the loopback archive ──────────────────────────────────────────────────


class ServedRequest(NamedTuple):
    """One request the loopback server answered."""

    method: str
    """``"HEAD"`` or ``"GET"``."""
    path: str
    """URL path, exactly as the client asked for it."""
    range_header: str | None
    """Verbatim ``Range`` request header, or ``None`` when the client asked for the whole file."""


class LoopbackArchive(NamedTuple):
    """A running loopback HTTP server, and what it has answered so far."""

    base_url: str
    """``http://127.0.0.1:<port>``; the keys of ``bodies`` hang off it."""
    served: list[ServedRequest]
    """Every request in order, so a test can assert *how* a file was fetched."""


@contextmanager
def serve(
    bodies: dict[str, bytes], *, body_limit: int | None = None, honor_ranges: bool = True, stall_once: str | None = None
) -> Iterator[LoopbackArchive]:
    """Serve ``bodies`` on a loopback port for the duration of the block.

    Args:
        bodies: URL path → whole file, exactly as the server should hand it over;
            any other path answers 404.
        body_limit: Truncate every GET body to this many bytes while still
            advertising the full length in HEAD — a server that hangs up early.
        honor_ranges: When False, answer every GET with the whole file and a 200,
            the way a range-blind server (or a CDN in front of one) does.
        stall_once: Truncate the **first** GET of a path with this suffix to half
            its length, the way the real archive hangs up mid-transfer. The next
            GET (a ``Range`` resume) is answered in full.

    Yields:
        The server's base URL and its request log.
    """
    served: list[ServedRequest] = []
    stalled: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _body(self) -> bytes | None:
            served.append(ServedRequest(method=self.command, path=self.path, range_header=self.headers.get("Range")))
            return bodies.get(self.path)

        def _not_found(self) -> None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_HEAD(self) -> None:
            whole: bytes | None = self._body()
            if whole is None:
                self._not_found()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(whole)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self) -> None:
            whole: bytes | None = self._body()
            if whole is None:
                self._not_found()
                return
            requested: str | None = self.headers.get("Range")
            start: int = 0 if requested is None or not honor_ranges else int(requested.removeprefix("bytes=").split("-")[0])
            body: bytes = whole[start:]
            if stall_once is not None and self.path.endswith(stall_once) and not stalled:
                stalled.append(self.path)
                body = body[: len(body) // 2]
            if body_limit is not None:
                body = body[:body_limit]
            self.send_response(206 if start else 200)
            if start:
                self.send_header("Content-Range", f"bytes {start}-{len(whole) - 1}/{len(whole)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *_args: object) -> None:
            """Keep pytest's captured output about dataforge, not about HTTP."""

    server: ThreadingHTTPServer = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread: threading.Thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield LoopbackArchive(base_url=f"http://127.0.0.1:{server.server_port}", served=served)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


# ── the official aria_calibrations JSON ───────────────────────────────────
#
# Only the tests read this file: ``convert`` takes every calibration out of the
# VRS device calibration, and the published JSON exists to cross-check that
# chain (``test_aria_vrs``) and to give the synthetic fixtures real extrinsics
# (``test_lamaria``). It therefore lives here rather than in the package.


@serde
@dataclass(frozen=True)
class PublishedResolution:
    """A published camera's image size."""

    width: int
    """Image width in pixels."""
    height: int
    """Image height in pixels."""


@serde
@dataclass(frozen=True)
class PublishedCamera:
    """One ``cam0``/``cam1`` entry of an ``aria_calibrations/<split>/<seq>.json``.

    This is the *published* calibration, kept only to cross-check the one read
    out of the VRS: it covers the two SLAM cameras and no RGB, and its 16-float
    ``params`` repeat the single Aria focal length as ``fx, fy``.
    """

    model: str
    """Always ``RAD_TAN_THIN_PRISM_FISHEYE``, Aria's FISHEYE624 under COLMAP's name."""
    resolution: PublishedResolution
    """The camera's image size."""
    params: list[float]
    """``[fx, fy, cx, cy, k0..k5, p0, p1, s0..s3]`` — 16 floats."""
    rig_T_cam: PublishedTransform = field(rename="T_b_s")
    """Published as ``T_b_s``; the body frame is imu-right, so this *is* ``rig_T_cam``."""


def read_calibration_json(path: Path) -> dict[str, PublishedCamera]:
    """Read an ``aria_calibrations/<split>/<seq>.json`` file's camera entries.

    The file's third entry, ``imu0``, is skipped: it is the body frame itself, so
    its transform is the identity by definition, and the rest of it is noise
    densities rather than a calibration.

    Args:
        path: The published calibration JSON.

    Returns:
        The camera entries, keyed by their published names (``cam0``, ``cam1``).
    """
    if not path.is_file():
        pytest.skip(f"published Aria calibration is absent: {path}")
    document: dict = json.loads(path.read_text())
    return {name: from_dict(PublishedCamera, entry) for name, entry in document.items() if name.startswith("cam")}


SHOW3D_RAW: Path = Path(__file__).parents[1] / "data/raw/show3d"
"""Local full-length SHOW3D assets shared by annotation tests."""


def read_chunks(rrd: Path) -> list[rrc.Chunk]:
    """Read every published chunk through the public RRD reader."""
    return list(rrc.RrdReader(rrd).stream())


def index_row(**overrides: str | int | bool) -> IndexRow:
    """Build a complete synthetic index row, overriding only the fields under test."""
    fields: dict[str, str | int | bool] = dict(
        subject_id="S", scene_id="keyboard_pick_up_abcd", num_frames=4, split="train",
        has_object_pose=True, has_hand_pose=True, has_caption=True,
        **{f"has_{camera.source_name}": True for camera in CAMERAS},
    )
    fields.update(overrides)
    return from_dict(IndexRow, fields)


class Show3dSceneInputs(NamedTuple):
    """Shared real scene, measured hands, and one decoded subject profile."""

    identity: SequenceIdentity
    scene: Scene
    hands: list[HandFrame]
    profile: HandProfileDoc


@pytest.fixture(scope="module", params=["SPI102/keyboard_toss-away_83ef", "LYA722/birdhousetoy_shaking_8eca"])
def show3d_scene_inputs(request: pytest.FixtureRequest) -> Show3dSceneInputs:
    """Skip absent local assets before reading the two reference scenes."""
    key: str = request.param
    identity: SequenceIdentity = SequenceIdentity("show3d", tuple(key.split("/")))
    scene_dir: Path = SHOW3D_RAW / "scenes" / key
    required: list[Path] = [
        SHOW3D_RAW / hand_pose_file(key),
        SHOW3D_RAW / hand_profile_file(identity.parts[0]),
        *(scene_dir / f"metadata/{name}.json" for name in ("recording_info", "frame_info")),
    ]
    for path in required:
        if not path.is_file():
            pytest.skip(f"SHOW3D scene asset absent: {path}")
    clock: FrameClock = read_frame_clock(scene_dir, key)
    for camera in clock.info.resolution:
        for relative in (f"camera_calibration/{camera}.json", f"blur_info/{camera}.mp4.json"):
            path: Path = scene_dir / relative
            if not path.is_file():
                pytest.skip(f"SHOW3D scene asset absent: {path}")
    scene: Scene = read_scene(scene_dir, scene_key=key)
    return Show3dSceneInputs(
        identity, scene,
        read_hand_frames(SHOW3D_RAW / hand_pose_file(key), scene),
        read_hand_profile(SHOW3D_RAW / hand_profile_file(identity.parts[0])),
    )
