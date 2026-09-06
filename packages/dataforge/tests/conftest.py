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
from jaxtyping import Float64, UInt8
from numpy import ndarray
from scipy.spatial.transform import Rotation
from serde import field, from_dict, serde

from dataforge import schema
from dataforge.logging_toolkit import require_av1_nvenc, resolve_ffmpeg

NOISE_CEILING: int = 96
"""Upper bound of the per-pixel noise, low enough that gradient + noise cannot wrap."""


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


def read_back(rrd: Path) -> rr.experimental.ChunkStore:
    """Load a saved rrd the way a consumer does: reader → store → queryable views.

    The stream is materialized because ``from_chunks`` declares ``Sequence[Chunk]``;
    these recordings are a few dozen rows, so the list costs nothing.
    """
    return rr.experimental.ChunkStore.from_chunks(list(rr.experimental.RrdReader(rrd).stream()))


def column_rows(store: rr.experimental.ChunkStore, column: str) -> pa.Table:
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


def read_back(rrd: Path) -> rr.experimental.ChunkStore:
    """Load a saved rrd the way a consumer does: reader → store → queryable views.

    The stream is materialized because ``from_chunks`` declares ``Sequence[Chunk]``;
    these recordings are a few dozen rows, so the list costs nothing.
    """
    return rr.experimental.ChunkStore.from_chunks(list(rr.experimental.RrdReader(rrd).stream()))


@pytest.fixture(scope="session")
def serving():
    """``serve``, as a fixture: a server's lifetime belongs to pytest, not to an import."""
    return serve


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
class PublishedTransform:
    """A published rigid transform: a quaternion in x, y, z, w order and a translation."""

    qvec: list[float]
    """Rotation as ``[x, y, z, w]`` — pycolmap's order, which the official tooling reads it with."""
    tvec: list[float]
    """Translation in metres."""

    def to_matrix(self) -> Float64[ndarray, "4 4"]:
        """The same transform as a 4x4."""
        matrix: Float64[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = Rotation.from_quat(np.asarray(self.qvec, dtype=np.float64)).as_matrix()
        matrix[:3, 3] = self.tvec
        return matrix


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
    document: dict = json.loads(path.read_text())
    return {name: from_dict(PublishedCamera, entry) for name, entry in document.items() if name.startswith("cam")}
