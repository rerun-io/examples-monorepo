"""The Insta360 base's lens videos: the size they are stored at, the camera model at that size, and the base as the source
of lens samples for the derived layers.

The base stores each lens re-encoded to AV1 at ``LENS_MAX_SIZE`` (one halving of the X5's 3840x3840 lenses). It is an
exception to dataforge's ~3 MP cap (``video_encoding.MAX_STORED_PIXELS``, which would halve 3840 twice, to 960): 1920 keeps
the detail slam-rs tracks, the masks read and the cube faces sample, at a twentieth of the camera's HEVC rate. A lens is
halved by a box filter (the mean of each ``s x s`` block), so a pixel centre ``c`` of the camera's image lies at
``(c + 0.5) / s - 0.5`` in the stored one; ``halved_camera`` scales the Mei model by that rule, so the stored camera
projects onto the stored pixels.

The cube and panorama layers read their pixels from the base through ``BaseLensVideos``, beside the calibration they read
from it (``insta360.cameras_from_recording``): pixels and camera model from one source, whatever size the base stores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import insta360_io
import numpy as np
import rerun as rr
import rerun.chunk as rrc
from jaxtyping import Bool, Int64, UInt8
from numpy import ndarray
from simplecv.catalog_video_codec import CatalogCodecName, catalog_codec_name
from simplecv.rerun_custom_types import DistortionCoefficientsBatch, DistortionModelBatch

LENS_MAX_SIZE: int = 1920
"""Widest lens the base stores: lenses wider are halved until they fit (3840 -> 1920 on the X5)."""


@runtime_checkable
class LensSamples(Protocol):
    """Where the lens decoders read encoded samples: the ``.insv`` clip (``insta360_io.InsvRig``) or the base (``BaseLensVideos``)."""

    @property
    def codec(self) -> CatalogCodecName:
        """``h264``, ``hevc`` or ``av1``."""
        ...

    def gops(self, lens: int) -> Int64[ndarray, "g 2"]:
        """``[start, stop)`` sample range of each GOP of one lens."""
        ...

    def read_packed(self, lens: int, start: int, stop: int) -> tuple[UInt8[ndarray, " n"], Int64[ndarray, " m"], Bool[ndarray, " k"]]:
        """Samples ``start .. stop - 1`` of one lens: their bytes end to end, the ``k + 1`` sample bounds, the keyframe flags."""
        ...


def lens_halvings(width: int) -> int:
    """How many times a lens ``width`` pixels wide is halved to fit ``LENS_MAX_SIZE``."""
    count: int = 0
    while (width >> count) > LENS_MAX_SIZE:
        count += 1
    return count


def halved_camera(camera: insta360_io.MeiCamera, halvings: int) -> insta360_io.MeiCamera:
    """``camera`` on its image box-filtered ``halvings`` times: focal ``f s``, principal point ``(c + 0.5) s - 0.5``,
    ``s = 2^-halvings``; the unitless coefficients (``xi``, the field and the distortion terms) are unchanged.

    The Mei projection is ``gamma m(x) + c`` with ``m`` unitless, so the halved camera projects every ray onto
    ``(p + 0.5) s - 0.5`` of the full camera's ``p``: the stored pixel whose box holds it.
    """
    scale: float = 1.0 / (1 << halvings)
    return insta360_io.MeiCamera(
        camera.model,
        camera.fx * scale,
        camera.fy * scale,
        (camera.cx + 0.5) * scale - 0.5,
        (camera.cy + 0.5) * scale - 0.5,
        list(camera.coefficients),
    )


def log_lens_camera(recording: rr.RecordingStream, entity: str, camera: insta360_io.MeiCamera, width: int, height: int) -> None:
    """Log a lens camera at its stored size the way ``insta360_io.InsvReader`` logs the camera's own: the paraxial
    ``gamma / (1 + xi)`` in the ``Pinhole`` (the viewer's approximation) and the Mei model under its unified name in the
    distortion coefficients (``gamma_x, gamma_y, *coefficients``; exoego:v2 section 4)."""
    xi: float = float(camera.coefficients[0])
    pinhole = rr.Pinhole(
        image_from_camera=[[camera.fx / (1.0 + xi), 0.0, camera.cx], [0.0, camera.fy / (1.0 + xi), camera.cy], [0.0, 0.0, 1.0]],
        width=width,
        height=height,
        camera_xyz=rr.ViewCoordinates.RDF,
        image_plane_distance=0.05,
    )
    distortion = (DistortionModelBatch(f"{camera.model}_unified"), DistortionCoefficientsBatch(np.asarray([[camera.fx, camera.fy, *camera.coefficients]], dtype=np.float32)))
    recording.log(entity, pinhole, [batch.described(batch.component_descriptor()) for batch in distortion], static=True)


class BaseLensVideos:
    """The two lens videos of a base recording as encoded samples (``LensSamples``), held in memory, and their stamps."""

    def __init__(self, base: Path, video_entities: tuple[str, str]) -> None:
        """Read both lens videos of ``base``.

        Args:
            base: The base recording.
            video_entities: ``cam_00``'s and ``cam_01``'s ``VideoStream`` entities.

        Raises:
            ValueError: If a lens video is missing, does not start on a keyframe, or the two lenses differ in stamps or codec.
        """
        store: rrc.LazyStore = rrc.RrdReader(base).store()
        self.samples: list[list[bytes]] = []
        self.keyframes: list[Bool[ndarray, " n"]] = []
        codecs: list[CatalogCodecName] = []
        stamps: list[Int64[ndarray, " n"]] = []
        for entity in video_entities:
            times: list[int] = []
            blobs: list[bytes] = []
            keyframe_times: set[int] = set()
            for chunk in store.stream().filter(content=[entity]).to_chunks():
                batch = chunk.to_record_batch()
                names: list[str] = batch.schema.names
                if "VideoStream:codec" in names:
                    codecs.append(catalog_codec_name(int(batch.column("VideoStream:codec").to_pylist()[0][0])))
                if "video_time" not in names:
                    continue
                stamp: list[int] = np.asarray(batch.column("video_time").to_numpy(zero_copy_only=False)).view(np.int64).tolist()
                if "VideoStream:sample" in names:
                    times += stamp
                    blobs += [bytes(row[0]) for row in batch.column("VideoStream:sample").to_pylist()]
                if "VideoStream:is_keyframe" in names:
                    keyframe_times.update(t for t, flags in zip(stamp, batch.column("VideoStream:is_keyframe").to_pylist(), strict=True) if flags and flags[0])
            if not blobs:
                raise ValueError(f"{base}: {entity} holds no video samples")
            order: Int64[ndarray, " n"] = np.argsort(np.asarray(times, dtype=np.int64), kind="stable")
            stamps.append(np.asarray(times, dtype=np.int64)[order])
            self.samples.append([blobs[k] for k in order.tolist()])
            self.keyframes.append(np.asarray([t in keyframe_times for t in stamps[-1].tolist()], dtype=np.bool_))
            if not self.keyframes[-1][0]:
                raise ValueError(f"{base}: {entity} does not start on a keyframe")
        if len(set(codecs)) != 1 or len(codecs) != 2:
            raise ValueError(f"{base}: the lens videos name codecs {codecs}, not one each and the same")
        if not np.array_equal(stamps[0], stamps[1]):
            raise ValueError(f"{base}: the two lens videos are not on the same stamps")
        self._codec: CatalogCodecName = codecs[0]
        self.times_ns: Int64[ndarray, " n"] = stamps[0]
        """``video_time`` of each frameset, nanoseconds."""

    @property
    def codec(self) -> CatalogCodecName:
        """The lens videos' codec."""
        return self._codec

    @property
    def num_frames(self) -> int:
        """Framesets in the base."""
        return self.times_ns.size

    def gops(self, lens: int) -> Int64[ndarray, "g 2"]:
        """``[start, stop)`` of each GOP of one lens, from its keyframe flags."""
        starts: Int64[ndarray, " g"] = np.flatnonzero(self.keyframes[lens]).astype(np.int64)
        return np.stack([starts, np.r_[starts[1:], self.num_frames]], axis=1)

    def read_packed(self, lens: int, start: int, stop: int) -> tuple[UInt8[ndarray, " n"], Int64[ndarray, " m"], Bool[ndarray, " k"]]:
        """Samples ``start .. stop - 1`` of one lens end to end, their bounds and keyframe flags."""
        chosen: list[bytes] = self.samples[lens][start:stop]
        bounds: Int64[ndarray, " m"] = np.r_[0, np.cumsum([len(sample) for sample in chosen])].astype(np.int64)
        return np.frombuffer(b"".join(chosen), dtype=np.uint8), bounds, self.keyframes[lens][start:stop].copy()
